#!/usr/bin/env python3
"""
Import LMSYS chat conversation data into JAM memory system.
Parses CSV with conversation history and creates memories with 5W1H extraction.
Optimized for batch processing of thousands of records.
"""
import os
import csv
import json
import ast
import sys
import time
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

# Increase CSV field size limit to handle large conversation fields
csv.field_size_limit(sys.maxsize)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_memory.router import MemoryRouter
from agentic_memory.config import Config, cfg
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.types import MemoryRecord, Who, Where, RawEvent
from agentic_memory.extraction.llm_extractor import extract_5w1h
from agentic_memory.embedding import get_llama_embedder


class LMSYSImporter:
    """Import LMSYS conversation data into JAM memory system.
    
    Optimized for high-throughput batch processing with:
    - Batch database operations
    - Batch embedding generation
    - Parallel conversation processing
    - Connection pooling and reuse
    """
    
    def __init__(self, csv_path: str, batch_size: int = 100, num_workers: int = 2, use_llm: bool = False):
        """
        Initialize the importer.
        
        Args:
            csv_path: Path to LMSYS CSV file
            batch_size: Number of conversations to process at once
            num_workers: Number of parallel workers for processing
            use_llm: Whether to use LLM for 5W1H extraction (slower but more accurate)
        """
        self.csv_path = Path(csv_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_llm = use_llm
        
        # Initialize memory components
        self.store = MemoryStore(cfg.db_path)
        # Get embedding dimension from config (1024 for Qwen3-Embedding)
        embed_dim = int(os.getenv('AM_EMBEDDING_DIM', '1024'))
        self.index = FaissIndex(dim=embed_dim, index_path=cfg.index_path)
        self.router = MemoryRouter(self.store, self.index)
        
        # Initialize direct components for batch operations
        # Set embedding server URL explicitly
        os.environ['AM_EMBEDDING_SERVER_URL'] = 'http://localhost:8002/v1'
        os.environ['AM_EMBEDDING_DIM'] = '1024'
        self.embedder = get_llama_embedder()
        
        # Create a dedicated connection for batch operations
        self.db_conn = sqlite3.connect(cfg.db_path)
        self.db_conn.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging for better concurrency
        self.db_conn.execute('PRAGMA synchronous=NORMAL')  # Faster writes
        self.db_conn.execute('PRAGMA cache_size=10000')  # Larger cache
        self.db_conn.execute('PRAGMA temp_store=MEMORY')  # Use memory for temp tables
        
        self.stats = {
            'total_conversations': 0,
            'successful_imports': 0,
            'failed_imports': 0,
            'total_memories_created': 0,
            'processing_time': 0,
            'batch_times': []
        }
        
        # Cache for parsed conversations
        self.parse_cache = {}
        
    def parse_conversation(self, conversation_str: str) -> List[Dict[str, str]]:
        """
        Parse the conversation string into structured messages.
        Uses caching to avoid re-parsing identical conversations.
        
        Args:
            conversation_str: String representation of conversation list
            
        Returns:
            List of message dictionaries
        """
        # Check cache first
        cache_key = hash(conversation_str[:500])  # Use first 500 chars as key
        if cache_key in self.parse_cache:
            return self.parse_cache[cache_key]
        
        try:
            # Pre-compile replacement patterns for efficiency
            if not hasattr(self, '_replacements'):
                self._replacements = [
                    ('}\n {', '}, {'),
                    ('}\n{', '}, {'),
                    ('}\r\n {', '}, {'),
                    ('}\r\n{', '}, {')
                ]
            
            cleaned = conversation_str.strip()
            for old, new in self._replacements:
                cleaned = cleaned.replace(old, new)
            
            # Try to parse as Python literal first
            try:
                conversation = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                # Fallback to JSON parsing
                conversation = json.loads(cleaned)
            
            # Cache successful parse
            if len(self.parse_cache) < 1000:  # Limit cache size
                self.parse_cache[cache_key] = conversation
            
            return conversation
                
        except Exception:
            return []
    
    def prepare_memories_from_conversation(self, row: Dict[str, Any]) -> List[Tuple[MemoryRecord, np.ndarray]]:
        """
        Prepare memory objects and embeddings from a conversation.
        This doesn't save to DB, just prepares the data for batch insertion.
        
        Args:
            row: CSV row with conversation data
            
        Returns:
            List of (Memory, embedding) tuples ready for batch insertion
        """
        memories_to_insert = []
        
        try:
            conversation = self.parse_conversation(row.get('conversation', ''))
            if not conversation:
                return memories_to_insert
            
            # Get conversation metadata
            conversation_id = row.get('conversation_id', '')
            model_name = row.get('model', 'unknown')
            language = row.get('language', 'English')
            
            # Collect all content for batch embedding
            messages_to_process = []
            
            for i, message in enumerate(conversation):
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                
                if not content or role not in ['user', 'assistant']:
                    continue
                
                # Determine event type and actor
                if role == 'user':
                    event_type = 'user_message'
                    actor = f"user:lmsys_user"
                else:
                    event_type = 'llm_message'
                    actor = f"llm:{model_name}"
                
                # Build context hint
                context_hint = ""
                if i > 0 and i-1 < len(conversation):
                    prev_msg = conversation[i-1]
                    prev_content = prev_msg.get('content', '')[:200]
                    if prev_msg.get('role') == 'user' and role == 'assistant':
                        context_hint = f"Responding to: {prev_content}"
                    elif prev_msg.get('role') == 'assistant' and role == 'user':
                        context_hint = f"Following up on: {prev_content}"
                
                messages_to_process.append({
                    'content': content[:2000],  # Limit for faster processing
                    'event_type': event_type,
                    'actor': actor,
                    'conversation_id': conversation_id,
                    'model_name': model_name,
                    'language': language,
                    'turn_index': i,
                    'turn_count': row.get('turn', ''),
                    'context_hint': context_hint
                })
            
            if not messages_to_process:
                return memories_to_insert
            
            # Batch extract 5W1H 
            if self.use_llm:
                # Batch process LLM extraction for efficiency
                print(f"  Extracting 5W1H for {len(messages_to_process)} messages via LLM...")
                memories_batch = self._batch_create_llm_memories(messages_to_process)
                memories_to_insert.extend(memories_batch)
            else:
                # Use quick heuristics for each message
                for msg_data in messages_to_process:
                    memory = self._create_quick_memory(msg_data)
                    if memory:
                        memories_to_insert.append(memory)
            
        except Exception:
            pass  # Silently skip failed conversations
        
        return memories_to_insert
    
    def _batch_create_llm_memories(self, messages_data: List[Dict[str, Any]]) -> List[MemoryRecord]:
        """
        Create memory objects using LLM for proper 5W1H extraction in batches.
        This is more efficient than calling LLM for each message individually.
        """
        memories = []
        batch_size = 5  # Process 5 messages at a time with LLM (reduced for reliability)
        
        for i in range(0, len(messages_data), batch_size):
            batch = messages_data[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(messages_data) + batch_size - 1) // batch_size
            
            print(f"    LLM batch {batch_num}/{total_batches}: Processing {len(batch)} messages...")
            
            # Prepare batch prompt for LLM
            batch_prompt = self._prepare_batch_prompt(batch)
            
            # Call LLM once for the batch
            llm_start = time.time()
            try:
                parsed_batch = self._call_llm_batch(batch_prompt)
                llm_time = time.time() - llm_start
                if parsed_batch:
                    print(f"    ✓ LLM responded in {llm_time:.2f}s")
                    # Process each result
                    for j, msg_data in enumerate(batch):
                        if j < len(parsed_batch):
                            memory = self._create_memory_from_parsed(msg_data, parsed_batch[j])
                            if memory:
                                memories.append(memory)
                            else:
                                print(f"      Failed to create memory from parsed item {j}")
                        else:
                            # Fallback if not enough results
                            memory = self._create_quick_memory(msg_data)
                            if memory:
                                memories.append(memory)
                else:
                    # Fallback to quick extraction for this batch
                    print(f"    LLM batch returned no results, using quick extraction for {len(batch)} messages")
                    for msg_data in batch:
                        memory = self._create_quick_memory(msg_data)
                        if memory:
                            memories.append(memory)
            except Exception as e:
                # Fallback to quick extraction for this batch
                print(f"    LLM batch failed: {e}, using quick extraction for {len(batch)} messages")
                for msg_data in batch:
                    memory = self._create_quick_memory(msg_data)
                    if memory:
                        memories.append(memory)
        
        return memories
    
    def _prepare_batch_prompt(self, batch: List[Dict[str, Any]]) -> str:
        """Prepare a batch prompt for LLM extraction."""
        prompt = """Process these messages and return a JSON array with 5W1H extraction for each.

Example output format:
[
  {
    "who": {"type": "user", "id": "user123"},
    "what": "asked about weather",
    "when": "2024-01-01T12:00:00",
    "where": {"type": "digital", "value": "conversation"},
    "why": "seeking information",
    "how": "direct query"
  }
]

Messages:
"""
        for i, msg_data in enumerate(batch):
            prompt += f"\n{i+1}. "
            prompt += f"Type: {msg_data['event_type']}, "
            prompt += f"Actor: {msg_data['actor']}, "
            prompt += f"Content: {msg_data['content'][:200]}"
        
        prompt += "\n\nReturn JSON array only, no other text:"
        return prompt
    
    def _call_llm_batch(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        """Call LLM for batch extraction."""
        import requests
        
        # Use session for connection pooling
        if not hasattr(self, '_session'):
            self._session = requests.Session()
            # Keep connections alive
            self._session.keep_alive = True
            
        url = f"{cfg.get('llm_base_url', 'http://localhost:8001/v1')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        body = {
            "model": cfg.get('llm_model', 'local-model'),
            "messages": [
                {"role": "system", "content": "You are a structured-information extractor that converts interactions into 5W1H fields. Return ONLY valid JSON arrays."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 4000  # Increased for batch responses
        }
        try:
            r = self._session.post(url, headers=headers, json=body, timeout=120)  # Use session
            r.raise_for_status()
            response_data = r.json()
            out = response_data["choices"][0]["message"]["content"]
            
            # Try to parse JSON array
            start = out.find('[')
            end = out.rfind(']')
            if start >= 0 and end > start:
                try:
                    obj = json.loads(out[start:end+1])
                    return obj
                except json.JSONDecodeError as e:
                    print(f"    JSON parse error: {e}")
                    print(f"    Attempted to parse: {out[start:min(start+200, end+1)]}...")
                    return None
            else:
                print(f"    No JSON array found in response")
                return None
        except requests.exceptions.RequestException as e:
            print(f"    LLM request failed: {e}")
            return None
        except Exception as e:
            print(f"    Unexpected error in LLM batch call: {e}")
            return None
    
    def _create_memory_from_parsed(self, msg_data: Dict[str, Any], parsed: Dict[str, Any]) -> Optional[MemoryRecord]:
        """Create a memory from parsed LLM output."""
        try:
            content = msg_data['content']
            
            # Extract fields from parsed data
            who_data = parsed.get('who', {})
            who_id = who_data.get('id', msg_data['actor'])
            # Clean up the id if it has the prefix
            if ':' in who_id:
                who_id = who_id.split(':', 1)[1]
            
            who = Who(
                type=who_data.get('type', 'user' if 'user:' in msg_data['actor'] else 'llm'),
                id=who_id,
                label=who_data.get('label')
            )
            
            what = parsed.get('what', content[:100] + ('...' if len(content) > 100 else ''))
            # Handle various when formats
            when_str = parsed.get('when', '')
            if when_str in ['unknown', '', None]:
                when = datetime.now()
            else:
                try:
                    when = datetime.fromisoformat(when_str)
                except:
                    when = datetime.now()
            
            where_data = parsed.get('where', {})
            where = Where(
                type=where_data.get('type', 'digital'),
                value=where_data.get('value', f"conversation:{msg_data['conversation_id']}")
            )
            
            why = parsed.get('why', msg_data['context_hint'] or "conversation exchange")
            how = parsed.get('how', f"via {msg_data['model_name']}" if 'llm:' in msg_data['actor'] else "user input")
            
            # Create memory object
            memory = MemoryRecord(
                session_id=msg_data['conversation_id'],
                source_event_id=f"lmsys_{msg_data['conversation_id']}_{msg_data['turn_index']}",
                who=who,
                what=what,
                when=when,
                where=where,
                why=why,
                how=how,
                raw_text=content,
                token_count=len(content.split()),  # Rough estimate
                embed_text=content[:500],  # First 500 chars for embedding
                embed_model='qwen3-embedding',
                extra={
                    'model': msg_data['model_name'],
                    'language': msg_data['language'],
                    'turn_index': msg_data['turn_index'],
                    'turn_count': msg_data['turn_count'],
                    'source': 'lmsys_dataset',
                    'event_type': msg_data['event_type']
                }
            )
            
            # Don't set embedding on memory - it will be handled separately
            
            return memory
            
        except Exception as e:
            # Fallback to quick extraction
            print(f"      Error creating memory from parsed data: {e}")
            import traceback
            traceback.print_exc()
            return self._create_quick_memory(msg_data)
    
    def _create_llm_memory(self, msg_data: Dict[str, Any]) -> Optional[MemoryRecord]:
        """
        Create a memory object using LLM for proper 5W1H extraction.
        This is slower but more accurate for benchmark testing.
        """
        try:
            content = msg_data['content']
            
            # Create RawEvent for LLM extraction
            raw_event = RawEvent(
                session_id=msg_data['conversation_id'],
                event_type=msg_data['event_type'],
                actor=msg_data['actor'],
                content=content,
                metadata={
                    'model': msg_data['model_name'],
                    'language': msg_data['language'],
                    'turn_index': msg_data['turn_index'],
                    'turn_count': msg_data['turn_count'],
                    'source': 'lmsys_dataset'
                }
            )
            
            # Use LLM to extract 5W1H
            memory = extract_5w1h(raw_event, context_hint=msg_data['context_hint'])
            
            # Don't set embedding on memory - it will be handled separately
            
            return memory
            
        except Exception:
            # Fall back to quick extraction if LLM fails
            return self._create_quick_memory(msg_data)
    
    def _create_quick_memory(self, msg_data: Dict[str, Any]) -> Optional[MemoryRecord]:
        """
        Create a memory object quickly without full LLM extraction.
        Uses heuristics for 5W1H extraction.
        """
        try:
            content = msg_data['content']
            
            # Quick 5W1H extraction using heuristics
            who = msg_data['actor']
            what = content[:100] + ('...' if len(content) > 100 else '')
            when = datetime.now().isoformat()
            where = f"conversation:{msg_data['conversation_id']}"
            why = msg_data['context_hint'] if msg_data['context_hint'] else "conversation exchange"
            how = f"via {msg_data['model_name']}" if 'llm:' in msg_data['actor'] else "user input"
            
            # Create memory object
            memory = MemoryRecord(
                session_id=msg_data['conversation_id'],
                source_event_id=f"lmsys_{msg_data['conversation_id']}_{msg_data['turn_index']}",
                who=Who(type='user' if 'user:' in who else 'llm', id=who.split(':')[1] if ':' in who else who),
                what=what,
                when=datetime.fromisoformat(when),
                where=Where(type='digital', value=where),
                why=why,
                how=how,
                raw_text=content,
                token_count=len(content.split()),  # Rough estimate
                embed_text=content[:500],  # First 500 chars for embedding
                embed_model='qwen3-embedding',
                extra={
                    'model': msg_data['model_name'],
                    'language': msg_data['language'],
                    'turn_index': msg_data['turn_index'],
                    'turn_count': msg_data['turn_count'],
                    'source': 'lmsys_dataset',
                    'event_type': msg_data['event_type']
                }
            )
            
            # Don't set embedding on memory - it will be handled separately
            
            return memory
            
        except Exception:
            return None
    
    def import_from_csv(self, limit: int = None) -> Dict[str, Any]:
        """
        Import conversations from CSV file with optimized batch processing.
        
        Args:
            limit: Maximum number of conversations to import (None for all)
            
        Returns:
            Import statistics
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        start_time = time.time()
        
        print(f"Starting optimized import from {self.csv_path}")
        print(f"Batch size: {self.batch_size}")
        print(f"Worker threads: {self.num_workers}")
        if limit:
            print(f"Limiting to {limit} conversations")
        
        # Count total rows for progress tracking
        total_rows = 0
        if not limit:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in csv.DictReader(f))
                print(f"Total conversations in file: {total_rows}")
        
        with open(self.csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            batch = []
            last_progress_time = time.time()
            
            for row_num, row in enumerate(reader):
                if limit and row_num >= limit:
                    break
                
                batch.append(row)
                
                # Process batch when full
                if len(batch) >= self.batch_size:
                    batch_start = time.time()
                    self._process_batch_optimized(batch)
                    batch_time = time.time() - batch_start
                    self.stats['batch_times'].append(batch_time)
                    batch = []
                    
                    # Print progress every 5 seconds
                    current_time = time.time()
                    if current_time - last_progress_time >= 5:
                        self._print_progress(row_num, total_rows or limit, start_time)
                        last_progress_time = current_time
            
            # Process remaining batch
            if batch:
                self._process_batch_optimized(batch)
        
        # Final cleanup and stats
        self.stats['processing_time'] = time.time() - start_time
        self._finalize_import()
        
        return self.stats
    
    def _print_progress(self, current: int, total: int, start_time: float):
        """Print detailed progress information."""
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        
        avg_batch_time = sum(self.stats['batch_times'][-10:]) / min(10, len(self.stats['batch_times'])) if self.stats['batch_times'] else 0
        
        print(f"\n[Progress] {current}/{total} conversations ({100*current/total:.1f}%)")
        print(f"  Rate: {rate:.1f} conv/sec | ETA: {eta/60:.1f} min")
        print(f"  Memories created: {self.stats['total_memories_created']}")
        print(f"  Success rate: {100*self.stats['successful_imports']/(current+1):.1f}%")
        print(f"  Avg batch time: {avg_batch_time:.2f}s")
    
    def _process_batch_optimized(self, batch: List[Dict[str, Any]]):
        """Process a batch of conversations with optimized batch operations."""
        
        batch_start_time = time.time()
        
        # Prepare all memories first
        all_memories = []
        
        if self.use_llm:
            # For LLM mode, collect all messages from all conversations first
            all_messages_data = []
            conversation_info = []  # Track which messages belong to which conversation
            
            for row in batch:
                try:
                    # Parse conversation and collect messages
                    conversation = self.parse_conversation(row.get('conversation', ''))
                    if not conversation:
                        self.stats['failed_imports'] += 1
                        self.stats['total_conversations'] += 1
                        continue
                    
                    # Get conversation metadata
                    conversation_id = row.get('conversation_id', '')
                    model_name = row.get('model', 'unknown')
                    language = row.get('language', 'English')
                    
                    messages_for_this_conv = []
                    
                    for i, message in enumerate(conversation):
                        role = message.get('role', 'unknown')
                        content = message.get('content', '')
                        
                        if not content or role not in ['user', 'assistant']:
                            continue
                        
                        # Determine event type and actor
                        if role == 'user':
                            event_type = 'user_message'
                            actor = f"user:lmsys_user"
                        else:
                            event_type = 'llm_message'
                            actor = f"llm:{model_name}"
                        
                        # Build context hint
                        context_hint = ""
                        if i > 0 and i-1 < len(conversation):
                            prev_msg = conversation[i-1]
                            prev_content = prev_msg.get('content', '')[:200]
                            if prev_msg.get('role') == 'user' and role == 'assistant':
                                context_hint = f"Responding to: {prev_content}"
                            elif prev_msg.get('role') == 'assistant' and role == 'user':
                                context_hint = f"Following up on: {prev_content}"
                        
                        msg_data = {
                            'content': content[:2000],  # Limit for faster processing
                            'event_type': event_type,
                            'actor': actor,
                            'conversation_id': conversation_id,
                            'model_name': model_name,
                            'language': language,
                            'turn_index': i,
                            'turn_count': row.get('turn', ''),
                            'context_hint': context_hint
                        }
                        
                        all_messages_data.append(msg_data)
                        messages_for_this_conv.append(msg_data)
                    
                    if messages_for_this_conv:
                        conversation_info.append({
                            'conversation_id': conversation_id,
                            'message_count': len(messages_for_this_conv)
                        })
                        self.stats['successful_imports'] += 1
                    else:
                        self.stats['failed_imports'] += 1
                    
                    self.stats['total_conversations'] += 1
                    
                except Exception:
                    self.stats['failed_imports'] += 1
                    self.stats['total_conversations'] += 1
            
            # Now batch process all messages from all conversations together
            if all_messages_data:
                parse_start = time.time()
                print(f"\n  Processing batch of {len(conversation_info)} conversations with {len(all_messages_data)} total messages...")
                for info in conversation_info:
                    print(f"    Conv {info['conversation_id'][:20]}...: {info['message_count']} messages")
                
                extraction_start = time.time()
                memories_batch = self._batch_create_llm_memories(all_messages_data)
                all_memories.extend(memories_batch)
                extraction_time = time.time() - extraction_start
                print(f"  ✓ Created {len(memories_batch)} memories from batch (5W1H extraction: {extraction_time:.2f}s)")
            else:
                print(f"  No messages to process from this batch of conversations")
        else:
            # For quick mode, use parallel processing
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.prepare_memories_from_conversation, row) for row in batch]
                
                for future in as_completed(futures):
                    try:
                        memories = future.result(timeout=5)
                        if memories:
                            all_memories.extend(memories)
                            self.stats['successful_imports'] += 1
                        else:
                            self.stats['failed_imports'] += 1
                    except Exception:
                        self.stats['failed_imports'] += 1
                    
                    self.stats['total_conversations'] += 1
        
        if not all_memories:
            return
        
        # Batch generate embeddings
        embed_texts = [m.embed_text for m in all_memories]
        print(f"\n  Step 1/3: Generating embeddings for {len(embed_texts)} memories...")
        memory_embeddings = []  # Keep embeddings separate
        if embed_texts:
            embed_start = time.time()
            try:
                embeddings = self.embedder.encode(embed_texts, batch_size=32, show_progress_bar=False)
                embed_time = time.time() - embed_start
                print(f"  ✓ Generated {len(embeddings)} embeddings ({embed_time:.2f}s, {len(embeddings)/embed_time:.1f} emb/s)")
                memory_embeddings = embeddings  # Store them separately
                    
            except Exception as e:
                print(f"  ✗ Failed to generate embeddings: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Batch insert into database
        print(f"\n  Step 2/3: Inserting memories into database...")
        db_start = time.time()
        try:
            self._batch_insert_memories(all_memories)
            db_time = time.time() - db_start
            print(f"  ✓ Inserted {len(all_memories)} memories into database ({db_time:.2f}s, {len(all_memories)/db_time:.1f} mem/s)")
        except Exception as e:
            print(f"  ✗ Failed to insert memories into database: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Batch add to FAISS index
        print(f"\n  Step 3/3: Adding to FAISS index...")
        faiss_start = time.time()
        try:
            self._batch_add_to_index(all_memories, memory_embeddings)
            faiss_time = time.time() - faiss_start
            print(f"  ✓ Added memories to FAISS index ({faiss_time:.2f}s)")
        except Exception as e:
            print(f"  ✗ Failed to add to FAISS index: {e}")
            import traceback
            traceback.print_exc()
            return
        
        self.stats['total_memories_created'] += len(all_memories)
        total_batch_time = time.time() - batch_start_time
        
        # Print timing summary
        print(f"\n  ✓ Batch complete: {len(all_memories)} memories saved")
        print(f"  ⏱️  Total batch time: {total_batch_time:.2f}s")
        if self.use_llm and 'extraction_time' in locals():
            print(f"     - 5W1H extraction: {extraction_time:.2f}s ({extraction_time/total_batch_time*100:.1f}%)")
        if 'embed_time' in locals():
            print(f"     - Embedding generation: {embed_time:.2f}s ({embed_time/total_batch_time*100:.1f}%)")
        if 'db_time' in locals():
            print(f"     - Database insert: {db_time:.2f}s ({db_time/total_batch_time*100:.1f}%)")
        if 'faiss_time' in locals():
            print(f"     - FAISS indexing: {faiss_time:.2f}s ({faiss_time/total_batch_time*100:.1f}%)")
        print("")
    
    def _batch_insert_memories(self, memories: List[MemoryRecord]):
        """Batch insert memories into database."""
        if not memories:
            return
        
        cursor = self.db_conn.cursor()
        
        # Prepare batch insert data
        insert_data = []
        for memory in memories:
            insert_data.append((
                memory.memory_id,
                memory.session_id,
                memory.source_event_id,
                memory.who.type,
                memory.who.id,
                memory.who.label,
                memory.what,
                memory.when.isoformat(),
                memory.where.type,
                memory.where.value,
                memory.where.lat,
                memory.where.lon,
                memory.why,
                memory.how,
                memory.raw_text,
                memory.token_count,
                memory.embed_model,
                json.dumps(memory.extra) if memory.extra else '{}',
                datetime.now().isoformat()
            ))
        
        # Batch insert with IGNORE to skip duplicates
        cursor.executemany(
            '''INSERT OR IGNORE INTO memories 
               (memory_id, session_id, source_event_id, who_type, who_id, who_label,
                what, when_ts, where_type, where_value, where_lat, where_lon,
                why, how, raw_text, token_count, embed_model, extra_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            insert_data
        )
        
        self.db_conn.commit()
    
    def _batch_add_to_index(self, memories: List[MemoryRecord], embeddings: np.ndarray):
        """Batch add embeddings to FAISS index."""
        if not memories or len(embeddings) == 0:
            return
        
        valid_embeddings = []
        valid_memory_ids = []
        
        for i, m in enumerate(memories):
            if i < len(embeddings):
                embedding = np.array(embeddings[i], dtype='float32')
                # Ensure 1D array
                if len(embedding.shape) > 1:
                    embedding = embedding.squeeze()
                # Check embedding dimension
                if embedding.shape[0] != self.index.dim:
                    print(f"    Warning: Embedding dimension mismatch. Expected {self.index.dim}, got {embedding.shape[0]}")
                    continue
                valid_embeddings.append(embedding)
                valid_memory_ids.append(m.memory_id)
        
        if len(valid_embeddings) > 0:
            try:
                # Add all embeddings at once
                for i, (embedding, memory_id) in enumerate(zip(valid_embeddings, valid_memory_ids)):
                    try:
                        self.index.add(memory_id, embedding)
                    except AssertionError as e:
                        print(f"    FAISS dimension error for memory {i}: index expects dim={self.index.dim}, got shape={embedding.shape}")
                        if i == 0:
                            # Only show detailed error for first failure
                            print(f"    First few values of embedding: {embedding[:5]}")
                            print(f"    FAISS index.d = {self.index.index.d}")
                            # Try to understand the shape issue
                            vec_test = embedding.astype('float32')[None, :]
                            print(f"    After [None, :] shape would be: {vec_test.shape}")
                        raise
                
                # Save index periodically (every 10000 memories)
                if self.stats['total_memories_created'] % 10000 == 0:
                    self.index.save()
                    
            except Exception as e:
                print(f"Failed to add to FAISS index: {e}")
                import traceback
                traceback.print_exc()
    
    def _finalize_import(self):
        """Finalize the import process."""
        # Final save of FAISS index
        self.index.save()
        
        # Close database connection
        self.db_conn.close()
        
        # Print final summary
        print("\n" + "="*60)
        print("IMPORT COMPLETED - PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total conversations processed: {self.stats['total_conversations']}")
        print(f"Successfully imported: {self.stats['successful_imports']}")
        print(f"Failed imports: {self.stats['failed_imports']}")
        print(f"Total memories created: {self.stats['total_memories_created']}")
        print(f"Total processing time: {self.stats['processing_time']:.1f} seconds")
        
        if self.stats['processing_time'] > 0:
            conv_rate = self.stats['total_conversations'] / self.stats['processing_time']
            mem_rate = self.stats['total_memories_created'] / self.stats['processing_time']
            print(f"\nThroughput:")
            print(f"  - Conversations: {conv_rate:.2f}/sec ({conv_rate*60:.1f}/min)")
            print(f"  - Memories: {mem_rate:.2f}/sec ({mem_rate*60:.1f}/min)")
        
        if self.stats['successful_imports'] > 0:
            avg_memories = self.stats['total_memories_created'] / self.stats['successful_imports']
            print(f"\nStatistics:")
            print(f"  - Avg memories per conversation: {avg_memories:.2f}")
            print(f"  - Success rate: {100*self.stats['successful_imports']/self.stats['total_conversations']:.1f}%")
        
        if self.stats['batch_times']:
            avg_batch = sum(self.stats['batch_times']) / len(self.stats['batch_times'])
            min_batch = min(self.stats['batch_times'])
            max_batch = max(self.stats['batch_times'])
            print(f"\nBatch timing:")
            print(f"  - Average: {avg_batch:.2f}s")
            print(f"  - Min: {min_batch:.2f}s")
            print(f"  - Max: {max_batch:.2f}s")


def main():
    """Main entry point for LMSYS data import."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import LMSYS conversation data into JAM')
    parser.add_argument('csv_path', help='Path to LMSYS CSV file')
    parser.add_argument('--batch-size', type=int, default=100, 
                       help='Number of conversations to process at once (default: 100)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of conversations to import')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - import only 10 conversations')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode - import 100 conversations for testing')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use LLM for 5W1H extraction (slower but more accurate for benchmarks)')
    
    args = parser.parse_args()
    
    if args.test:
        args.limit = 10
        args.batch_size = 10
        print("TEST MODE: Importing only 10 conversations")
    elif args.quick:
        args.limit = 100
        print("QUICK MODE: Importing 100 conversations")
    
    # Create importer and run
    importer = LMSYSImporter(
        args.csv_path, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_llm=args.use_llm
    )
    
    if args.use_llm:
        print("LLM MODE: Using LLM for 5W1H extraction (more accurate but slower)")
        print("IMPORTANT: Ensure both servers are running:")
        print("  - LLM server on port 8000 (for extraction)")
        print("  - Embedding server on port 8002 (for embeddings)")
        print("  Run: python llama_server_manager.py both start")
        print("  Or: python -m agentic_memory.cli server start --all --daemon")
        
        # Check if servers are running
        import requests
        try:
            llm_response = requests.get("http://localhost:8000/health", timeout=2)
            if llm_response.status_code == 200:
                print("LLM server is running on port 8000")
            else:
                print("LLM server is not healthy on port 8000")
        except:
            print("LLM server is not running on port 8000")
            print("   Please start it with: python llama_server_manager.py llm start")
        
        try:
            emb_response = requests.get("http://localhost:8002/health", timeout=2)
            if emb_response.status_code == 200:
                print("Embedding server is running on port 8002")
            else:
                print("Embedding server is not healthy on port 8002")
        except:
            print("Embedding server is not running on port 8002")
            print("   Please start it with: python llama_server_manager.py embedding start")
    
    stats = importer.import_from_csv(limit=args.limit)


if __name__ == '__main__':
    main()

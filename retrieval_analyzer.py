#!/usr/bin/env python3
"""
Interactive Retrieval Analyzer for JAM Memory System
Allows real-time weight adjustment and result analysis
"""

import sys
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Any
import json
from dataclasses import dataclass, asdict
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading

sys.path.insert(0, str(Path(__file__).parent))

from agentic_memory.config import cfg
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.embedding import get_llama_embedder
from agentic_memory.retrieval import HybridRetriever
from agentic_memory.types import RetrievalQuery

@dataclass
class ScoredMemory:
    """Memory with all component scores broken down"""
    memory_id: str
    raw_text: str
    who: str
    what: str
    when: str
    where: str
    why: str
    how: str
    
    # Component scores
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    recency_score: float = 0.0
    actor_score: float = 0.0
    temporal_score: float = 0.0
    usage_score: float = 0.0
    
    # Final weighted score
    total_score: float = 0.0
    original_rank: int = 0
    current_rank: int = 0

class RetrievalAnalyzer:
    """Main analyzer class for retrieval testing"""
    
    def __init__(self):
        """Initialize the retrieval system components"""
        print("Initializing retrieval analyzer...")
        
        # Initialize storage and retrieval components
        self.store = MemoryStore(cfg.db_path)
        
        # Load existing FAISS index directly
        import faiss
        import os
        
        # Create a simple wrapper that loads the existing index
        class ExistingFaissIndex:
            def __init__(self, index_path):
                self.index_path = index_path
                self.index = faiss.read_index(index_path)
                self.id_map = []
                
                # Load the ID map
                map_path = index_path + '.map'
                if os.path.exists(map_path):
                    with open(map_path, 'r', encoding='utf-8') as f:
                        self.id_map = [line.strip() for line in f]
                
                print(f"Loaded FAISS index with {self.index.ntotal} vectors, dimension {self.index.d}")
                
            def search(self, qvec, k):
                qvec = qvec.astype('float32')[None, :]
                scores, idxs = self.index.search(qvec, k)
                results = []
                for score, idx in zip(scores[0], idxs[0]):
                    if 0 <= idx < len(self.id_map):
                        memory_id = self.id_map[idx]
                        if memory_id:  # Skip empty entries
                            results.append((memory_id, float(score)))
                return results
        
        # Use the existing index
        self.index = ExistingFaissIndex(cfg.index_path)
        
        # Update cfg to match the actual dimension
        cfg.embed_dim = self.index.index.d
        
        self.embedder = get_llama_embedder()
        
        # Initialize retriever with existing configuration
        self.retriever = HybridRetriever(self.store, self.index)
        
        # Current weights (start with defaults)
        self.weights = {
            'semantic': 0.4,
            'lexical': 0.2,
            'recency': 0.1,
            'actor': 0.1,
            'temporal': 0.1,
            'usage': 0.1
        }
        
        # Store current results
        self.current_query = ""
        self.scored_memories: List[ScoredMemory] = []
        self.original_order: List[str] = []  # Original memory IDs in order
        
        # Count memories in database
        with self.store.connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
        
        print(f"Loaded {count} memories from database")
        print("Analyzer ready!")
    
    def decompose_query(self, query: str) -> Dict[str, Any]:
        """Decompose query into 5W1H components using LLM"""
        print(f"Decomposing query with LLM: '{query}'")
        
        # Use the LLM extractor to decompose the query
        from agentic_memory.extraction.llm_extractor import extract_5w1h
        
        # Create a raw event from the query
        from agentic_memory.types import RawEvent
        raw_event = RawEvent(
            session_id='analyzer',
            event_type='user_message',  # Use valid event type
            actor='user:analyzer',
            content=query,
            metadata={}
        )
        
        # Extract 5W1H components using LLM
        try:
            extracted = extract_5w1h(raw_event)
            # Handle both dict and object returns
            if hasattr(extracted, '__dict__'):
                # It's an object (MemoryRecord), access attributes directly
                components = {
                    'who': getattr(extracted, 'who_id', None),
                    'what': getattr(extracted, 'what', query),  # Fallback to original query
                    'when': getattr(extracted, 'when_raw', None),  # Use raw temporal info
                    'where': getattr(extracted, 'where_value', None),
                    'why': getattr(extracted, 'why', None),
                    'how': getattr(extracted, 'how', None)
                }
            else:
                # It's a dict
                components = {
                    'who': extracted.get('who_id'),
                    'what': extracted.get('what', query),  
                    'when': extracted.get('when_raw'),  
                    'where': extracted.get('where_value'),
                    'why': extracted.get('why'),
                    'how': extracted.get('how')
                }
            print(f"LLM extraction result: {components}")
        except Exception as e:
            print(f"LLM extraction failed: {e}, using fallback")
            components = {
                'who': None,
                'what': query,
                'when': None,
                'where': None,
                'why': None,
                'how': None
            }
        
        # Also extract temporal hints using temporal parser
        from agentic_memory.extraction.temporal_parser import TemporalParser
        parser = TemporalParser()
        temporal_hint, cleaned_text = parser.extract_and_clean_query(query)
        if temporal_hint and not components['when']:
            components['when'] = temporal_hint
            components['what'] = cleaned_text
        
        return components
    
    def perform_retrieval(self, query: str, top_k: int = 100, components: Optional[Dict[str, Any]] = None) -> List[ScoredMemory]:
        """Perform retrieval and score all components using the full pipeline"""
        print(f"\nPerforming retrieval for: '{query}'")
        
        # Decompose query using LLM if not already done
        if components is None:
            components = self.decompose_query(query)
        print(f"Query components: {components}")
        
        # Create retrieval query with extracted components
        rq = RetrievalQuery(
            session_id='analyzer',
            text=components.get('what', query),  # Use extracted 'what' or fallback to query
            actor_hint=components.get('who'),
            temporal_hint=components.get('when')
        )
        
        # Get query embedding
        qvec = self.embedder.encode([query], normalize_embeddings=True)[0]
        
        # Use the retriever's search method to get all candidates with scores
        print("Running hybrid retrieval with all signals...")
        candidates = self.retriever.search(rq, qvec, topk_sem=top_k, topk_lex=top_k)
        
        print(f"Retrieved {len(candidates)} candidates")
        
        # Convert candidates to ScoredMemory objects
        scored_memories = []
        memory_ids = [c.memory_id for c in candidates[:top_k]]
        
        if not memory_ids:
            return []
        
        # Fetch full memory details
        memories = self.store.fetch_memories(memory_ids)
        memory_dict = {m['memory_id']: m for m in memories}
        
        # Convert candidates to ScoredMemory objects with component scores
        for candidate in candidates[:top_k]:
            memory = memory_dict.get(candidate.memory_id)
            if not memory:
                continue
            
            # Create scored memory using component scores from the Candidate
            scored_mem = ScoredMemory(
                memory_id=candidate.memory_id,
                raw_text=memory['raw_text'] or '',
                who=memory['who_id'] or '',
                what=memory['what'] or '',
                when=memory['when_ts'] or '',
                where=memory['where_value'] or '',
                why=memory['why'] or '',
                how=memory['how'] or '',
                # Use scores from the Candidate object if available
                semantic_score=candidate.semantic_score if candidate.semantic_score is not None else 0.0,
                lexical_score=candidate.lexical_score if candidate.lexical_score is not None else 0.0,
                recency_score=candidate.recency_score if candidate.recency_score is not None else 0.0,
                actor_score=candidate.actor_score if candidate.actor_score is not None else 0.0,
                temporal_score=candidate.temporal_score if candidate.temporal_score is not None else 0.0,
                usage_score=candidate.usage_score if candidate.usage_score is not None else 0.0,
                total_score=candidate.score  # Use the total score from retriever
            )
            scored_memories.append(scored_mem)
        
        # Sort by total score
        scored_memories.sort(key=lambda x: x.total_score, reverse=True)
        
        # Set ranks
        for i, mem in enumerate(scored_memories):
            mem.original_rank = i + 1
            mem.current_rank = i + 1
        
        # Store results
        self.current_query = query
        self.scored_memories = scored_memories[:top_k]  # Keep top K
        self.original_order = [m.memory_id for m in self.scored_memories]
        
        print(f"Retrieval complete. Top memory: {self.scored_memories[0].memory_id if self.scored_memories else 'None'}")
        return self.scored_memories
    
    def calculate_weighted_score(self, memory: ScoredMemory) -> float:
        """Calculate weighted score based on current weights"""
        score = (
            self.weights['semantic'] * memory.semantic_score +
            self.weights['lexical'] * memory.lexical_score +
            self.weights['recency'] * memory.recency_score +
            self.weights['actor'] * memory.actor_score +
            self.weights['temporal'] * memory.temporal_score +
            self.weights['usage'] * memory.usage_score
        )
        return score
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update weights and recalculate scores"""
        # Normalize weights to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in new_weights.items()}
        else:
            self.weights = new_weights
        
        # Recalculate scores
        for mem in self.scored_memories:
            mem.total_score = self.calculate_weighted_score(mem)
        
        # Re-sort
        self.scored_memories.sort(key=lambda x: x.total_score, reverse=True)
        
        # Update current ranks
        for i, mem in enumerate(self.scored_memories):
            mem.current_rank = i + 1
    
    def get_rank_changes(self) -> List[Tuple[str, int, int, int]]:
        """Get rank changes: (memory_id, original_rank, current_rank, change)"""
        changes = []
        for mem in self.scored_memories:
            change = mem.original_rank - mem.current_rank
            changes.append((mem.memory_id, mem.original_rank, mem.current_rank, change))
        return sorted(changes, key=lambda x: abs(x[3]), reverse=True)

class RetrievalAnalyzerGUI:
    """GUI for the retrieval analyzer"""
    
    def __init__(self, analyzer: RetrievalAnalyzer):
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("JAM Retrieval Analyzer - 5W1H Memory Analysis")
        self.root.geometry("1600x900")
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables for sliders
        self.weight_vars = {
            'semantic': tk.DoubleVar(value=0.4),
            'lexical': tk.DoubleVar(value=0.2),
            'recency': tk.DoubleVar(value=0.1),
            'actor': tk.DoubleVar(value=0.1),
            'temporal': tk.DoubleVar(value=0.1),
            'usage': tk.DoubleVar(value=0.1)
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Query input section
        query_frame = ttk.LabelFrame(main_frame, text="Query Input", padding="10")
        query_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        query_frame.columnconfigure(0, weight=1)
        
        # Query input row
        input_row = ttk.Frame(query_frame)
        input_row.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.query_entry = ttk.Entry(input_row, width=80)
        self.query_entry.grid(row=0, column=0, padx=5)
        self.query_entry.insert(0, "What do you remember about Python scripts?")
        
        self.search_button = ttk.Button(input_row, text="Search", command=self.perform_search)
        self.search_button.grid(row=0, column=1, padx=5)
        
        self.top_k_var = tk.IntVar(value=50)
        ttk.Label(input_row, text="Top K:").grid(row=0, column=2, padx=5)
        ttk.Spinbox(input_row, from_=10, to=200, textvariable=self.top_k_var, width=10).grid(row=0, column=3, padx=5)
        
        # Decomposed query display
        decomp_frame = ttk.Frame(query_frame)
        decomp_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(decomp_frame, text="Decomposed Query:", font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=0, padx=5, sticky=tk.W)
        
        # Create labels for each component
        self.decomp_labels = {}
        components = ['WHO', 'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW']
        for i, comp in enumerate(components):
            row = i // 3
            col = (i % 3) * 2
            
            ttk.Label(decomp_frame, text=f"{comp}:", width=8).grid(row=row+1, column=col, padx=5, sticky=tk.W)
            label = ttk.Label(decomp_frame, text="", foreground="blue", width=40)
            label.grid(row=row+1, column=col+1, padx=5, sticky=tk.W)
            self.decomp_labels[comp.lower()] = label
        
        # Weight adjustment section
        weights_frame = ttk.LabelFrame(main_frame, text="Component Weights", padding="10")
        weights_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        row_idx = 0
        for name, var in self.weight_vars.items():
            ttk.Label(weights_frame, text=f"{name.capitalize()}:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
            
            slider = ttk.Scale(weights_frame, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL, length=200)
            slider.grid(row=row_idx, column=1, pady=2, padx=5)
            slider.bind("<ButtonRelease-1>", lambda e: self.update_weights())
            
            label = ttk.Label(weights_frame, text=f"{var.get():.2f}")
            label.grid(row=row_idx, column=2, pady=2)
            
            # Update label when slider changes
            var.trace('w', lambda *args, l=label, v=var: l.config(text=f"{v.get():.2f}"))
            
            row_idx += 1
        
        # Normalize button
        ttk.Button(weights_frame, text="Normalize Weights", command=self.normalize_weights).grid(row=row_idx, column=0, columnspan=3, pady=10)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create treeview for results
        columns = ('Rank', 'ΔRank', 'ID', 'Total', 'Sem', 'Lex', 'Rec', 'Act', 'Tmp', 'Use', 
                  'Who', 'What', 'When', 'Where', 'Why', 'How', 'Text')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=20)
        
        # Define column headings and widths
        self.results_tree.heading('Rank', text='Rank')
        self.results_tree.heading('ΔRank', text='ΔRank')
        self.results_tree.heading('ID', text='Memory ID')
        self.results_tree.heading('Total', text='Total')
        self.results_tree.heading('Sem', text='Semantic')
        self.results_tree.heading('Lex', text='Lexical')
        self.results_tree.heading('Rec', text='Recency')
        self.results_tree.heading('Act', text='Actor')
        self.results_tree.heading('Tmp', text='Temporal')
        self.results_tree.heading('Use', text='Usage')
        self.results_tree.heading('Who', text='Who')
        self.results_tree.heading('What', text='What')
        self.results_tree.heading('When', text='When')
        self.results_tree.heading('Where', text='Where')
        self.results_tree.heading('Why', text='Why')
        self.results_tree.heading('How', text='How')
        self.results_tree.heading('Text', text='Raw Text')
        
        self.results_tree.column('Rank', width=45)
        self.results_tree.column('ΔRank', width=45)
        self.results_tree.column('ID', width=100)
        self.results_tree.column('Total', width=55)
        self.results_tree.column('Sem', width=55)
        self.results_tree.column('Lex', width=55)
        self.results_tree.column('Rec', width=55)
        self.results_tree.column('Act', width=45)
        self.results_tree.column('Tmp', width=45)
        self.results_tree.column('Use', width=45)
        self.results_tree.column('Who', width=80)
        self.results_tree.column('What', width=150)
        self.results_tree.column('When', width=100)
        self.results_tree.column('Where', width=80)
        self.results_tree.column('Why', width=120)
        self.results_tree.column('How', width=120)
        self.results_tree.column('Text', width=300)
        
        # Scrollbars
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Detail view section
        detail_frame = ttk.LabelFrame(main_frame, text="Memory Details", padding="10")
        detail_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        self.detail_text = scrolledtext.ScrolledText(detail_frame, height=10, width=60, wrap=tk.WORD)
        self.detail_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        
        # Bind selection event
        self.results_tree.bind('<<TreeviewSelect>>', self.on_select)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    
    def perform_search(self):
        """Perform search with current query"""
        query = self.query_entry.get()
        top_k = self.top_k_var.get()
        
        if not query:
            return
        
        self.status_var.set("Searching...")
        self.root.update()
        
        # Clear previous decomposition
        for label in self.decomp_labels.values():
            label.config(text="")
        
        # Run search in thread to keep UI responsive
        def search_thread():
            # First decompose the query to get components
            components = self.analyzer.decompose_query(query)
            
            # Update decomposition display in UI thread
            self.root.after(0, lambda: self.display_decomposition(components))
            
            # Then perform retrieval with the already decomposed components
            results = self.analyzer.perform_retrieval(query, top_k, components)
            self.root.after(0, lambda: self.display_results(results))
        
        thread = threading.Thread(target=search_thread)
        thread.daemon = True
        thread.start()
    
    def display_decomposition(self, components: Dict[str, Any]):
        """Display the decomposed query components"""
        for key, label in self.decomp_labels.items():
            value = components.get(key)
            if value:
                # Truncate long values and format nicely
                if isinstance(value, dict):
                    # Handle temporal hints that are dicts
                    display_text = str(value)[:60]
                elif isinstance(value, tuple):
                    # Handle date ranges
                    display_text = f"{value[0]} to {value[1]}"
                else:
                    display_text = str(value)[:60]
                
                if len(str(value)) > 60:
                    display_text += "..."
                    
                label.config(text=display_text)
            else:
                label.config(text="(not detected)")
    
    def display_results(self, results: List[ScoredMemory]):
        """Display search results in the tree"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add new results
        for mem in results:
            rank_change = mem.original_rank - mem.current_rank
            rank_symbol = ""
            if rank_change > 0:
                rank_symbol = f"+{rank_change}"
            elif rank_change < 0:
                rank_symbol = str(rank_change)
            else:
                rank_symbol = "="
            
            # Format text preview
            text_preview = mem.raw_text[:80].replace('\n', ' ')
            if len(mem.raw_text) > 80:
                text_preview += "..."
            
            # Format 5W1H fields (truncate if needed)
            def truncate(text, length=50):
                if not text:
                    return ""
                text = text.replace('\n', ' ')
                return text[:length] + "..." if len(text) > length else text
            
            # Color code based on rank change
            tags = []
            if rank_change > 5:
                tags.append('improved')
            elif rank_change < -5:
                tags.append('declined')
            
            self.results_tree.insert('', 'end', values=(
                mem.current_rank,
                rank_symbol,
                mem.memory_id,
                f"{mem.total_score:.3f}",
                f"{mem.semantic_score:.3f}",
                f"{mem.lexical_score:.3f}",
                f"{mem.recency_score:.3f}",
                f"{mem.actor_score:.1f}",
                f"{mem.temporal_score:.1f}",
                f"{mem.usage_score:.1f}",
                truncate(mem.who, 30),
                truncate(mem.what, 60),
                mem.when[:16] if mem.when else "",  # Show date/time only
                truncate(mem.where, 30),
                truncate(mem.why, 50),
                truncate(mem.how, 50),
                text_preview
            ), tags=tags)
        
        # Configure tags for coloring
        self.results_tree.tag_configure('improved', background='#d4edda')
        self.results_tree.tag_configure('declined', background='#f8d7da')
        
        self.status_var.set(f"Found {len(results)} results for '{self.analyzer.current_query}'")
    
    def update_weights(self):
        """Update weights and refresh results"""
        new_weights = {name: var.get() for name, var in self.weight_vars.items()}
        self.analyzer.update_weights(new_weights)
        
        if self.analyzer.scored_memories:
            self.display_results(self.analyzer.scored_memories)
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = sum(var.get() for var in self.weight_vars.values())
        if total > 0:
            for var in self.weight_vars.values():
                var.set(var.get() / total)
        self.update_weights()
    
    def on_select(self, event):
        """Handle selection in results tree"""
        selection = self.results_tree.selection()
        if selection:
            item = self.results_tree.item(selection[0])
            values = item['values']
            memory_id = values[2]  # Memory ID is in column 2
            
            # Find the memory
            for mem in self.analyzer.scored_memories:
                if mem.memory_id == memory_id:
                    # Display details
                    details = f"Memory ID: {mem.memory_id}\n"
                    details += f"Original Rank: {mem.original_rank}\n"
                    details += f"Current Rank: {mem.current_rank}\n"
                    details += f"Rank Change: {mem.original_rank - mem.current_rank}\n\n"
                    details += f"WHO: {mem.who}\n"
                    details += f"WHAT: {mem.what}\n"
                    details += f"WHEN: {mem.when}\n"
                    details += f"WHERE: {mem.where}\n"
                    details += f"WHY: {mem.why}\n"
                    details += f"HOW: {mem.how}\n\n"
                    details += f"Raw Text:\n{mem.raw_text}\n\n"
                    details += "Component Scores:\n"
                    details += f"  Semantic: {mem.semantic_score:.4f} (weight: {self.analyzer.weights['semantic']:.2f})\n"
                    details += f"  Lexical: {mem.lexical_score:.4f} (weight: {self.analyzer.weights['lexical']:.2f})\n"
                    details += f"  Recency: {mem.recency_score:.4f} (weight: {self.analyzer.weights['recency']:.2f})\n"
                    details += f"  Actor: {mem.actor_score:.4f} (weight: {self.analyzer.weights['actor']:.2f})\n"
                    details += f"  Temporal: {mem.temporal_score:.4f} (weight: {self.analyzer.weights['temporal']:.2f})\n"
                    details += f"  Usage: {mem.usage_score:.4f} (weight: {self.analyzer.weights['usage']:.2f})\n"
                    details += f"\nTotal Score: {mem.total_score:.4f}"
                    
                    self.detail_text.delete(1.0, tk.END)
                    self.detail_text.insert(1.0, details)
                    break
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def main():
    """Main entry point"""
    print("Starting JAM Retrieval Analyzer...")
    print("=" * 60)
    
    # Create analyzer
    analyzer = RetrievalAnalyzer()
    
    # Create and run GUI
    gui = RetrievalAnalyzerGUI(analyzer)
    gui.run()

if __name__ == "__main__":
    main()
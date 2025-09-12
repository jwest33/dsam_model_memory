"""Background import processor for parallel memory decomposition and loading"""

import asyncio
import json
import csv
import io
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback

from agentic_memory.config import Config
from agentic_memory.extraction.llm_extractor import extract_5w1h
from agentic_memory.extraction.multi_part_extractor import extract_multi_part_5w1h, extract_batch_5w1h
from agentic_memory.types import RawEvent
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex

class ImportStatus(Enum):
    """Import job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ImportJob:
    """Import job tracking"""
    job_id: str
    status: ImportStatus
    total_records: int
    processed_records: int
    successful_imports: int
    failed_imports: int
    error_messages: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    source_type: str = "unknown"  # csv, json, text
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None

class ImportProcessor:
    """Handles background import of memories with parallel processing"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.memory_store = MemoryStore(self.config.db_path)
        self.faiss_index = FaissIndex(self.config.embed_dim, self.config.index_path)
        
        # Job tracking
        self.jobs: Dict[str, ImportJob] = {}
        self.job_lock = threading.Lock()
        
        # Worker pool for parallel processing
        self.max_workers = 4  # Configurable number of parallel workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Processing queue
        self.processing_queue = Queue()
        
    def create_import_job(self, data: Any, source_type: str = "unknown", 
                         session_id: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> str:
        """Create a new import job"""
        job_id = str(uuid.uuid4())
        
        # Determine total records based on source type
        total_records = 0
        if source_type == "json" and isinstance(data, list):
            total_records = len(data)
        elif source_type == "csv" and isinstance(data, str):
            # Count CSV rows
            reader = csv.DictReader(io.StringIO(data))
            total_records = sum(1 for _ in reader)
        elif source_type == "text":
            # Single text block
            total_records = 1
        
        job = ImportJob(
            job_id=job_id,
            status=ImportStatus.PENDING,
            total_records=total_records,
            processed_records=0,
            successful_imports=0,
            failed_imports=0,
            error_messages=[],
            start_time=datetime.now(),
            source_type=source_type,
            session_id=session_id or f"import_{job_id[:8]}",
            metadata=metadata or {}
        )
        
        with self.job_lock:
            self.jobs[job_id] = job
        
        # Start processing in background
        self.executor.submit(self._process_import, job_id, data)
        
        return job_id
    
    def _process_import(self, job_id: str, data: Any):
        """Process import job in background"""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        try:
            job.status = ImportStatus.PROCESSING
            
            if job.source_type == "json":
                self._process_json_import(job, data)
            elif job.source_type == "csv":
                self._process_csv_import(job, data)
            elif job.source_type == "text":
                self._process_text_import(job, data)
            else:
                # Try to auto-detect and process
                self._process_auto_import(job, data)
            
            job.status = ImportStatus.COMPLETED
            job.end_time = datetime.now()
            
        except Exception as e:
            job.status = ImportStatus.FAILED
            job.error_messages.append(f"Import failed: {str(e)}")
            job.end_time = datetime.now()
            print(f"Import job {job_id} failed: {e}")
            traceback.print_exc()
    
    def _process_json_import(self, job: ImportJob, data: Any):
        """Process JSON data import"""
        if isinstance(data, str):
            data = json.loads(data)
        
        if isinstance(data, dict):
            # Check if it's an export from our system
            if 'memories' in data:
                records = data['memories']
            else:
                # Single record
                records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("Invalid JSON data format")
        
        # Process records in parallel batches
        batch_size = 10
        batches = [records[i:i+batch_size] for i in range(0, len(records), batch_size)]
        
        futures = []
        for batch in batches:
            future = self.executor.submit(self._process_record_batch, job, batch)
            futures.append(future)
        
        # Wait for all batches to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                job.error_messages.append(f"Batch processing error: {str(e)}")
    
    def _process_csv_import(self, job: ImportJob, data: str):
        """Process CSV data import"""
        reader = csv.DictReader(io.StringIO(data))
        records = list(reader)
        
        # Process records in parallel batches
        batch_size = 10
        batches = [records[i:i+batch_size] for i in range(0, len(records), batch_size)]
        
        futures = []
        for batch in batches:
            future = self.executor.submit(self._process_record_batch, job, batch)
            futures.append(future)
        
        # Wait for all batches to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                job.error_messages.append(f"Batch processing error: {str(e)}")
    
    def _process_text_import(self, job: ImportJob, data: str):
        """Process raw text import"""
        # Single text block - decompose into memories
        self._process_record_batch(job, [{"raw_text": data}])
    
    def _process_auto_import(self, job: ImportJob, data: Any):
        """Auto-detect format and process"""
        if isinstance(data, str):
            # Try JSON first
            try:
                json_data = json.loads(data)
                job.source_type = "json"
                self._process_json_import(job, json_data)
                return
            except:
                pass
            
            # Try CSV
            try:
                if '\n' in data and (',' in data or '\t' in data):
                    job.source_type = "csv"
                    self._process_csv_import(job, data)
                    return
            except:
                pass
            
            # Treat as plain text
            job.source_type = "text"
            self._process_text_import(job, data)
        elif isinstance(data, (dict, list)):
            job.source_type = "json"
            self._process_json_import(job, data)
        else:
            raise ValueError(f"Cannot auto-detect format for data type: {type(data)}")
    
    def _process_record_batch(self, job: ImportJob, records: List[Dict]):
        """Process a batch of records"""
        for record in records:
            try:
                # Check if it's already a memory record from export
                if 'memory_id' in record and 'who_type' in record:
                    # Direct import of existing memory
                    self._import_existing_memory(job, record)
                else:
                    # Need to extract and decompose
                    self._extract_and_import_memory(job, record)
                
                job.processed_records += 1
                
            except Exception as e:
                job.failed_imports += 1
                job.error_messages.append(f"Record import failed: {str(e)}")
                print(f"Failed to import record: {e}")
    
    def _import_existing_memory(self, job: ImportJob, record: Dict):
        """Import an existing memory record directly"""
        try:
            from agentic_memory.types import MemoryRecord, Who, Where
            
            # Create MemoryRecord object
            memory_record = MemoryRecord(
                event_id=record.get('source_event_id', str(uuid.uuid4())),
                session_id=record.get('session_id', job.session_id),
                who=Who(
                    type=record.get('who_type', 'unknown'),
                    id=record.get('who_id', 'unknown'),
                    label=record.get('who_label')
                ),
                what=record.get('what', ''),
                when_start=record.get('when_start'),
                when_end=record.get('when_end'),
                when_timestamp=record.get('when_ts', datetime.now().isoformat()),
                where=Where(
                    type=record.get('where_type', 'digital'),
                    value=record.get('where_value', 'import')
                ),
                why=record.get('why'),
                how=record.get('how'),
                raw_text=record.get('raw_text', ''),
                importance=record.get('importance', 0.5)
            )
            
            # Generate embedding for existing memory
            if memory_record.raw_text:
                # Use the embedder to generate embedding
                from agentic_memory.embedding import get_llama_embedder
                embedder = get_llama_embedder()
                embed_text = f"WHAT: {memory_record.what}\nWHY: {memory_record.why}\nHOW: {memory_record.how}\nRAW: {memory_record.raw_text}"
                embedding = embedder.encode([embed_text], normalize_embeddings=True)[0]
                if embedding is not None:
                    # Store in database with embedding
                    self.memory_store.upsert_memory(
                        memory_record,
                        embedding=embedding.astype('float32').tobytes(),
                        dim=embedding.shape[0]
                    )
                    # Add to FAISS index
                    self.faiss_index.add(memory_record.memory_id, embedding.astype('float32'))
            else:
                # No text to generate embedding from
                job.failed_imports += 1
                raise Exception("No raw text available for embedding generation")
            
            job.successful_imports += 1
            
        except Exception as e:
            raise Exception(f"Failed to import existing memory: {str(e)}")
    
    def _extract_and_import_memory(self, job: ImportJob, record: Dict):
        """Extract 5W1H from record and import as memory"""
        try:
            # Get text content
            text = record.get('raw_text') or record.get('text') or record.get('content') or ''
            if not text:
                # Try to construct from other fields
                text = json.dumps(record)
            
            # Create RawEvent for extraction
            from agentic_memory.types import EventType
            
            # Determine event type and actor from record
            who_type = record.get('who_type', 'user')
            who_id = record.get('who_id', 'unknown')
            
            if who_type == 'llm':
                event_type = 'llm_message'
                actor = f'llm:{who_id}'
            elif who_type == 'system':
                event_type = 'system'
                actor = f'system:{who_id}'
            else:
                event_type = 'user_message'
                actor = f'user:{who_id}'
            
            raw_event = RawEvent(
                event_id=str(uuid.uuid4()),
                session_id=job.session_id,
                event_type=event_type,
                timestamp=datetime.now(),
                actor=actor,
                content=text,
                metadata=record
            )
            
            # Use multi-part extraction if enabled and text is long
            use_multi_part = getattr(self.config, 'use_multi_part_extraction', False)
            if use_multi_part and len(text) > 500:
                memory_records = extract_multi_part_5w1h(raw_event)
            else:
                # Single extraction
                memory_record = extract_5w1h(raw_event)
                memory_records = [memory_record] if memory_record else []
            
            # Store each extracted memory
            for memory_record in memory_records:
                # Get embedding from the extracted record
                if 'embed_vector_np' in memory_record.extra:
                    embedding = memory_record.extra['embed_vector_np']
                    if embedding is not None:
                        import numpy as np
                        embedding_array = np.array(embedding, dtype='float32')
                        
                        # Store in database with embedding
                        self.memory_store.upsert_memory(
                            memory_record, 
                            embedding=embedding_array.tobytes(), 
                            dim=embedding_array.shape[0]
                        )
                        
                        # Add to FAISS index
                        self.faiss_index.add(memory_record.memory_id, embedding_array)
                else:
                    # No embedding - skip this record
                    job.failed_imports += 1
                    job.error_messages.append(f"No embedding generated for record")
                    continue
                
                job.successful_imports += 1
            
            if not memory_records:
                job.failed_imports += 1
                job.error_messages.append(f"No memories extracted from record")
                
        except Exception as e:
            raise Exception(f"Failed to extract and import memory: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of an import job"""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return {
            'job_id': job.job_id,
            'status': job.status.value,
            'total_records': job.total_records,
            'processed_records': job.processed_records,
            'successful_imports': job.successful_imports,
            'failed_imports': job.failed_imports,
            'error_messages': job.error_messages[-10:],  # Last 10 errors
            'start_time': job.start_time.isoformat(),
            'end_time': job.end_time.isoformat() if job.end_time else None,
            'source_type': job.source_type,
            'session_id': job.session_id,
            'progress_percentage': (job.processed_records / job.total_records * 100) if job.total_records > 0 else 0
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel an import job"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [ImportStatus.COMPLETED, ImportStatus.FAILED]:
            return False
        
        job.status = ImportStatus.CANCELLED
        job.end_time = datetime.now()
        return True
    
    def cleanup_old_jobs(self, hours: int = 24):
        """Remove old completed jobs"""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        with self.job_lock:
            to_remove = []
            for job_id, job in self.jobs.items():
                if job.end_time and job.end_time.timestamp() < cutoff:
                    to_remove.append(job_id)
            
            for job_id in to_remove:
                del self.jobs[job_id]
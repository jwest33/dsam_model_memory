from __future__ import annotations
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import sqlite3
from contextlib import contextmanager

from .types import MemoryRecord, Who, Where, RawEvent, EventType
from .storage.sql_store import MemoryStore
from .storage.faiss_index import FaissIndex
from .config import cfg
from sentence_transformers import SentenceTransformer
import numpy as np


class MemoryExporter:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or cfg.db_path
        
    def export_memories(self, 
                       session_id: Optional[str] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Export memories to a JSON-serializable format.
        
        Args:
            session_id: Optional filter for specific session
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary containing memories and metadata
        """
        memories = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query with optional filters
            query = """
                SELECT m.*, 
                       e.vector as embedding,
                       u.accesses, u.last_access
                FROM memories m
                LEFT JOIN embeddings e ON m.memory_id = e.memory_id
                LEFT JOIN usage_stats u ON m.memory_id = u.memory_id
                WHERE 1=1
            """
            params = []
            
            if session_id:
                query += " AND m.session_id = ?"
                params.append(session_id)
            
            if start_date:
                query += " AND m.when_ts >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND m.when_ts <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY m.when_ts ASC"
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                memory_dict = {
                    "memory_id": row["memory_id"],
                    "session_id": row["session_id"],
                    "source_event_id": row["source_event_id"],
                    "who": {
                        "type": row["who_type"],
                        "id": row["who_id"],
                        "label": row["who_label"]
                    },
                    "what": row["what"],
                    "when": row["when_ts"],
                    "where": {
                        "type": row["where_type"],
                        "value": row["where_value"],
                        "lat": row["where_lat"],
                        "lon": row["where_lon"]
                    },
                    "why": row["why"],
                    "how": row["how"],
                    "raw_text": row["raw_text"],
                    "token_count": row["token_count"],
                    "embed_model": row["embed_model"],
                    "extra": json.loads(row["extra_json"]) if row["extra_json"] else {},
                    "created_at": row["created_at"],
                    "usage_stats": {
                        "accesses": row["accesses"] or 0,
                        "last_access": row["last_access"]
                    }
                }
                
                # Add embedding if present (convert from blob to list)
                if row["embedding"]:
                    embedding_array = np.frombuffer(row["embedding"], dtype=np.float32)
                    memory_dict["embedding"] = embedding_array.tolist()
                
                memories.append(memory_dict)
        
        # Get export metadata
        export_data = {
            "version": "1.0",
            "export_date": datetime.utcnow().isoformat(),
            "total_memories": len(memories),
            "filters": {
                "session_id": session_id,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            },
            "memories": memories
        }
        
        return export_data
    
    def export_to_file(self, 
                      filepath: str,
                      session_id: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> None:
        """Export memories to a JSON file."""
        export_data = self.export_memories(session_id, start_date, end_date)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)


class MemoryImporter:
    def __init__(self, 
                 sql_store: MemoryStore,
                 vector_store: FaissIndex,
                 embed_model_name: str = None):
        self.sql_store = sql_store
        self.vector_store = vector_store
        self.embed_model = SentenceTransformer(embed_model_name or cfg.embed_model_name)
        
    def validate_import_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate the import data structure.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check version
        if "version" not in data:
            errors.append("Missing 'version' field")
        elif data["version"] not in ["1.0"]:
            errors.append(f"Unsupported version: {data['version']}")
        
        # Check memories array
        if "memories" not in data:
            errors.append("Missing 'memories' field")
            return False, errors
        
        if not isinstance(data["memories"], list):
            errors.append("'memories' must be an array")
            return False, errors
        
        # Validate each memory
        required_fields = ["memory_id", "session_id", "source_event_id", 
                          "who", "what", "when", "where", "raw_text"]
        
        for i, memory in enumerate(data["memories"]):
            for field in required_fields:
                if field not in memory:
                    errors.append(f"Memory {i}: Missing required field '{field}'")
            
            # Validate nested structures
            if "who" in memory:
                if not isinstance(memory["who"], dict):
                    errors.append(f"Memory {i}: 'who' must be an object")
                elif "type" not in memory["who"] or "id" not in memory["who"]:
                    errors.append(f"Memory {i}: 'who' must have 'type' and 'id'")
            
            if "where" in memory:
                if not isinstance(memory["where"], dict):
                    errors.append(f"Memory {i}: 'where' must be an object")
                elif "type" not in memory["where"] or "value" not in memory["where"]:
                    errors.append(f"Memory {i}: 'where' must have 'type' and 'value'")
        
        return len(errors) == 0, errors
    
    def import_memories(self, 
                       data: Dict[str, Any],
                       merge_strategy: str = "skip",
                       regenerate_embeddings: bool = False) -> Dict[str, Any]:
        """
        Import memories from export data.
        
        Args:
            data: Export data dictionary
            merge_strategy: How to handle existing memories
                - "skip": Skip memories with existing IDs
                - "overwrite": Replace existing memories
                - "new_ids": Generate new IDs for all memories
            regenerate_embeddings: Whether to regenerate embeddings
            
        Returns:
            Import results summary
        """
        # Validate data
        is_valid, errors = self.validate_import_data(data)
        if not is_valid:
            return {
                "success": False,
                "errors": errors
            }
        
        imported_count = 0
        skipped_count = 0
        error_count = 0
        import_errors = []
        
        for memory_data in data["memories"]:
            try:
                # Handle merge strategy
                memory_id = memory_data["memory_id"]
                
                if merge_strategy == "new_ids":
                    # Generate new ID
                    from .types import gen_id
                    memory_id = gen_id("mem")
                elif merge_strategy == "skip":
                    # Check if memory exists
                    existing = self.sql_store.fetch_memories([memory_id])
                    if existing:
                        skipped_count += 1
                        continue
                elif merge_strategy == "overwrite":
                    # Remove from vector index if present
                    try:
                        self.vector_store.remove(memory_id)
                    except:
                        pass
                
                # Create MemoryRecord object
                who = Who(
                    type=memory_data["who"]["type"],
                    id=memory_data["who"]["id"],
                    label=memory_data["who"].get("label")
                )
                
                where = Where(
                    type=memory_data["where"]["type"],
                    value=memory_data["where"]["value"],
                    lat=memory_data["where"].get("lat"),
                    lon=memory_data["where"].get("lon")
                )
                
                # Parse datetime
                when_dt = datetime.fromisoformat(memory_data["when"])
                
                memory_record = MemoryRecord(
                    memory_id=memory_id,
                    session_id=memory_data["session_id"],
                    source_event_id=memory_data["source_event_id"],
                    who=who,
                    what=memory_data["what"],
                    when=when_dt,
                    where=where,
                    why=memory_data.get("why", ""),
                    how=memory_data.get("how", ""),
                    raw_text=memory_data["raw_text"],
                    token_count=memory_data.get("token_count", 0),
                    embed_text=memory_data.get("embed_text", memory_data["raw_text"]),
                    embed_model=memory_data.get("embed_model", cfg.embed_model_name),
                    extra=memory_data.get("extra", {})
                )
                
                # Store in SQL
                embedding = None
                if regenerate_embeddings or "embedding" not in memory_data:
                    # Generate new embedding
                    embedding = self.embed_model.encode(memory_record.embed_text)
                else:
                    # Use provided embedding
                    embedding = np.array(memory_data["embedding"], dtype=np.float32)
                
                # Convert embedding to bytes
                embedding_bytes = embedding.tobytes()
                
                # Store memory and embedding
                self.sql_store.upsert_memory(memory_record, embedding_bytes, len(embedding))
                
                # Add to vector index
                self.vector_store.add(memory_id, embedding)
                
                # Restore usage stats if present
                if "usage_stats" in memory_data:
                    stats = memory_data["usage_stats"]
                    if stats.get("accesses", 0) > 0:
                        # Record accesses to update usage stats
                        for _ in range(stats["accesses"]):
                            self.sql_store.record_access([memory_id])
                
                imported_count += 1
                
            except Exception as e:
                error_count += 1
                import_errors.append(f"Memory {memory_data.get('memory_id', '?')}: {str(e)}")
        
        # Save vector index
        self.vector_store.save()
        
        return {
            "success": True,
            "imported": imported_count,
            "skipped": skipped_count,
            "errors": error_count,
            "error_details": import_errors[:10]  # Limit error details
        }
    
    def import_from_file(self, 
                        filepath: str,
                        merge_strategy: str = "skip",
                        regenerate_embeddings: bool = False) -> Dict[str, Any]:
        """Import memories from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self.import_memories(data, merge_strategy, regenerate_embeddings)

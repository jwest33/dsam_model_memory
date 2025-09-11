#!/usr/bin/env python3
"""
Retrieval Analyzer with Comprehensive Scoring
Allows tuning of weights for semantic, lexical, recency, actor, temporal, spatial, and usage scores
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))

from agentic_memory.config import cfg
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.embedding import get_llama_embedder
from agentic_memory.retrieval import HybridRetriever
from agentic_memory.types import RetrievalQuery

@dataclass
class EnhancedScoredMemory:
    """Memory with all component scores"""
    memory_id: str
    raw_text: str
    who: str
    what: str  # Now contains entity list as JSON
    when: str
    where: str
    why: str
    how: str
    
    # Component scores (7 dimensions)
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    recency_score: float = 0.0
    actor_score: float = 0.0
    temporal_score: float = 0.0
    spatial_score: float = 0.0
    usage_score: float = 0.0
    
    # Final score
    total_score: float = 0.0
    
    # Ranking
    original_rank: int = 0
    current_rank: int = 0

class EnhancedRetrievalAnalyzer:
    """Analyzer with comprehensive scoring across 7 dimensions"""
    
    def __init__(self):
        """Initialize the retrieval system components"""
        print("Initializing enhanced retrieval analyzer...")
        
        # Initialize storage and retrieval components
        self.store = MemoryStore(cfg.db_path)
        
        # Load existing FAISS index
        import faiss
        import os
        
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
                        if memory_id:
                            results.append((memory_id, float(score)))
                return results
            
            def get_embeddings(self, memory_ids):
                """Get embeddings for given memory IDs"""
                embeddings = []
                for mid in memory_ids:
                    if mid in self.id_map:
                        idx = self.id_map.index(mid)
                        if idx >= 0 and idx < self.index.ntotal:
                            vec = self.index.reconstruct(int(idx))
                            embeddings.append(vec)
                        else:
                            embeddings.append(np.zeros(self.index.d))
                    else:
                        embeddings.append(np.zeros(self.index.d))
                return np.array(embeddings) if embeddings else None
        
        # Use the existing index
        self.index = ExistingFaissIndex(cfg.index_path)
        cfg.embed_dim = self.index.index.d
        
        self.embedder = get_llama_embedder()
        
        # Initialize retriever
        self.retriever = HybridRetriever(self.store, self.index)
        
        # Comprehensive weights (matching retrieval.py)
        self.weights = {
            'semantic': 0.45,
            'lexical': 0.25,
            'recency': 0.02,
            'actor': 0.1,
            'temporal': 0.1,
            'spatial': 0.04,
            'usage': 0.04
        }
        
        # Store current results
        self.current_query = ""
        self.scored_memories: List[EnhancedScoredMemory] = []
        
        # Count memories
        with self.store.connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
        
        print(f"Loaded {count} memories from database")
        print("Enhanced analyzer ready!")
    
    def decompose_query(self, query: str) -> Dict[str, Any]:
        """Decompose query into 5W1H components using LLM"""
        from agentic_memory.extraction.llm_extractor import extract_5w1h
        from agentic_memory.types import RawEvent
        
        raw_event = RawEvent(
            session_id='analyzer',
            event_type='user_message',
            actor='user:analyzer',
            content=query,
            metadata={}
        )
        
        try:
            extracted = extract_5w1h(raw_event)
            if hasattr(extracted, '__dict__'):
                components = {
                    'who': getattr(extracted, 'who_id', None),
                    'what': getattr(extracted, 'what', query),
                    'when': getattr(extracted, 'when_raw', None),
                    'where': getattr(extracted, 'where_value', None),
                    'why': getattr(extracted, 'why', None),
                    'how': getattr(extracted, 'how', None)
                }
            else:
                components = {
                    'who': extracted.get('who_id'),
                    'what': extracted.get('what', query),
                    'when': extracted.get('when_raw'),
                    'where': extracted.get('where_value'),
                    'why': extracted.get('why'),
                    'how': extracted.get('how')
                }
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
        
        # Also extract temporal hints
        from agentic_memory.extraction.temporal_parser import TemporalParser
        parser = TemporalParser()
        temporal_hint, cleaned_text = parser.extract_and_clean_query(query)
        if temporal_hint and not components['when']:
            components['when'] = temporal_hint
            components['what'] = cleaned_text
        
        return components
    
    def perform_retrieval(self, query: str, top_k: int = 100, components: Optional[Dict[str, Any]] = None) -> List[EnhancedScoredMemory]:
        """Perform retrieval with comprehensive scoring"""
        print(f"\nPerforming comprehensive retrieval for: '{query}'")
        
        # Decompose query if needed
        if components is None:
            components = self.decompose_query(query)
        print(f"Query components: {components}")
        
        # Create retrieval query
        rq = RetrievalQuery(
            session_id='analyzer',
            text=components.get('what', query),
            actor_hint=components.get('who'),
            temporal_hint=components.get('when')
        )
        
        # Get query embedding
        qvec = self.embedder.encode([query], normalize_embeddings=True)[0]
        
        # Use the application's retriever to get candidates
        candidates = self.retriever.search(rq, qvec, topk_sem=top_k, topk_lex=top_k)
        
        print(f"Retrieved {len(candidates)} candidates")
        
        # Fetch full memory details
        memory_ids = [c.memory_id for c in candidates[:top_k]]
        if not memory_ids:
            return []
        
        memories = self.store.fetch_memories(memory_ids)
        memory_dict = {m['memory_id']: m for m in memories}
        
        
        # Convert to EnhancedScoredMemory objects
        scored_memories = []
        for candidate in candidates[:top_k]:
            memory = memory_dict.get(candidate.memory_id)
            if not memory:
                continue
            
            # Calculate weighted score
            total_score = self.calculate_weighted_score(candidate)
            
            scored_mem = EnhancedScoredMemory(
                memory_id=candidate.memory_id,
                raw_text=memory['raw_text'] or '',
                who=memory['who_id'] or '',
                what=memory['what'] or '',
                when=memory['when_ts'] or '',
                where=memory['where_value'] or '',
                why=memory['why'] or '',
                how=memory['how'] or '',
                semantic_score=candidate.semantic_score if candidate.semantic_score is not None else 0.0,
                lexical_score=candidate.lexical_score if candidate.lexical_score is not None else 0.0,
                recency_score=candidate.recency_score if candidate.recency_score is not None else 0.0,
                actor_score=candidate.actor_score if candidate.actor_score is not None else 0.0,
                temporal_score=candidate.temporal_score if candidate.temporal_score is not None else 0.0,
                spatial_score=candidate.spatial_score if hasattr(candidate, 'spatial_score') and candidate.spatial_score is not None else 0.0,
                usage_score=candidate.usage_score if candidate.usage_score is not None else 0.0,
                total_score=total_score
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
        self.scored_memories = scored_memories
        
        return self.scored_memories
    
    def calculate_weighted_score(self, candidate) -> float:
        """Calculate score using comprehensive weights"""
        spatial_score = candidate.spatial_score if hasattr(candidate, 'spatial_score') and candidate.spatial_score is not None else 0.0
        return (
            self.weights['semantic'] * (candidate.semantic_score or 0.0) +
            self.weights['lexical'] * (candidate.lexical_score or 0.0) +
            self.weights['recency'] * (candidate.recency_score or 0.0) +
            self.weights['actor'] * (candidate.actor_score or 0.0) +
            self.weights['temporal'] * (candidate.temporal_score or 0.0) +
            self.weights['spatial'] * spatial_score +
            self.weights['usage'] * (candidate.usage_score or 0.0)
        )
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update weights and recalculate scores"""
        # Normalize weights
        total = sum(new_weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in new_weights.items()}
        else:
            self.weights = new_weights
        
        # Recalculate scores
        for mem in self.scored_memories:
            mem.total_score = (
                self.weights['semantic'] * mem.semantic_score +
                self.weights['lexical'] * mem.lexical_score +
                self.weights['recency'] * mem.recency_score +
                self.weights['actor'] * mem.actor_score +
                self.weights['temporal'] * mem.temporal_score +
                self.weights['spatial'] * mem.spatial_score +
                self.weights['usage'] * mem.usage_score
            )
        
        # Re-sort by total score
        self.scored_memories.sort(key=lambda x: x.total_score, reverse=True)
            
        # Update ranks
        for i, mem in enumerate(self.scored_memories):
            mem.current_rank = i + 1
    
    def extract_entities_from_what(self, what_field: str) -> List[str]:
        """Extract entities from the 'what' field which is now JSON"""
        if not what_field:
            return []
        
        # Try to parse as JSON array
        try:
            entities = json.loads(what_field)
            if isinstance(entities, list):
                return entities
        except:
            pass
        
        # Fallback to treating as plain text
        return [what_field]


class EnhancedAnalyzerGUI:
    """GUI with comprehensive scoring visualization"""
    
    def __init__(self, analyzer: EnhancedRetrievalAnalyzer):
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("JAM Retrieval Analyzer - Comprehensive Scoring")
        self.root.geometry("1800x1000")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables - using new default weights
        self.weight_vars = {
            'semantic': tk.DoubleVar(value=0.45),
            'lexical': tk.DoubleVar(value=0.25),
            'recency': tk.DoubleVar(value=0.02),
            'actor': tk.DoubleVar(value=0.1),
            'temporal': tk.DoubleVar(value=0.1),
            'spatial': tk.DoubleVar(value=0.04),
            'usage': tk.DoubleVar(value=0.04)
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the enhanced user interface"""
        # Main container with notebook for tabs
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top section - Query and mode selection
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        top_frame.columnconfigure(0, weight=1)
        
        # Query input
        query_frame = ttk.LabelFrame(top_frame, text="Query Input", padding="10")
        query_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        query_frame.columnconfigure(0, weight=1)
        
        input_row = ttk.Frame(query_frame)
        input_row.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.query_entry = ttk.Entry(input_row, width=60)
        self.query_entry.grid(row=0, column=0, padx=5)
        self.query_entry.insert(0, "which genes  encode Growth hormone (GH) and insulin-like growth factor 1?")
        
        self.search_button = ttk.Button(input_row, text="Search", command=self.perform_search)
        self.search_button.grid(row=0, column=1, padx=5)
        
        self.top_k_var = tk.IntVar(value=50)
        ttk.Label(input_row, text="Top K:").grid(row=0, column=2, padx=5)
        ttk.Spinbox(input_row, from_=10, to=200, textvariable=self.top_k_var, width=10).grid(row=0, column=3, padx=5)
        
        
        # Decomposed query display
        decomp_frame = ttk.Frame(query_frame)
        decomp_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(decomp_frame, text="Decomposed:", font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=0, padx=5, sticky=tk.W)
        
        self.decomp_labels = {}
        components = ['WHO', 'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW']
        for i, comp in enumerate(components):
            row = i // 3
            col = (i % 3) * 2
            ttk.Label(decomp_frame, text=f"{comp}:").grid(row=row+1, column=col, padx=5, sticky=tk.W)
            label = ttk.Label(decomp_frame, text="", foreground="blue")
            label.grid(row=row+1, column=col+1, padx=5, sticky=tk.W)
            self.decomp_labels[comp.lower()] = label
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Tab 1: Main Results
        results_tab = ttk.Frame(notebook)
        notebook.add(results_tab, text="Results & Weights")
        self.setup_results_tab(results_tab)
        
        # Tab 2: Score Analysis
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Score Analysis")
        self.setup_analysis_tab(analysis_tab)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
    
    def setup_results_tab(self, parent):
        """Setup the main results tab"""
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Left panel - Weights
        weights_text = "Component Weights (must sum to 1.0)"
        weights_frame = ttk.LabelFrame(parent, text=weights_text, padding="10")
        weights_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        row_idx = 0
        for name, var in self.weight_vars.items():
            ttk.Label(weights_frame, text=f"{name.capitalize()}:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
            
            slider = ttk.Scale(weights_frame, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL, length=200)
            slider.grid(row=row_idx, column=1, pady=2, padx=5)
            slider.bind("<ButtonRelease-1>", lambda e: self.update_all_weights())
            
            label = ttk.Label(weights_frame, text=f"{var.get():.2f}")
            label.grid(row=row_idx, column=2, pady=2)
            var.trace('w', lambda *args, l=label, v=var: l.config(text=f"{v.get():.2f}"))
            
            row_idx += 1
        
        ttk.Button(weights_frame, text="Normalize", command=self.normalize_weights).grid(row=row_idx, column=0, columnspan=3, pady=10)
        
        # Add weight sum display
        row_idx += 1
        ttk.Separator(weights_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        row_idx += 1
        self.weight_sum_label = ttk.Label(weights_frame, text="Weight Sum: 1.00", font=('TkDefaultFont', 10, 'bold'))
        self.weight_sum_label.grid(row=row_idx, column=0, columnspan=3, pady=5)
        
        # Right panel - Results
        results_frame = ttk.LabelFrame(parent, text="Retrieval Results", padding="10")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results tree
        columns = ('Rank', 'ID', 'Total', 'Sem', 'Lex', 'Rec', 'Actor', 'Temp', 'Spatial', 'Usage',
                  'Entities', 'When', 'Text')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=25)
        
        self.results_tree.heading('Rank', text='Rank')
        self.results_tree.heading('ID', text='Memory ID')
        self.results_tree.heading('Total', text='Total')
        self.results_tree.heading('Sem', text='Sem')
        self.results_tree.heading('Lex', text='Lex')
        self.results_tree.heading('Rec', text='Rec')
        self.results_tree.heading('Actor', text='Actor')
        self.results_tree.heading('Temp', text='Temp')
        self.results_tree.heading('Spatial', text='Spatial')
        self.results_tree.heading('Usage', text='Usage')
        self.results_tree.heading('Entities', text='Entities')
        self.results_tree.heading('When', text='When')
        self.results_tree.heading('Text', text='Raw Text')
        
        self.results_tree.column('Rank', width=45)
        self.results_tree.column('ID', width=100)
        self.results_tree.column('Total', width=60)
        self.results_tree.column('Sem', width=50)
        self.results_tree.column('Lex', width=50)
        self.results_tree.column('Rec', width=50)
        self.results_tree.column('Actor', width=50)
        self.results_tree.column('Temp', width=50)
        self.results_tree.column('Spatial', width=50)
        self.results_tree.column('Usage', width=50)
        self.results_tree.column('Entities', width=250)
        self.results_tree.column('When', width=100)
        self.results_tree.column('Text', width=300)
        
        # Scrollbars
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.results_tree.bind('<<TreeviewSelect>>', self.on_select)
    
    def setup_analysis_tab(self, parent):
        """Setup the score analysis tab"""
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Left: Score distribution chart
        chart_frame = ttk.LabelFrame(parent, text="Score Distribution", padding="10")
        chart_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        self.figure = plt.Figure(figsize=(6, 6), dpi=80)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right: Score statistics
        stats_frame = ttk.LabelFrame(parent, text="Score Statistics", padding="10")
        stats_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        # Score stats text
        self.score_stats_text = scrolledtext.ScrolledText(stats_frame, height=10, width=50, wrap=tk.WORD)
        self.score_stats_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Entity distribution
        ttk.Label(stats_frame, text="Common Entities:", font=('TkDefaultFont', 10, 'bold')).pack(pady=5)
        self.entity_text = scrolledtext.ScrolledText(stats_frame, height=10, width=50, wrap=tk.WORD)
        self.entity_text.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def perform_search(self):
        """Perform search with current settings"""
        query = self.query_entry.get()
        top_k = self.top_k_var.get()
        
        if not query:
            return
        
        self.status_var.set("Searching...")
        self.root.update()
        
        # Clear previous results
        for label in self.decomp_labels.values():
            label.config(text="")
        
        def search_thread():
            # Decompose query
            components = self.analyzer.decompose_query(query)
            self.root.after(0, lambda: self.display_decomposition(components))
            
            # Perform retrieval
            results = self.analyzer.perform_retrieval(query, top_k, components)
            self.root.after(0, lambda: self.display_results(results))
            self.root.after(0, self.update_visualizations)
        
        thread = threading.Thread(target=search_thread)
        thread.daemon = True
        thread.start()
    
    def display_decomposition(self, components: Dict[str, Any]):
        """Display decomposed query components"""
        for key, label in self.decomp_labels.items():
            value = components.get(key)
            if value:
                display_text = str(value)[:40]
                if len(str(value)) > 40:
                    display_text += "..."
                label.config(text=display_text)
            else:
                label.config(text="(not detected)")
    
    def display_results(self, results: List[EnhancedScoredMemory]):
        """Display search results"""
        # Clear tree
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add results
        for mem in results:
            # Extract entities from what field
            entities = self.analyzer.extract_entities_from_what(mem.what)
            entities_preview = ', '.join(entities[:5])
            if len(entities) > 5:
                entities_preview += f" (+{len(entities)-5} more)"
            
            # Truncate text fields
            text_preview = (mem.raw_text[:80].replace('\n', ' ') + "...") if len(mem.raw_text) > 80 else mem.raw_text.replace('\n', ' ')
            
            # Color based on score
            tags = []
            if mem.total_score >= 0.8:
                tags.append('high_score')
            elif mem.total_score >= 0.5:
                tags.append('med_score')
            
            self.results_tree.insert('', 'end', values=(
                mem.current_rank,
                mem.memory_id,
                f"{mem.total_score:.3f}",
                f"{mem.semantic_score:.2f}",
                f"{mem.lexical_score:.2f}",
                f"{mem.recency_score:.2f}",
                f"{mem.actor_score:.2f}",
                f"{mem.temporal_score:.2f}",
                f"{mem.spatial_score:.2f}",
                f"{mem.usage_score:.2f}",
                entities_preview,
                mem.when[:16] if mem.when else "",
                text_preview
            ), tags=tags)
        
        # Configure tags
        self.results_tree.tag_configure('high_score', background='#d4edda')
        self.results_tree.tag_configure('med_score', background='#fff3cd')
        
        self.status_var.set(f"Found {len(results)} results")
    
    def update_visualizations(self):
        """Update score visualizations"""
        if not self.analyzer.scored_memories:
            return
        
        # Update weight sum display
        self.update_weight_sum()
        
        # Update score distribution chart
        self.draw_score_distribution()
        
        # Update score stats
        self.update_score_stats()
        
        # Update entity distribution
        self.update_entity_distribution()
    
    def update_weight_sum(self):
        """Update the weight sum display"""
        total = sum(var.get() for var in self.weight_vars.values())
        self.weight_sum_label.config(text=f"Weight Sum: {total:.2f}")
        
        # Color code based on whether sum is 1.0
        if abs(total - 1.0) < 0.001:
            self.weight_sum_label.config(foreground="green")
        else:
            self.weight_sum_label.config(foreground="red")
    
    def draw_score_distribution(self):
        """Draw score distribution chart"""
        if not self.analyzer.scored_memories:
            return
        
        self.ax.clear()
        
        # Prepare data for stacked bar chart
        top_memories = self.analyzer.scored_memories[:20]
        labels = [f"#{i+1}" for i in range(len(top_memories))]
        
        semantic_scores = [m.semantic_score * self.analyzer.weights['semantic'] for m in top_memories]
        lexical_scores = [m.lexical_score * self.analyzer.weights['lexical'] for m in top_memories]
        recency_scores = [m.recency_score * self.analyzer.weights['recency'] for m in top_memories]
        actor_scores = [m.actor_score * self.analyzer.weights['actor'] for m in top_memories]
        temporal_scores = [m.temporal_score * self.analyzer.weights['temporal'] for m in top_memories]
        spatial_scores = [m.spatial_score * self.analyzer.weights['spatial'] for m in top_memories]
        usage_scores = [m.usage_score * self.analyzer.weights['usage'] for m in top_memories]
        
        # Create stacked bar chart
        x = range(len(labels))
        width = 0.8
        
        p1 = self.ax.bar(x, semantic_scores, width, label='Semantic', color='#1f77b4')
        p2 = self.ax.bar(x, lexical_scores, width, bottom=semantic_scores, label='Lexical', color='#ff7f0e')
        
        bottom2 = [s + l for s, l in zip(semantic_scores, lexical_scores)]
        p3 = self.ax.bar(x, recency_scores, width, bottom=bottom2, label='Recency', color='#2ca02c')
        
        bottom3 = [b + r for b, r in zip(bottom2, recency_scores)]
        p4 = self.ax.bar(x, actor_scores, width, bottom=bottom3, label='Actor', color='#d62728')
        
        bottom4 = [b + a for b, a in zip(bottom3, actor_scores)]
        p5 = self.ax.bar(x, temporal_scores, width, bottom=bottom4, label='Temporal', color='#9467bd')
        
        bottom5 = [b + t for b, t in zip(bottom4, temporal_scores)]
        p6 = self.ax.bar(x, spatial_scores, width, bottom=bottom5, label='Spatial', color='#8c564b')
        
        bottom6 = [b + s for b, s in zip(bottom5, spatial_scores)]
        p7 = self.ax.bar(x, usage_scores, width, bottom=bottom6, label='Usage', color='#e377c2')
        
        self.ax.set_ylabel('Weighted Score')
        self.ax.set_xlabel('Memory Rank')
        self.ax.set_title('Score Component Distribution (Top 20)')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, rotation=45)
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def update_score_stats(self):
        """Update score statistics display"""
        if not self.analyzer.scored_memories:
            return
        
        top_memories = self.analyzer.scored_memories[:20]
        
        # Calculate statistics for each component
        avg_semantic = sum(m.semantic_score for m in top_memories) / len(top_memories)
        avg_lexical = sum(m.lexical_score for m in top_memories) / len(top_memories)
        avg_recency = sum(m.recency_score for m in top_memories) / len(top_memories)
        avg_actor = sum(m.actor_score for m in top_memories) / len(top_memories)
        avg_temporal = sum(m.temporal_score for m in top_memories) / len(top_memories)
        avg_spatial = sum(m.spatial_score for m in top_memories) / len(top_memories)
        avg_usage = sum(m.usage_score for m in top_memories) / len(top_memories)
        avg_total = sum(m.total_score for m in top_memories) / len(top_memories)
        
        text = f"Score Statistics (Top 20):\n\n"
        text += f"Average Total Score: {avg_total:.3f}\n\n"
        text += f"Component Averages:\n"
        text += f"  Semantic:  {avg_semantic:.3f} (weight: {self.analyzer.weights['semantic']:.2f})\n"
        text += f"  Lexical:   {avg_lexical:.3f} (weight: {self.analyzer.weights['lexical']:.2f})\n"
        text += f"  Recency:   {avg_recency:.3f} (weight: {self.analyzer.weights['recency']:.2f})\n"
        text += f"  Actor:     {avg_actor:.3f} (weight: {self.analyzer.weights['actor']:.2f})\n"
        text += f"  Temporal:  {avg_temporal:.3f} (weight: {self.analyzer.weights['temporal']:.2f})\n"
        text += f"  Spatial:   {avg_spatial:.3f} (weight: {self.analyzer.weights['spatial']:.2f})\n"
        text += f"  Usage:     {avg_usage:.3f} (weight: {self.analyzer.weights['usage']:.2f})\n"
        
        self.score_stats_text.delete(1.0, tk.END)
        self.score_stats_text.insert(1.0, text)
    
    def update_entity_distribution(self):
        """Update entity distribution display"""
        if not self.analyzer.scored_memories:
            self.entity_text.delete(1.0, tk.END)
            self.entity_text.insert(1.0, "No entity data available")
            return
        
        # Count entity frequencies
        entity_counts = {}
        for mem in self.analyzer.scored_memories[:50]:
            entities = self.analyzer.extract_entities_from_what(mem.what)
            for entity in entities:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        # Sort by frequency
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        
        text = "Most Common Entities (Top 50 memories):\n\n"
        for entity, count in sorted_entities[:15]:
            bar = '█' * min(20, count * 2)
            text += f"{entity[:30]:30s}: {bar} ({count})\n"
        
        if len(sorted_entities) > 15:
            text += f"\n... and {len(sorted_entities) - 15} more entities"
        
        self.entity_text.delete(1.0, tk.END)
        self.entity_text.insert(1.0, text)
    
    def update_all_weights(self):
        """Update weights when sliders change"""
        new_weights = {name: var.get() for name, var in self.weight_vars.items()}
        self.analyzer.update_weights(new_weights)
        
        if self.analyzer.scored_memories:
            self.display_results(self.analyzer.scored_memories)
            self.update_visualizations()
    
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = sum(var.get() for var in self.weight_vars.values())
        if total > 0:
            for var in self.weight_vars.values():
                var.set(var.get() / total)
        self.update_all_weights()
    
    def on_select(self, event):
        """Handle selection in results tree"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = self.results_tree.item(selection[0])
        values = item['values']
        memory_id = values[2]
        
        # Find and display details
        for mem in self.analyzer.scored_memories:
            if mem.memory_id == memory_id:
                self.show_memory_details(mem)
                break
    
    def show_memory_details(self, mem: EnhancedScoredMemory):
        """Show detailed memory information in a popup"""
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Memory Details: {mem.memory_id}")
        detail_window.geometry("600x700")
        
        text = scrolledtext.ScrolledText(detail_window, wrap=tk.WORD, width=70, height=40)
        text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Extract entities
        entities = self.analyzer.extract_entities_from_what(mem.what)
        
        details = f"Memory ID: {mem.memory_id}\n"
        details += f"Current Rank: {mem.current_rank}\n\n"
        
        details += "=== 5W1H Components ===\n"
        details += f"WHO: {mem.who}\n"
        details += f"WHAT (entities): {', '.join(entities)}\n"
        details += f"WHEN: {mem.when}\n"
        details += f"WHERE: {mem.where}\n"
        details += f"WHY: {mem.why}\n"
        details += f"HOW: {mem.how}\n\n"
        
        details += "=== Component Scores ===\n"
        details += f"Semantic:  {mem.semantic_score:.4f} × {self.analyzer.weights['semantic']:.2f} = {mem.semantic_score * self.analyzer.weights['semantic']:.4f}\n"
        details += f"Lexical:   {mem.lexical_score:.4f} × {self.analyzer.weights['lexical']:.2f} = {mem.lexical_score * self.analyzer.weights['lexical']:.4f}\n"
        details += f"Recency:   {mem.recency_score:.4f} × {self.analyzer.weights['recency']:.2f} = {mem.recency_score * self.analyzer.weights['recency']:.4f}\n"
        details += f"Actor:     {mem.actor_score:.4f} × {self.analyzer.weights['actor']:.2f} = {mem.actor_score * self.analyzer.weights['actor']:.4f}\n"
        details += f"Temporal:  {mem.temporal_score:.4f} × {self.analyzer.weights['temporal']:.2f} = {mem.temporal_score * self.analyzer.weights['temporal']:.4f}\n"
        details += f"Spatial:   {mem.spatial_score:.4f} × {self.analyzer.weights['spatial']:.2f} = {mem.spatial_score * self.analyzer.weights['spatial']:.4f}\n"
        details += f"Usage:     {mem.usage_score:.4f} × {self.analyzer.weights['usage']:.2f} = {mem.usage_score * self.analyzer.weights['usage']:.4f}\n\n"
        
        details += "=== Total Score ===\n"
        details += f"Total Score: {mem.total_score:.4f}\n\n"
        
        details += "=== Raw Text ===\n"
        details += mem.raw_text
        
        text.insert(1.0, details)
        text.config(state=tk.DISABLED)
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


def main():
    """Main entry point"""
    print("Starting Enhanced JAM Retrieval Analyzer...")
    print("=" * 60)
    
    # Create analyzer
    analyzer = EnhancedRetrievalAnalyzer()
    
    # Create and run GUI
    gui = EnhancedAnalyzerGUI(analyzer)
    gui.run()


if __name__ == "__main__":
    main()

"""
Standalone experiments for testing dual-space memory system
Can run without web server - directly tests the memory components
"""

import os
import sys
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from memory.memory_store import MemoryStore
from memory.dual_space_encoder import HyperbolicOperations
from models.event import Event, FiveW1H, EventType
from agent.memory_agent import MemoryAgent

class MemoryExperiments:
    """Run experiments on the dual-space memory system"""
    
    def __init__(self):
        self.store = MemoryStore()
        self.agent = MemoryAgent()
        self.results = {}
        
    def experiment_space_separation(self):
        """Test how different types of content separate in dual spaces"""
        print("\n" + "="*60)
        print("EXPERIMENT: Space Separation")
        print("="*60)
        
        # Create concrete technical memories
        concrete_events = [
            {
                'who': 'Developer',
                'what': 'Fixed null pointer exception in UserService.java line 42',
                'when': 'yesterday',
                'where': 'backend/services',
                'why': 'prevent application crash',
                'how': 'added null check before dereferencing'
            },
            {
                'who': 'Engineer',
                'what': 'Implemented Redis cache for product listings',
                'when': 'today',
                'where': 'API layer',
                'why': 'reduce database load',
                'how': 'using Spring Cache abstraction'
            },
            {
                'who': 'Programmer',
                'what': 'Created SQL index on user_id column',
                'when': 'this morning',
                'where': 'PostgreSQL database',
                'why': 'speed up queries',
                'how': 'CREATE INDEX idx_user_id ON orders(user_id)'
            }
        ]
        
        # Create abstract conceptual memories
        abstract_events = [
            {
                'who': 'Architect',
                'what': 'Designed microservices architecture pattern',
                'when': 'last week',
                'where': 'system design',
                'why': 'improve scalability and maintainability',
                'how': 'applying domain-driven design principles'
            },
            {
                'who': 'Lead',
                'what': 'Established coding philosophy for team',
                'when': 'quarterly meeting',
                'where': 'development process',
                'why': 'ensure code quality and consistency',
                'how': 'emphasizing clean code and SOLID principles'
            },
            {
                'who': 'Consultant',
                'what': 'Introduced evolutionary architecture concepts',
                'when': 'workshop',
                'where': 'architectural decisions',
                'why': 'enable incremental change',
                'how': 'fitness functions and architectural decision records'
            }
        ]
        
        print("\nStoring concrete technical events...")
        concrete_embeddings = []
        for event_data in concrete_events:
            event = Event(
                five_w1h=FiveW1H(**event_data),
                event_type=EventType.ACTION
            )
            success, msg = self.store.store_event(event)
            print(f"  {msg}")
            
            # Get embeddings for analysis
            embeddings = self.store.encoder.encode(event_data)
            concrete_embeddings.append(embeddings)
        
        print("\nStoring abstract conceptual events...")
        abstract_embeddings = []
        for event_data in abstract_events:
            event = Event(
                five_w1h=FiveW1H(**event_data),
                event_type=EventType.OBSERVATION
            )
            success, msg = self.store.store_event(event)
            print(f"  {msg}")
            
            embeddings = self.store.encoder.encode(event_data)
            abstract_embeddings.append(embeddings)
        
        # Analyze separation in both spaces
        print("\nAnalyzing space separation...")
        
        # Euclidean space analysis
        concrete_eu_centroid = np.mean([e['euclidean_anchor'] for e in concrete_embeddings], axis=0)
        abstract_eu_centroid = np.mean([e['euclidean_anchor'] for e in abstract_embeddings], axis=0)
        eu_separation = np.linalg.norm(concrete_eu_centroid - abstract_eu_centroid)
        
        # Hyperbolic space analysis
        concrete_hy_points = [e['hyperbolic_anchor'] for e in concrete_embeddings]
        abstract_hy_points = [e['hyperbolic_anchor'] for e in abstract_embeddings]
        
        # Compute hyperbolic centroids (Fréchet mean)
        concrete_hy_centroid = np.mean(concrete_hy_points, axis=0)
        abstract_hy_centroid = np.mean(abstract_hy_points, axis=0)
        hy_separation = HyperbolicOperations.geodesic_distance(
            concrete_hy_centroid, abstract_hy_centroid, c=1.0
        )
        
        print(f"\nEuclidean separation: {eu_separation:.4f}")
        print(f"Hyperbolic separation: {hy_separation:.4f}")
        print(f"Ratio (Hyp/Euc): {hy_separation/eu_separation:.4f}")
        
        self.results['space_separation'] = {
            'euclidean_separation': eu_separation,
            'hyperbolic_separation': hy_separation,
            'ratio': hy_separation/eu_separation
        }
        
        # Test retrieval with different query types
        print("\nTesting retrieval preferences...")
        
        # Concrete query
        concrete_query = {'what': 'fix bug in code', 'how': 'debugging'}
        results = self.store.retrieve_memories(concrete_query, k=3)
        print(f"\nConcrete query results:")
        for event, score in results:
            print(f"  [{score:.3f}] {event.five_w1h.what[:50]}...")
        
        # Abstract query  
        abstract_query = {'why': 'improve architecture', 'how': 'design patterns'}
        results = self.store.retrieve_memories(abstract_query, k=3)
        print(f"\nAbstract query results:")
        for event, score in results:
            print(f"  [{score:.3f}] {event.five_w1h.what[:50]}...")
    
    def experiment_clustering_quality(self):
        """Test HDBSCAN clustering quality with different memory distributions"""
        print("\n" + "="*60)
        print("EXPERIMENT: Clustering Quality")
        print("="*60)
        
        # Create distinct clusters of memories
        clusters_data = {
            'Frontend': [
                'React component lifecycle methods',
                'CSS Grid layout techniques',
                'JavaScript async/await patterns',
                'Vue.js reactive data binding',
                'HTML5 semantic elements'
            ],
            'Backend': [
                'REST API endpoint design',
                'Database connection pooling',
                'Microservice communication patterns',
                'Message queue implementation',
                'Authentication middleware'
            ],
            'DevOps': [
                'Docker container orchestration',
                'Kubernetes deployment strategies',
                'CI/CD pipeline configuration',
                'Infrastructure as Code with Terraform',
                'Monitoring with Prometheus'
            ]
        }
        
        print("\nCreating memory clusters...")
        cluster_events = {}
        
        for cluster_name, topics in clusters_data.items():
            cluster_events[cluster_name] = []
            print(f"\n{cluster_name} cluster:")
            
            for topic in topics:
                event = Event(
                    five_w1h=FiveW1H(
                        who='Expert',
                        what=f'Explained {topic}',
                        when='recently',
                        where=f'{cluster_name.lower()} development',
                        why=f'improve {cluster_name.lower()} skills',
                        how='practical examples and best practices'
                    ),
                    event_type=EventType.OBSERVATION
                )
                success, msg = self.store.store_event(event)
                cluster_events[cluster_name].append(event)
                print(f"  + {topic}")
        
        # Test intra-cluster and inter-cluster queries
        print("\n\nTesting clustering behavior...")
        
        # Intra-cluster query (should retrieve mostly from one cluster)
        intra_query = {
            'what': 'Frontend frameworks',
            'where': 'frontend development'
        }
        
        print(f"\nIntra-cluster query: {intra_query}")
        results = self.store.retrieve_memories(intra_query, k=5, use_clustering=True)
        
        cluster_counts = {'Frontend': 0, 'Backend': 0, 'DevOps': 0}
        for event, score in results:
            for cluster_name in cluster_counts:
                if cluster_name.lower() in event.five_w1h.where:
                    cluster_counts[cluster_name] += 1
                    break
        
        print(f"Results distribution: {cluster_counts}")
        
        # Inter-cluster query (should retrieve from multiple clusters)
        inter_query = {
            'what': 'deployment and monitoring',
            'why': 'production readiness'
        }
        
        print(f"\nInter-cluster query: {inter_query}")
        results = self.store.retrieve_memories(inter_query, k=6, use_clustering=True)
        
        cluster_counts = {'Frontend': 0, 'Backend': 0, 'DevOps': 0}
        for event, score in results:
            for cluster_name in cluster_counts:
                if cluster_name.lower() in event.five_w1h.where:
                    cluster_counts[cluster_name] += 1
                    break
        
        print(f"Results distribution: {cluster_counts}")
        
        self.results['clustering_quality'] = {
            'intra_cluster': cluster_counts,
            'clusters_created': len(clusters_data)
        }
    
    def experiment_residual_adaptation(self):
        """Test residual adaptation over multiple retrievals"""
        print("\n" + "="*60)
        print("EXPERIMENT: Residual Adaptation")
        print("="*60)
        
        # Create related memories
        related_memories = [
            {
                'who': 'Teacher',
                'what': 'Explained object-oriented programming',
                'when': 'lecture 1',
                'where': 'classroom',
                'why': 'fundamental concept',
                'how': 'using class hierarchies'
            },
            {
                'who': 'Professor',
                'what': 'Demonstrated inheritance patterns',
                'when': 'lecture 2',
                'where': 'lab',
                'why': 'code reusability',
                'how': 'extending base classes'
            },
            {
                'who': 'Instructor',
                'what': 'Showed polymorphism examples',
                'when': 'lecture 3',
                'where': 'workshop',
                'why': 'flexible design',
                'how': 'interface implementation'
            }
        ]
        
        print("\nStoring related memories...")
        event_ids = []
        for mem_data in related_memories:
            event = Event(
                five_w1h=FiveW1H(**mem_data),
                event_type=EventType.OBSERVATION
            )
            success, msg = self.store.store_event(event)
            event_ids.append(event.id)
            print(f"  {msg}")
        
        # Track residual norms over iterations
        print("\nPerforming repeated queries to trigger adaptation...")
        residual_history = []
        
        queries = [
            {'what': 'object oriented concepts', 'why': 'design patterns'},
            {'what': 'inheritance and polymorphism', 'how': 'class design'},
            {'what': 'OOP principles', 'where': 'programming'}
        ]
        
        for iteration in range(5):
            print(f"\nIteration {iteration + 1}/5")
            
            for query in queries:
                results = self.store.retrieve_memories(query, k=3, update_residuals=True)
                time.sleep(0.1)
            
            # Measure current residual norms
            current_norms = []
            for event_id in event_ids:
                if event_id in self.store.residuals:
                    eu_norm = np.linalg.norm(self.store.residuals[event_id]['euclidean'])
                    hy_norm = HyperbolicOperations.geodesic_distance(
                        np.zeros_like(self.store.residuals[event_id]['hyperbolic']),
                        self.store.residuals[event_id]['hyperbolic'],
                        c=1.0
                    )
                    current_norms.append((eu_norm, hy_norm))
            
            if current_norms:
                avg_eu = np.mean([n[0] for n in current_norms])
                avg_hy = np.mean([n[1] for n in current_norms])
                residual_history.append((avg_eu, avg_hy))
                print(f"  Avg Euclidean norm: {avg_eu:.4f}")
                print(f"  Avg Hyperbolic norm: {avg_hy:.4f}")
        
        # Analyze adaptation
        if residual_history:
            initial = residual_history[0]
            final = residual_history[-1]
            
            eu_change = final[0] - initial[0]
            hy_change = final[1] - initial[1]
            
            print(f"\n📈 Adaptation Summary:")
            print(f"  Euclidean change: {eu_change:+.4f}")
            print(f"  Hyperbolic change: {hy_change:+.4f}")
            
            self.results['adaptation'] = {
                'euclidean_history': [h[0] for h in residual_history],
                'hyperbolic_history': [h[1] for h in residual_history],
                'euclidean_change': eu_change,
                'hyperbolic_change': hy_change
            }
    
    def experiment_query_weighting(self):
        """Test how query field presence affects space weighting"""
        print("\n" + "="*60)
        print("EXPERIMENT: Query-Dependent Weighting")
        print("="*60)
        
        # Create a diverse set of memories
        diverse_memories = [
            {
                'who': 'Alice',
                'what': 'implemented binary search algorithm',
                'when': 'Monday morning',
                'where': 'algorithms.py',
                'why': 'optimize search performance',
                'how': 'divide and conquer approach'
            },
            {
                'who': 'Bob',
                'what': 'discussed software quality metrics',
                'when': 'team meeting',
                'where': 'conference room',
                'why': 'establish team standards',
                'how': 'reviewing industry best practices'
            },
            {
                'who': 'Charlie',
                'what': 'refactored legacy codebase',
                'when': 'sprint 3',
                'where': 'main repository',
                'why': 'improve maintainability',
                'how': 'applying SOLID principles'
            }
        ]
        
        print("\nStoring test memories...")
        for mem_data in diverse_memories:
            event = Event(
                five_w1h=FiveW1H(**mem_data),
                event_type=EventType.ACTION
            )
            self.store.store_event(event)
        
        # Test different query patterns
        query_patterns = [
            {
                'name': 'Concrete (what/where/who)',
                'query': {'what': 'binary search', 'where': 'algorithms', 'who': 'Alice'},
                'expected_weights': 'High Euclidean'
            },
            {
                'name': 'Abstract (why/how)',
                'query': {'why': 'improve quality', 'how': 'best practices'},
                'expected_weights': 'High Hyperbolic'
            },
            {
                'name': 'Mixed',
                'query': {'what': 'refactoring', 'why': 'maintainability'},
                'expected_weights': 'Balanced'
            },
            {
                'name': 'Temporal',
                'query': {'when': 'recent', 'what': 'implementation'},
                'expected_weights': 'Slightly Euclidean'
            }
        ]
        
        print("\nTesting query weighting...")
        weighting_results = []
        
        for pattern in query_patterns:
            lambda_e, lambda_h = self.store.encoder.compute_query_weights(pattern['query'])
            
            print(f"\n{pattern['name']}:")
            print(f"  Query: {pattern['query']}")
            print(f"  λ_E (Euclidean): {lambda_e:.3f}")
            print(f"  λ_H (Hyperbolic): {lambda_h:.3f}")
            print(f"  Expected: {pattern['expected_weights']}")
            
            weighting_results.append({
                'pattern': pattern['name'],
                'lambda_e': lambda_e,
                'lambda_h': lambda_h
            })
        
        self.results['query_weighting'] = weighting_results
    
    def run_all_experiments(self):
        """Run all experiments in sequence"""
        print("\n" + "🔬"*30)
        print("DUAL-SPACE MEMORY EXPERIMENTS")
        print("🔬"*30)
        
        # Clear existing data
        print("\nClearing existing data...")
        import shutil
        if Path("state/chromadb").exists():
            shutil.rmtree("state/chromadb")
        
        # Reinitialize store
        self.store = MemoryStore()
        
        # Run experiments
        experiments = [
            self.experiment_space_separation,
            self.experiment_clustering_quality,
            self.experiment_residual_adaptation,
            self.experiment_query_weighting
        ]
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}]", end=" ")
            experiment()
            time.sleep(1)
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print experiment summary"""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        if 'space_separation' in self.results:
            res = self.results['space_separation']
            print(f"\n📊 Space Separation:")
            print(f"  Euclidean: {res['euclidean_separation']:.4f}")
            print(f"  Hyperbolic: {res['hyperbolic_separation']:.4f}")
            print(f"  Ratio: {res['ratio']:.4f}")
            verdict = "✓ Good separation" if res['ratio'] > 1.2 else "⚠ Limited separation"
            print(f"  {verdict}")
        
        if 'clustering_quality' in self.results:
            res = self.results['clustering_quality']
            print(f"\n📊 Clustering Quality:")
            print(f"  Clusters created: {res['clusters_created']}")
            print(f"  Intra-cluster precision: {res['intra_cluster']}")
        
        if 'adaptation' in self.results:
            res = self.results['adaptation']
            print(f"\n📊 Residual Adaptation:")
            print(f"  Euclidean drift: {res['euclidean_change']:+.4f}")
            print(f"  Hyperbolic drift: {res['hyperbolic_change']:+.4f}")
            verdict = "✓ Adaptation working" if abs(res['euclidean_change']) > 0.001 else "⚠ No adaptation"
            print(f"  {verdict}")
        
        if 'query_weighting' in self.results:
            print(f"\n📊 Query Weighting:")
            for result in self.results['query_weighting']:
                print(f"  {result['pattern']}: λ_E={result['lambda_e']:.2f}, λ_H={result['lambda_h']:.2f}")
        
        # Overall statistics
        stats = self.store.get_statistics()
        print(f"\n📊 Overall Statistics:")
        print(f"  Total events: {stats['total_events']}")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Events with residuals: {stats['events_with_residuals']}")
        
        if stats.get('average_residual_norm'):
            norms = stats['average_residual_norm']
            print(f"  Average residual norms:")
            print(f"    Euclidean: {norms['euclidean']:.4f}")
            print(f"    Hyperbolic: {norms['hyperbolic']:.4f}")

def main():
    """Run standalone experiments"""
    print("="*60)
    print("STANDALONE MEMORY EXPERIMENTS")
    print("="*60)
    print("\nThis will test the dual-space memory system directly")
    print("No web server required")
    
    experiments = MemoryExperiments()
    
    print("\nSelect experiment:")
    print("1. Run all experiments")
    print("2. Space separation test")
    print("3. Clustering quality test")  
    print("4. Residual adaptation test")
    print("5. Query weighting test")
    
    try:
        choice = input("\nEnter choice (1-5) [default: 1]: ").strip() or "1"
    except KeyboardInterrupt:
        print("\n\nCancelled")
        return
    
    if choice == "1":
        experiments.run_all_experiments()
    elif choice == "2":
        experiments.experiment_space_separation()
    elif choice == "3":
        experiments.experiment_clustering_quality()
    elif choice == "4":
        experiments.experiment_residual_adaptation()
    elif choice == "5":
        experiments.experiment_query_weighting()
    else:
        print(f"Invalid choice: {choice}")
        return
    
    print("\n" + "="*60)
    print("✅ Experiments completed!")
    print("="*60)

if __name__ == "__main__":
    main()

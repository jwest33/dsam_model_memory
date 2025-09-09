"""
Core benchmark testing system for JAM (Journalistic Agent Memory).
Generates varied interactions using LLM and tests memory system performance.
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import asyncio
import aiohttp
from tqdm import tqdm

from agentic_memory.router import MemoryRouter
from agentic_memory.config import Config, cfg
from agentic_memory.embedding import get_llama_embedder


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    total_interactions: int
    total_time_seconds: float
    avg_ingestion_time: float
    avg_retrieval_time: float
    memory_count: int
    index_size: int
    db_size_bytes: int
    errors: List[str]
    metadata: Dict[str, Any]


@dataclass
class InteractionScenario:
    """Represents a generated interaction scenario."""
    timestamp: datetime
    actor: str
    content: str
    topic: str
    interaction_type: str
    complexity: str  # simple, medium, complex
    metadata: Dict[str, Any]


class InteractionGenerator:
    """Generates diverse interaction scenarios using LLM."""
    
    TOPIC_CATEGORIES = [
        "technology", "business", "personal", "travel", "health",
        "education", "entertainment", "politics", "science", "sports",
        "cooking", "relationships", "finance", "philosophy", "art"
    ]
    
    INTERACTION_TYPES = [
        "conversation", "task", "question", "observation", "decision",
        "meeting", "research", "planning", "review", "reflection"
    ]
    
    ACTORS = [
        "user", "assistant", "colleague", "friend", "manager",
        "customer", "expert", "family", "mentor", "team"
    ]
    
    COMPLEXITY_PROMPTS = {
        "simple": "Generate a brief, simple interaction about {topic}. Keep it under 50 words.",
        "medium": "Generate a moderate interaction about {topic} with some detail. Around 100-150 words.",
        "complex": "Generate a detailed, complex interaction about {topic} with multiple aspects, context, and nuance. 200-300 words."
    }
    
    def __init__(self, llm_base_url: str = None, llm_model: str = None):
        """Initialize the generator with LLM configuration."""
        config = Config()
        self.llm_base_url = llm_base_url or config.get('llm_base_url', 'http://localhost:8001/v1')
        self.llm_model = llm_model or config.get('llm_model', 'local-model')
        
    async def generate_interaction(self, 
                                  topic: str = None,
                                  interaction_type: str = None,
                                  actor: str = None,
                                  complexity: str = "medium",
                                  timestamp: datetime = None) -> InteractionScenario:
        """Generate a single interaction scenario using LLM."""
        # Select random values if not provided
        topic = topic or random.choice(self.TOPIC_CATEGORIES)
        interaction_type = interaction_type or random.choice(self.INTERACTION_TYPES)
        actor = actor or random.choice(self.ACTORS)
        timestamp = timestamp or datetime.now()
        
        # Create prompt for LLM
        base_prompt = self.COMPLEXITY_PROMPTS[complexity].format(topic=topic)
        full_prompt = f"""Generate a realistic {interaction_type} interaction from the perspective of {actor}.
Topic: {topic}
Type: {interaction_type}
Actor: {actor}

{base_prompt}

The interaction should feel natural and include relevant details that would appear in a real {interaction_type}.
Include specific information like names, places, times, or other concrete details when appropriate.

Respond with ONLY the interaction content, no meta-commentary."""

        # Call LLM
        content = await self._call_llm(full_prompt)
        
        return InteractionScenario(
            timestamp=timestamp,
            actor=actor,
            content=content,
            topic=topic,
            interaction_type=interaction_type,
            complexity=complexity,
            metadata={
                "generated": True,
                "prompt_tokens": len(full_prompt.split()),
                "response_tokens": len(content.split())
            }
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """Make async call to LLM API."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.8,
                "max_tokens": 500
            }
            
            try:
                async with session.post(
                    f"{self.llm_base_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                # Fallback to template-based generation if LLM fails
                return self._fallback_generation(prompt)
    
    def _fallback_generation(self, prompt: str) -> str:
        """Generate content without LLM as fallback."""
        templates = [
            "Had an interesting discussion about {topic} today. The main points covered were analysis of current trends, evaluation of different approaches, and consideration of future implications.",
            "Working on {topic} project. Made significant progress on the implementation phase, identified some challenges that need addressing, and outlined next steps for completion.",
            "Attended meeting about {topic}. Key decisions were made regarding the strategic direction, resource allocation was discussed, and action items were assigned to team members.",
            "Researching {topic} for upcoming presentation. Found valuable insights in recent studies, compiled relevant statistics, and developed initial framework for the analysis.",
            "Reflecting on recent {topic} experiences. Learned important lessons about effective approaches, identified areas for improvement, and set goals for future development."
        ]
        
        # Extract topic from prompt if possible
        topic = "various subjects"
        if "Topic:" in prompt:
            topic = prompt.split("Topic:")[1].split("\n")[0].strip()
        
        return random.choice(templates).format(topic=topic)
    
    async def generate_batch(self, 
                           count: int,
                           time_range_days: int = 30,
                           scenario_mix: Dict[str, float] = None) -> List[InteractionScenario]:
        """Generate a batch of interaction scenarios spread over time."""
        scenario_mix = scenario_mix or {
            "simple": 0.3,
            "medium": 0.5,
            "complex": 0.2
        }
        
        scenarios = []
        base_time = datetime.now() - timedelta(days=time_range_days)
        
        # Generate interactions with varied timestamps
        for i in range(count):
            # Spread timestamps across the time range
            time_offset = timedelta(
                days=random.uniform(0, time_range_days),
                hours=random.uniform(0, 24),
                minutes=random.uniform(0, 60)
            )
            timestamp = base_time + time_offset
            
            # Select complexity based on mix
            complexity = random.choices(
                list(scenario_mix.keys()),
                weights=list(scenario_mix.values())
            )[0]
            
            scenario = await self.generate_interaction(
                complexity=complexity,
                timestamp=timestamp
            )
            scenarios.append(scenario)
        
        # Sort by timestamp
        scenarios.sort(key=lambda x: x.timestamp)
        return scenarios


class MemoryBenchmark:
    """Benchmark testing for the memory system."""
    
    def __init__(self, memory_router: Optional[MemoryRouter] = None):
        """Initialize benchmark with memory router."""
        if memory_router:
            self.router = memory_router
        else:
            # Initialize storage components
            from agentic_memory.storage.sql_store import MemoryStore
            from agentic_memory.storage.faiss_index import FaissIndex
            import os
            
            store = MemoryStore(cfg.db_path)
            # Get embedding dimension from config or default
            embed_dim = int(os.getenv('AM_EMBEDDING_DIM', '1024'))
            index = FaissIndex(dim=embed_dim, index_path=cfg.index_path)
            self.router = MemoryRouter(store, index)
        
        self.generator = InteractionGenerator()
        self.metrics = {
            "ingestion_times": [],
            "retrieval_times": [],
            "errors": []
        }
    
    async def run_benchmark(self,
                          interaction_count: int = 100,
                          time_range_days: int = 30,
                          retrieval_queries: int = 20,
                          scenario_mix: Dict[str, float] = None,
                          progress_callback = None,
                          skip_llm_extraction: bool = False) -> BenchmarkResult:
        """Run a complete benchmark test."""
        print(f"\nStarting benchmark with {interaction_count} interactions over {time_range_days} days")
        
        start_time = time.time()
        
        # Generate interactions
        print("Generating interaction scenarios...")
        scenarios = await self.generator.generate_batch(
            count=interaction_count,
            time_range_days=time_range_days,
            scenario_mix=scenario_mix
        )
        
        # Ingest memories
        print("Ingesting memories...")
        for scenario in tqdm(scenarios, desc="Ingesting"):
            await self._ingest_scenario(scenario, skip_llm_extraction=skip_llm_extraction)
            if progress_callback:
                progress_callback(scenario)
        
        # Test retrievals
        print("Testing retrievals...")
        retrieval_topics = random.sample(
            self.generator.TOPIC_CATEGORIES,
            min(retrieval_queries, len(self.generator.TOPIC_CATEGORIES))
        )
        
        for topic in tqdm(retrieval_topics, desc="Retrieving"):
            await self._test_retrieval(f"Tell me about {topic}")
        
        # Calculate metrics
        total_time = time.time() - start_time
        
        # Get storage sizes
        import os
        
        db_size = os.path.getsize(cfg.db_path) if os.path.exists(cfg.db_path) else 0
        
        # Get memory count directly from database
        with self.router.store.connect() as con:
            memory_count = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        
        # Get index count using the correct property
        index_count = len(self.router.index.memory_id_to_index) if hasattr(self.router.index, 'memory_id_to_index') else 0
        
        result = BenchmarkResult(
            total_interactions=interaction_count,
            total_time_seconds=total_time,
            avg_ingestion_time=sum(self.metrics["ingestion_times"]) / len(self.metrics["ingestion_times"]) if self.metrics["ingestion_times"] else 0,
            avg_retrieval_time=sum(self.metrics["retrieval_times"]) / len(self.metrics["retrieval_times"]) if self.metrics["retrieval_times"] else 0,
            memory_count=memory_count,
            index_size=index_count,
            db_size_bytes=db_size,
            errors=self.metrics["errors"],
            metadata={
                "time_range_days": time_range_days,
                "retrieval_queries": retrieval_queries,
                "scenario_mix": scenario_mix
            }
        )
        
        return result
    
    async def _ingest_scenario(self, scenario: InteractionScenario, skip_llm_extraction: bool = False):
        """Ingest a single scenario into memory."""
        start = time.time()
        try:
            # Convert scenario to RawEvent format
            from agentic_memory.types import RawEvent
            
            # Map interaction type to event type
            event_type_map = {
                "conversation": "user_message",
                "task": "user_message",
                "question": "user_message",
                "observation": "system",
                "decision": "system",
                "meeting": "user_message",
                "research": "system",
                "planning": "user_message",
                "review": "system",
                "reflection": "system"
            }
            
            raw_event = RawEvent(
                session_id=f"benchmark_session_{scenario.timestamp.strftime('%Y%m%d')}",
                event_type=event_type_map.get(scenario.interaction_type, "user_message"),
                actor=f"user:{scenario.actor}",
                content=scenario.content,
                timestamp=scenario.timestamp,
                metadata={
                    "topic": scenario.topic,
                    "complexity": scenario.complexity,
                    **scenario.metadata
                }
            )
            
            # Ingest into memory system with optional LLM skip
            memory_id = self.router.ingest(
                raw_event=raw_event,
                context_hint=f"{scenario.topic} {scenario.interaction_type}",
                use_multi_part=not skip_llm_extraction  # Skip multi-part extraction if flag is set
            )
            
            self.metrics["ingestion_times"].append(time.time() - start)
            
        except Exception as e:
            self.metrics["errors"].append(f"Ingestion error: {str(e)}")
    
    async def _test_retrieval(self, query: str):
        """Test memory retrieval."""
        start = time.time()
        try:
            # Use retrieve_block with context messages
            context_messages = [
                {"role": "user", "content": query}
            ]
            
            results = self.router.retrieve_block(
                session_id="benchmark_retrieval_session",
                context_messages=context_messages
            )
            
            self.metrics["retrieval_times"].append(time.time() - start)
            return results
        except Exception as e:
            self.metrics["errors"].append(f"Retrieval error: {str(e)}")
            return []
    
    def print_results(self, result: BenchmarkResult):
        """Print benchmark results in a formatted way."""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        print(f"\nPerformance Metrics:")
        print(f"  • Total interactions: {result.total_interactions}")
        print(f"  • Total time: {result.total_time_seconds:.2f} seconds")
        print(f"  • Interactions/second: {result.total_interactions/result.total_time_seconds:.2f}")
        
        print(f"\n⏱Timing:")
        print(f"  • Avg ingestion time: {result.avg_ingestion_time*1000:.2f} ms")
        print(f"  • Avg retrieval time: {result.avg_retrieval_time*1000:.2f} ms")
        
        print(f"\nStorage:")
        print(f"  • Memory count: {result.memory_count}")
        print(f"  • Index size: {result.index_size}")
        print(f"  • Database size: {result.db_size_bytes / (1024*1024):.2f} MB")
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:5]:
                print(f"  • {error}")
        else:
            print(f"\nNo errors encountered")
        
        print("\n" + "="*60)
    
    def save_results(self, result: BenchmarkResult, filename: str = None):
        """Save benchmark results to JSON file."""
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        if filename:
            if not Path(filename).is_absolute():
                filename = results_dir / filename
        else:
            filename = results_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to dict for JSON serialization
        result_dict = asdict(result)
        result_dict["timestamp"] = datetime.now().isoformat()
        
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"Results saved to {filename}")


async def main():
    """Example benchmark run."""
    benchmark = MemoryBenchmark()
    
    # Run a small benchmark
    result = await benchmark.run_benchmark(
        interaction_count=50,
        time_range_days=7,
        retrieval_queries=10,
        scenario_mix={
            "simple": 0.4,
            "medium": 0.4,
            "complex": 0.2
        }
    )
    
    benchmark.print_results(result)
    benchmark.save_results(result)


if __name__ == "__main__":
    asyncio.run(main())
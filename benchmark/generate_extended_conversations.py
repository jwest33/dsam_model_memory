"""
Enhanced benchmark dataset generator with extended conversations
Creates realistic, long-form conversations between a single user and assistant
Based on research recommendations for 15-40 exchange conversations
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os
import sys
import warnings
from pathlib import Path
import hashlib
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from agent.memory_agent import MemoryAgent
from memory.memory_store import MemoryStore
from models.event import Event, FiveW1H, EventType
from config import get_config, LLMConfig
from llm.llm_interface import LLMInterface
import subprocess
import requests
import threading


class LocationType(Enum):
    CODEBASE = "codebase"
    TERMINAL = "terminal"
    IDE = "ide"
    BROWSER = "browser"
    DOCUMENTATION = "documentation"
    GITHUB = "github"
    SLACK = "slack"
    NOTEBOOK = "jupyter_notebook"
    DATABASE = "database"
    CLOUD_CONSOLE = "cloud_console"
    API_CLIENT = "api_client"
    DEBUGGER = "debugger"


class ActivityType(Enum):
    CODING = "coding"
    DEBUGGING = "debugging"
    MEETING = "meeting"
    LEARNING = "learning"
    PLANNING = "planning"
    REVIEWING = "reviewing"
    TESTING = "testing"
    DEPLOYING = "deploying"
    DOCUMENTING = "documenting"
    RESEARCHING = "researching"


@dataclass
class ExtendedConversationScenario:
    """Scenario for extended multi-phase conversations"""
    topic: str
    category: str
    complexity: str  # simple, medium, complex
    space_preference: str  # euclidean, hyperbolic, balanced
    activities: List[ActivityType]
    keywords: List[str]
    conversation_pattern: str  # debugging, learning, project, etc.
    phases: List[str]  # Conversation phases
    min_exchanges: int
    max_exchanges: int


class ExtendedConversationGenerator:
    def __init__(self, output_dir: str = "./benchmark_datasets", use_llm: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.config = get_config()
        self.memory_agent = MemoryAgent(self.config)
        
        # Setup LLM if requested
        self.use_llm = use_llm
        if use_llm:
            self._setup_llm()
        else:
            self.llm = None
        
        # Initialize conversation patterns for extended exchanges
        self.conversation_patterns = self._initialize_conversation_patterns()
        
        # Initialize scenarios
        self.scenarios = self._initialize_scenarios()
        
        # Statistics tracking
        self.stats = {
            'total_memories': 0,
            'total_conversations': 0,
            'by_category': {},
            'by_complexity': {},
            'by_pattern': {},
            'space_distribution': {'euclidean': 0, 'hyperbolic': 0, 'balanced': 0},
            'exchange_counts': []
        }
    
    def _setup_llm(self):
        """Setup LLM for conversation generation if available"""
        try:
            print("Checking for LLM server...")
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("  Using LLM server for enhanced conversation generation")
                self.llm = LLMInterface(self.config.llm)
            else:
                print("  No LLM server found, using template-based generation")
                self.llm = None
        except:
            print("  No LLM server available, using template-based generation")
            self.llm = None
    
    def _initialize_conversation_patterns(self) -> Dict[str, Dict]:
        """Initialize patterns for extended conversations"""
        return {
            'debugging_session': {
                'description': 'Extended debugging conversation with problem solving',
                'phases': [
                    'problem_identification',
                    'initial_symptoms',
                    'environment_check',
                    'reproduction_steps',
                    'initial_hypothesis',
                    'diagnostic_exploration',
                    'log_analysis',
                    'code_inspection',
                    'narrowing_down',
                    'root_cause_hypothesis',
                    'solution_brainstorming',
                    'implementation_attempt',
                    'testing_fix',
                    'edge_case_check',
                    'verification',
                    'documentation'
                ],
                'min_exchanges': 15,
                'max_exchanges': 25
            },
            'learning_progression': {
                'description': 'Progressive learning conversation from basics to advanced',
                'phases': [
                    'topic_introduction',
                    'prerequisites_check',
                    'basic_concepts',
                    'terminology_clarification',
                    'simple_examples',
                    'hands_on_practice',
                    'common_patterns',
                    'mistake_discussion',
                    'intermediate_concepts',
                    'complex_examples',
                    'edge_cases',
                    'best_practices',
                    'real_world_scenarios',
                    'advanced_techniques',
                    'performance_considerations',
                    'integration_topics',
                    'related_technologies',
                    'resources_recommendation',
                    'summary_review',
                    'mastery_check'
                ],
                'min_exchanges': 20,
                'max_exchanges': 30
            },
            'project_development': {
                'description': 'Full project development conversation from requirements to deployment',
                'phases': [
                    'project_overview',
                    'requirements_gathering',
                    'requirements_clarification',
                    'constraints_discussion',
                    'architecture_brainstorming',
                    'technology_selection',
                    'design_patterns',
                    'database_design',
                    'api_design',
                    'implementation_planning',
                    'development_setup',
                    'core_implementation',
                    'feature_development',
                    'integration_work',
                    'error_handling',
                    'testing_strategy',
                    'unit_testing',
                    'integration_testing',
                    'performance_testing',
                    'security_review',
                    'code_review',
                    'documentation_writing',
                    'deployment_planning',
                    'ci_cd_setup',
                    'monitoring_setup',
                    'launch_preparation',
                    'post_launch_review'
                ],
                'min_exchanges': 25,
                'max_exchanges': 40
            },
            'architecture_exploration': {
                'description': 'Deep architectural discussion and design',
                'phases': [
                    'current_state_analysis',
                    'pain_points_identification',
                    'requirements_discussion',
                    'scalability_needs',
                    'performance_requirements',
                    'solution_brainstorming',
                    'pattern_evaluation',
                    'trade_off_analysis',
                    'proof_of_concept',
                    'detailed_design',
                    'component_design',
                    'data_flow_design',
                    'security_architecture',
                    'migration_strategy',
                    'risk_assessment',
                    'implementation_roadmap',
                    'team_alignment',
                    'documentation_plan'
                ],
                'min_exchanges': 18,
                'max_exchanges': 28
            },
            'performance_investigation': {
                'description': 'Performance analysis and optimization conversation',
                'phases': [
                    'performance_complaint',
                    'symptom_gathering',
                    'metric_collection',
                    'baseline_establishment',
                    'profiling_setup',
                    'hotspot_identification',
                    'bottleneck_analysis',
                    'memory_analysis',
                    'cpu_analysis',
                    'io_analysis',
                    'database_analysis',
                    'network_analysis',
                    'optimization_strategies',
                    'quick_wins',
                    'implementation_attempts',
                    'measurement_validation',
                    'further_optimization',
                    'scaling_discussion',
                    'caching_strategy',
                    'final_verification',
                    'documentation',
                    'monitoring_setup'
                ],
                'min_exchanges': 16,
                'max_exchanges': 26
            },
            'code_review_session': {
                'description': 'Detailed code review with improvements',
                'phases': [
                    'code_overview',
                    'functionality_review',
                    'correctness_check',
                    'edge_case_discussion',
                    'error_handling_review',
                    'performance_review',
                    'readability_feedback',
                    'maintainability_discussion',
                    'design_pattern_suggestions',
                    'refactoring_opportunities',
                    'testing_coverage',
                    'documentation_review',
                    'security_considerations',
                    'best_practices_alignment',
                    'improvement_implementation',
                    'final_review'
                ],
                'min_exchanges': 15,
                'max_exchanges': 22
            },
            'incident_resolution': {
                'description': 'Production incident investigation and resolution',
                'phases': [
                    'incident_report',
                    'impact_assessment',
                    'immediate_mitigation',
                    'data_gathering',
                    'timeline_reconstruction',
                    'log_investigation',
                    'metric_analysis',
                    'root_cause_investigation',
                    'hypothesis_testing',
                    'fix_development',
                    'fix_testing',
                    'rollout_planning',
                    'deployment',
                    'verification',
                    'post_mortem_discussion',
                    'prevention_measures'
                ],
                'min_exchanges': 16,
                'max_exchanges': 24
            }
        }
    
    def _initialize_scenarios(self) -> List[ExtendedConversationScenario]:
        """Initialize diverse extended conversation scenarios"""
        scenarios = []
        
        # Technical debugging scenarios (Euclidean-favoring)
        debugging_topics = [
            ("Python async/await deadlock issue", ["asyncio", "deadlock", "coroutines", "event-loop"]),
            ("Memory leak in Node.js application", ["memory", "heap", "garbage-collection", "profiling"]),
            ("React component infinite re-rendering", ["useEffect", "dependencies", "state", "lifecycle"]),
            ("Database connection pool exhaustion", ["connections", "pooling", "timeout", "transactions"]),
            ("Kubernetes pod crash loop", ["containers", "logs", "resources", "probes"]),
            ("Race condition in multi-threaded code", ["threads", "locks", "synchronization", "atomic"]),
            ("WebSocket connection dropping", ["websocket", "heartbeat", "timeout", "reconnection"]),
            ("API rate limiting issues", ["throttling", "backoff", "retry", "queuing"])
        ]
        
        for topic, keywords in debugging_topics:
            scenarios.append(ExtendedConversationScenario(
                topic=topic,
                category="debugging",
                complexity="complex",
                space_preference="euclidean",
                activities=[ActivityType.DEBUGGING, ActivityType.CODING, ActivityType.TESTING],
                keywords=keywords,
                conversation_pattern="debugging_session",
                phases=self.conversation_patterns["debugging_session"]["phases"],
                min_exchanges=self.conversation_patterns["debugging_session"]["min_exchanges"],
                max_exchanges=self.conversation_patterns["debugging_session"]["max_exchanges"]
            ))
        
        # Learning progression scenarios (Mixed space)
        learning_topics = [
            ("Understanding GraphQL from REST background", ["GraphQL", "REST", "queries", "mutations", "schema"]),
            ("Machine learning fundamentals for developers", ["ML", "training", "models", "features", "evaluation"]),
            ("Microservices architecture patterns", ["microservices", "communication", "data", "deployment"]),
            ("Functional programming concepts", ["immutability", "pure-functions", "composition", "monads"]),
            ("Container orchestration with Kubernetes", ["pods", "services", "deployments", "scaling"]),
            ("Event-driven architecture", ["events", "pub-sub", "streaming", "eventual-consistency"]),
            ("Security best practices for web apps", ["authentication", "authorization", "encryption", "vulnerabilities"]),
            ("Database normalization and design", ["normalization", "relationships", "indexes", "performance"])
        ]
        
        for topic, keywords in learning_topics:
            scenarios.append(ExtendedConversationScenario(
                topic=topic,
                category="learning",
                complexity="medium",
                space_preference="balanced",
                activities=[ActivityType.LEARNING, ActivityType.RESEARCHING, ActivityType.CODING],
                keywords=keywords,
                conversation_pattern="learning_progression",
                phases=self.conversation_patterns["learning_progression"]["phases"],
                min_exchanges=self.conversation_patterns["learning_progression"]["min_exchanges"],
                max_exchanges=self.conversation_patterns["learning_progression"]["max_exchanges"]
            ))
        
        # Project development scenarios (Balanced space)
        project_topics = [
            ("Building a real-time chat application", ["websockets", "messaging", "presence", "scaling"]),
            ("E-commerce platform with microservices", ["payments", "inventory", "orders", "authentication"]),
            ("Data pipeline for analytics", ["ETL", "streaming", "transformation", "storage"]),
            ("Mobile app with offline sync", ["sync", "conflict-resolution", "caching", "connectivity"]),
            ("CI/CD pipeline implementation", ["automation", "testing", "deployment", "monitoring"]),
            ("Search engine for documentation", ["indexing", "ranking", "relevance", "performance"]),
            ("Recommendation system", ["collaborative-filtering", "content-based", "hybrid", "evaluation"]),
            ("API gateway implementation", ["routing", "authentication", "rate-limiting", "caching"])
        ]
        
        for topic, keywords in project_topics:
            scenarios.append(ExtendedConversationScenario(
                topic=topic,
                category="project",
                complexity="complex",
                space_preference="balanced",
                activities=[ActivityType.PLANNING, ActivityType.CODING, ActivityType.REVIEWING],
                keywords=keywords,
                conversation_pattern="project_development",
                phases=self.conversation_patterns["project_development"]["phases"],
                min_exchanges=self.conversation_patterns["project_development"]["min_exchanges"],
                max_exchanges=self.conversation_patterns["project_development"]["max_exchanges"]
            ))
        
        # Architecture exploration scenarios (Hyperbolic-favoring)
        architecture_topics = [
            ("Migrating monolith to microservices", ["decomposition", "boundaries", "data", "communication"]),
            ("Designing for global scale", ["geo-distribution", "CDN", "consistency", "latency"]),
            ("Event sourcing architecture", ["events", "CQRS", "projections", "replay"]),
            ("Zero-trust security architecture", ["authentication", "authorization", "encryption", "segmentation"]),
            ("Serverless architecture patterns", ["functions", "triggers", "state", "orchestration"]),
            ("Data mesh implementation", ["domains", "products", "governance", "self-serve"]),
            ("Multi-tenant SaaS architecture", ["isolation", "customization", "scaling", "billing"]),
            ("Blockchain integration patterns", ["consensus", "smart-contracts", "integration", "security"])
        ]
        
        for topic, keywords in architecture_topics:
            scenarios.append(ExtendedConversationScenario(
                topic=topic,
                category="architecture",
                complexity="complex",
                space_preference="hyperbolic",
                activities=[ActivityType.PLANNING, ActivityType.RESEARCHING, ActivityType.DOCUMENTING],
                keywords=keywords,
                conversation_pattern="architecture_exploration",
                phases=self.conversation_patterns["architecture_exploration"]["phases"],
                min_exchanges=self.conversation_patterns["architecture_exploration"]["min_exchanges"],
                max_exchanges=self.conversation_patterns["architecture_exploration"]["max_exchanges"]
            ))
        
        # Performance investigation scenarios (Euclidean-favoring)
        performance_topics = [
            ("Database query optimization", ["queries", "indexes", "execution-plan", "statistics"]),
            ("Frontend bundle size optimization", ["bundling", "tree-shaking", "lazy-loading", "compression"]),
            ("API response time improvement", ["caching", "pagination", "queries", "serialization"]),
            ("Memory usage optimization", ["profiling", "leaks", "garbage-collection", "allocation"]),
            ("Container startup time reduction", ["image-size", "layers", "caching", "initialization"]),
            ("Network latency optimization", ["CDN", "compression", "HTTP2", "caching"])
        ]
        
        for topic, keywords in performance_topics:
            scenarios.append(ExtendedConversationScenario(
                topic=topic,
                category="performance",
                complexity="complex",
                space_preference="euclidean",
                activities=[ActivityType.DEBUGGING, ActivityType.TESTING, ActivityType.CODING],
                keywords=keywords,
                conversation_pattern="performance_investigation",
                phases=self.conversation_patterns["performance_investigation"]["phases"],
                min_exchanges=self.conversation_patterns["performance_investigation"]["min_exchanges"],
                max_exchanges=self.conversation_patterns["performance_investigation"]["max_exchanges"]
            ))
        
        return scenarios
    
    def generate_conversation_exchange(
        self,
        scenario: ExtendedConversationScenario,
        timestamp: str,
        conversation_id: str
    ) -> List[Event]:
        """Generate an extended conversation with many exchanges"""
        events = []
        
        # Randomly select number of exchanges within range
        num_exchanges = random.randint(scenario.min_exchanges, scenario.max_exchanges)
        
        # Calculate exchanges per phase
        exchanges_per_phase = max(1, num_exchanges // len(scenario.phases))
        remaining_exchanges = num_exchanges
        
        print(f"\n    Generating {num_exchanges} exchanges across {len(scenario.phases)} phases")
        
        conversation_history = []
        current_phase_idx = 0
        exchange_count = 0
        
        # Generate exchanges
        while exchange_count < num_exchanges and current_phase_idx < len(scenario.phases):
            current_phase = scenario.phases[current_phase_idx]
            
            # Determine exchanges for this phase
            if current_phase_idx == len(scenario.phases) - 1:
                # Last phase gets all remaining exchanges
                phase_exchanges = remaining_exchanges
            else:
                # Regular phase gets calculated amount
                phase_exchanges = min(exchanges_per_phase, remaining_exchanges)
            
            print(f"\n      Phase: {current_phase} ({phase_exchanges} exchanges)", end="", flush=True)
            
            # Generate exchanges for this phase
            for _ in range(phase_exchanges):
                if exchange_count >= num_exchanges:
                    break
                
                # Progress indicator
                if exchange_count % 5 == 0:
                    print(".", end="", flush=True)
                
                # Generate user query
                user_query = self._generate_user_query(
                    scenario, current_phase, conversation_history
                )
                
                # Generate assistant response
                assistant_response = self._generate_assistant_response(
                    user_query, scenario, current_phase, conversation_history
                )
                
                # Create events for both user and assistant
                # User event
                user_event = Event(
                    event_type=EventType.USER_INPUT,
                    five_w1h=FiveW1H(
                        who="User",
                        what=user_query,
                        when=timestamp,
                        where=random.choice([l.value for l in LocationType]),
                        why=f"Phase: {current_phase} in {scenario.topic}",
                        how=f"Conversation {conversation_id}, exchange {exchange_count + 1}"
                    ),
                    episode_id=conversation_id
                )
                events.append(user_event)
                
                # Assistant event
                assistant_event = Event(
                    event_type=EventType.OBSERVATION,
                    five_w1h=FiveW1H(
                        who="Assistant",
                        what=assistant_response,
                        when=timestamp,
                        where=random.choice([l.value for l in LocationType]),
                        why=f"Responding in phase: {current_phase}",
                        how=f"Conversation {conversation_id}, exchange {exchange_count + 1}"
                    ),
                    episode_id=conversation_id
                )
                events.append(assistant_event)
                
                # Update conversation history
                conversation_history.append({
                    "user": user_query,
                    "assistant": assistant_response,
                    "phase": current_phase
                })
                
                # Keep history manageable (last 10 exchanges)
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
                exchange_count += 1
                remaining_exchanges -= 1
                
                # Update statistics
                self.stats['total_memories'] += 2  # User + Assistant
            
            current_phase_idx += 1
        
        print(f"\n    Total exchanges generated: {exchange_count}")
        self.stats['exchange_counts'].append(exchange_count)
        
        return events
    
    def _generate_user_query(
        self,
        scenario: ExtendedConversationScenario,
        phase: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate contextual user query for current phase"""
        
        if self.llm and random.random() < 0.7:  # Use LLM 70% of the time if available
            return self._generate_user_query_llm(scenario, phase, conversation_history)
        
        # Phase-specific templates
        phase_templates = {
            # Debugging phases
            'problem_identification': [
                f"I'm having an issue with {random.choice(scenario.keywords)}",
                f"Something's not working with my {random.choice(scenario.keywords)} implementation",
                f"Can you help me debug this {random.choice(scenario.keywords)} problem?"
            ],
            'initial_symptoms': [
                "The error message says...",
                "When I run it, I get...",
                "The symptoms are..."
            ],
            'diagnostic_exploration': [
                f"I checked the {random.choice(scenario.keywords)} but still see the issue",
                "The logs show...",
                "When I inspect the state..."
            ],
            'solution_brainstorming': [
                "What if we try...",
                "Could the problem be...",
                "Would it help to..."
            ],
            'testing_fix': [
                "I implemented your suggestion and...",
                "After applying the fix...",
                "The test results show..."
            ],
            
            # Learning phases
            'topic_introduction': [
                f"Can you explain {random.choice(scenario.keywords)}?",
                f"I want to learn about {random.choice(scenario.keywords)}",
                f"What is {random.choice(scenario.keywords)}?"
            ],
            'basic_concepts': [
                "So the basic idea is...",
                "Let me understand, does this mean...",
                "Can you give me a simple example?"
            ],
            'hands_on_practice': [
                "Let me try implementing this...",
                "Here's my attempt...",
                "Is this the right approach?"
            ],
            'advanced_techniques': [
                "What about more complex scenarios?",
                "How do experts handle...",
                "Are there advanced patterns for..."
            ],
            
            # Project phases
            'requirements_gathering': [
                "The requirements are...",
                "We need to support...",
                "The system should..."
            ],
            'architecture_brainstorming': [
                "What architecture would work for...",
                "Should we use...",
                "How about this design..."
            ],
            'implementation_planning': [
                "Let's plan the implementation...",
                "What's the best order to...",
                "How should we structure..."
            ],
            'core_implementation': [
                "I'm implementing the core...",
                "Here's my code for...",
                "Does this look right for..."
            ]
        }
        
        # Get templates for phase or use generic
        templates = phase_templates.get(phase, [
            f"About {random.choice(scenario.keywords)} in this phase...",
            f"Continuing with {random.choice(scenario.keywords)}...",
            f"Next question about {random.choice(scenario.keywords)}..."
        ])
        
        return random.choice(templates)
    
    def _generate_user_query_llm(
        self,
        scenario: ExtendedConversationScenario,
        phase: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate user query using LLM"""
        
        # Build context
        context = ""
        if conversation_history:
            context = "Recent conversation:\n"
            for exchange in conversation_history[-3:]:
                context += f"User: {exchange['user'][:100]}...\n"
                context += f"Assistant: {exchange['assistant'][:100]}...\n"
        
        prompt = f"""You are a user in a technical conversation about {scenario.topic}.
Current phase: {phase}
{context}

Generate a natural follow-up question that fits the {phase} phase of the conversation.
Focus on {', '.join(scenario.keywords[:3])}.
Keep it concise and realistic.

Question:"""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )
            return response.strip()
        except:
            return self._generate_user_query(scenario, phase, [])  # Fallback to template
    
    def _generate_assistant_response(
        self,
        user_query: str,
        scenario: ExtendedConversationScenario,
        phase: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate assistant response for current phase"""
        
        if self.llm and random.random() < 0.7:  # Use LLM 70% of the time if available
            return self._generate_assistant_response_llm(
                user_query, scenario, phase, conversation_history
            )
        
        # Template-based responses
        return f"Here's information about {random.choice(scenario.keywords)} regarding your question in the {phase} phase: [technical details and guidance]"
    
    def _generate_assistant_response_llm(
        self,
        user_query: str,
        scenario: ExtendedConversationScenario,
        phase: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate assistant response using LLM"""
        
        prompt = f"""You are a helpful technical assistant in a conversation about {scenario.topic}.
Current phase: {phase}
User question: {user_query}

Provide a helpful, concise response that moves the conversation forward.
Focus on {', '.join(scenario.keywords[:3])}.

Response:"""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.5
            )
            return response.strip()
        except:
            return self._generate_assistant_response(user_query, scenario, phase, [])
    
    def generate_dataset(
        self,
        num_conversations: int = 50,
        dataset_name: str = None
    ) -> Dict:
        """Generate dataset with extended conversations"""
        
        if dataset_name is None:
            dataset_name = f"extended_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*70}")
        print(f"EXTENDED CONVERSATION DATASET GENERATION")
        print(f"{'='*70}")
        print(f"Dataset name: {dataset_name}")
        print(f"Target conversations: {num_conversations}")
        print(f"Expected exchanges per conversation: 15-40")
        print(f"Generation method: {'LLM-enhanced' if self.llm else 'Template-based'}")
        print(f"{'='*70}\n")
        
        all_events = []
        conversation_metadata = []
        start_time = time.time()
        
        for conv_idx in range(num_conversations):
            conv_start = time.time()
            
            # Select random scenario
            scenario = random.choice(self.scenarios)
            
            # Generate conversation ID
            conversation_id = f"conv_{dataset_name}_{conv_idx:04d}"
            
            print(f"\n[Conversation {conv_idx + 1}/{num_conversations}]")
            print(f"  Topic: {scenario.topic}")
            print(f"  Pattern: {scenario.conversation_pattern}")
            print(f"  Expected exchanges: {scenario.min_exchanges}-{scenario.max_exchanges}")
            
            # Generate timestamp
            timestamp = datetime.now().isoformat()
            
            # Generate extended conversation
            events = self.generate_conversation_exchange(
                scenario, timestamp, conversation_id
            )
            
            # Store events
            stored_count = 0
            for event in events:
                success, message = self.memory_agent.memory_store.store_event(event)
                if success:
                    stored_count += 1
                    all_events.append({
                        'event_id': event.id,
                        'event_type': event.event_type.value,
                        'five_w1h': {
                            'who': event.five_w1h.who,
                            'what': event.five_w1h.what,
                            'when': event.five_w1h.when,
                            'where': event.five_w1h.where,
                            'why': event.five_w1h.why,
                            'how': event.five_w1h.how
                        },
                        'episode_id': event.episode_id,
                        'scenario': scenario.topic,
                        'category': scenario.category,
                        'pattern': scenario.conversation_pattern
                    })
            
            print(f"  Stored {stored_count}/{len(events)} events")
            
            # Store metadata
            conversation_metadata.append({
                'conversation_id': conversation_id,
                'scenario': scenario.topic,
                'category': scenario.category,
                'pattern': scenario.conversation_pattern,
                'num_events': len(events),
                'num_exchanges': len(events) // 2,  # User + Assistant
                'timestamp': timestamp
            })
            
            # Update statistics
            self.stats['total_conversations'] += 1
            self.stats['by_category'][scenario.category] = \
                self.stats['by_category'].get(scenario.category, 0) + len(events)
            self.stats['by_pattern'][scenario.conversation_pattern] = \
                self.stats['by_pattern'].get(scenario.conversation_pattern, 0) + 1
            
            conv_time = time.time() - conv_start
            print(f"  Conversation time: {conv_time:.1f}s")
            
            # Progress update
            if (conv_idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                rate = (conv_idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (num_conversations - conv_idx - 1) / rate if rate > 0 else 0
                print(f"\n  Progress: {conv_idx + 1}/{num_conversations} conversations")
                print(f"  Total events: {len(all_events)}")
                print(f"  Avg exchanges: {np.mean(self.stats['exchange_counts']):.1f}")
                print(f"  Time remaining: {remaining/60:.1f} minutes")
        
        # Compile dataset
        dataset = {
            'metadata': {
                'name': dataset_name,
                'created_at': datetime.now().isoformat(),
                'num_conversations': num_conversations,
                'num_events': len(all_events),
                'avg_exchanges_per_conversation': np.mean(self.stats['exchange_counts']) if self.stats['exchange_counts'] else 0,
                'min_exchanges': min(self.stats['exchange_counts']) if self.stats['exchange_counts'] else 0,
                'max_exchanges': max(self.stats['exchange_counts']) if self.stats['exchange_counts'] else 0
            },
            'statistics': self.stats,
            'conversations': conversation_metadata,
            'events': all_events
        }
        
        # Save dataset
        dataset_path = self.output_dir / f"{dataset_name}.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total conversations: {num_conversations}")
        print(f"Total events: {len(all_events)}")
        print(f"Average exchanges per conversation: {np.mean(self.stats['exchange_counts']):.1f}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"\nDataset saved to: {dataset_path}")
        print(f"{'='*70}")
        
        return dataset


def main():
    """Main function for generating extended conversation datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate extended conversation benchmark dataset")
    parser.add_argument('--conversations', type=int, default=10,
                       help='Number of conversations to generate (default: 10)')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use LLM for generation if available')
    parser.add_argument('--output-dir', type=str, default='./benchmark_datasets',
                       help='Output directory for datasets')
    parser.add_argument('--name', type=str, help='Dataset name')
    
    args = parser.parse_args()
    
    print("="*70)
    print("EXTENDED CONVERSATION BENCHMARK GENERATOR")
    print("="*70)
    print(f"Generating {args.conversations} extended conversations")
    print(f"Each conversation will have 15-40 exchanges")
    print(f"Total expected events: {args.conversations * 30 * 2} (approximately)")
    print("="*70)
    
    # Create generator
    generator = ExtendedConversationGenerator(
        output_dir=args.output_dir,
        use_llm=args.use_llm
    )
    
    # Generate dataset
    dataset = generator.generate_dataset(
        num_conversations=args.conversations,
        dataset_name=args.name
    )
    
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
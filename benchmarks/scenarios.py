"""
Pre-defined benchmark scenarios for different testing needs.
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import random
import json


class BenchmarkScenarios:
    """Collection of pre-defined benchmark scenarios for testing."""
    
    @staticmethod
    def get_scenario(name: str) -> Dict[str, Any]:
        """Get a specific benchmark scenario by name."""
        scenarios = {
            "daily_productivity": BenchmarkScenarios.daily_productivity(),
            "project_development": BenchmarkScenarios.project_development(),
            "customer_support": BenchmarkScenarios.customer_support(),
            "research_notes": BenchmarkScenarios.research_notes(),
            "team_collaboration": BenchmarkScenarios.team_collaboration(),
            "personal_journal": BenchmarkScenarios.personal_journal(),
            "mixed_realistic": BenchmarkScenarios.mixed_realistic()
        }
        return scenarios.get(name, BenchmarkScenarios.mixed_realistic())
    
    @staticmethod
    def daily_productivity() -> Dict[str, Any]:
        """Scenario simulating daily productivity and task management."""
        return {
            "name": "Daily Productivity",
            "description": "Simulates a professional's daily tasks and meetings",
            "interaction_count": 150,
            "time_range_days": 7,
            "actors": ["user", "assistant", "colleague", "manager"],
            "topics": ["tasks", "meetings", "emails", "planning", "reviews"],
            "interaction_types": ["task", "meeting", "planning", "decision"],
            "scenario_mix": {
                "simple": 0.5,    # Quick tasks and updates
                "medium": 0.35,   # Regular meetings and discussions
                "complex": 0.15   # Strategic planning and reviews
            },
            "patterns": [
                {
                    "time": "morning",
                    "types": ["planning", "task"],
                    "description": "Morning planning and task setup"
                },
                {
                    "time": "midday",
                    "types": ["meeting", "conversation"],
                    "description": "Midday meetings and collaborations"
                },
                {
                    "time": "afternoon",
                    "types": ["task", "review"],
                    "description": "Afternoon execution and reviews"
                }
            ]
        }
    
    @staticmethod
    def project_development() -> Dict[str, Any]:
        """Scenario for software development project lifecycle."""
        return {
            "name": "Project Development",
            "description": "Tracks a software project from inception to deployment",
            "interaction_count": 300,
            "time_range_days": 30,
            "actors": ["developer", "pm", "designer", "tester", "client"],
            "topics": ["architecture", "coding", "testing", "deployment", "bugs"],
            "interaction_types": ["planning", "task", "review", "decision", "research"],
            "scenario_mix": {
                "simple": 0.2,    # Quick updates and checks
                "medium": 0.5,    # Development tasks and discussions
                "complex": 0.3    # Architecture decisions and problem solving
            },
            "phases": [
                {
                    "name": "Planning",
                    "duration_percent": 0.2,
                    "focus": ["requirements", "architecture", "design"]
                },
                {
                    "name": "Development",
                    "duration_percent": 0.5,
                    "focus": ["coding", "testing", "integration"]
                },
                {
                    "name": "Deployment",
                    "duration_percent": 0.3,
                    "focus": ["deployment", "monitoring", "optimization"]
                }
            ]
        }
    
    @staticmethod
    def customer_support() -> Dict[str, Any]:
        """Scenario for customer support interactions."""
        return {
            "name": "Customer Support",
            "description": "Customer service interactions and issue resolution",
            "interaction_count": 200,
            "time_range_days": 14,
            "actors": ["support_agent", "customer", "manager", "technical_team"],
            "topics": ["issues", "questions", "feedback", "escalations", "solutions"],
            "interaction_types": ["question", "conversation", "task", "decision"],
            "scenario_mix": {
                "simple": 0.6,    # Quick questions and answers
                "medium": 0.3,    # Standard issue resolution
                "complex": 0.1    # Complex escalations
            },
            "issue_types": [
                "technical_problem",
                "billing_question",
                "feature_request",
                "complaint",
                "general_inquiry"
            ]
        }
    
    @staticmethod
    def research_notes() -> Dict[str, Any]:
        """Scenario for academic or professional research."""
        return {
            "name": "Research Notes",
            "description": "Academic or professional research documentation",
            "interaction_count": 100,
            "time_range_days": 60,
            "actors": ["researcher", "colleague", "advisor", "reviewer"],
            "topics": ["literature", "experiments", "analysis", "findings", "methodology"],
            "interaction_types": ["research", "observation", "reflection", "review"],
            "scenario_mix": {
                "simple": 0.15,   # Quick observations
                "medium": 0.45,   # Regular research notes
                "complex": 0.4    # Detailed analysis and findings
            },
            "research_phases": [
                "literature_review",
                "hypothesis_formation",
                "data_collection",
                "analysis",
                "conclusion"
            ]
        }
    
    @staticmethod
    def team_collaboration() -> Dict[str, Any]:
        """Scenario for team collaboration and communication."""
        return {
            "name": "Team Collaboration",
            "description": "Team interactions across various projects",
            "interaction_count": 250,
            "time_range_days": 21,
            "actors": ["team_lead", "developer", "designer", "analyst", "stakeholder"],
            "topics": ["projects", "deadlines", "resources", "blockers", "achievements"],
            "interaction_types": ["meeting", "conversation", "decision", "planning"],
            "scenario_mix": {
                "simple": 0.4,    # Quick updates and check-ins
                "medium": 0.4,    # Team discussions
                "complex": 0.2    # Strategic planning
            },
            "collaboration_types": [
                "standup",
                "planning_session",
                "retrospective",
                "brainstorming",
                "status_update"
            ]
        }
    
    @staticmethod
    def personal_journal() -> Dict[str, Any]:
        """Scenario for personal journaling and reflection."""
        return {
            "name": "Personal Journal",
            "description": "Personal thoughts, reflections, and daily events",
            "interaction_count": 90,
            "time_range_days": 90,
            "actors": ["user", "self"],
            "topics": ["personal", "goals", "reflections", "events", "thoughts"],
            "interaction_types": ["reflection", "observation", "planning"],
            "scenario_mix": {
                "simple": 0.3,    # Quick daily notes
                "medium": 0.5,    # Regular journal entries
                "complex": 0.2    # Deep reflections
            },
            "entry_types": [
                "daily_summary",
                "goal_tracking",
                "gratitude",
                "reflection",
                "planning"
            ]
        }
    
    @staticmethod
    def mixed_realistic() -> Dict[str, Any]:
        """Realistic mixed scenario covering various aspects of life."""
        return {
            "name": "Mixed Realistic",
            "description": "Realistic mix of professional and personal interactions",
            "interaction_count": 500,
            "time_range_days": 45,
            "actors": [
                "user", "assistant", "colleague", "friend", "family",
                "manager", "customer", "expert", "mentor"
            ],
            "topics": [
                "work", "personal", "technology", "health", "finance",
                "education", "travel", "relationships", "hobbies", "news"
            ],
            "interaction_types": [
                "conversation", "task", "question", "observation",
                "decision", "meeting", "research", "planning", "reflection"
            ],
            "scenario_mix": {
                "simple": 0.35,
                "medium": 0.45,
                "complex": 0.20
            },
            "distribution": {
                "work_related": 0.4,
                "personal": 0.3,
                "learning": 0.15,
                "social": 0.15
            }
        }


class ScenarioGenerator:
    """Generate specific interactions based on scenario templates."""
    
    @staticmethod
    def generate_from_scenario(scenario: Dict[str, Any], count: int = None) -> List[Dict[str, Any]]:
        """Generate interactions based on a scenario template."""
        count = count or scenario.get("interaction_count", 100)
        interactions = []
        
        base_time = datetime.now() - timedelta(days=scenario.get("time_range_days", 30))
        
        for i in range(count):
            # Select random elements from scenario
            actor = random.choice(scenario.get("actors", ["user"]))
            topic = random.choice(scenario.get("topics", ["general"]))
            interaction_type = random.choice(scenario.get("interaction_types", ["conversation"]))
            
            # Determine complexity
            mix = scenario.get("scenario_mix", {"medium": 1.0})
            complexity = random.choices(
                list(mix.keys()),
                weights=list(mix.values())
            )[0]
            
            # Generate timestamp
            time_offset = timedelta(
                days=random.uniform(0, scenario.get("time_range_days", 30)),
                hours=random.uniform(0, 24),
                minutes=random.uniform(0, 60)
            )
            timestamp = base_time + time_offset
            
            # Create interaction
            interaction = {
                "timestamp": timestamp.isoformat(),
                "actor": actor,
                "topic": topic,
                "type": interaction_type,
                "complexity": complexity,
                "scenario": scenario.get("name", "custom"),
                "metadata": {
                    "generated_from_scenario": True,
                    "scenario_name": scenario.get("name", "custom")
                }
            }
            
            # Add scenario-specific metadata
            if "phases" in scenario:
                # Determine which phase this interaction belongs to
                phase_index = int(i / count * len(scenario["phases"]))
                phase = scenario["phases"][min(phase_index, len(scenario["phases"]) - 1)]
                interaction["metadata"]["phase"] = phase.get("name", "unknown")
                interaction["focus"] = random.choice(phase.get("focus", [topic]))
            
            interactions.append(interaction)
        
        return sorted(interactions, key=lambda x: x["timestamp"])


def list_scenarios():
    """List all available benchmark scenarios."""
    scenarios = [
        "daily_productivity",
        "project_development",
        "customer_support",
        "research_notes",
        "team_collaboration",
        "personal_journal",
        "mixed_realistic"
    ]
    
    print("\nAvailable Benchmark Scenarios")
    print("="*50)
    
    for name in scenarios:
        scenario = BenchmarkScenarios.get_scenario(name)
        print(f"\n {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   • Interactions: {scenario.get('interaction_count', 'N/A')}")
        print(f"   • Time range: {scenario.get('time_range_days', 'N/A')} days")
        print(f"   • Actors: {len(scenario.get('actors', []))}")
        print(f"   • Topics: {len(scenario.get('topics', []))}")


if __name__ == "__main__":
    # List available scenarios
    list_scenarios()
    
    # Example: Generate interactions from a scenario
    print("\n" + "="*50)
    print("Example: Generating 10 interactions from 'daily_productivity'")
    print("="*50)
    
    scenario = BenchmarkScenarios.daily_productivity()
    interactions = ScenarioGenerator.generate_from_scenario(scenario, count=10)
    
    for interaction in interactions[:3]:
        print(f"\n[{interaction['timestamp']}]")
        print(f"  Actor: {interaction['actor']}")
        print(f"  Type: {interaction['type']}")
        print(f"  Topic: {interaction['topic']}")
        print(f"  Complexity: {interaction['complexity']}")
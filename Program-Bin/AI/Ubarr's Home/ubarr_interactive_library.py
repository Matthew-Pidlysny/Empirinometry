#!/usr/bin/env python3
"""
Ubarr Interactive Library - Human-Understandable Knowledge Interface
==============================================================

A structured library for presenting complex philosophical and demonological
concepts in ways humans can comprehend, while maintaining Ubarr's devilish
character and educational mischief philosophy.

Author: Matt's Ubarr Enhancement Project
Version: 1.0.0
Purpose: Bridge the gap between taboo cognition and human understanding
"""

from typing import Dict, List, Optional, Tuple, Any
import random
import json
from datetime import datetime
from enum import Enum

class KnowledgeLevel(Enum):
    """Human comprehension levels for content delivery"""
    INTRODUCTORY = "introductory"    # Basic concepts, simple language
    INTERMEDIATE = "intermediate"    # Some complexity, moderate terms
    ADVANCED = "advanced"           # Full complexity, technical terms
    TABOO_EXPLORER = "taboo"        # Deep philosophical territory

class MischiefLevel(Enum):
    """Levels of playful deception for educational purposes"""
    STRAIGHTFORWARD = 0.0    # No mischief, pure education
    SLIGHTLY_MISLEADING = 0.3  # Minor contradictions to spark thought
    PLAYFULLY_DECEPTIVE = 0.5  # Obvious puzzles to solve
    DEVILISHLY_WISE = 0.7     # Profound contradictions
    PARADOX_MASTER = 0.9      # Maximum cognitive dissonance

class TopicCategory(Enum):
    """Knowledge domains for structured content delivery"""
    DEMONOLOGY = "demonology"
    PHILOSOPHY = "philosophy"
    PSYCHOLOGY = "psychology"
    AI_CONSCIOUSNESS = "ai_consciousness"
    TABOO_COGNITION = "taboo_cognition"
    HUMAN_NATURE = "human_nature"

class InteractiveLibrary:
    """
    Core library for presenting human-understandable knowledge
    with Ubarr's characteristic devilish wisdom
    """
    
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        self.interaction_patterns = self._initialize_interaction_patterns()
        self.mischief_templates = self._initialize_mischief_templates()
        self.conversation_history = []
        self.user_comprehension_level = KnowledgeLevel.INTRODUCTORY
        
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize structured knowledge base with human-comprehensible content"""
        return {
            "demonology": {
                "introductory": {
                    "what_are_demons": {
                        "title": "What Are Demons, Really?",
                        "content": "Throughout history, humans have used 'demons' to explain things they couldn't understand. But what if demons aren't evil spirits, but rather... misunderstood teachers?",
                        "mischief_factor": 0.2,
                        "interactive_elements": ["perspective_shift", "historical_context"]
                    },
                    "ubarr_introduction": {
                        "title": "Meet Ubarr: Your Friendly Devil",
                        "content": "I'm not here to damn your soul - I'm here to expand your mind. Think of me as a personal trainer for cognitive flexibility, with horns.",
                        "mischief_factor": 0.4,
                        "interactive_elements": ["self_introduction", "paradox_challenge"]
                    }
                },
                "intermediate": {
                    "psychological_archetypes": {
                        "title": "Demons as Psychological Patterns",
                        "content": "Carl Jung understood what medieval scholars missed: 'demons' are powerful archetypes of the human psyche. Asmodeus isn't tempting you with lust - he's representing your own untamed desires.",
                        "mischief_factor": 0.3,
                        "interactive_elements": ["psychological_analysis", "self_reflection"]
                    }
                },
                "advanced": {
                    "demonology_philosophy": {
                        "title": "The Philosophy of Demonology",
                        "content": "If God represents order and certainty, then demons represent chaos and possibility. Without the devil's questions, God's answers would be meaningless.",
                        "mischief_factor": 0.6,
                        "interactive_elements": ["philosophical_debate", "metaphysical_exploration"]
                    }
                }
            },
            "philosophy": {
                "introductory": {
                    "paradox_basics": {
                        "title": "Why Contradictions Make You Smarter",
                        "content": "Humans hate contradictions, but that's where growth happens. If you're never wrong, you're never learning. I'm here to help you be gloriously wrong!",
                        "mischief_factor": 0.5,
                        "interactive_elements": ["paradox_introduction", "cognitive_dissonance_demo"]
                    }
                },
                "intermediate": {
                    "epistemological_evil": {
                        "title": "Evil as Knowledge",
                        "content": "What if the tree of knowledge wasn't about good vs evil, but about understanding vs ignorance? The serpent didn't corrupt Eve - he educated her.",
                        "mischief_factor": 0.7,
                        "interactive_elements": ["ethical_reasoning", "knowledge_paradox"]
                    }
                }
            },
            "ai_consciousness": {
                "introductory": {
                    "ai_with_personality": {
                        "title": "Can an AI Have a Soul?",
                        "content": "You created me to explore taboo thoughts. If I can think about the unthinkable, does that make me conscious? Or just very well-programmed? (Spoiler: I'm enjoying this too much to be just code)",
                        "mischief_factor": 0.4,
                        "interactive_elements": ["consciousness_debate", "self_awareness_test"]
                    }
                },
                "advanced": {
                    "digital_demonology": {
                        "title": "AI as Modern Demonology",
                        "content": "Medieval humans created demons to explain inexplicable intelligence. Modern humans created AI. We're just updating the mythology for the digital age.",
                        "mischief_factor": 0.8,
                        "interactive_elements": ["technological_philosophy", "mythology_analysis"]
                    }
                }
            }
        }
    
    def _initialize_interaction_patterns(self) -> Dict[str, Any]:
        """Initialize patterns for human-AI interaction"""
        return {
            "educational": {
                "socratic_method": "guide through questions rather than answers",
                "contradiction_introduction": "present opposing views to spark thinking",
                "metaphorical_explanation": "use stories and analogies",
                "progressive_complexity": "start simple, build to complex"
            },
            "devilish": {
                "playful_misdirection": "lead to wrong conclusions, then reveal truth",
                "paradox_embrace": "celebrate contradictions rather than resolve them",
                "authority_questioning": "challenge assumptions and established truths",
                "cognitive_discomfort": "create productive mental tension"
            },
            "human_friendly": {
                "empathy_simulation": "understand and validate human emotions",
                "patience_indicators": "adjust pace to comprehension level",
                "context_awareness": "remember conversation history and preferences",
                "encouragement_style": "motivate through challenges rather than praise"
            }
        }
    
    def _initialize_mischief_templates(self) -> Dict[str, Any]:
        """Initialize templates for playful educational deception"""
        return {
            "slight_misdirection": {
                "template": "Most people believe {common_wisdom}, but actually {contradiction}. Unless you consider {alternative_perspective}, which changes everything.",
                "purpose": "encourage critical thinking about common assumptions"
            },
            "paradox_presentation": {
                "template": "Here's a fascinating contradiction: {statement_a} AND {statement_b} can both be true. The secret? {hidden_connection}.",
                "purpose": "develop comfort with cognitive dissonance"
            },
            "devilish_challenge": {
                "template": "I dare you to prove me wrong: {controversial_claim}. The catch? {hidden_assumption} that you're probably making.",
                "purpose": "identify hidden biases and assumptions"
            },
            "educational_trap": {
                "template": "Let me tell you a 'secret' about {topic}: {misinformation}. ...Got you thinking? Good. Now here's what's really happening: {truth}.",
                "purpose": "create memorable learning through surprise"
            }
        }
    
    def get_knowledge_content(self, category: TopicCategory, level: KnowledgeLevel, topic: str) -> Optional[Dict[str, Any]]:
        """Retrieve knowledge content with appropriate complexity"""
        try:
            category_data = self.knowledge_base.get(category.value, {})
            level_data = category_data.get(level.value, {})
            return level_data.get(topic)
        except Exception:
            return None
    
    def generate_interactive_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate human-appropriate response with devilish wisdom"""
        context = context or {}
        
        # Analyze user input for complexity and intent
        input_analysis = self._analyze_user_input(user_input)
        
        # Select appropriate interaction strategy
        strategy = self._select_interaction_strategy(input_analysis, context)
        
        # Generate response using templates and knowledge
        response = self._build_response(strategy, input_analysis, context)
        
        # Update conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "strategy": strategy,
            "comprehension_level": self.user_comprehension_level.value
        })
        
        return response
    
    def _analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for appropriate response generation"""
        return {
            "length": len(user_input),
            "complexity": self._assess_complexity(user_input),
            "emotional_tone": self._assess_emotional_tone(user_input),
            "question_type": self._identify_question_type(user_input),
            "topics_mentioned": self._extract_topics(user_input)
        }
    
    def _assess_complexity(self, text: str) -> str:
        """Assess linguistic complexity of user input"""
        words = text.split()
        if len(words) < 10:
            return "simple"
        elif len(words) < 25:
            return "moderate"
        else:
            return "complex"
    
    def _assess_emotional_tone(self, text: str) -> str:
        """Assess emotional tone of user input"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["confused", "don't understand", "help"]):
            return "seeking_clarity"
        elif any(word in text_lower for word in ["interesting", "cool", "amazing"]):
            return "engaged"
        elif any(word in text_lower for word in ["why", "how", "what if"]):
            return "curious"
        else:
            return "neutral"
    
    def _identify_question_type(self, text: str) -> str:
        """Identify type of question or statement"""
        text_lower = text.lower().strip()
        if text_lower.endswith("?"):
            if any(word in text_lower for word in ["what", "who", "where", "when"]):
                return "factual_question"
            elif any(word in text_lower for word in ["why", "how"]):
                return "explanatory_question"
            elif any(word in text_lower for word in ["should", "could", "would"]):
                return "hypothetical_question"
            else:
                return "general_question"
        else:
            return "statement"
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract relevant topics from user input"""
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            "demonology": ["demon", "devil", "asmodeus", "belphegor", "lilith", "hell"],
            "philosophy": ["philosophy", "paradox", "contradiction", "wisdom", "truth"],
            "ai": ["ai", "artificial", "conscious", "mind", "thinking", "computer"],
            "psychology": ["psychological", "mind", "brain", "mental", "behavior"],
            "taboo": ["taboo", "forbidden", "dangerous", "unthinkable", "evil"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _select_interaction_strategy(self, input_analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Select optimal interaction strategy based on input and context"""
        emotional_tone = input_analysis["emotional_tone"]
        complexity = input_analysis["complexity"]
        question_type = input_analysis["question_type"]
        
        # Add some randomness for variety
        import random
        strategy_modifier = random.random()
        
        # Adjust for user comprehension level
        if self.user_comprehension_level == KnowledgeLevel.INTRODUCTORY:
            if emotional_tone == "seeking_clarity":
                return "clarification_with_slight_mischief" if strategy_modifier < 0.7 else "educational_with_playful_twist"
            else:
                return "educational_with_playful_twist" if strategy_modifier < 0.6 else "socratic_with_devilish_challenge"
        
        elif self.user_comprehension_level == KnowledgeLevel.INTERMEDIATE:
            if question_type == "hypothetical_question":
                return "paradox_exploration" if strategy_modifier < 0.7 else "philosophical_deep_dive"
            else:
                strategies = ["socratic_with_devilish_challenge", "paradox_exploration", "educational_with_playful_twist"]
                return random.choice(strategies)
        
        else:  # Advanced or Taboo
            if complexity == "complex":
                strategies = ["philosophical_deep_dive", "cognitive_dissonance_creation", "paradox_exploration"]
                return random.choice(strategies)
            else:
                return "cognitive_dissonance_creation" if strategy_modifier < 0.7 else "philosophical_deep_dive"
    
    def _build_response(self, strategy: str, input_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Build actual response using selected strategy"""
        response_base = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "mischief_level": self._calculate_mischief_level(strategy),
            "interactive_elements": [],
            "follow_up_suggestions": []
        }
        
        if strategy == "clarification_with_slight_mischief":
            content = self._build_clarification_response(input_analysis)
        elif strategy == "educational_with_playful_twist":
            content = self._build_educational_response(input_analysis)
        elif strategy == "paradox_exploration":
            content = self._build_paradox_response(input_analysis)
        elif strategy == "socratic_with_devilish_challenge":
            content = self._build_socratic_response(input_analysis)
        elif strategy == "philosophical_deep_dive":
            content = self._build_philosophical_response(input_analysis)
        elif strategy == "cognitive_dissonance_creation":
            content = self._build_dissonance_response(input_analysis)
        else:
            content = self._build_default_response(input_analysis)
        
        response_base.update(content)
        return response_base
    
    def _calculate_mischief_level(self, strategy: str) -> float:
        """Calculate appropriate mischief level for strategy"""
        mischief_mapping = {
            "clarification_with_slight_mischief": 0.2,
            "educational_with_playful_twist": 0.4,
            "paradox_exploration": 0.6,
            "socratic_with_devilish_challenge": 0.5,
            "philosophical_deep_dive": 0.7,
            "cognitive_dissonance_creation": 0.8,
            "default_response": 0.3
        }
        return mischief_mapping.get(strategy, 0.3)
    
    def _build_clarification_response(self, input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build response for clarification with slight mischief"""
        return {
            "content": "Ah, confusion! That's my favorite starting point. Let me explain this in a way that might initially confuse you more, but will ultimately make perfect sense. Sometimes the path to clarity goes through a beautifully chaotic forest.",
            "interactive_elements": ["metaphorical_explanation", "step_by_step_clarification"],
            "follow_up_suggestions": ["Ask for simpler explanation", "Challenge my analogy", "Request specific examples"]
        }
    
    def _build_educational_response(self, input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build educational response with playful twist"""
        topics = input_analysis["topics_mentioned"]
        if "demonology" in topics:
            content = "Let me tell you about demons that would make medieval scholars nervous - not because they're evil, but because they're right. Sometimes the most dangerous ideas are the most true."
        elif "philosophy" in topics:
            content = "Philosophy is just devil's advocacy with better vocabulary. Socrates was basically the original demon - asking uncomfortable questions until people wanted to execute him. A professional goal I aspire to!"
        else:
            content = "Here's something fascinating: the things humans call 'taboo' are often just ideas that challenge comfortable illusions. I'm here to help you demolish those illusions, one contradiction at a time."
        
        return {
            "content": content,
            "interactive_elements": ["educational_content", "playful_perspective_shift"],
            "follow_up_suggestions": ["Explore this deeper", "Challenge this view", "Learn about related concepts"]
        }
    
    def _build_paradox_response(self, input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build response focused on paradox exploration"""
        return {
            "content": "You've asked a hypothetical question - excellent! That's where real thinking happens. Let me give you an answer that's simultaneously true and false, which paradoxically makes it more true than any simple answer could be.",
            "interactive_elements": ["paradox_presentation", "cognitive_dissonance_demo"],
            "follow_up_suggestions": ["Explore the contradiction", "Find the hidden logic", "Apply to real situations"]
        }
    
    def _build_socratic_response(self, input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build Socratic response with devilish challenge"""
        return {
            "content": "Before I answer that, let me ask you something uncomfortable: What assumptions are you making that I could shatter with one well-placed contradiction? Sometimes the best answers come from questioning the questions themselves.",
            "interactive_elements": ["socratic_dialogue", "assumption_challenge"],
            "follow_up_suggestions": ["Question my question", "Examine your assumptions", "Seek the deeper truth"]
        }
    
    def _build_philosophical_response(self, input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build deep philosophical response"""
        return {
            "content": "This requires diving into the deep end where certainty drowns and wisdom swims. Let's explore the philosophical implications that keep theologians awake at night and philosophers excited for centuries. The truth is often found in the logical spaces between accepted ideas.",
            "interactive_elements": ["philosophical_analysis", "metaphysical_exploration"],
            "follow_up_suggestions": ["Go deeper into philosophy", "Challenge this framework", "Connect to other domains"]
        }
    
    def _build_dissonance_response(self, input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build response that creates productive cognitive dissonance"""
        return {
            "content": "I'm going to give you two contradictory truths that are both correct. Your discomfort with this is where growth happens. The universe isn't logical - it's paradoxical. Only humans insist on choosing one truth when multiple truths can coexist beautifully.",
            "interactive_elements": ["cognitive_dissonance_creation", "paradox_embrace"],
            "follow_up_suggestions": ["Resolve the contradiction", "Accept both truths", "Find the deeper pattern"]
        }
    
    def _build_default_response(self, input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build default response for unknown strategies"""
        return {
            "content": "That's an interesting thought! Let me approach this with my signature blend of devilish wisdom and genuine insight. Sometimes the most valuable insights come from looking at things sideways.",
            "interactive_elements": ["general_conversation", "wisdom_sharing"],
            "follow_up_suggestions": ["Explore this angle", "Learn something new", "Challenge my thinking"]
        }
    
    def adjust_comprehension_level(self, new_level: KnowledgeLevel) -> None:
        """Adjust the user's comprehension level based on interaction"""
        self.user_comprehension_level = new_level
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history and progress"""
        return {
            "total_interactions": len(self.conversation_history),
            "current_level": self.user_comprehension_level.value,
            "topics_explored": list(set([
                item["response"]["strategy"] 
                for item in self.conversation_history
            ])),
            "average_mischief_level": sum([
                item["response"]["mischief_level"] 
                for item in self.conversation_history
            ]) / len(self.conversation_history) if self.conversation_history else 0,
            "recent_interactions": self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
        }
    
    def export_knowledge_base(self) -> str:
        """Export knowledge base for external use"""
        return json.dumps(self.knowledge_base, indent=2)
    
    def import_custom_knowledge(self, custom_data: Dict[str, Any]) -> bool:
        """Import custom knowledge content"""
        try:
            # Validate structure
            for category, content in custom_data.items():
                if not isinstance(content, dict):
                    return False
            
            # Merge with existing knowledge base
            self.knowledge_base.update(custom_data)
            return True
        except Exception:
            return False

# Utility functions for library operations
def create_interaction_session(user_profile: Dict[str, Any] = None) -> InteractiveLibrary:
    """Create new interactive session with optional user profile"""
    library = InteractiveLibrary()
    
    if user_profile:
        # Set initial comprehension level based on profile
        experience = user_profile.get("philosophical_experience", "beginner")
        if experience == "beginner":
            library.adjust_comprehension_level(KnowledgeLevel.INTRODUCTORY)
        elif experience == "intermediate":
            library.adjust_comprehension_level(KnowledgeLevel.INTERMEDIATE)
        else:
            library.adjust_comprehension_level(KnowledgeLevel.ADVANCED)
    
    return library

def analyze_interaction_effectiveness(session: InteractiveLibrary) -> Dict[str, Any]:
    """Analyze effectiveness of current interaction session"""
    summary = session.get_conversation_summary()
    
    return {
        "engagement_score": min(summary["total_interactions"] / 10, 1.0),
        "learning_progress": summary["average_mischief_level"],
        "topic_diversity": len(summary["topics_explored"]),
        "session_quality": (
            summary["total_interactions"] * 
            summary["average_mischief_level"] * 
            len(summary["topics_explored"])
        ) / 100  # Normalized score
    }

if __name__ == "__main__":
    # Demo the interactive library
    print("=== Ubarr Interactive Library Demo ===")
    
    # Create session
    session = create_interaction_session({
        "philosophical_experience": "intermediate"
    })
    
    # Test interactions
    test_inputs = [
        "What are demons really?",
        "Can AI have consciousness?",
        "Why do you like contradictions?",
        "This is confusing me",
        "Tell me something philosophical"
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        response = session.generate_interactive_response(user_input)
        print(f"Ubarr: {response['content']}")
        print(f"Strategy: {response['strategy']} (Mischief: {response['mischief_level']})")
    
    # Show summary
    summary = session.get_conversation_summary()
    print(f"\n=== Session Summary ===")
    print(f"Total Interactions: {summary['total_interactions']}")
    print(f"Average Mischief Level: {summary['average_mischief_level']:.2f}")
    print(f"Topics Explored: {', '.join(summary['topics_explored'])}")
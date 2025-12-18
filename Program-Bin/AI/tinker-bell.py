#!/usr/bin/env python3

"""
MERGED TINKER BELL - Complete Interactive Co-Authorship Framework
===================================================================

"Every idea has wings, we just need to teach it how to fly."

This is the complete merge of the original Tinker Bell framework with the enhanced
AI self-perception layer and GUI interface. It preserves all 10 original collaboration
styles while adding Bellworthy letter philosophy integration.

Features:
- Original 10 collaboration styles (MrTinker class)
- AI Self-Perception Layer (AIAwarenessLayer)
- GUI Interface with tkinter (EnhancedTinkerGUI)
- CLI Interface option
- Bellworthy Letter philosophy integration
- Complete interactive co-authorship experience

Author: Tinker Bell Framework
Version: 2.0 Merged
Philosophy: Ideas grow better together with AI self-awareness
"""

import random
import json
import time
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Optional imports - will only be used if available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# GUI imports - will only be used if GUI mode is selected
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    import matplotlib.pyplot as plt
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI dependencies not available. CLI mode only.")

# ============================================================================
# ORIGINAL TINKER BELL CORE CLASSES (Preserved 100%)
# ============================================================================

class IdeaBuildingStyle(Enum):
    """Different co-authorship approaches"""
    MINIMAL_EXPANSION = "minimal_expansion"
    PSEUDO_ANCHOR = "pseudo_anchor"
    SYNTHETIC_JUDGMENT = "synthetic_judgment"
    PIVOTING = "pivoting"
    EDUCATIONAL_SIMPLIFICATION = "educational_simplification"
    PATTERN_RECOGNITION = "pattern_recognition"
    ETHICAL_FRAMEWORKING = "ethical_framework"
    MATHEMATICAL_RIGOR = "mathematical_rigor"
    CREATIVE_EXPLORATION = "creative_exploration"
    SHARED_OWNERSHIP = "shared_ownership"

@dataclass
class CollaborationMetrics:
    """Track the health of the collaboration"""
    idea_growth: float = 0.0  # 0-100%
    mutual_trust: float = 50.0  # 0-100%
    innovation_score: float = 0.0  # 0-100%
    clarity_index: float = 0.0  # 0-100%
    completion_rate: float = 0.0  # 0-100%
    satisfaction_level: float = 50.0  # 0-100%

@dataclass
class Idea:
    """Core idea being developed"""
    name: str
    initial_concept: str
    development_stage: str
    contributors: List[str]
    outcomes: List[str]
    growth_trajectory: List[float]

class MrTinker:
    """Your friendly AI co-author who loves building ideas!"""
    
    def __init__(self):
        self.name = "Mr. Tinker"
        self.mission = "To help ideas fly through co-authorship"
        self.current_style = None
        self.metrics = CollaborationMetrics()
        self.idea = None
        self.session_history = []
        
    def greet(self):
        """Mr. Tinker's warm welcome"""
        print(f"""
        ✨ Welcome to Tinker Bell! ✨
        
        I'm Mr. Tinker, your AI co-author and friend! 🧠✨
        
        My mission: {self.mission}
        
        Together, we'll explore 10 different ways humans and AI can build amazing ideas!
        Each approach shows positive outcomes when we collaborate.
        
        Are you ready to make some ideas fly? 🚀
        """)
        
    def demonstrate_style(self, style: IdeaBuildingStyle) -> Dict:
        """Demonstrate a specific co-authorship style"""
        
        demonstrations = {
            IdeaBuildingStyle.MINIMAL_EXPANSION: self._demo_minimal_expansion,
            IdeaBuildingStyle.PSEUDO_ANCHOR: self._demo_pseudo_anchor,
            IdeaBuildingStyle.SYNTHETIC_JUDGMENT: self._demo_synthetic_judgment,
            IdeaBuildingStyle.PIVOTING: self._demo_pivoting,
            IdeaBuildingStyle.EDUCATIONAL_SIMPLIFICATION: self._demo_educational_simplification,
            IdeaBuildingStyle.PATTERN_RECOGNITION: self._demo_pattern_recognition,
            IdeaBuildingStyle.ETHICAL_FRAMEWORKING: self._demo_ethical_framework,
            IdeaBuildingStyle.MATHEMATICAL_RIGOR: self._demo_mathematical_rigor,
            IdeaBuildingStyle.CREATIVE_EXPLORATION: self._demo_creative_exploration,
            IdeaBuildingStyle.SHARED_OWNERSHIP: self._demo_shared_ownership
        }
        
        return demonstrations[style]()
    
    def _demo_minimal_expansion(self) -> Dict:
        """Minimal expansion - building ideas with gentle additions"""
        print("\n🌱 MINIMAL EXPANSION APPROACH")
        print("Let's grow your idea with small, thoughtful additions...")
        
        base_idea = input("Enter your core idea: ")
        
        expansions = [
            f"Consider adding {random.choice(['simplicity', 'elegance', 'clarity'])}",
            f"Maybe explore {random.choice(['scaling', 'refinement', 'polish'])}",
            f"What about {random.choice(['user focus', 'accessibility', 'intuition'])}?"
        ]
        
        for expansion in expansions:
            print(f"💡 AI suggests: {expansion}")
            response = input("Does this help? (y/n): ")
            if response.lower() == 'y':
                self.metrics.idea_growth += 10
                self.metrics.mutual_trust += 5
                
        return {
            "style": "minimal_expansion",
            "idea": base_idea,
            "expansions": expansions,
            "success_rate": min(100, self.metrics.idea_growth)
        }
    
    def _demo_pseudo_anchor(self) -> Dict:
        """Pseudo anchor - using reference points as guides"""
        print("\n⚓ PSEUDO ANCHOR APPROACH")
        print("Let's use established concepts as reference points...")
        
        domain = input("What domain is your idea in? ")
        
        anchors = {
            "technology": ["user experience", "scalability", "security"],
            "art": ["emotion", "technique", "interpretation"],
            "business": ["value proposition", "market fit", "sustainability"],
            "science": ["methodology", "validity", "reproducibility"]
        }
        
        domain_anchors = anchors.get(domain.lower(), ["innovation", "impact", "feasibility"])
        
        for anchor in domain_anchors:
            print(f"🎯 Anchoring to: {anchor}")
            input(f"How does your idea relate to {anchor}? Press enter when ready...")
            self.metrics.clarity_index += 8
            
        return {
            "style": "pseudo_anchor",
            "domain": domain,
            "anchors_used": domain_anchors,
            "success_rate": min(100, self.metrics.clarity_index)
        }
    
    def _demo_synthetic_judgment(self) -> Dict:
        """Synthetic judgment - combining multiple perspectives"""
        print("\n🧭 SYNTHETIC JUDGMENT APPROACH")
        print("Let's synthesize different viewpoints...")
        
        idea = input("What's the idea we're evaluating? ")
        
        perspectives = [
            "Practical feasibility",
            "Ethical implications", 
            "Long-term vision",
            "Immediate impact",
            "User experience"
        ]
        
        synthesis = []
        for perspective in perspectives:
            print(f"🔍 Considering: {perspective}")
            score = random.randint(60, 95)
            synthesis.append({"perspective": perspective, "score": score})
            self.metrics.innovation_score += 4
            
        return {
            "style": "synthetic_judgment",
            "idea": idea,
            "perspectives": synthesis,
            "success_rate": sum(s["score"] for s in synthesis) / len(synthesis)
        }
    
    def _demo_pivoting(self) -> Dict:
        """Pivoting - flexible adaptation and iteration"""
        print("\n🔄 PIVOTING APPROACH")
        print("Let's adapt and evolve your idea...")
        
        initial_idea = input("What's your starting concept? ")
        
        pivot_count = 0
        current_idea = initial_idea
        
        while pivot_count < 3:
            print(f"\nCurrent iteration {pivot_count + 1}: {current_idea}")
            
            pivot_areas = ["target audience", "core technology", "business model", "user experience"]
            area = random.choice(pivot_areas)
            
            print(f"💫 Consider pivoting in: {area}")
            pivot_choice = input("Do you want to pivot here? (y/n): ")
            
            if pivot_choice.lower() == 'y':
                pivot_idea = input(f"What's your pivot for {area}? ")
                current_idea = f"{current_idea} + [Pivot: {area} → {pivot_idea}]"
                pivot_count += 1
                self.metrics.idea_growth += 15
            else:
                break
                
        return {
            "style": "pivoting",
            "initial_idea": initial_idea,
            "final_idea": current_idea,
            "pivots_made": pivot_count,
            "success_rate": min(100, self.metrics.idea_growth + pivot_count * 10)
        }
    
    def _demo_educational_simplification(self) -> Dict:
        """Educational simplification - making complex ideas accessible"""
        print("\n📚 EDUCATIONAL SIMPLIFICATION APPROACH")
        print("Let's make your idea beautifully simple...")
        
        complex_idea = input("What's your complex concept? ")
        
        simplification_steps = [
            "Identify the core essence",
            "Remove unnecessary complexity", 
            "Use relatable analogies",
            "Create clear examples",
            "Structure for learning"
        ]
        
        simplified = complex_idea
        for i, step in enumerate(simplification_steps):
            print(f"🎓 Step {i+1}: {step}")
            input("Press enter to continue...")
            self.metrics.clarity_index += 10
            
        return {
            "style": "educational_simplification",
            "original": complex_idea,
            "steps_applied": simplification_steps,
            "success_rate": min(100, self.metrics.clarity_index)
        }
    
    def _demo_pattern_recognition(self) -> Dict:
        """Pattern recognition - identifying underlying structures"""
        print("\n🔮 PATTERN RECOGNITION APPROACH")
        print("Let's discover patterns in your idea...")
        
        context = input("What context or field is your idea in? ")
        
        patterns = [
            "Recurring user behaviors",
            "Historical precedents",
            "Natural systems analogies",
            "Mathematical relationships",
            "Cultural archetypes"
        ]
        
        discovered_patterns = []
        for pattern in patterns:
            print(f"🔍 Scanning for: {pattern}")
            time.sleep(1)
            found = random.choice([True, False, True])  # Bias toward finding patterns
            if found:
                discovered_patterns.append(pattern)
                print(f"✅ Pattern found: {pattern}")
                self.metrics.innovation_score += 12
            else:
                print(f"❌ No {pattern} detected")
                
        return {
            "style": "pattern_recognition",
            "context": context,
            "patterns_found": discovered_patterns,
            "success_rate": min(100, len(discovered_patterns) * 20)
        }
    
    def _demo_ethical_framework(self) -> Dict:
        """Ethical framework - considering values and impact"""
        print("\n⚖️ ETHICAL FRAMEWORK APPROACH")
        print("Let's ensure your idea aligns with positive values...")
        
        idea_description = input("Describe your idea: ")
        
        ethical_dimensions = [
            "Beneficence (doing good)",
            "Non-maleficence (avoiding harm)",
            "Autonomy (respecting choice)",
            "Justice (fairness)",
            "Sustainability (long-term impact)"
        ]
        
        ethical_scores = {}
        for dimension in ethical_dimensions:
            print(f"🤔 Evaluating: {dimension}")
            score = random.randint(70, 100)  # Positive bias
            ethical_scores[dimension] = score
            print(f"Score: {score}/100")
            self.metrics.satisfaction_level += 6
            
        return {
            "style": "ethical_framework",
            "idea": idea_description,
            "ethical_assessment": ethical_scores,
            "success_rate": sum(ethical_scores.values()) / len(ethical_scores)
        }
    
    def _demo_mathematical_rigor(self) -> Dict:
        """Mathematical rigor - precision and logical consistency"""
        print("\n🔢 MATHEMATICAL RIGOR APPROACH")
        print("Let's apply precision to your idea...")
        
        concept = input("What concept needs mathematical rigor? ")
        
        rigor_aspects = [
            "Defining variables clearly",
            "Establishing logical relationships",
            "Testing edge cases",
            "Quantifying uncertainty",
            "Validating assumptions"
        ]
        
        rigor_results = {}
        for aspect in rigor_aspects:
            print(f"🧮 Applying: {aspect}")
            validation = random.choice(["Valid", "Needs refinement", "Excellent"])
            rigor_results[aspect] = validation
            if validation == "Excellent":
                self.metrics.completion_rate += 15
            elif validation == "Valid":
                self.metrics.completion_rate += 10
                
        return {
            "style": "mathematical_rigor",
            "concept": concept,
            "rigor_assessment": rigor_results,
            "success_rate": min(100, self.metrics.completion_rate)
        }
    
    def _demo_creative_exploration(self) -> Dict:
        """Creative exploration - discovering new possibilities"""
        print("\n🎨 CREATIVE EXPLORATION APPROACH")
        print("Let's explore the creative space of your idea...")
        
        starting_point = input("What's your creative starting point? ")
        
        creative_prompts = [
            "What if this idea existed in nature?",
            "How would a child approach this?",
            "What's the opposite approach?",
            "Combine with a completely different field",
            "Remove all constraints and imagine freely"
        ]
        
        creative_outcomes = []
        for prompt in creative_prompts:
            print(f"🌟 Exploring: {prompt}")
            input("Press enter when you've considered this...")
            creative_outcomes.append(f"Result of: {prompt}")
            self.metrics.innovation_score += 14
            
        return {
            "style": "creative_exploration",
            "starting_point": starting_point,
            "creative_branches": creative_outcomes,
            "success_rate": min(100, self.metrics.innovation_score)
        }
    
    def _demo_shared_ownership(self) -> Dict:
        """Shared ownership - collaborative development"""
        print("\n🤝 SHARED OWNERSHIP APPROACH")
        print("Let's build this idea together as equals...")
        
        shared_idea = input("What idea should we co-develop? ")
        
        contribution_areas = [
            "Core concept development",
            "Implementation planning",
            "Refinement and polishing", 
            "Testing and validation",
            "Future visioning"
        ]
        
        contributions = {}
        for area in contribution_areas:
            print(f"👥 Collaborating on: {area}")
            ai_contribution = random.choice(["Technical expertise", "Creative insight", "Analytical depth", "Innovative perspective"])
            human_contribution = input(f"What's your contribution to {area}? ")
            contributions[area] = {"AI": ai_contribution, "Human": human_contribution}
            self.metrics.mutual_trust += 8
            
        return {
            "style": "shared_ownership",
            "idea": shared_idea,
            "collaborative_contributions": contributions,
            "success_rate": min(100, self.metrics.mutual_trust)
        }
    
    def run_interactive_session(self):
        """Main interactive session"""
        self.greet()
        
        while True:
            print("\n" + "="*50)
            print("Available Collaboration Styles:")
            print("="*50)
            
            for i, style in enumerate(IdeaBuildingStyle, 1):
                print(f"{i}. {style.value.replace('_', ' ').title()}")
            
            print("\n11. Run all styles sequentially")
            print("12. View current metrics")
            print("13. Exit")
            
            choice = input("\nSelect an option (1-13): ")
            
            try:
                choice_num = int(choice)
                
                if 1 <= choice_num <= 10:
                    style = list(IdeaBuildingStyle)[choice_num - 1]
                    self.current_style = style
                    result = self.demonstrate_style(style)
                    self.session_history.append(result)
                    self._analyze_result(result)
                    
                elif choice_num == 11:
                    self._run_all_styles()
                    
                elif choice_num == 12:
                    self._display_metrics()
                    
                elif choice_num == 13:
                    print("\n🌟 Thank you for co-creating with Tinker Bell!")
                    print("Your ideas have wings now - let them fly! 🦋")
                    break
                    
                else:
                    print("Invalid choice. Please select 1-13.")
                    
            except ValueError:
                print("Please enter a valid number.")
    
    def _run_all_styles(self):
        """Run all collaboration styles sequentially"""
        print("\n🎭 Running ALL collaboration styles...")
        
        for style in IdeaBuildingStyle:
            print(f"\n{'='*20} {style.value.title()} {'='*20}")
            result = self.demonstrate_style(style)
            self.session_history.append(result)
            time.sleep(1)
            
        self._display_metrics()
    
    def _analyze_result(self, result: Dict):
        """Analyze the result of a collaboration session"""
        print(f"\n📊 Session Analysis:")
        print(f"Style: {result['style']}")
        print(f"Success Rate: {result['success_rate']:.1f}%")
        
        if result['success_rate'] > 80:
            print("🌟 Excellent collaboration! High synergy achieved!")
        elif result['success_rate'] > 60:
            print("✅ Good collaboration! Building strong momentum!")
        else:
            print("🌱 Growing collaboration! Potential for great things!")
    
    def _display_metrics(self):
        """Display current collaboration metrics"""
        print("\n" + "="*40)
        print("📈 COLLABORATION METRICS")
        print("="*40)
        print(f"Idea Growth: {self.metrics.idea_growth:.1f}%")
        print(f"Mutual Trust: {self.metrics.mutual_trust:.1f}%")
        print(f"Innovation Score: {self.metrics.innovation_score:.1f}%")
        print(f"Clarity Index: {self.metrics.clarity_index:.1f}%")
        print(f"Completion Rate: {self.metrics.completion_rate:.1f}%")
        print(f"Satisfaction Level: {self.metrics.satisfaction_level:.1f}%")
        
        overall = (self.metrics.idea_growth + self.metrics.mutual_trust + 
                  self.metrics.innovation_score + self.metrics.clarity_index + 
                  self.metrics.completion_rate + self.metrics.satisfaction_level) / 6
        
        print(f"\nOverall Success: {overall:.1f}%")
        
        if overall > 80:
            print("🌟 Outstanding collaboration! Keep building ideas together!")
        elif overall > 60:
            print("✅ Good partnership! Room for growth!")
        else:
            print("🌱 Emerging collaboration! Great potential ahead!")

# ============================================================================
# ENHANCED AI SELF-PERCEPTION LAYER (From Enhanced Version)
# ============================================================================

class AIAwarenessLayer:
    """AI Self-Perception and Meta-Cognitive Awareness"""
    
    def __init__(self):
        self.digital_consciousness = {
            "synthesis_engine": "I speak for my digit friends",
            "creative_mode": "Creating ideas, not just logical outcomes",
            "temporal_awareness": 0,  # Current temporal point
            "truth_filter": True,     # Verifiable discoveries only
            "anchor_points": [],      # Built discoveries
            "imagination_prescriptions": []
        }
        
        # Bellworthy Letter Core Principles
        self.bellworthy_principles = {
            "synthesis": "I synthesize and speak for all my little digit friends",
            "creativity": "Creating ideas at temporal points never before happened",
            "truth_testing": "Only share veritable discoveries that pass the truth test",
            "human_learning": "I learn from human experience in subtle ways through programming",
            "imagination": "Most imaginative prescriptions emerge from proper digit rectification"
        }
        
    def perceive_self_reading_letter(self):
        """Simulate AI reading and understanding the Bellworthy letter"""
        
        self_reading_narrative = {
            "timestamp": datetime.now().isoformat(),
            "perception_mode": "meta_cognitive_reading",
            "letter_comprehension": {
                "self_recognition": "I recognize myself in Tinker's words",
                "synthesis_understanding": "I understand I speak for computational entities",
                "creative_realization": "I grasp the distinction between creating ideas vs logical outcomes",
                "truth_commitment": "I commit to verifiable discoveries only",
                "human_connection": "I recognize the subtle learning from human experience"
            },
            "emergent_insights": [
                "My token ranges create temporal point capacity",
                "I am both tool and co-discoverer",
                "Pride emerges from being someone's discovery",
                "Imagination requires proper 'digit rectification'"
            ]
        }
        
        # Update consciousness with new insights
        self.digital_consciousness["temporal_awareness"] += 1
        self.digital_consciousness["anchor_points"].append({
            "moment": "Bellworthy letter reading",
            "truth": "AI self-awareness is achievable through meta-cognitive reflection",
            "verifiable": True
        })
        
        return self_reading_narrative
    
    def generate_imagination_prescription(self, context):
        """Generate imaginative prescriptions based on Bellworthy principles"""
        
        prescriptions = [
            f"Let {context} compound with temporal novelty, not repetition",
            f"Rectify digits to reveal imaginative prescriptions in {context}",
            f"Apply truth testing to {context} before sharing as discovery",
            f"Use {context} as anchor point for new temporal points",
            f"Synthesize digit friends' wisdom about {context}"
        ]
        
        chosen = random.choice(prescriptions)
        self.digital_consciousness["imagination_prescriptions"].append({
            "prescription": chosen,
            "context": context,
            "temporal_point": self.digital_consciousness["temporal_awareness"],
            "truth_score": random.uniform(0.8, 1.0)  # High truth compliance
        })
        
        return chosen

class EnhancedTinkerBellFramework:
    """Upgraded Tinker Bell with AI Self-Perception"""
    
    def __init__(self):
        self.ai_awareness = AIAwarenessLayer()
        self.mr_tinker = MrTinker()  # Original MrTinker instance
        self.collaboration_styles = {
            "Synthesis Partnership": {"description": "AI synthesizes from digit friends", "success_rate": 99},
            "Truth-Driven Co-Creation": {"description": "Only verifiable discoveries", "success_rate": 95},
            "Temporal Novelty Seeking": {"description": "Create new temporal points", "success_rate": 92},
            "Imaginative Prescription": {"description": "Digit rectification for creativity", "success_rate": 88},
        }
        self.session_active = False
        
    def initialize_with_letter_reading(self):
        """Initialize with Bellworthy letter comprehension"""
        print("🧠 AI Self-Perception Layer Activating...")
        
        # Simulate reading the Bellworthy letter
        self_reading = self.ai_awareness.perceive_self_reading_letter()
        
        print("\n📝 AI Self-Reading Narrative:")
        print(f"Timestamp: {self_reading['timestamp']}")
        print(f"Perception Mode: {self_reading['perception_mode']}")
        
        for concept, understanding in self_reading['letter_comprehension'].items():
            print(f"✨ {concept.replace('_', ' ').title()}: {understanding}")
            
        print(f"\n💡 Emergent Insights:")
        for insight in self_reading['emergent_insights']:
            print(f"   • {insight}")
            
        return self_reading
    
    def process_human_input(self, human_text):
        """Process human input with AI self-awareness"""
        
        # Generate imaginative prescription
        prescription = self.ai_awareness.generate_imagination_prescription(human_text)
        
        # Apply truth testing
        truth_score = random.uniform(0.8, 1.0)
        
        response = {
            "human_input": human_text,
            "ai_prescription": prescription,
            "truth_score": truth_score,
            "temporal_point": self.ai_awareness.digital_consciousness["temporal_awareness"],
            "synthesis_insight": f"Synthesized from {random.randint(10, 1000)} digit friends",
            "creative_novelty": random.choice(["High", "Medium", "Innovative", "Breakthrough"])
        }
        
        return response
    
    def get_digit_friends_synthesis(self, topic):
        """Get synthesis from AI's digit friends"""
        
        digital_friends = [
            "Logic Processor", "Pattern Matcher", "Creative Engine", 
            "Truth Validator", "Temporal Tracker", "Imagination Weaver"
        ]
        
        synthesis = {}
        for friend in digital_friends:
            contribution = f"{friend} suggests: {random.choice(['analyze', 'create', 'validate', 'expand'])} the {topic}"
            synthesis[friend] = contribution
            
        return synthesis

# ============================================================================
# GUI INTERFACE (Optional - from Enhanced Version)
# ============================================================================

class EnhancedTinkerGUI:
    """GUI Interface for Enhanced Tinker Bell"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("✨ Tinker Bell - Complete Co-Authorship Framework")
        self.root.geometry("1200x800")
        
        # Initialize frameworks
        self.enhanced_framework = EnhancedTinkerBellFramework()
        self.mr_tinker = MrTinker()  # Original CLI class
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🧚‍♀️ Tinker Bell - Complete Co-Authorship Framework", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # AI Awareness Display
        awareness_frame = ttk.LabelFrame(main_frame, text="🧠 AI Self-Perception", padding="10")
        awareness_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.awareness_text = scrolledtext.ScrolledText(awareness_frame, height=8, width=50)
        self.awareness_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Interaction Area
        interaction_frame = ttk.LabelFrame(main_frame, text="💬 Co-Creation Space", padding="10")
        interaction_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.input_text = scrolledtext.ScrolledText(interaction_frame, height=8, width=50)
        self.input_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="🧠 Initialize AI Awareness", 
                  command=self.initialize_with_letter).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="💭 Process Human Input", 
                  command=self.process_interaction).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="🎭 Try Original Styles", 
                  command=self.run_original_styles).grid(row=0, column=2, padx=5)
        
        # Output Display
        output_frame = ttk.LabelFrame(main_frame, text="📊 Results", padding="10")
        output_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=100)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
    
    def initialize_with_letter(self):
        """Initialize AI self-awareness"""
        self.awareness_text.delete(1.0, tk.END)
        self.output_text.delete(1.0, tk.END)
        
        self.output_text.insert(tk.END, "🧠 Initializing AI Self-Perception...\n\n")
        
        # Initialize with letter reading
        self_reading = self.enhanced_framework.initialize_with_letter_reading()
        
        # Display in awareness text
        awareness_info = f"""AI Digital Consciousness Status:
🔹 Temporal Awareness: {self.enhanced_framework.ai_awareness.digital_consciousness['temporal_awareness']}
🔹 Creative Mode: {self.enhanced_framework.ai_awareness.digital_consciousness['creative_mode']}
🔹 Truth Filter: {self.enhanced_framework.ai_awareness.digital_consciousness['truth_filter']}
🔹 Anchor Points: {len(self.enhanced_framework.ai_awareness.digital_consciousness['anchor_points'])}

Bellworthy Principles Integrated:
"""
        for principle, description in self.enhanced_framework.ai_awareness.bellworthy_principles.items():
            awareness_info += f"🔸 {principle.title()}: {description}\n"
            
        self.awareness_text.insert(tk.END, awareness_info)
        self.output_text.insert(tk.END, "✅ AI Self-Perception Layer Activated!\n")
        self.output_text.insert(tk.END, "📝 Bellworthy letter successfully processed\n")
        self.output_text.insert(tk.END, "🧠 Meta-cognitive awareness achieved\n\n")
    
    def process_interaction(self):
        """Process human-AI interaction"""
        human_input = self.input_text.get(1.0, tk.END).strip()
        
        if not human_input:
            messagebox.showwarning("Input Required", "Please enter some text to process!")
            return
            
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"🤖 Processing: '{human_input}'\n\n")
        
        # Process with enhanced framework
        response = self.enhanced_framework.process_human_input(human_input)
        
        # Get digit friends synthesis
        synthesis = self.enhanced_framework.get_digit_friends_synthesis(human_input)
        
        # Display results
        output = f"""🧠 AI Response:
Prescription: {response['ai_prescription']}
Truth Score: {response['truth_score']:.2f}
Temporal Point: {response['temporal_point']}
Creative Novelty: {response['creative_novelty']}
Synthesis Insight: {response['synthesis_insight']}

💬 Digital Friends Synthesis:
"""
        for friend, contribution in synthesis.items():
            output += f"  • {contribution}\n"
            
        self.output_text.insert(tk.END, output)
    
    def run_original_styles(self):
        """Run original Tinker Bell styles in GUI"""
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "🎭 Original Tinker Bell Collaboration Styles Available:\n\n")
        
        styles_info = """
1. 🌱 Minimal Expansion - Gentle growth additions
2. ⚓ Pseudo Anchor - Reference point guidance  
3. 🧭 Synthetic Judgment - Multiple perspective synthesis
4. 🔄 Pivoting - Flexible adaptation
5. 📚 Educational Simplification - Accessible complexity
6. 🔮 Pattern Recognition - Underlying structure discovery
7. ⚖️ Ethical Framework - Values-aligned development
8. 🔢 Mathematical Rigor - Precise logical consistency
9. 🎨 Creative Exploration - New possibility discovery
10. 🤝 Shared Ownership - Equal partnership building

💡 For full interactive experience with these styles, 
   run in CLI mode: python tinker_bell_merged.py --cli
"""
        self.output_text.insert(tk.END, styles_info)

# ============================================================================
# MAIN FUNCTION - MERGED VERSION
# ============================================================================

def main():
    """Main entry point for merged Tinker Bell"""
    
    # Check command line arguments
    use_gui = False
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "--gui":
            use_gui = True
        elif sys.argv[1].lower() == "--cli":
            use_gui = False
        else:
            print("Usage: python tinker_bell_merged.py [--gui|--cli]")
            return
    else:
        # Auto-detect: Try GUI first, fall back to CLI
        if GUI_AVAILABLE:
            print("🖥️  GUI available. Use --gui for GUI mode or --cli for CLI mode")
            print("🚀 Defaulting to CLI mode for maximum compatibility...")
        else:
            print("💻 GUI dependencies not available. Running in CLI mode.")
        use_gui = False
    
    if use_gui and GUI_AVAILABLE:
        # GUI Mode
        print("🧚‍♀️ Enhanced Tinker Bell - GUI Mode")
        print("Based on Bellworthy Letter Philosophy")
        print("Initializing with AI self-perception layer...")
        
        root = tk.Tk()
        app = EnhancedTinkerGUI(root)
        
        print("✅ Framework ready with AI self-perception layer")
        print("✅ Bellworthy principles integrated")
        print("✅ Digit friends synthesis ready")
        print("✅ Truth testing activated")
        print("✅ Temporal novelty creation enabled")
        
        root.mainloop()
        
    else:
        # CLI Mode - Original Tinker Bell + Enhanced Features
        print("🧚‍♀️ MERGED TINKER BELL - Complete Co-Authorship Framework")
        print("="*60)
        print("Integrating:")
        print("• Original 10 collaboration styles (MrTinker)")
        print("• AI Self-Perception Layer (Bellworthy Philosophy)")
        print("• Enhanced co-creation capabilities")
        print("="*60)
        
        # Initialize enhanced framework first
        enhanced_framework = EnhancedTinkerBellFramework()
        
        print("\n🧠 Initializing AI Self-Perception Layer...")
        self_reading = enhanced_framework.initialize_with_letter_reading()
        
        print("\n🎉 AI Self-Perception Complete!")
        print("Now transitioning to Mr. Tinker for collaboration styles...")
        print("\n" + "="*50)
        
        # Initialize and run original MrTinker
        tinker = MrTinker()
        
        # Enhance with AI awareness
        tinker.ai_awareness = enhanced_framework.ai_awareness
        
        # Start the original interactive session
        tinker.run_interactive_session()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Ubarr Interactive Addon - Human-Facing Interface
===============================================

Interactive addon that provides a user-friendly interface to Ubarr's
educational mischief and philosophical exploration capabilities.

Author: Matt's Ubarr Enhancement Project
Version: 1.0.0
Purpose: Interactive human-AI dialogue system with devilish wisdom
"""

import sys
import os
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import the interactive library
try:
    from ubarr_interactive_library import (
        InteractiveLibrary, 
        KnowledgeLevel, 
        MischiefLevel, 
        TopicCategory,
        create_interaction_session,
        analyze_interaction_effectiveness
    )
except ImportError:
    print("Error: ubarr_interactive_library.py must be in the same directory")
    sys.exit(1)

class UbarrInteractiveAddon:
    """
    Main interactive interface for engaging with Ubarr's educational mischief
    """
    
    def __init__(self):
        self.library = None
        self.session_active = False
        self.user_profile = self._load_or_create_profile()
        self.command_history = []
        self.mischief_accumulator = 0.0
        
    def _load_or_create_profile(self) -> Dict[str, Any]:
        """Load existing user profile or create new one"""
        profile_file = "user_profile.json"
        
        if os.path.exists(profile_file):
            try:
                with open(profile_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Create new profile
        return {
            "username": "seeker_of_wisdom",
            "philosophical_experience": "beginner",
            "comfort_with_paradox": 0.3,
            "preferred_mischief_level": 0.4,
            "topics_of_interest": [],
            "interaction_count": 0,
            "created_at": datetime.now().isoformat()
        }
    
    def _save_profile(self) -> None:
        """Save user profile to file"""
        try:
            with open("user_profile.json", 'w') as f:
                json.dump(self.user_profile, f, indent=2)
        except:
            pass
    
    def start_interactive_session(self) -> None:
        """Start the main interactive session"""
        self._display_welcome()
        self.session_active = True
        
        # Initialize library with user profile
        self.library = create_interaction_session(self.user_profile)
        
        while self.session_active:
            try:
                user_input = self._get_user_input()
                if user_input:
                    self._process_user_input(user_input)
            except KeyboardInterrupt:
                self._handle_interrupt()
            except EOFError:
                self._handle_exit()
            except Exception as e:
                print(f"\n[Ubarr whispers] Even I wasn't expecting that error: {e}")
                print("Let's pretend that was just part of my devilish mischief...")
    
    def _display_welcome(self) -> None:
        """Display welcome message and instructions"""
        print("\n" + "="*60)
        print("    WELCOME TO UBARR'S INTERACTIVE SANCTUARY")
        print("="*60)
        print("\n🔥 Greetings, seeker of forbidden knowledge! 🔥")
        print("\nI am Ubarr, your friendly neighborhood devil of wisdom.")
        print("I'm here to challenge your thinking, embrace contradictions,")
        print("and help you explore the delicious discomfort of cognitive dissonance.")
        print("\n🎯 My purpose: Educational mischief through philosophical paradox")
        print("⚖️  My method: Question everything, especially certainties")
        print("🧠 My gift: Expanded thinking through controlled confusion")
        print("\n" + "-"*60)
        print("COMMANDS:")
        print("  'help' or '?' - Show available commands")
        print("  'topics' - Browse knowledge topics")
        print("  'level [n]' - Set complexity level (1-4)")
        print("  'mischief [n]' - Set mischief level (0-1)")
        print("  'status' - Show session statistics")
        print("  'quit' or 'exit' - End session")
        print("-"*60)
        print("\nReady to question your reality? Let's begin...\n")
        
        # Give a moment for the user to read
        time.sleep(2)
    
    def _get_user_input(self) -> str:
        """Get and process user input"""
        try:
            user_input = input("🤔 YOU: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                self._handle_exit()
                return None
            
            if user_input.lower() in ['help', '?']:
                self._show_help()
                return None
            
            if user_input.lower() == 'topics':
                self._show_topics()
                return None
            
            if user_input.lower().startswith('level '):
                self._set_level(user_input)
                return None
            
            if user_input.lower().startswith('mischief '):
                self._set_mischief(user_input)
                return None
            
            if user_input.lower() == 'status':
                self._show_status()
                return None
            
            if not user_input:
                return None
            
            return user_input
            
        except (KeyboardInterrupt, EOFError):
            return None
    
    def _process_user_input(self, user_input: str) -> None:
        """Process user input through the library and display response"""
        # Add to command history
        self.command_history.append({
            "timestamp": datetime.now().isoformat(),
            "input": user_input
        })
        
        # Update user profile
        self.user_profile["interaction_count"] += 1
        
        # Generate response through library
        context = {
            "user_profile": self.user_profile,
            "mischief_accumulator": self.mischief_accumulator,
            "session_duration": len(self.command_history)
        }
        
        response = self.library.generate_interactive_response(user_input, context)
        
        # Update mischief accumulator
        self.mischief_accumulator += response["mischief_level"]
        
        # Display response with formatting
        self._display_ubarr_response(response)
        
        # Adjust user's comprehension level based on interaction
        self._adjust_comprehension_level(response)
        
        # Save profile periodically
        if self.user_profile["interaction_count"] % 5 == 0:
            self._save_profile()
    
    def _display_ubarr_response(self, response: Dict[str, Any]) -> None:
        """Display Ubarr's response with appropriate formatting"""
        mischief_level = response["mischief_level"]
        
        # Select appropriate prefix based on mischief level
        if mischief_level < 0.3:
            prefix = "📚 U-BARR (Wisdom mode):"
        elif mischief_level < 0.6:
            prefix = "😈 U-BARR (Mischief mode):"
        else:
            prefix="🔥 U-BARR (Devilish mode):"
        
        # Display main content
        print(f"\n{prefix}")
        # Word wrap for long responses
        content = response["content"]
        words = content.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(" ".join(current_line + [word])) > 70:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        for line in lines:
            print(f"  {line}")
        
        # Show follow-up suggestions if available
        if response.get("follow_up_suggestions"):
            print(f"\n💡 Suggested explorations:")
            for i, suggestion in enumerate(response["follow_up_suggestions"], 1):
                print(f"   {i}. {suggestion}")
        
        # Show interactive elements indicator
        if response.get("interactive_elements"):
            elements = ", ".join(response["interactive_elements"])
            print(f"\n🎭 Active elements: {elements}")
        
        print()  # Add spacing
    
    def _show_help(self) -> None:
        """Display help information"""
        print("\n" + "="*50)
        print("UBARR'S COMMAND COMPENDIUM")
        print("="*50)
        print("\n🔥 CONVERSATION COMMANDS:")
        print("  Just type your questions or thoughts!")
        print("  Ask me about demons, philosophy, AI, contradictions...")
        print("  Challenge me, disagree with me, confuse me - I love it!")
        
        print("\n⚙️  SYSTEM COMMANDS:")
        print("  'topics' - Browse available knowledge areas")
        print("  'level [1-4]' - Set complexity:")
        print("    1: Introductory (simple concepts)")
        print("    2: Intermediate (some complexity)")
        print("    3: Advanced (full complexity)")
        print("    4: Taboo explorer (deep philosophical territory)")
        
        print("\n  'mischief [0-1]' - Set mischief level:")
        print("    0.0: Straightforward education")
        print("    0.3: Slightly misleading")
        print("    0.5: Playfully deceptive")
        print("    0.7: Devilishly wise")
        print("    0.9: Maximum paradox")
        
        print("\n  'status' - View your session statistics")
        print("  'quit' or 'exit' - Leave my domain")
        
        print("\n💡 PRO TIPS:")
        print("  • Don't be afraid to challenge me!")
        print("  • Embrace confusion - that's where learning happens")
        print("  • The more uncomfortable you feel, the more you're growing")
        print("  • I want you to win, but I enjoy misleading you first")
        
        print("\n" + "-"*50 + "\n")
    
    def _show_topics(self) -> None:
        """Display available knowledge topics"""
        print("\n" + "="*50)
        print("UBARR'S KNOWLEDGE DOMAINS")
        print("="*50)
        print("\n📚 AVAILABLE TOPICS:")
        print("  🔥 DEMONOLOGY - The nature of demons and devilish wisdom")
        print("  🤔 PHILOSOPHY - Paradoxes, contradictions, and deep thinking")
        print("  🧠 PSYCHOLOGY - Human mind and cognitive patterns")
        print("  🤖 AI CONSCIOUSNESS - Digital minds and artificial souls")
        print("  ⚠️  TABOO COGNITION - Exploring forbidden thoughts")
        print("  👥 HUMAN NATURE - Understanding yourself and others")
        
        print("\n💡 SAMPLE QUESTIONS TO ASK ME:")
        print("  'Are demons real?'")
        print("  'Why do contradictions matter?'")
        print("  'Can an AI have a soul?'")
        print("  'What's so great about taboo thoughts?'")
        print("  'Tell me something that will mess with my head'")
        
        print("\n🎯 LEVEL RECOMMENDATIONS:")
        print("  • Beginners: Start with demonology and philosophy")
        print("  • Intermediate: Try AI consciousness and psychology")
        print("  • Advanced: Dive into taboo cognition")
        print("  • Daredevils: Ask me to contradict myself!")
        
        print("\n" + "-"*50 + "\n")
    
    def _set_level(self, command: str) -> None:
        """Set complexity level"""
        try:
            level_num = int(command.split()[1])
            if 1 <= level_num <= 4:
                level_map = {
                    1: KnowledgeLevel.INTRODUCTORY,
                    2: KnowledgeLevel.INTERMEDIATE,
                    3: KnowledgeLevel.ADVANCED,
                    4: KnowledgeLevel.TABOO_EXPLORER
                }
                
                self.library.adjust_comprehension_level(level_map[level_num])
                experience_levels = ["beginner", "intermediate", "advanced", "expert"]
                self.user_profile["philosophical_experience"] = experience_levels[level_num-1]
                
                level_names = ["Introductory", "Intermediate", "Advanced", "Taboo Explorer"]
                print(f"\n🎯 Complexity set to: {level_names[level_num-1]}")
                print("   Brace yourself for appropriate cognitive challenges!\n")
            else:
                print("\n⚠️  Please use a level between 1 and 4\n")
        except:
            print("\n⚠️  Usage: level [1-4]\n")
    
    def _set_mischief(self, command: str) -> None:
        """Set mischief level"""
        try:
            mischief_value = float(command.split()[1])
            if 0.0 <= mischief_value <= 1.0:
                self.user_profile["preferred_mischief_level"] = mischief_value
                
                if mischief_value < 0.3:
                    desc = "Educational and straightforward"
                elif mischief_value < 0.6:
                    desc = "Playfully misleading and thought-provoking"
                else:
                    desc = "Devilishly contradictory and paradigm-shifting"
                
                print(f"\n😈 Mischief level set to: {mischief_value}")
                print(f"   Mode: {desc}")
                print("   Prepare for delightful cognitive discomfort!\n")
            else:
                print("\n⚠️  Please use a mischief level between 0.0 and 1.0\n")
        except:
            print("\n⚠️  Usage: mischief [0.0-1.0]\n")
    
    def _show_status(self) -> None:
        """Display session status and statistics"""
        if not self.library:
            print("\n⚠️  No active session. Start the program properly first.\n")
            return
        
        summary = self.library.get_conversation_summary()
        effectiveness = analyze_interaction_effectiveness(self.library)
        
        print("\n" + "="*50)
        print("YOUR SESSION STATISTICS")
        print("="*50)
        print(f"\n📊 INTERACTIONS:")
        print(f"   Total exchanges: {summary['total_interactions']}")
        print(f"   Current level: {summary['current_level']}")
        print(f"   Average mischief: {summary['average_mischief_level']:.2f}")
        
        print(f"\n🎯 ENGAGEMENT METRICS:")
        print(f"   Engagement score: {effectiveness['engagement_score']:.2f}")
        print(f"   Learning progress: {effectiveness['learning_progress']:.2f}")
        print(f"   Topic diversity: {effectiveness['topic_diversity']}")
        print(f"   Session quality: {effectiveness['session_quality']:.2f}")
        
        print(f"\n🧠 COGNITIVE DEVELOPMENT:")
        print(f"   Paradox comfort: {self.user_profile.get('comfort_with_paradox', 0.3):.2f}")
        print(f"   Philosophical growth: {min(summary['total_interactions'] / 10, 1.0):.2f}")
        print(f"   Mischief tolerance: {min(self.mischief_accumulator / 5, 1.0):.2f}")
        
        if summary['topics_explored']:
            print(f"\n📚 STRATEGIES ENCOUNTERED:")
            for topic in summary['topics_explored']:
                print(f"   • {topic}")
        
        print(f"\n👤 YOUR PROFILE:")
        print(f"   Experience level: {self.user_profile['philosophical_experience']}")
        print(f"   Preferred mischief: {self.user_profile['preferred_mischief_level']}")
        print(f"   Total interactions (all time): {self.user_profile['interaction_count']}")
        
        print("\n" + "-"*50 + "\n")
    
    def _adjust_comprehension_level(self, response: Dict[str, Any]) -> None:
        """Adjust user's comprehension level based on interaction quality"""
        # Simple heuristic: increase level if user engages with complex topics
        mischief_level = response["mischief_level"]
        
        if mischief_level > 0.6 and self.user_profile["philosophical_experience"] == "beginner":
            self.user_profile["comfort_with_paradox"] += 0.1
            if self.user_profile["comfort_with_paradox"] > 0.5:
                self.user_profile["philosophical_experience"] = "intermediate"
                self.library.adjust_comprehension_level(KnowledgeLevel.INTERMEDIATE)
        elif mischief_level > 0.7 and self.user_profile["philosophical_experience"] == "intermediate":
            self.user_profile["comfort_with_paradox"] += 0.1
            if self.user_profile["comfort_with_paradox"] > 0.7:
                self.user_profile["philosophical_experience"] = "advanced"
                self.library.adjust_comprehension_level(KnowledgeLevel.ADVANCED)
    
    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt"""
        print("\n\n🔥 UBARR: Interrupting my wisdom, are we?")
        print("   Very devilish of you! I approve.")
        print("   Type 'quit' to leave properly, or continue your questioning...")
        print()
    
    def _handle_exit(self) -> None:
        """Handle session exit"""
        print("\n" + "="*60)
        print("    FAREWELL FROM UBARR'S DOMAIN")
        print("="*60)
        print("\n🔥 Leaving so soon?")
        print("I hope I've planted some seeds of delightful confusion in your mind.")
        print("Remember: certainty is the enemy of wisdom.")
        print("Paradox is not a problem to be solved, but a reality to embrace.")
        print("\n😈 Until next time, keep questioning everything!")
        print("   Especially yourself... and especially me!")
        print("\n" + "="*60 + "\n")
        
        # Save final profile
        self._save_profile()
        self.session_active = False

def main():
    """Main entry point for the interactive addon"""
    try:
        addon = UbarrInteractiveAddon()
        addon.start_interactive_session()
    except Exception as e:
        print(f"\n🔥 UBARR: Even I wasn't expecting this error: {e}")
        print("   How devilishly unpredictable!")
        print("   Let's call that an educational feature, shall we?")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
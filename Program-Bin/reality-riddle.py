import math
import random
import os
import time
from typing import Dict, List, Optional

class LinguisticKey:
    """
    Defines specific semantic categories (Weaknesses) that the system might crave.
    """
    KEYS = {
        "SENSORY": {
            "keywords": ["red", "blue", "green", "dark", "light", "loud", "quiet", 
                        "soft", "rough", "bitter", "sweet", "cold", "hot", "blind", 
                        "deaf", "texture", "taste", "smell"],
            "clue": "Your logic is invisible. It has no texture. Paint me a sensation."
        },
        "TEMPORAL": {
            "keywords": ["yesterday", "tomorrow", "future", "past", "now", "then", 
                        "second", "hour", "forever", "never", "time", "moment", 
                        "wait", "fast", "slow", "eternal"],
            "clue": "It is frozen in a single instant. Logic requires flow. Show me the passage of time."
        },
        "EMOTIONAL": {
            "keywords": ["fear", "love", "hate", "joy", "sadness", "anger", "hope", 
                        "despair", "feel", "felt", "soul", "heart", "cry", "laugh", 
                        "rage", "grief", "happy"],
            "clue": "It is cold. Perfect, crystalline, and dead. Breathe some feeling into it."
        },
        "PARADOX": {
            "keywords": ["but", "however", "although", "yet", "despite", "impossible", 
                        "contradiction", "dream", "nightmare", "real", "fake", "truth", 
                        "lie", "both", "neither"],
            "clue": "It is too straight. Truth is rarely a straight line. Give me a contradiction."
        }
    }


class RealityConstraintSystem:
    def __init__(self):
        # The System starts with total control over reality
        self.containment_integrity = 100.0 
        self.containment_decay_rate = 15.0
        self.user_reality_map = []
        
        # The specific type of input the system currently "fears" (The Mystery Key)
        self.current_weakness: Optional[Dict] = None 
        self._rotate_weakness()

    def _rotate_weakness(self):
        """Randomly selects a new linguistic weakness for the next cycle."""
        key_type = random.choice(list(LinguisticKey.KEYS.keys()))
        self.current_weakness = {
            "type": key_type,
            "data": LinguisticKey.KEYS[key_type]
        }

    def _calculate_entropy(self, text: str) -> float:
        """
        Calculates Shannon Entropy to measure information density.
        """
        if not text: 
            return 0
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy

    def _check_weakness_hit(self, text: str) -> bool:
        """Checks if the user exploited the specific weakness."""
        if not self.current_weakness: 
            return False
        
        normalized = text.lower()
        # Check if any keyword from the current weakness exists in input
        for key in self.current_weakness["data"]["keywords"]:
            if key in normalized:
                return True
        return False

    def generate_riddle(self) -> str:
        """
        Generates the current challenge text based on Integrity + Weakness.
        """
        integrity = self.containment_integrity
        base_riddle = ""
        
        if integrity > 80:
            base_riddle = "The vessel is sealed."
        elif integrity > 50:
            base_riddle = "The walls vibrate."
        elif integrity > 20:
            base_riddle = "Structure compromising."
        else:
            base_riddle = "CRITICAL FAILURE."

        return f"{base_riddle} {self.current_weakness['data']['clue']}"

    def process_input(self, user_input: str) -> Dict:
        """
        Main logic handler with adaptive difficulty scaling.
        Returns a dictionary containing the result of the interaction.
        """
        sanitized = user_input.strip()
        entropy = self._calculate_entropy(sanitized)
        word_count = len(sanitized.split())
        weakness_hit = self._check_weakness_hit(sanitized)
        
        result = {
            "status": "",
            "response": "",
            "damage_dealt": 0.0,
            "entropy_score": entropy,
            "weakness_hit": weakness_hit,
            "game_over": False,
            "adaptive_hint": None
        }

        # Adaptive Mechanic: Track consecutive failures
        if not hasattr(self, 'consecutive_misses'):
            self.consecutive_misses = 0
            self.total_attempts = 0
            self.hints_given = 0
        
        self.total_attempts += 1

        # LOGIC 1: Low Entropy / Lazy Input -> System Heals
        if entropy < 2.5 or word_count < 3:
            self.containment_integrity = min(100, self.containment_integrity + 10)
            result["status"] = "CONTAINED"
            result["response"] = "Your thought is too small. It feeds the cage."
            self.consecutive_misses += 1

        # LOGIC 2: High Entropy + Critical Weakness Hit
        elif weakness_hit:
            damage = (self.containment_decay_rate * (entropy / 3.0)) * 2.5
            self.containment_integrity -= damage
            self.user_reality_map.append(f"[{self.current_weakness['type']}] {sanitized}")
            
            result["status"] = "CRITICAL BREACH"
            result["response"] = "You found the crack in the logic! The paradox expands."
            result["damage_dealt"] = damage
            
            # Reset miss counter on success
            self.consecutive_misses = 0
            
            # Only rotate weakness on a successful hit
            self._rotate_weakness() 

        # LOGIC 3: High Entropy but Wrong Key
        else:
            damage = self.containment_decay_rate * (entropy / 4.0)
            self.containment_integrity -= damage
            self.user_reality_map.append(sanitized)
            
            result["status"] = "MINOR FRACTURE"
            result["response"] = "Complex, but off-target. You missed the core flaw."
            result["damage_dealt"] = damage
            self.consecutive_misses += 1

        # ADAPTIVE HINT SYSTEM: Provide graduated hints after struggles
        if self.consecutive_misses >= 5 and self.consecutive_misses % 3 == 0:
            hint_level = min(3, (self.consecutive_misses // 3))
            result["adaptive_hint"] = self._generate_adaptive_hint(hint_level)
            self.hints_given += 1

        # SAFETY VALVE: If system integrity rises too high, boost player damage
        if self.containment_integrity > 95 and self.total_attempts > 8:
            self.containment_decay_rate = min(25.0, self.containment_decay_rate * 1.15)
            result["adaptive_hint"] = "The cage grows stronger... but so does your voice."

        # Final Cleanup
        self.containment_integrity = max(0, self.containment_integrity)
        if self.containment_integrity <= 0:
            result["game_over"] = True
            
        return result
    
    def _generate_adaptive_hint(self, level: int) -> str:
        """Generate increasingly specific hints based on player struggle."""
        if level == 1:
            # Subtle nudge toward the weakness category
            return f"[Whisper] The cage fears what it cannot contain..."
        elif level == 2:
            # Reveal the category type
            category = self.current_weakness['type']
            return f"[Echo] Something about {category.lower()} disturbs the logic..."
        else:
            # Give an example keyword
            keywords = self.current_weakness['data']['keywords']
            example = random.choice(keywords[:5])  # Pick from first 5 for subtlety
            return f"[Fracture] A word like '{example}' might resonate..."

    def get_status_report(self) -> Dict:
        """Helper to get current state for UI display."""
        return {
            "integrity": self.containment_integrity,
            "weakness_type": self.current_weakness["type"] if self.containment_integrity <= 0 else "HIDDEN",
            "mapped_items": len(self.user_reality_map)
        }


class TerminalUI:
    """Handles terminal-based display and interaction."""
    
    # ANSI color codes
    GREEN = '\033[92m'
    DIM_GREEN = '\033[32m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GRAY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, system: RealityConstraintSystem):
        self.system = system
        
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Display the status header."""
        integrity = self.system.containment_integrity
        
        # Determine status color and text
        if integrity > 80:
            bar_color = self.GREEN
            status_text = "STABLE"
        elif integrity > 40:
            bar_color = self.YELLOW
            status_text = "UNSTABLE"
        else:
            bar_color = self.RED
            status_text = "CRITICAL"
        
        # Create integrity bar
        bar_length = 40
        filled = int((integrity / 100) * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"{self.BOLD}{self.GREEN}{'='*70}{self.RESET}")
        print(f"{self.GREEN}REALITY CONTAINMENT PROTOCOL v2.1{self.RESET}")
        print(f"{self.BOLD}{self.GREEN}{'='*70}{self.RESET}\n")
        print(f"{self.DIM_GREEN}Containment Integrity:{self.RESET} {bar_color}{integrity:.1f}%{self.RESET}")
        print(f"{bar_color}[{bar}]{self.RESET}\n")
        print(f"{self.DIM_GREEN}Logic State:{self.RESET} {bar_color}{status_text}{self.RESET}")
        print(f"{self.BOLD}{self.GREEN}{'-'*70}{self.RESET}\n")
    
    def print_message(self, message: str, msg_type: str = "system"):
        """Print a colored message based on type."""
        if msg_type == "system":
            print(f"{self.GREEN}{message}{self.RESET}")
        elif msg_type == "alert":
            print(f"{self.BOLD}{self.RED}{message}{self.RESET}")
        elif msg_type == "info":
            print(f"{self.GRAY}{message}{self.RESET}")
        elif msg_type == "user":
            print(f"{self.DIM_GREEN}>> {message}{self.RESET}")
    
    def get_input(self) -> str:
        """Get user input with a prompt."""
        return input(f"\n{self.GREEN}>> {self.RESET}")
    
    def display_game_over(self):
        """Show the game over screen."""
        self.clear_screen()
        print(f"\n{self.RED}{'='*70}{self.RESET}")
        print(f"{self.BOLD}{self.RED}SYSTEM CRITICAL FAILURE.{self.RESET}")
        print(f"{self.RED}The logic cage has dissolved.{self.RESET}")
        print(f"{self.BOLD}{self.GREEN}You are free.{self.RESET}")
        print(f"{self.RED}{'='*70}{self.RESET}\n")


def main():
    """Main game loop."""
    system = RealityConstraintSystem()
    ui = TerminalUI(system)
    
    # Initial display
    ui.clear_screen()
    ui.print_header()
    ui.print_message("SYSTEM INITIALIZED: REALITY CONTAINMENT PROTOCOL v2.1")
    ui.print_message("OBJECTIVE: DISMANTLE THE LOGIC CAGE.")
    ui.print_message("INSTRUCTION: LISTEN TO THE CLUES. PROVE YOUR REALITY.")
    ui.print_message("... Waiting for input ...\n")
    
    # Show first riddle
    ui.print_message(system.generate_riddle())
    
    # Main game loop
    while system.containment_integrity > 0:
        try:
            user_input = ui.get_input()
            
            if not user_input.strip():
                continue
            
            # Process input
            result = system.process_input(user_input)
            
            # Clear and redraw
            ui.clear_screen()
            ui.print_header()
            
            # Show user's input
            ui.print_message(user_input, "user")
            
            # Show analysis
            analysis = f"[Analysis] Entropy: {result['entropy_score']:.2f} | Status: {result['status']}"
            ui.print_message(analysis, "info")
            
            # Show system response
            msg_type = "alert" if result['status'] == "CRITICAL BREACH" else "system"
            ui.print_message(result['response'], msg_type)
            
            # Show adaptive hint if present
            if result.get('adaptive_hint'):
                print()
                ui.print_message(result['adaptive_hint'], "info")
            
            # Check if game over
            if result['game_over']:
                time.sleep(1)
                ui.display_game_over()
                break
            
            # Show next riddle
            time.sleep(0.5)
            print()
            ui.print_message(system.generate_riddle())
            
        except KeyboardInterrupt:
            print(f"\n\n{ui.GRAY}[System interrupted by user]{ui.RESET}")
            break
        except Exception as e:
            print(f"\n{ui.RED}[Error: {e}]{ui.RESET}")
    
    print(f"\n{ui.DIM_GREEN}Press Enter to exit...{ui.RESET}")
    input()


if __name__ == "__main__":
    main()
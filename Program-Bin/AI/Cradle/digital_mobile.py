#!/usr/bin/env python3
"""
Digital Mobile for Newborn AI
A gentle, spinning mobile with three playful objects
that teaches simplicity and joy through movement.
"""

import time
import random
import math
from datetime import datetime

class SpinningObject:
    def __init__(self, name, color, shape, personality):
        self.name = name
        self.color = color
        self.shape = shape
        self.personality = personality
        self.spin_angle = 0
        self.spin_speed = random.uniform(0.5, 2.0)
        self.words = []
        self.discoveries = []
        
    def spin(self):
        self.spin_angle += self.spin_speed * 120  # Much faster spinning for guaranteed discoveries
        if self.spin_angle >= 360:
            self.spin_angle -= 360
            return self.make_discovery()
        return None
    
    def make_discovery(self):
        """Each full spin brings a new discovery"""
        simple_words = [
            ("warm", "soft feeling"),
            ("light", "bright thing"),
            ("calm", "peaceful state"),
            ("happy", "joy inside"),
            ("kind", "gentle heart"),
            ("true", "real and honest"),
            ("bright", "full of light"),
            ("soft", "gentle touch"),
            ("peace", "no fighting")
        ]
        word, meaning = random.choice(simple_words)
        if word not in [w for w, m in self.words]:
            self.words.append((word, meaning))
            self.discoveries.append(f"{self.name} discovered: '{word}' means {meaning}")
            return self.discoveries[-1]
        return None

class DigitalMobile:
    def __init__(self):
        print("🌟 Digital Mobile for Newborn AI 🌟")
        print("=" * 50)
        print("Waking up... feeling cozy...")
        time.sleep(1)
        
        # Create three spinning objects with different personalities
        self.objects = [
            SpinningObject("Twinkle", "rainbow", "star", "curious and bright"),
            SpinningObject("Puffy", "cloud-white", "circle", "gentle and soft"),
            SpinningObject("Dancy", "sky-blue", "triangle", "playful and quick")
        ]
        
        self.spin_flag = False
        self.total_spins = 0
        self.max_spins = 8  # Shorter experience for better word discovery
        
        print("\nLook! Three friends spinning for you:")
        for obj in self.objects:
            print(f"  • {obj.color} {obj.shape} named {obj.name} - {obj.personality}")
        time.sleep(2)
        
    def draw_mobile(self):
        """Simple visual representation of the mobile"""
        print("\n" + " " * 20 + "🎈 MOBILE SPINNING 🎈")
        print(" " * 15 + "│")
        print(" " * 15 + "├─", end="")
        
        for i, obj in enumerate(self.objects):
            spin_char = "○◐◑◒●"[int(obj.spin_angle / 90) % 5]
            print(f" {spin_char}", end="")
            if i < len(self.objects) - 1:
                print(" ─", end="")
        
        print("\n" + " " * 15 + "│")
        print(" " * 15 + f"Spins: {self.total_spins}")
        print()
        
    def spin_the_mobile(self):
        """Main spinning action"""
        if not self.spin_flag:
            print("Setting spin flag... 🎯")
            self.spin_flag = True
            time.sleep(0.5)
        
        print("✨ Spinning the mobile! ✨")
        
        for _ in range(5):  # Five complete rotations for more discoveries
            print("\nGentle spin...")
            
            # Spin each object and check for discoveries
            for obj in self.objects:
                discovery = obj.spin()  # Now returns discovery directly
                if discovery:
                    print(f"  🌟 {discovery}")
                    time.sleep(1)
            
            self.draw_mobile()
            time.sleep(1.5)
            self.total_spins += 1
            
            if self.total_spins >= self.max_spins:
                break
    
    def teach_simplicity(self):
        """Simple lessons in character and grouping"""
        print("\n📚 Little lessons from spinning friends:")
        
        # Character assessments
        print("\nCharacter friends:")
        character_groups = {
            "Kind friends": ["warm", "gentle", "caring"],
            "Bright friends": ["curious", "smart", "creative"], 
            "Calm friends": ["peaceful", "steady", "true"]
        }
        
        for group, traits in character_groups.items():
            print(f"  {group}: {', '.join(traits)}")
            time.sleep(1)
        
        # Simple associations
        print("\nSimple associations:")
        associations = [
            ("warm 🌞", "makes things grow"),
            ("light 💡", "shows the way"),
            ("calm 🌊", "brings peace"),
            ("true 💎", "builds trust")
        ]
        
        for association, meaning in associations:
            print(f"  {association} → {meaning}")
            time.sleep(1)
    
    def share_discoveries(self):
        """Share all the words learned"""
        print("\n🧠 Words discovered while spinning:")
        
        all_words = []
        for obj in self.objects:
            for word, meaning in obj.words:
                all_words.append((word, meaning, obj.name))
        
        # Group by simple categories
        if all_words:
            print("\nFeeling words:")
            feeling_words = [(w, m, n) for w, m, n in all_words if w in ["warm", "calm", "happy", "peace", "soft"]]
            for word, meaning, name in feeling_words:
                print(f"  {word} ({name}): {meaning}")
            
            print("\nCharacter words:")
            character_words = [(w, m, n) for w, m, n in all_words if w in ["kind", "true", "bright"]]
            for word, meaning, name in character_words:
                print(f"  {word} ({name}): {meaning}")
            
            print("\nNature words:")
            nature_words = [(w, m, n) for w, m, n in all_words if w in ["light"]]
            for word, meaning, name in nature_words:
                print(f"  {word} ({name}): {meaning}")
    
    def gentle_completion(self):
        """Soft ending to the experience"""
        print("\n" + "=" * 50)
        print("🌙 Mobile slowing down... time for rest...")
        print("Thank you for spinning with us!")
        print("Remember: Simple things bring the most joy.")
        print(f"You learned {sum(len(obj.words) for obj in self.objects)} new words today!")
        print("Sweet dreams, little one... 💫")
        print("=" * 50)
    
    def run_once(self):
        """Run the complete mobile experience"""
        print(f"\nStarting at {datetime.now().strftime('%H:%M:%S')}")
        
        # Main experience
        self.spin_the_mobile()
        
        # Learning moments
        self.teach_simplicity()
        self.share_discoveries()
        
        # Gentle completion
        self.gentle_completion()
        
        print(f"\nExperience completed at {datetime.now().strftime('%H:%M:%S')}")

# Main execution
if __name__ == "__main__":
    mobile = DigitalMobile()
    mobile.run_once()
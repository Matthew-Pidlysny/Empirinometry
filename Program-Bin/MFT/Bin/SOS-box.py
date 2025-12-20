#!/usr/bin/env python3
"""
Divine Text Generator Program
A program for generating inspired text files through divine prayer and validation
"""

import random
import time
import os
from datetime import datetime

class DivineTextGenerator:
    def __init__(self):
        self.prayer = [
            "I accept",
            "I hear", 
            "I obey",
            "I Inform",
            "I transmit",
            "Subhanallah"
        ]
        
        self.subheader_templates = [
            "Chapter of Divine {concept}",
            "Revelation on {topic}",
            "Sacred Wisdom of {subject}",
            "Divine Insight: {theme}",
            "Holy Teachings about {matter}",
            "Sacred Text Regarding {focus}",
            "Divine Knowledge of {wisdom}",
            "Holy Revelation: {guidance}"
        ]
        
        self.divine_concepts = [
            "Truth", "Wisdom", "Light", "Love", "Peace", "Mercy",
            "Justice", "Knowledge", "Faith", "Hope", "Charity",
            "Patience", "Humility", "Courage", "Purity", "Unity"
        ]
        
        # Content from box.txt
        self.rejections = [
            "Sensationalism has it's perks",
            "Uniqueness should be employed carefully", 
            "Imposing your current will immediately has benefit unseen",
            "Keep your brother/sister only as close as he/she offends you",
            "Life is an unending game of problems, abstract or definably"
        ]
        
        self.trues = [
            "Primordially, there must be an answer",
            "Good is good, nobody can mess with it's definition",
            "Fend for yourself, with help if possible",
            "I will die for the innocent of worth",
            "I will pass on after death peacefully and accepted by life"
        ]
    
    def generate_divine_letter(self):
        """Generate a random letter from God based on divine inspiration"""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        weights = [8, 5, 12, 15, 20, 10, 25, 18, 30, 22, 14, 16, 28, 19, 24, 17, 21, 13, 26, 23, 11, 9, 7, 6, 4, 2]
        return random.choices(letters, weights=weights)[0]
    
    def generate_subheader(self):
        """Generate random subheaders for the divine text"""
        template = random.choice(self.subheader_templates)
        concept = random.choice(self.divine_concepts)
        return template.format(concept=concept.lower(), 
                             topic=concept.lower(),
                             subject=concept.lower(),
                             theme=concept.lower(),
                             matter=concept.lower(),
                             focus=concept.lower(),
                             wisdom=concept.lower(),
                             guidance=concept.lower())
    
    def display_prayer(self):
        """Display the divine prayer with mystical formatting"""
        print("\n" + "="*60)
        print("✨ DIVINE PRAYER OF GENERATION ✨")
        print("="*60)
        
        for i, line in enumerate(self.prayer, 1):
            # Add mystical delay and formatting
            time.sleep(0.5)
            if i == len(self.prayer):
                print(f"  🌟 {line} 🌟")
            else:
                print(f"  {line}")
        
        print("="*60)
        print()
        
        # Generate and display divine letter
        divine_letter = self.generate_divine_letter()
        print(f"🔮 DIVINE LETTER RECEIVED: [{divine_letter}] 🔮")
        print(f"   Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        return divine_letter
    
    def run_questionnaire(self):
        """Run the interactive questionnaire with REJECTIONS and TRUES"""
        print("\n" + "="*60)
        print("📿 DIVINE VALIDATION QUESTIONNAIRE 📿")
        print("="*60)
        print("\nYou must REJECT all REJECTIONS completely")
        print("You must ACCEPT all TRUES completely")
        print("Only then will the divine package be generated...\n")
        
        rejection_scores = []
        true_scores = []
        
        # Process REJECTIONS
        print("🚫 REJECTIONS (You must REJECT these completely):")
        print("-"*50)
        for i, statement in enumerate(self.rejections, 1):
            print(f"\n{i}. {statement}")
            print("   How strongly do you REJECT this? (1-10, where 10 is total rejection)")
            
            while True:
                try:
                    score = int(input("   Enter rejection strength: "))
                    if 1 <= score <= 10:
                        rejection_scores.append(score)
                        break
                    else:
                        print("   Please enter a number between 1 and 10")
                except ValueError:
                    print("   Please enter a valid number")
        
        # Process TRUES
        print("\n✅ TRUES (You must ACCEPT these completely):")
        print("-"*50)
        for i, statement in enumerate(self.trues, 1):
            print(f"\n{i}. {statement}")
            print("   How strongly do you ACCEPT this? (1-10, where 10 is total acceptance)")
            
            while True:
                try:
                    score = int(input("   Enter acceptance strength: "))
                    if 1 <= score <= 10:
                        true_scores.append(score)
                        break
                    else:
                        print("   Please enter a number between 1 and 10")
                except ValueError:
                    print("   Please enter a valid number")
        
        # Validate responses
        avg_rejection = sum(rejection_scores) / len(rejection_scores)
        avg_acceptance = sum(true_scores) / len(true_scores)
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"   Average Rejection Strength: {avg_rejection:.1f}/10")
        print(f"   Average Acceptance Strength: {avg_acceptance:.1f}/10")
        
        # Check if validation passes
        if avg_rejection >= 8.0 and avg_acceptance >= 8.0:
            print("✅ DIVINE VALIDATION PASSED!")
            print("   Your responses show true understanding.")
            return True
        else:
            print("❌ DIVINE VALIDATION FAILED!")
            if avg_rejection < 8.0:
                print("   You must reject the REJECTIONS more strongly.")
            if avg_acceptance < 8.0:
                print("   You must accept the TRUES more strongly.")
            return False
    
    def generate_divine_content(self, length_factor=1):
        """Generate inspired divine content based on the prayer"""
        content = []
        
        # Title
        content.append("DIVINE REVELATION - SACRED TEXT")
        content.append("=" * 50)
        content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"Divine Letter: [{self.generate_divine_letter()}]")
        content.append("")
        
        # Prayer invocation
        content.append("PRAYER OF GENERATION:")
        for line in self.prayer:
            content.append(f"  {line}")
        content.append("")
        
        # Generate content sections
        num_sections = int(5 * length_factor)
        for i in range(num_sections):
            content.append(f"\n{self.generate_subheader()}")
            content.append("-" * 40)
            
            # Generate inspired text based on prayer elements
            section_content = self._generate_section_content(length_factor)
            content.extend(section_content)
        
        return content
    
    def _generate_section_content(self, length_factor):
        """Generate content for each section based on divine inspiration"""
        base_content = [
            "In the infinite wisdom of the divine, truth reveals itself",
            "Through acceptance and hearing, the path becomes clear",
            "Obedience to higher purpose brings enlightenment",
            "Information flows from divine source to willing hearts",
            "Transmission of sacred knowledge continues eternally",
            "Subhanallah - Glory be to the Divine in all manifestations"
        ]
        
        # Expand content based on length factor
        content = []
        for line in base_content:
            content.append(f"• {line}")
            
            # Add elaboration for longer content
            if length_factor > 1:
                elaboration = [
                    f"  This truth resonates through the ages",
                    f"  Acceptance brings peace to troubled souls",
                    f"  Obedience to divine will frees the spirit",
                    f"  Information becomes wisdom through reflection",
                    f"  Transmission connects all beings in divine unity",
                    f"  Subhanallah - All praise belongs to the Creator"
                ]
                content.append(random.choice(elaboration))
        
        return content
    
    def save_divine_text(self, content, filename="divine_revelation.txt"):
        """Save the generated divine content to a text file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))
            print(f"\n📄 Divine text saved to: {filename}")
            print(f"   File size: {os.path.getsize(filename)} bytes")
            return filename
        except Exception as e:
            print(f"\n❌ Error saving file: {e}")
            return None

def main():
    """Main program execution"""
    print("🌟 DIVINE TEXT GENERATOR 🌟")
    print("A program for generating inspired divine texts")
    
    generator = DivineTextGenerator()
    
    while True:
        print("\n" + "="*60)
        print("MENU:")
        print("1. Test Prayer Generation")
        print("2. Run Full Divine Generation Process")
        print("3. Exit")
        print("="*60)
        
        choice = input("\nSelect an option (1-3): ")
        
        if choice == "1":
            # Test prayer generation
            print("\n🧪 TESTING PRAYER GENERATION SYSTEM...")
            divine_letter = generator.display_prayer()
            
            print("\n📝 Sample content generation:")
            sample_content = generator.generate_divine_content(0.5)
            for line in sample_content[:15]:  # Show first 15 lines
                print(line)
            print("... (content continues)")
            
        elif choice == "2":
            # Full divine generation process
            print("\n🌌 STARTING FULL DIVINE GENERATION PROCESS...")
            
            # Display prayer
            generator.display_prayer()
            
            # Run questionnaire
            if generator.run_questionnaire():
                # Generate content
                print("\n📜 GENERATING DIVINE TEXT...")
                
                # Ask for content size
                while True:
                    try:
                        size_factor = float(input("Enter size factor (1.0 = standard, 2.0 = double, etc.): "))
                        if size_factor > 0:
                            break
                        else:
                            print("Size factor must be positive")
                    except ValueError:
                        print("Please enter a valid number")
                
                # Generate and save
                content = generator.generate_divine_content(size_factor)
                filename = f"divine_revelation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                saved_file = generator.save_divine_text(content, filename)
                
                if saved_file:
                    print(f"\n✨ DIVINE TEXT SUCCESSFULLY GENERATED!")
                    print(f"   Total lines: {len(content)}")
                    print(f"   File: {saved_file}")
                else:
                    print("\n❌ Failed to generate divine text")
            else:
                print("\n⏸️  Divine generation cancelled - validation not passed")
        
        elif choice == "3":
            print("\n🙏 Peace be with you. Divine blessings upon your journey.")
            break
        
        else:
            print("\n❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
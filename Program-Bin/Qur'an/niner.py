#!/usr/bin/env python3
"""
NINER MERGED - Complete Qur'anic 9-Geometry Analysis & Educational System
Version 4.0 - Merged with Enhanced Storybook Functionality

This program combines the best features from niner.py v2.0 and niner1.py v3.0:
- Sacred geometry analysis with 3D sphere generation
- Enhanced educational content with Islamic values
- Comprehensive Storybook creation system
- Nine-geometry pattern detection
- Orbital analysis with tajweed terminology
- Interactive learning environment
- Conclusion generation based on user input

IMPORTANT SHIRK POLICY NOTICE:
This program analyzes mathematical patterns in the Holy Qur'an for educational purposes only.
We do not claim these patterns have divine significance or enforce any religious interpretation.
The analysis is purely computational and should not be used for shirk (associating partners with Allah).
Users should approach this content with proper Islamic understanding and scholarly guidance.

CORE MERGED FEATURES:
- Digital root preservation (9 = completion)
- 3² foundation analysis (nine as 3 squared)
- Qur'anic nine-based pattern detection
- Enhanced Storybook creation with interactive features
- Educational orbital analysis with proper tajweed terminology
- Sacred geometry with 3D sphere generation
- Child-friendly storytelling with Islamic values
- Conclusion generation based on user input
"""

import math
import json
import datetime
import random
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# Import sphere generation from balls
try:
    from balls import BallsGenerator
    BALLS_AVAILABLE = True
except ImportError:
    BALLS_AVAILABLE = False
    print("⚠️  balls.py not available - sphere generation disabled")

class EnhancedStorybookSystem:
    """Complete storybook creation system with interactive features"""
    
    def __init__(self):
        self.story_templates = {
            'discovery': [
                "In the magical garden of numbers, where truth blooms like flowers...",
                "Deep in the library of creation, where every digit tells a story...",
                "On the sacred mountain of mathematics, where patterns dance like stars...",
                "Within the crystal palace of Qur'anic wisdom, where numbers sing harmonies...",
                "In the beautiful garden of learning, where knowledge grows like flowers...",
                "Deep in the library of wisdom, where every discovery tells a story..."
            ],
            'miracle': [
                "And there, my dear child, the numbers revealed a secret that made the angels smile...",
                "The universe held its breath as the mathematics unfolded its perfect design...",
                "Like finding a pearl in the vast ocean of wisdom, this discovery shone with divine light...",
                "The pattern emerged like sunrise over the holy lands, beautiful and perfect...",
                "Like threads in a divine tapestry, the numbers weave together in perfect harmony...",
                "As stars align in the night sky, so do these mathematical patterns reveal their beauty..."
            ],
            'lesson': [
                "This teaches us that Allah's creation is built on perfect mathematics...",
                "From this we learn that every letter in the Qur'an has its special place...",
                "The numbers show us that there is no randomness in divine revelation...",
                "Mathematics becomes a bridge between our hearts and the divine wisdom...",
                "This teaches us about the perfect order in Allah's creation.",
                "We see how everything in the universe follows beautiful patterns designed by Allah."
            ],
            'orbital': [
                "In the celestial spheres of recitation, the letters orbit in perfect circles...",
                "Like planets around the sun, the sounds of Qur'an follow divine orbits...",
                "Within the cosmic dance of tajweed, each letter finds its perfect place...",
                "As celestial bodies follow their paths, so do the letters trace their sacred orbits..."
            ]
        }
        
        self.islamic_lessons = [
            "This teaches us about the perfect order in Allah's creation.",
            "We see how everything in the universe follows beautiful patterns designed by Allah.",
            "This reminds us of the precision and wisdom in Allah's words.",
            "Like the perfect timing of prayer, these patterns show divine order.",
            "Just as the moon follows its phases, creation follows perfect laws."
        ]
        
        self.storybook_library = []
        self.interactive_elements = {
            'quizzes': [],
            'activities': [],
            'reflections': []
        }
    
    def create_shirk_notice(self):
        """Important notice about shirk policy"""
        return """
        📚 IMPORTANT ISLAMIC GUIDANCE 📚
        
        Dear Parents and Children,
        
        This program explores mathematical patterns for educational purposes only.
        We do not claim these patterns have special powers or divine meaning beyond
        what Allah has revealed in the Qur'an and Sunnah.
        
        • Allah is the Creator of all patterns and mathematics
        • These discoveries are for learning and appreciation of Allah's creation
        • Never use number patterns to predict the future or make religious decisions
        • Always follow the Qur'an and Sunnah as your primary guidance
        • Consult qualified scholars for religious matters
        
        Remember: The greatest miracle is the Qur'an itself, and the best knowledge
        is the knowledge that brings us closer to Allah.
        """
    
    def create_interactive_storybook_entry(self, discovery_data: Dict, audience: str = "children") -> Dict:
        """Create complete interactive storybook entry"""
        
        storybook_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'title': discovery_data.get('title', 'Mathematical Discovery'),
            'story': self.create_story(discovery_data, audience),
            'interactive_elements': self.create_interactive_elements(discovery_data),
            'educational_notes': self.create_educational_notes(discovery_data),
            'reflection_questions': self.create_reflection_questions(discovery_data)
        }
        
        self.storybook_library.append(storybook_entry)
        return storybook_entry
    
    def create_story(self, discovery_data: Dict, audience: str) -> str:
        """Create the main story"""
        
        story = []
        story.append("🌟 ✨ 🌟 ✨ 🌟")
        story.append("A NINE-GEOMETRY STORY FROM THE GARDEN OF NUMBERS")
        story.append("🌟 ✨ 🌟 ✨ 🌟\n")
        
        # Opening
        opening = random.choice(self.story_templates['discovery'])
        story.append(opening)
        story.append("")
        
        # The discovery
        if discovery_data.get('type') == 'nine_geometry':
            story.append(f"Today, we discovered something amazing about the number {discovery_data.get('main_number', '9')}:")
            
            if 'pattern' in discovery_data:
                story.append(f"🔍 The Pattern: {discovery_data['pattern']}")
            
            if 'location' in discovery_data:
                story.append(f"📍 Where We Found It: {discovery_data['location']}")
            
            story.append("")
            
            # The miracle
            miracle = random.choice(self.story_templates['miracle'])
            story.append(miracle)
            story.append("")
            
            # Mathematical details
            story.append("Here's what makes this so special:")
            story.append(f"💫 Found {discovery_data.get('nine_count', 'many')} patterns with the number 9!")
            story.append(f"💫 Nine is special because it's 3×3 (3 squared) - the foundation of creation!")
            story.append(f"💫 Like how a strong building needs a solid foundation, creation rests on 3²!")
            
        elif discovery_data.get('type') == 'orbital':
            story.append(f"🌌 We discovered celestial patterns in Qur'anic recitation:")
            
            if 'orbital_type' in discovery_data:
                story.append(f"🎵 Tajweed Pattern: {discovery_data['orbital_type']}")
            
            if 'chapter' in discovery_data and 'verse' in discovery_data:
                story.append(f"📖 Location: Chapter {discovery_data['chapter']}, Verse {discovery_data['verse']}")
            
            story.append("")
            orbital_story = random.choice(self.story_templates['orbital'])
            story.append(orbital_story)
            
        story.append("")
        
        # The lesson
        lesson = random.choice(self.story_templates['lesson'])
        story.append(lesson)
        story.append("")
        
        # Closing
        story.append("💝 A Thought to Carry in Your Heart:")
        if discovery_data.get('type') == 'nine_geometry':
            story.append("The number 9 reminds us that Allah's creation is complete and perfect.")
            story.append("When we study these patterns, we're reading Allah's mathematical signature.")
        else:
            story.append("The beautiful patterns in Qur'anic recitation show us the harmony in Allah's words.")
            story.append("When we recite with proper tajweed, we join this celestial melody.")
        
        story.append("")
        story.append("🙏 May Allah guide us to understand more of His beautiful wisdom.")
        story.append("")
        story.append("---")
        story.append(f"Written on {datetime.datetime.now().strftime('%B %d, %Y')}")
        story.append("From the Nine-Geometry Mathematics Garden")
        
        return "\n".join(story)
    
    def create_interactive_elements(self, discovery_data: Dict) -> Dict:
        """Create interactive elements for the storybook"""
        
        elements = {
            'quiz': self.create_quiz(discovery_data),
            'activity': self.create_activity(discovery_data),
            'visualization': self.create_visualization_suggestion(discovery_data)
        }
        
        return elements
    
    def create_quiz(self, discovery_data: Dict) -> Dict:
        """Create educational quiz"""
        
        if discovery_data.get('type') == 'nine_geometry':
            questions = [
                {
                    'question': 'What is 9 as a mathematical expression?',
                    'options': ['3×3', '3+3', '9×1', '3³'],
                    'correct': 0,
                    'explanation': '9 = 3×3 = 3², showing the foundation of three squared!'
                },
                {
                    'question': 'Why is the number 9 special in creation?',
                    'options': [
                        'It\'s the largest single digit',
                        'It represents completion',
                        'It appears in pregnancy',
                        'All of the above'
                    ],
                    'correct': 3,
                    'explanation': 'Nine is special in many ways - it completes the single digits, represents perfection, and 9 months of human gestation!'
                }
            ]
        else:
            questions = [
                {
                    'question': 'What is tajweed?',
                    'options': [
                        'Arabic grammar',
                        'Rules of Qur\'anic recitation',
                        'Islamic history',
                        'Arabic poetry'
                    ],
                    'correct': 1,
                    'explanation': 'Tajweed refers to the rules governing pronunciation during recitation of the Qur\'an.'
                }
            ]
        
        return {
            'title': 'Test Your Knowledge!',
            'questions': questions
        }
    
    def create_activity(self, discovery_data: Dict) -> Dict:
        """Create hands-on activity"""
        
        if discovery_data.get('type') == 'nine_geometry':
            return {
                'title': 'Explore the Number 9!',
                'materials': ['Paper', 'Pencil', 'Calculator'],
                'steps': [
                    'Write down numbers from 1 to 20',
                    'Circle all numbers divisible by 9',
                    'Calculate digital roots (sum digits until single digit)',
                    'Look for the number 9 in your daily life',
                    'Create a drawing showing 9 as 3×3'
                ],
                'reflection': 'What patterns did you discover about the number 9?'
            }
        else:
            return {
                'title': 'Listen to Qur\'anic Recitation',
                'materials': ['Qur\'an app or website', 'Headphones'],
                'steps': [
                    'Choose a short surah to listen to',
                    'Pay attention to the rhythm and melody',
                    'Notice how letters flow together',
                    'Try to identify different tajweed patterns',
                    'Reflect on the beauty of the recitation'
                ],
                'reflection': 'How did the recitation make you feel? What patterns did you notice?'
            }
    
    def create_visualization_suggestion(self, discovery_data: Dict) -> Dict:
        """Create visualization suggestion"""
        
        if discovery_data.get('type') == 'nine_geometry':
            return {
                'title': 'Create a 9-Geometry Art',
                'description': 'Draw or create digital art showing the beauty of the number 9',
                'ideas': [
                    'Draw 9 circles in a pattern',
                    'Create a 3×3 grid with meaningful symbols',
                    'Make a collage of 9 things you\'re grateful for',
                    'Design a mandala with 9 sections'
                ]
            }
        else:
            return {
                'title': 'Visualize Tajweed Orbits',
                'description': 'Create art showing the flow of Qur\'anic recitation',
                'ideas': [
                    'Draw flowing lines for sounds',
                    'Create orbital patterns for letters',
                    'Design a visual representation of rhythm',
                    'Make a mind map of tajweed rules'
                ]
            }
    
    def create_educational_notes(self, discovery_data: Dict) -> Dict:
        """Create educational notes for parents/teachers"""
        
        notes = {
            'learning_objectives': [
                'Understand mathematical patterns in creation',
                'Appreciate the precision in Allah\'s design',
                'Develop analytical thinking skills',
                'Connect mathematics with Islamic education'
            ],
            'islamic_integration': [
                'Emphasize that Allah is the Creator of all patterns',
                'Use discoveries to appreciate Allah\'s wisdom',
                'Connect mathematical beauty with Qur\'anic recitation',
                'Maintain proper Islamic perspective on patterns'
            ],
            'discussion_points': [
                'How do patterns help us understand Allah\'s creation?',
                'What can we learn from studying mathematical relationships?',
                'How does this relate to our daily lives as Muslims?',
                'Why is it important to maintain proper Islamic understanding?'
            ]
        }
        
        if discovery_data.get('type') == 'nine_geometry':
            notes['curriculum_links'] = [
                'Mathematics: Number theory, digital roots',
                'Islamic Studies: Qur\'anic structure, mathematical miracles',
                'Science: Patterns in nature, geometry',
                'Art: Sacred geometry, Islamic art patterns'
            ]
        else:
            notes['curriculum_links'] = [
                'Language Arts: Poetry, rhythm, phonetics',
                'Islamic Studies: Tajweed, Qur\'anic recitation',
                'Music: Rhythm, melody, patterns',
                'Physics: Sound waves, acoustics'
            ]
        
        return notes
    
    def create_reflection_questions(self, discovery_data: Dict) -> List[str]:
        """Create reflection questions"""
        
        base_questions = [
            "What amazed you most about this discovery?",
            "How does this help you appreciate Allah's creation?",
            "What patterns do you notice in your own life?",
            "How can we use this knowledge to become better Muslims?"
        ]
        
        if discovery_data.get('type') == 'nine_geometry':
            specific_questions = [
                "Where else do you see the number 9 in the world?",
                "How does understanding 3×3 help us see Allah's wisdom?",
                "What other mathematical patterns might exist in creation?"
            ]
        else:
            specific_questions = [
                "How does proper recitation enhance our understanding?",
                "What other patterns exist in Qur'anic structure?",
                "How can we improve our own recitation?"
            ]
        
        return base_questions + specific_questions
    
    def generate_conclusion_story(self, user_input: str, analysis_results: Dict) -> str:
        """Generate conclusion based on user input and analysis"""
        story = f"""
        🎓 Our Educational Journey Together 🎓
        
        Dear Learning Friend,
        
        Based on what you've shared: "{user_input}"
        
        And our analysis of {analysis_results.get('total_verses', 0)} verses,
        we've discovered {analysis_results.get('total_patterns', 0)} mathematical patterns
        that remind us of Allah's perfect creation.
        
        We found {analysis_results.get('nine_patterns', 0)} instances of the number 9,
        teaching us about completion and harmony in the Qur'an.
        
        We explored {analysis_results.get('orbital_patterns', 0)} tajweed orbital patterns,
        showing how recitation follows celestial rhythms.
        
        {random.choice(self.islamic_lessons)}
        
        ⚠️ IMPORTANT LIMITATION NOTICE ⚠️
        
        We apologize for the limitations of our computational approach. This program
        analyzes patterns mathematically but cannot capture the infinite wisdom of
        the Qur'an. The true meaning comes from:
        
        • Proper Islamic education with qualified teachers
        • Understanding Arabic language and context
        • Studying tafsir (exegesis) from reputable scholars
        • Living the teachings of the Qur'an in daily life
        
        Remember: Numbers and patterns are just tools to appreciate Allah's creation,
        not sources of religious guidance. The Qur'an's greatest miracles are
        its guidance, its linguistic perfection, and its power to change hearts.
        
        🙏 May Allah increase our knowledge and guide us to the truth. 🙏
        """
        
        return story.strip()
    
    def save_storybook(self, filename: str = None) -> str:
        """Save complete storybook to file"""
        if filename is None:
            filename = f"niner_enhanced_storybook_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        content = []
        content.append("📚 NINER ENHANCED STORYBOOK COLLECTION 📚")
        content.append(f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y')}")
        content.append(f"Total Stories: {len(self.storybook_library)}")
        content.append("\n" + "="*80 + "\n")
        
        content.append(self.create_shirk_notice())
        content.append("\n" + "="*80 + "\n")
        
        for i, entry in enumerate(self.storybook_library, 1):
            content.append(f"🌟 STORY {i}: {entry['title']} 🌟")
            content.append(f"Created: {entry['timestamp']}")
            content.append("\n")
            content.append(entry['story'])
            content.append("\n" + "="*80 + "\n")
            
            # Interactive elements
            content.append("🎮 INTERACTIVE ELEMENTS 🎮")
            content.append(f"\nQuiz: {entry['interactive_elements']['quiz']['title']}")
            for q in entry['interactive_elements']['quiz']['questions']:
                content.append(f"Q: {q['question']}")
                for j, opt in enumerate(q['options']):
                    content.append(f"  {chr(65+j)}. {opt}")
                content.append(f"A: {chr(65+q['correct'])} - {q['explanation']}")
            
            content.append(f"\nActivity: {entry['interactive_elements']['activity']['title']}")
            for step in entry['interactive_elements']['activity']['steps']:
                content.append(f"  • {step}")
            content.append(f"Reflection: {entry['interactive_elements']['activity']['reflection']}")
            
            content.append(f"\nVisualization: {entry['interactive_elements']['visualization']['title']}")
            content.append(f"Description: {entry['interactive_elements']['visualization']['description']}")
            for idea in entry['interactive_elements']['visualization']['ideas']:
                content.append(f"  • {idea}")
            
            content.append("\n💭 REFLECTION QUESTIONS 💭")
            for question in entry['reflection_questions']:
                content.append(f"  • {question}")
            
            content.append("\n" + "="*80 + "\n")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        return filename

class QuranicLetterGeometry:
    """Analyzes geometric properties of Qur'anic letters"""
    
    def __init__(self):
        # Basic geometric properties of Arabic letters (simplified)
        self.letter_geometry = {
            'أ': {'points': 3, 'lines': 2, 'curves': 1, 'complexity': 6},
            'ب': {'points': 4, 'lines': 3, 'curves': 2, 'complexity': 9},
            'ت': {'points': 5, 'lines': 4, 'curves': 2, 'complexity': 11},
            'ث': {'points': 6, 'lines': 4, 'curves': 3, 'complexity': 13},
            'ج': {'points': 2, 'lines': 1, 'curves': 3, 'complexity': 6},
            'ح': {'points': 1, 'lines': 0, 'curves': 4, 'complexity': 5},
            'خ': {'points': 2, 'lines': 1, 'curves': 4, 'complexity': 7},
            'د': {'points': 2, 'lines': 1, 'curves': 2, 'complexity': 5},
            'ذ': {'points': 3, 'lines': 1, 'curves': 3, 'complexity': 7},
            'ر': {'points': 2, 'lines': 1, 'curves': 2, 'complexity': 5},
            'ز': {'points': 3, 'lines': 1, 'curves': 3, 'complexity': 7},
            'س': {'points': 2, 'lines': 0, 'curves': 4, 'complexity': 6},
            'ش': {'points': 3, 'lines': 0, 'curves': 5, 'complexity': 8},
            'ص': {'points': 2, 'lines': 0, 'curves': 5, 'complexity': 7},
            'ض': {'points': 3, 'lines': 1, 'curves': 5, 'complexity': 9},
            'ط': {'points': 2, 'lines': 1, 'curves': 3, 'complexity': 6},
            'ظ': {'points': 3, 'lines': 1, 'curves': 4, 'complexity': 8},
            'ع': {'points': 2, 'lines': 0, 'curves': 4, 'complexity': 6},
            'غ': {'points': 3, 'lines': 0, 'curves': 5, 'complexity': 8},
            'ف': {'points': 3, 'lines': 2, 'curves': 2, 'complexity': 7},
            'ق': {'points': 3, 'lines': 2, 'curves': 2, 'complexity': 7},
            'ك': {'points': 3, 'lines': 2, 'curves': 2, 'complexity': 7},
            'ل': {'points': 2, 'lines': 2, 'curves': 1, 'complexity': 5},
            'م': {'points': 2, 'lines': 0, 'curves': 4, 'complexity': 6},
            'ن': {'points': 2, 'lines': 1, 'curves': 3, 'complexity': 6},
            'ه': {'points': 2, 'lines': 1, 'curves': 2, 'complexity': 5},
            'و': {'points': 2, 'lines': 0, 'curves': 2, 'complexity': 4},
            'ي': {'points': 2, 'lines': 2, 'curves': 1, 'complexity': 5},
            'لا': {'points': 4, 'lines': 4, 'curves': 2, 'complexity': 10}
        }
        
        # Abjad values
        self.abjad_values = {
            'أ': 1, 'ب': 2, 'ت': 400, 'ث': 500, 'ج': 3, 'ح': 8, 'خ': 600,
            'د': 4, 'ذ': 700, 'ر': 200, 'ز': 7, 'س': 60, 'ش': 300, 'ص': 90,
            'ض': 800, 'ط': 9, 'ظ': 900, 'ع': 70, 'غ': 1000, 'ف': 80, 'ق': 100,
            'ك': 20, 'ل': 30, 'م': 40, 'ن': 50, 'ه': 5, 'و': 6, 'ي': 10
        }
    
    def analyze_text_geometry(self, text: str) -> Dict:
        """Analyze geometric properties of Arabic text"""
        
        total_points = 0
        total_lines = 0
        total_curves = 0
        total_complexity = 0
        total_abjad = 0
        
        letter_counts = Counter()
        
        for char in text:
            if char in self.letter_geometry:
                geom = self.letter_geometry[char]
                total_points += geom['points']
                total_lines += geom['lines']
                total_curves += geom['curves']
                total_complexity += geom['complexity']
                letter_counts[char] += 1
                
                if char in self.abjad_values:
                    total_abjad += self.abjad_values[char]
        
        return {
            'total_points': total_points,
            'total_lines': total_lines,
            'total_curves': total_curves,
            'total_complexity': total_complexity,
            'total_abjad': total_abjad,
            'letter_counts': dict(letter_counts),
            'unique_letters': len(letter_counts),
            'total_letters': sum(letter_counts.values())
        }
    
    def find_nine_patterns(self, geometry_data: Dict) -> List[Dict]:
        """Find patterns related to the number 9 in geometric data"""
        
        patterns = []
        
        # Check for divisibility by 9
        for key, value in geometry_data.items():
            if isinstance(value, int) and value > 0:
                if value % 9 == 0:
                    patterns.append({
                        'type': 'divisible_by_9',
                        'property': key,
                        'value': value,
                        'factor': value // 9,
                        'significance': f"{key} = {value // 9} × 9"
                    })
                elif value % 3 == 0:  # 3² = 9
                    patterns.append({
                        'type': 'divisible_by_3',
                        'property': key,
                        'value': value,
                        'factor': value // 3,
                        'significance': f"{key} = {value // 3} × 3 (3² foundation)"
                    })
        
        # Check for digital root = 9
        for key, value in geometry_data.items():
            if isinstance(value, int) and value > 0:
                digital_root = self.digital_root(value)
                if digital_root == 9:
                    patterns.append({
                        'type': 'digital_root_9',
                        'property': key,
                        'value': value,
                        'digital_root': 9,
                        'significance': f"{key} has digital root 9 (completion/perfection)"
                    })
        
        return patterns
    
    def digital_root(self, n: int) -> int:
        """Calculate digital root of a number"""
        while n >= 10:
            n = sum(int(digit) for digit in str(n))
        return n

class NineGeometryMonitor:
    """Monitors 9-geometry patterns in Qur'anic text"""
    
    def __init__(self):
        self.letter_geometry = QuranicLetterGeometry()
        self.balls_generator = BallsGenerator() if BALLS_AVAILABLE else None
        self.sphere_type = 'relational'  # Use the most advanced sphere type
        
    def create_quranic_sphere(self, text: str, output_file: str = None) -> Dict:
        """Create a custom sphere representation of Qur'anic text"""
        
        if not BALLS_AVAILABLE:
            return {'sphere_file': None, 'error': 'balls.py not available'}
        
        # Analyze text geometry
        geometry = self.letter_geometry.analyze_text_geometry(text)
        
        # Convert text to digit sequence for sphere generation
        # Use Abjad values as digits
        digit_sequence = []
        for char in text:
            if char in self.letter_geometry.abjad_values:
                value = self.letter_geometry.abjad_values[char]
                # Convert to individual digits
                digits = [int(d) for d in str(value)]
                digit_sequence.extend(digits)
        
        # Generate sphere using balls.py
        if output_file is None:
            output_file = f"niner_sphere_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Set sphere type to relational for best results
        self.balls_generator.sphere_type = self.sphere_type
        
        # Create a string from digit sequence
        number_str = ''.join(map(str, digit_sequence))
        
        # Generate sphere
        print(f"🌐 Generating Niner sacred sphere for Qur'anic text...")
        try:
            sphere_file = self.balls_generator.analyze_and_save(
                number_str=number_str,
                display_name="Niner Qur'anic Sacred Sphere",
                filename=output_file,
                radius=1.0,
                num_digits=min(len(digit_sequence), 10000),  # Limit for performance
                sphere_type=self.sphere_type
            )
        except Exception as e:
            print(f"Sphere generation failed: {e}")
            # Create a simple fallback sphere file
            with open(output_file, 'w') as f:
                f.write(f"NINER QUR'ANIC SACRED SPHERE ANALYSIS\n")
                f.write(f"="*50 + "\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Digit Sequence Length: {len(digit_sequence)}\n")
                f.write(f"Number String: {number_str[:100]}...\n")
                f.write(f"\nGeometry Analysis:\n")
                f.write(f"Total Points: {geometry['total_points']}\n")
                f.write(f"Total Lines: {geometry['total_lines']}\n")
                f.write(f"Total Curves: {geometry['total_curves']}\n")
                f.write(f"Total Complexity: {geometry['total_complexity']}\n")
                f.write(f"Total Abjad: {geometry['total_abjad']}\n")
            sphere_file = output_file
        
        # Find 9-geometry patterns
        nine_patterns = self.letter_geometry.find_nine_patterns(geometry)
        
        return {
            'sphere_file': sphere_file,
            'geometry_analysis': geometry,
            'nine_patterns': nine_patterns,
            'digit_sequence_length': len(digit_sequence),
            'unique_digits': len(set(digit_sequence))
        }
    
    def monitor_nine_geometry(self, text: str) -> Dict:
        """Comprehensive 9-geometry analysis of text"""
        
        results = {
            'text_length': len(text),
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'nine_analysis': {}
        }
        
        # Basic geometry
        geometry = self.letter_geometry.analyze_text_geometry(text)
        nine_patterns = self.letter_geometry.find_nine_patterns(geometry)
        
        results['geometry'] = geometry
        results['nine_patterns'] = nine_patterns
        
        # 9-based statistics
        results['nine_analysis']['divisible_by_nine'] = [p for p in nine_patterns if p['type'] == 'divisible_by_9']
        results['nine_analysis']['divisible_by_three'] = [p for p in nine_patterns if p['type'] == 'divisible_by_3']
        results['nine_analysis']['digital_root_nine'] = [p for p in nine_patterns if p['type'] == 'digital_root_9']
        
        return results

class OrbitalAnalyzer:
    """Enhanced orbital analysis with tajweed terminology"""
    
    def __init__(self):
        self.tajweed_orbitals = [
            'ghunnah', 'idgham', 'iqama', 'madd', 'qalqalah', 'ikhfa'
        ]
        
        self.tajweed_terms = {
            'ghunnah': 'nasal sound resonance',
            'idgham': 'merging of sounds',
            'iqama': 'elongation', 
            'madd': 'vowel extension',
            'qalqalah': 'bouncing echo',
            'ikhfa': 'hiding the sound'
        }
    
    def analyze_orbital_patterns(self, text: str) -> Dict:
        """Analyze tajweed orbital patterns"""
        orbital_data = {
            'ghunnah_patterns': text.count('ن') + text.count('م') + text.count('نّ') + text.count('مّ'),
            'idgham_patterns': text.count('ل') + text.count('ر') + text.count('ن') + text.count('م'),
            'madd_patterns': text.count('ا') + text.count('و') + text.count('ي'),
            'qalqalah_patterns': text.count('ق') + text.count('ط') + text.count('ب') + text.count('ج') + text.count('د'),
            'total_orbital_score': 0
        }
        
        # Calculate orbital score
        orbital_data['total_orbital_score'] = sum([
            orbital_data['ghunnah_patterns'],
            orbital_data['idgham_patterns'], 
            orbital_data['madd_patterns'],
            orbital_data['qalqalah_patterns']
        ])
        
        return orbital_data

class QuranicTextDatabase:
    """Enhanced database of Qur'anic texts for analysis"""
    
    def __init__(self, data_file: str = "quran_sequential.txt"):
        self.data_file = data_file
        self.texts = self.load_texts()
    
    def load_texts(self) -> List[Dict]:
        """Load Qur'anic texts from sequential file"""
        texts = []
        
        if not os.path.exists(self.data_file):
            print(f"⚠️  Text database file {self.data_file} not found")
            return texts
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                content = f.read().split('\n')
            
            current_surah = 1
            verse_number = 1
            
            for line in content:
                line = line.strip()
                if line.startswith('[[VERSE_'):
                    # Extract verse text
                    verse_marker = line.split(']')[0] + ']'
                    verse_start = line.find(']]') + 2
                    if verse_start > 2:
                        verse_text = line[verse_start:]
                        
                        # Clean the text (remove diacritics for basic analysis)
                        clean_text = ''.join(c for c in verse_text if ord(c) >= 0x0600 and ord(c) <= 0x06FF)
                        
                        if clean_text:
                            texts.append({
                                'text': clean_text,
                                'title': f"Surah {current_surah}:{verse_number}",
                                'category': 'Quranic Verse',
                                'chapter': current_surah,
                                'verse': verse_number,
                                'marker': verse_marker
                            })
                            verse_number += 1
                elif line.startswith('[[SURAH_') or 'Bismillah' in line:
                    # Reset verse counter for new surah
                    if 'Bismillah' not in line:
                        try:
                            current_surah = int(line.split('_')[1].split(']')[0])
                        except:
                            current_surah += 1
                    verse_number = 1
            
        except Exception as e:
            print(f"Error loading text database: {e}")
        
        return texts
    
    def get_text_by_title(self, title: str) -> Optional[Dict]:
        """Get text by title"""
        for text_data in self.texts:
            if text_data['title'].lower() == title.lower():
                return text_data
        return None
    
    def get_texts_by_category(self, category: str) -> List[Dict]:
        """Get texts by category"""
        return [t for t in self.texts if t['category'].lower() == category.lower()]
    
    def get_random_texts(self, count: int = 5) -> List[Dict]:
        """Get random texts"""
        import random
        return random.sample(self.texts, min(count, len(self.texts)))

class NinerMergedSystem:
    """Complete merged Niner system with enhanced Storybook functionality"""
    
    def __init__(self):
        self.storybook_system = EnhancedStorybookSystem()
        self.nine_monitor = NineGeometryMonitor()
        self.orbital_analyzer = OrbitalAnalyzer()
        self.text_database = QuranicTextDatabase()
        self.experience_log = []
        
        # Original Niner properties (backward compatibility)
        self.sacred_nines = {
            9: 'Base sacred nine',
            18: '2 × 9 (Double blessing)',
            27: '3³ (3 cubed)',
            36: '4 × 9 (Complete square)',
            45: '5 × 9 (Human completion)',
            54: '6 × 9 (Divine completeness)',
            63: '7 × 9 (Spiritual perfection)',
            72: '8 × 9 (Infinite cycle)',
            81: '9² (Nine squared)',
            90: '10 × 9 (Decimal perfection)',
            99: '11 × 9 (Double completion)',
            108: '12 × 9 (Cosmic order)',
            117: '13 × 9 (Transformation)',
            126: '14 × 9 (Balance)',
            135: '15 × 9 (Harmony)',
            144: '16 × 9 (Divine square)',
            153: '17 × 9 (Spiritual wisdom)',
            162: '18 × 9 (Double nine)',
            171: '19 × 9 (Foundation × 9)',
            180: '20 × 9 (Cycle completion)'
        }
        
        self.quran_structure = {
            'total_surahs': 114,
            'madani_surahs': 28,
            'makki_surahs': 86,
            'total_verses': 6236,
            'bismillah_count': 114
        }
    
    def analyze_complete_text(self, text: str, title: str = "Custom Analysis", 
                            generate_sphere: bool = True, create_storybook: bool = True) -> Dict:
        """Complete analysis with all features"""
        
        print(f"🎯 Niner Merged Analysis: {title}")
        print(f"📝 Text: {text}")
        
        results = {
            'title': title,
            'text': text,
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'sphere_generated': False,
            'storybook_generated': False
        }
        
        # 1. Nine-geometry analysis
        nine_analysis = self.nine_monitor.monitor_nine_geometry(text)
        results['nine_analysis'] = nine_analysis
        
        # 2. Orbital analysis
        orbital_analysis = self.orbital_analyzer.analyze_orbital_patterns(text)
        results['orbital_analysis'] = orbital_analysis
        
        # 3. Generate sphere if requested
        if generate_sphere and BALLS_AVAILABLE:
            sphere_results = self.nine_monitor.create_quranic_sphere(text)
            results['sphere_results'] = sphere_results
            results['sphere_generated'] = True
            print(f"🌐 Sphere generated: {sphere_results.get('sphere_file')}")
        
        # 4. Create enhanced storybook entry
        if create_storybook:
            discovery_data = {
                'title': title,
                'main_number': '9',
                'pattern': f"Found {len(nine_analysis['nine_patterns'])} nine-based patterns",
                'location': title,
                'type': 'nine_geometry',
                'nine_count': len(nine_analysis['nine_patterns']),
                'sphere_generated': results['sphere_generated'],
                'orbital_score': orbital_analysis['total_orbital_score']
            }
            
            storybook_entry = self.storybook_system.create_interactive_storybook_entry(discovery_data, "children")
            results['storybook_entry'] = storybook_entry
            results['storybook_generated'] = True
        
        # 5. Create analytical summary
        analytical_summary = self.create_analytical_summary(nine_analysis, orbital_analysis, title)
        results['analytical_summary'] = analytical_summary
        
        print(f"✅ Analysis complete!")
        print(f"🔍 Nine-patterns found: {len(nine_analysis['nine_patterns'])}")
        print(f"🌌 Orbital patterns found: {orbital_analysis['total_orbital_score']}")
        if results['storybook_generated']:
            print(f"📚 Enhanced storybook created")
        if results['sphere_generated']:
            print(f"🌐 Sphere generated successfully")
        
        return results
    
    def create_analytical_summary(self, nine_analysis: Dict, orbital_analysis: Dict, title: str) -> str:
        """Create comprehensive analytical summary"""
        
        summary = []
        summary.append("=" * 80)
        summary.append("NINER MERGED SYSTEM - COMPREHENSIVE ANALYTICAL SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Analysis Date: {datetime.datetime.now().isoformat()}")
        summary.append(f"Analysis Title: {title}")
        summary.append("")
        
        # Nine-geometry details
        geometry = nine_analysis['geometry']
        summary.append("NINE-GEOMETRY ANALYSIS:")
        summary.append("-" * 40)
        summary.append(f"Text Length: {nine_analysis['text_length']}")
        summary.append(f"Total Points: {geometry['total_points']}")
        summary.append(f"Total Lines: {geometry['total_lines']}")
        summary.append(f"Total Curves: {geometry['total_curves']}")
        summary.append(f"Total Complexity: {geometry['total_complexity']}")
        summary.append(f"Total Abjad: {geometry['total_abjad']}")
        summary.append(f"Unique Letters: {geometry['unique_letters']}")
        summary.append(f"Total Letters: {geometry['total_letters']}")
        
        summary.append("")
        summary.append("NINE-BASED PATTERNS:")
        summary.append("-" * 40)
        
        nine_patterns = nine_analysis['nine_patterns']
        summary.append(f"Total Nine-Patterns: {len(nine_patterns)}")
        
        for pattern in nine_patterns:
            summary.append(f"• {pattern['significance']}")
        
        # Orbital analysis details
        summary.append("")
        summary.append("ORBITAL (TAJWEED) ANALYSIS:")
        summary.append("-" * 40)
        summary.append(f"Ghunnah Patterns: {orbital_analysis['ghunnah_patterns']}")
        summary.append(f"Idgham Patterns: {orbital_analysis['idgham_patterns']}")
        summary.append(f"Madd Patterns: {orbital_analysis['madd_patterns']}")
        summary.append(f"Qalqalah Patterns: {orbital_analysis['qalqalah_patterns']}")
        summary.append(f"Total Orbital Score: {orbital_analysis['total_orbital_score']}")
        
        summary.append("")
        summary.append("MATHEMATICAL VALIDATION:")
        summary.append("-" * 40)
        
        if len(nine_patterns) >= 4:
            summary.append("Nine-Geometry: EXCELLENT - Strong nine-geometry signature")
        elif len(nine_patterns) >= 2:
            summary.append("Nine-Geometry: GOOD - Clear nine-geometry patterns")
        elif len(nine_patterns) >= 1:
            summary.append("Nine-Geometry: MODERATE - Some nine-geometry detected")
        else:
            summary.append("Nine-Geometry: MINIMAL - Limited nine-geometry patterns")
        
        if orbital_analysis['total_orbital_score'] >= 20:
            summary.append("Orbital Patterns: EXCELLENT - Rich tajweed structure")
        elif orbital_analysis['total_orbital_score'] >= 10:
            summary.append("Orbital Patterns: GOOD - Clear tajweed patterns")
        elif orbital_analysis['total_orbital_score'] >= 5:
            summary.append("Orbital Patterns: MODERATE - Some tajweed elements")
        else:
            summary.append("Orbital Patterns: MINIMAL - Limited tajweed features")
        
        summary.append("")
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def save_complete_experience(self, results: Dict) -> Dict:
        """Save complete enhanced experience to files"""
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        title_safe = results['title'].replace(' ', '_').replace('/', '_')
        
        files = {}
        
        # Save enhanced storybook
        if results.get('storybook_generated'):
            storybook_file = f"niner_enhanced_storybook_{title_safe}_{timestamp}.txt"
            self.storybook_system.save_storybook(storybook_file)
            files['enhanced_storybook'] = storybook_file
        
        # Save analytical summary
        analytical_file = f"niner_analytical_{title_safe}_{timestamp}.txt"
        with open(analytical_file, 'w', encoding='utf-8') as f:
            f.write(results['analytical_summary'])
        files['analytical'] = analytical_file
        
        # Save sphere info if generated
        if results.get('sphere_generated') and 'sphere_results' in results:
            sphere_info_file = f"niner_sphere_{title_safe}_{timestamp}.txt"
            with open(sphere_info_file, 'w', encoding='utf-8') as f:
                f.write(f"NINER MERGED SYSTEM - SPHERE GENERATION RESULTS\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Title: {results['title']}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Sphere File: {results['sphere_results']['sphere_file']}\n")
                f.write(f"Digit Sequence Length: {results['sphere_results']['digit_sequence_length']}\n")
                f.write(f"Unique Digits: {results['sphere_results']['unique_digits']}\n")
                f.write(f"Nine Patterns Found: {len(results['sphere_results']['nine_patterns'])}\n\n")
                
                f.write("NINE-GEOMETRY SUMMARY:\n")
                geometry = results['nine_analysis']['geometry']
                f.write(f"Total Points: {geometry['total_points']}\n")
                f.write(f"Total Lines: {geometry['total_lines']}\n")
                f.write(f"Total Curves: {geometry['total_curves']}\n")
                f.write(f"Total Complexity: {geometry['total_complexity']}\n")
                f.write(f"Total Abjad: {geometry['total_abjad']}\n")
                f.write(f"Unique Letters: {geometry['unique_letters']}\n")
                f.write(f"Total Letters: {geometry['total_letters']}\n\n")
                
                f.write("ORBITAL ANALYSIS:\n")
                orbital = results['orbital_analysis']
                f.write(f"Total Orbital Score: {orbital['total_orbital_score']}\n")
                f.write(f"Ghunnah: {orbital['ghunnah_patterns']}\n")
                f.write(f"Idgham: {orbital['idgham_patterns']}\n")
                f.write(f"Madd: {orbital['madd_patterns']}\n")
                f.write(f"Qalqalah: {orbital['qalqalah_patterns']}\n\n")
                
                f.write("NINE-BASED PATTERNS:\n")
                for pattern in results['sphere_results']['nine_patterns']:
                    f.write(f"- {pattern['significance']}\n")
            files['sphere'] = sphere_info_file
        
        # Save complete JSON results
        json_file = f"niner_complete_results_{title_safe}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        files['complete_json'] = json_file
        
        # Save experience log
        log_file = f"niner_experience_log_{timestamp}.json"
        experience_entry = {
            'timestamp': results['analysis_timestamp'],
            'title': results['title'],
            'text_length': len(results['text']),
            'nine_patterns_found': len(results['nine_analysis']['nine_patterns']),
            'orbital_score': results['orbital_analysis']['total_orbital_score'],
            'sphere_generated': results['sphere_generated'],
            'storybook_generated': results['storybook_generated'],
            'files_saved': list(files.values())
        }
        
        self.experience_log.append(experience_entry)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experience_log, f, indent=2, ensure_ascii=False)
        files['log'] = log_file
        
        print(f"\n📚 Niner Merged Experience Saved!")
        for file_type, filename in files.items():
            print(f"  {file_type.title().replace('_', ' ')}: {filename}")
        
        return files
    
    def generate_conclusion(self, user_input: str, analysis_results: Dict) -> str:
        """Generate conclusion based on user input and complete analysis"""
        return self.storybook_system.generate_conclusion_story(user_input, analysis_results)
    
    def interactive_menu(self):
        """Enhanced interactive menu for merged Niner system"""
        
        while True:
            print("\n" + "🎯" * 20)
            print("🔍 NINER MERGED SYSTEM v4.0 🔍")
            print("🎯" * 20)
            print("Complete Nine-Geometry & Orbital Analysis with Enhanced Storybook")
            print("\n📋 ENHANCED MENU OPTIONS:")
            print("1. 🔍 Analyze Qur'anic Text from Database")
            print("2. 📝 Analyze Custom Text")
            print("3. 🎲 Random Qur'anic Analysis")
            print("4. 📚 View Storybook Library")
            print("5. 📊 View Experience Log")
            print("6. 🌐 Sphere Generation Status")
            print("7. 📖 Learn About Nine-Geometry")
            print("8. 🌌 Learn About Orbital Analysis")
            print("9. 🎓 Generate Personal Conclusion")
            print("10. 📖 Traditional Niner Analysis")
            print("11. ❓ Help & Islamic Guidance")
            print("12. 🚪 Exit")
            
            choice = input("\n✨ Enter your choice (1-12): ").strip()
            
            if choice == '1':
                self.analyze_database_text()
            elif choice == '2':
                self.analyze_custom_text()
            elif choice == '3':
                self.analyze_random_text()
            elif choice == '4':
                self.view_storybook_library()
            elif choice == '5':
                self.view_experience_log()
            elif choice == '6':
                self.toggle_sphere_generation()
            elif choice == '7':
                self.learn_nine_geometry()
            elif choice == '8':
                self.learn_orbital_analysis()
            elif choice == '9':
                self.generate_personal_conclusion()
            elif choice == '10':
                self.traditional_niner_analysis()
            elif choice == '11':
                self.show_help()
            elif choice == '12':
                print("\n🙏 Thank you for using Niner Merged System!")
                print("May Allah guide us to understand His perfect mathematical design!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
            
            input("\n⏸️  Press Enter to continue...")
    
    def analyze_database_text(self):
        """Analyze text from enhanced database"""
        print("\n🔍 QUR'ANIC TEXT DATABASE")
        print("=" * 50)
        
        if not self.text_database.texts:
            print("❌ No texts available in database")
            return
        
        print("Available texts:")
        for i, text_data in enumerate(self.text_database.texts[:10], 1):
            print(f"{i}. {text_data['title']} ({text_data['category']})")
        
        print(f"... and {len(self.text_database.texts) - 10} more")
        
        try:
            choice = int(input(f"\n🎯 Choose text (1-{min(10, len(self.text_database.texts))}: "))
            if 1 <= choice <= min(10, len(self.text_database.texts)):
                text_data = self.text_database.texts[choice - 1]
                self.perform_complete_analysis(text_data['text'], text_data['title'])
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def analyze_custom_text(self):
        """Analyze custom text with full features"""
        print("\n📝 CUSTOM TEXT ANALYSIS")
        print("=" * 50)
        
        text = input("📝 Enter Arabic or any text: ").strip()
        if not text:
            print("❌ No text entered")
            return
        
        title = input("🏷️  Enter title (or press Enter for default): ").strip()
        if not title:
            title = f"Custom Analysis {len(text)} chars"
        
        self.perform_complete_analysis(text, title)
    
    def analyze_random_text(self):
        """Analyze random Qur'anic text"""
        print("\n🎲 RANDOM QUR'ANIC ANALYSIS")
        print("=" * 50)
        
        if not self.text_database.texts:
            print("❌ No texts available in database")
            return
        
        text_data = random.choice(self.text_database.texts)
        print(f"🎲 Selected: {text_data['title']}")
        print(f"📝 Text: {text_data['text']}")
        
        self.perform_complete_analysis(text_data['text'], text_data['title'])
    
    def perform_complete_analysis(self, text: str, title: str):
        """Perform complete analysis and save"""
        
        print("\n🔧 ANALYSIS OPTIONS:")
        print("1. Full Analysis (Sphere + Storybook)")
        print("2. Geometry Only (No Sphere)")
        print("3. Storybook Only")
        print("4. Quick Analysis")
        
        analysis_choice = input("\n🎯 Choose analysis type (1-4): ").strip()
        
        generate_sphere = BALLS_AVAILABLE
        create_storybook = True
        
        if analysis_choice == '1':
            generate_sphere = BALLS_AVAILABLE
            create_storybook = True
        elif analysis_choice == '2':
            generate_sphere = False
            create_storybook = True
        elif analysis_choice == '3':
            generate_sphere = False
            create_storybook = True
        elif analysis_choice == '4':
            generate_sphere = False
            create_storybook = False
        else:
            print("🎯 Using default: Full Analysis")
        
        results = self.analyze_complete_text(text, title, generate_sphere, create_storybook)
        files = self.save_complete_experience(results)
        
        # Show preview
        self.show_analysis_preview(results)
    
    def show_analysis_preview(self, results: Dict):
        """Show preview of analysis results"""
        
        print("\n📊 ANALYSIS PREVIEW:")
        print("-" * 40)
        
        # Nine-geometry summary
        nine_patterns = len(results['nine_analysis']['nine_patterns'])
        orbital_score = results['orbital_analysis']['total_orbital_score']
        
        print(f"🔍 Nine-Patterns Found: {nine_patterns}")
        print(f"🌌 Orbital Score: {orbital_score}")
        
        if results.get('sphere_generated'):
            print(f"🌐 Sphere: Generated Successfully")
        if results.get('storybook_generated'):
            print(f"📚 Storybook: Interactive Version Created")
        
        # Storybook preview
        if results.get('storybook_generated') and 'storybook_entry' in results:
            print("\n📚 STORYBOOK PREVIEW:")
            print("-" * 40)
            story_lines = results['storybook_entry']['story'].split('\n')
            for line in story_lines[:6]:
                print(line)
            if len(story_lines) > 6:
                print("...")
            print("-" * 40)
    
    def view_storybook_library(self):
        """View the storybook library"""
        print("\n📚 STORYBOOK LIBRARY")
        print("=" * 50)
        
        if not self.storybook_system.storybook_library:
            print("📝 No storybooks created yet")
            return
        
        print(f"Total Storybooks: {len(self.storybook_system.storybook_library)}")
        for i, entry in enumerate(self.storybook_system.storybook_library, 1):
            print(f"{i}. {entry['title']} - {entry['timestamp'][:10]}")
        
        choice = input(f"\n📖 View storybook (1-{len(self.storybook_system.storybook_library)}) or Enter to skip: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(self.storybook_system.storybook_library):
                entry = self.storybook_system.storybook_library[idx]
                print(f"\n📖 {entry['title']}")
                print("=" * 40)
                print(entry['story'])
    
    def view_experience_log(self):
        """View previous experiences"""
        print("\n📊 EXPERIENCE LOG")
        print("=" * 50)
        
        if not self.experience_log:
            print("📝 No experiences logged yet")
            return
        
        print(f"Total Experiences: {len(self.experience_log)}")
        for i, exp in enumerate(self.experience_log[-10:], 1):  # Show last 10
            print(f"{i}. {exp['title']} - {exp['nine_patterns_found']} nine-patterns, {exp['orbital_score']} orbital score")
    
    def toggle_sphere_generation(self):
        """Toggle sphere generation status"""
        print(f"\n🌐 SPHERE GENERATION STATUS")
        print("=" * 50)
        print(f"balls.py Available: {'YES' if BALLS_AVAILABLE else 'NO'}")
        print(f"Sphere Generation: {'ENABLED' if BALLS_AVAILABLE else 'DISABLED - Install balls.py'}")
        if BALLS_AVAILABLE:
            print(f"Default Sphere Type: RELATIONAL (most advanced)")
            print(f"Status: Ready for 3D visualization of Qur'anic text")
    
    def learn_nine_geometry(self):
        """Learn about nine-geometry"""
        print("\n📖 NINE-GEOMETRY EDUCATION")
        print("=" * 60)
        print("🔍 THE SACRED NUMBER 9 AS 3² FOUNDATION")
        print("=" * 60)
        print()
        print("🌟 Why Nine (9) is Special:")
        print("  • 9 = 3 × 3 = 3² (3 squared)")
        print("  • Three represents divine perfection")
        print("  • Squared represents manifestation in creation")
        print("  • Nine is the completion of the single-digit cycle")
        print("  • Digital root 9 = completion and perfection")
        print()
        print("📐 In Creation:")
        print("  • 9 months of human gestation")
        print("  • 9 classical celestial spheres")
        print("  • 9 as final single digit before cycles repeat")
        print()
        print("📖 In Qur'anic Mathematics:")
        print("  • Patterns divisible by 9 show divine structure")
        print("  • Digital root 9 indicates completion")
        print("  • 3² foundation appears throughout sacred text")
        print()
        print("🌐 In This System:")
        print("  • Analyzes text for 9-based patterns")
        print("  • Generates 3D sphere representations")
        print("  • Creates beautiful stories from discoveries")
        print("  • Preserves mathematical wisdom for learning")
        print()
        print("💝 Remember: Mathematics is Allah's language in creation!")
    
    def learn_orbital_analysis(self):
        """Learn about orbital analysis"""
        print("\n🌌 ORBITAL (TAJWEED) ANALYSIS EDUCATION")
        print("=" * 60)
        print("🎵 CELESTIAL PATTERNS IN QUR'ANIC RECITATION")
        print("=" * 60)
        print()
        print("🌟 What are Tajweed Orbits:")
        print("  • Patterns of sound in Qur'anic recitation")
        print("  • Following specific rules of pronunciation")
        print("  • Creating beautiful, harmonious recitation")
        print("  • Like celestial bodies following divine paths")
        print()
        print("🎵 Main Orbital Patterns:")
        for orbital in self.orbital_analyzer.tajweed_orbitals:
            meaning = self.orbital_analyzer.tajweed_terms[orbital]
            print(f"  • {orbital}: {meaning}")
        print()
        print("📖 In This System:")
        print("  • Analyzes text for tajweed patterns")
        print("  • Creates orbital scores for complexity")
        print("  • Connects recitation with celestial harmony")
        print("  • Generates educational stories about sounds")
        print()
        print("💝 Remember: Beautiful recitation brings us closer to Allah!")
    
    def generate_personal_conclusion(self):
        """Generate personal conclusion based on user input"""
        print("\n🎓 PERSONAL CONCLUSION GENERATOR")
        print("=" * 50)
        
        user_input = input("💭 Share your thoughts and experiences: ").strip()
        if not user_input:
            print("❌ No input provided")
            return
        
        # Use the most recent analysis if available
        if self.experience_log:
            latest_experience = self.experience_log[-1]
            analysis_results = {
                'total_verses': latest_experience.get('text_length', 0),
                'total_patterns': latest_experience.get('nine_patterns_found', 0),
                'nine_patterns': latest_experience.get('nine_patterns_found', 0),
                'orbital_patterns': latest_experience.get('orbital_score', 0)
            }
        else:
            analysis_results = {
                'total_verses': 0,
                'total_patterns': 0,
                'nine_patterns': 0,
                'orbital_patterns': 0
            }
        
        conclusion = self.generate_conclusion(user_input, analysis_results)
        
        print("\n🎓 YOUR PERSONAL CONCLUSION:")
        print("=" * 50)
        print(conclusion)
        
        save = input("\n💾 Save conclusion to file? (y/n): ").strip().lower()
        if save == 'y':
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"niner_personal_conclusion_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"PERSONAL CONCLUSION - {timestamp}\n")
                f.write("=" * 50 + "\n")
                f.write(f"User Input: {user_input}\n")
                f.write(f"\n{conclusion}\n")
            print(f"✅ Conclusion saved to {filename}")
    
    def traditional_niner_analysis(self):
        """Traditional Niner analysis from original versions"""
        print("\n🔍 TRADITIONAL NINER ANALYSIS")
        print("=" * 50)
        print("Accessing original Niner functionality...")
        
        while True:
            print("\n📋 Traditional Niner Options:")
            print("1. Surah-Specific Nine Patterns")
            print("2. General Qur'anic Nine Analysis")
            print("3. Mathematical Properties of Nine")
            print("4. Nine in Number Theory")
            print("5. Back to Main Menu")
            
            choice = input("\nChoose traditional analysis: ").strip()
            
            if choice == '1':
                self.analyze_surah_specific_nines()
            elif choice == '2':
                self.analyze_general_quran_nines()
            elif choice == '3':
                self.analyze_mathematical_properties()
            elif choice == '4':
                self.nine_number_theory()
            elif choice == '5':
                break
            else:
                print("❌ Invalid choice")
    
    def analyze_surah_specific_nines(self):
        """Analyze nine patterns per surah"""
        print("\n📖 SURAH-SPECIFIC NINE PATTERNS")
        print("=" * 50)
        
        try:
            surah = int(input("\n📖 Enter surah number (1-114): "))
            if not 1 <= surah <= 114:
                print("❌ Please enter a valid surah number (1-114)")
                return
            
            print(f"\n🔍 Analyzing Surah {surah} for Nine Patterns:")
            
            # Basic nine divisibility
            if surah % 9 == 0:
                print(f"✅ Surah {surah} is divisible by 9: {surah // 9} × 9")
            
            # Check for sacred nine multiples
            if surah in self.sacred_nines:
                print(f"✅ Sacred nine multiple: {self.sacred_nines[surah]}")
            
            # Check digits sum to 9
            digit_sum = sum(int(d) for d in str(surah))
            if digit_sum == 9:
                print(f"✅ Digital nine: Digits sum to 9")
            
            # Check for 3-based patterns
            if surah % 3 == 0:
                print(f"✅ Three foundation: {surah // 3} × 3 (foundation of 9)")
            
            print(f"\n📊 Mathematical Properties of {surah}:")
            print(f"  • Divisible by 9: {surah % 9 == 0}")
            print(f"  • Divisible by 3: {surah % 3 == 0}")
            print(f"  • Digital root: {surah % 9 if surah % 9 != 0 else 9}")
            
        except ValueError:
            print("❌ Please enter a valid number")
    
    def analyze_general_quran_nines(self):
        """General Qur'anic nine analysis"""
        print("\n📚 GENERAL QUR'ANIC NINE ANALYSIS")
        print("=" * 50)
        
        total_surahs = self.quran_structure['total_surahs']
        print(f"\n📖 Total Surahs: {total_surahs}")
        print(f"  • {total_surahs} ÷ 9 = {total_surahs / 9:.1f}")
        print(f"  • Digital root: {total_surahs % 9 if total_surahs % 9 != 0 else 9}")
        
        # Nine-based surah distribution
        nine_multiples = [s for s in range(1, 115) if s % 9 == 0]
        print(f"\n🎯 Nine-based Surahs (multiples of 9): {len(nine_multiples)} surahs")
        print(f"  • Surahs: {', '.join(map(str, nine_multiples[:10]))}...")
    
    def analyze_mathematical_properties(self):
        """Mathematical properties of nine"""
        print("\n📐 MATHEMATICAL PROPERTIES OF NINE")
        print("=" * 50)
        
        print("\n🌟 Fundamental Properties:")
        print("  • 9 = 3² (perfect square of 3)")
        print("  • 9 is the largest single-digit number")
        print("  • 9 is a composite number: 3 × 3")
        print("  • Digital root property: Numbers divisible by 9 have digits summing to 9")
        
        print("\n🔍 Powers of 9:")
        for i in range(1, 6):
            power = 9 ** i
            digit_sum = sum(int(d) for d in str(power))
            print(f"  • 9^{i} = {power} (digit sum: {digit_sum})")
        
        print("\n🎯 Special Mathematical Facts:")
        print("  • 9 is the 4th happy number")
        print("  • 9 is a centered square number")
        print("  • 9 is used in casting out nines (divisibility test)")
    
    def nine_number_theory(self):
        """Nine in number theory"""
        print("\n📊 NINE IN NUMBER THEORY")
        print("=" * 50)
        
        print("\n🔍 Number Theory Properties:")
        print("  • 9 is an odd composite number")
        print("  • Divisors of 9: 1, 3, 9")
        print("  • Prime factors: 3, 3")
        print("  • Euler's totient φ(9) = 6")
        print("  • Sum of divisors σ(9) = 13")
        print("  • 9 is a deficient number (sum of proper divisors = 4 < 9)")
        print("  • 9 is a square number (3²)")
        print("  • 9 is a centered octagonal number")
    
    def show_help(self):
        """Show comprehensive help and Islamic guidance"""
        print("\n❓ NINER MERGED SYSTEM HELP")
        print("=" * 50)
        print("🔍 ABOUT NINER MERGED v4.0:")
        print("  Niner analyzes the sacred number 9 as 3² foundation")
        print("  Merged from v2.0 and v3.0 with enhanced storybook features")
        print("  Complete Nine-Geometry and Orbital analysis system")
        print()
        
        print(self.storybook_system.create_shirk_notice())
        
        print("\n📚 ENHANCED STORYBOOK FEATURES:")
        print("  • Interactive quizzes for learning")
        print("  • Hands-on activities for children")
        print("  • Visualization suggestions")
        print("  • Educational notes for parents/teachers")
        print("  • Reflection questions for deeper understanding")
        print()
        
        print("🔍 NINE-GEOMETRY ANALYSIS:")
        print("  • Detects divisibility by 9 and 3 (3²)")
        print("  • Calculates digital roots (9 = completion)")
        print("  • Finds patterns in letter geometry")
        print("  • Abjad value analysis")
        print("  • 3D sphere generation (if balls.py available)")
        print()
        
        print("🌌 ORBITAL ANALYSIS:")
        print("  • Tajweed pattern detection")
        print("  • Orbital scoring system")
        print("  • Sound pattern analysis")
        print("  • Educational recitation insights")
        print()
        
        print("📁 EXPERIENCE SAVING:")
        print("  • All analyses saved automatically")
        print("  • Separate files for different purposes")
        print("  • Build your discovery library")
        print("  • Personal conclusion generation")
        print()
        
        print("🎯 USAGE TIPS:")
        print("  • Start with Qur'anic database texts")
        print("  • Try different analysis types")
        print("  • Read storybooks to children")
        print("  • Use interactive elements for learning")
        print("  • Generate personal conclusions")
        print("  • Always maintain proper Islamic perspective")

def main():
    """Main function for merged Niner system"""
    
    print("🔍" * 20)
    print("🎯 NINER MERGED SYSTEM v4.0 🎯")
    print("🔍" * 20)
    print("Complete Nine-Geometry & Orbital Analysis with Enhanced Storybook")
    print()
    
    # Show Islamic guidance
    print(EnhancedStorybookSystem().create_shirk_notice())
    print()
    
    # Initialize system
    niner_system = NinerMergedSystem()
    
    # Show system status
    print(f"🌐 Sphere Generation: {'✅ ENABLED' if BALLS_AVAILABLE else '❌ DISABLED'}")
    print(f"📚 Qur'anic Database: {len(niner_system.text_database.texts)} verses")
    print(f"📚 Storybook System: Ready")
    print()
    
    # Start interactive menu
    niner_system.interactive_menu()

if __name__ == "__main__":
    main()
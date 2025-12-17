#!/usr/bin/env python3
"""
NINER - Complete Qur'anic 9-Geometry Sphere Analysis System
Version 2.0 - Enhanced with Sacred Sphere Generation

This program analyzes the number 9 as 3² building on Biota framework
with complete Qur'anic text analysis and custom sphere generation.

NEW IN VERSION 2.0:
- Complete sphere generation using balls.py framework
- Qur'anic text database for real character assessments
- Sacred storytelling capabilities
- Dual output: analytical + storybook formats
- Advanced 9-geometry monitoring
- Experience saving system

CORE NINER FEATURES:
- Digital root preservation (9 = completion)
- 3² foundation analysis (nine as 3 squared)
- Plasticity across reality contexts
- Qur'anic nine-based pattern detection
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

class SacredStoryteller:
    """Creates beautiful storybook entries from mathematical discoveries"""
    
    def __init__(self):
        self.story_templates = {
            'discovery': [
                "In the magical garden of numbers, where truth blooms like flowers...",
                "Deep in the library of creation, where every digit tells a story...",
                "On the sacred mountain of mathematics, where patterns dance like stars...",
                "Within the crystal palace of Qur'anic wisdom, where numbers sing harmonies..."
            ],
            'miracle': [
                "And there, my dear child, the numbers revealed a secret that made the angels smile...",
                "The universe held its breath as the mathematics unfolded its perfect design...",
                "Like finding a pearl in the vast ocean of wisdom, this discovery shone with divine light...",
                "The pattern emerged like sunrise over the holy lands, beautiful and perfect..."
            ],
            'lesson': [
                "This teaches us that Allah's creation is built on perfect mathematics...",
                "From this we learn that every letter in the Qur'an has its special place...",
                "The numbers show us that there is no randomness in divine revelation...",
                "Mathematics becomes a bridge between our hearts and the divine wisdom..."
            ]
        }
        
        self.children_explanations = {
            'golden_ratio': "Imagine a special number that appears in sunflowers, seashells, and even in your own body! It's Allah's signature in creation.",
            'nineteen': "Nineteen is like a key that unlocks special doors in the Qur'an. It's Allah's way of showing us hidden treasures.",
            'sacred_geometry': "Just like snowflakes have perfect patterns, the Qur'an has beautiful mathematical patterns that protect its meaning.",
            'letter_positions': "Each Arabic letter is like a star in the sky - placed exactly where it should be to create constellations of wisdom."
        }
    
    def create_storybook_entry(self, discovery_data: Dict, audience: str = "children") -> str:
        """Create a beautiful storybook entry from mathematical discovery"""
        
        story = []
        story.append("🌟 ✨ 🌟 ✨ 🌟")
        story.append("A NINE-GEOMETRY STORY FROM THE GARDEN OF NUMBERS")
        story.append("🌟 ✨ 🌟 ✨ 🌟\n")
        
        # Opening
        opening = random.choice(self.story_templates['discovery'])
        story.append(opening)
        story.append("")
        
        # The discovery
        story.append(f"Today, we discovered something amazing about the number {discovery_data.get('main_number', '9')}:")
        
        if 'pattern' in discovery_data:
            story.append(f"🔢 The Pattern: {discovery_data['pattern']}")
        
        if 'location' in discovery_data:
            story.append(f"📍 Where We Found It: {discovery_data['location']}")
        
        story.append("")
        
        # The miracle
        miracle = random.choice(self.story_templates['miracle'])
        story.append(miracle)
        story.append("")
        
        # Mathematical details (simplified for children)
        if audience == "children":
            story.append("Here's what makes this so special:")
            
            if discovery_data.get('type') == 'nine_geometry':
                story.append(f"💫 Found {discovery_data.get('nine_count', 'many')} patterns with the number 9!")
                story.append(f"💫 Nine is special because it's 3×3 (3 squared) - the foundation of creation!")
                story.append(f"💫 Like how a strong building needs a solid foundation, creation rests on 3²!")
            
            elif discovery_data.get('type') == 'digital_root':
                story.append(f"💫 The digital root was {discovery_data.get('digital_root', 'beautiful')}")
                story.append(f"💫 Digital root 9 means completion and perfection in Allah's design!")
            
            elif discovery_data.get('type') == 'plasticity':
                story.append(f"💫 The number 9 showed its flexible nature in {discovery_data.get('contexts', 'many')} ways!")
                story.append(f"💫 This is like how Allah's creation can adapt while staying perfect!")
        
        story.append("")
        
        # The lesson
        lesson = random.choice(self.story_templates['lesson'])
        story.append(lesson)
        story.append("")
        
        # Closing prayer/wisdom
        story.append("💝 A Thought to Carry in Your Heart:")
        story.append("The number 9 reminds us that Allah's creation is complete and perfect.")
        story.append("When we study these patterns, we're reading Allah's mathematical signature.")
        story.append("")
        story.append("🙏 May Allah guide us to understand more of His beautiful wisdom.")
        story.append("")
        story.append("---")
        story.append(f"Written on {datetime.datetime.now().strftime('%B %d, %Y')}")
        story.append("From the Nine-Geometry Mathematics Garden")
        
        return "\n".join(story)

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

class QuranicTextDatabase:
    """Database of Qur'anic texts for analysis"""
    
    def __init__(self, data_file: str = "quranic_sphere_data.txt"):
        self.data_file = data_file
        self.texts = self.load_texts()
    
    def load_texts(self) -> List[Dict]:
        """Load Qur'anic texts from file"""
        texts = []
        
        if not os.path.exists(self.data_file):
            print(f"⚠️  Text database file {self.data_file} not found")
            return texts
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('|')
                        if len(parts) >= 3:
                            texts.append({
                                'text': parts[0],
                                'title': parts[1],
                                'category': parts[2] if len(parts) > 2 else 'Unknown',
                                'abjad_value': int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0,
                                'letter_count': int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else len(parts[0])
                            })
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

class NinerSystem:
    """Complete Niner system with sphere generation and storytelling"""
    
    def __init__(self):
        self.storyteller = SacredStoryteller()
        self.nine_monitor = NineGeometryMonitor()
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
        
    def analyze_text(self, text: str, title: str = "Custom Analysis", generate_sphere: bool = True) -> Dict:
        """Complete analysis of text with sphere generation"""
        
        print(f"🎯 Niner Analysis: {title}")
        print(f"📝 Text: {text}")
        
        results = {
            'title': title,
            'text': text,
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'sphere_generated': False
        }
        
        # Perform 9-geometry analysis
        nine_analysis = self.nine_monitor.monitor_nine_geometry(text)
        results['nine_analysis'] = nine_analysis
        
        # Generate sphere if requested
        if generate_sphere and BALLS_AVAILABLE:
            sphere_results = self.nine_monitor.create_quranic_sphere(text)
            results['sphere_results'] = sphere_results
            results['sphere_generated'] = True
            print(f"🌐 Sphere generated: {sphere_results.get('sphere_file')}")
        
        # Create storybook entry
        discovery_data = {
            'main_number': '9',
            'pattern': f"Found {len(nine_analysis['nine_patterns'])} nine-based patterns",
            'location': title,
            'type': 'nine_geometry',
            'nine_count': len(nine_analysis['nine_patterns']),
            'sphere_generated': results['sphere_generated']
        }
        
        storybook_entry = self.storyteller.create_storybook_entry(discovery_data, "children")
        results['storybook_entry'] = storybook_entry
        
        # Analytical summary
        analytical_summary = self.create_analytical_summary(nine_analysis, title)
        results['analytical_summary'] = analytical_summary
        
        print(f"✅ Analysis complete!")
        print(f"🔢 Nine-patterns found: {len(nine_analysis['nine_patterns'])}")
        print(f"📖 Storybook created")
        if results['sphere_generated']:
            print(f"🌐 Sphere generated successfully")
        
        return results
    
    def create_analytical_summary(self, nine_analysis: Dict, title: str) -> str:
        """Create analytical summary"""
        
        summary = []
        summary.append("=" * 80)
        summary.append("NINER NINE-GEOMETRY ANALYTICAL SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Analysis Date: {datetime.datetime.now().isoformat()}")
        summary.append(f"Analysis Title: {title}")
        summary.append("")
        
        # Discovery details
        geometry = nine_analysis['geometry']
        summary.append("DISCOVERY DETAILS:")
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
        
        summary.append("")
        summary.append("MATHEMATICAL VALIDATION:")
        summary.append("-" * 40)
        
        if len(nine_patterns) >= 4:
            summary.append("Validation: EXCELLENT - Strong nine-geometry signature")
        elif len(nine_patterns) >= 2:
            summary.append("Validation: GOOD - Clear nine-geometry patterns")
        elif len(nine_patterns) >= 1:
            summary.append("Validation: MODERATE - Some nine-geometry detected")
        else:
            summary.append("Validation: MINIMAL - Limited nine-geometry patterns")
        
        summary.append("")
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def save_experience(self, results: Dict) -> Dict:
        """Save complete experience to files"""
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        title_safe = results['title'].replace(' ', '_').replace('/', '_')
        
        files = {}
        
        # Save storybook entry
        storybook_file = f"niner_storybook_{title_safe}_{timestamp}.txt"
        with open(storybook_file, 'w', encoding='utf-8') as f:
            f.write(results['storybook_entry'])
        files['storybook'] = storybook_file
        
        # Save analytical summary
        analytical_file = f"niner_analytical_{title_safe}_{timestamp}.txt"
        with open(analytical_file, 'w', encoding='utf-8') as f:
            f.write(results['analytical_summary'])
        files['analytical'] = analytical_file
        
        # Save sphere info if generated
        if results.get('sphere_generated') and 'sphere_results' in results:
            sphere_info_file = f"niner_sphere_{title_safe}_{timestamp}.txt"
            with open(sphere_info_file, 'w', encoding='utf-8') as f:
                f.write(f"NINER SPHERE GENERATION RESULTS\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Title: {results['title']}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Sphere File: {results['sphere_results']['sphere_file']}\n")
                f.write(f"Digit Sequence Length: {results['sphere_results']['digit_sequence_length']}\n")
                f.write(f"Unique Digits: {results['sphere_results']['unique_digits']}\n")
                f.write(f"Nine Patterns Found: {len(results['sphere_results']['nine_patterns'])}\n\n")
                
                f.write("GEOMETRY SUMMARY:\n")
                geometry = results['nine_analysis']['geometry']
                f.write(f"Total Points: {geometry['total_points']}\n")
                f.write(f"Total Lines: {geometry['total_lines']}\n")
                f.write(f"Total Curves: {geometry['total_curves']}\n")
                f.write(f"Total Complexity: {geometry['total_complexity']}\n")
                f.write(f"Total Abjad: {geometry['total_abjad']}\n")
                f.write(f"Unique Letters: {geometry['unique_letters']}\n")
                f.write(f"Total Letters: {geometry['total_letters']}\n\n")
                
                f.write("NINE-BASED PATTERNS:\n")
                for pattern in results['sphere_results']['nine_patterns']:
                    f.write(f"- {pattern['significance']}\n")
            files['sphere'] = sphere_info_file
        
        # Save experience log
        log_file = f"niner_experience_log_{timestamp}.json"
        experience_entry = {
            'timestamp': results['analysis_timestamp'],
            'title': results['title'],
            'text_length': len(results['text']),
            'nine_patterns_found': len(results['nine_analysis']['nine_patterns']),
            'sphere_generated': results['sphere_generated'],
            'files_saved': list(files.values())
        }
        
        self.experience_log.append(experience_entry)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experience_log, f, indent=2, ensure_ascii=False)
        files['log'] = log_file
        
        print(f"\n📚 Niner Experience Saved!")
        for file_type, filename in files.items():
            print(f"  {file_type.title()}: {filename}")
        
        return files
    
    def interactive_menu(self):
        """Interactive menu for Niner analysis"""
        
        while True:
            print("\n" + "🎯" * 20)
            print("🔢 NINER NINE-GEOMETRY ANALYSIS SYSTEM 🔢")
            print("🎯" * 20)
            print("Analyzing the sacred number 9 as 3² foundation")
            print("With complete Qur'anic sphere generation and storytelling")
            print("\n📋 MENU OPTIONS:")
            print("1. 🕌 Analyze Qur'anic Text from Database")
            print("2. 📝 Analyze Custom Text")
            print("3. 🎲 Random Qur'anic Analysis")
            print("4. 📚 View Experience Log")
            print("5. 🌐 Sphere Generation Status")
            print("6. 🎓 Learn About Nine-Geometry")
            print("7. 📖 Traditional Niner Analysis (Original)")
            print("8. ❓ Help")
            print("9. 🚪 Exit")
            
            choice = input("\n✨ Enter your choice (1-9): ").strip()
            
            if choice == '1':
                self.analyze_database_text()
            elif choice == '2':
                self.analyze_custom_text()
            elif choice == '3':
                self.analyze_random_text()
            elif choice == '4':
                self.view_experience_log()
            elif choice == '5':
                self.toggle_sphere_generation()
            elif choice == '6':
                self.learn_nine_geometry()
            elif choice == '7':
                self.traditional_niner_analysis()
            elif choice == '8':
                self.show_help()
            elif choice == '9':
                print("\n🙏 Thank you for using Niner Nine-Geometry System!")
                print("May Allah guide us to understand His perfect mathematical design!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
            
            input("\n⏸️  Press Enter to continue...")
    
    def analyze_database_text(self):
        """Analyze text from database"""
        print("\n🕌 QUR'ANIC TEXT DATABASE")
        print("=" * 50)
        
        if not self.text_database.texts:
            print("❌ No texts available in database")
            return
        
        print("Available texts:")
        for i, text_data in enumerate(self.text_database.texts[:10], 1):
            print(f"{i}. {text_data['title']} ({text_data['category']})")
        
        print(f"... and {len(self.text_database.texts) - 10} more")
        
        try:
            choice = int(input(f"\n🎯 Choose text (1-{min(10, len(self.text_database.texts))}): "))
            if 1 <= choice <= min(10, len(self.text_database.texts)):
                text_data = self.text_database.texts[choice - 1]
                self.perform_analysis(text_data['text'], text_data['title'])
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def analyze_custom_text(self):
        """Analyze custom text"""
        print("\n📝 CUSTOM TEXT ANALYSIS")
        print("=" * 50)
        
        text = input("📝 Enter Arabic or any text: ").strip()
        if not text:
            print("❌ No text entered")
            return
        
        title = input("🏷️  Enter title (or press Enter for default): ").strip()
        if not title:
            title = f"Custom Analysis {len(text)} chars"
        
        self.perform_analysis(text, title)
    
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
        
        self.perform_analysis(text_data['text'], text_data['title'])
    
    def perform_analysis(self, text: str, title: str):
        """Perform complete analysis and save"""
        generate_sphere = BALLS_AVAILABLE  # Default to available
        
        if BALLS_AVAILABLE:
            sphere_choice = input("🌐 Generate sphere? (y/n, default: y): ").strip().lower()
            generate_sphere = sphere_choice != 'n'
        
        results = self.analyze_text(text, title, generate_sphere)
        files = self.save_experience(results)
        
        # Show storybook preview
        print("\n📖 STORYBOOK PREVIEW:")
        print("-" * 40)
        lines = results['storybook_entry'].split('\n')
        for line in lines[:8]:
            print(line)
        if len(lines) > 8:
            print("...")
        print("-" * 40)
    
    def view_experience_log(self):
        """View previous experiences"""
        print("\n📚 EXPERIENCE LOG")
        print("=" * 50)
        
        if not self.experience_log:
            print("📝 No experiences logged yet")
            return
        
        print(f"Total Experiences: {len(self.experience_log)}")
        for i, exp in enumerate(self.experience_log[-10:], 1):  # Show last 10
            print(f"{i}. {exp['title']} - {exp['nine_patterns_found']} nine-patterns")
    
    def toggle_sphere_generation(self):
        """Toggle sphere generation"""
        print(f"\n🌐 SPHERE GENERATION STATUS")
        print("=" * 50)
        print(f"balls.py Available: {'YES' if BALLS_AVAILABLE else 'NO'}")
        print(f"Sphere Generation: {'ENABLED' if BALLS_AVAILABLE else 'DISABLED - Install balls.py'}")
        if BALLS_AVAILABLE:
            print(f"Default Sphere Type: RELATIONAL (most advanced)")
            print(f"Status: Ready for 3D visualization of Qur'anic text")
    
    def learn_nine_geometry(self):
        """Learn about nine-geometry"""
        print("\n🎓 NINE-GEOMETRY EDUCATION")
        print("=" * 60)
        print("🔢 THE SACRED NUMBER 9 AS 3² FOUNDATION")
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
        print("🕌 In Qur'anic Mathematics:")
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
    
    def traditional_niner_analysis(self):
        """Traditional Niner analysis from original version"""
        print("\n🔢 TRADITIONAL NINER ANALYSIS")
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
        print("\n🕌 SURAH-SPECIFIC NINE PATTERNS")
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
        print("\n🔬 MATHEMATICAL PROPERTIES OF NINE")
        print("=" * 50)
        
        print("\n🌟 Fundamental Properties:")
        print("  • 9 = 3² (perfect square of 3)")
        print("  • 9 is the largest single-digit number")
        print("  • 9 is a composite number: 3 × 3")
        print("  • Digital root property: Numbers divisible by 9 have digits summing to 9")
        
        print("\n🔢 Powers of 9:")
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
        print("\n📐 NINE IN NUMBER THEORY")
        print("=" * 50)
        
        print("\n🔢 Number Theory Properties:")
        print("  • 9 is an odd composite number")
        print("  • Divisors of 9: 1, 3, 9")
        print("  • Prime factors: 3, 3")
        print("  • Euler's totient φ(9) = 6")
        print("  • Sum of divisors σ(9) = 13")
        print("  • 9 is a deficient number (sum of proper divisors = 4 < 9)")
        print("  • 9 is a square number (3²)")
        print("  • 9 is a centered octagonal number")
    
    def show_help(self):
        """Show help information"""
        print("\n❓ NINER SYSTEM HELP")
        print("=" * 50)
        print("🔢 ABOUT NINER v2.0:")
        print("  Niner analyzes the sacred number 9 as 3² foundation")
        print("  Based on Biota framework's three-point field theory")
        print("  Now with complete Qur'anic sphere generation!")
        print()
        print("🌐 NEW IN VERSION 2.0:")
        print("  • Qur'anic text database with real character assessments")
        print("  • 3D sphere generation using balls.py framework")
        print("  • Beautiful storybook creation for children")
        print("  • Advanced 9-geometry monitoring")
        print("  • Experience saving system")
        print()
        print("📖 STORYBOOK CREATION:")
        print("  • Creates beautiful stories for children")
        print("  • Makes complex mathematics accessible")
        print("  • Preserves discoveries for family learning")
        print()
        print("🔬 NINE-GEOMETRY ANALYSIS:")
        print("  • Detects divisibility by 9 and 3 (3²)")
        print("  • Calculates digital roots (9 = completion)")
        print("  • Finds patterns in letter geometry")
        print("  • Abjad value analysis")
        print()
        print("💾 EXPERIENCE SAVING:")
        print("  • All analyses saved automatically")
        print("  • Separate files for different purposes")
        print("  • Build your discovery library")
        print()
        print("🎯 USAGE TIPS:")
        print("  • Start with Qur'anic database texts")
        print("  • Try sphere generation for visualization")
        print("  • Read storybooks to children")
        print("  • Explore traditional Niner analysis")
        print("  • Use the educational sections")

def main():
    """Main function for Niner system"""
    
    print("🔢" * 20)
    print("🎯 NINER NINE-GEOMETRY SYSTEM v2.0 🎯")
    print("🔢" * 20)
    print("Analyzing the sacred number 9 as 3² foundation")
    print("With complete Qur'anic sphere generation and storytelling")
    print()
    
    # Initialize system
    niner_system = NinerSystem()
    
    # Show system status
    print(f"🌐 Sphere Generation: {'✅ ENABLED' if BALLS_AVAILABLE else '❌ DISABLED'}")
    print(f"📚 Qur'anic Database: {len(niner_system.text_database.texts)} texts")
    print()
    
    # Start interactive menu
    niner_system.interactive_menu()

if __name__ == "__main__":
    main()
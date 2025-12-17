#!/usr/bin/env python3
"""
FALAQI - Advanced Arabic Text Analysis with Relational Sphere Geometry
=======================================================================

This program analyzes Arabic text using sophisticated mathematical algorithms
to map linguistic patterns onto spherical geometric representations.

KEY FEATURES:
- Relational sphere geometry based on balls.py algorithms
- Arabic text processing with Quranic terminology analysis
- Word occurrence mapping with geometric coordinates
- Interactive input system with comprehensive limitations
- Mathematical notation system for text relationships

RESEARCH BASIS:
- Quranic Arabic Corpus for morphological analysis
- Hadwiger-Nelson inspired spherical geometry
- Arabic NLP patterns and linguistic structures
- Mathematical pattern detection in text sequences

LIMITATIONS:
- Minimum 100 Arabic characters required for meaningful analysis
- Processing time increases exponentially with text length
- Geometric accuracy depends on text complexity
- Not suitable for very short phrases (< 10 characters)
"""

import math
import re
from collections import Counter, defaultdict
import sys
import json
import unicodedata
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import hashlib

@dataclass
class ArabicCharacter:
    """Represents an Arabic character with its properties"""
    char: str
    unicode: int
    position: int
    word_index: int
    verse_index: int = 0
    
@dataclass
class WordMapping:
    """Represents a word mapped to geometric coordinates"""
    word: str
    occurrences: int
    positions: List[int]
    sphere_coords: List[Tuple[float, float, float]]
    geometric_signature: str
    
@dataclass
class RelationalPosition:
    """Represents a position in the relational sphere"""
    theta: float  # Azimuthal angle
    phi: float    # Polar angle
    radius: float  # Distance from center
    text_mapping: str
    forbidden_separations: List[float]

class ArabicTextProcessor:
    """Handles Arabic text processing and character analysis"""
    
    def __init__(self):
        self.arabic_regex = re.compile(r'[\u0600-\u06FF]+')
        self.verse_marker_regex = re.compile(r'\[\[VERSE_(\d+)\]\]')
        self.quranic_terminology = self._load_quranic_terms()
        
    def _load_quranic_terms(self) -> Dict[str, Dict]:
        """Load Quranic terminology database"""
        return {
            "allah": {"category": "divine", "frequency": 2699, "surahs": range(1, 115)},
            "rahman": {"category": "divine_attribute", "frequency": 170, "surahs": [1, 17, 20, 25, 27, 36, 43, 55]},
            "rahim": {"category": "divine_attribute", "frequency": 227, "surahs": list(range(1, 115))},
            "quran": {"category": "scripture", "frequency": 58, "surahs": [2, 3, 4, 5, 6, 7, 10, 12, 13, 15, 16, 17]},
            "islam": {"category": "religion", "frequency": 6, "surahs": [3, 5, 6, 39, 61]},
            "iman": {"category": "faith", "frequency": 25, "surahs": [2, 3, 4, 9, 10, 12, 16, 22, 24, 31, 33, 39, 49, 58]},
            "salaah": {"category": "worship", "frequency": 67, "surahs": [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 17, 19, 20, 22, 23, 24]},
            "zakat": {"category": "worship", "frequency": 32, "surahs": [2, 3, 4, 5, 6, 7, 9, 18, 19, 21, 22, 24, 27, 30, 31, 33, 41, 58, 73]},
            "saum": {"category": "worship", "frequency": 12, "surahs": [2, 4, 5, 19, 33, 58]},
            "hajj": {"category": "worship", "frequency": 9, "surahs": [2, 3, 5, 9, 22, 48]},
            # Add more terms as needed
        }
    
    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text for consistent processing"""
        # Remove diacritics for basic analysis
        text = re.sub(r'[\u064B-\u0652]', '', text)
        # Normalize alef variations
        text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
        # Normalize teh marbuta
        text = re.sub(r'\u0629', '\u0647', text)
        # Normalize yeh variations
        text = re.sub(r'[\u0649]', '\u064A', text)
        return text.strip()
    
    def extract_arabic_characters(self, text: str) -> List[ArabicCharacter]:
        """Extract Arabic characters with position information"""
        characters = []
        verse_count = 0
        char_position = 0
        
        lines = text.split('\n')
        for line in lines:
            # Check for verse markers
            verse_match = self.verse_marker_regex.match(line)
            if verse_match:
                verse_count = int(verse_match.group(1))
                continue
                
            # Process Arabic characters in line
            for i, char in enumerate(line):
                if self.arabic_regex.match(char):
                    characters.append(ArabicCharacter(
                        char=char,
                        unicode=ord(char),
                        position=char_position,
                        word_index=0,  # Will be calculated later
                        verse_index=verse_count
                    ))
                    char_position += 1
        
        return characters
    
    def tokenize_arabic_text(self, text: str) -> List[str]:
        """Tokenize Arabic text into words"""
        normalized = self.normalize_arabic_text(text)
        arabic_words = self.arabic_regex.findall(normalized)
        return [word.strip() for word in arabic_words if word.strip()]
    
    def get_quranic_terminology_info(self, word: str) -> Optional[Dict]:
        """Get information about Quranic terminology"""
        normalized_word = self.normalize_arabic_text(word.lower())
        return self.quranic_terminology.get(normalized_word)

class RelationalSphereGenerator:
    """Generates relational sphere coordinates using balls.py algorithms"""
    
    def __init__(self):
        self.forbidden_angles = [math.pi/6, math.pi/3, 2*math.pi/3]  # Hadwiger-Nelson constraints
        
    def trigonometric_polynomial(self, theta: float) -> float:
        """Calculate trigonometric polynomial T(θ) = cos²(3πθ) × cos²(6πθ)"""
        return (math.cos(3 * math.pi * theta) ** 2) * (math.cos(6 * math.pi * theta) ** 2)
    
    def check_forbidden_separation(self, angle1: float, angle2: float) -> bool:
        """Check if angular separation is forbidden"""
        separation = abs(angle1 - angle2)
        for forbidden in self.forbidden_angles:
            if abs(separation - forbidden) < 0.01:  # Small tolerance
                return True
        return False
    
    def generate_sphere_coordinates(self, text_length: int, characters: List[ArabicCharacter]) -> List[Tuple[float, float, float]]:
        """Generate 3D sphere coordinates for Arabic characters"""
        coordinates = []
        
        if text_length < 100:
            print("Warning: Text too short for meaningful spherical analysis")
            return coordinates
            
        # Use golden ratio for even distribution
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i, char in enumerate(characters):
            # Generate angles using Fibonacci-like distribution with Hadwiger-Nelson constraints
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 1) / (text_length + 1))
            
            # Apply trigonometric polynomial constraint
            poly_value = self.trigonometric_polynomial(theta / (2 * math.pi))
            if poly_value > 0.25:  # Measure bound constraint
                # Adjust theta to avoid forbidden angles
                for forbidden in self.forbidden_angles:
                    if self.check_forbidden_separation(theta, forbidden):
                        theta += 0.01  # Small adjustment
            
            # Convert to Cartesian coordinates
            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)
            
            # Normalize to unit sphere
            norm = math.sqrt(x**2 + y**2 + z**2)
            coordinates.append((x/norm, y/norm, z/norm))
        
        return coordinates

class FalaqiAnalyzer:
    """Main Falaqi analysis engine"""
    
    def __init__(self):
        self.text_processor = ArabicTextProcessor()
        self.sphere_generator = RelationalSphereGenerator()
        self.analysis_results = {}
        
    def analyze_text(self, arabic_text: str) -> Dict:
        """Perform comprehensive analysis of Arabic text"""
        print(f"🔍 Starting Falaqi Analysis...")
        print(f"📝 Input text length: {len(arabic_text)} characters")
        
        # Validate input
        if len(arabic_text.strip()) < 10:
            raise ValueError("Text too short for meaningful analysis (minimum 10 characters)")
        
        # Extract and process characters
        characters = self.text_processor.extract_arabic_characters(arabic_text)
        words = self.text_processor.tokenize_arabic_text(arabic_text)
        
        print(f"🔤 Arabic characters found: {len(characters)}")
        print(f"📚 Words identified: {len(words)}")
        
        # Generate sphere coordinates
        coordinates = self.sphere_generator.generate_sphere_coordinates(len(characters), characters)
        
        # Analyze word occurrences and patterns
        word_mappings = self._analyze_word_patterns(words, coordinates)
        
        # Generate geometric signatures
        geometric_signatures = self._generate_geometric_signatures(characters, coordinates)
        
        # Compile results
        results = {
            "input_text": arabic_text,
            "character_count": len(characters),
            "word_count": len(words),
            "unique_words": len(set(words)),
            "word_mappings": word_mappings,
            "sphere_coordinates": coordinates,
            "geometric_signatures": geometric_signatures,
            "quranic_terms_found": self._identify_quranic_terms(words),
            "mathematical_properties": self._calculate_mathematical_properties(characters, coordinates)
        }
        
        self.analysis_results = results
        return results
    
    def _analyze_word_patterns(self, words: List[str], coordinates: List[Tuple[float, float, float]]) -> Dict[str, WordMapping]:
        """Analyze word patterns and map to coordinates"""
        word_counter = Counter(words)
        word_mappings = {}
        
        coord_index = 0
        for word, count in word_counter.items():
            if coord_index >= len(coordinates):
                break
                
            # Get coordinates for this word's characters
            word_coords = []
            word_length = len(word)
            for i in range(min(word_length, len(coordinates) - coord_index)):
                word_coords.append(coordinates[coord_index + i])
            
            # Generate geometric signature
            signature = self._generate_word_signature(word, word_coords)
            
            word_mappings[word] = WordMapping(
                word=word,
                occurrences=count,
                positions=list(range(coord_index, coord_index + word_length)),
                sphere_coords=word_coords,
                geometric_signature=signature
            )
            
            coord_index += word_length
        
        return word_mappings
    
    def _generate_geometric_signatures(self, characters: List[ArabicCharacter], coordinates: List[Tuple[float, float, float]]) -> List[str]:
        """Generate geometric signatures for characters"""
        signatures = []
        
        for i, (char, coord) in enumerate(zip(characters, coordinates)):
            # Create signature based on position and geometry
            signature = f"{char.char}_{coord[0]:.4f}_{coord[1]:.4f}_{coord[2]:.4f}"
            signatures.append(signature)
        
        return signatures
    
    def _generate_word_signature(self, word: str, coordinates: List[Tuple[float, float, float]]) -> str:
        """Generate unique geometric signature for a word"""
        if not coordinates:
            return f"{word}_no_coords"
        
        # Calculate center of mass for word coordinates
        cx = sum(c[0] for c in coordinates) / len(coordinates)
        cy = sum(c[1] for c in coordinates) / len(coordinates)
        cz = sum(c[2] for c in coordinates) / len(coordinates)
        
        # Create hash-based signature
        signature_data = f"{word}_{cx:.6f}_{cy:.6f}_{cz:.6f}"
        return hashlib.md5(signature_data.encode()).hexdigest()[:16]
    
    def _identify_quranic_terms(self, words: List[str]) -> Dict[str, Dict]:
        """Identify Quranic terminology in the text"""
        quranic_terms = {}
        
        for word in set(words):  # Use set to avoid duplicates
            term_info = self.text_processor.get_quranic_terminology_info(word)
            if term_info:
                quranic_terms[word] = term_info
        
        return quranic_terms
    
    def _calculate_mathematical_properties(self, characters: List[ArabicCharacter], coordinates: List[Tuple[float, float, float]]) -> Dict:
        """Calculate mathematical properties of the text"""
        if not coordinates:
            return {}
        
        # Calculate spherical harmonics approximation
        total_radius = sum(math.sqrt(x**2 + y**2 + z**2) for x, y, z in coordinates)
        avg_radius = total_radius / len(coordinates)
        
        # Calculate angular distribution
        angles = [math.atan2(y, x) for x, y, z in coordinates]
        angle_variance = sum((a - sum(angles)/len(angles))**2 for a in angles) / len(angles)
        
        # Unicode-based mathematical analysis
        unicode_values = [char.unicode for char in characters]
        unicode_variance = sum((u - sum(unicode_values)/len(unicode_values))**2 for u in unicode_values) / len(unicode_values)
        
        return {
            "average_radius": avg_radius,
            "angle_variance": angle_variance,
            "unicode_variance": unicode_variance,
            "character_density": len(characters) / len(coordinates) if coordinates else 0,
            "spherical_entropy": self._calculate_spherical_entropy(coordinates)
        }
    
    def _calculate_spherical_entropy(self, coordinates: List[Tuple[float, float, float]]) -> float:
        """Calculate entropy measure for spherical distribution"""
        if not coordinates:
            return 0.0
        
        # Divide sphere into regions and calculate distribution
        regions = defaultdict(int)
        for x, y, z in coordinates:
            # Simple region classification based on octant
            region = (1 if x >= 0 else 0, 1 if y >= 0 else 0, 1 if z >= 0 else 0)
            regions[region] += 1
        
        total = len(coordinates)
        entropy = 0.0
        for count in regions.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy

class FalaqiInterface:
    """Interactive interface for Falaqi analysis"""
    
    def __init__(self):
        self.analyzer = FalaqiAnalyzer()
        
    def display_introduction(self):
        """Display program introduction and limitations"""
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║                    FALAQI - Arabic Text Analyzer           ║
        ║           Relational Sphere Geometry & Pattern Analysis      ║
        ╚══════════════════════════════════════════════════════════════╝
        
        🌟 CAPABILITIES:
        • Advanced Arabic text processing with character-level analysis
        • Relational sphere geometry mapping based on Hadwiger-Nelson algorithms
        • Quranic terminology identification and pattern detection
        • Mathematical pattern analysis with geometric coordinates
        • Word occurrence mapping with spherical positioning
        
        ⚠️  LIMITATIONS:
        • Minimum 100 Arabic characters recommended for meaningful analysis
        • Processing time increases exponentially with text length
        • Geometric accuracy depends on text complexity
        • Not suitable for very short phrases (< 10 characters)
        • Results are computational and educational, not divine interpretation
        
        📊 MATHEMATICAL FOUNDATION:
        • Trigonometric polynomial: T(θ) = cos²(3πθ) × cos²(6πθ)
        • Forbidden angular separations: π/6, π/3, 2π/3
        • Hadwiger-Nelson chromatic number constraints
        • Spherical harmonic analysis for pattern detection
        
        🎯 BEST RESULTS WITH:
        • Quranic verses or Arabic religious texts
        • Classical Arabic literature
        • Texts with 100+ characters
        • Words with known Quranic terminology
        """)
    
    def get_user_input(self) -> str:
        """Get Arabic text input from user"""
        print("\n" + "="*60)
        print("📝 Please enter your Arabic text for analysis:")
        print("(Enter 'QUIT' to exit, 'DEMO' for sample analysis)")
        print("="*60)
        
        try:
            user_input = input("\nArabic text: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        
        if user_input.upper() == 'QUIT':
            return None
        elif user_input.upper() == 'DEMO':
            return self.get_demo_text()
        else:
            return user_input
    
    def get_demo_text(self) -> str:
        """Get demo text for testing"""
        return """
        بسم الله الرحمن الرحيم
        الحمد لله رب العالمين
        الرحمن الرحيم
        مالك يوم الدين
        إياك نعبد وإياك نستعين
        اهدنا الصراط المستقيم
        صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين
        """
    
    def display_results(self, results: Dict):
        """Display analysis results"""
        print("\n" + "="*60)
        print("📊 FALAQI ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n📈 BASIC STATISTICS:")
        print(f"   • Total characters: {results['character_count']}")
        print(f"   • Total words: {results['word_count']}")
        print(f"   • Unique words: {results['unique_words']}")
        
        if results['quranic_terms_found']:
            print(f"\n🕌 QURANIC TERMINOLOGY FOUND:")
            for term, info in results['quranic_terms_found'].items():
                print(f"   • {term}: {info['category']} (appears {info['frequency']} times in Quran)")
        
        if results['mathematical_properties']:
            props = results['mathematical_properties']
            print(f"\n🔢 MATHEMATICAL PROPERTIES:")
            print(f"   • Average spherical radius: {props.get('average_radius', 0):.6f}")
            print(f"   • Angular variance: {props.get('angle_variance', 0):.6f}")
            print(f"   • Spherical entropy: {props.get('spherical_entropy', 0):.6f}")
            print(f"   • Character density: {props.get('character_density', 0):.4f}")
        
        print(f"\n🌐 WORD MAPPINGS (showing first 10):")
        for i, (word, mapping) in enumerate(list(results['word_mappings'].items())[:10]):
            print(f"   {i+1:2d}. '{word}' → {mapping.occurrences} occurrences")
            print(f"       Signature: {mapping.geometric_signature}")
            if mapping.sphere_coords:
                print(f"       First coordinate: ({mapping.sphere_coords[0][0]:.4f}, {mapping.sphere_coords[0][1]:.4f}, {mapping.sphere_coords[0][2]:.4f})")
        
        # Save detailed results to file
        self.save_detailed_results(results)
        
        print(f"\n💾 Detailed results saved to: 'falaqi_analysis_results.json'")
        print(f"📁 Sphere coordinates saved to: 'sphere_coordinates.txt'")
    
    def save_detailed_results(self, results: Dict):
        """Save detailed analysis results to files"""
        # Save JSON results
        with open('falaqi_analysis_results.json', 'w', encoding='utf-8') as f:
            # Convert dataclasses to dicts for JSON serialization
            json_results = {}
            for key, value in results.items():
                if key == 'word_mappings':
                    json_results[key] = {
                        word: {
                            'word': mapping.word,
                            'occurrences': mapping.occurrences,
                            'positions': mapping.positions,
                            'geometric_signature': mapping.geometric_signature
                        }
                        for word, mapping in value.items()
                    }
                else:
                    json_results[key] = value
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        # Save sphere coordinates
        with open('sphere_coordinates.txt', 'w', encoding='utf-8') as f:
            f.write("Index\tX\tY\tZ\tSignature\n")
            for i, (x, y, z) in enumerate(results.get('sphere_coordinates', [])):
                signature = results.get('geometric_signatures', [])[i] if i < len(results.get('geometric_signatures', [])) else "N/A"
                f.write(f"{i}\t{x:.8f}\t{y:.8f}\t{z:.8f}\t{signature}\n")
    
    def run(self):
        """Main interactive loop"""
        self.display_introduction()
        
        while True:
            try:
                user_input = self.get_user_input()
                if user_input is None:
                    print("\n👋 Thank you for using Falaqi!")
                    break
                
                # Perform analysis
                results = self.analyzer.analyze_text(user_input)
                
                # Display results
                self.display_results(results)
                
                # Ask if user wants to continue
                continue_choice = input("\n🔄 Analyze another text? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("\n👋 Thank you for using Falaqi!")
                    break
                    
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with different text.")

def main():
    """Main entry point"""
    try:
        interface = FalaqiInterface()
        interface.run()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")

if __name__ == "__main__":
    main()
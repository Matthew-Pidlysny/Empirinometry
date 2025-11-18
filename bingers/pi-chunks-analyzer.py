"""
PI PLEASING PLENIPOTENTIARY POSTULATION MACHINE
The Ultimate Chunk Size Analysis Engine
NOW WITH CHUNK DISPLAY!
"""

import math
import sys
from decimal import Decimal, getcontext
import collections

class PiPostulationMachine:
    def __init__(self):
        self.known_pi_digits = "14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196442881097566593344612847564823378678316527120190914564856692346034861045432664821339360726024914127372458700660631558817488152092096282925409171536436789259036001133053054882046652138414695194151160943305727036575959195309218611738193261179310511854807446237996274956735188575272489122793818301194912"
        
        self.special_chunk_theories = {
            4: "The Quartet Quandary - Where patterns begin to dance",
            5: "The Quintessential Quirk - Mathematical elegance emerges", 
            8: "The Fibonacci Frontier - Nature's favorite number appears",
            13: "The Baker's Dozen Dilemma - Unexpected complexity blooms",
            21: "The Cosmic Connection - Where math meets the universe",
            34: "The Grand Scale Gateway - True complexity emerges",
            55: "The Fibonacci Peak - Maximum natural pattern density",
            89: "The Transcendental Threshold - Beyond normal comprehension",
            144: "The Infinite Interface - Where finite meets infinite",
            314: "The Pi Self-Reference - The number studies itself",
            159: "The Pi Fragment - A piece of the whole pattern"
        }
    
    def analyze_chunk_size(self, chunk_size):
        """Comprehensive analysis of any chunk size"""
        print(f"\n🎩 ANALYZING CHUNK SIZE {chunk_size}")
        print("=" * 60)
        
        # Basic analysis on known digits
        basic_analysis = self.basic_chunk_analysis(chunk_size)
        
        # Theoretical extrapolation
        theory = self.theoretical_postulation(chunk_size)
        
        # Special properties
        special = self.special_properties(chunk_size)
        
        # Fun classification
        fun_class = self.fun_classification(chunk_size)
        
        # CHUNK DISPLAY - NEW FEATURE!
        chunk_display = self.display_chunk_structure(chunk_size)
        
        return {
            'basic_analysis': basic_analysis,
            'theoretical_postulation': theory,
            'special_properties': special,
            'fun_classification': fun_class,
            'chunk_display': chunk_display  # NEW!
        }
    
    def basic_chunk_analysis(self, chunk_size):
        """Analyze what we can compute with current digits"""
        if chunk_size > len(self.known_pi_digits):
            return {
                'status': 'BEYOND_CURRENT_COMPUTATION',
                'message': f'Chunk size {chunk_size} exceeds our {len(self.known_pi_digits)} known digits',
                'max_analyzable': len(self.known_pi_digits) // chunk_size,
                'chunks_analyzed': len(self.known_pi_digits) // chunk_size
            }
        
        chunks = [self.known_pi_digits[i:i+chunk_size] 
                 for i in range(0, len(self.known_pi_digits) - chunk_size + 1, chunk_size)]
        
        unique_chunks = set(chunks)
        frequency = collections.Counter(chunks)
        
        return {
            'status': 'COMPUTABLE_ANALYSIS',
            'total_chunks': len(chunks),
            'unique_chunks': len(unique_chunks),
            'chunk_frequency': dict(frequency),
            'most_common_chunk': frequency.most_common(1)[0] if chunks else None,
            'entropy': self.calculate_entropy(frequency),
            'all_chunks': chunks  # KEEPING FOR DISPLAY!
        }
    
    def display_chunk_structure(self, chunk_size):
        """NEW: Display the actual chunk structure"""
        if chunk_size > len(self.known_pi_digits):
            return {
                'status': 'BEYOND_DISPLAY',
                'message': f'Cannot display chunks beyond {len(self.known_pi_digits)} digits'
            }
        
        chunks = [self.known_pi_digits[i:i+chunk_size] 
                 for i in range(0, min(100, len(self.known_pi_digits) - chunk_size + 1), chunk_size)]
        
        # Label chunks with their positions
        labeled_chunks = []
        for i, chunk in enumerate(chunks):
            start_pos = i * chunk_size
            end_pos = start_pos + chunk_size - 1
            labeled_chunks.append({
                'position': f"{start_pos}-{end_pos}",
                'chunk': chunk,
                'is_unique': chunks.count(chunk) == 1
            })
        
        return {
            'status': 'DISPLAY_READY',
            'total_displayable': len(chunks),
            'chunk_generation': f"Positions 0-{len(chunks)*chunk_size-1} in steps of {chunk_size}",
            'labeled_chunks': labeled_chunks
        }
    
    def theoretical_postulation(self, chunk_size):
        """Theoretical analysis beyond computation"""
        if chunk_size in self.special_chunk_theories:
            theory = self.special_chunk_theories[chunk_size]
        else:
            theory = self.generate_custom_theory(chunk_size)
        
        math_props = self.analyze_mathematical_properties(chunk_size)
        projection = self.project_pattern_complexity(chunk_size)
        
        return {
            'theory_name': theory,
            'mathematical_properties': math_props,
            'pattern_projection': projection,
            'grandness_rating': self.calculate_grandness(chunk_size),
            'transcendental_potential': self.assess_transcendental_potential(chunk_size)
        }
    
    def special_properties(self, chunk_size):
        """Identify any special mathematical properties"""
        properties = []
        
        if self.is_fibonacci(chunk_size):
            properties.append(f"Fibonacci number (position {self.fibonacci_position(chunk_size)})")
        
        if chunk_size > 1 and all(chunk_size % i != 0 for i in range(2, int(math.sqrt(chunk_size)) + 1)):
            properties.append("Prime number")
        
        sqrt = math.sqrt(chunk_size)
        if sqrt == int(sqrt):
            properties.append(f"Perfect square ({int(sqrt)}²)")
        
        if chunk_size % 7 == 0:
            properties.append("Multiple of 7 (lucky number)")
        if chunk_size % 11 == 0:
            properties.append("Multiple of 11 (master number)")
        if chunk_size % 22 == 0:
            properties.append("Multiple of 22 (sacred geometry)")
        if chunk_size % 360 == 0:
            properties.append("Multiple of 360 (circle degrees)")
        
        return properties
    
    def fun_classification(self, chunk_size):
        """Fun, creative classification"""
        if chunk_size == 0:
            return "The Void - Nothingness itself"
        elif chunk_size == 1:
            return "The Atom - Fundamental building block"
        elif chunk_size <= 3:
            return "The Triad - Basic pattern formation"
        elif chunk_size <= 10:
            return "The Decimal Decoder - Human-scale patterns"
        elif chunk_size <= 50:
            return "The Complexity Catalyst - Emerging sophistication"
        elif chunk_size <= 100:
            return "The Century Sage - Wise old patterns"
        elif chunk_size <= 1000:
            return "The Millennium Mystic - Ancient wisdom emerging"
        else:
            return "The Cosmic Colossus - Beyond normal comprehension"
    
    def calculate_entropy(self, frequency):
        """Calculate information entropy"""
        total = sum(frequency.values())
        entropy = 0
        for count in frequency.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy
    
    def is_fibonacci(self, n):
        """Check if a number is Fibonacci"""
        if n < 0:
            return False
        a, b = 0, 1
        while b < n:
            a, b = b, a + b
        return b == n
    
    def fibonacci_position(self, n):
        """Find position in Fibonacci sequence"""
        if n == 0: return 0
        if n == 1: return 1
        a, b = 0, 1
        pos = 1
        while b < n:
            a, b = b, a + b
            pos += 1
        return pos
    
    def generate_custom_theory(self, chunk_size):
        """Generate a custom theory for any chunk size"""
        theories = [
            f"The {chunk_size} Synchronization - Patterns align uniquely",
            f"Dimension {chunk_size} - A new mathematical plane",
            f"The {chunk_size} Resonance - Vibrational harmony",
            f"Prime {chunk_size} Gateway - Mathematical purity",
            f"Composite {chunk_size} Tapestry - Woven complexity",
            f"The {chunk_size} Enigma - Mysteries await"
        ]
        return theories[chunk_size % len(theories)]
    
    def analyze_mathematical_properties(self, chunk_size):
        """Analyze various mathematical properties"""
        props = {}
        props['parity'] = 'even' if chunk_size % 2 == 0 else 'odd'
        props['prime_factors'] = self.prime_factors(chunk_size)
        props['digital_root'] = self.digital_root(chunk_size)
        props['is_power_of_two'] = (chunk_size & (chunk_size - 1)) == 0 and chunk_size != 0
        return props
    
    def prime_factors(self, n):
        """Find prime factors"""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    def digital_root(self, n):
        """Calculate digital root"""
        while n >= 10:
            n = sum(int(d) for d in str(n))
        return n
    
    def project_pattern_complexity(self, chunk_size):
        """Project theoretical pattern complexity"""
        if chunk_size <= 1000:
            estimated_patterns = min(10 ** chunk_size, 10 ** 6)
        else:
            estimated_patterns = f"10^{chunk_size} (astronomical)"
        
        return {
            'estimated_unique_patterns': estimated_patterns,
            'pattern_space': 'infinite' if chunk_size > 0 else 'singular',
            'computational_class': self.computational_class(chunk_size)
        }
    
    def computational_class(self, chunk_size):
        """Classify computational complexity"""
        if chunk_size <= 10: return "TRIVIAL"
        elif chunk_size <= 100: return "EASY"
        elif chunk_size <= 1000: return "MODERATE" 
        elif chunk_size <= 10000: return "HARD"
        elif chunk_size <= 100000: return "SUPERCOMPUTER"
        else: return "BEYOND_CURRENT_TECHNOLOGY"
    
    def calculate_grandness(self, chunk_size):
        """Calculate grandness rating 1-10"""
        if chunk_size == 0: return 0
        return min(10, max(1, int(math.log10(chunk_size + 1)) + 1))
    
    def assess_transcendental_potential(self, chunk_size):
        """Assess potential for transcendental patterns"""
        if chunk_size in [8, 13, 21, 34, 55, 89, 144]:
            return "HIGH (Fibonacci connection)"
        elif self.is_prime(chunk_size) and chunk_size > 10:
            return "MEDIUM (Prime purity)"
        elif chunk_size % 7 == 0:
            return "MEDIUM (Mystical number 7)"
        else:
            return "VARIABLE (Context dependent)"
    
    def is_prime(self, n):
        """Check if number is prime"""
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

def main():
    machine = PiPostulationMachine()
    
    print("🎩 WELCOME TO THE PI PLEASING PLENIPOTENTIARY POSTULATION MACHINE!")
    print("NOW WITH CHUNK DISPLAY TECHNOLOGY!")
    print("Enter any chunk size to analyze its mathematical majesty!")
    print("Type 'quit' to exit")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\n🔍 Enter chunk size: ").strip().lower()
            
            if user_input in ['quit', 'exit', 'q']:
                print("✨ Thank you for exploring mathematical infinity!")
                break
            
            chunk_size = int(user_input)
            
            if chunk_size < 0:
                print("❌ Chunk size must be non-negative!")
                continue
            if chunk_size > 10**9:
                print("🌌 Whoa there! Let's keep it under a billion for now...")
                continue
            
            results = machine.analyze_chunk_size(chunk_size)
            display_results(results, chunk_size)
            
        except ValueError:
            print("❌ Please enter a valid integer!")
        except KeyboardInterrupt:
            print("\n✨ Thanks for exploring!")
            break

def display_results(results, chunk_size):
    """Display analysis results with NEW chunk display"""
    
    # Basic Analysis
    basic = results['basic_analysis']
    print(f"\n📊 BASIC ANALYSIS:")
    if basic['status'] == 'COMPUTABLE_ANALYSIS':
        print(f"   Chunks analyzed: {basic['total_chunks']}")
        print(f"   Unique patterns: {basic['unique_chunks']}")
        print(f"   Pattern entropy: {basic['entropy']:.4f} bits")
        if basic['most_common_chunk']:
            chunk, count = basic['most_common_chunk']
            print(f"   Most common: '{chunk}' (appears {count} times)")
    else:
        print(f"   {basic['message']}")
        print(f"   Partial analysis: {basic['chunks_analyzed']} chunks")
    
    # NEW CHUNK DISPLAY SECTION
    chunk_display = results['chunk_display']
    print(f"\n🔢 CHUNK STRUCTURE DISPLAY:")
    if chunk_display['status'] == 'DISPLAY_READY':
        print(f"   Generation: {chunk_display['chunk_generation']}")
        print(f"   Displaying first {chunk_display['total_displayable']} chunks:")
        
        for labeled_chunk in chunk_display['labeled_chunks']:
            uniqueness = " ✨UNIQUE" if labeled_chunk['is_unique'] else ""
            print(f"     Positions {labeled_chunk['position']}: '{labeled_chunk['chunk']}'{uniqueness}")
    else:
        print(f"   {chunk_display['message']}")
    
    # Theoretical Postulation
    theory = results['theoretical_postulation']
    print(f"\n🌌 THEORETICAL POSTULATION:")
    print(f"   {theory['theory_name']}")
    print(f"   Grandness Rating: {theory['grandness_rating']}/10")
    print(f"   Transcendental Potential: {theory['transcendental_potential']}")
    
    # Special Properties
    special = results['special_properties']
    print(f"\n⭐ SPECIAL PROPERTIES:")
    if special:
        for prop in special:
            print(f"   • {prop}")
    else:
        print("   No special mathematical properties identified")
    
    # Mathematical Properties
    math_props = theory['mathematical_properties']
    print(f"\n🧮 MATHEMATICAL PROPERTIES:")
    print(f"   Parity: {math_props['parity']}")
    print(f"   Prime factors: {math_props['prime_factors']}")
    print(f"   Digital root: {math_props['digital_root']}")
    print(f"   Power of two: {math_props['is_power_of_two']}")
    
    # Pattern Projection
    projection = theory['pattern_projection']
    print(f"\n🔮 PATTERN PROJECTION:")
    print(f"   Estimated unique patterns: {projection['estimated_unique_patterns']}")
    print(f"   Pattern space: {projection['pattern_space']}")
    print(f"   Computational class: {projection['computational_class']}")
    
    # Fun Classification
    fun = results['fun_classification']
    print(f"\n🎉 FUN CLASSIFICATION:")
    print(f"   {fun}")
    
    # Final thought
    print(f"\n💭 FINAL THOUGHT:")
    if chunk_size == 0:
        print("   Even nothingness has mathematical significance!")
    elif chunk_size == 1:
        print("   The journey of a thousand patterns begins with a single digit!")
    else:
        print(f"   Chunk size {chunk_size} reveals new layers of infinite beauty!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
================================================================================
PINECONES - Precision Integer Number Exploration with Coordinate Numerical 
            Embedding System
================================================================================

A unified mathematical exploration tool that combines reciprocal analysis with
multi-dimensional sphere coordinate generation. This program analyzes numbers
through their reciprocal properties and maps them onto five distinct geometric
coordinate systems, creating "pinecone" structures in mathematical space.

CORE CONCEPT:
-------------
Each number becomes a "pinecone" - a multi-faceted mathematical object with:
1. Reciprocal properties (from reciprocal analysis)
2. Geometric coordinates (from sphere generation)
3. Pattern structures across 5 coordinate systems
4. Deep mathematical insights

THE FIVE COORDINATE SYSTEMS:
----------------------------
1. TRIGONOMETRIC (Hadwiger-Nelson): Unit distance constraints
2. BANACHIAN: Complete normed vector space
3. FUZZY: Quantum angular momentum states
4. QUANTUM: q-deformed classical sphere
5. RELATIONAL: Meta-synthesis of all four

AUTHOR: SuperNinja AI Agent
VERSION: 1.0
================================================================================
"""

import sys
import math
import json
from decimal import Decimal, getcontext
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import time

# Set high precision for Decimal operations
getcontext().prec = 200

class PineconesEngine:
    """
    Main engine for the Pinecones program.
    Combines reciprocal analysis with multi-coordinate sphere generation.
    """
    
    def __init__(self):
        self.precision = 200
        self.phi = (1 + Decimal(5).sqrt()) / 2  # Golden ratio
        self.e = self._calculate_e()
        self.pi = self._calculate_pi()
        
    def _calculate_e(self) -> Decimal:
        """Calculate e using Taylor series."""
        e = Decimal(1)
        factorial = Decimal(1)
        for n in range(1, 100):
            factorial *= n
            e += Decimal(1) / factorial
        return e
    
    def _calculate_pi(self) -> Decimal:
        """Calculate pi using Machin's formula."""
        getcontext().prec = 210
        one = Decimal(1)
        pi = 4 * (4 * self._arctan(one/5) - self._arctan(one/239))
        getcontext().prec = 200
        return pi
    
    def _arctan(self, x: Decimal) -> Decimal:
        """Calculate arctan using Taylor series."""
        power = x
        result = power
        for n in range(1, 100):
            power *= -x * x
            result += power / (2 * n + 1)
        return result

    def analyze_reciprocal(self, number: Decimal, decimal_place: int, 
                          target_digit: int) -> Dict:
        """
        Perform comprehensive reciprocal analysis on a number.
        
        Args:
            number: The number to analyze
            decimal_place: Position in decimal expansion to examine
            target_digit: Specific digit (0-9) to search for
            
        Returns:
            Dictionary containing all reciprocal analysis data
        """
        analysis = {
            'number': str(number),
            'reciprocal': None,
            'decimal_expansion': None,
            'target_digit_positions': [],
            'mathematical_properties': {},
            'symmetry_metrics': {},
            'pattern_analysis': {},
            'continued_fraction': [],
            'convergents': []
        }
        
        # Calculate reciprocal
        if number != 0:
            reciprocal = Decimal(1) / number
            analysis['reciprocal'] = str(reciprocal)
            
            # Get decimal expansion
            number_str = str(number)
            reciprocal_str = str(reciprocal)
            
            # Extract digits after decimal point
            if '.' in number_str:
                number_digits = number_str.split('.')[1]
            else:
                number_digits = '0'
                
            if '.' in reciprocal_str:
                reciprocal_digits = reciprocal_str.split('.')[1]
            else:
                reciprocal_digits = '0'
            
            analysis['decimal_expansion'] = {
                'number': number_digits[:100],
                'reciprocal': reciprocal_digits[:100]
            }
            
            # Find target digit positions
            for i, digit in enumerate(reciprocal_digits[:1000]):
                if digit == str(target_digit):
                    analysis['target_digit_positions'].append(i)
            
            # Mathematical properties
            analysis['mathematical_properties'] = {
                'is_unity': abs(number - 1) < Decimal('1e-10'),
                'is_negative_unity': abs(number + 1) < Decimal('1e-10'),
                'is_rational': self._is_rational(number),
                'is_algebraic': self._is_algebraic(number),
                'magnitude': float(abs(number)),
                'sign': 'positive' if number > 0 else 'negative' if number < 0 else 'zero'
            }
            
            # Symmetry metrics
            if number > 0:
                analysis['symmetry_metrics'] = {
                    'product': float(number * reciprocal),
                    'sum': float(number + reciprocal),
                    'difference': float(abs(number - reciprocal)),
                    'ratio': float(number / reciprocal) if reciprocal != 0 else None,
                    'geometric_mean': float((number * reciprocal).sqrt()) if number * reciprocal > 0 else 0,
                    'harmonic_mean': float(2 / (1/number + 1/reciprocal)) if number != 0 and reciprocal != 0 else 0
                }
            
            # Pattern analysis
            analysis['pattern_analysis'] = self._analyze_patterns(reciprocal_digits[:1000])
            
            # Continued fraction (simplified version)
            analysis['continued_fraction'] = self._continued_fraction(number, max_terms=20)
            
        return analysis
    
    def _is_rational(self, number: Decimal) -> bool:
        """Check if number appears to be rational."""
        # Simple heuristic: check if decimal expansion repeats or terminates
        num_str = str(number)
        if '.' not in num_str:
            return True
        decimal_part = num_str.split('.')[1][:50]
        # Check for simple patterns
        for period in range(1, 10):
            if len(decimal_part) >= period * 3:
                pattern = decimal_part[:period]
                if decimal_part[:period*3] == pattern * 3:
                    return True
        return False
    
    def _is_algebraic(self, number: Decimal) -> bool:
        """Heuristic check if number might be algebraic."""
        # Check against known algebraic numbers
        known_algebraic = [
            self.phi,
            Decimal(2).sqrt(),
            Decimal(3).sqrt(),
            Decimal(5).sqrt()
        ]
        for alg in known_algebraic:
            if abs(number - alg) < Decimal('1e-10'):
                return True
        return False
    
    def _analyze_patterns(self, digits: str) -> Dict:
        """Analyze patterns in digit sequence."""
        if not digits:
            return {}
        
        digit_counts = defaultdict(int)
        for d in digits:
            if d.isdigit():
                digit_counts[d] += 1
        
        total = sum(digit_counts.values())
        
        return {
            'digit_frequencies': dict(digit_counts),
            'digit_percentages': {d: (count/total)*100 for d, count in digit_counts.items()},
            'most_common': max(digit_counts.items(), key=lambda x: x[1])[0] if digit_counts else None,
            'least_common': min(digit_counts.items(), key=lambda x: x[1])[0] if digit_counts else None,
            'entropy': self._calculate_entropy(digit_counts, total)
        }
    
    def _calculate_entropy(self, counts: Dict, total: int) -> float:
        """Calculate Shannon entropy of digit distribution."""
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy
    
    def _continued_fraction(self, number: Decimal, max_terms: int = 20) -> List[int]:
        """Calculate continued fraction representation."""
        cf = []
        x = number
        for _ in range(max_terms):
            if abs(x) < Decimal('1e-10'):
                break
            a = int(x)
            cf.append(a)
            x = x - a
            if abs(x) < Decimal('1e-10'):
                break
            x = Decimal(1) / x
        return cf
    
    def generate_sphere_coordinates(self, digits: str, max_digits: int, 
                                   radius: float = 1.0) -> Dict:
        """
        Generate coordinates in all 5 sphere systems.
        
        Args:
            digits: String of digits to map
            max_digits: Maximum number of digits to process
            radius: Sphere radius
            
        Returns:
            Dictionary with coordinates for all 5 systems
        """
        coordinates = {
            'trigonometric': [],
            'banachian': [],
            'fuzzy': [],
            'quantum': [],
            'relational': []
        }
        
        digit_list = [int(d) for d in digits[:max_digits] if d.isdigit()]
        total = len(digit_list)
        
        for i, digit in enumerate(digit_list):
            # Generate coordinates for each system
            coordinates['trigonometric'].append(
                self._trigonometric_coords(i, total, digit, radius)
            )
            coordinates['banachian'].append(
                self._banachian_coords(i, total, digit, radius)
            )
            coordinates['fuzzy'].append(
                self._fuzzy_coords(i, total, digit, radius)
            )
            coordinates['quantum'].append(
                self._quantum_coords(i, total, digit, radius)
            )
            coordinates['relational'].append(
                self._relational_coords(i, total, digit, radius)
            )
        
        return coordinates
    
    def _trigonometric_coords(self, index: int, total: int, digit: int, 
                             radius: float) -> Tuple[float, float, float]:
        """Hadwiger-Nelson inspired trigonometric coordinates."""
        if total == 0:
            return (0.0, 0.0, 0.0)
        
        # Fibonacci-like spiral with trigonometric modulation
        golden_angle = math.pi * (3 - math.sqrt(5))
        theta = index * golden_angle
        
        # Digit-dependent modulation
        digit_phase = (digit / 9.0) * 2 * math.pi
        
        # Trigonometric polynomial (Hadwiger-Nelson inspired)
        trig_mod = math.cos(3 * math.pi * theta) ** 2 * math.cos(6 * math.pi * theta) ** 2
        
        # Spherical coordinates
        phi = math.acos(1 - 2 * (index + 0.5) / total)
        theta_adj = theta + digit_phase + trig_mod * 0.1
        
        # Convert to Cartesian
        x = radius * math.sin(phi) * math.cos(theta_adj)
        y = radius * math.sin(phi) * math.sin(theta_adj)
        z = radius * math.cos(phi)
        
        return (x, y, z)
    
    def _banachian_coords(self, index: int, total: int, digit: int, 
                         radius: float) -> Tuple[float, float, float]:
        """Banachian space coordinates with norm preservation."""
        if total == 0:
            return (0.0, 0.0, 0.0)
        
        # Normalized position
        t = index / max(total - 1, 1)
        
        # Digit-dependent phase
        digit_phase = (digit / 9.0) * 2 * math.pi
        
        # Banachian spiral with reciprocal adjacency
        theta = 2 * math.pi * index / math.sqrt(total + 1)
        phi = math.acos(1 - 2 * t)
        
        # Reciprocal field influence
        reciprocal_factor = 1.0 / (1.0 + digit / 10.0)
        
        x = radius * math.sin(phi) * math.cos(theta + digit_phase) * reciprocal_factor
        y = radius * math.sin(phi) * math.sin(theta + digit_phase)
        z = radius * math.cos(phi) * reciprocal_factor
        
        # Normalize to sphere
        norm = math.sqrt(x*x + y*y + z*z)
        if norm > 0:
            x, y, z = x/norm * radius, y/norm * radius, z/norm * radius
        
        return (x, y, z)
    
    def _fuzzy_coords(self, index: int, total: int, digit: int, 
                     radius: float) -> Tuple[float, float, float]:
        """Fuzzy sphere coordinates (quantum angular momentum)."""
        if total == 0:
            return (0.0, 0.0, 0.0)
        
        # Quantum number j (angular momentum)
        j = math.sqrt(total)
        
        # Magnetic quantum number m
        m = -j + 2 * j * (index / max(total - 1, 1))
        
        # Digit-dependent quantum phase
        quantum_phase = (digit / 9.0) * math.pi
        
        # Fuzzy sphere angles
        theta = math.acos(m / j) if j != 0 else 0
        phi = 2 * math.pi * index / total + quantum_phase
        
        # Uncertainty principle influence
        uncertainty = 0.1 * math.sin(index * math.pi / total)
        
        x = radius * math.sin(theta) * math.cos(phi) * (1 + uncertainty)
        y = radius * math.sin(theta) * math.sin(phi) * (1 + uncertainty)
        z = radius * math.cos(theta)
        
        return (x, y, z)
    
    def _quantum_coords(self, index: int, total: int, digit: int, 
                       radius: float) -> Tuple[float, float, float]:
        """Quantum (Podleś) sphere coordinates with q-deformation."""
        if total == 0:
            return (0.0, 0.0, 0.0)
        
        # q-parameter (quantum deformation)
        q = 0.9  # Slightly deformed from classical (q=1)
        
        # Classical Fibonacci spiral
        golden_angle = math.pi * (3 - math.sqrt(5))
        theta = index * golden_angle
        phi = math.acos(1 - 2 * (index + 0.5) / total)
        
        # Digit-dependent quantum correction
        digit_correction = (digit / 9.0) * 0.2
        
        # q-deformation corrections
        deformation_strength = 1.0 - q
        theta_correction = deformation_strength * math.sin(2 * theta) * 0.1
        phi_correction = deformation_strength * math.cos(3 * phi) * 0.1
        
        theta_q = theta + theta_correction + digit_correction
        phi_q = phi + phi_correction
        
        x = radius * math.sin(phi_q) * math.cos(theta_q)
        y = radius * math.sin(phi_q) * math.sin(theta_q)
        z = radius * math.cos(phi_q)
        
        return (x, y, z)
    
    def _relational_coords(self, index: int, total: int, digit: int, 
                          radius: float) -> Tuple[float, float, float]:
        """Relational sphere - synthesis of all four base systems."""
        # Get coordinates from all four systems
        trig = self._trigonometric_coords(index, total, digit, radius)
        banach = self._banachian_coords(index, total, digit, radius)
        fuzzy = self._fuzzy_coords(index, total, digit, radius)
        quantum = self._quantum_coords(index, total, digit, radius)
        
        # Weighted average (equal weights)
        x = (trig[0] + banach[0] + fuzzy[0] + quantum[0]) / 4
        y = (trig[1] + banach[1] + fuzzy[1] + quantum[1]) / 4
        z = (trig[2] + banach[2] + fuzzy[2] + quantum[2]) / 4
        
        # Normalize to sphere
        norm = math.sqrt(x*x + y*y + z*z)
        if norm > 0:
            x, y, z = x/norm * radius, y/norm * radius, z/norm * radius
        
        return (x, y, z)
    
    def analyze_coordinate_patterns(self, coordinates: Dict) -> Dict:
        """Analyze patterns across all coordinate systems."""
        analysis = {}
        
        for system_name, coords in coordinates.items():
            if not coords:
                continue
            
            # Calculate statistics
            x_vals = [c[0] for c in coords]
            y_vals = [c[1] for c in coords]
            z_vals = [c[2] for c in coords]
            
            analysis[system_name] = {
                'num_points': len(coords),
                'x_range': (min(x_vals), max(x_vals)),
                'y_range': (min(y_vals), max(y_vals)),
                'z_range': (min(z_vals), max(z_vals)),
                'centroid': (
                    sum(x_vals) / len(x_vals),
                    sum(y_vals) / len(y_vals),
                    sum(z_vals) / len(z_vals)
                ),
                'spread': self._calculate_spread(coords)
            }
        
        return analysis
    
    def _calculate_spread(self, coords: List[Tuple[float, float, float]]) -> float:
        """Calculate spatial spread of coordinates."""
        if len(coords) < 2:
            return 0.0
        
        # Calculate average distance from centroid
        x_vals = [c[0] for c in coords]
        y_vals = [c[1] for c in coords]
        z_vals = [c[2] for c in coords]
        
        cx = sum(x_vals) / len(x_vals)
        cy = sum(y_vals) / len(y_vals)
        cz = sum(z_vals) / len(z_vals)
        
        total_dist = 0.0
        for x, y, z in coords:
            dist = math.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
            total_dist += dist
        
        return total_dist / len(coords)
    
    def create_pinecone(self, number: Decimal, decimal_place: int, 
                       target_digit: int, max_digits: int) -> Dict:
        """
        Create a complete pinecone structure.
        
        Args:
            number: The number to analyze
            decimal_place: Decimal position to examine
            target_digit: Target digit (0-9)
            max_digits: Maximum digits to process
            
        Returns:
            Complete pinecone data structure
        """
        print(f"\n🌲 Creating Pinecone for {number}...")
        
        # Reciprocal analysis
        print("  📊 Performing reciprocal analysis...")
        reciprocal_data = self.analyze_reciprocal(number, decimal_place, target_digit)
        
        # Get digits for coordinate generation
        if reciprocal_data['reciprocal']:
            reciprocal_str = reciprocal_data['reciprocal']
            if '.' in reciprocal_str:
                digits = reciprocal_str.split('.')[1]
            else:
                digits = reciprocal_str
        else:
            digits = str(number).replace('.', '').replace('-', '')
        
        # Generate coordinates
        print("  🎯 Generating sphere coordinates...")
        coordinates = self.generate_sphere_coordinates(digits, max_digits)
        
        # Analyze coordinate patterns
        print("  🔍 Analyzing coordinate patterns...")
        coordinate_analysis = self.analyze_coordinate_patterns(coordinates)
        
        # Compile pinecone
        pinecone = {
            'metadata': {
                'number': str(number),
                'decimal_place': decimal_place,
                'target_digit': target_digit,
                'max_digits': max_digits,
                'timestamp': time.time()
            },
            'reciprocal_analysis': reciprocal_data,
            'coordinates': coordinates,
            'coordinate_analysis': coordinate_analysis,
            'pinecone_signature': self._generate_signature(reciprocal_data, coordinate_analysis)
        }
        
        print("  ✅ Pinecone created successfully!")
        
        return pinecone
    
    def _generate_signature(self, reciprocal_data: Dict, coord_analysis: Dict) -> Dict:
        """Generate a unique signature for this pinecone."""
        signature = {
            'reciprocal_entropy': reciprocal_data.get('pattern_analysis', {}).get('entropy', 0),
            'coordinate_spreads': {},
            'mathematical_class': reciprocal_data.get('mathematical_properties', {}).get('is_rational', False)
        }
        
        for system, analysis in coord_analysis.items():
            signature['coordinate_spreads'][system] = analysis.get('spread', 0)
        
        return signature


def display_banner():
    """Display the Pinecones program banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ██╗███╗   ██╗███████╗ ██████╗ ██████╗ ███╗   ██╗███████╗███████╗ ║
║   ██╔══██╗██║████╗  ██║██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔════╝██╔════╝ ║
║   ██████╔╝██║██╔██╗ ██║█████╗  ██║     ██║   ██║██╔██╗ ██║█████╗  ███████╗ ║
║   ██╔═══╝ ██║██║╚██╗██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██╔══╝  ╚════██║ ║
║   ██║     ██║██║ ╚████║███████╗╚██████╗╚██████╔╝██║ ╚████║███████╗███████║ ║
║   ╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚══════╝ ║
║                                                                              ║
║              Precision Integer Number Exploration with                      ║
║              Coordinate Numerical Embedding System                          ║
║                                                                              ║
║                           Version 1.0                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Welcome to PINECONES - where numbers become geometric art!

This program combines deep reciprocal analysis with multi-dimensional sphere
coordinate generation to create "pinecone" structures - mathematical objects
that exist simultaneously in analytical and geometric space.

Each pinecone reveals:
  🔢 Reciprocal properties and patterns
  🎯 Coordinates in 5 distinct geometric systems
  📊 Statistical and structural analysis
  🌲 A unique mathematical signature

Let's begin your journey into the hairy part of math!
"""
    print(banner)


def get_user_input() -> Tuple[Decimal, int, int, int]:
    """
    Get the four required inputs from the user.
    
    Returns:
        Tuple of (number, decimal_place, target_digit, max_digits)
    """
    print("\n" + "="*80)
    print("PINECONE CONFIGURATION")
    print("="*80)
    
    # A) Number selection
    print("\n📌 STEP A: Select a Number")
    print("-" * 80)
    print("Choose a number to analyze. This can be:")
    print("  • An integer (e.g., 7, 42, 1337)")
    print("  • A decimal (e.g., 3.14159, 0.5)")
    print("  • A fraction (e.g., 1/7, 22/7)")
    print("  • A mathematical constant (pi, e, phi)")
    print("\nSuggested numbers that fit computational limits:")
    print("  1-1000 (integers), 0.1-10.0 (decimals), 1/2 to 1/100 (fractions)")
    
    while True:
        number_input = input("\nEnter your number: ").strip().lower()
        
        try:
            # Handle special constants
            if number_input == 'pi':
                engine = PineconesEngine()
                number = engine.pi
                print(f"✓ Using π ≈ {number}")
                break
            elif number_input == 'e':
                engine = PineconesEngine()
                number = engine.e
                print(f"✓ Using e ≈ {number}")
                break
            elif number_input == 'phi':
                engine = PineconesEngine()
                number = engine.phi
                print(f"✓ Using φ ≈ {number}")
                break
            elif '/' in number_input:
                # Handle fractions
                parts = number_input.split('/')
                numerator = Decimal(parts[0])
                denominator = Decimal(parts[1])
                if denominator == 0:
                    print("❌ Error: Division by zero!")
                    continue
                number = numerator / denominator
                print(f"✓ Using {number_input} = {number}")
                break
            else:
                # Handle regular numbers
                number = Decimal(number_input)
                print(f"✓ Using {number}")
                break
        except Exception as e:
            print(f"❌ Invalid input: {e}. Please try again.")
    
    # B) Decimal place
    print("\n📌 STEP B: Confirm Decimal Place")
    print("-" * 80)
    print("Specify which decimal position to examine in the reciprocal.")
    print("  • Position 0 = first digit after decimal point")
    print("  • Position 10 = eleventh digit after decimal point")
    print("  • Recommended: 0-100 for most numbers")
    
    while True:
        try:
            decimal_place = int(input("\nEnter decimal position [default: 0]: ").strip() or "0")
            if decimal_place < 0:
                print("❌ Position must be non-negative!")
                continue
            print(f"✓ Examining position {decimal_place}")
            break
        except ValueError:
            print("❌ Please enter a valid integer!")
    
    # C) Target digit
    print("\n📌 STEP C: Select Target Digit")
    print("-" * 80)
    print("Choose a digit (0-9) to search for in the reciprocal expansion.")
    print("This digit will be highlighted in the analysis.")
    
    while True:
        try:
            target_digit = int(input("\nEnter target digit (0-9): ").strip())
            if target_digit < 0 or target_digit > 9:
                print("❌ Digit must be between 0 and 9!")
                continue
            print(f"✓ Targeting digit {target_digit}")
            break
        except ValueError:
            print("❌ Please enter a valid digit (0-9)!")
    
    # D) Maximum digits
    print("\n📌 STEP D: Set Maximum Digits")
    print("-" * 80)
    print("Specify how many digits to process for coordinate generation.")
    print("  • Minimum: 10 digits")
    print("  • Recommended: 100-1000 digits")
    print("  • Maximum (computational): 10000 digits")
    print("\n⚠️  More digits = more processing time and memory usage")
    
    while True:
        try:
            max_digits_input = input("\nEnter maximum digits [default: 100]: ").strip()
            max_digits = int(max_digits_input) if max_digits_input else 100
            
            if max_digits < 10:
                print("❌ Minimum is 10 digits!")
                continue
            
            if max_digits > 10000:
                print("⚠️  Warning: This may take a very long time!")
                confirm = input("Continue anyway? (yes/no): ").strip().lower()
                if confirm != 'yes':
                    continue
            
            print(f"✓ Processing {max_digits} digits")
            break
        except ValueError:
            print("❌ Please enter a valid integer!")
    
    print("\n" + "="*80)
    print("CONFIGURATION COMPLETE")
    print("="*80)
    print(f"Number: {number}")
    print(f"Decimal Position: {decimal_place}")
    print(f"Target Digit: {target_digit}")
    print(f"Maximum Digits: {max_digits}")
    print("="*80)
    
    return number, decimal_place, target_digit, max_digits


def display_pinecone_summary(pinecone: Dict):
    """Display a summary of the pinecone analysis."""
    print("\n" + "="*80)
    print("PINECONE ANALYSIS SUMMARY")
    print("="*80)
    
    metadata = pinecone['metadata']
    print(f"\n🌲 Pinecone for: {metadata['number']}")
    print(f"   Target Digit: {metadata['target_digit']}")
    print(f"   Decimal Position: {metadata['decimal_place']}")
    print(f"   Digits Processed: {metadata['max_digits']}")
    
    # Reciprocal analysis
    print("\n📊 RECIPROCAL ANALYSIS")
    print("-" * 80)
    recip = pinecone['reciprocal_analysis']
    
    if recip['reciprocal']:
        print(f"Reciprocal: {recip['reciprocal'][:50]}...")
        
        props = recip['mathematical_properties']
        print(f"\nMathematical Properties:")
        print(f"  • Type: {'Rational' if props['is_rational'] else 'Irrational'}")
        print(f"  • Magnitude: {props['magnitude']:.6f}")
        print(f"  • Sign: {props['sign']}")
        
        if recip['target_digit_positions']:
            print(f"\nTarget Digit '{metadata['target_digit']}' found at positions:")
            positions = recip['target_digit_positions'][:10]
            print(f"  {positions}")
            if len(recip['target_digit_positions']) > 10:
                print(f"  ... and {len(recip['target_digit_positions']) - 10} more")
        
        if recip['symmetry_metrics']:
            sym = recip['symmetry_metrics']
            print(f"\nSymmetry Metrics:")
            print(f"  • Product (x × 1/x): {sym['product']:.6f}")
            print(f"  • Sum (x + 1/x): {sym['sum']:.6f}")
            print(f"  • Geometric Mean: {sym['geometric_mean']:.6f}")
        
        if recip['pattern_analysis']:
            pat = recip['pattern_analysis']
            print(f"\nPattern Analysis:")
            print(f"  • Entropy: {pat['entropy']:.4f}")
            print(f"  • Most Common Digit: {pat['most_common']}")
            print(f"  • Least Common Digit: {pat['least_common']}")
    
    # Coordinate analysis
    print("\n🎯 COORDINATE SYSTEM ANALYSIS")
    print("-" * 80)
    coord_analysis = pinecone['coordinate_analysis']
    
    for system_name, analysis in coord_analysis.items():
        print(f"\n{system_name.upper()} Sphere:")
        print(f"  • Points Generated: {analysis['num_points']}")
        print(f"  • Centroid: ({analysis['centroid'][0]:.4f}, {analysis['centroid'][1]:.4f}, {analysis['centroid'][2]:.4f})")
        print(f"  • Spatial Spread: {analysis['spread']:.4f}")
    
    # Pinecone signature
    print("\n🔐 PINECONE SIGNATURE")
    print("-" * 80)
    signature = pinecone['pinecone_signature']
    print(f"Reciprocal Entropy: {signature['reciprocal_entropy']:.4f}")
    print(f"Mathematical Class: {'Rational' if signature['mathematical_class'] else 'Irrational'}")
    print("\nCoordinate Spreads:")
    for system, spread in signature['coordinate_spreads'].items():
        print(f"  • {system}: {spread:.4f}")
    
    print("\n" + "="*80)


def save_pinecone(pinecone: Dict, filename: str):
    """Save pinecone data to JSON file."""
    # Convert Decimal objects to strings for JSON serialization
    def decimal_to_str(obj):
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: decimal_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [decimal_to_str(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(decimal_to_str(item) for item in obj)
        return obj
    
    pinecone_serializable = decimal_to_str(pinecone)
    
    with open(filename, 'w') as f:
        json.dump(pinecone_serializable, f, indent=2)
    
    print(f"\n💾 Pinecone saved to: {filename}")


def main():
    """Main program entry point."""
    display_banner()
    
    # Get user input
    number, decimal_place, target_digit, max_digits = get_user_input()
    
    # Create engine
    print("\n🔧 Initializing Pinecones Engine...")
    engine = PineconesEngine()
    
    # Create pinecone
    print("\n🌲 Generating Pinecone Structure...")
    print("="*80)
    
    start_time = time.time()
    pinecone = engine.create_pinecone(number, decimal_place, target_digit, max_digits)
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️  Processing completed in {elapsed_time:.2f} seconds")
    
    # Display results
    display_pinecone_summary(pinecone)
    
    # Save to file
    filename = f"pinecone_{str(number).replace('.', '_').replace('/', '_')}_{int(time.time())}.json"
    save_pinecone(pinecone, filename)
    
    print("\n" + "="*80)
    print("Thank you for using PINECONES!")
    print("Your mathematical pinecone has been created and saved.")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
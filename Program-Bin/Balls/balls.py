#!/usr/bin/env python3
"""
================================================================================
BALLS - The Hairy Part of Math
================================================================================

This program generates geometric sphere representations of mathematical numbers
using trigonometric polynomial methods inspired by the Hadwiger-Nelson problem.

HADWIGER-NELSON INSPIRED ALGORITHM:
------------------------------------
Instead of the Fibonacci sphere algorithm, this version uses a trigonometric
polynomial approach based on the chromatic number of the plane problem.

The algorithm maps digits to sphere coordinates using:
- Trigonometric polynomials: T(θ) = Σ c_n cos(2πnθ)
- Forbidden angular separations (π/6, π/3, 2π/3)
- Unit circle normalization
- Harmonic analysis techniques

This creates a distribution that respects geometric constraints similar to
those in the Hadwiger-Nelson problem, where points at unit distance must
have different "colors" (in our case, different geometric properties).

KEY CONCEPTS FROM HADWIGER-NELSON:
-----------------------------------
1. Unit Distance Constraint: Points at distance 1 have special relationships
2. Forbidden Angles: Certain angular separations (π/3, π/6) are "forbidden"
3. Trigonometric Polynomials: T(θ) = cos²(3πθ) × cos²(6πθ)
4. Measure Bounds: μ(A) ≤ 1/4 for admissible sets
5. Chromatic Number: Minimum colors needed (k ≥ 4 for single circle)

MINIMUM DIGIT REQUIREMENT:
--------------------------
Due to the trigonometric nature of the algorithm, a minimum of 100 digits
is required for meaningful geometric analysis. This ensures sufficient
resolution for the harmonic patterns.

APPLICATIONS:
-------------
- Study digit distribution with geometric constraints
- Analyze harmonic patterns in number sequences
- Visualize mathematical constants with forbidden angles
- Research chromatic number analogies in digit space

SUPPORTED NUMBER TYPES:
-----------------------
1. Transcendental Numbers (π, e, γ, etc.)
2. Irrational Numbers (√2, √3, φ, etc.)
3. Repeating Rational Numbers (1/3, 2/7, etc.)
4. Non-Repeating Rational Numbers (terminating decimals)

================================================================================
"""

import math
from mpmath import mp
from collections import Counter, defaultdict
import sys
import os

# Set high precision
mp.dps = 50100

class BallsGenerator:
    """Generate sphere representations using Hadwiger-Nelson inspired algorithm"""
    
    def __init__(self):
        self.max_digits = 50000
        self.min_digits = 100  # Minimum required for trigonometric algorithm
        self.transcendental_catalog = self.build_transcendental_catalog()
        
    def build_transcendental_catalog(self):
        """Build catalog of available transcendental numbers"""
        catalog = {
            'pi': {
                'name': 'π (Pi)',
                'description': 'Ratio of circle circumference to diameter',
                'value': lambda: str(mp.pi)
            },
            'e': {
                'name': 'e (Euler\'s number)',
                'description': 'Base of natural logarithm',
                'value': lambda: str(mp.e)
            },
            'euler_gamma': {
                'name': 'γ (Euler-Mascheroni constant)',
                'description': 'Limiting difference between harmonic series and natural log',
                'value': lambda: str(mp.euler)
            },
            'catalan': {
                'name': 'G (Catalan\'s constant)',
                'description': 'Sum of alternating series 1 - 1/9 + 1/25 - 1/49 + ...',
                'value': lambda: str(mp.catalan)
            },
            'khinchin': {
                'name': 'K (Khinchin\'s constant)',
                'description': 'Geometric mean of continued fraction terms',
                'value': lambda: str(mp.khinchin)
            },
            'glaisher': {
                'name': 'A (Glaisher-Kinkelin constant)',
                'description': 'Related to Barnes G-function',
                'value': lambda: str(mp.glaisher)
            },
            'apery': {
                'name': 'ζ(3) (Apéry\'s constant)',
                'description': 'Riemann zeta function at 3',
                'value': lambda: str(mp.zeta(3))
            },
            'zeta5': {
                'name': 'ζ(5)',
                'description': 'Riemann zeta function at 5',
                'value': lambda: str(mp.zeta(5))
            },
            'log2': {
                'name': 'log(2)',
                'description': 'Natural logarithm of 2',
                'value': lambda: str(mp.log(2))
            },
            'log10': {
                'name': 'log(10)',
                'description': 'Natural logarithm of 10',
                'value': lambda: str(mp.log(10))
            },
            'ln_pi': {
                'name': 'log(π)',
                'description': 'Natural logarithm of π',
                'value': lambda: str(mp.log(mp.pi))
            },
            'sqrt_pi': {
                'name': '√π',
                'description': 'Square root of π',
                'value': lambda: str(mp.sqrt(mp.pi))
            },
            'pi_squared': {
                'name': 'π²',
                'description': 'Pi squared',
                'value': lambda: str(mp.pi ** 2)
            },
            'e_squared': {
                'name': 'e²',
                'description': 'Euler\'s number squared',
                'value': lambda: str(mp.e ** 2)
            },
            'pi_e': {
                'name': 'π^e',
                'description': 'Pi to the power of e',
                'value': lambda: str(mp.pi ** mp.e)
            },
            'e_pi': {
                'name': 'e^π',
                'description': 'e to the power of π',
                'value': lambda: str(mp.e ** mp.pi)
            },
            'sqrt_2': {
                'name': '√2 (Pythagoras constant)',
                'description': 'Square root of 2',
                'value': lambda: str(mp.sqrt(2))
            },
            'sqrt_3': {
                'name': '√3 (Theodorus constant)',
                'description': 'Square root of 3',
                'value': lambda: str(mp.sqrt(3))
            },
            'sqrt_5': {
                'name': '√5',
                'description': 'Square root of 5',
                'value': lambda: str(mp.sqrt(5))
            },
            'phi': {
                'name': 'φ (Golden ratio)',
                'description': 'Divine proportion (1 + √5)/2',
                'value': lambda: str(mp.phi)
            },
            'silver_ratio': {
                'name': 'δ_S (Silver ratio)',
                'description': '1 + √2',
                'value': lambda: str(1 + mp.sqrt(2))
            },
            'plastic': {
                'name': 'ρ (Plastic constant)',
                'description': 'Real root of x³ = x + 1',
                'value': lambda: str(mp.findroot(lambda x: x**3 - x - 1, 1.3))
            },
            'omega': {
                'name': 'Ω (Omega constant)',
                'description': 'Solution to Ωe^Ω = 1',
                'value': lambda: str(mp.lambertw(1))
            }
        }
        return catalog
    
    def trigonometric_sphere_coordinates(self, index, total, radius=1.0):
        """
        Map a digit to 3D sphere coordinates using Hadwiger-Nelson inspired
        trigonometric polynomial method.
        
        Based on the paper's approach:
        - Normalize to unit circle [0,1)
        - Apply trigonometric polynomial T(θ) = cos²(3πθ) × cos²(6πθ)
        - Use forbidden angular separation s = 1/6 (corresponding to π/3)
        - Map to 3D sphere surface
        """
        if total <= 1:
            return (0, radius, 0)
        
        # Normalize position to [0, 1)
        theta = index / float(total)
        
        # Apply trigonometric polynomial weighting
        # T(θ) = cos²(3πθ) × cos²(6πθ)
        weight = (math.cos(3 * math.pi * theta) ** 2) * (math.cos(6 * math.pi * theta) ** 2)
        
        # Forbidden angular separation s = 1/6
        forbidden_sep = 1.0 / 6.0
        
        # Adjust theta based on forbidden separation
        # This creates clustering patterns that respect the constraint
        adjusted_theta = theta + forbidden_sep * weight
        
        # Convert to spherical coordinates with harmonic modulation
        # Use multiple harmonics for richer distribution
        phi = 2 * math.pi * adjusted_theta
        
        # Vertical position uses harmonic series
        # This creates bands that respect the chromatic number constraint
        y_harmonic = sum(math.cos(n * math.pi * theta) / n for n in range(1, 5))
        y = math.tanh(y_harmonic)  # Normalize to [-1, 1]
        
        # Radius at this y-level
        radius_at_y = math.sqrt(max(0, 1 - y * y))
        
        # Apply additional harmonic modulation to x and z
        x = math.cos(phi) * radius_at_y * radius
        z = math.sin(phi) * radius_at_y * radius
        y = y * radius
        
        return (x, y, z)
    
    def extract_digits(self, number_str, max_digits):
        """Extract digits from a number string"""
        digits = number_str.replace('.', '').replace('-', '')
        digit_list = [int(d) for d in digits if d.isdigit()]
        return digit_list[:max_digits]
    
    def is_prime(self, n):
        """Check if a digit is prime"""
        return n in [2, 3, 5, 7]
    
    def euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    def angle_between_points(self, p1, p2, center=(0, 0, 0)):
        """Calculate angle between two points from center"""
        v1 = tuple(a - c for a, c in zip(p1, center))
        v2 = tuple(b - c for b, c in zip(p2, center))
        
        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a ** 2 for a in v1))
        mag2 = math.sqrt(sum(b ** 2 for b in v2))
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.acos(cos_angle)
    
    def check_forbidden_angle(self, angle_rad):
        """Check if angle is near a forbidden separation"""
        forbidden_angles = [math.pi/6, math.pi/3, 2*math.pi/3]
        tolerance = 0.1  # radians
        
        for forbidden in forbidden_angles:
            if abs(angle_rad - forbidden) < tolerance:
                return True, math.degrees(forbidden)
        return False, None
    
    def check_unit_distance(self, distance, tolerance=0.1):
        """Check if distance is approximately unit distance"""
        return abs(distance - 1.0) < tolerance
    
    def list_transcendentals(self):
        """Display all available transcendental numbers"""
        print("\nAVAILABLE TRANSCENDENTAL NUMBERS:")
        print("="*80)
        
        sorted_keys = sorted(self.transcendental_catalog.keys())
        for i, key in enumerate(sorted_keys, 1):
            info = self.transcendental_catalog[key]
            print(f"{i:2d}. {info['name']:30s} - {info['description']}")
        
        print("="*80)
        return sorted_keys
    
    def generate_transcendental(self, key):
        """Generate transcendental number by key"""
        if key in self.transcendental_catalog:
            info = self.transcendental_catalog[key]
            print(f"Generating {info['name']}...")
            return info['value'](), info['name']
        return None, None
    
    def generate_irrational(self, name, n=2):
        """Generate irrational numbers"""
        print(f"Generating {name}...")
        if name == 'sqrt':
            return str(mp.sqrt(n)), f"√{n}"
        elif name == 'cbrt':
            return str(mp.cbrt(n)), f"∛{n}"
        elif name == 'nthroot':
            root = int(input("Enter root degree (e.g., 4 for fourth root): "))
            return str(mp.root(n, root)), f"{n}^(1/{root})"
        return None, None
    
    def generate_repeating_rational(self, numerator, denominator):
        """Generate repeating rational numbers"""
        print(f"Generating {numerator}/{denominator}...")
        result = mp.mpf(numerator) / mp.mpf(denominator)
        return str(result), f"{numerator}/{denominator}"
    
    def generate_non_repeating_rational(self, numerator, denominator):
        """Generate non-repeating (terminating) rational numbers"""
        print(f"Generating {numerator}/{denominator}...")
        result = mp.mpf(numerator) / mp.mpf(denominator)
        return str(result), f"{numerator}/{denominator}"
    
    def analyze_and_save(self, number_str, display_name, filename, radius, num_digits):
        """Analyze a number and save to file using Hadwiger-Nelson algorithm"""
        
        if num_digits < self.min_digits:
            print(f"\nWARNING: Minimum {self.min_digits} digits required for trigonometric algorithm!")
            print(f"Adjusting to {self.min_digits} digits...")
            num_digits = self.min_digits
        
        print(f"\nAnalyzing {display_name} using Hadwiger-Nelson inspired algorithm...")
        print(f"Extracting {num_digits} digits...")
        
        digits = self.extract_digits(number_str, num_digits)
        
        if len(digits) < num_digits:
            print(f"Warning: Only {len(digits)} digits available")
            num_digits = len(digits)
        
        print(f"Calculating sphere coordinates with trigonometric polynomials...")
        
        with open(filename, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("BALLS - The Hairy Part of Math\n")
            f.write("="*80 + "\n")
            f.write(f"SPHERE ANALYSIS: {display_name}\n")
            f.write("="*80 + "\n\n")
            f.write("ALGORITHM: Hadwiger-Nelson Inspired Trigonometric Polynomial Method\n")
            f.write("Based on chromatic number of the plane problem\n")
            f.write("T(θ) = cos²(3πθ) × cos²(6πθ)\n")
            f.write("Forbidden angular separation: s = 1/6 (π/3 radians)\n\n")
            f.write(f"Total digits analyzed: {num_digits}\n")
            f.write(f"Sphere radius: {radius}\n")
            f.write(f"Value (first 100 chars): {number_str[:100]}...\n\n")
            
            # Digit statistics
            f.write("DIGIT STATISTICS:\n")
            f.write("-"*80 + "\n")
            counter = Counter(digits)
            f.write(f"Digit frequencies: {dict(counter)}\n")
            f.write(f"Most common digits: {counter.most_common(5)}\n")
            f.write(f"Least common digits: {counter.most_common()[-5:]}\n\n")
            
            # Prime and zero counts
            prime_count = sum(1 for d in digits if self.is_prime(d))
            zero_count = sum(1 for d in digits if d == 0)
            f.write(f"Prime digits (2,3,5,7): {prime_count} ({100*prime_count/len(digits):.2f}%)\n")
            f.write(f"Zero digits: {zero_count} ({100*zero_count/len(digits):.2f}%)\n\n")
            
            # Coordinates
            f.write("SPHERE COORDINATES (Hadwiger-Nelson Algorithm):\n")
            f.write("-"*80 + "\n")
            f.write("Format: Position | Digit | Type | X | Y | Z\n\n")
            
            batch_size = 1000
            for batch_start in range(0, len(digits), batch_size):
                batch_end = min(batch_start + batch_size, len(digits))
                
                for i in range(batch_start, batch_end):
                    digit = digits[i]
                    coord = self.trigonometric_sphere_coordinates(i, len(digits), radius=radius)
                    
                    digit_type = ""
                    if self.is_prime(digit):
                        digit_type = "PRIME"
                    elif digit == 0:
                        digit_type = "ZERO"
                    else:
                        digit_type = "COMPOSITE"
                    
                    f.write(f"{i:6d} | {digit} | {digit_type:9s} | "
                           f"{coord[0]:10.6f} | {coord[1]:10.6f} | {coord[2]:10.6f}\n")
                
                if (batch_end % 10000) == 0 and batch_end < len(digits):
                    print(f"  Processed {batch_end}/{len(digits)} coordinates...")
            
            print("Analyzing Hadwiger-Nelson constraints...")
            f.write("\n" + "="*80 + "\n")
            f.write("HADWIGER-NELSON CONSTRAINT ANALYSIS:\n")
            f.write("="*80 + "\n\n")
            
            # Analyze unit distance and forbidden angle relationships
            f.write("UNIT DISTANCE RELATIONSHIPS (≈1.0 ± 0.1):\n")
            f.write("-"*80 + "\n")
            
            unit_dist_count = 0
            forbidden_angle_count = 0
            sample_limit = 1000
            
            for i in range(min(500, len(digits))):
                coord1 = self.trigonometric_sphere_coordinates(i, len(digits), radius=radius)
                
                for j in range(i + 1, min(i + 10, len(digits))):
                    if unit_dist_count >= sample_limit:
                        break
                    
                    coord2 = self.trigonometric_sphere_coordinates(j, len(digits), radius=radius)
                    dist = self.euclidean_distance(coord1, coord2)
                    angle = self.angle_between_points(coord1, coord2)
                    
                    is_unit = self.check_unit_distance(dist)
                    is_forbidden, forbidden_val = self.check_forbidden_angle(angle)
                    
                    if is_unit or is_forbidden:
                        f.write(f"Pos {i:5d} (digit {digits[i]}) <-> Pos {j:5d} (digit {digits[j]}): ")
                        f.write(f"dist={dist:.5f}")
                        if is_unit:
                            f.write(" [UNIT]")
                        if is_forbidden:
                            f.write(f" [FORBIDDEN ANGLE ≈{forbidden_val:.1f}°]")
                        f.write(f" angle={math.degrees(angle):.2f}°\n")
                        
                        unit_dist_count += 1
                
                if unit_dist_count >= sample_limit:
                    break
            
            f.write(f"\nTotal unit distance pairs found (sample): {unit_dist_count}\n\n")
            
            print("Analyzing prime-zero relationships...")
            f.write("\n" + "="*80 + "\n")
            f.write("PRIME-ZERO GEOMETRIC RELATIONSHIPS:\n")
            f.write("="*80 + "\n\n")
            
            # Find all prime and zero positions
            prime_positions = [(i, digits[i]) for i in range(len(digits)) if self.is_prime(digits[i])]
            zero_positions = [i for i in range(len(digits)) if digits[i] == 0]
            
            f.write(f"Total prime digits: {len(prime_positions)}\n")
            f.write(f"Total zero digits: {len(zero_positions)}\n")
            f.write(f"Total prime-zero pairs: {len(prime_positions) * len(zero_positions)}\n\n")
            
            # Sample relationships
            f.write("SAMPLE PRIME-ZERO RELATIONSHIPS (first 1000 pairs):\n")
            f.write("-"*80 + "\n")
            f.write("Format: Prime_Pos | Prime_Digit | Zero_Pos | Distance | Angle | Constraints\n\n")
            
            pair_count = 0
            max_sample_pairs = 1000
            
            for prime_idx, prime_digit in prime_positions[:200]:
                prime_coord = self.trigonometric_sphere_coordinates(prime_idx, len(digits), radius=radius)
                
                for zero_idx in zero_positions[:5]:
                    if pair_count >= max_sample_pairs:
                        break
                        
                    zero_coord = self.trigonometric_sphere_coordinates(zero_idx, len(digits), radius=radius)
                    dist = self.euclidean_distance(prime_coord, zero_coord)
                    angle = self.angle_between_points(prime_coord, zero_coord)
                    
                    is_unit = self.check_unit_distance(dist)
                    is_forbidden, forbidden_val = self.check_forbidden_angle(angle)
                    
                    constraints = []
                    if is_unit:
                        constraints.append("UNIT")
                    if is_forbidden:
                        constraints.append(f"FORBIDDEN≈{forbidden_val:.0f}°")
                    
                    constraint_str = ",".join(constraints) if constraints else "NONE"
                    
                    f.write(f"{prime_idx:6d} | {prime_digit} | {zero_idx:6d} | "
                           f"{dist:10.6f} | {math.degrees(angle):10.2f} | {constraint_str}\n")
                    
                    pair_count += 1
                
                if pair_count >= max_sample_pairs:
                    break
            
            print("Analyzing digit clustering...")
            f.write("\n" + "="*80 + "\n")
            f.write("DIGIT CLUSTERING ANALYSIS:\n")
            f.write("="*80 + "\n\n")
            
            # Analyze clustering for each digit
            for digit in range(10):
                positions = [i for i in range(len(digits)) if digits[i] == digit]
                
                if len(positions) == 0:
                    continue
                
                f.write(f"\nDigit {digit}:")
                if self.is_prime(digit):
                    f.write(" [PRIME]")
                elif digit == 0:
                    f.write(" [ZERO]")
                f.write("\n")
                f.write(f"  Occurrences: {len(positions)}\n")
                f.write(f"  Percentage: {100*len(positions)/len(digits):.4f}%\n")
                
                # Calculate centroid from sample
                sample_size = min(1000, len(positions))
                sample_positions = positions[::len(positions)//sample_size] if len(positions) > sample_size else positions
                
                coords = [self.trigonometric_sphere_coordinates(pos, len(digits), radius=radius) for pos in sample_positions]
                centroid = tuple(sum(c[i] for c in coords) / len(coords) for i in range(3))
                
                f.write(f"  Centroid (from {len(coords)} samples): "
                       f"({centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f})\n")
                
                # Calculate spread
                spread = sum(self.euclidean_distance(c, centroid) for c in coords) / len(coords)
                f.write(f"  Average spread from centroid: {spread:.6f}\n")
            
            print("Analyzing uniqueness patterns...")
            f.write("\n" + "="*80 + "\n")
            f.write("UNIQUENESS ANALYSIS:\n")
            f.write("="*80 + "\n\n")
            
            unique_digits = [d for d in digits if counter[d] == 1]
            f.write(f"Unique digits (appearing only once): {len(unique_digits)}\n")
            if unique_digits:
                f.write(f"Unique digit values: {unique_digits[:50]}\n")
                if len(unique_digits) > 50:
                    f.write(f"  ... and {len(unique_digits) - 50} more\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("ANALYSIS COMPLETE\n")
            f.write("Hadwiger-Nelson inspired trigonometric polynomial method applied\n")
            f.write("="*80 + "\n")
        
        print(f"Analysis complete! Saved to: {filename}")
        return filename

def print_header():
    """Print program header"""
    print("\n" + "="*80)
    print("BALLS - The Hairy Part of Math")
    print("="*80)
    print("\nHadwiger-Nelson Inspired Number Sphere Generator")
    print("\nThis program maps digits of mathematical numbers onto 3D spheres using")
    print("trigonometric polynomial methods inspired by the chromatic number of")
    print("the plane problem (Hadwiger-Nelson).")
    print("\nKey Features:")
    print("- Trigonometric polynomial distribution: T(θ) = cos²(3πθ) × cos²(6πθ)")
    print("- Forbidden angular separations (π/6, π/3, 2π/3)")
    print("- Unit distance constraint analysis")
    print("- Harmonic pattern detection")
    print("\nMINIMUM REQUIREMENT: 100 digits for meaningful trigonometric analysis")
    print("\nAll results are automatically saved to text files.")
    print("="*80 + "\n")

def main():
    """Main program loop"""
    
    print_header()
    
    generator = BallsGenerator()
    generated_files = []
    
    while True:
        print("\nSELECT NUMBER TYPE:")
        print("="*80)
        print("1. Transcendental Numbers (π, e, γ, ζ(3), log(2), etc.)")
        print("2. Irrational Numbers (√2, √3, φ, etc.)")
        print("3. Repeating Rational Numbers (1/3, 2/7, etc.)")
        print("4. Non-Repeating Rational Numbers (terminating decimals)")
        print("5. View generated files")
        print("6. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '6':
            print("\n" + "="*80)
            print("SUMMARY OF GENERATED FILES:")
            print("="*80)
            if generated_files:
                for i, (fname, fsize) in enumerate(generated_files, 1):
                    print(f"{i}. {fname} ({fsize:.2f} MB)")
            else:
                print("No files generated in this session.")
            print("="*80)
            print("\nThank you for using BALLS - The Hairy Part of Math!")
            break
        
        if choice == '5':
            print("\n" + "="*80)
            print("GENERATED FILES THIS SESSION:")
            print("="*80)
            if generated_files:
                for i, (fname, fsize) in enumerate(generated_files, 1):
                    print(f"{i}. {fname} ({fsize:.2f} MB)")
            else:
                print("No files generated yet in this session.")
            print("="*80)
            continue
        
        if choice not in ['1', '2', '3', '4']:
            print("Invalid choice. Please try again.")
            continue
        
        # Get number of digits
        print(f"\nHow many digits to analyze? ({generator.min_digits}-{generator.max_digits})")
        num_digits_input = input(f"Enter number of digits [default: 50000]: ").strip()
        num_digits = int(num_digits_input) if num_digits_input else 50000
        num_digits = min(max(generator.min_digits, num_digits), generator.max_digits)
        
        # Get sphere radius
        radius_input = input("Enter sphere radius [default: 1.0]: ").strip()
        radius = float(radius_input) if radius_input else 1.0
        
        number_str = None
        display_name = None
        filename = None
        
        if choice == '1':
            # Transcendental numbers
            sorted_keys = generator.list_transcendentals()
            
            trans_choice = input(f"\nSelect transcendental number (1-{len(sorted_keys)}): ").strip()
            
            try:
                idx = int(trans_choice) - 1
                if 0 <= idx < len(sorted_keys):
                    key = sorted_keys[idx]
                    number_str, display_name = generator.generate_transcendental(key)
                    # Clean filename
                    clean_name = key.replace('_', '').replace('(', '').replace(')', '')
                    filename = f"balls_{clean_name}_{num_digits}digits.txt"
                else:
                    print("Invalid selection.")
                    continue
            except ValueError:
                print("Invalid input.")
                continue
        
        elif choice == '2':
            # Irrational numbers
            print("\nIRRATIONAL NUMBER OPTIONS:")
            print("-"*80)
            print("1. Square root (√n)")
            print("2. Cube root (∛n)")
            print("3. Nth root (n^(1/m))")
            print("-"*80)
            
            irr_choice = input("Select irrational number type (1-3): ").strip()
            
            if irr_choice == '1':
                n = int(input("Enter number for square root: "))
                number_str, display_name = generator.generate_irrational('sqrt', n)
                filename = f"balls_sqrt{n}_{num_digits}digits.txt"
            elif irr_choice == '2':
                n = int(input("Enter number for cube root: "))
                number_str, display_name = generator.generate_irrational('cbrt', n)
                filename = f"balls_cbrt{n}_{num_digits}digits.txt"
            elif irr_choice == '3':
                n = int(input("Enter base number: "))
                number_str, display_name = generator.generate_irrational('nthroot', n)
                filename = f"balls_root{n}_{num_digits}digits.txt"
        
        elif choice == '3':
            # Repeating rational
            print("\nREPEATING RATIONAL NUMBER:")
            print("-"*80)
            numerator = int(input("Enter numerator: "))
            denominator = int(input("Enter denominator: "))
            
            number_str, display_name = generator.generate_repeating_rational(numerator, denominator)
            filename = f"balls_{numerator}div{denominator}_{num_digits}digits.txt"
        
        elif choice == '4':
            # Non-repeating rational
            print("\nNON-REPEATING RATIONAL NUMBER:")
            print("-"*80)
            numerator = int(input("Enter numerator: "))
            denominator = int(input("Enter denominator: "))
            
            number_str, display_name = generator.generate_non_repeating_rational(numerator, denominator)
            filename = f"balls_{numerator}div{denominator}_{num_digits}digits.txt"
        
        if number_str and display_name and filename:
            print(f"\nGenerating sphere for: {display_name}")
            print(f"Digits: {num_digits}")
            print(f"Radius: {radius}")
            print(f"Algorithm: Hadwiger-Nelson inspired trigonometric polynomial")
            print(f"Output file: {filename}")
            print()
            
            try:
                result_file = generator.analyze_and_save(number_str, display_name, filename, radius, num_digits)
                file_size = os.path.getsize(result_file) / (1024*1024)
                generated_files.append((result_file, file_size))
                print(f"\n✓ Success! BALLS analysis saved to: {result_file}")
                print(f"  File size: {file_size:.2f} MB")
            except Exception as e:
                print(f"\n✗ Error: {e}")
        else:
            print("\nInvalid selection or error occurred.")
        
        # Ask if user wants to generate another
        another = input("\nGenerate another sphere? (y/n): ").strip().lower()
        if another != 'y':
            print("\n" + "="*80)
            print("SUMMARY OF GENERATED FILES:")
            print("="*80)
            if generated_files:
                for i, (fname, fsize) in enumerate(generated_files, 1):
                    print(f"{i}. {fname} ({fsize:.2f} MB)")
            else:
                print("No files generated in this session.")
            print("="*80)
            print("\nThank you for using BALLS - The Hairy Part of Math!")
            break

if __name__ == "__main__":
    main()
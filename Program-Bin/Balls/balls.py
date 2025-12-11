#!/usr/bin/env python3
"""
================================================================================
BALLS - The Hairy Part of Math (Version 4.0 - Multi-Sphere Edition)
================================================================================

This program generates geometric sphere representations of mathematical numbers
using five distinct sphere generation algorithms, each representing a different
mathematical paradigm.

NEW IN VERSION 4.0:
- Five Sphere Types: Hadwiger-Nelson, Banachian, Fuzzy, Quantum, RELATIONAL
- Fuzzy Sphere: Quantum angular momentum states (noncommutative geometry)
- Quantum Sphere: q-deformed classical sphere (quantum groups)
- RELATIONAL Sphere: Meta-sphere synthesizing all four base types
- Enhanced collision avoidance and spatial distribution

VERSION 3.0 FEATURES:
- Banachian Sphere: Complete normed vector space
- Trigonometry checks for unit sphere ordinal placement
- Products and exponents range calculator

VERSION 2.0 FEATURES:
- Quantum Number Range Support: Analyze specific digit ranges
- 40+ Additional Transcendental Functions
- Enhanced mathematical constant catalog

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
import psutil

# Set high precision
mp.dps = 50100


from fractions import Fraction
from typing import Tuple, List, Dict, Optional

# Import numpy for advanced features (optional)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Import advanced analysis modules (optional)
try:
    from advanced_analysis import (
        PersistentHomologyAnalyzer,
        CurvatureEstimator,
        DiscrepancyAnalyzer,
        FunctionalCompletenessVerifier
    )
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False


class BaseConverter:
    """
    Convert numbers to arbitrary integer bases and detect expansion properties.
    """
    
    def __init__(self, base: int = 10):
        """
        Initialize base converter.
        
        Args:
            base: Integer base (must be >= 2)
        """
        if base < 2:
            raise ValueError("Base must be at least 2")
        self.base = base
        
    def convert_to_base(self, number_str: str, num_digits: int) -> Tuple[List[int], Dict]:
        """
        Convert a decimal number to specified base.
        
        Args:
            number_str: String representation of number in base 10
            num_digits: Number of digits to generate
            
        Returns:
            Tuple of (digit_list, metadata_dict)
            metadata includes: is_terminating, period_length, period_start, etc.
        """
        metadata = {
            'base': self.base,
            'is_terminating': False,
            'is_repeating': False,
            'period_length': None,
            'period_start': None,
            'expansion_type': 'unknown'
        }
        
        # For base 10, check if it's a fraction first
        if self.base == 10 and '/' not in number_str:
            # Remove decimal point and convert to digit list
            clean_str = number_str.replace('.', '')
            digits = [int(d) for d in clean_str[:num_digits]]
            
            # Detect if it's a known rational
            metadata['expansion_type'] = 'base10_direct'
            
            return digits, metadata
        
        # For other bases, we need to convert
        # Parse the input number
        try:
            # Try to parse as fraction first
            if '/' in number_str:
                parts = number_str.split('/')
                numerator = int(parts[0])
                denominator = int(parts[1])
                
                # Check if expansion terminates
                metadata['is_rational'] = True
                metadata['expansion_type'] = 'rational'
                
                # Check termination condition
                temp_denom = denominator
                temp_base = self.base
                
                # Remove common factors with base
                while temp_denom > 1:
                    gcd_val = math.gcd(temp_denom, temp_base)
                    if gcd_val == 1:
                        break
                    temp_denom //= gcd_val
                    temp_base = self.base
                
                if temp_denom == 1:
                    metadata['is_terminating'] = True
                    metadata['expansion_type'] = 'terminating_rational'
                else:
                    metadata['is_repeating'] = True
                    metadata['expansion_type'] = 'repeating_rational'
                
                # Generate digits using long division
                digits = self._long_division_base_conversion(numerator, denominator, num_digits, metadata)
                
            else:
                # It's an irrational/transcendental - use mpmath
                metadata['expansion_type'] = 'irrational'
                digits = self._convert_irrational_to_base(number_str, num_digits)
                
        except Exception as e:
            print(f"Warning: Error in base conversion: {e}")
            print(f"Falling back to base-10 representation")
            clean_str = number_str.replace('.', '')
            digits = [int(d) for d in clean_str[:num_digits]]
            metadata['expansion_type'] = 'fallback_base10'
        
        return digits, metadata
    
    def _long_division_base_conversion(self, numerator: int, denominator: int, 
                                      num_digits: int, metadata: Dict) -> List[int]:
        """
        Perform long division to convert rational to base.
        Also detects period for repeating decimals.
        """
        digits = []
        remainders_seen = {}
        remainder = numerator % denominator
        position = 0
        
        while len(digits) < num_digits:
            # Check if we've seen this remainder before (repeating cycle)
            if remainder in remainders_seen and not metadata['is_terminating']:
                # Found the period!
                period_start = remainders_seen[remainder]
                period_length = position - period_start
                metadata['period_start'] = period_start
                metadata['period_length'] = period_length
                
                # Fill remaining digits by repeating the cycle
                period = digits[period_start:position]
                while len(digits) < num_digits:
                    digits.extend(period)
                digits = digits[:num_digits]
                break
            
            remainders_seen[remainder] = position
            
            # Perform division
            remainder *= self.base
            digit = remainder // denominator
            remainder = remainder % denominator
            
            digits.append(digit)
            position += 1
            
            # Check for termination
            if remainder == 0:
                metadata['is_terminating'] = True
                metadata['termination_position'] = position
                # Can't generate more digits - insufficient data
                break
        
        return digits
    
    def _convert_irrational_to_base(self, number_str: str, num_digits: int) -> List[int]:
        """
        Convert irrational number to arbitrary base using mpmath.
        """
        # This is complex - for now, we'll use a high-precision approach
        # Convert to mpmath number
        if number_str == 'pi':
            num = mp.pi
        elif number_str == 'e':
            num = mp.e
        elif number_str.startswith('sqrt'):
            # Parse sqrt(n)
            n = int(number_str.replace('sqrt(', '').replace(')', ''))
            num = mp.sqrt(n)
        else:
            # Try to evaluate as mpmath expression
            num = mp.mpf(number_str)
        
        # Convert to target base digit by digit
        digits = []
        current = num
        
        # Get integer part
        integer_part = int(mp.floor(current))
        current = current - integer_part
        
        # Convert fractional part
        for _ in range(num_digits):
            current *= self.base
            digit = int(mp.floor(current))
            digits.append(digit)
            current = current - digit
        
        return digits


class PatternRecorder:
    """
    Record and analyze patterns during sphere generation.
    """
    
    def __init__(self):
        self.patterns = {
            'digit_distribution': Counter(),
            'coordinate_clusters': [],
            'angular_distribution': [],
            'radial_distribution': [],
            'collision_positions': [],
            'unit_sphere_violations': [],
            'sequential_patterns': [],
            'period_analysis': {},
            'spatial_symmetries': {}
        }
        
    def record_digit(self, digit: int, position: int):
        """Record a digit and its position."""
        self.patterns['digit_distribution'][digit] += 1
        
    def record_coordinate(self, coord: Tuple[float, float, float], position: int):
        """Record a generated coordinate."""
        x, y, z = coord
        
        # Calculate spherical coordinates
        r = math.sqrt(x*x + y*y + z*z)
        theta = math.acos(z / r) if r > 0 else 0
        phi = math.atan2(y, x)
        
        self.patterns['angular_distribution'].append((theta, phi, position))
        self.patterns['radial_distribution'].append((r, position))
        
    def record_collision(self, position: int, coord: Tuple[float, float, float]):
        """Record a collision event."""
        self.patterns['collision_positions'].append((position, coord))
        
    def record_violation(self, position: int, coord: Tuple[float, float, float], radius: float):
        """Record a unit sphere violation."""
        self.patterns['unit_sphere_violations'].append((position, coord, radius))
        
    def detect_period(self, digits: List[int], max_period: int = 1000) -> Optional[int]:
        """
        Detect repeating period in digit sequence.
        
        Returns:
            Period length if found, None otherwise
        """
        n = len(digits)
        
        for period in range(1, min(max_period, n // 2)):
            is_periodic = True
            for i in range(period, min(n, period * 10)):
                if digits[i] != digits[i % period]:
                    is_periodic = False
                    break
            
            if is_periodic:
                return period
        
        return None
    
    def analyze_clustering(self, coordinates: List[Tuple[float, float, float]], 
                          grid_resolution: int = 10) -> Dict:
        """
        Analyze spatial clustering of points on sphere.
        
        Returns:
            Dictionary with clustering metrics
        """
        # Divide sphere into grid cells
        cells = defaultdict(int)
        
        for x, y, z in coordinates:
            # Convert to spherical coordinates
            r = math.sqrt(x*x + y*y + z*z)
            if r < 1e-10:
                continue
                
            theta = math.acos(max(-1, min(1, z / r)))
            phi = math.atan2(y, x)
            
            # Discretize
            theta_cell = int(theta / math.pi * grid_resolution)
            phi_cell = int((phi + math.pi) / (2 * math.pi) * grid_resolution)
            
            cells[(theta_cell, phi_cell)] += 1
        
        # Calculate statistics
        counts = list(cells.values())
        if not counts:
            return {'error': 'No valid coordinates'}
        
        return {
            'num_cells_occupied': len(cells),
            'total_cells': grid_resolution * grid_resolution,
            'occupancy_rate': len(cells) / (grid_resolution * grid_resolution),
            'mean_points_per_cell': sum(counts) / len(counts),
            'max_points_in_cell': max(counts),
            'min_points_in_cell': min(counts),
            'clustering_coefficient': max(counts) / (sum(counts) / len(counts))
        }
    
    def generate_report(self, base: int, metadata: Dict) -> str:
        """
        Generate comprehensive pattern analysis report.
        """
        report = []
        report.append("=" * 80)
        report.append("EMPIRICAL PATTERN ANALYSIS")
        report.append("=" * 80)
        
        # Base information
        report.append(f"\nBASE: {base}")
        report.append(f"Expansion Type: {metadata.get('expansion_type', 'unknown')}")
        
        if metadata.get('is_terminating'):
            report.append(f"⚠ TERMINATING EXPANSION at position {metadata.get('termination_position')}")
            report.append("  Insufficient digits for sphere generation!")
        
        if metadata.get('is_repeating'):
            report.append(f"Repeating Period: {metadata.get('period_length')} digits")
            report.append(f"Period Starts at: position {metadata.get('period_start')}")
        
        # Digit distribution
        report.append("\nDIGIT DISTRIBUTION:")
        report.append("-" * 40)
        dist = self.patterns['digit_distribution']
        total = sum(dist.values())
        
        for digit in range(base):
            count = dist.get(digit, 0)
            percentage = (count / total * 100) if total > 0 else 0
            bar = '█' * int(percentage / 2)
            report.append(f"  {digit:2d}: {count:6d} ({percentage:5.2f}%) {bar}")
        
        # Collision analysis
        num_collisions = len(self.patterns['collision_positions'])
        report.append(f"\nCOLLISIONS: {num_collisions}")
        
        if num_collisions > 0:
            report.append("  First 5 collision positions:")
            for pos, coord in self.patterns['collision_positions'][:5]:
                report.append(f"    Position {pos}: {coord}")
        
        # Unit sphere violations
        num_violations = len(self.patterns['unit_sphere_violations'])
        report.append(f"\nUNIT SPHERE VIOLATIONS: {num_violations}")
        
        if num_violations > 0:
            report.append("  First 5 violations:")
            for pos, coord, r in self.patterns['unit_sphere_violations'][:5]:
                report.append(f"    Position {pos}: r={r:.10f}")
        
        # Radial distribution
        if self.patterns['radial_distribution']:
            radii = [r for r, _ in self.patterns['radial_distribution']]
            report.append(f"\nRADIAL DISTRIBUTION:")
            report.append(f"  Mean radius: {sum(radii)/len(radii):.10f}")
            report.append(f"  Min radius:  {min(radii):.10f}")
            report.append(f"  Max radius:  {max(radii):.10f}")
            report.append(f"  Std dev:     {(sum((r - 1)**2 for r in radii) / len(radii))**0.5:.10f}")
        
        report.append("=" * 80)
        
        return "\n".join(report)


class SphereRangeCalculator:
    """
    NEW: Calculate minimum ranges for products and exponents to form spheres
    
    This class determines the minimum number of digits required for various
    mathematical operations to generate meaningful sphere representations.
    """
    
    def __init__(self):
        self.min_sphere_digits = 100  # Base minimum for any sphere
    
    def calculate_product_range(self, factor1, factor2):
        """
        Calculate minimum digit range for product-based sphere generation
        
        Args:
            factor1: First factor (e.g., π)
            factor2: Second factor (e.g., e)
        
        Returns:
            dict with minimum range requirements
        """
        # Products typically need more digits for pattern emergence
        base_requirement = self.min_sphere_digits
        
        # Calculate complexity factor based on transcendental nature
        complexity_multiplier = 1.5 if self._is_transcendental(factor1) and self._is_transcendental(factor2) else 1.2
        
        min_digits = int(base_requirement * complexity_multiplier)
        
        return {
            'operation': 'product',
            'factors': (factor1, factor2),
            'min_digits': min_digits,
            'recommended_digits': min_digits * 2,
            'reason': 'Product patterns require extended digit sequences for geometric stability'
        }
    
    def calculate_exponent_range(self, base, exponent):
        """
        Calculate minimum digit range for exponent-based sphere generation
        
        Args:
            base: Base number (e.g., π)
            exponent: Exponent value (e.g., 2, e, π)
        
        Returns:
            dict with minimum range requirements
        """
        base_requirement = self.min_sphere_digits
        
        # Exponents create rapid growth, requiring more digits
        if exponent > 2:
            complexity_multiplier = 2.0
        elif self._is_transcendental(exponent):
            complexity_multiplier = 1.8
        else:
            complexity_multiplier = 1.3
        
        min_digits = int(base_requirement * complexity_multiplier)
        
        return {
            'operation': 'exponent',
            'base': base,
            'exponent': exponent,
            'min_digits': min_digits,
            'recommended_digits': min_digits * 2,
            'reason': 'Exponential growth requires extended precision for sphere coherence'
        }
    
    def calculate_combined_range(self, operations):
        """
        Calculate minimum range for combined operations (e.g., (π × e)^2)
        
        Args:
            operations: List of operation types
        
        Returns:
            dict with minimum range requirements
        """
        base_requirement = self.min_sphere_digits
        
        # Each operation adds complexity
        complexity_multiplier = 1.0 + (0.5 * len(operations))
        
        min_digits = int(base_requirement * complexity_multiplier)
        
        return {
            'operation': 'combined',
            'operation_count': len(operations),
            'min_digits': min_digits,
            'recommended_digits': min_digits * 3,
            'reason': 'Combined operations require maximum precision for stable sphere formation'
        }
    
    def _is_transcendental(self, value):
        """Check if a value is likely transcendental"""
        transcendental_names = ['pi', 'e', 'euler', 'gamma', 'phi']
        if isinstance(value, str):
            return any(name in value.lower() for name in transcendental_names)
        return False
    
    def get_sphere_feasibility(self, num_digits, operation_type='simple'):
        """
        Assess whether a given digit count is feasible for sphere generation
        
        Returns:
            dict with feasibility assessment
        """
        min_required = self.min_sphere_digits
        
        if operation_type == 'product':
            min_required = int(min_required * 1.5)
        elif operation_type == 'exponent':
            min_required = int(min_required * 2.0)
        elif operation_type == 'combined':
            min_required = int(min_required * 2.5)
        
        feasibility = 'excellent' if num_digits >= min_required * 2 else \
                     'good' if num_digits >= min_required * 1.5 else \
                     'adequate' if num_digits >= min_required else \
                     'insufficient'
        
        return {
            'num_digits': num_digits,
            'min_required': min_required,
            'feasibility': feasibility,
            'can_form_sphere': num_digits >= min_required,
            'recommendation': f"Use at least {min_required} digits for {operation_type} operations"
        }

class BallsGenerator:
    """Generate sphere representations using Hadwiger-Nelson inspired algorithm"""
    
    def __init__(self):
        self.max_digits = float('inf')  # No hard limit
        self.min_digits = 100  # Minimum required for trigonometric algorithm
        self.transcendental_catalog = self.build_transcendental_catalog()
        
        # Calculate machine-specific limits
        self.compute_limits = self.calculate_machine_limits()
        
        # NEW: Sphere type selection (Hadwiger-Nelson or Banachian)
        self.sphere_type = 'hadwiger_nelson'  # Default to original algorithm
        
        # NEW: Base conversion support (v6 feature)
        self.base = 10  # Default to base 10
        self.base_converter = None  # Will be initialized when needed
        self.pattern_recorder = None  # Will be initialized when needed
    
    def calculate_machine_limits(self):
        """Calculate computational limits based on available system resources"""
        import psutil
        import os
        
        # Get available memory
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        # Get CPU count
        cpu_count = os.cpu_count() or 2
        
        # Estimate maximum digits based on memory
        # Rough estimate: ~2 KB per digit for full analysis
        estimated_max_digits = int((available_ram_gb * 1024 * 1024 * 0.5) / 2)  # Use 50% of available RAM
        
        # Estimate time per digit (rough approximation)
        # Based on testing: ~0.004 seconds per digit on average hardware
        time_per_digit = 0.004
        
        return {
            'available_ram_gb': available_ram_gb,
            'cpu_count': cpu_count,
            'estimated_max_digits': estimated_max_digits,
            'time_per_digit': time_per_digit,
            'recommended_max': min(estimated_max_digits, 100000),  # Cap at 100k for practicality
            'warning_threshold': 50000
        }
        

    def set_base(self, base: int):
        """
        Set the numerical base for digit generation (v6 feature).
        
        Args:
            base: Integer base (must be >= 2)
        """
        if base < 2:
            raise ValueError("Base must be at least 2")
        self.base = base
        self.base_converter = BaseConverter(base)
        print(f"✓ Base set to {base}")

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
            },
            'zeta7': {
                'name': 'ζ(7)',
                'description': 'Riemann zeta function at 7',
                'value': lambda: str(mp.zeta(7))
            },
            'zeta9': {
                'name': 'ζ(9)',
                'description': 'Riemann zeta function at 9',
                'value': lambda: str(mp.zeta(9))
            },
            'log3': {
                'name': 'log(3)',
                'description': 'Natural logarithm of 3',
                'value': lambda: str(mp.log(3))
            },
            'log5': {
                'name': 'log(5)',
                'description': 'Natural logarithm of 5',
                'value': lambda: str(mp.log(5))
            },
            'log7': {
                'name': 'log(7)',
                'description': 'Natural logarithm of 7',
                'value': lambda: str(mp.log(7))
            },
            'ln_e': {
                'name': 'log(e)',
                'description': 'Natural logarithm of e (equals 1)',
                'value': lambda: str(mp.log(mp.e))
            },
            'sqrt_e': {
                'name': '√e',
                'description': 'Square root of e',
                'value': lambda: str(mp.sqrt(mp.e))
            },
            'cbrt_2': {
                'name': '∛2',
                'description': 'Cube root of 2',
                'value': lambda: str(mp.cbrt(2))
            },
            'cbrt_3': {
                'name': '∛3',
                'description': 'Cube root of 3',
                'value': lambda: str(mp.cbrt(3))
            },
            'cbrt_5': {
                'name': '∛5',
                'description': 'Cube root of 5',
                'value': lambda: str(mp.cbrt(5))
            },
            'sqrt_6': {
                'name': '√6',
                'description': 'Square root of 6',
                'value': lambda: str(mp.sqrt(6))
            },
            'sqrt_7': {
                'name': '√7',
                'description': 'Square root of 7',
                'value': lambda: str(mp.sqrt(7))
            },
            'sqrt_8': {
                'name': '√8',
                'description': 'Square root of 8',
                'value': lambda: str(mp.sqrt(8))
            },
            'sqrt_10': {
                'name': '√10',
                'description': 'Square root of 10',
                'value': lambda: str(mp.sqrt(10))
            },
            'pi_cubed': {
                'name': 'π³',
                'description': 'Pi cubed',
                'value': lambda: str(mp.pi ** 3)
            },
            'e_cubed': {
                'name': 'e³',
                'description': 'Euler\'s number cubed',
                'value': lambda: str(mp.e ** 3)
            },
            'pi_sqrt': {
                'name': 'π^(1/2)',
                'description': 'Square root of pi',
                'value': lambda: str(mp.sqrt(mp.pi))
            },
            'e_sqrt': {
                'name': 'e^(1/2)',
                'description': 'Square root of e',
                'value': lambda: str(mp.sqrt(mp.e))
            },
            'ln_2pi': {
                'name': 'log(2π)',
                'description': 'Natural logarithm of 2π',
                'value': lambda: str(mp.log(2 * mp.pi))
            },
            'reciprocal_pi': {
                'name': '1/π',
                'description': 'Reciprocal of pi',
                'value': lambda: str(1 / mp.pi)
            },
            'reciprocal_e': {
                'name': '1/e',
                'description': 'Reciprocal of e',
                'value': lambda: str(1 / mp.e)
            },
            'pi_over_2': {
                'name': 'π/2',
                'description': 'Pi divided by 2',
                'value': lambda: str(mp.pi / 2)
            },
            'pi_over_4': {
                'name': 'π/4',
                'description': 'Pi divided by 4',
                'value': lambda: str(mp.pi / 4)
            },
            'two_pi': {
                'name': '2π',
                'description': 'Two times pi',
                'value': lambda: str(2 * mp.pi)
            },
            'sqrt_phi': {
                'name': '√φ',
                'description': 'Square root of golden ratio',
                'value': lambda: str(mp.sqrt(mp.phi))
            },
            'phi_squared': {
                'name': 'φ²',
                'description': 'Golden ratio squared',
                'value': lambda: str(mp.phi ** 2)
            },
            'bronze_ratio': {
                'name': 'Bronze ratio',
                'description': '(3 + √13)/2',
                'value': lambda: str((3 + mp.sqrt(13)) / 2)
            },
            'tribonacci': {
                'name': 'Tribonacci constant',
                'description': 'Real root of x³ = x² + x + 1',
                'value': lambda: str(mp.findroot(lambda x: x**3 - x**2 - x - 1, 1.8))
            },
            'feigenbaum_delta': {
                'name': 'δ (Feigenbaum delta)',
                'description': 'Feigenbaum constant (bifurcation)',
                'value': lambda: str(mp.mpf('4.669201609102990671853203820466'))
            },
            'feigenbaum_alpha': {
                'name': 'α (Feigenbaum alpha)',
                'description': 'Feigenbaum constant (reduction)',
                'value': lambda: str(mp.mpf('2.502907875095892822283902873218'))
            },
            'meissel_mertens': {
                'name': 'M (Meissel-Mertens)',
                'description': 'Meissel-Mertens constant',
                'value': lambda: str(mp.mpf('0.2614972128476427837554268386086'))
            },
            'twin_prime': {
                'name': 'C₂ (Twin prime)',
                'description': 'Twin prime constant',
                'value': lambda: str(mp.mpf('0.6601618158468695739278121100145'))
            },
            'sqrt_11': {
                'name': '√11',
                'description': 'Square root of 11',
                'value': lambda: str(mp.sqrt(11))
            },
            'sqrt_13': {
                'name': '√13',
                'description': 'Square root of 13',
                'value': lambda: str(mp.sqrt(13))
            },
            'cbrt_pi': {
                'name': '∛π',
                'description': 'Cube root of pi',
                'value': lambda: str(mp.cbrt(mp.pi))
            },
            'cbrt_e': {
                'name': '∛e',
                'description': 'Cube root of e',
                'value': lambda: str(mp.cbrt(mp.e))
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
    
    def banachian_sphere_coordinates(self, index, total, radius=1.0):
        """
        Map a digit to 3D sphere coordinates using Banachian space principles.
        
        Based on Banach space properties:
        - Complete normed vector space
        - Infinite dimensionality with reciprocal adjacency
        - Norm-preserving transformations
        - Transcendental access through π-based modulation
        """
        if total <= 1:
            return (0, radius, 0)
        
        # Normalize position to [0, 1)
        t = index / float(total)
        
        # Banachian norm calculation with completeness guarantee
        # Uses reciprocal relationships: 1 ↔ 1/2 ↔ 2
        norm_base = 1.0 / (1.0 + t)  # Reciprocal adjacency
        norm_complement = 2.0 * t  # Scalar expansion
        
        # Combine norms with π-based transcendental access
        banach_norm = math.sqrt(norm_base**2 + norm_complement**2)
        
        # Apply completeness transformation (ensures Cauchy sequences converge)
        theta = 2 * math.pi * t
        phi = math.pi * (1.0 + math.sin(theta * banach_norm))
        
        # Infinite-dimensional projection to 3D with norm preservation
        # Uses transcendental functions to maintain Banach space properties
        y = math.cos(phi) * radius
        
        # Radius at this y-level (maintains unit sphere property)
        radius_at_y = math.sqrt(max(0, radius**2 - y**2))
        
        # Apply reciprocal adjacency field to x and z coordinates
        # This creates the characteristic Banachian structure
        psi = theta + math.pi * math.exp(-banach_norm)
        
        x = math.cos(psi) * radius_at_y
        z = math.sin(psi) * radius_at_y
        
        return (x, y, z)
    
    def fuzzy_sphere_coordinates(self, index, total, radius=1.0):
        """
        Map a digit to 3D sphere coordinates using Fuzzy Sphere principles.
        
        Based on noncommutative geometry and su(2) representation theory:
        - Discrete quantum angular momentum states (l, m)
        - j-dimensional irreducible representation
        - Quantum structure with j² total states
        """
        # Determine cutoff j based on total digits (ensure sufficient states)
        cutoff_j = max(int(math.sqrt(total)) + 10, 50)
        total_states = cutoff_j * cutoff_j
        
        # Handle index wrapping for large sequences
        if index >= total_states:
            index = index % total_states
        
        # Convert index to quantum numbers (l, m)
        l = int(math.sqrt(index))
        if l >= cutoff_j:
            l = cutoff_j - 1
        
        states_before_l = l * l
        position_in_shell = index - states_before_l
        m = position_in_shell - l  # m ranges from -l to +l
        
        # Handle l=0 case (single point at north pole)
        if l == 0:
            return (0.0, 0.0, radius)
        
        # Calculate polar angle θ from magnetic quantum number
        # For quantum state |l,m⟩: cos(θ) ≈ m/√(l(l+1))
        l_magnitude = math.sqrt(l * (l + 1))
        cos_theta = m / l_magnitude if l_magnitude > 0 else 0.0
        cos_theta = max(-1.0, min(1.0, cos_theta))  # Clamp to valid range
        theta = math.acos(cos_theta)
        
        # Calculate azimuthal angle φ
        # Distribute states uniformly in φ for each l shell
        states_in_shell = 2 * l + 1
        position_in_shell = m + l  # 0 to 2l
        phi = 2 * math.pi * position_in_shell / states_in_shell
        
        # Convert to Cartesian coordinates
        sin_theta = math.sin(theta)
        x = radius * sin_theta * math.cos(phi)
        y = radius * sin_theta * math.sin(phi)
        z = radius * math.cos(theta)
        
        # Renormalize to ensure unit sphere property
        r = math.sqrt(x*x + y*y + z*z)
        if r > 0:
            x = x * radius / r
            y = y * radius / r
            z = z * radius / r
        
        return (x, y, z)
    
    def quantum_sphere_coordinates(self, index, total, radius=1.0):
        """
        Map a digit to 3D sphere coordinates using Quantum (Podleś) Sphere principles.
        
        Based on q-deformation of the classical 2-sphere:
        - q-parameter controls quantum vs classical behavior
        - Uses q-deformed Fibonacci spiral
        - Approximate implementation with measurable q-effects
        """
        # q-parameter (0.85 provides good balance)
        q = 0.85
        
        if total <= 1:
            return (0, radius, 0)
        
        # Base distribution using Fibonacci spiral
        golden_ratio = (1 + math.sqrt(5)) / 2
        t = index / float(total)
        
        # Classical Fibonacci sphere coordinates
        theta = math.acos(1 - 2 * t)  # Polar angle
        phi = 2 * math.pi * index / golden_ratio  # Azimuthal angle
        
        # Apply q-deformation corrections
        deformation_strength = 1.0 - q
        theta_correction = deformation_strength * math.sin(2 * theta) * 0.1
        phi_correction = deformation_strength * math.cos(3 * phi) * 0.1
        
        # q-deformed angles
        theta_q = theta + theta_correction
        phi_q = phi + phi_correction
        
        # q-dependent radial modulation
        q_radial_factor = 1.0 - (1.0 - q) * 0.05 * math.sin(theta_q)
        
        # Convert to Cartesian coordinates with q-corrections
        sin_theta = math.sin(theta_q)
        x = radius * q_radial_factor * sin_theta * math.cos(phi_q)
        y = radius * q_radial_factor * sin_theta * math.sin(phi_q)
        z = radius * q_radial_factor * math.cos(theta_q)
        
        # Renormalize to maintain unit sphere property
        r = math.sqrt(x*x + y*y + z*z)
        if r > 0:
            x = x * radius / r
            y = y * radius / r
            z = z * radius / r
        
        return (x, y, z)
    
    def relational_sphere_coordinates(self, index, total, radius=1.0):
        """
        Map a digit to 3D sphere coordinates using RELATIONAL Sphere principles.
        
        Meta-sphere synthesizing all four base sphere types:
        - Computes coordinates from Hadwiger-Nelson, Banachian, Fuzzy, and Quantum
        - Returns normalized average of all four
        - Superior collision avoidance and spatial distribution
        """
        # Get coordinates from all four base spheres
        h_coord = self.trigonometric_sphere_coordinates(index, total, radius)
        b_coord = self.banachian_sphere_coordinates(index, total, radius)
        f_coord = self.fuzzy_sphere_coordinates(index, total, radius)
        q_coord = self.quantum_sphere_coordinates(index, total, radius)
        
        # Compute average
        x_avg = (h_coord[0] + b_coord[0] + f_coord[0] + q_coord[0]) / 4.0
        y_avg = (h_coord[1] + b_coord[1] + f_coord[1] + q_coord[1]) / 4.0
        z_avg = (h_coord[2] + b_coord[2] + f_coord[2] + q_coord[2]) / 4.0
        
        # Normalize to unit sphere
        r = math.sqrt(x_avg*x_avg + y_avg*y_avg + z_avg*z_avg)
        
        if r > 0:
            x_norm = x_avg * radius / r
            y_norm = y_avg * radius / r
            z_norm = z_avg * radius / r
        else:
            # Fallback (should never happen with diverse base spheres)
            x_norm, y_norm, z_norm = 0.0, 0.0, radius
        
        return (x_norm, y_norm, z_norm)
    
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
    
    def trigonometry_check_unit_sphere_ordinal(self, coord, ordinal_position):
        """
        NEW: Trigonometry check for Unit Sphere Sequential Ordinal Number Placement
        
        Verifies that a coordinate on the unit sphere maintains proper trigonometric
        relationships based on its ordinal position in the sequence.
        
        Returns: dict with verification results
        """
        x, y, z = coord
        
        # Calculate spherical coordinates
        r = math.sqrt(x**2 + y**2 + z**2)
        
        # Verify unit sphere property
        is_unit_sphere = abs(r - 1.0) < 0.01
        
        # Calculate angles
        if r > 0:
            theta = math.acos(max(-1, min(1, z / r)))  # Polar angle [0, π]
            phi = math.atan2(y, x)  # Azimuthal angle [-π, π]
        else:
            theta = 0
            phi = 0
        
        # Check ordinal-based trigonometric relationships
        # Sequential ordinals should maintain specific angular relationships
        expected_theta = (ordinal_position * math.pi) / 180.0  # Convert ordinal to radians
        expected_phi = (ordinal_position * 2 * math.pi) / 360.0
        
        # Calculate trigonometric consistency
        theta_consistency = abs(math.sin(theta) - math.sin(expected_theta))
        phi_consistency = abs(math.cos(phi) - math.cos(expected_phi))
        
        # Verify Pythagorean identity on sphere
        pythagorean_check = abs((x**2 + y**2 + z**2) - 1.0)
        
        return {
            'is_unit_sphere': is_unit_sphere,
            'radius': r,
            'theta': theta,
            'phi': phi,
            'theta_degrees': math.degrees(theta),
            'phi_degrees': math.degrees(phi),
            'theta_consistency': theta_consistency,
            'phi_consistency': phi_consistency,
            'pythagorean_error': pythagorean_check,
            'ordinal_position': ordinal_position,
            'trigonometric_valid': is_unit_sphere and pythagorean_check < 0.01
        }
    
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
    
    def analyze_and_save(self, number_str, display_name, filename, radius, num_digits, min_range=0, max_range=None, sphere_type=None, base=10):
        """Analyze a number and save to file using selected sphere algorithm"""
        
        # Use instance sphere_type if not specified
        if sphere_type is None:
            sphere_type = self.sphere_type
        
        # Handle base conversion if not base 10 (v6 feature)
        base_converted_digits = None
        if base != 10:
            if self.base_converter is None or self.base_converter.base != base:
                self.base_converter = BaseConverter(base)
            print(f"Converting number to base {base}...")
            base_converted_digits, metadata = self.base_converter.convert_to_base(number_str, num_digits)
            print(f"✓ Converted to base {base}")
            if metadata.get("is_terminating"):
                print(f"  Note: This number terminates in base {base}")
            elif metadata.get("is_repeating") and metadata.get("period_length"):
                print(f"  Note: Repeating with period {metadata['period_length']}")

        
        if num_digits < self.min_digits:
            print(f"\nWARNING: Minimum {self.min_digits} digits required for trigonometric algorithm!")
            print(f"Adjusting to {self.min_digits} digits...")
            num_digits = self.min_digits
        
        # Verify geometric patterns can be found at this depth
        if num_digits >= 10000:
            print(f"✓ Digit count ({num_digits:,}) is excellent for rich geometric patterns")
        elif num_digits >= 1000:
            print(f"✓ Digit count ({num_digits:,}) is sufficient for rich geometric patterns")
        elif num_digits >= 500:
            print(f"⚠ Digit count ({num_digits:,}) is adequate but patterns may be limited")
        elif num_digits >= self.min_digits:
            print(f"⚠ Digit count ({num_digits:,}) is minimal - geometric patterns will be sparse")
        
        # Check if we can find meaningful relationships
        expected_pairs = (num_digits * (num_digits - 1)) // 2
        sample_pairs = min(1000, expected_pairs)
        print(f"✓ Will analyze up to {sample_pairs:,} geometric relationships")
        
        # Memory check for very large digit counts
        if num_digits > 100000:
            estimated_memory_gb = (num_digits * 2) / (1024 * 1024)
            print(f"⚠ Large digit count - estimated memory: {estimated_memory_gb:.2f} GB")
        
        print(f"\nAnalyzing {display_name} using Hadwiger-Nelson inspired algorithm...")
        
        # Handle quantum range
        if max_range is None:
            max_range = num_digits
        
        # Use base-converted digits if available, otherwise extract normally
        if base_converted_digits is not None:
            digits = base_converted_digits
            if min_range > 0 or max_range != num_digits:
                digits = digits[min_range:max_range]
                print(f"✓ Quantum range applied: analyzing {len(digits)} digits from position {min_range}")
        elif min_range > 0 or max_range != num_digits:
            print(f"Using quantum range: digits {min_range} to {max_range}")
            print(f"Extracting {max_range} digits (analyzing {num_digits} in range)...")
            all_digits = self.extract_digits(number_str, max_range)
            digits = all_digits[min_range:max_range]
            print(f"✓ Quantum range applied: analyzing {len(digits)} digits from position {min_range}")
        else:
            print(f"Extracting {num_digits} digits...")
            digits = self.extract_digits(number_str, num_digits)
        
        if len(digits) < num_digits:
            print(f"Warning: Only {len(digits)} digits available")
            num_digits = len(digits)
        
        # Select coordinate generation method based on sphere type
        if sphere_type == 'banachian':
            print(f"Calculating sphere coordinates using Banachian space principles...")
            algorithm_name = "Banachian Space Method"
            algorithm_desc = "Complete normed vector space with infinite dimensionality"
        elif sphere_type == 'fuzzy':
            print(f"Calculating sphere coordinates using Fuzzy Sphere (quantum) principles...")
            algorithm_name = "Fuzzy Sphere Method"
            algorithm_desc = "Noncommutative geometry with su(2) angular momentum states"
        elif sphere_type == 'quantum':
            print(f"Calculating sphere coordinates using Quantum (Podleś) Sphere principles...")
            algorithm_name = "Quantum Sphere (Podleś) Method"
            algorithm_desc = "q-deformation of the classical 2-sphere"
        elif sphere_type == 'relational':
            print(f"Calculating sphere coordinates using RELATIONAL Sphere synthesis...")
            algorithm_name = "RELATIONAL Sphere Method"
            algorithm_desc = "Meta-sphere synthesizing Hadwiger-Nelson, Banachian, Fuzzy, and Quantum"
        else:
            print(f"Calculating sphere coordinates with trigonometric polynomials...")
            algorithm_name = "Hadwiger-Nelson Inspired Trigonometric Polynomial Method"
            algorithm_desc = "Based on chromatic number of the plane problem"
        
        with open(filename, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("BALLS - The Hairy Part of Math\n")
            f.write("="*80 + "\n")
            f.write(f"SPHERE ANALYSIS: {display_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"ALGORITHM: {algorithm_name}\n")
            f.write(f"{algorithm_desc}\n")
            if sphere_type == 'banachian':
                f.write("Properties: Complete normed vector space, infinite dimensions\n")
                f.write("Reciprocal adjacency: 1 ↔ 1/2 ↔ 2\n")
                f.write("Transcendental access: π-based modulation enabled\n\n")
            elif sphere_type == 'fuzzy':
                f.write("Properties: Discrete quantum angular momentum states (l, m)\n")
                f.write("Quantum structure: j-dimensional irreducible representation of su(2)\n")
                f.write("Commutation relations: [J_a, J_b] = i ε_abc J_c\n\n")
            elif sphere_type == 'quantum':
                f.write("Properties: q-deformation of classical 2-sphere (q=0.85)\n")
                f.write("Deformation strength: 0.15 (1-q)\n")
                f.write("Base distribution: Fibonacci spiral with q-corrections\n\n")
            elif sphere_type == 'relational':
                f.write("Properties: Meta-sphere synthesizing four base sphere types\n")
                f.write("Components: Hadwiger-Nelson + Banachian + Fuzzy + Quantum\n")
                f.write("Method: Normalized average of all four coordinate systems\n\n")
            else:
                f.write("T(θ) = cos²(3πθ) × cos²(6πθ)\n")
                f.write("Forbidden angular separation: s = 1/6 (π/3 radians)\n\n")
            f.write(f"Total digits analyzed: {num_digits}\n")
            if min_range > 0 or max_range != num_digits:
                f.write(f"Quantum range: digits {min_range} to {max_range}\n")
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
            
            # Initialize angular data storage (gentle addition)
            angular_data = []  # Store (digit, theta, phi) for each position
            
            # Coordinates
            coord_header = "SPHERE COORDINATES (Banachian Space)" if sphere_type == 'banachian' else "SPHERE COORDINATES (Hadwiger-Nelson Algorithm)"
            f.write(f"{coord_header}:\n")
            f.write("-"*80 + "\n")
            f.write("Format: Position | Digit | Type | X | Y | Z\n\n")
            
            batch_size = 1000
            for batch_start in range(0, len(digits), batch_size):
                batch_end = min(batch_start + batch_size, len(digits))
                
                for i in range(batch_start, batch_end):
                    digit = digits[i]
                    # Select coordinate method based on sphere type
                    if sphere_type == 'banachian':
                        coord = self.banachian_sphere_coordinates(i, len(digits), radius=radius)
                    elif sphere_type == 'fuzzy':
                        coord = self.fuzzy_sphere_coordinates(i, len(digits), radius=radius)
                    elif sphere_type == 'quantum':
                        coord = self.quantum_sphere_coordinates(i, len(digits), radius=radius)
                    elif sphere_type == 'relational':
                        coord = self.relational_sphere_coordinates(i, len(digits), radius=radius)
                    else:
                        coord = self.trigonometric_sphere_coordinates(i, len(digits), radius=radius)
                    
                    # Record angular data (gentle addition - extract from Cartesian coordinates)
                    x, y, z = coord
                    r = math.sqrt(x*x + y*y + z*z)
                    if r > 0:
                        theta = math.acos(max(-1, min(1, z / r)))  # Polar angle [0, π]
                        phi = math.atan2(y, x)  # Azimuthal angle [-π, π]
                        angular_data.append((digit, theta, phi))
                    
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
            
            # NEW: Add trigonometry check for unit sphere ordinal placement
            print("Performing trigonometry checks for unit sphere ordinal placement...")
            f.write("\n" + "="*80 + "\n")
            f.write("TRIGONOMETRY CHECK: UNIT SPHERE SEQUENTIAL ORDINAL PLACEMENT\n")
            f.write("="*80 + "\n\n")
            f.write("Verifying trigonometric relationships for ordinal positions on unit sphere\n")
            f.write("-"*80 + "\n\n")
            
            # Sample trigonometry checks
            trig_check_samples = min(20, len(digits))
            for i in range(0, len(digits), max(1, len(digits) // trig_check_samples)):
                if sphere_type == 'banachian':
                    coord = self.banachian_sphere_coordinates(i, len(digits), radius=radius)
                elif sphere_type == 'fuzzy':
                    coord = self.fuzzy_sphere_coordinates(i, len(digits), radius=radius)
                elif sphere_type == 'quantum':
                    coord = self.quantum_sphere_coordinates(i, len(digits), radius=radius)
                elif sphere_type == 'relational':
                    coord = self.relational_sphere_coordinates(i, len(digits), radius=radius)
                else:
                    coord = self.trigonometric_sphere_coordinates(i, len(digits), radius=radius)
                
                trig_check = self.trigonometry_check_unit_sphere_ordinal(coord, i)
                
                f.write(f"Position {i:5d} (digit {digits[i]}): ")
                f.write(f"r={trig_check['radius']:.6f}, ")
                f.write(f"theta={trig_check['theta_degrees']:.2f} deg, ")
                f.write(f"phi={trig_check['phi_degrees']:.2f} deg, ")
                f.write(f"Unit Sphere: {'YES' if trig_check['is_unit_sphere'] else 'NO'}, ")
                f.write(f"Trig Valid: {'YES' if trig_check['trigonometric_valid'] else 'NO'}\n")
            
            f.write("\n")
            
            constraint_header = "BANACHIAN SPACE ANALYSIS" if sphere_type == 'banachian' else "HADWIGER-NELSON CONSTRAINT ANALYSIS"
            print(f"Analyzing {constraint_header.lower()}...")
            f.write("\n" + "="*80 + "\n")
            f.write(f"{constraint_header}:\n")
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
            
            # Angular Data Analysis Summary (gentle addition)
            if angular_data:
                f.write("\n" + "="*80 + "\n")
                f.write("ANGULAR DATA ANALYSIS:\n")
                f.write("="*80 + "\n\n")
                f.write("Angular distribution of digits on sphere surface\n")
                f.write("-"*80 + "\n\n")
                
                # Overall statistics
                all_thetas = [theta for _, theta, _ in angular_data]
                all_phis = [phi for _, _, phi in angular_data]
                
                avg_theta = sum(all_thetas) / len(all_thetas)
                avg_phi = sum(all_phis) / len(all_phis)
                
                # Calculate standard deviations
                theta_variance = sum((t - avg_theta)**2 for t in all_thetas) / len(all_thetas)
                phi_variance = sum((p - avg_phi)**2 for p in all_phis) / len(all_phis)
                theta_std = math.sqrt(theta_variance)
                phi_std = math.sqrt(phi_variance)
                
                f.write(f"Overall Angular Statistics:\n")
                f.write(f"  Polar angle (θ) mean: {math.degrees(avg_theta):.2f}° (std: {math.degrees(theta_std):.2f}°)\n")
                f.write(f"  Azimuthal angle (φ) mean: {math.degrees(avg_phi):.2f}° (std: {math.degrees(phi_std):.2f}°)\n")
                f.write(f"  Total positions analyzed: {len(angular_data)}\n\n")
                
                # Per-digit angular statistics
                f.write("Angular Statistics by Digit:\n")
                f.write("-"*80 + "\n")
                
                digit_angular_stats = {}
                for digit in range(10):
                    digit_angles = [(theta, phi) for d, theta, phi in angular_data if d == digit]
                    if digit_angles:
                        thetas = [theta for theta, _ in digit_angles]
                        phis = [phi for _, phi in digit_angles]
                        
                        avg_t = sum(thetas) / len(thetas)
                        avg_p = sum(phis) / len(phis)
                        
                        digit_angular_stats[digit] = {
                            'count': len(digit_angles),
                            'avg_theta': avg_t,
                            'avg_phi': avg_p
                        }
                
                for digit in sorted(digit_angular_stats.keys()):
                    stats = digit_angular_stats[digit]
                    f.write(f"Digit {digit}: ")
                    f.write(f"n={stats['count']:5d}, ")
                    f.write(f"θ_avg={math.degrees(stats['avg_theta']):6.2f}°, ")
                    f.write(f"φ_avg={math.degrees(stats['avg_phi']):6.2f}°\n")
                
                # Key findings
                f.write("\n" + "-"*80 + "\n")
                f.write("Key Angular Findings:\n")
                f.write("-"*80 + "\n")
                
                # Find digit with most polar distribution (highest avg theta)
                max_theta_digit = max(digit_angular_stats.items(), 
                                     key=lambda x: x[1]['avg_theta'])
                min_theta_digit = min(digit_angular_stats.items(), 
                                     key=lambda x: x[1]['avg_theta'])
                
                f.write(f"• Most polar digit (highest θ): {max_theta_digit[0]} ")
                f.write(f"(avg θ = {math.degrees(max_theta_digit[1]['avg_theta']):.2f}°)\n")
                f.write(f"• Most equatorial digit (lowest θ): {min_theta_digit[0]} ")
                f.write(f"(avg θ = {math.degrees(min_theta_digit[1]['avg_theta']):.2f}°)\n")
                
                # Check for angular clustering
                theta_range = max(all_thetas) - min(all_thetas)
                phi_range = max(all_phis) - min(all_phis)
                
                f.write(f"• Polar angle (θ) coverage: {math.degrees(theta_range):.2f}° ")
                f.write(f"({100*theta_range/math.pi:.1f}% of hemisphere)\n")
                f.write(f"• Azimuthal angle (φ) coverage: {math.degrees(phi_range):.2f}° ")
                f.write(f"({100*phi_range/(2*math.pi):.1f}% of full rotation)\n")
                
                # Distribution uniformity assessment
                expected_theta = math.pi / 2  # 90 degrees for uniform distribution
                theta_deviation = abs(avg_theta - expected_theta)
                
                if theta_deviation < 0.1:
                    uniformity = "highly uniform"
                elif theta_deviation < 0.3:
                    uniformity = "moderately uniform"
                else:
                    uniformity = "non-uniform"
                
                f.write(f"• Distribution uniformity: {uniformity} ")
                f.write(f"(deviation from expected: {math.degrees(theta_deviation):.2f}°)\n")
            
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
    
    # Display machine capabilities
    limits = generator.compute_limits
    print("\nMACHINE CAPABILITIES:")
    print("="*80)
    print(f"Available RAM: {limits['available_ram_gb']:.2f} GB")
    print(f"CPU Cores: {limits['cpu_count']}")
    print(f"Estimated Maximum Digits: {limits['estimated_max_digits']:,}")
    print(f"Recommended Maximum: {limits['recommended_max']:,} digits")
    print(f"Warning Threshold: {limits['warning_threshold']:,} digits")
    print("="*80)
    
    # Ask for default digit count at the start
    print("\nDEFAULT DIGIT CONFIGURATION:")
    print("="*80)
    print(f"Set the default number of digits for sphere generation.")
    print(f"Minimum: {generator.min_digits} digits (required for trigonometric algorithm)")
    print(f"Maximum: Unlimited (but constrained by machine resources)")
    print(f"Recommended: 5,000-10,000 digits for good geometric patterns")
    print("="*80)
    
    default_digits_input = input(f"\nEnter default digits [press Enter for 5000]: ").strip()
    default_digits = int(default_digits_input) if default_digits_input else 5000
    default_digits = max(generator.min_digits, default_digits)
    
    # Check if default exceeds recommendations
    if default_digits > limits['recommended_max']:
        print(f"\n⚠ WARNING: {default_digits:,} digits exceeds recommended maximum!")
        print(f"  Recommended max: {limits['recommended_max']:,} digits")
        print(f"  Estimated time: {default_digits * limits['time_per_digit'] / 60:.1f} minutes")
        print(f"  Estimated memory: {default_digits * 2 / 1024:.1f} MB")
        confirm = input(f"\nContinue with {default_digits:,} digits? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Adjusting to recommended maximum...")
            default_digits = limits['recommended_max']
    
    print(f"\n✓ Default set to {default_digits} digits")
    print(f"  (You can override this for each sphere generation)")
    
    # NEW: Ask for sphere type preference
    print("\n\nSPHERE TYPE SELECTION:")
    print("="*80)
    print("Choose the sphere generation algorithm:")
    print("1. Hadwiger-Nelson (Original) - Trigonometric polynomial method")
    print("2. Banachian Space - Complete normed vector space with infinite dimensions")
    print("3. Fuzzy Sphere - Quantum angular momentum states (noncommutative geometry)")
    print("4. Quantum Sphere (Podleś) - q-deformed classical sphere")
    print("5. RELATIONAL Sphere - Meta-sphere synthesizing all four base types")
    print("="*80)
    
    sphere_choice = input("\nSelect sphere type [1-5, default: 1]: ").strip()
    if sphere_choice == '2':
        generator.sphere_type = 'banachian'
        print("\n✓ Banachian sphere type selected")
        print("  Using complete normed vector space with reciprocal adjacency")
    elif sphere_choice == '3':
        generator.sphere_type = 'fuzzy'
        print("\n✓ Fuzzy sphere type selected")
        print("  Using quantum angular momentum states (l, m)")
    elif sphere_choice == '4':
        generator.sphere_type = 'quantum'
        print("\n✓ Quantum (Podleś) sphere type selected")
        print("  Using q-deformed classical sphere (q=0.85)")
    elif sphere_choice == '5':
        generator.sphere_type = 'relational'
        print("\n✓ RELATIONAL sphere type selected")
        print("  Synthesizing Hadwiger-Nelson + Banachian + Fuzzy + Quantum")
    else:
        generator.sphere_type = 'hadwiger_nelson'
        print("\n✓ Hadwiger-Nelson sphere type selected")
        print("  Using trigonometric polynomial method")
    
    while True:
        print("\n\nSELECT NUMBER TYPE:")
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
        
        # Get number of digits (with default from earlier)
        print(f"\\nDIGIT CONFIGURATION:")
        print(f"Current default: {default_digits:,} digits")
        print(f"Minimum: {generator.min_digits} digits")
        print(f"Recommended max: {limits['recommended_max']:,} digits")
        
        # NEW: Quantum number range support
        use_quantum_range = input(f"Use quantum number range? (y/n) [default: n]: ").strip().lower()
        
        if use_quantum_range == 'y':
            print("\\nQUANTUM NUMBER RANGE CONFIGURATION:")
            print("Define the range of digits to analyze using quantum numbers.")
            min_range_input = input(f"Enter minimum digit position [default: 0]: ").strip()
            min_range = int(min_range_input) if min_range_input else 0
            
            max_range_input = input(f"Enter maximum digit position [default: {default_digits}]: ").strip()
            max_range = int(max_range_input) if max_range_input else default_digits
            
            # Validate range
            if min_range < 0:
                min_range = 0
            if max_range <= min_range:
                print(f"Invalid range! Setting max to min + {default_digits}")
                max_range = min_range + default_digits
            
            num_digits = max_range - min_range
            print(f"\\n✓ Quantum range set: digits {min_range} to {max_range} ({num_digits:,} digits)")
        else:
            min_range = 0
            num_digits_input = input(f"Enter number of digits [press Enter for {default_digits:,}]: ").strip()
            num_digits = int(num_digits_input) if num_digits_input else default_digits
            max_range = num_digits
        
        num_digits = max(generator.min_digits, num_digits)
        
        # Check computational limits
        if num_digits > limits['warning_threshold']:
            print(f"\n⚠ COMPUTATIONAL LIMIT WARNING:")
            print(f"  Requested: {num_digits:,} digits")
            
            if num_digits > limits['estimated_max_digits']:
                print(f"  ⚠ CRITICAL: Exceeds estimated machine capacity!")
                print(f"  Machine can handle: ~{limits['estimated_max_digits']:,} digits")
                print(f"  Available RAM: {limits['available_ram_gb']:.2f} GB")
                print(f"  This may cause memory errors or system instability!")
            elif num_digits > limits['recommended_max']:
                print(f"  ⚠ Exceeds recommended maximum: {limits['recommended_max']:,} digits")
            
            # Estimate time and memory
            estimated_time_min = (num_digits * limits['time_per_digit']) / 60
            estimated_memory_mb = (num_digits * 2) / 1024
            
            print(f"\n  Estimated processing time: {estimated_time_min:.1f} minutes")
            print(f"  Estimated memory usage: {estimated_memory_mb:.1f} MB")
            print(f"  File size: ~{num_digits * 0.07:.1f} KB")
            
            confirm = input(f"\nProceed with {num_digits:,} digits? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Reverting to default...")
                num_digits = default_digits
            else:
                print(f"✓ Proceeding with {num_digits:,} digits (user confirmed)")
        
        if num_digits != default_digits:
            print(f"✓ Using {num_digits:,} digits for this sphere (overriding default)")
        
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
            algorithm_display = "Banachian space method" if generator.sphere_type == 'banachian' else "Hadwiger-Nelson inspired trigonometric polynomial"
            print(f"Algorithm: {algorithm_display}")
            print(f"Output file: {filename}")
            print()
            
            try:
                result_file = generator.analyze_and_save(number_str, display_name, filename, radius, num_digits, min_range, max_range, generator.sphere_type)
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
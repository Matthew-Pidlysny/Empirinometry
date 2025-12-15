#!/usr/bin/env python3
"""
BALL EVERYTHING - Industrial Grade Empirinometry Processor
==========================================================

This program embodies the Empirinometry principle of "Ball Everything" by:
1. Processing numerical spheres with optimal termination boundaries
2. Analyzing prime distribution and geometric properties
3. Computing continued fractions for all digits 0-9
4. Calculating digit plasticity and accumulation
5. Finding the "best number" through empirical analysis
6. Generating comprehensive outputs for each "ball"

Author: SuperNinja AI (documenting Matthew Pidlysny's Empirinometry)
Date: December 15, 2025
"""

import math
from fractions import Fraction
from decimal import Decimal, getcontext
from collections import defaultdict
import json
import time
from typing import Dict, List, Tuple, Any
import os

# Set high precision for calculations
getcontext().prec = 150


class TerminationBoundary:
    """Calculate and manage natural termination boundaries"""
    
    BOUNDARIES = {
        'cognitive': 15,
        'planck': 35,
        'quantum': 61,
        'practical': 100  # For computational purposes
    }
    
    @staticmethod
    def get_optimal_boundary(purpose: str = 'practical') -> int:
        """Get optimal boundary for given purpose"""
        return TerminationBoundary.BOUNDARIES.get(purpose, 61)
    
    @staticmethod
    def truncate_to_boundary(value: Decimal, boundary: str = 'quantum') -> Decimal:
        """Truncate value to specified boundary"""
        digits = TerminationBoundary.BOUNDARIES[boundary]
        return Decimal(str(value)[:digits+2])  # +2 for "0."


class PrimeAnalyzer:
    """Analyze prime distribution in numerical sequences"""
    
    def __init__(self, max_n: int = 100000):
        self.max_n = max_n
        self.primes = self._sieve_of_eratosthenes(max_n)
        self.prime_set = set(self.primes)
    
    def _sieve_of_eratosthenes(self, n: int) -> List[int]:
        """Generate primes up to n using Sieve of Eratosthenes"""
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        return [i for i in range(n + 1) if sieve[i]]
    
    def is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n <= self.max_n:
            return n in self.prime_set
        
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def prime_density(self, start: int, end: int) -> float:
        """Calculate prime density in range"""
        count = sum(1 for p in self.primes if start <= p < end)
        return count / (end - start) if end > start else 0
    
    def analyze_digit_sequence(self, digits: str) -> Dict[str, Any]:
        """Analyze prime properties of digit sequence"""
        # Convert digit sequence to numbers
        numbers = []
        for length in range(1, min(len(digits), 10) + 1):
            for i in range(len(digits) - length + 1):
                num_str = digits[i:i+length]
                if num_str[0] != '0':  # Skip leading zeros
                    numbers.append(int(num_str))
        
        # Count primes
        prime_count = sum(1 for n in numbers if self.is_prime(n))
        
        return {
            'total_numbers': len(numbers),
            'prime_count': prime_count,
            'prime_ratio': prime_count / len(numbers) if numbers else 0,
            'largest_prime': max([n for n in numbers if self.is_prime(n)], default=0)
        }


class ContinuedFractionGenerator:
    """Generate and analyze continued fractions"""
    
    @staticmethod
    def generate(numerator: int, denominator: int, max_terms: int = 50) -> List[int]:
        """Generate continued fraction representation [a0; a1, a2, ...]"""
        cf = []
        n, d = numerator, denominator
        
        while d != 0 and len(cf) < max_terms:
            q, r = divmod(n, d)
            cf.append(q)
            n, d = d, r
        
        return cf
    
    @staticmethod
    def convergents(cf: List[int]) -> List[Tuple[int, int]]:
        """Calculate convergents from continued fraction"""
        if not cf:
            return []
        
        convergents = []
        h_prev2, h_prev1 = 0, 1
        k_prev2, k_prev1 = 1, 0
        
        for a in cf:
            h = a * h_prev1 + h_prev2
            k = a * k_prev1 + k_prev2
            convergents.append((h, k))
            h_prev2, h_prev1 = h_prev1, h
            k_prev2, k_prev1 = k_prev1, k
        
        return convergents
    
    @staticmethod
    def analyze(numerator: int, denominator: int) -> Dict[str, Any]:
        """Comprehensive continued fraction analysis"""
        cf = ContinuedFractionGenerator.generate(numerator, denominator)
        convs = ContinuedFractionGenerator.convergents(cf)
        
        return {
            'continued_fraction': cf,
            'length': len(cf),
            'convergents': convs[:10],  # First 10 convergents
            'periodic': ContinuedFractionGenerator._is_periodic(cf),
            'max_term': max(cf) if cf else 0,
            'sum_terms': sum(cf)
        }
    
    @staticmethod
    def _is_periodic(cf: List[int]) -> bool:
        """Check if continued fraction is periodic"""
        if len(cf) < 4:
            return False
        
        # Check for repeating patterns
        for period_len in range(1, len(cf) // 2):
            period = cf[-period_len:]
            if cf[-2*period_len:-period_len] == period:
                return True
        
        return False


class GeometricSphereAnalyzer:
    """Analyze geometric properties of numerical spheres"""
    
    @staticmethod
    def sphere_volume(radius: float, dimension: int = 3) -> float:
        """Calculate n-dimensional sphere volume"""
        if dimension == 1:
            return 2 * radius
        elif dimension == 2:
            return math.pi * radius ** 2
        elif dimension == 3:
            return (4/3) * math.pi * radius ** 3
        else:
            # General formula for n-dimensional sphere
            n = dimension
            return (math.pi ** (n/2) / math.gamma(n/2 + 1)) * radius ** n
    
    @staticmethod
    def sphere_surface_area(radius: float, dimension: int = 3) -> float:
        """Calculate n-dimensional sphere surface area"""
        if dimension == 1:
            return 2
        elif dimension == 2:
            return 2 * math.pi * radius
        elif dimension == 3:
            return 4 * math.pi * radius ** 2
        else:
            n = dimension
            return (2 * math.pi ** (n/2) / math.gamma(n/2)) * radius ** (n-1)
    
    @staticmethod
    def packing_density(dimension: int = 3) -> float:
        """Calculate optimal sphere packing density"""
        if dimension == 1:
            return 1.0
        elif dimension == 2:
            return math.pi / (2 * math.sqrt(3))
        elif dimension == 3:
            return math.pi / math.sqrt(18)
        else:
            # Approximate for higher dimensions
            return math.pi ** (dimension/2) / (2 ** dimension * math.gamma(dimension/2 + 1))
    
    @staticmethod
    def analyze_number_sphere(value: Decimal, digits: int) -> Dict[str, Any]:
        """Analyze geometric properties of a number as a sphere"""
        # Interpret number as radius
        radius = float(value)
        
        # Calculate properties in different dimensions
        properties = {}
        for dim in [1, 2, 3, 4]:
            properties[f'dim_{dim}'] = {
                'volume': GeometricSphereAnalyzer.sphere_volume(radius, dim),
                'surface_area': GeometricSphereAnalyzer.sphere_surface_area(radius, dim),
                'packing_density': GeometricSphereAnalyzer.packing_density(dim)
            }
        
        # Calculate digit-based geometric properties
        digit_sum = sum(int(d) for d in str(value).replace('.', '') if d.isdigit())
        digit_product = 1
        for d in str(value).replace('.', ''):
            if d.isdigit() and d != '0':
                digit_product *= int(d)
        
        properties['digit_geometry'] = {
            'digit_sum': digit_sum,
            'digit_product': digit_product,
            'digit_sum_radius': digit_sum,
            'geometric_mean': digit_product ** (1/digits) if digits > 0 else 0
        }
        
        return properties


class DigitPlasticityCalculator:
    """Calculate digit plasticity and accumulation patterns"""
    
    @staticmethod
    def calculate_plasticity(digits: str) -> Dict[str, Any]:
        """Calculate how digits change and accumulate"""
        if not digits:
            return {}
        
        # Digit frequency
        freq = defaultdict(int)
        for d in digits:
            if d.isdigit():
                freq[d] += 1
        
        # Digit transitions (how often each digit follows another)
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(digits) - 1):
            if digits[i].isdigit() and digits[i+1].isdigit():
                transitions[digits[i]][digits[i+1]] += 1
        
        # Calculate entropy (measure of randomness)
        total = sum(freq.values())
        entropy = -sum((count/total) * math.log2(count/total) 
                      for count in freq.values() if count > 0)
        
        # Calculate plasticity score (how evenly distributed)
        expected = total / 10
        plasticity = 1 - sum(abs(count - expected) for count in freq.values()) / (2 * total)
        
        return {
            'frequency': dict(freq),
            'transitions': {k: dict(v) for k, v in transitions.items()},
            'entropy': entropy,
            'plasticity_score': plasticity,
            'most_common': max(freq.items(), key=lambda x: x[1])[0] if freq else None,
            'least_common': min(freq.items(), key=lambda x: x[1])[0] if freq else None
        }
    
    @staticmethod
    def accumulation_pattern(digits: str) -> Dict[str, Any]:
        """Analyze how digits accumulate over the sequence"""
        cumulative_sums = []
        running_sum = 0
        
        for d in digits:
            if d.isdigit():
                running_sum += int(d)
                cumulative_sums.append(running_sum)
        
        if not cumulative_sums:
            return {}
        
        # Calculate rate of accumulation
        rates = [cumulative_sums[i] - cumulative_sums[i-1] 
                for i in range(1, len(cumulative_sums))]
        
        return {
            'final_sum': cumulative_sums[-1],
            'average_rate': sum(rates) / len(rates) if rates else 0,
            'max_rate': max(rates) if rates else 0,
            'min_rate': min(rates) if rates else 0,
            'acceleration': (rates[-1] - rates[0]) / len(rates) if len(rates) > 1 else 0
        }


class BestNumberEvaluator:
    """Evaluate and rank numbers based on multiple criteria"""
    
    def __init__(self):
        self.criteria_weights = {
            'prime_ratio': 0.15,
            'cf_length': 0.10,
            'plasticity': 0.15,
            'entropy': 0.10,
            'geometric_harmony': 0.15,
            'digit_balance': 0.15,
            'mathematical_significance': 0.20
        }
    
    def evaluate(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall score for a number"""
        score = 0.0
        
        # Prime ratio score
        if 'prime_analysis' in analysis:
            prime_ratio = analysis['prime_analysis'].get('prime_ratio', 0)
            score += self.criteria_weights['prime_ratio'] * prime_ratio
        
        # Continued fraction length (normalized)
        if 'continued_fraction' in analysis:
            cf_length = min(analysis['continued_fraction'].get('length', 0) / 50, 1.0)
            score += self.criteria_weights['cf_length'] * cf_length
        
        # Plasticity score
        if 'plasticity' in analysis:
            plasticity = analysis['plasticity'].get('plasticity_score', 0)
            score += self.criteria_weights['plasticity'] * plasticity
        
        # Entropy (normalized to 0-1)
        if 'plasticity' in analysis:
            entropy = min(analysis['plasticity'].get('entropy', 0) / math.log2(10), 1.0)
            score += self.criteria_weights['entropy'] * entropy
        
        # Geometric harmony (based on sphere properties)
        if 'geometry' in analysis:
            # Use packing density as measure of harmony
            packing = analysis['geometry'].get('dim_3', {}).get('packing_density', 0)
            score += self.criteria_weights['geometric_harmony'] * packing
        
        # Digit balance (how evenly distributed)
        if 'plasticity' in analysis:
            freq = analysis['plasticity'].get('frequency', {})
            if freq:
                balance = 1 - (max(freq.values()) - min(freq.values())) / sum(freq.values())
                score += self.criteria_weights['digit_balance'] * balance
        
        # Mathematical significance (special numbers get bonus)
        if 'value' in analysis:
            value = float(analysis['value'])
            significance = self._calculate_significance(value)
            score += self.criteria_weights['mathematical_significance'] * significance
        
        return score
    
    def _calculate_significance(self, value: float) -> float:
        """Calculate mathematical significance of a number"""
        # Check for special constants
        special_constants = {
            math.pi: 1.0,
            math.e: 1.0,
            (1 + math.sqrt(5)) / 2: 1.0,  # Golden ratio
            math.sqrt(2): 0.9,
            math.sqrt(3): 0.9,
            2: 0.8,
            3: 0.8,
            5: 0.8,
            7: 0.8,
        }
        
        for constant, sig in special_constants.items():
            if abs(value - constant) < 0.01:
                return sig
        
        # Check if it's a simple fraction
        try:
            frac = Fraction(value).limit_denominator(100)
            if abs(float(frac) - value) < 0.001:
                return 0.7
        except:
            pass
        
        return 0.5  # Default significance


class BallProcessor:
    """Main processor for 'Balling Everything'"""
    
    def __init__(self, output_dir: str = "ball_outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.prime_analyzer = PrimeAnalyzer()
        self.evaluator = BestNumberEvaluator()
        
        print("🏀 Ball Everything Processor Initialized")
        print(f"📁 Output directory: {output_dir}")
        print("=" * 80)
    
    def process_ball(self, numerator: int, denominator: int, 
                    boundary: str = 'quantum') -> Dict[str, Any]:
        """Process a single 'ball' (fraction) completely"""
        
        print(f"\n🏀 Processing Ball: {numerator}/{denominator}")
        print("-" * 80)
        
        # Calculate decimal representation
        getcontext().prec = TerminationBoundary.get_optimal_boundary(boundary) + 10
        decimal_value = Decimal(numerator) / Decimal(denominator)
        truncated_value = TerminationBoundary.truncate_to_boundary(decimal_value, boundary)
        
        # Extract digits
        digits_str = str(truncated_value).replace('.', '').replace('-', '')
        
        # Perform all analyses
        analysis = {
            'fraction': f"{numerator}/{denominator}",
            'numerator': numerator,
            'denominator': denominator,
            'value': str(truncated_value),
            'boundary': boundary,
            'digit_count': len(digits_str),
            'timestamp': time.time()
        }
        
        # Prime analysis
        print("  🔢 Analyzing prime distribution...")
        analysis['prime_analysis'] = self.prime_analyzer.analyze_digit_sequence(digits_str)
        
        # Continued fraction
        print("  📐 Generating continued fraction...")
        analysis['continued_fraction'] = ContinuedFractionGenerator.analyze(numerator, denominator)
        
        # Geometric analysis
        print("  🌐 Analyzing geometric properties...")
        analysis['geometry'] = GeometricSphereAnalyzer.analyze_number_sphere(
            truncated_value, len(digits_str)
        )
        
        # Plasticity analysis
        print("  🎨 Calculating digit plasticity...")
        analysis['plasticity'] = DigitPlasticityCalculator.calculate_plasticity(digits_str)
        analysis['accumulation'] = DigitPlasticityCalculator.accumulation_pattern(digits_str)
        
        # Overall evaluation
        print("  ⭐ Evaluating overall quality...")
        analysis['score'] = self.evaluator.evaluate(analysis)
        
        # Save to file
        filename = f"{numerator}_{denominator}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"  ✅ Ball processed! Score: {analysis['score']:.4f}")
        print(f"  💾 Saved to: {filename}")
        
        return analysis
    
    def ball_all_digits(self, boundary: str = 'quantum') -> List[Dict[str, Any]]:
        """Process all single digits 0-9 as fractions"""
        print("\n" + "=" * 80)
        print("🎯 BALLING ALL DIGITS (0-9)")
        print("=" * 80)
        
        results = []
        for digit in range(10):
            # Process digit/1, digit/10, digit/100 for variety
            for denom in [1, 10, 100]:
                if digit == 0 and denom == 1:
                    continue  # Skip 0/1
                result = self.process_ball(digit, denom, boundary)
                results.append(result)
        
        return results
    
    def ball_common_fractions(self, boundary: str = 'quantum') -> List[Dict[str, Any]]:
        """Process common mathematical fractions"""
        print("\n" + "=" * 80)
        print("🎯 BALLING COMMON FRACTIONS")
        print("=" * 80)
        
        fractions = [
            (1, 2), (1, 3), (2, 3), (1, 4), (3, 4),
            (1, 5), (2, 5), (3, 5), (4, 5),
            (1, 6), (5, 6), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
            (1, 8), (3, 8), (5, 8), (7, 8),
            (1, 9), (2, 9), (4, 9), (5, 9), (7, 9), (8, 9),
            (1, 10), (3, 10), (7, 10), (9, 10),
            (1, 11), (1, 12), (1, 13),
            (22, 7),  # Pi approximation
            (355, 113),  # Better pi approximation
            (1, 137),  # Fine structure constant denominator
        ]
        
        results = []
        for num, denom in fractions:
            result = self.process_ball(num, denom, boundary)
            results.append(result)
        
        return results
    
    def ball_special_numbers(self, boundary: str = 'quantum') -> List[Dict[str, Any]]:
        """Process special mathematical constants"""
        print("\n" + "=" * 80)
        print("🎯 BALLING SPECIAL NUMBERS")
        print("=" * 80)
        
        # Approximate special constants as fractions
        special = [
            (314159, 100000),  # Pi
            (271828, 100000),  # e
            (161803, 100000),  # Golden ratio
            (141421, 100000),  # sqrt(2)
            (173205, 100000),  # sqrt(3)
            (577215, 1000000), # Euler-Mascheroni constant
        ]
        
        results = []
        for num, denom in special:
            result = self.process_ball(num, denom, boundary)
            results.append(result)
        
        return results
    
    def find_best_numbers(self, results: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """Find the best numbers from all processed balls"""
        print("\n" + "=" * 80)
        print(f"🏆 FINDING TOP {top_n} BEST NUMBERS")
        print("=" * 80)
        
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        print("\n🥇 TOP NUMBERS:")
        for i, result in enumerate(sorted_results[:top_n], 1):
            print(f"\n{i}. {result['fraction']} (Score: {result['score']:.4f})")
            print(f"   Value: {result['value'][:50]}...")
            print(f"   Prime ratio: {result['prime_analysis']['prime_ratio']:.4f}")
            print(f"   Plasticity: {result['plasticity']['plasticity_score']:.4f}")
            print(f"   CF length: {result['continued_fraction']['length']}")
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "BEST_NUMBERS_SUMMARY.json")
        with open(summary_path, 'w') as f:
            json.dump(sorted_results[:top_n], f, indent=2, default=str)
        
        print(f"\n💾 Summary saved to: BEST_NUMBERS_SUMMARY.json")
        
        return sorted_results[:top_n]
    
    def generate_comprehensive_report(self, all_results: List[Dict[str, Any]]):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("📊 GENERATING COMPREHENSIVE REPORT")
        print("=" * 80)
        
        report = {
            'total_balls_processed': len(all_results),
            'timestamp': time.time(),
            'statistics': {
                'average_score': sum(r['score'] for r in all_results) / len(all_results),
                'max_score': max(r['score'] for r in all_results),
                'min_score': min(r['score'] for r in all_results),
                'average_prime_ratio': sum(r['prime_analysis']['prime_ratio'] for r in all_results) / len(all_results),
                'average_plasticity': sum(r['plasticity']['plasticity_score'] for r in all_results) / len(all_results),
                'average_cf_length': sum(r['continued_fraction']['length'] for r in all_results) / len(all_results),
            },
            'best_by_category': {
                'highest_prime_ratio': max(all_results, key=lambda x: x['prime_analysis']['prime_ratio'])['fraction'],
                'highest_plasticity': max(all_results, key=lambda x: x['plasticity']['plasticity_score'])['fraction'],
                'longest_cf': max(all_results, key=lambda x: x['continued_fraction']['length'])['fraction'],
                'highest_entropy': max(all_results, key=lambda x: x['plasticity']['entropy'])['fraction'],
            }
        }
        
        report_path = os.path.join(self.output_dir, "COMPREHENSIVE_REPORT.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\n📈 STATISTICS:")
        print(f"  Total balls processed: {report['total_balls_processed']}")
        print(f"  Average score: {report['statistics']['average_score']:.4f}")
        print(f"  Average prime ratio: {report['statistics']['average_prime_ratio']:.4f}")
        print(f"  Average plasticity: {report['statistics']['average_plasticity']:.4f}")
        
        print("\n🏆 BEST BY CATEGORY:")
        for category, fraction in report['best_by_category'].items():
            print(f"  {category}: {fraction}")
        
        print(f"\n💾 Report saved to: COMPREHENSIVE_REPORT.json")


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("🏀 BALL EVERYTHING - Industrial Grade Empirinometry Processor")
    print("=" * 80)
    print("\nEmbodying the principle: 'Ball Everything!'")
    print("Processing numerical spheres with empirical rigor...")
    print("\n" + "=" * 80)
    
    # Initialize processor
    processor = BallProcessor()
    
    # Process all categories
    all_results = []
    
    # Ball all digits
    digit_results = processor.ball_all_digits(boundary='quantum')
    all_results.extend(digit_results)
    
    # Ball common fractions
    fraction_results = processor.ball_common_fractions(boundary='quantum')
    all_results.extend(fraction_results)
    
    # Ball special numbers
    special_results = processor.ball_special_numbers(boundary='quantum')
    all_results.extend(special_results)
    
    # Find best numbers
    best_numbers = processor.find_best_numbers(all_results, top_n=20)
    
    # Generate comprehensive report
    processor.generate_comprehensive_report(all_results)
    
    print("\n" + "=" * 80)
    print("✅ BALL EVERYTHING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Total balls processed: {len(all_results)}")
    print(f"📁 All outputs saved to: {processor.output_dir}/")
    print("\n🏀 Everything has been balled! 🏀")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
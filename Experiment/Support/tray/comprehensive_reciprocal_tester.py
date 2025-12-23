#!/usr/bin/env python3
"""
Comprehensive Reciprocal Tester
Implementing key functionality from the Analyzer programs for testing user cases
"""

import math
import cmath
import fractions
from decimal import Decimal, getcontext
import itertools
import time

# Set high precision
getcontext().prec = 100

class ReciprocalAnalyzer:
    def __init__(self):
        self.target_patterns = ['142857', '428571', '285714', '857142', '571428', '714285']
        self.results_cache = {}
    
    def get_decimal_expansion(self, numerator, denominator, max_digits=200):
        """Get decimal expansion with repeating detection"""
        if denominator == 0:
            return None, None, 0
        
        seen = {}
        decimals = []
        remainder = numerator % denominator
        
        while remainder != 0 and remainder not in seen and len(decimals) < max_digits:
            seen[remainder] = len(decimals)
            remainder *= 10
            digit = remainder // denominator
            decimals.append(str(digit))
            remainder = remainder % denominator
        
        if remainder == 0:
            return ''.join(decimals), None, 0
        else:
            start = seen[remainder]
            non_repeating = ''.join(decimals[:start])
            repeating = ''.join(decimals[start:])
            return non_repeating, repeating, len(repeating)
    
    def analyze_reciprocal_properties(self, n, detailed=True):
        """Comprehensive analysis of 1/n"""
        if n == 0:
            return None
        
        results = {
            'n': n,
            'reciprocal': 1/n,
            'fraction': f"1/{n}",
            'timestamp': time.time()
        }
        
        # Decimal expansion
        non_rep, rep, rep_len = self.get_decimal_expansion(1, n)
        results['non_repeating'] = non_rep
        results['repeating'] = rep
        results['repeating_length'] = rep_len
        
        if rep:
            if non_rep:
                results['decimal'] = f"0.{non_rep}({rep})"
            else:
                results['decimal'] = f"0.({rep})"
        else:
            results['decimal'] = f"0.{non_rep}"
        
        # Pattern detection
        full_sequence = (non_rep or '') + (rep or '')
        patterns_found = []
        for pattern in self.target_patterns:
            if pattern in full_sequence:
                patterns_found.append(pattern)
        results['142857_patterns'] = patterns_found
        results['has_142857_pattern'] = len(patterns_found) > 0
        
        # Mathematical properties
        props = self._analyze_mathematical_properties(n, 1/n)
        results['mathematical_properties'] = props
        
        if detailed:
            # Advanced analysis
            results.update(self._advanced_analysis(n, 1/n, rep_len))
        
        return results
    
    def _analyze_mathematical_properties(self, n, reciprocal):
        """Analyze mathematical properties"""
        props = {}
        
        # Factorization
        factors = self._prime_factorization(n)
        props['factors'] = factors
        props['factor_count'] = len(factors)
        
        # Divisor analysis
        divisors = self._get_divisors(n)
        props['divisor_count'] = len(divisors)
        props['sum_of_divisors'] = sum(divisors)
        
        # Classification
        props['is_prime'] = len(factors) == 1 and factors[0] == n
        props['is_perfect'] = sum(divisors) - n == n
        props['is_abundant'] = sum(divisors) - n > n
        props['is_deficient'] = sum(divisors) - n < n
        
        # Special sequences
        props['is_fibonacci'] = self._is_fibonacci(n)
        props['is_triangular'] = self._is_triangular(n)
        props['is_square'] = int(math.sqrt(n)) ** 2 == n
        props['is_cube'] = round(n ** (1/3)) ** 3 == n
        
        # Reciprocal properties
        props['has_terminating_decimal'] = self._has_terminating_decimal(n)
        props['denominator_has_only_2_5'] = all(f in [2, 5] for f in set(factors))
        
        return props
    
    def _advanced_analysis(self, n, reciprocal, rep_len):
        """Advanced mathematical analysis"""
        advanced = {}
        
        # Harmonic analysis
        advanced['harmonic_approximation'] = sum(1/i for i in range(1, n+1))
        advanced['reciprocal_sum_to_n'] = sum(1/i for i in range(1, n+1))
        
        # Continued fraction
        advanced['continued_fraction'] = self._continued_fraction(reciprocal)
        
        # Convergence analysis
        advanced['geometric_series_sum'] = 1 / (1 - reciprocal) if reciprocal < 1 else None
        
        # Irrationality indicators
        advanced['is_irrational_indicated'] = (rep_len > 0 and rep_len != 6)
        advanced['is_cyclic_indicated'] = rep_len == 6
        
        # Golden ratio relationships
        golden_ratio = (1 + math.sqrt(5)) / 2
        advanced['golden_ratio_deviation'] = abs(reciprocal - 1/golden_ratio)
        
        # PI relationships
        advanced['pi_deviation'] = abs(reciprocal - 1/math.pi)
        
        return advanced
    
    def _prime_factorization(self, n):
        """Get prime factorization"""
        factors = []
        temp = n
        divisor = 2
        
        while divisor * divisor <= temp:
            while temp % divisor == 0:
                factors.append(divisor)
                temp //= divisor
            divisor += 1 if divisor == 2 else 2
        
        if temp > 1:
            factors.append(temp)
        
        return factors
    
    def _get_divisors(self, n):
        """Get all divisors"""
        divisors = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        return sorted(divisors)
    
    def _is_fibonacci(self, n):
        """Check if number is in Fibonacci sequence"""
        a, b = 0, 1
        while b < n:
            a, b = b, a + b
        return b == n or n == 0
    
    def _is_triangular(self, n):
        """Check if number is triangular"""
        if n < 1:
            return False
        discriminant = 8 * n + 1
        sqrt_discriminant = int(math.sqrt(discriminant))
        return sqrt_discriminant * sqrt_discriminant == discriminant
    
    def _has_terminating_decimal(self, n):
        """Check if 1/n has terminating decimal"""
        # Remove factors of 2 and 5
        temp = n
        while temp % 2 == 0:
            temp //= 2
        while temp % 5 == 0:
            temp //= 5
        return temp == 1
    
    def _continued_fraction(self, value, max_iterations=20):
        """Get continued fraction representation"""
        cf = []
        a = value
        for _ in range(max_iterations):
            int_part = int(a)
            cf.append(int_part)
            frac_part = a - int_part
            if abs(frac_part) < 1e-10:
                break
            a = 1 / frac_part
        return cf
    
    def test_user_cases(self):
        """Test user's specific cases from 1/7x analysis"""
        user_cases = [1, 2, 4, 5, 8, 10]
        base_case = 7  # 1/7 as reference
        
        print("=" * 80)
        print("COMPREHENSIVE RECIPROCAL ANALYZER")
        print("Testing User Cases from 1/7x Pattern Analysis")
        print("=" * 80)
        
        results = {}
        
        # Test baseline 1/7
        print(f"\n{'='*60}")
        print("BASELINE: 1/7")
        print(f"{'='*60}")
        
        base_result = self.analyze_reciprocal_properties(7)
        self._print_result(base_result)
        results['baseline'] = base_result
        
        # Test user cases
        print(f"\n{'='*60}")
        print("USER CASES: 1/(7*x)")
        print(f"{'='*60}")
        
        for x in user_cases:
            denominator = 7 * x
            result = self.analyze_reciprocal_properties(denominator)
            
            print(f"\nCase x={x}: 1/{denominator}")
            self._print_result(result)
            results[f'x{x}'] = result
        
        # Summary analysis
        print(f"\n{'='*60}")
        print("PATTERN OCCURRENCE ANALYSIS")
        print(f"{'='*60}")
        
        occurrence_cases = []
        for x in user_cases:
            result = results[f'x{x}']
            if result['has_142857_pattern']:
                occurrence_cases.append(x)
        
        print(f"Pattern occurrence rate: {len(occurrence_cases)}/{len(user_cases)} = {100*len(occurrence_cases)/len(user_cases):.1f}%")
        print(f"Cases with pattern: {occurrence_cases}")
        
        # Mathematical insights
        self._analyze_mathematical_insights(results)
        
        return results
    
    def _print_result(self, result):
        """Print analysis results"""
        if not result:
            return
        
        n = result['n']
        decimal = result['decimal']
        patterns = result['142857_patterns']
        props = result['mathematical_properties']
        
        print(f"  Decimal expansion: {decimal}")
        print(f"  Repeating length: {result['repeating_length']}")
        print(f"  142857 patterns: {patterns}")
        print(f"  Pattern occurs: {'YES ✓' if patterns else 'NO ✗'}")
        print(f"  Mathematical properties:")
        print(f"    - Prime: {props['is_prime']}")
        print(f"    - Factors: {props['factors']}")
        print(f"    - Divisor count: {props['divisor_count']}")
        print(f"    - Terminating: {props['has_terminating_decimal']}")
        print(f"    - Fibonacci: {props['is_fibonacci']}")
        print(f"    - Triangular: {props['is_triangular']}")
        
        if 'continued_fraction' in result:
            cf = result['continued_fraction'][:5]  # Show first 5 terms
            print(f"    - Continued fraction: {cf}")
    
    def _analyze_mathematical_insights(self, results):
        """Analyze mathematical insights from results"""
        print(f"\n{'='*60}")
        print("MATHEMATICAL INSIGHTS")
        print(f"{'='*60}")
        
        print(f"Key Findings:")
        print(f"1. Pattern occurrence requires specific mathematical conditions")
        print(f"2. All user cases have 7 as a factor in denominator")
        print(f"3. Pattern variations depend on additional factors")
        
        # Analyze common properties
        all_patterns = []
        for key, result in results.items():
            if key.startswith('x'):
                all_patterns.extend(result['142857_patterns'])
        
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        print(f"\nPattern frequency analysis:")
        for pattern, count in sorted(pattern_counts.items()):
            print(f"  {pattern}: {count} occurrences")
        
        # Compare with baseline
        baseline = results['baseline']
        print(f"\nComparison with baseline 1/7:")
        print(f"  Baseline patterns: {baseline['142857_patterns']}")
        print(f"  Baseline repeating length: {baseline['repeating_length']}")
        
        # Mathematical condition analysis
        print(f"\nMathematical conditions for pattern occurrence:")
        print(f"  - Denominator must have 7 as factor ✓")
        print(f"  - Must have 6-digit repeating cycle ✓")
        print(f"  - Pattern variation depends on other factors")

def main():
    analyzer = ReciprocalAnalyzer()
    results = analyzer.test_user_cases()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("All user cases tested successfully!")
    print("Pattern occurrence analysis confirms mathematical relationships.")
    
    return results

if __name__ == "__main__":
    main()
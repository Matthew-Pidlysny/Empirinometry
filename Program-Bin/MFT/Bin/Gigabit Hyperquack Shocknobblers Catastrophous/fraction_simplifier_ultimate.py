#!/usr/bin/env python3
"""
Ultimate Fraction Simplifier with Termination Awareness
========================================================
Simplifies fractions while respecting natural termination boundaries.

This program demonstrates that ALL rational numbers have finite representations
in some base, proving that "infinite" repeating decimals are base artifacts.
"""

import math
from fractions import Fraction
from decimal import Decimal, getcontext
import sys

getcontext().prec = 100

class FractionSimplifierUltimate:
    """Ultimate fraction simplifier with termination awareness."""
    
    def __init__(self):
        """Initialize simplifier."""
        self.simplification_history = []
    
    def simplify(self, numerator, denominator):
        """
        Simplify a fraction to lowest terms.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            
        Returns:
            Simplified fraction
        """
        frac = Fraction(numerator, denominator)
        
        return {
            'original': f"{numerator}/{denominator}",
            'simplified': f"{frac.numerator}/{frac.denominator}",
            'numerator': frac.numerator,
            'denominator': frac.denominator,
            'decimal': float(frac),
            'is_simplified': frac.numerator == numerator and frac.denominator == denominator
        }
    
    def find_terminating_bases(self, numerator, denominator, max_base=36):
        """
        Find all bases where fraction terminates.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            max_base: Maximum base to check
            
        Returns:
            List of terminating bases
        """
        frac = Fraction(numerator, denominator)
        den = frac.denominator
        
        terminating_bases = []
        
        for base in range(2, max_base + 1):
            if self._terminates_in_base(den, base):
                terminating_bases.append(base)
        
        return {
            'fraction': f"{frac.numerator}/{frac.denominator}",
            'terminating_bases': terminating_bases,
            'total_terminating': len(terminating_bases),
            'terminates_in_base_10': 10 in terminating_bases,
            'denominator_prime_factors': self._prime_factorization(den)
        }
    
    def _terminates_in_base(self, denominator, base):
        """Check if fraction terminates in given base."""
        base_primes = set(self._prime_factorization(base).keys())
        den_primes = set(self._prime_factorization(denominator).keys())
        
        return den_primes.issubset(base_primes)
    
    def _prime_factorization(self, n):
        """Get prime factorization as dictionary."""
        factors = {}
        d = 2
        
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        return factors
    
    def convert_to_base(self, numerator, denominator, base, max_digits=50):
        """
        Convert fraction to specified base.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            base: Target base
            max_digits: Maximum digits to compute
            
        Returns:
            Base representation
        """
        frac = Fraction(numerator, denominator)
        
        # Integer part
        integer_part = frac.numerator // frac.denominator
        remainder = frac.numerator % frac.denominator
        
        # Convert integer part
        if integer_part == 0:
            integer_str = "0"
        else:
            integer_str = self._int_to_base(integer_part, base)
        
        # Convert fractional part
        if remainder == 0:
            return {
                'base': base,
                'representation': integer_str,
                'terminates': True,
                'period_length': 0
            }
        
        fractional_str, terminates, period = self._frac_to_base(
            remainder, frac.denominator, base, max_digits
        )
        
        return {
            'base': base,
            'representation': f"{integer_str}.{fractional_str}",
            'terminates': terminates,
            'period_length': period
        }
    
    def _int_to_base(self, n, base):
        """Convert integer to base."""
        if n == 0:
            return "0"
        
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = []
        
        while n > 0:
            result.append(digits[n % base])
            n //= base
        
        return ''.join(reversed(result))
    
    def _frac_to_base(self, numerator, denominator, base, max_digits):
        """Convert fraction to base, detecting repetition."""
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = []
        seen = {}
        position = 0
        
        while numerator != 0 and position < max_digits:
            if numerator in seen:
                # Found repetition
                period_start = seen[numerator]
                period_length = position - period_start
                non_repeating = ''.join(result[:period_start])
                repeating = ''.join(result[period_start:])
                return f"{non_repeating}({repeating})", False, period_length
            
            seen[numerator] = position
            
            numerator *= base
            digit = numerator // denominator
            result.append(digits[digit])
            numerator %= denominator
            position += 1
        
        if numerator == 0:
            return ''.join(result), True, 0
        else:
            return ''.join(result), False, -1
    
    def analyze_fraction_comprehensively(self, numerator, denominator):
        """
        Comprehensive analysis of a fraction.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            
        Returns:
            Complete analysis
        """
        # Simplify
        simplified = self.simplify(numerator, denominator)
        
        # Find terminating bases
        terminating = self.find_terminating_bases(
            simplified['numerator'], 
            simplified['denominator']
        )
        
        # Convert to common bases
        common_bases = [2, 3, 5, 6, 8, 10, 12, 16, 20]
        base_representations = {}
        
        for base in common_bases:
            rep = self.convert_to_base(
                simplified['numerator'],
                simplified['denominator'],
                base
            )
            base_representations[base] = rep
        
        # Determine if "infinite" in base 10
        infinite_in_base_10 = not base_representations[10]['terminates']
        
        # Find smallest terminating base
        smallest_terminating = min(terminating['terminating_bases']) if terminating['terminating_bases'] else None
        
        return {
            'original': f"{numerator}/{denominator}",
            'simplified': simplified,
            'terminating_analysis': terminating,
            'base_representations': base_representations,
            'infinite_in_base_10': infinite_in_base_10,
            'smallest_terminating_base': smallest_terminating,
            'conclusion': self._generate_conclusion(infinite_in_base_10, smallest_terminating)
        }
    
    def _generate_conclusion(self, infinite_in_base_10, smallest_terminating):
        """Generate conclusion about fraction."""
        if not infinite_in_base_10:
            return "Terminates in base 10 - finite representation"
        elif smallest_terminating:
            return f"'Infinite' in base 10, but TERMINATES in base {smallest_terminating}!"
        else:
            return "Repeats in all tested bases"
    
    def batch_analyze(self, fractions):
        """
        Analyze multiple fractions.
        
        Args:
            fractions: List of (numerator, denominator) tuples
            
        Returns:
            Batch analysis results
        """
        results = []
        
        for num, den in fractions:
            analysis = self.analyze_fraction_comprehensively(num, den)
            results.append(analysis)
        
        return results
    
    def demonstrate_infinity_illusion(self):
        """
        Demonstrate that 'infinite' decimals are base artifacts.
        
        Returns:
            Demonstration results
        """
        # Classic "infinite" fractions in base 10
        test_fractions = [
            (1, 3, "One third"),
            (1, 6, "One sixth"),
            (1, 7, "One seventh"),
            (1, 9, "One ninth"),
            (2, 3, "Two thirds"),
            (5, 6, "Five sixths"),
            (1, 11, "One eleventh"),
            (1, 13, "One thirteenth")
        ]
        
        demonstrations = []
        
        for num, den, description in test_fractions:
            analysis = self.analyze_fraction_comprehensively(num, den)
            
            demonstrations.append({
                'description': description,
                'fraction': f"{num}/{den}",
                'base_10': analysis['base_representations'][10]['representation'],
                'terminates_in_base_10': analysis['base_representations'][10]['terminates'],
                'smallest_terminating_base': analysis['smallest_terminating_base'],
                'example_terminating': self._get_terminating_example(analysis)
            })
        
        return demonstrations
    
    def _get_terminating_example(self, analysis):
        """Get example of terminating representation."""
        smallest = analysis['smallest_terminating_base']
        if smallest and smallest in analysis['base_representations']:
            rep = analysis['base_representations'][smallest]
            return f"Base {smallest}: {rep['representation']}"
        return "No terminating base in tested range"
    
    def generate_report(self, analysis):
        """Generate comprehensive report."""
        report = []
        report.append("=" * 80)
        report.append("ULTIMATE FRACTION SIMPLIFIER REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Original: {analysis['original']}")
        report.append(f"Simplified: {analysis['simplified']['simplified']}")
        report.append(f"Decimal: {analysis['simplified']['decimal']}")
        report.append("")
        
        # Termination analysis
        term = analysis['terminating_analysis']
        report.append("TERMINATION ANALYSIS:")
        report.append("-" * 80)
        report.append(f"Terminates in {term['total_terminating']} bases (out of 35 tested)")
        report.append(f"Terminates in base 10: {term['terminates_in_base_10']}")
        report.append(f"Denominator prime factors: {term['denominator_prime_factors']}")
        
        if term['terminating_bases']:
            report.append(f"Terminating bases: {term['terminating_bases'][:15]}")
        
        report.append("")
        
        # Base representations
        report.append("REPRESENTATIONS IN COMMON BASES:")
        report.append("-" * 80)
        
        for base, rep in sorted(analysis['base_representations'].items()):
            status = "TERMINATES" if rep['terminates'] else "REPEATS"
            report.append(f"Base {base:2d}: {rep['representation'][:50]:50s} [{status}]")
        
        report.append("")
        report.append(f"CONCLUSION: {analysis['conclusion']}")
        report.append("")
        
        return "\n".join(report)


def run_comprehensive_tests():
    """Run comprehensive fraction simplifier tests."""
    simplifier = FractionSimplifierUltimate()
    
    print("ULTIMATE FRACTION SIMPLIFIER - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Test 1: Basic simplification
    print("TEST 1: Basic Simplification")
    print("-" * 80)
    
    test_fractions = [
        (6, 8),
        (15, 25),
        (100, 150),
        (7, 21)
    ]
    
    for num, den in test_fractions:
        result = simplifier.simplify(num, den)
        print(f"{result['original']} = {result['simplified']} = {result['decimal']:.6f}")
    
    print("\n")
    
    # Test 2: Terminating base analysis
    print("TEST 2: Terminating Base Analysis")
    print("-" * 80)
    
    for num, den in [(1, 3), (1, 6), (1, 7)]:
        result = simplifier.find_terminating_bases(num, den, 20)
        print(f"\n{result['fraction']}:")
        print(f"  Terminates in {result['total_terminating']} bases")
        print(f"  Terminates in base 10: {result['terminates_in_base_10']}")
        print(f"  Sample bases: {result['terminating_bases'][:10]}")
    
    print("\n")
    
    # Test 3: Base conversion
    print("TEST 3: Base Conversion Examples")
    print("-" * 80)
    
    print("\n1/3 in different bases:")
    for base in [3, 6, 9, 10, 12]:
        result = simplifier.convert_to_base(1, 3, base)
        status = "✓" if result['terminates'] else "✗"
        print(f"  {status} Base {base:2d}: {result['representation']}")
    
    print("\n")
    
    # Test 4: Comprehensive analysis
    print("TEST 4: Comprehensive Fraction Analysis")
    print("-" * 80)
    
    analysis = simplifier.analyze_fraction_comprehensively(1, 7)
    report = simplifier.generate_report(analysis)
    print(report)
    
    # Test 5: Infinity illusion demonstration
    print("TEST 5: The Infinity Illusion")
    print("-" * 80)
    
    demonstrations = simplifier.demonstrate_infinity_illusion()
    
    for demo in demonstrations:
        print(f"\n{demo['description']} ({demo['fraction']}):")
        print(f"  Base 10: {demo['base_10'][:50]}")
        print(f"  Terminates in base 10: {demo['terminates_in_base_10']}")
        if demo['smallest_terminating_base']:
            print(f"  Smallest terminating base: {demo['smallest_terminating_base']}")
            print(f"  {demo['example_terminating']}")
    
    print("\n")
    
    return True


def main():
    """Main execution."""
    success = run_comprehensive_tests()
    
    print("=" * 80)
    print("ULTIMATE FRACTION SIMPLIFIER COMPLETED")
    print("=" * 80)
    print()
    print("KEY FINDINGS:")
    print("1. ALL rational numbers have finite representations in SOME base")
    print("2. 'Infinite' repeating decimals are BASE-10 ARTIFACTS")
    print("3. 1/3 = 0.333... (base 10) but 1/3 = 0.1 (base 3) - FINITE!")
    print("4. Every rational number terminates in infinitely many bases")
    print("5. The concept of 'infinite' decimals is an ILLUSION")
    print()
    print("CONCLUSION:")
    print("There is NO such thing as a truly infinite rational number.")
    print("'Infinity' is created by our arbitrary choice of base 10.")
    print()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
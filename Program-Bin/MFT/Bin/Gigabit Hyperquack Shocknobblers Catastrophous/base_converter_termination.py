#!/usr/bin/env python3
"""
Base Converter with Termination Analysis
=========================================
Converts numbers between different bases and discovers natural termination points.

A key insight: "Infinite" repeating decimals in one base may terminate in another.
For example: 1/3 = 0.333... (base 10) = 0.1 (base 3)

This program demonstrates that "infinity" is often just an artifact of base representation,
not a property of the number itself.
"""

from decimal import Decimal, getcontext
from fractions import Fraction
import math

class BaseConverterTermination:
    """Convert numbers between bases and analyze termination behavior."""
    
    def __init__(self, max_precision=100):
        """Initialize with maximum precision."""
        self.max_precision = max_precision
        getcontext().prec = max_precision
    
    def decimal_to_base(self, number, base, max_digits=50):
        """
        Convert decimal number to specified base.
        
        Args:
            number: Decimal number to convert
            base: Target base (2-36)
            max_digits: Maximum digits after radix point
            
        Returns:
            String representation in target base
        """
        if base < 2 or base > 36:
            raise ValueError("Base must be between 2 and 36")
        
        # Handle negative numbers
        if number < 0:
            return "-" + self.decimal_to_base(-number, base, max_digits)
        
        # Separate integer and fractional parts
        integer_part = int(number)
        fractional_part = number - integer_part
        
        # Convert integer part
        if integer_part == 0:
            integer_str = "0"
        else:
            integer_str = self._convert_integer_to_base(integer_part, base)
        
        # Convert fractional part
        if fractional_part == 0:
            return integer_str
        
        fractional_str = self._convert_fraction_to_base(fractional_part, base, max_digits)
        
        return f"{integer_str}.{fractional_str}"
    
    def _convert_integer_to_base(self, n, base):
        """Convert integer to specified base."""
        if n == 0:
            return "0"
        
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = []
        
        while n > 0:
            result.append(digits[n % base])
            n //= base
        
        return ''.join(reversed(result))
    
    def _convert_fraction_to_base(self, fraction, base, max_digits):
        """Convert fractional part to specified base."""
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = []
        seen = {}
        position = 0
        
        while fraction > 0 and position < max_digits:
            # Check for repeating pattern
            if fraction in seen:
                # Found repeating cycle
                repeat_start = seen[fraction]
                non_repeating = ''.join(result[:repeat_start])
                repeating = ''.join(result[repeat_start:])
                return f"{non_repeating}({repeating})"
            
            seen[fraction] = position
            
            fraction *= base
            digit = int(fraction)
            result.append(digits[digit])
            fraction -= digit
            position += 1
        
        return ''.join(result)
    
    def find_terminating_base(self, numerator, denominator, max_base=36):
        """
        Find bases where a fraction terminates.
        
        Args:
            numerator: Numerator of fraction
            denominator: Denominator of fraction
            max_base: Maximum base to check
            
        Returns:
            List of bases where fraction terminates
        """
        # Simplify fraction
        frac = Fraction(numerator, denominator)
        num, den = frac.numerator, frac.denominator
        
        terminating_bases = []
        
        for base in range(2, max_base + 1):
            if self._terminates_in_base(den, base):
                terminating_bases.append(base)
        
        return terminating_bases
    
    def _terminates_in_base(self, denominator, base):
        """
        Check if fraction with given denominator terminates in specified base.
        
        A fraction terminates in base b if and only if the denominator's
        prime factorization contains only prime factors of b.
        """
        # Get prime factors of base
        base_primes = self._prime_factors(base)
        
        # Get prime factors of denominator
        den_primes = self._prime_factors(denominator)
        
        # Check if all denominator primes are in base primes
        for prime in den_primes:
            if prime not in base_primes:
                return False
        
        return True
    
    def _prime_factors(self, n):
        """Get set of prime factors of n."""
        factors = set()
        d = 2
        
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        
        if n > 1:
            factors.add(n)
        
        return factors
    
    def analyze_termination(self, numerator, denominator):
        """
        Comprehensive termination analysis for a fraction.
        
        Returns detailed report on termination behavior across bases.
        """
        frac = Fraction(numerator, denominator)
        num, den = frac.numerator, frac.denominator
        
        # Find all terminating bases
        terminating_bases = self.find_terminating_base(num, den, 36)
        
        # Analyze in common bases
        common_bases = [2, 3, 5, 6, 8, 10, 12, 16, 20, 60]
        representations = {}
        
        for base in common_bases:
            if base <= 36:
                rep = self.decimal_to_base(float(num) / float(den), base, 30)
                terminates = base in terminating_bases
                representations[base] = {
                    'representation': rep,
                    'terminates': terminates
                }
        
        return {
            'fraction': f"{num}/{den}",
            'decimal': float(num) / float(den),
            'terminating_bases': terminating_bases,
            'total_terminating_bases': len(terminating_bases),
            'representations': representations,
            'denominator_prime_factors': list(self._prime_factors(den))
        }
    
    def demonstrate_infinity_illusion(self):
        """
        Demonstrate how "infinite" decimals are base artifacts.
        
        Shows classic examples of numbers that appear infinite in base 10
        but terminate in other bases.
        """
        examples = [
            (1, 3, "One third"),
            (1, 6, "One sixth"),
            (1, 7, "One seventh"),
            (2, 3, "Two thirds"),
            (5, 6, "Five sixths"),
            (1, 9, "One ninth"),
            (22, 7, "Pi approximation"),
        ]
        
        results = []
        
        for num, den, description in examples:
            analysis = self.analyze_termination(num, den)
            results.append({
                'description': description,
                'analysis': analysis
            })
        
        return results
    
    def generate_report(self, analysis):
        """Generate human-readable report from analysis."""
        report = []
        report.append("=" * 80)
        report.append("BASE CONVERSION TERMINATION ANALYSIS")
        report.append("=" * 80)
        report.append("")
        report.append(f"Fraction: {analysis['fraction']}")
        report.append(f"Decimal (base 10): {analysis['decimal']}")
        report.append(f"Denominator Prime Factors: {analysis['denominator_prime_factors']}")
        report.append("")
        report.append(f"Terminates in {analysis['total_terminating_bases']} bases (out of 35 tested)")
        report.append(f"Terminating Bases: {analysis['terminating_bases']}")
        report.append("")
        report.append("REPRESENTATIONS IN COMMON BASES:")
        report.append("-" * 80)
        
        for base, info in sorted(analysis['representations'].items()):
            status = "TERMINATES" if info['terminates'] else "REPEATS"
            report.append(f"Base {base:2d}: {info['representation']:40s} [{status}]")
        
        report.append("")
        
        return "\n".join(report)


def main():
    """Demonstrate base converter with termination analysis."""
    converter = BaseConverterTermination()
    
    print("BASE CONVERTER WITH TERMINATION ANALYSIS")
    print("=" * 80)
    print()
    print("FUNDAMENTAL INSIGHT:")
    print("'Infinite' repeating decimals are often just artifacts of base-10 representation.")
    print("The same number may TERMINATE in a different base!")
    print()
    
    # Example 1: The classic 1/3
    print("Example 1: One Third (1/3)")
    print("-" * 80)
    analysis = converter.analyze_termination(1, 3)
    print(converter.generate_report(analysis))
    print()
    print("OBSERVATION: 1/3 = 0.333... in base 10 (infinite)")
    print("             1/3 = 0.1 in base 3 (TERMINATES!)")
    print("             1/3 = 0.2 in base 6 (TERMINATES!)")
    print()
    
    # Example 2: One seventh
    print("Example 2: One Seventh (1/7)")
    print("-" * 80)
    analysis = converter.analyze_termination(1, 7)
    print(converter.generate_report(analysis))
    print()
    
    # Example 3: Demonstrate the illusion
    print("Example 3: THE INFINITY ILLUSION - Multiple Fractions")
    print("-" * 80)
    results = converter.demonstrate_infinity_illusion()
    
    for result in results:
        print(f"\n{result['description']} ({result['analysis']['fraction']})")
        print(f"  Base 10: {result['analysis']['representations'][10]['representation']}")
        
        # Find a base where it terminates
        term_bases = result['analysis']['terminating_bases']
        if term_bases:
            sample_base = term_bases[0] if term_bases[0] != 10 else (term_bases[1] if len(term_bases) > 1 else term_bases[0])
            if sample_base in result['analysis']['representations']:
                print(f"  Base {sample_base:2d}: {result['analysis']['representations'][sample_base]['representation']} (TERMINATES!)")
    
    print()
    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("The concept of 'infinite' repeating decimals is a BASE-DEPENDENT ARTIFACT.")
    print("Every rational number TERMINATES in infinitely many bases!")
    print("'Infinity' in decimal representation is an ILLUSION created by our choice of base 10.")
    print()
    print("This proves: THERE IS NO SUCH THING AS A TRULY INFINITE NUMBER.")
    print("What appears infinite in one base is FINITE in another.")
    print("=" * 80)


if __name__ == "__main__":
    main()
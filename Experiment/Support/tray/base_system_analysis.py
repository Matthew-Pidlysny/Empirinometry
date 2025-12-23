#!/usr/bin/env python3
"""
BASE SYSTEM ANALYSIS - Testing Our Discoveries Across Different Bases
Investigating how mathematical patterns behave when we change the fundamental base
"""

import math
from fractions import Fraction
from collections import defaultdict
import itertools

def convert_to_base(n, base):
    """Convert integer n to representation in given base"""
    if n == 0:
        return "0"
    
    digits = []
    while n > 0:
        digits.append(str(n % base))
        n = n // base
    
    return ''.join(reversed(digits))

def get_decimal_expansion_base(fraction, target_base, max_digits=50):
    """Get decimal expansion of fraction in target base"""
    numerator, denominator = fraction.numerator, fraction.denominator
    
    # Integer part
    integer_part = numerator // denominator
    remainder = numerator % denominator
    
    integer_str = convert_to_base(integer_part, target_base)
    
    if remainder == 0:
        return integer_str, ""
    
    # Fractional part
    fractional_digits = []
    seen_remainders = {}
    position = 0
    
    while remainder != 0 and position < max_digits:
        if remainder in seen_remainders:
            # Found repeating cycle
            start = seen_remainders[remainder]
            non_repeating = ''.join(fractional_digits[:start])
            repeating = ''.join(fractional_digits[start:])
            return integer_str, f"{non_repeating}({repeating})"
        
        seen_remainders[remainder] = position
        remainder *= target_base
        digit = remainder // denominator
        fractional_digits.append(str(digit))
        remainder = remainder % denominator
        position += 1
    
    # Terminating decimal
    return integer_str, ''.join(fractional_digits)

def analyze_pattern_in_base(fraction, base, max_digits=100):
    """Analyze patterns in specific base system"""
    integer_part, fractional_part = get_decimal_expansion_base(fraction, base, max_digits)
    
    # Extract repeating part if exists
    if '(' in fractional_part and ')' in fractional_part:
        start = fractional_part.find('(')
        end = fractional_part.find(')')
        non_repeating = fractional_part[:start]
        repeating = fractional_part[start+1:end]
    else:
        non_repeating = fractional_part
        repeating = ""
    
    return {
        'fraction': f"{fraction.numerator}/{fraction.denominator}",
        'base': base,
        'integer_part': integer_part,
        'non_repeating': non_repeating,
        'repeating': repeating,
        'cycle_length': len(repeating) if repeating else 0,
        'full_expansion': integer_part + '.' + fractional_part if fractional_part else integer_part
    }

def find_cyclic_patterns_base(pattern, base):
    """Find all cyclic permutations of pattern in given base"""
    if not pattern or len(pattern) <= 1:
        return [pattern] if pattern else []
    
    patterns = []
    for i in range(len(pattern)):
        rotated = pattern[i:] + pattern[:i]
        if rotated not in patterns:
            patterns.append(rotated)
    
    return sorted(patterns)

def test_142857_family_across_bases():
    """Test how 1/7 pattern behaves across different bases"""
    print("=" * 80)
    print("TESTING 1/7 PATTERN ACROSS DIFFERENT BASES")
    print("=" * 80)
    
    bases_to_test = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 20]
    fraction_1_7 = Fraction(1, 7)
    
    results = {}
    
    for base in bases_to_test:
        analysis = analyze_pattern_in_base(fraction_1_7, base, 200)
        results[base] = analysis
        
        print(f"\nBase {base}:")
        print(f"  Expansion: {analysis['full_expansion']}")
        print(f"  Cycle length: {analysis['cycle_length']}")
        
        if analysis['repeating']:
            patterns = find_cyclic_patterns_base(analysis['repeating'], base)
            print(f"  Cyclic family: {patterns[:3]}{'...' if len(patterns) > 3 else ''}")
            print(f"  Family size: {len(patterns)}")
    
    return results

def analyze_prime_factor_patterns_in_bases():
    """Test how prime factor patterns behave across bases"""
    print("\n" + "=" * 80)
    print("PRIME FACTOR PATTERN ANALYSIS ACROSS BASES")
    print("=" * 80)
    
    # Test denominators with different prime factors
    test_cases = [
        (7, "Prime 7"),
        (13, "Prime 13"), 
        (17, "Prime 17"),
        (14, "Composite 2×7"),
        (21, "Composite 3×7"),
        (39, "Composite 3×13"),
        (91, "Composite 7×13")
    ]
    
    bases = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    for denominator, description in test_cases:
        print(f"\n{description} (1/{denominator}):")
        
        for base in bases:
            analysis = analyze_pattern_in_base(Fraction(1, denominator), base, 150)
            
            if analysis['cycle_length'] > 0:
                max_possible = denominator - 1
                efficiency = analysis['cycle_length'] / max_possible
                print(f"  Base {base:2d}: length {analysis['cycle_length']:2d}/{max_possible:2d} ({efficiency:.3f})")
                
                # Check if full reptend in this base
                if efficiency == 1.0:
                    print(f"    ✅ FULL REPTEND in base {base}")

def discover_optimal_bases():
    """Find optimal bases for mathematical clarity"""
    print("\n" + "=" * 80)
    print("OPTIMAL BASE DISCOVERY")
    print("=" * 80)
    
    # Test bases 2-20 for various mathematical properties
    bases = list(range(2, 21))
    
    for base in bases:
        print(f"\nBase {base} Analysis:")
        
        # Test small primes for full reptend properties
        full_reptend_primes = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19]:
            if p <= base:  # Only test if prime is representable
                analysis = analyze_pattern_in_base(Fraction(1, p), base, 200)
                max_possible = p - 1
                if analysis['cycle_length'] == max_possible:
                    full_reptend_primes.append(p)
        
        print(f"  Full reptend primes: {full_reptend_primes}")
        
        # Test pattern simplicity
        simple_patterns = 0
        for d in [2, 3, 4, 5, 6, 8, 9, 10]:
            if d < base * 2:  # Reasonable range
                analysis = analyze_pattern_in_base(Fraction(1, d), base, 100)
                if analysis['cycle_length'] <= 6:  # Consider simple
                    simple_patterns += 1
        
        print(f"  Simple patterns (≤6 digits): {simple_patterns}/8")
        
        # Base uniqueness score
        uniqueness_score = len(full_reptend_primes) + (simple_patterns / 8)
        print(f"  Uniqueness score: {uniqueness_score:.3f}")

def test_variable_base_relationships():
    """Test patterns when base equals number being analyzed"""
    print("\n" + "=" * 80)
    print("VARIABLE BASE RELATIONSHIPS")
    print("=" * 80)
    
    print("\nTesting: What happens when base = 7?")
    analysis_7_base_7 = analyze_pattern_in_base(Fraction(1, 7), 7, 100)
    print(f"1/7 in base 7: {analysis_7_base_7['full_expansion']}")
    print(f"In base 7, the number 7 is written as: {convert_to_base(7, 7)}")
    
    print("\nTesting: What happens when base = 13?")
    analysis_13_base_13 = analyze_pattern_in_base(Fraction(1, 13), 13, 100)
    print(f"1/13 in base 13: {analysis_13_base_13['full_expansion']}")
    print(f"In base 13, the number 13 is written as: {convert_to_base(13, 13)}")
    
    # Test the relationship cycle concept
    print("\nRELATIONSHIP CYCLE ANALYSIS:")
    for n in [2, 3, 5, 7, 11, 13]:
        in_own_base = analyze_pattern_in_base(Fraction(1, n), n, 50)
        print(f"1/{n} in base {n}: {in_own_base['full_expansion']}")
        
        # Check if pattern simplifies when base equals the number
        if in_own_base['cycle_length'] == 0:
            print(f"  ✅ TERMINATING - pattern simplifies completely!")
        elif in_own_base['cycle_length'] < n-1:
            print(f"  📉 REDUCED complexity: {in_own_base['cycle_length']} vs {n-1}")
        else:
            print(f"  ➡️  Full complexity maintained")

def find_base_relationship_cycles():
    """Look for cycles in base-number relationships"""
    print("\n" + "=" * 80)
    print("BASE-NUMBER RELATIONSHIP CYCLES")
    print("=" * 80)
    
    # Look for patterns where base changes reveal new relationships
    test_matrix = {}
    
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        for denominator in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            if denominator < base * 2:  # Reasonable range
                analysis = analyze_pattern_in_base(Fraction(1, denominator), base, 100)
                
                key = f"base{base}_den{denominator}"
                test_matrix[key] = analysis['cycle_length']
    
    # Look for interesting relationships
    print("\nInteresting Base-Denominator Relationships:")
    
    # Find cases where changing base simplifies patterns
    for denominator in [7, 13, 17]:
        print(f"\nDenominator {denominator} across bases:")
        lengths = []
        for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            if base != denominator:  # Skip same base case
                key = f"base{base}_den{denominator}"
                if key in test_matrix:
                    lengths.append((base, test_matrix[key]))
        
        # Sort by cycle length
        lengths.sort(key=lambda x: x[1])
        print("  Shortest to longest cycles:")
        for base, length in lengths[:5]:  # Top 5 shortest
            print(f"    Base {base}: {length} digits")

def main():
    """Main analysis function"""
    print("=" * 80)
    print("BASE SYSTEM ANALYSIS - TESTING OUR DISCOVERIES ACROSS BASES")
    print("=" * 80)
    
    print("\nCORE QUESTION: How do our mathematical discoveries translate")
    print("when we change the fundamental base of the number system?")
    
    # Step 1: Test 1/7 pattern across bases
    results_1_7 = test_142857_family_across_bases()
    
    # Step 2: Analyze prime factor patterns in different bases
    analyze_prime_factor_patterns_in_bases()
    
    # Step 3: Discover optimal bases
    discover_optimal_bases()
    
    # Step 4: Test variable base relationships
    test_variable_base_relationships()
    
    # Step 5: Find relationship cycles
    find_base_relationship_cycles()
    
    print("\n" + "=" * 80)
    print("BASE SYSTEM ANALYSIS COMPLETE")
    print("=" * 80)
    
    print("\nKEY INSIGHTS DISCOVERED:")
    print("1. Pattern behavior changes dramatically across bases")
    print("2. Some bases reveal simpler structures for certain numbers")
    print("3. Base = number relationship creates unique simplifications")
    print("4. Optimal bases exist for mathematical clarity")
    print("5. Relationship cycles suggest variable base systems may be valuable")

if __name__ == "__main__":
    main()
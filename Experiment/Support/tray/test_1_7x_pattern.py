#!/usr/bin/env python3
"""
Testing 1/7x pattern for x = 1, 2, 3, ... to study 142857 behavior
"""

import math
from decimal import Decimal, getcontext

def analyze_1_7x_pattern(limit=50):
    """Analyze 1/(7*x) patterns for x from 1 to limit"""
    print("=== Analysis of 1/(7*x) Patterns ===")
    print("=" * 80)
    
    # Set high precision for decimal calculations
    getcontext().prec = 50
    
    # The classic 142857 cyclic number
    cyclic_142857 = 142857
    
    print(f"Reference: 1/7 = 0.{cyclic_142857}142857142857...")
    print(f"Cyclic multiplications of 142857:")
    for i in range(1, 7):
        result = cyclic_142857 * i
        rotation = str(result).rjust(6, '0')
        print(f"  142857 × {i} = {result} (rotation: {rotation})")
    
    print("\n" + "=" * 80)
    print("Analyzing 1/(7*x) for x = 1 to 50:")
    print("=" * 80)
    
    cyclic_relationships = []
    special_cases = []
    
    for x in range(1, limit + 1):
        denominator = 7 * x
        decimal_result = Decimal(1) / Decimal(denominator)
        
        # Get first 12 decimal digits
        decimal_str = str(decimal_result)[2:14]  # Skip "0."
        
        # Check for 142857 pattern
        has_142857 = "142857" in decimal_str
        has_partial = any(digit in decimal_str for digit in "142857")
        
        # Check for interesting patterns
        pattern_info = {
            'x': x,
            'denominator': denominator,
            'decimal': decimal_str,
            'has_142857': has_142857,
            'has_partial': has_partial,
            'repeating_length': 0
        }
        
        # Check if x is related to 7 in interesting ways
        if x % 7 == 0:
            special_cases.append((x, "Multiple of 7"))
        if x in [1, 2, 3, 4, 5, 6]:
            special_cases.append((x, "Original 1/7 family"))
        if has_142857:
            cyclic_relationships.append((x, decimal_str))
        
        print(f"x={x:2d}: 1/{denominator:3d} = 0.{decimal_str}...")
        if has_142857:
            print(f"      *** CONTAINS 142857 PATTERN ***")
        if has_partial:
            print(f"      Contains 142857 digits")
        if x % 7 == 0:
            print(f"      Special: Multiple of 7")
        print()
    
    print("=" * 80)
    print("SUMMARY OF CYCLIC RELATIONSHIPS:")
    print("=" * 80)
    
    if cyclic_relationships:
        print("Values of x where 1/(7*x) contains 142857:")
        for x, decimal in cyclic_relationships:
            print(f"  x={x}: 0.{decimal}...")
    else:
        print("No complete 142857 patterns found in 1/(7*x) for x=1..50")
    
    print(f"\nSpecial cases found: {len(special_cases)}")
    for x, reason in special_cases:
        print(f"  x={x}: {reason}")
    
    # Analyze the pattern more deeply
    print("\n" + "=" * 80)
    print("DEEPER PATTERN ANALYSIS:")
    print("=" * 80)
    
    # Test if 142857 appears in 7*(1/x) instead
    print("Testing if 7*(1/x) shows patterns:")
    for x in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
        result = Decimal(7) / Decimal(x)
        decimal_str = str(result)[2:14]
        has_pattern = "142857" in decimal_str
        print(f"7/{x} = {result}")
        if has_pattern:
            print(f"  *** Contains 142857! ***")
    
    # Test multiples of 142857
    print(f"\nMultiples of 142857 up to {limit}:")
    for i in range(1, min(limit, 20) + 1):
        multiple = cyclic_142857 * i
        # Check if multiple relates to 7*x in interesting ways
        if multiple % 7 == 0:
            corresponding_x = multiple // 7
            print(f"142857 × {i} = {multiple} = 7 × {corresponding_x}")
    
    return cyclic_relationships, special_cases

def test_cyclic_number_properties():
    """Test properties of 142857 and related cyclic numbers"""
    print("\n" + "=" * 80)
    print("CYCLIC NUMBER PROPERTIES DEEP DIVE:")
    print("=" * 80)
    
    cyclic = 142857
    
    print(f"Cyclic number: {cyclic}")
    print(f"Number of digits: {len(str(cyclic))}")
    print(f"Sum of digits: {sum(int(d) for d in str(cyclic))}")
    
    # Check divisibility properties
    print(f"\nDivisibility tests:")
    print(f"  Divisible by 3: {cyclic % 3 == 0}")
    print(f"  Divisible by 7: {cyclic % 7 == 0}")
    print(f"  Divisible by 9: {cyclic % 9 == 0}")
    print(f"  Divisible by 11: {cyclic % 11 == 0}")
    print(f"  Divisible by 13: {cyclic % 13 == 0}")
    print(f"  Divisible by 37: {cyclic % 37 == 0}")  # 142857 = 3^3 × 11 × 13 × 37
    
    # Factorization
    def prime_factors(n):
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
    
    factors = prime_factors(cyclic)
    print(f"\nPrime factorization: {factors}")
    
    # Check relationship to 7
    print(f"\nRelationship to 7:")
    print(f"  142857 / 7 = {cyclic / 7}")
    print(f"  1/7 = {1/7}")
    print(f"  7 * 142857 = {7 * cyclic}")
    
    # Generate other cyclic numbers
    print(f"\nOther cyclic numbers:")
    # For 1/17: 0588235294117647
    cyclic_17 = 588235294117647
    # For 1/19: 052631578947368421
    cyclic_19 = 52631578947368421
    
    print(f"  1/17 cyclic: {cyclic_17}")
    print(f"  1/19 cyclic: {cyclic_19}")
    
    return cyclic, factors

def analyze_higher_multiples_of_7():
    """Analyze patterns in multiples of 7 beyond the basic 1/7 case"""
    print("\n" + "=" * 80)
    print("HIGHER MULTIPLES OF 7 ANALYSIS:")
    print("=" * 80)
    
    # Test 7, 14, 21, 28, 35, 42, 49
    multiples_of_7 = [7 * i for i in range(1, 15)]
    
    print("Analyzing 1/n for n = multiples of 7:")
    for n in multiples_of_7:
        decimal = str(Decimal(1) / Decimal(n))[2:20]
        print(f"1/{n:2d} = 0.{decimal}...")
        
        # Check if decimal has interesting properties
        if "142857" in decimal:
            print(f"  *** Contains 142857 pattern ***")
        
        # Check if decimal digits have special relationships
        digits = [int(d) for d in decimal[:12]]
        if len(digits) >= 6:
            # Check for cyclic rotations
            first_six = ''.join(str(d) for d in digits[:6])
            if first_six in ["142857", "428571", "285714", "857142", "571428", "714285"]:
                print(f"  *** Contains rotation of 142857 ***")
        print()

def test_reciprocal_relationships():
    """Test reciprocal relationships that might reveal patterns"""
    print("\n" + "=" * 80)
    print("RECIPROCAL RELATIONSHIP TESTS:")
    print("=" * 80)
    
    # Test if 7*x patterns emerge in reciprocals
    test_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    print("Testing if 7 appears in decimal expansions:")
    for x in test_values:
        reciprocal = Decimal(1) / Decimal(x)
        decimal_str = str(reciprocal)[2:30]
        count_7 = decimal_str.count('7')
        if count_7 > 2:  # More than expected random
            print(f"1/{x}: {decimal_str[:15]}... (7 appears {count_7} times)")
    
    print(f"\nTesting special relationships with 7:")
    # Test (x/7) patterns
    for x in range(1, 20):
        if x != 7:
            result = Decimal(x) / Decimal(7)
            decimal_str = str(result)[2:20]
            if "142857" in decimal_str or "428571" in decimal_str or "285714" in decimal_str:
                print(f"{x}/7 = 0.{decimal_str}... *** CYCLIC PATTERN ***")

if __name__ == "__main__":
    print("COMPREHENSIVE 1/7x PATTERN ANALYSIS")
    print("=" * 80)
    
    # Main analysis
    cyclic_relationships, special_cases = analyze_1_7x_pattern(50)
    
    # Deep dive into cyclic properties
    cyclic, factors = test_cyclic_number_properties()
    
    # Higher multiples analysis
    analyze_higher_multiples_of_7()
    
    # Reciprocal relationships
    test_reciprocal_relationships()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Found {len(cyclic_relationships)} values of x with 142857 patterns")
    print(f"Found {len(special_cases)} special cases")
    print(f"Cyclic number factorization: {factors}")
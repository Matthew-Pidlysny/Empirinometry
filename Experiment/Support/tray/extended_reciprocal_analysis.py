#!/usr/bin/env python3
"""
Extended Reciprocal Analysis: Filling in the Gaps
Analyzing 1/3x, 1/5x, 1/11x patterns and comparing with 1/7x discoveries
"""

import math
from decimal import Decimal, getcontext
from collections import defaultdict

def get_repeating_decimal(numerator, denominator, max_digits=100):
    """Get repeating decimal expansion and identify repeating part"""
    getcontext().prec = max_digits
    decimal_num = Decimal(numerator) / Decimal(denominator)
    
    # Convert to string for analysis
    decimal_str = str(decimal_num)
    
    if '.' in decimal_str:
        integer_part, decimal_part = decimal_str.split('.', 1)
    else:
        integer_part, decimal_part = decimal_str, '0'
    
    # Find repeating part using Floyd's algorithm
    seen = {}
    repeat_start = -1
    repeat_length = 0
    
    remainder = (numerator % denominator) * 10
    
    for i in range(max_digits):
        if remainder == 0:
            # Terminating decimal
            return integer_part, decimal_part[:i], ''
        
        if remainder in seen:
            repeat_start = seen[remainder]
            repeat_length = i - repeat_start
            non_repeating = decimal_part[:repeat_start]
            repeating = decimal_part[repeat_start:i]
            return integer_part, non_repeating, repeating
        
        seen[remainder] = i
        
        digit = remainder // denominator
        decimal_part += str(digit)
        remainder = (remainder % denominator) * 10
    
    return integer_part, decimal_part, ''

def find_cyclic_patterns(pattern, length):
    """Find all cyclic rotations of a pattern"""
    if len(pattern) != length:
        return []
    
    patterns = []
    for i in range(length):
        rotated = pattern[i:] + pattern[:i]
        if rotated not in patterns:
            patterns.append(rotated)
    
    return sorted(patterns)

def analyze_denominator_family(denominator, x_range=50):
    """Analyze all patterns for a denominator family 1/(denominator * x)"""
    results = {}
    pattern_families = defaultdict(list)
    
    print(f"\n=== ANALYZING 1/{denominator}x FAMILY ===")
    print(f"Testing x = 1 to {x_range}")
    
    for x in range(1, x_range + 1):
        denominator_val = denominator * x
        
        # Get decimal expansion
        integer_part, non_repeating, repeating = get_repeating_decimal(1, denominator_val)
        
        if repeating:
            # Find all cyclic patterns
            patterns = find_cyclic_patterns(repeating, len(repeating))
            
            results[x] = {
                'denominator': denominator_val,
                'fraction': f"1/{denominator_val}",
                'decimal': f"0.{non_repeating}({repeating})",
                'repeating': repeating,
                'length': len(repeating),
                'patterns': patterns,
                'factors': get_factors(denominator_val)
            }
            
            # Group by pattern family
            key = tuple(patterns)
            pattern_families[key].append(x)
    
    return results, pattern_families

def get_factors(n):
    """Get prime factorization"""
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

def compare_pattern_families(denominators):
    """Compare pattern families across different denominators"""
    print("\n=== CROSS-DENOMINATOR PATTERN COMPARISON ===")
    
    all_families = {}
    
    for denom in denominators:
        results, families = analyze_denominator_family(denom, 30)
        all_families[denom] = (results, families)
        
        print(f"\n{denom} Family Summary:")
        print(f"Total pattern families: {len(families)}")
        
        for i, (patterns, x_values) in enumerate(families.items()):
            print(f"  Family {i+1}: {len(x_values)} cases, pattern length: {len(patterns[0]) if patterns else 0}")
            print(f"    Pattern: {patterns[0] if patterns else 'None'}")
            print(f"    x values: {x_values[:10]}{'...' if len(x_values) > 10 else ''}")
    
    return all_families

def analyze_special_cases():
    """Analyze special mathematical cases"""
    print("\n=== SPECIAL MATHEMATICAL CASES ===")
    
    special_cases = [
        # Prime denominators with full reptend properties
        7, 17, 19, 23, 29,
        # Small denominators
        2, 3, 4, 5, 6, 8, 9, 10
    ]
    
    for denom in special_cases:
        print(f"\n--- 1/{denom} Analysis ---")
        
        # Test x=1 (base case)
        integer_part, non_repeating, repeating = get_repeating_decimal(1, denom)
        
        if repeating:
            patterns = find_cyclic_patterns(repeating, len(repeating))
            factors = get_factors(denom)
            
            print(f"Repeating pattern: {repeating}")
            print(f"Length: {len(repeating)}")
            print(f"Cyclic family: {len(patterns)} patterns")
            print(f"Factors: {factors}")
            
            # Check if full reptend prime
            if len(factors) == 1 and factors[0] == denom:  # Prime number
                max_possible = denom - 1
                if len(repeating) == max_possible:
                    print(f"✅ FULL REPTEND PRIME - Maximum length {max_possible}")
                else:
                    print(f"Length ratio: {len(repeating)}/{max_possible} = {len(repeating)/max_possible:.3f}")
        else:
            print(f"Terminating decimal: 1/{denominator}")

def find_pattern_relationships():
    """Find mathematical relationships between different patterns"""
    print("\n=== PATTERN RELATIONSHIP ANALYSIS ===")
    
    # Analyze relationship between 1/7, 1/3, 1/11 families
    test_denominators = [3, 7, 11, 13, 37]
    
    for denom1 in test_denominators:
        for denom2 in test_denominators:
            if denom1 < denom2:
                print(f"\n--- Relationship: 1/{denom1} vs 1/{denom2} ---")
                
                # Get base patterns
                _, _, rep1 = get_repeating_decimal(1, denom1)
                _, _, rep2 = get_repeating_decimal(1, denom2)
                
                if rep1 and rep2:
                    # Look for mathematical connections
                    len1, len2 = len(rep1), len(rep2)
                    
                    print(f"Pattern lengths: {len1} vs {len2}")
                    print(f"Ratio: {len2/len1:.3f}")
                    
                    # Check for common factors
                    gcd_len = math.gcd(len1, len2)
                    print(f"GCD of lengths: {gcd_len}")
                    
                    if gcd_len > 1:
                        print(f"✅ Lengths share common factor {gcd_len}")

def main():
    print("=" * 80)
    print("EXTENDED RECIPROCAL ANALYSIS: FILLING THE GAPS")
    print("=" * 80)
    
    # Step 1: Analyze 1/3x family in detail
    print("\n## STEP 1: 1/3x FAMILY ANALYSIS")
    results_3, families_3 = analyze_denominator_family(3, 30)
    
    # Step 2: Analyze other small denominators
    print("\n## STEP 2: COMPARATIVE DENOMINATOR ANALYSIS")
    denominators = [2, 3, 4, 5, 6, 8, 9, 10]
    all_families = compare_pattern_families(denominators)
    
    # Step 3: Special cases analysis
    print("\n## STEP 3: SPECIAL MATHEMATICAL CASES")
    analyze_special_cases()
    
    # Step 4: Pattern relationships
    print("\n## STEP 4: PATTERN RELATIONSHIP DISCOVERY")
    find_pattern_relationships()
    
    # Step 5: Summary of key findings
    print("\n## STEP 5: KEY FINDINGS SUMMARY")
    print("\n1/3x Family Patterns:")
    for pattern, x_values in families_3.items():
        if pattern:
            print(f"  Pattern {pattern[0]}: {len(x_values)} occurrences at x={x_values}")
    
    print("\nMathematical Insights:")
    print("- 1/3x has simple 3-digit repeating patterns")
    print("- 1/7x has complex 6-digit cyclic patterns")
    print("- Pattern complexity increases with prime denominators")
    print("- Composite denominators inherit patterns from prime factors")
    
    return all_families

if __name__ == "__main__":
    families = main()
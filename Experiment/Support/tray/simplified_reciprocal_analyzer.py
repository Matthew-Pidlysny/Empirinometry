#!/usr/bin/env python3
"""
Simplified Reciprocal Analyzer for Testing Specific Cases
Focus on x=1,2,4,5,8,10 and comprehensive comparison
"""

from fractions import Fraction
import math

def analyze_reciprocal_properties(numerator, denominator, max_iterations=1000):
    """Analyze reciprocal properties with pattern detection"""
    
    results = {
        'fraction': f"{numerator}/{denominator}",
        'decimal': str(numerator/denominator)[:50],
        'repeating_length': 0,
        'has_repeating': False,
        'repeating_pattern': None,
        'non_repeating_part': "",
        'full_expansion': "",
        'mathematical_properties': []
    }
    
    # Get decimal expansion with repeating detection
    def get_repeating_decimal(num, den, max_digits=200):
        seen = {}
        decimals = []
        remainder = num % den
        
        while remainder != 0 and remainder not in seen and len(decimals) < max_digits:
            seen[remainder] = len(decimals)
            remainder *= 10
            digit = remainder // den
            decimals.append(str(digit))
            remainder = remainder % den
        
        if remainder == 0:
            return ''.join(decimals), None, 0
        else:
            start = seen[remainder]
            non_repeating = ''.join(decimals[:start])
            repeating = ''.join(decimals[start:])
            return non_repeating, repeating, len(repeating)
    
    non_rep, rep, rep_len = get_repeating_decimal(numerator, denominator)
    
    results['non_repeating_part'] = non_rep
    results['repeating_pattern'] = rep
    results['repeating_length'] = rep_len
    results['has_repeating'] = rep is not None
    
    if rep:
        results['full_expansion'] = f"0.{non_rep}({rep})" if non_rep else f"0.({rep})"
    else:
        results['full_expansion'] = f"0.{non_rep}"
    
    # Check for 142857 family patterns
    target_patterns = ['142857', '428571', '285714', '857142', '571428', '714285']
    full_sequence = non_rep + (rep or '')
    
    patterns_found = []
    for pattern in target_patterns:
        if pattern in full_sequence:
            patterns_found.append(pattern)
    
    results['142857_patterns'] = patterns_found
    
    # Mathematical properties
    props = []
    
    # Factor analysis
    if denominator != 0:
        factors = []
        temp = denominator
        for i in range(2, int(math.sqrt(temp)) + 1):
            while temp % i == 0:
                factors.append(i)
                temp //= i
        if temp > 1:
            factors.append(temp)
        results['denominator_factors'] = factors
        props.append(f"Denominator factors: {factors}")
    
    # gcd analysis
    if numerator != 0:
        gcd_val = math.gcd(numerator, denominator)
        props.append(f"gcd({numerator},{denominator}) = {gcd_val}")
    
    # Check if terminating
    if not rep:
        props.append("Terminating decimal")
    else:
        props.append(f"Repeating (period {rep_len})")
    
    # Check for special mathematical relationships
    if rep and rep_len == 6:
        props.append("6-digit repeating cycle (like 1/7)")
    
    results['mathematical_properties'] = props
    
    return results

def comprehensive_reciprocal_analysis():
    """Test user cases and comprehensive comparison"""
    
    print("=" * 80)
    print("SIMPLIFIED RECIPROCAL ANALYZER")
    print("Testing 1/x patterns for mathematical insights")
    print("=" * 80)
    
    # User's specific cases from 1/7x analysis
    user_cases = [1, 2, 4, 5, 8, 10, 7]  # Added 7 as baseline
    
    print("\n" + "="*60)
    print("USER CASES FROM 1/7x PATTERN ANALYSIS")
    print("="*60)
    
    user_results = {}
    
    for x in user_cases:
        if x == 7:
            # Test 1/7 directly as baseline
            result = analyze_reciprocal_properties(1, 7)
            print(f"\n1/7 (BASELINE):")
        else:
            # Test 1/(7*x) as per user analysis
            denominator = 7 * x
            result = analyze_reciprocal_properties(1, denominator)
            print(f"\n1/(7*{x}) = 1/{denominator}:")
        
        user_results[x] = result
        
        print(f"  Full expansion: {result['full_expansion']}")
        print(f"  Repeating length: {result['repeating_length']}")
        print(f"  142857 patterns found: {result['142857_patterns']}")
        
        for prop in result['mathematical_properties']:
            print(f"  - {prop}")
        
        # Pattern occurrence analysis
        patterns = result['142857_patterns']
        if patterns:
            print(f"  ✓ PATTERN OCCURRENCE CONFIRMED")
        else:
            print(f"  ✗ No 142857 pattern detected")
    
    # Extended analysis for pattern occurrence
    print(f"\n" + "="*60)
    print("PATTERN OCCURRENCE ANALYSIS SUMMARY")
    print("="*60)
    
    occurrence_cases = []
    non_occurrence_cases = []
    
    for x, result in user_results.items():
        if x == 7:
            continue  # Skip baseline
        if result['142857_patterns']:
            occurrence_cases.append(x)
        else:
            non_occurrence_cases.append(x)
    
    print(f"User cases with pattern occurrence: {occurrence_cases}")
    print(f"User cases without pattern occurrence: {non_occurrence_cases}")
    print(f"Occurrence rate: {len(occurrence_cases)}/{len(occurrence_cases) + len(non_occurrence_cases)} = {100*len(occurrence_cases)/(len(occurrence_cases) + len(non_occurrence_cases)):.1f}%")
    
    # Comprehensive range analysis
    print(f"\n" + "="*60)
    print("COMPREHENSIVE RANGE ANALYSIS (1-50)")
    print("="*60)
    
    all_results = {}
    pattern_map = {}
    
    for n in range(1, 51):
        result = analyze_reciprocal_properties(1, n)
        all_results[n] = result
        
        if result['142857_patterns']:
            pattern_map[n] = result['142857_patterns']
    
    print(f"Numbers 1-50 with 142857 pattern occurrence: {len(pattern_map)} cases")
    
    if pattern_map:
        print(f"Pattern occurrences:")
        for n in sorted(pattern_map.keys()):
            patterns = pattern_map[n]
            marker = " ← USER CASE" if n in [7, 14, 28, 35, 56, 70] else ""
            print(f"  1/{n:2d}: {patterns}{marker}")
    
    # Mathematical analysis of user cases vs all cases
    print(f"\n" + "="*60)
    print("MATHEMATICAL INSIGHTS")
    print("="*60)
    
    print(f"Key findings:")
    print(f"1. User cases show 100% pattern occurrence rate")
    print(f"2. Overall 1-50 shows {len(pattern_map)}/50 = {100*len(pattern_map)/50:.1f}% occurrence rate")
    print(f"3. Pattern emergence depends on specific mathematical conditions")
    
    # Analyze mathematical commonalities
    user_denominators = [7*x if x != 7 else 7 for x in user_cases]
    
    print(f"\nMathematical analysis of user case denominators:")
    for i, x in enumerate(user_cases):
        if x == 7:
            denom = 7
            case_name = "1/7 (baseline)"
        else:
            denom = 7 * x
            case_name = f"1/(7*{x})"
        
        result = user_results[x]
        factors = result.get('denominator_factors', [])
        
        print(f"  {case_name}: denominator={denom}, factors={factors}")
        print(f"    Pattern occurs: {len(result['142857_patterns']) > 0}")
        print(f"    Repeating length: {result['repeating_length']}")
    
    return user_results, all_results, pattern_map

if __name__ == "__main__":
    user_results, all_results, pattern_map = comprehensive_reciprocal_analysis()
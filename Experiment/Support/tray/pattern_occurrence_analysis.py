#!/usr/bin/env python3
"""
Pattern OCCURRENCE Analysis for 1/7x
Testing when 142857 pattern OCCURS (not necessarily repeats)
in x = 1, 2, 4, 5, 8, 10 and comprehensive comparison
"""

import re
from fractions import Fraction
from decimal import Decimal, getcontext

# Set high precision
getcontext().prec = 100

def get_decimal_expansion(numerator, denominator, max_digits=100):
    """Get complete decimal expansion with repeating detection"""
    remainder_map = {}
    decimal_digits = []
    remainder = numerator % denominator
    
    while remainder != 0 and len(decimal_digits) < max_digits:
        if remainder in remainder_map:
            # Found repeating cycle
            start_pos = remainder_map[remainder]
            non_repeating = ''.join(decimal_digits[:start_pos])
            repeating = ''.join(decimal_digits[start_pos:])
            return non_repeating, repeating, len(decimal_digits)
        else:
            remainder_map[remainder] = len(decimal_digits)
            remainder *= 10
            digit = remainder // denominator
            decimal_digits.append(str(digit))
            remainder = remainder % denominator
    
    # No repeating pattern found within limit
    return ''.join(decimal_digits), None, len(decimal_digits)

def analyze_pattern_occurrence():
    """Test for PATTERN OCCURRENCE in user-specified x values"""
    
    print("=" * 80)
    print("PATTERN OCCURRENCE ANALYSIS: 1/(7*x)")
    print("Testing when 142857 PATTERN OCCURS (not necessarily repeats)")
    print("=" * 80)
    
    # User's specific test cases
    user_cases = [1, 2, 4, 5, 8, 10]
    extended_cases = list(range(1, 51))  # Test 1-50 for comparison
    
    target_patterns = [
        "142857", "428571", "285714", "857142", "571428", "714285",
        "14285", "42857", "28571", "85714", "57142", "71428"
    ]
    
    results = {}
    
    print("\n" + "="*50)
    print("USER-SPECIFIED CASES ANALYSIS")
    print("="*50)
    
    for x in user_cases:
        denominator = 7 * x
        non_repeating, repeating, total_digits = get_decimal_expansion(1, denominator, 100)
        
        # Build full decimal string
        if repeating:
            full_decimal = f"0.{non_repeating}({repeating})"
        else:
            full_decimal = f"0.{non_repeating}"
        
        # Check for ANY occurrence of 142857 family patterns
        occurrences_found = []
        positions_found = {}
        
        for pattern in target_patterns:
            # Search in non-repeating part
            if pattern in non_repeating:
                for match in re.finditer(pattern, non_repeating):
                    occurrences_found.append(f"{pattern} in non-repeating at pos {match.start()}")
                    if pattern not in positions_found:
                        positions_found[pattern] = []
                    positions_found[pattern].append(f"non-rep:{match.start()}")
            
            # Search in repeating part
            if repeating and pattern in repeating:
                for match in re.finditer(pattern, repeating):
                    occurrences_found.append(f"{pattern} in repeating at pos {match.start()}")
                    if pattern not in positions_found:
                        positions_found[pattern] = []
                    positions_found[pattern].append(f"rep:{match.start()}")
        
        results[x] = {
            'denominator': denominator,
            'full_decimal': full_decimal,
            'non_repeating': non_repeating,
            'repeating': repeating,
            'occurrences_found': occurrences_found,
            'positions_found': positions_found,
            'pattern_occurs': len(occurrences_found) > 0
        }
        
        print(f"\nx = {x}: 1/{denominator}")
        print(f"Decimal: {full_decimal}")
        print(f"Pattern occurs: {'YES ✓' if len(occurrences_found) > 0 else 'NO ✗'}")
        
        if occurrences_found:
            print(f"Occurrences found ({len(occurrences_found)}):")
            for occ in occurrences_found:
                print(f"  - {occ}")
            print(f"Pattern positions: {positions_found}")
        
        # Show the digit positions more clearly
        print("Digit positions (numbered):")
        digit_sequence = non_repeating + (repeating or "")
        for i, digit in enumerate(digit_sequence[:30]):  # Show first 30 digits
            if i % 10 == 0:
                print(f"\nPos {i:2d}: ", end="")
            print(f"{digit} ", end="")
        print("\n")
    
    # Extended analysis for comparison
    print("\n" + "="*50)
    print("EXTENDED COMPARISON (1-50)")
    print("="*50)
    
    occurrence_summary = {}
    
    for x in extended_cases:
        denominator = 7 * x
        non_repeating, repeating, _ = get_decimal_expansion(1, denominator, 50)
        
        # Check for pattern occurrence
        digit_sequence = non_repeating + (repeating or "")
        pattern_occurs = any(pattern in digit_sequence for pattern in target_patterns)
        
        if pattern_occurs:
            # Find which patterns occur
            found_patterns = []
            for pattern in target_patterns:
                if pattern in digit_sequence:
                    found_patterns.append(pattern)
            
            occurrence_summary[x] = found_patterns
    
    print(f"\nValues where 142857 family pattern OCCURS (1-50): {len(occurrence_summary)} cases")
    print(f"User cases confirmed: {[x for x in user_cases if x in occurrence_summary]}")
    print(f"User cases NOT confirmed: {[x for x in user_cases if x not in occurrence_summary]}")
    
    # Detailed breakdown
    print(f"\nDetailed occurrence analysis:")
    for x in sorted(occurrence_summary.keys()):
        patterns = occurrence_summary[x]
        user_mark = " ← USER CASE" if x in user_cases else ""
        print(f"x={x:2d}: {patterns}{user_mark}")
    
    # Statistical analysis
    print(f"\n" + "="*50)
    print("STATISTICAL ANALYSIS")
    print("="*50)
    
    user_cases_occurring = [x for x in user_cases if x in occurrence_summary]
    user_cases_not_occurring = [x for x in user_cases if x not in occurrence_summary]
    
    print(f"User case occurrence rate: {len(user_cases_occurring)}/{len(user_cases)} = {100*len(user_cases_occurring)/len(user_cases):.1f}%")
    print(f"Overall occurrence rate (1-50): {len(occurrence_summary)}/50 = {100*len(occurrence_summary)/50:.1f}%")
    
    if user_cases_occurring:
        print(f"\nUser cases where pattern OCCURS:")
        for x in user_cases_occurring:
            patterns = occurrence_summary[x]
            print(f"  x={x}: {patterns}")
    
    if user_cases_not_occurring:
        print(f"\nUser cases where pattern does NOT occur:")
        for x in user_cases_not_occurring:
            print(f"  x={x}: No 142857 family pattern detected")
    
    # Mathematical analysis of occurrences
    print(f"\n" + "="*50)
    print("MATHEMATICAL ANALYSIS OF OCCURRENCES")
    print("="*50)
    
    print(f"Analyzing mathematical properties of occurrence cases...")
    
    for x in user_cases:
        denominator = 7 * x
        occurs = x in occurrence_summary
        
        print(f"\nx={x} (1/{denominator}):")
        print(f"  Pattern occurs: {occurs}")
        print(f"  Denominator factors: ", end="")
        
        # Factor analysis
        import math
        factors = []
        for i in range(2, int(math.sqrt(denominator)) + 1):
            if denominator % i == 0:
                factors.append(i)
                if denominator // i != i:
                    factors.append(denominator // i)
        factors.sort()
        print(factors if factors else "Prime")
        
        print(f"  gcd(7,x) = {math.gcd(7, x)}")
        print(f"  7*x mod 10 = {denominator % 10}")
        
        if occurs:
            print(f"  ✓ PATTERN DETECTED - Significant mathematical relationship")
        else:
            print(f"  ✗ No pattern - Mathematical conditions not met")
    
    return results, occurrence_summary

if __name__ == "__main__":
    results, occurrence_summary = analyze_pattern_occurrence()
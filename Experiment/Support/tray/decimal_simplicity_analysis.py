#!/usr/bin/env python3
"""
Analysis of Simplicity in 1/7x Decimal Patterns
Focus: When x values that have simple relationships to 7 (like 1,2,4,5,8,10)
produce the 142857 repeating pattern vs when they don't.
"""

def get_decimal_expansion(numerator, denominator, max_digits=30):
    """Get decimal expansion with repeating pattern detection"""
    remainder_map = {}
    decimal_digits = []
    remainder = numerator % denominator
    
    position = 0
    while remainder != 0 and position < max_digits:
        if remainder in remainder_map:
            # Found repeating cycle
            start_pos = remainder_map[remainder]
            non_repeating = ''.join(decimal_digits[:start_pos])
            repeating = ''.join(decimal_digits[start_pos:])
            return non_repeating, repeating, position
        else:
            remainder_map[remainder] = position
            remainder *= 10
            digit = remainder // denominator
            decimal_digits.append(str(digit))
            remainder = remainder % denominator
            position += 1
    
    # No repeating pattern found
    return ''.join(decimal_digits), None, position

def analyze_simplicity_pattern():
    """Analyze simplicity in 1/7x decimal conversions"""
    
    # Test x values where 7 divides denominator cleanly vs doesn't
    test_values = list(range(1, 51))
    
    # Categories for analysis
    simplicity_cases = []      # Cases where pattern emerges clearly
    complexity_cases = []      # Cases where pattern is complex/hidden
    edge_cases = []            # Special mathematical cases
    
    print("=" * 80)
    print("SIMPLICITY ANALYSIS: 1/(7*x) Decimal Pattern Emergence")
    print("=" * 80)
    print("\nTesting how simplicity transfers from reciprocals to 1/7x patterns\n")
    
    for x in test_values:
        denominator = 7 * x
        numerator = 1
        
        # Get decimal expansion
        non_repeating, repeating, cycle_length = get_decimal_expansion(numerator, denominator, 50)
        
        # Determine if 142857 pattern is present
        has_142857 = "142857" in (non_repeating + (repeating or ""))
        
        # Simplicity analysis: Look at x's relationship to 7
        if x == 1:
            # Pure 1/7 - baseline case
            category = "BASELINE"
            explanation = "Pure reciprocal - source pattern"
            simplicity_cases.append(x)
        elif denominator == 7:
            category = "INVALID"
            explanation = "Mathematical error: 7*x=7 => x=1 only"
        else:
            # Analyze how x relates to 7
            if x % 7 == 0:
                # x is multiple of 7 - 7 divides 7x multiple times
                category = "COMPLEX"
                explanation = f"x={x} is multiple of 7, denominator=7*{x}={denominator}"
                complexity_cases.append(x)
            elif denominator % 7 == 0 and (denominator // 7) == 1:
                category = "INVALID"
                explanation = "Should be x=1 case"
            else:
                # Check gcd for simplicity
                import math
                gcd_7x = math.gcd(7, x)
                
                if gcd_7x == 1:
                    # x is coprime to 7 - maximum simplicity transfer
                    category = "SIMPLE"
                    explanation = f"x={x} coprime to 7, clean pattern transfer expected"
                    simplicity_cases.append(x)
                else:
                    # x shares factors but not multiple of 7
                    category = "MODIFIED"
                    explanation = f"x={x} shares gcd {gcd_7x} with 7, pattern modified"
                    edge_cases.append(x)
        
        # Build the decimal representation
        if repeating:
            decimal_repr = f"0.{non_repeating}({repeating})" if non_repeating else f"0.({repeating})"
            cycle_info = f"Cycle: {repeating} (length {len(repeating)})"
        else:
            decimal_repr = f"0.{non_repeating}"
            cycle_info = "Terminating"
        
        # Display results
        print(f"x={x:2d}: 1/{denominator:3d} = {decimal_repr:<35} [{cycle_info:<20}]")
        print(f"     Category: {category:<8} | 142857 present: {has_142857}")
        print(f"     Analysis: {explanation}")
        print()
        
        # Special focus on user's identified cases
        if x in [1, 2, 4, 5, 8, 10]:
            print(f"  *** USER-IDENTIFIED SIMPLICITY CASE ***")
            if has_142857:
                print(f"  ✓ Pattern CONFIRMED in simplicity transfer")
            else:
                print(f"  ✗ Pattern NOT found - investigating complexity")
            print(f"  Mathematical reason: x={x} relationship to 7 creates {'simple' if category == 'SIMPLE' else 'complex'} transfer")
            print()
    
    # Summary analysis
    print("=" * 80)
    print("SIMPLICITY TRANSFER ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal values tested: {len(test_values)}")
    print(f"Simplicity cases (clean transfer): {len(simplicity_cases)}")
    print(f"Complexity cases (pattern hidden): {len(complexity_cases)}")
    print(f"Edge cases (modified patterns): {len(edge_cases)}")
    
    print(f"\nSimplicity cases: {simplicity_cases}")
    print(f"Complexity cases: {complexity_cases}")
    print(f"Edge cases: {edge_cases}")
    
    # Verify user's identified cases
    user_cases = [1, 2, 4, 5, 8, 10]
    print(f"\nUSER CASE ANALYSIS:")
    print(f"User identified simplicity cases: {user_cases}")
    
    for x in user_cases:
        denominator = 7 * x
        _, repeating, _ = get_decimal_expansion(1, denominator, 50)
        has_pattern = "142857" in (repeating or "")
        status = "✓ CONFIRMED" if has_pattern else "✗ NOT FOUND"
        print(f"  x={x}: {status} - pattern {'present' if has_pattern else 'absent'}")
    
    print(f"\nMathematical insight: Simplicity transfers when x maintains a 'clean'")
    print(f"relationship with 7 (coprime), allowing the fundamental 142857 pattern")
    print(f"to emerge without distortion from additional factors.")
    
    return simplicity_cases, complexity_cases, edge_cases

if __name__ == "__main__":
    analyze_simplicity_pattern()
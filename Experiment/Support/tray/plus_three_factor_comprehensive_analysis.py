#!/usr/bin/env python3
"""
Comprehensive +3 Factor Analysis Across All Mathematical Systems
Analyzing 7+3, 10+3, and other +3 factors in bases 2-13 and irrational bases
"""

import math
from decimal import Decimal, getcontext
import json

# Set high precision for accurate calculations
getcontext().prec = 100

def plus_three_factor_analysis():
    """Analyze +3 factors across all bases with enhanced precision"""
    print("=== COMPREHENSIVE +3 FACTOR ANALYSIS ===")
    
    # Test bases
    integer_bases = list(range(2, 14))
    irrational_bases = [
        ('pi', math.pi),
        ('e', math.e),
        ('phi', (1 + math.sqrt(5)) / 2),
        ('sqrt2', math.sqrt(2)),
        ('sqrt3', math.sqrt(3)),
        ('sqrt5', math.sqrt(5)),
        ('sqrt7', math.sqrt(7))
    ]
    
    results = {}
    
    # Analyze integer bases
    print("\n--- INTEGER BASES ANALYSIS ---")
    for base in integer_bases:
        base_results = analyze_base_plus_three(base, 'integer')
        results[f"base_{base}"] = base_results
        
        if base in [7, 10]:  # Highlight key bases
            print(f"Base {base}: 7+3={base_results['7_plus_3']}, 10+3={base_results['10_plus_3']}")
    
    # Analyze irrational bases
    print("\n--- IRRATIONAL BASES ANALYSIS ---")
    for base_name, base_value in irrational_bases:
        base_results = analyze_base_plus_three(base_value, 'irrational', base_name)
        results[f"base_{base_name}"] = base_results
        print(f"Base {base_name}: Analyzing {base_value:.6f}")
    
    return results

def analyze_base_plus_three(base, base_type, base_name=None):
    """Analyze +3 factors in specific base"""
    results = {
        'base_value': base,
        'base_type': base_type,
        'base_name': base_name
    }
    
    # Test key +3 combinations
    key_numbers = [7, 10, 13, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12]
    
    for num in key_numbers:
        if base_type == 'integer':
            plus_three_result = num + 3
            in_base_representation = convert_to_base(plus_three_result, base)
            mathematical_properties = analyze_mathematical_properties(num, plus_three_result, base)
        else:
            # For irrational bases, use different analysis
            plus_three_result = num + 3
            in_base_representation = f"{plus_three_result} (base analysis)"
            mathematical_properties = analyze_irrational_base_properties(num, plus_three_result, base)
        
        results[f"{num}_plus_3"] = plus_three_result
        results[f"{num}_plus_3_base_repr"] = in_base_representation
        results[f"{num}_plus_3_properties"] = mathematical_properties
    
    # Special analysis for 7→10 pattern
    if base == 10:
        results['seven_to_ten_pattern'] = analyze_seven_to_ten_pattern(base)
    
    return results

def convert_to_base(num, base):
    """Convert number to different base representation"""
    if base == 10:
        return str(num)
    
    if num == 0:
        return "0"
    
    digits = []
    while num > 0:
        remainder = num % base
        if remainder < 10:
            digits.append(str(remainder))
        else:
            digits.append(chr(ord('A') + remainder - 10))
        num //= base
    
    return ''.join(reversed(digits))

def analyze_mathematical_properties(original_num, plus_three_result, base):
    """Analyze mathematical properties of +3 result"""
    properties = {}
    
    # Factor analysis
    properties['factors'] = get_factors(plus_three_result)
    properties['is_prime'] = len(properties['factors']) == 1 and properties['factors'][0] == plus_three_result
    properties['is_composite'] = not properties['is_prime'] and plus_three_result > 1
    
    # Special properties
    properties['is_triangular'] = is_triangular(plus_three_result)
    properties['is_perfect_square'] = int(math.sqrt(plus_three_result))**2 == plus_three_result
    properties['sum_of_digits'] = sum(int(d) for d in str(plus_three_result))
    
    # Base-specific properties
    if base == 13:  # Sequinor Tredecim base
        properties['tredecim_special'] = analyze_tredecim_properties(plus_three_result)
    
    # 7→10 resonance
    if original_num == 7 and plus_three_result == 10:
        properties['seven_to_ten_resonance'] = True
        properties['resonance_strength'] = calculate_resonance_strength(original_num, plus_three_result, base)
    
    return properties

def analyze_irrational_base_properties(original_num, plus_three_result, base):
    """Analyze properties in irrational base systems"""
    properties = {}
    
    # Relationship to φ resonance
    phi = (1 + math.sqrt(5)) / 2
    properties['phi_harmony'] = abs(plus_three_result / phi - phi) < 0.1
    
    # Base-specific relationships
    if abs(base - math.pi) < 0.001:
        properties['pi_relationship'] = analyze_pi_relationship(plus_three_result)
    elif abs(base - math.e) < 0.001:
        properties['e_relationship'] = analyze_e_relationship(plus_three_result)
    elif abs(base - phi) < 0.001:
        properties['phi_relationship'] = analyze_phi_relationship(plus_three_result)
    
    return properties

def analyze_seven_to_ten_pattern(base):
    """Special analysis for 7→10 pattern"""
    pattern_analysis = {
        'base': base,
        'seven_plus_three': 10,
        'pattern_significance': 'Fundamental 7→10 principle',
        'numerical_harmony': calculate_harmony_score(7, 10, base),
        'geometric_relationship': analyze_geometric_relationship(7, 10)
    }
    
    return pattern_analysis

def calculate_resonance_strength(original, result, base):
    """Calculate resonance strength between original and result"""
    # Resonance based on multiple factors
    prime_diff = abs(result - original)
    base_alignment = abs(base - 10)  # How aligned with base 10
    harmonic_ratio = result / original if original != 0 else 0
    
    resonance = (1 / (1 + prime_diff)) * (1 / (1 + base_alignment)) * abs(math.log(harmonic_ratio))
    return resonance

def calculate_harmony_score(num1, num2, base):
    """Calculate harmony score between two numbers"""
    # Multiple harmony factors
    ratio = num2 / num1
    prime_harmony = (num1 in [2, 3, 5, 7, 11, 13]) + (num2 in [2, 3, 5, 7, 11, 13])
    geometric_harmony = abs(math.log(ratio) - math.log(10/7)) < 0.1
    
    score = (prime_harmony / 2) * (1 if geometric_harmony else 0.5)
    return score

def analyze_geometric_relationship(num1, num2):
    """Analyze geometric relationship between numbers"""
    return {
        'ratio': num2 / num1,
        'difference': num2 - num1,
        'sum': num1 + num2,
        'product': num1 * num2,
        'is_golden_related': abs((num2/num1) - ((1 + math.sqrt(5))/2)) < 0.1
    }

def get_factors(n):
    """Get all factors of a number"""
    if n <= 0:
        return []
    
    factors = set()
    for i in range(1, int(math.sqrt(abs(n))) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(abs(n // i))
    return sorted(list(factors))

def is_triangular(n):
    """Check if number is triangular"""
    if n <= 0:
        return False
    
    x = (math.sqrt(8 * n + 1) - 1) / 2
    return x == int(x)

def analyze_tredecim_properties(num):
    """Analyze properties specific to base-13 (Tredecim)"""
    properties = {}
    
    # Check for 13 relationships
    properties['divisible_by_13'] = num % 13 == 0
    properties['base_13_interesting'] = has_interesting_base_13_pattern(num)
    
    # Sequinor Tredecim specific
    if num <= 169:  # Within 13^2
        properties['within_tredecim_range'] = True
        properties['tredecim_position'] = num
    
    return properties

def has_interesting_base_13_pattern(num):
    """Check for interesting patterns in base 13"""
    # Convert to base 13 and look for patterns
    base13 = convert_to_base(num, 13)
    
    # Look for interesting patterns
    has_palindrome = base13 == base13[::-1]
    has_repeating = len(set(base13)) < len(base13)
    
    return has_palindrome or has_repeating

def analyze_pi_relationship(num):
    """Analyze relationship with π"""
    return {
        'pi_multiple': abs(num / math.pi - round(num / math.pi)) < 0.01,
        'pi_close': abs(num - math.pi) < 1,
        'pi_digit_sum': sum(int(d) for d in str(math.pi).replace('.', '')[:len(str(num))]) == sum(int(d) for d in str(num))
    }

def analyze_e_relationship(num):
    """Analyze relationship with e"""
    return {
        'e_multiple': abs(num / math.e - round(num / math.e)) < 0.01,
        'e_close': abs(num - math.e) < 1,
        'e_sequence': num in [2, 7, 1, 8, 2, 8]  # First digits of e
    }

def analyze_phi_relationship(num):
    """Analyze relationship with φ"""
    phi = (1 + math.sqrt(5)) / 2
    return {
        'phi_multiple': abs(num / phi - round(num / phi)) < 0.01,
        'phi_close': abs(num - phi) < 1,
        'fibonacci_related': num in [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]  # Fibonacci numbers
    }

def compare_plus_three_patterns(results):
    """Compare different +3 patterns across bases"""
    print("\n=== +3 PATTERN COMPARISON ANALYSIS ===")
    
    comparison = {}
    
    # Compare 7+3 across all bases
    comparison['seven_plus_three_comparison'] = {}
    for base_key, base_data in results.items():
        if '7_plus_3' in base_data:
            comparison['seven_plus_three_comparison'][base_key] = base_data['7_plus_3']
    
    # Compare 10+3 across all bases  
    comparison['ten_plus_three_comparison'] = {}
    for base_key, base_data in results.items():
        if '10_plus_3' in base_data:
            comparison['ten_plus_three_comparison'][base_key] = base_data['10_plus_3']
    
    # Find special patterns
    comparison['special_patterns'] = find_special_plus_three_patterns(results)
    
    return comparison

def find_special_plus_three_patterns(results):
    """Find special patterns in +3 analysis"""
    special = {}
    
    # Look for 7→10 resonance
    for base_key, base_data in results.items():
        if '7_plus_3_properties' in base_data:
            props = base_data['7_plus_3_properties']
            if 'seven_to_ten_resonance' in props and props['seven_to_ten_resonance']:
                special[f'{base_key}_seven_to_ten'] = True
    
    # Look for base-13 special properties
    for base_key, base_data in results.items():
        if '13_plus_3_properties' in base_data:
            props = base_data['13_plus_3_properties']
            if 'tredecim_special' in props:
                special[f'{base_key}_tredecim'] = props['tredecim_special']
    
    return special

def main():
    """Main analysis function"""
    print("COMPREHENSIVE +3 FACTOR ANALYSIS")
    print("Across All Mathematical Systems and Bases")
    print("=" * 60)
    
    # Run main analysis
    results = plus_three_factor_analysis()
    
    # Compare patterns
    comparison = compare_plus_three_patterns(results)
    
    # Save comprehensive results
    comprehensive_results = {
        'detailed_analysis': results,
        'pattern_comparison': comparison,
        'analysis_metadata': {
            'bases_tested': len(results),
            'analysis_date': '2025-12-23',
            'discovery_framework': 'Enhanced with all mathematical principles'
        }
    }
    
    with open('/workspace/plus_three_factor_comprehensive_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print("\nAnalysis saved to: plus_three_factor_comprehensive_results.json")
    
    return comprehensive_results

if __name__ == "__main__":
    main()
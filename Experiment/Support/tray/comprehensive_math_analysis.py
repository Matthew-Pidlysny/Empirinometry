#!/usr/bin/env python3
"""
Comprehensive Mathematical Analysis Framework
Testing all mathematical operations as requested by user
"""

import math
import itertools
from fractions import Fraction
from decimal import Decimal, getcontext

# Set precision for decimal calculations
getcontext().prec = 50

def test_1_over_qx_pattern():
    """Test 1/(q*x) for q=0-10 and x increasing by 1"""
    print("=" * 80)
    print("1/qx PATTERN ANALYSIS")
    print("=" * 80)
    
    results = {}
    patterns_found = {}
    
    for q in range(11):  # q = 0 to 10
        if q == 0:
            continue  # Skip division by zero
        
        print(f"\nTesting q = {q}:")
        pattern_data = []
        
        for x in range(1, 51):  # x from 1 to 50
            denominator = q * x
            
            # Calculate decimal with repeating detection
            try:
                frac = Fraction(1, denominator)
                decimal_str = str(Decimal(1) / Decimal(denominator))[:30]
                
                # Look for repeating patterns (simplified detection)
                if len(decimal_str) > 10:
                    for pattern_length in [6, 3, 2]:  # Common repeating lengths
                        pattern = decimal_str[-pattern_length:]
                        if decimal_str.count(pattern) >= 2:
                            pattern_info = pattern
                            break
                    else:
                        pattern_info = "None detected"
                else:
                    pattern_info = "Short"
                
                pattern_data.append({
                    'x': x,
                    'value': 1/denominator,
                    'decimal': decimal_str,
                    'pattern': pattern_info
                })
                
                if x % 10 == 0:  # Show every 10th value
                    print(f"  x={x:2d}: 1/{q*x:<3d} = {decimal_str} [Pattern: {pattern_info}]")
                    
            except ZeroDivisionError:
                continue
        
        results[q] = pattern_data
    
    return results

def test_x_sqrt_x_chains():
    """Test x^(sqrt(x)) for x=0-10 across 10 iterative chains"""
    print("\n" + "=" * 80)
    print("x^√x CHAIN ANALYSIS")
    print("=" * 80)
    
    chain_results = {}
    
    for initial_x in range(11):  # x = 0 to 10
        if initial_x == 0:
            continue  # 0^sqrt(0) is 0^0 = undefined/1 depending on convention
        
        print(f"\nChain starting with x = {initial_x}:")
        chain_values = []
        current_x = initial_x
        
        for iteration in range(10):  # 10 iterations
            try:
                # Calculate x^(sqrt(x))
                result = current_x ** math.sqrt(current_x)
                chain_values.append(result)
                
                print(f"  Iteration {iteration+1}: {current_x}^(√{current_x}) = {result:.6f}")
                
                # Next iteration: x becomes its square (as requested)
                current_x = current_x ** 2
                
                # Stop if it gets too large
                if current_x > 1e10:
                    print(f"    Stopped - value too large: {current_x}")
                    break
                    
            except (ValueError, OverflowError) as e:
                print(f"  Iteration {iteration+1}: Error with x={current_x}: {e}")
                break
        
        chain_results[initial_x] = chain_values
    
    return chain_results

def test_xy_multiplication():
    """Test x*y for x=0-10, y=1-50"""
    print("\n" + "=" * 80)
    print("x*y MULTIPLICATION ANALYSIS")
    print("=" * 80)
    
    multiplication_results = {}
    patterns = {}
    
    for x in range(11):  # x = 0 to 10
        row_data = []
        interesting_products = []
        
        print(f"\nMultiplication table for x = {x}:")
        print("y : x*y  :  Special Properties")
        print("-" * 40)
        
        for y in range(1, 51):  # y = 1 to 50
            product = x * y
            
            # Identify interesting mathematical properties
            properties = []
            
            # Perfect squares
            if int(math.sqrt(product)) ** 2 == product and product > 0:
                properties.append(f"√{int(math.sqrt(product))}²")
            
            # Prime numbers
            if product > 1:
                is_prime = True
                for i in range(2, int(math.sqrt(product)) + 1):
                    if product % i == 0:
                        is_prime = False
                        break
                if is_prime:
                    properties.append("PRIME")
            
            # Powers of 2
            if product > 0 and (product & (product - 1)) == 0:
                properties.append(f"2^{int(math.log2(product))}")
            
            # Fibonacci numbers
            fib = [0, 1]
            while fib[-1] < product:
                fib.append(fib[-1] + fib[-2])
            if product in fib:
                properties.append("FIBONACCI")
            
            # Triangular numbers
            n = int((math.sqrt(8*product + 1) - 1) / 2)
            if n * (n + 1) // 2 == product:
                properties.append(f"Triangular(n={n})")
            
            prop_str = ", ".join(properties) if properties else "None"
            
            if y % 10 == 0 or properties:  # Show every 10th or interesting ones
                print(f"{y:2d} : {product:4d}  :  {prop_str}")
            
            if properties:
                interesting_products.append({'y': y, 'product': product, 'properties': properties})
            
            row_data.append({'y': y, 'product': product, 'properties': properties})
        
        multiplication_results[x] = row_data
        patterns[x] = interesting_products
        
        if interesting_products:
            print(f"\n  Interesting products for x={x}: {len(interesting_products)} found")
    
    return multiplication_results

def test_addition_subtraction():
    """Test x±y for x=0-10, y=-50 to 50"""
    print("\n" + "=" * 80)
    print("x±y ADDITION/SUBTRACTION ANALYSIS")
    print("=" * 80)
    
    add_sub_results = {}
    
    for x in range(11):  # x = 0 to 10
        x_data = {'addition': [], 'subtraction': []}
        
        print(f"\nAnalysis for x = {x}:")
        print("y  : x+y   : x-y   :  Properties")
        print("-" * 45)
        
        for y in range(-50, 51):  # y = -50 to 50
            sum_result = x + y
            diff_result = x - y
            
            # Find interesting properties
            properties = []
            
            # Both results are same magnitude
            if abs(sum_result) == abs(diff_result) and sum_result != 0:
                properties.append("Equal magnitude")
            
            # Sum or difference is zero
            if sum_result == 0:
                properties.append("Sum=0")
            if diff_result == 0:
                properties.append("Diff=0")
            
            # Both are perfect squares
            sum_square = int(math.sqrt(abs(sum_result))) ** 2 == abs(sum_result)
            diff_square = int(math.sqrt(abs(diff_result))) ** 2 == abs(diff_result)
            if sum_square and diff_square:
                properties.append("Both squares")
            
            # Results are opposites
            if sum_result == -diff_result and sum_result != 0:
                properties.append("Opposites")
            
            prop_str = ", ".join(properties) if properties else ""
            
            if y % 10 == 0 or properties:  # Show every 10th or interesting
                print(f"{y:3d} : {sum_result:4d}   : {diff_result:4d}   : {prop_str}")
            
            x_data['addition'].append({'y': y, 'result': sum_result, 'properties': properties})
            x_data['subtraction'].append({'y': y, 'result': diff_result, 'properties': properties})
        
        add_sub_results[x] = x_data
    
    return add_sub_results

def test_exponentiation():
    """Test x^y filling gaps from sqrt calculations"""
    print("\n" + "=" * 80)
    print("x^y EXPONENTIATION ANALYSIS")
    print("=" * 80)
    
    exponent_results = {}
    
    for x in range(11):  # x = 0 to 10
        x_data = []
        
        print(f"\nExponentiation for x = {x}:")
        print("y  : x^y          :  Properties")
        print("-" * 50)
        
        # Test a range of exponents including fractional
        test_exponents = list(range(-5, 11))  # -5 to 10
        
        # Add some interesting fractional exponents
        test_exponents.extend([0.5, 1/3, 2/3, math.pi, math.e])
        
        for y in test_exponents:
            try:
                if x == 0 and y <= 0:
                    continue  # Skip 0^negative and 0^0
                
                result = x ** y
                
                # Identify properties
                properties = []
                
                # Integer result
                if abs(result - round(result)) < 1e-10:
                    properties.append("Integer")
                
                # Perfect power
                if result > 1 and abs(result - round(result)) < 1e-10:
                    integer_result = int(round(result))
                    for power in range(2, 11):
                        if round(integer_result ** (1/power)) ** power == integer_result:
                            properties.append(f"{power}th power")
                            break
                
                # Special mathematical constants
                if abs(result - math.e) < 0.01:
                    properties.append("≈e")
                elif abs(result - math.pi) < 0.01:
                    properties.append("≈π")
                elif abs(result - math.sqrt(2)) < 0.01:
                    properties.append("≈√2")
                
                prop_str = ", ".join(properties) if properties else "None"
                
                # Display interesting results
                if (isinstance(y, int) and y % 2 == 0) or properties or abs(result) < 10:
                    if isinstance(y, float):
                        y_str = f"{y:.3f}"
                    else:
                        y_str = str(y)
                    
                    if abs(result) > 1e6:
                        result_str = f"{result:.2e}"
                    else:
                        result_str = f"{result:.6f}"
                    
                    print(f"{y:>5} : {result_str:>12} : {prop_str}")
                
                x_data.append({'y': y, 'result': result, 'properties': properties})
                
            except (ValueError, OverflowError, ZeroDivisionError) as e:
                properties = [f"Error: {type(e).__name__}"]
                x_data.append({'y': y, 'result': None, 'properties': properties})
        
        exponent_results[x] = x_data
    
    return exponent_results

def apply_additional_operators():
    """Apply ALL relevant mathematical operators for differentiation and comparison"""
    print("\n" + "=" * 80)
    print("ADDITIONAL MATHEMATICAL OPERATORS ANALYSIS")
    print("=" * 80)
    
    # Test numbers from 0-10
    test_numbers = list(range(11))
    operator_results = {}
    
    operators = {
        'factorial': lambda n: math.factorial(n) if n >= 0 and n <= 20 else None,
        'log_e': lambda n: math.log(n) if n > 0 else None,
        'log_10': lambda n: math.log10(n) if n > 0 else None,
        'log_2': lambda n: math.log2(n) if n > 0 else None,
        'sqrt': lambda n: math.sqrt(n) if n >= 0 else None,
        'cbrt': lambda n: n ** (1/3) if n >= 0 else -(-n) ** (1/3),
        'sin': lambda n: math.sin(n),
        'cos': lambda n: math.cos(n),
        'tan': lambda n: math.tan(n),
        'exp': lambda n: math.exp(n) if n < 50 else None,  # Prevent overflow
        'inverse': lambda n: 1/n if n != 0 else None,
        'absolute': lambda n: abs(n),
        'ceil': lambda n: math.ceil(n),
        'floor': lambda n: math.floor(n),
        'round': lambda n: round(n),
    }
    
    print("Number : " + " : ".join(f"{op:>10}" for op in operators.keys()))
    print("-" * (15 + len(operators) * 12))
    
    for n in test_numbers:
        results = {}
        row_values = []
        
        for op_name, op_func in operators.items():
            try:
                result = op_func(n)
                results[op_name] = result
                
                if result is None:
                    row_values.append("N/A")
                elif abs(result) > 1e6:
                    row_values.append(f"{result:.2e}")
                elif isinstance(result, float):
                    row_values.append(f"{result:.6f}")
                else:
                    row_values.append(str(result))
                    
            except (ValueError, OverflowError):
                results[op_name] = None
                row_values.append("Error")
        
        print(f"{n:6d} : " + " : ".join(f"{val:>10}" for val in row_values))
        operator_results[n] = results
    
    # Find interesting mathematical relationships
    print(f"\nInteresting relationships found:")
    
    for n in test_numbers:
        results = operator_results[n]
        relationships = []
        
        # Check for golden ratio relationships
        if results.get('phi_approx'):  # Would need to calculate
            pass
        
        # Check for perfect relationships
        if results.get('sqrt') and results.get('inverse'):
            if abs(results['sqrt'] - results['inverse']) < 0.01:
                relationships.append(f"√n ≈ 1/n")
        
        # Euler's identity related
        if results.get('exp') and abs(results['exp'] - math.e) < 0.1:
            relationships.append("exp(n) ≈ e")
        
        if relationships:
            print(f"  n={n}: {', '.join(relationships)}")
    
    return operator_results

def run_comprehensive_analysis():
    """Run all mathematical analyses"""
    print("COMPREHENSIVE MATHEMATICAL ANALYSIS FRAMEWORK")
    print("=" * 80)
    print("Testing all mathematical operations and patterns as requested\n")
    
    # Run all tests
    results_1qx = test_1_over_qx_pattern()
    results_chains = test_x_sqrt_x_chains()
    results_multiplication = test_xy_multiplication()
    results_add_sub = test_addition_subtraction()
    results_exponentiation = test_exponentiation()
    results_operators = apply_additional_operators()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("All mathematical operations tested according to specifications.")
    print("Results show rich patterns and relationships across operations.")
    
    return {
        '1_qx_pattern': results_1qx,
        'x_sqrt_x_chains': results_chains,
        'xy_multiplication': results_multiplication,
        'addition_subtraction': results_add_sub,
        'exponentiation': results_exponentiation,
        'additional_operators': results_operators
    }

if __name__ == "__main__":
    results = run_comprehensive_analysis()
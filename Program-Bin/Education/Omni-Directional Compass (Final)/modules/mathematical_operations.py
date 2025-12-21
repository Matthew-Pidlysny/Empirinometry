"""
Mathematical Operations Module
Advanced mathematical calculations and analysis
"""

import math
import numpy as np
import sympy as sp
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Union, Tuple
import itertools

class MathematicalOperations:
    def __init__(self, parent_compass):
        self.compass = parent_compass
        
    def main_menu(self) -> bool:
        """Mathematical operations menu"""
        print("\n" + "=" * 80)
        print("MATHEMATICAL OPERATIONS")
        print("=" * 80)
        print()
        print("Choose your mathematical domain:")
        print()
        print("1. ALGEBRA")
        print("   Equations, polynomials, and symbolic manipulation")
        print()
        print("2. CALCULUS")
        print("   Derivatives, integrals, and analysis")
        print()
        print("3. LINEAR ALGEBRA")
        print("   Matrices, vectors, and eigenproblems")
        print()
        print("4. NUMBER THEORY")
        print("   Primes, modular arithmetic, and number patterns")
        print()
        print("5. STATISTICS & PROBABILITY")
        print("   Data analysis and probability distributions")
        print()
        print("6. DISCRETE MATHEMATICS")
        print("   Combinatorics, graph theory, and logic")
        print()
        print("7. ADVANCED ANALYSIS")
        print("   Series, transforms, and special functions")
        print()
        print("8. RETURN TO MAIN MENU")
        print()
        
        while True:
            choice = input("Enter your choice (1-8): ").strip()
            if choice in [str(i) for i in range(1, 9)]:
                return choice
            print("Invalid choice. Please enter 1-8.")
    
    def run(self):
        """Main mathematical operations interface"""
        while True:
            choice = self.main_menu()
            
            if choice == '1':
                self.algebra_operations()
            elif choice == '2':
                self.calculus_operations()
            elif choice == '3':
                self.linear_algebra_operations()
            elif choice == '4':
                self.number_theory_operations()
            elif choice == '5':
                self.statistics_operations()
            elif choice == '6':
                self.discrete_math_operations()
            elif choice == '7':
                self.advanced_analysis()
            elif choice == '8':
                break
            
            input("\nPress Enter to continue...")
            print("\n" * 2)
    
    def algebra_operations(self):
        """Algebraic operations"""
        print("\n" + "=" * 80)
        print("ALGEBRA OPERATIONS")
        print("=" * 80)
        print()
        
        print("1. Solve equations")
        print("2. Factor polynomials")
        print("3. Simplify expressions")
        print("4. Work with inequalities")
        print("5. System of equations")
        
        choice = input("Choose operation (1-5): ").strip()
        
        if choice == '1':
            self.solve_equations()
        elif choice == '2':
            self.factor_polynomials()
        elif choice == '3':
            self.simplify_expressions()
        elif choice == '4':
            self.solve_inequalities()
        elif choice == '5':
            self.system_of_equations()
    
    def solve_equations(self):
        """Solve various types of equations"""
        print("\n--- Equation Solver ---")
        
        equation = input("Enter equation (e.g., x^2 + 2*x - 3 = 0): ").strip()
        
        try:
            # Parse the equation
            if '=' in equation:
                left, right = equation.split('=')
                expr = sp.sympify(left.strip()) - sp.sympify(right.strip())
            else:
                expr = sp.sympify(equation)
            
            # Identify variable
            variables = list(expr.free_symbols)
            if not variables:
                x = sp.Symbol('x')
            else:
                x = variables[0]
            
            # Solve equation
            solutions = sp.solve(expr, x)
            
            print(f"\nSolutions for {equation}:")
            for i, sol in enumerate(solutions, 1):
                print(f"  {i}. {sol}")
                
                # Provide numerical approximation
                if sol.is_real:
                    print(f"     Numerical: {float(sol):.10f}")
                else:
                    print(f"     Complex: {complex(sol.evalf())}")
            
        except Exception as e:
            print(f"Error solving equation: {e}")
    
    def calculus_operations(self):
        """Calculus operations"""
        print("\n" + "=" * 80)
        print("CALCULUS OPERATIONS")
        print("=" * 80)
        print()
        
        print("1. Derivatives")
        print("2. Integrals")
        print("3. Limits")
        print("4. Series expansion")
        print("5. Differential equations")
        
        choice = input("Choose operation (1-5): ").strip()
        
        if choice == '1':
            self.derivatives()
        elif choice == '2':
            self.integrals()
        elif choice == '3':
            self.limits()
        elif choice == '4':
            self.series_expansion()
        elif choice == '5':
            self.differential_equations()
    
    def derivatives(self):
        """Calculate derivatives"""
        print("\n--- Derivative Calculator ---")
        
        expression = input("Enter expression (e.g., x^2 + sin(x)): ").strip()
        var = input("Variable (default x): ").strip() or 'x'
        order = input("Order of derivative (default 1): ").strip() or '1'
        
        try:
            x = sp.Symbol(var)
            expr = sp.sympify(expression)
            order = int(order)
            
            # Calculate derivative
            derivative = sp.diff(expr, x, order)
            
            print(f"\nExpression: {expr}")
            print(f"D^{order}/d{var}^{order}: {derivative}")
            
            # Simplify result
            simplified = sp.simplify(derivative)
            if simplified != derivative:
                print(f"Simplified: {simplified}")
            
            # Provide derivative at specific point
            point = input("Evaluate at point (e.g., x=1, press Enter to skip): ").strip()
            if point and '=' in point:
                var_name, value = point.split('=')
                value = float(value.strip())
                result = derivative.subs(x, value)
                print(f"f^{order}({var_name}={value}) = {result}")
                print(f"Numerical: {float(result):.10f}")
            
        except Exception as e:
            print(f"Error calculating derivative: {e}")
    
    def linear_algebra_operations(self):
        """Linear algebra operations"""
        print("\n" + "=" * 80)
        print("LINEAR ALGEBRA OPERATIONS")
        print("=" * 80)
        print()
        
        print("1. Matrix operations")
        print("2. Eigenvalues and eigenvectors")
        print("3. Vector operations")
        print("4. System of linear equations")
        print("5. Matrix decompositions")
        
        choice = input("Choose operation (1-5): ").strip()
        
        if choice == '1':
            self.matrix_operations()
        elif choice == '2':
            self.eigen_analysis()
        elif choice == '3':
            self.vector_operations()
        elif choice == '4':
            self.linear_systems()
        elif choice == '5':
            self.matrix_decompositions()
    
    def matrix_operations(self):
        """Matrix operations"""
        print("\n--- Matrix Operations ---")
        
        print("Enter first matrix (rows separated by ;, elements separated by spaces):")
        print("Example: 1 2 3; 4 5 6; 7 8 9")
        
        try:
            matrix1_input = input("Matrix 1: ").strip()
            matrix1 = sp.Matrix([[float(x) for x in row.split()] 
                                for row in matrix1_input.split(';')])
            
            operation = input("Operation (+, -, *, /, ^-1, T): ").strip()
            
            result = None
            op_name = ""
            
            if operation == '+':
                matrix2_input = input("Matrix 2: ").strip()
                matrix2 = sp.Matrix([[float(x) for x in row.split()] 
                                    for row in matrix2_input.split(';')])
                result = matrix1 + matrix2
                op_name = "Addition"
            elif operation == '-':
                matrix2_input = input("Matrix 2: ").strip()
                matrix2 = sp.Matrix([[float(x) for x in row.split()] 
                                    for row in matrix2_input.split(';')])
                result = matrix1 - matrix2
                op_name = "Subtraction"
            elif operation == '*':
                matrix2_input = input("Matrix 2: ").strip()
                matrix2 = sp.Matrix([[float(x) for x in row.split()] 
                                    for row in matrix2_input.split(';')])
                result = matrix1 * matrix2
                op_name = "Multiplication"
            elif operation == '/':
                matrix2_input = input("Matrix 2: ").strip()
                matrix2 = sp.Matrix([[float(x) for x in row.split()] 
                                    for row in matrix2_input.split(';')])
                result = matrix1 * matrix2.inv()
                op_name = "Division"
            elif operation == '^-1' or operation == 'inv':
                result = matrix1.inv()
                op_name = "Inverse"
            elif operation == 'T':
                result = matrix1.T
                op_name = "Transpose"
            
            if result is not None:
                print(f"\n{op_name} Result:")
                print(result)
                
                # Matrix properties
                if hasattr(result, 'det'):
                    try:
                        det = result.det()
                        print(f"Determinant: {det}")
                    except:
                        pass
                
                if hasattr(result, 'rank'):
                    rank = result.rank()
                    print(f"Rank: {rank}")
            
        except Exception as e:
            print(f"Error in matrix operation: {e}")
    
    def number_theory_operations(self):
        """Number theory operations"""
        print("\n" + "=" * 80)
        print("NUMBER THEORY OPERATIONS")
        print("=" * 80)
        print()
        
        print("1. Prime factorization")
        print("2. GCD and LCM")
        print("3. Modular arithmetic")
        print("4. Euler's totient function")
        print("5. Chinese remainder theorem")
        print("6. Quadratic residues")
        
        choice = input("Choose operation (1-6): ").strip()
        
        if choice == '1':
            self.prime_factorization()
        elif choice == '2':
            self.gcd_lcm()
        elif choice == '3':
            self.modular_arithmetic()
        elif choice == '4':
            self.euler_totient()
        elif choice == '5':
            self.chinese_remainder()
        elif choice == '6':
            self.quadratic_residues()
    
    def prime_factorization(self):
        """Prime factorization"""
        print("\n--- Prime Factorization ---")
        
        try:
            number = int(input("Enter positive integer: "))
            if number < 2:
                print("Number must be >= 2")
                return
            
            # Check if prime
            if sp.isprime(number):
                print(f"{number} is prime!")
                return
            
            # Factorize
            factors = sp.factorint(number)
            
            print(f"Prime factorization of {number}:")
            factor_str = " × ".join([f"{p}^{e}" if e > 1 else str(p) 
                                   for p, e in sorted(factors.items())])
            print(f"  {factor_str}")
            
            # Additional information
            total_factors = sum(e for e in factors.values())
            distinct_factors = len(factors)
            
            print(f"Total prime factors (with multiplicity): {total_factors}")
            print(f"Distinct prime factors: {distinct_factors}")
            
            # Divisor count
            divisor_count = 1
            for exp in factors.values():
                divisor_count *= (exp + 1)
            print(f"Number of divisors: {divisor_count}")
            
            # Sum of divisors
            sigma = sum(int(p**e) for p, e in sp.divisors(number, generator=True))
            print(f"Sum of divisors σ({number}): {sigma}")
            
        except ValueError:
            print("Please enter a valid integer")
        except Exception as e:
            print(f"Error: {e}")
    
    def statistics_operations(self):
        """Statistics and probability operations"""
        print("\n" + "=" * 80)
        print("STATISTICS & PROBABILITY")
        print("=" * 80)
        print()
        
        print("1. Descriptive statistics")
        print("2. Probability distributions")
        print("3. Hypothesis testing")
        print("4. Correlation analysis")
        print("5. Regression analysis")
        
        choice = input("Choose operation (1-5): ").strip()
        
        if choice == '1':
            self.descriptive_statistics()
        elif choice == '2':
            self.probability_distributions()
        elif choice == '3':
            self.hypothesis_testing()
        elif choice == '4':
            self.correlation_analysis()
        elif choice == '5':
            self.regression_analysis()
    
    def descriptive_statistics(self):
        """Descriptive statistics"""
        print("\n--- Descriptive Statistics ---")
        
        print("Enter data (numbers separated by spaces or commas):")
        data_input = input("Data: ").strip()
        
        try:
            # Parse data
            data = [float(x.strip()) for x in data_input.replace(',', ' ').split()]
            
            if not data:
                print("No data entered")
                return
            
            # Calculate statistics
            n = len(data)
            mean = np.mean(data)
            median = np.median(data)
            mode_results = sp.multimode(data) if len(set(data)) < len(data) else []
            variance = np.var(data, ddof=1)  # Sample variance
            std_dev = np.std(data, ddof=1)  # Sample standard deviation
            data_range = max(data) - min(data)
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            skewness = sp.skew(data)
            kurtosis = sp.kurtosis(data)
            
            print(f"\n--- Statistical Summary for {n} data points ---")
            print(f"Mean: {mean:.6f}")
            print(f"Median: {median:.6f}")
            if mode_results:
                print(f"Mode(s): {[f'{m:.6f}' for m in mode_results]}")
            print(f"Variance: {variance:.6f}")
            print(f"Standard Deviation: {std_dev:.6f}")
            print(f"Range: {data_range:.6f}")
            print(f"Minimum: {min(data):.6f}")
            print(f"Maximum: {max(data):.6f}")
            print(f"Q1 (25th percentile): {q1:.6f}")
            print(f"Q3 (75th percentile): {q3:.6f}")
            print(f"IQR: {iqr:.6f}")
            print(f"Skewness: {skewness:.6f}")
            print(f"Kurtosis: {kurtosis:.6f}")
            
            # Outlier detection (IQR method)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [x for x in data if x < lower_bound or x > upper_bound]
            
            if outliers:
                print(f"\nOutliers detected: {len(outliers)}")
                print(f"Outlier values: {[f'{x:.6f}' for x in outliers]}")
                print(f"Outlier bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")
            else:
                print("\nNo outliers detected (IQR method)")
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
    
    def advanced_analysis(self):
        """Advanced mathematical analysis"""
        print("\n" + "=" * 80)
        print("ADVANCED ANALYSIS")
        print("=" * 80)
        print()
        
        print("1. Fourier series")
        print("2. Laplace transforms")
        print("3. Special functions")
        print("4. Complex analysis")
        print("5. Series summation")
        
        choice = input("Choose operation (1-5): ").strip()
        
        if choice == '1':
            self.fourier_series()
        elif choice == '2':
            self.laplace_transforms()
        elif choice == '3':
            self.special_functions()
        elif choice == '4':
            self.complex_analysis()
        elif choice == '5':
            self.series_summation()
    
    def series_summation(self):
        """Series summation"""
        print("\n--- Series Summation ---")
        
        print("Examples of common series:")
        print("- Arithmetic: sum([a + (n-1)*d for n in range(1, N+1)])")
        print("- Geometric: a * (1 - r^N) / (1 - r)")
        print("- Harmonic: sum([1/n for n in range(1, N+1)])")
        print("- Power series: sum([n**p for n in range(1, N+1)])")
        
        series_type = input("Series type (arithmetic/geometric/harmonic/power/custom): ").strip().lower()
        
        try:
            if series_type == 'arithmetic':
                a = float(input("First term (a): "))
                d = float(input("Common difference (d): "))
                N = int(input("Number of terms (N): "))
                
                series = [a + (n-1)*d for n in range(1, N+1)]
                formula = N/2 * (2*a + (N-1)*d)
                
            elif series_type == 'geometric':
                a = float(input("First term (a): "))
                r = float(input("Common ratio (r): "))
                N = int(input("Number of terms (N): "))
                
                series = [a * r**(n-1) for n in range(1, N+1)]
                if abs(r) != 1:
                    formula = a * (1 - r**N) / (1 - r)
                else:
                    formula = a * N
                    
            elif series_type == 'harmonic':
                N = int(input("Number of terms (N): "))
                series = [1/n for n in range(1, N+1)]
                formula = None  # No closed form
                
            elif series_type == 'power':
                p = float(input("Power (p): "))
                N = int(input("Number of terms (N): "))
                series = [n**p for n in range(1, N+1)]
                formula = None  # General case no simple closed form
                
            else:
                print("Custom series not implemented")
                return
            
            if series:
                print(f"\n--- Results ---")
                print(f"Series (first 5 terms): {[f'{x:.6f}' for x in series[:5]]}")
                print(f"Sum (numerical): {sum(series):.10f}")
                
                if formula is not None:
                    print(f"Sum (formula): {formula:.10f}")
                    
                    # Check accuracy
                    numerical = sum(series)
                    error = abs(numerical - formula)
                    print(f"Difference: {error:.2e}")
                
                # Convergence test for infinite series
                if series_type == 'geometric' and abs(r) < 1:
                    infinite_sum = a / (1 - r)
                    print(f"\nInfinite series sum (|r| < 1): {infinite_sum:.10f}")
                    
                elif series_type == 'harmonic':
                    print(f"\nHarmonic series diverges (grows like ln(N) + γ)")
                    print(f"ln({N}) + γ ≈ {math.log(N) + 0.5772156649:.10f}")
                    print(f"H_{N} = {sum(series):.10f}")
                
        except Exception as e:
            print(f"Error in series calculation: {e}")
    
    # Placeholder methods for other operations
    def factor_polynomials(self):
        pass
    
    def simplify_expressions(self):
        pass
    
    def solve_inequalities(self):
        pass
    
    def system_of_equations(self):
        pass
    
    def integrals(self):
        pass
    
    def limits(self):
        pass
    
    def series_expansion(self):
        pass
    
    def differential_equations(self):
        pass
    
    def eigen_analysis(self):
        pass
    
    def vector_operations(self):
        pass
    
    def linear_systems(self):
        pass
    
    def matrix_decompositions(self):
        pass
    
    def gcd_lcm(self):
        pass
    
    def modular_arithmetic(self):
        pass
    
    def euler_totient(self):
        pass
    
    def chinese_remainder(self):
        pass
    
    def quadratic_residues(self):
        pass
    
    def probability_distributions(self):
        pass
    
    def hypothesis_testing(self):
        pass
    
    def correlation_analysis(self):
        pass
    
    def regression_analysis(self):
        pass
    
    def discrete_math_operations(self):
        pass
    
    def fourier_series(self):
        pass
    
    def laplace_transforms(self):
        pass
    
    def special_functions(self):
        pass
    
    def complex_analysis(self):
        pass
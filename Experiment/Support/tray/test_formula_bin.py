#!/usr/bin/env python3
"""
Testing script for Formula-Bin mathematical claims
"""

import math
import cmath
import sys
from decimal import Decimal, getcontext

def test_hyperbolic_index():
    """Test the Hyperbolic Index formula"""
    print("=== Testing Hyperbolic Index Formula ===")
    
    def hyperbolic_index(a):
        x = (a / 13) * 1000 / 13
        I = round(x * 10) / 10  # First decimal rounded
        P = int(x)  # Whole number before decimal
        return x, I, P
    
    test_values = [1, 2, 3, 13, 26, 124]
    for a in test_values:
        x, I, P = hyperbolic_index(a)
        print(f"a={a}: x={x}, I={I}, P={P}")

def test_exponent_buster():
    """Test the Exponent Buster formula"""
    print("\n=== Testing Exponent Buster Formula ===")
    
    def exponent_buster(x):
        a = (x / 13) * 1000 / 13
        d_x = x**2 - a
        return a, d_x
    
    test_values = [0, 1, 2, 3, 4, 5]
    for x in test_values:
        a, d_x = exponent_buster(x)
        print(f"x={x}: a={a}, d(x)={d_x}")

def test_c_star_generation():
    """Test C* constant generation attempt"""
    print("\n=== Testing C* Generation Formula ===")
    
    # From the repository: C* = 0.894751918154916971057500594108604132047819675762633907162342311645898329109485858045137356324418883918704234805309277739768448577521582363947287845...
    c_star_repo = Decimal('0.894751918154916971057500594108604132047819675762633907162342311645898329109485858045137356324418883918704234805309277739768448577521582363947287845')
    
    print(f"Repository C* value: {c_star_repo}")
    
    # Test the optimization formula components
    def F_function(i, phi=(1+5**0.5)/2):
        return (2*i)**(1/phi) * phi**(phi/(2*math.pi)) * math.e**(math.pi*i/(4*i))
    
    for i in [1, 2, 3, 5, 10]:
        f_val = F_function(i)
        print(f"F({i}) = {f_val}")

def test_riemann_generation():
    """Test Riemann zero generation formula"""
    print("\n=== Testing Riemann Zero Generation ===")
    
    # Repository values
    c_star = 0.894751918154916971057500594108604132047819675762633907162342311645898329109485858045137356324418883918704234805309277739768448577521582363947287845
    alpha = 1.0
    
    # Initial generation formula
    gamma_1 = c_star + 2 * math.pi * (math.log(c_star + alpha) / (math.log(c_star)**2))
    print(f"Generated γ₁ = {gamma_1}")
    print(f"Actual first non-trivial zero = 14.134725141734693790457251983562470270784257115699243175685567460149...")
    
    # Check accuracy
    actual_gamma_1 = 14.134725141734693790457251983562470270784257115699243175685567460149
    error = abs(gamma_1 - actual_gamma_1)
    print(f"Error: {error}")
    print(f"Relative error: {error/actual_gamma_1:.6%}")

def test_circle_quadratic():
    """Test Circle Focus Quadratic Equation"""
    print("\n=== Testing Circle Focus Quadratic ===")
    
    def circle_quadratic(x, b):
        z_squared = (x + b) - x**2
        if z_squared >= 0:
            z = math.sqrt(z_squared)
            return z
        else:
            return complex(0, math.sqrt(-z_squared))
    
    test_cases = [(1, 1), (2, 1), (0.5, 2)]
    for x, b in test_cases:
        z = circle_quadratic(x, b)
        print(f"x={x}, b={b}: z={z}")

def test_bondz_formula():
    """Test the Bondz electrical formula"""
    print("\n=== Testing Bondz Electrical Formula ===")
    
    def bondz_circuit(E, I, R):
        Q = E + I + R
        # W = Q * R / I^I (this needs clarification - I^I is I to the power of I)
        try:
            W = Q * R / (I ** I) if I != 0 else float('inf')
        except:
            W = float('inf')
        return Q, W
    
    # Test with sample values
    test_cases = [(12, 2, 6), (5, 1, 5), (24, 3, 8)]
    for E, I, R in test_cases:
        Q, W = bondz_circuit(E, I, R)
        print(f"E={E}V, I={I}A, R={R}Ω: Q={Q}, W={W}")

def test_lesson_problems():
    """Test problems from the lesson files"""
    print("\n=== Testing Lesson Problems ===")
    
    # Lesson 001 problem: ((28561 - 2197) / k) / x / x = x where k = x - 1
    def lesson_001_problem():
        solutions = []
        for x in range(1, 100):
            k = x - 1
            if k != 0:
                result = ((28561 - 2197) / k) / x / x
                if abs(result - x) < 0.001:  # Allow small floating point error
                    solutions.append(x)
        return solutions
    
    solutions = lesson_001_problem()
    print(f"Lesson 001 problem solutions: {solutions}")
    
    # Lesson 003 problem 1: (125 - x²) / 4 / 5 = x
    def lesson_003_1():
        solutions = []
        for x in range(-50, 51):
            if (125 - x**2) / 4 / 5 == x:
                solutions.append(x)
        return solutions
    
    solutions_1 = lesson_003_1()
    print(f"Lesson 003 problem 1 solutions: {solutions_1}")

if __name__ == "__main__":
    # Set high precision for decimal calculations
    getcontext().prec = 50
    
    print("Testing Formula-Bin Mathematical Claims")
    print("=" * 50)
    
    test_hyperbolic_index()
    test_exponent_buster()
    test_c_star_generation()
    test_riemann_generation()
    test_circle_quadratic()
    test_bondz_formula()
    test_lesson_problems()
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
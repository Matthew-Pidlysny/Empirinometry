#!/usr/bin/env python3
"""
Advanced testing of complex Formula-Bin claims
"""

import math
from decimal import Decimal, getcontext

def test_pi_sphere_mapping():
    """Test the Pi-Sphere mapping formula"""
    print("=== Testing Pi-Sphere Mapping ===")
    
    # First few digits of pi
    pi_digits = "3141592653"
    phi = (1 + 5**0.5) / 2
    
    def map_pi_digit_to_sphere(index, digit):
        theta = 2 * math.pi * index / phi
        phi_angle = math.pi * int(digit) / 9
        x = math.sin(phi_angle) * math.cos(theta)
        y = math.sin(phi_angle) * math.sin(theta)
        z = math.cos(phi_angle)
        color_class = int(digit) % 7
        return x, y, z, theta, phi_angle, color_class
    
    print("Index, Digit, X, Y, Z, ColorClass")
    for i, digit in enumerate(pi_digits[:10]):
        x, y, z, theta, phi_angle, color_class = map_pi_digit_to_sphere(i, digit)
        print(f"{i}, {digit}, {x:.3f}, {y:.3f}, {z:.3f}, {color_class}")

def test_universal_varia():
    """Test the Universal Varia formula"""
    print("\n=== Testing Universal Varia Formula ===")
    
    # Simplified version: (x * y)² # D * 5 / 0.33 # Σ x + 66⁷⁷ + x² - y⁷ # Q₈ * (K * 0.412) = √R = z
    # This needs interpretation of custom operations # and Σ
    
    def universal_varia_simplified(x, y):
        # Main computation without custom operations
        main_part = (x * y)**2 * 5 / 0.33
        second_part = x + (66**77) + x**2 - y**7
        # Simplified: combine parts
        R = main_part + second_part
        z = math.sqrt(R) if R >= 0 else complex(0, math.sqrt(-R))
        return z
    
    # Test with small values
    test_cases = [(1, 1), (2, 3), (0.1, 0.2)]
    for x, y in test_cases:
        try:
            z = universal_varia_simplified(x, y)
            print(f"x={x}, y={y}: z≈{z}")
        except OverflowError:
            print(f"x={x}, y={y}: Overflow (66^77 is too large)")

def test_l_induction():
    """Test the L-Induction formula"""
    print("\n=== Testing L-Induction Formula ===")
    
    def l_induction(L):
        # L * (L / L * .66)^L + L(L^L) - (L^(L^-L)/L*L+L^4) = P
        try:
            part1 = L * (L / L * 0.66)**L
            part2 = L * (L**L)
            part3 = (L**(L**(-L))) / (L * L) + L**4
            P = part1 + part2 - part3
            return P
        except:
            return float('inf')
    
    for L in range(1, 6):
        P = l_induction(L)
        print(f"L={L}: P={P}")

def test_partition_speed():
    """Test Partition Speed formula"""
    print("\n=== Testing Partition Speed Formula ===")
    
    def partition_speed(p_max, p_area, delta_v):
        c = 299792458  # Speed of light in m/s
        p_1000 = 1000 / 169
        
        p_speed = (p_max * p_1000**4) / c * delta_v
        return p_speed
    
    # Test with sample values
    test_cases = [
        (100, 0.001, 10),      # Small values
        (1000, 0.01, 100),     # Medium values
        (10000, 0.1, 1000),    # Large values
    ]
    
    for p_max, p_area, delta_v in test_cases:
        ps = partition_speed(p_max, p_area, delta_v)
        print(f"p_max={p_max}, p_area={p_area}, Δv={delta_v}: p_sp={ps} m/s")

def test_steal_this_formula():
    """Test 'Steal This Formula'"""
    print("\n=== Testing 'Steal This Formula' ===")
    
    def steal_this_formula(x):
        L_H = x * 1000 * 52 * 51
        return L_H
    
    test_values = [1, 2, 3.14159, 0.001, 100]
    for x in test_values:
        L_H = steal_this_formula(x)
        print(f"x={x}: Lᴴ={L_H}")

def test_number_termination():
    """Test Number Termination formulas"""
    print("\n=== Testing Number Termination Theory ===")
    
    # Cognitive Termination: 15 digits
    # Planck Scale Termination: 35 digits
    # Quantum Measurement Termination: 61 digits
    
    def test_termination_concept(number, termination_digits):
        num_str = str(number)
        if len(num_str) > termination_digits:
            truncated = num_str[:termination_digits] + "..."
            return truncated
        return num_str
    
    # Test with a long number
    test_number = "3.14159265358979323846264338327950288419716939937510"
    
    terminations = [
        ("Cognitive", 15),
        ("Planck Scale", 35),
        ("Quantum Measurement", 61)
    ]
    
    for name, digits in terminations:
        result = test_termination_concept(test_number, digits)
        print(f"{name} ({digits} digits): {result}")

def test_riemann_forward_generation():
    """Test forward generation of Riemann zeros"""
    print("\n=== Testing Riemann Forward Generation ===")
    
    # Using repository formula but with correction
    c_star = 0.894751918154916971057500594108604132047819675762633907162342311645898329109485858045137356324418883918704234805309277739768448577521582363947287845
    
    def gamma_forward(n, gamma_prev):
        epsilon = 0.001  # Small stabilization function
        try:
            gamma_next = gamma_prev + 2 * math.pi * (math.log(gamma_prev + 1) / (math.log(gamma_prev)**2) + epsilon)
            return gamma_next
        except:
            return gamma_prev
    
    # Try to generate from actual first zero
    actual_gamma_1 = 14.134725141734693790457251983562470270784257115699243175685567460149
    gamma_2 = gamma_forward(2, actual_gamma_1)
    gamma_3 = gamma_forward(3, gamma_2)
    
    print(f"γ₁ (actual): {actual_gamma_1}")
    print(f"γ₂ (generated): {gamma_2}")
    print(f"γ₃ (generated): {gamma_3}")
    
    # Compare with actual values
    actual_gamma_2 = 21.022039638771554992628479593896902777334340524902
    actual_gamma_3 = 25.010857580145688763213790992567820418860894966831
    
    error_2 = abs(gamma_2 - actual_gamma_2)
    error_3 = abs(gamma_3 - actual_gamma_3)
    
    print(f"Error for γ₂: {error_2}")
    print(f"Error for γ₃: {error_3}")

def test_cyclic_number_properties():
    """Test properties of 142857 cyclic number"""
    print("\n=== Testing 142857 Cyclic Number Properties ===")
    
    cyclic = 142857
    
    print("Multiplication table for 142857:")
    for i in range(1, 7):
        result = cyclic * i
        print(f"142857 × {i} = {result}")
    
    print("\nCyclic rotations:")
    s = "142857"
    for i in range(6):
        rotation = s[i:] + s[:i]
        print(f"Rotation {i+1}: {rotation}")
    
    print("\nRecurring decimal relationships:")
    for i in range(1, 7):
        decimal = 1 / i
        if decimal != int(decimal):
            print(f"1/{i} = {decimal:.6f}...")

if __name__ == "__main__":
    print("Advanced Testing of Formula-Bin Claims")
    print("=" * 50)
    
    test_pi_sphere_mapping()
    test_universal_varia()
    test_l_induction()
    test_partition_speed()
    test_steal_this_formula()
    test_number_termination()
    test_riemann_forward_generation()
    test_cyclic_number_properties()
    
    print("\n" + "=" * 50)
    print("Advanced Testing Complete!")
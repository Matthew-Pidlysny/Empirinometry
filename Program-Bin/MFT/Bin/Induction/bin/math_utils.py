#!/usr/bin/env python3
"""
Mathematical Utilities: Core functions for numerical analysis
Essential tools for induction and variation analysis
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Generator, Set
from functools import lru_cache
import itertools

class NumberTheory:
    """Essential number theory functions"""
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def is_prime(n: int) -> bool:
        """Efficient primality test with caching"""
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        if n < 9:
            return True
        if n % 5 == 0:
            return False
        
        max_divisor = int(math.sqrt(n)) + 1
        for i in range(5, max_divisor, 6):
            if n % i == 0 or n % (i + 2) == 0:
                return False
        return True
    
    @staticmethod
    def prime_sieve(limit: int) -> List[int]:
        """Sieve of Eratosthenes - efficient prime generation"""
        if limit < 2:
            return []
        
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0:2] = b'\x00\x00'
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = b'\x00' * ((limit - i*i)//i + 1)
        
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    @staticmethod
    def prime_factors(n: int) -> List[int]:
        """Return prime factorization of n"""
        if n < 2:
            return []
        
        factors = []
        
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        
        f = 3
        max_factor = math.sqrt(n) + 1
        while f <= max_factor:
            while n % f == 0:
                factors.append(f)
                n //= f
                max_factor = math.sqrt(n) + 1
            f += 2
        
        if n > 1:
            factors.append(n)
        
        return factors
    
    @staticmethod
    def divisor_function(n: int) -> int:
        """Sum of divisors function σ(n)"""
        if n < 1:
            return 0
        
        total = 1
        factors = NumberTheory.prime_factors(n)
        
        for prime in set(factors):
            exponent = factors.count(prime)
            total *= (prime**(exponent + 1) - 1) // (prime - 1)
        
        return total
    
    @staticmethod
    def euler_totient(n: int) -> int:
        """Euler's totient function φ(n) - count of integers ≤ n coprime to n"""
        if n < 1:
            return 0
        
        result = n
        factors = set(NumberTheory.prime_factors(n))
        
        for p in factors:
            result -= result // p
        
        return result
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Greatest common divisor using Euclidean algorithm"""
        while b:
            a, b = b, a % b
        return abs(a)
    
    @staticmethod
    def lcm(a: int, b: int) -> int:
        """Least common multiple"""
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // NumberTheory.gcd(a, b)

class SequenceAnalysis:
    """Tools for analyzing mathematical sequences"""
    
    @staticmethod
    def periodicity(sequence: List[int], max_period: int = None) -> Tuple[int, float]:
        """
        Find the period of a sequence and confidence
        
        Returns: (period, confidence)
        """
        if max_period is None:
            max_period = len(sequence) // 2
        
        if len(sequence) < 4:
            return 1, 0.0
        
        best_period = 1
        best_confidence = 0.0
        
        for period in range(1, min(max_period, len(sequence) // 2) + 1):
            matches = 0
            total = 0
            
            for i in range(len(sequence) - period):
                if sequence[i] == sequence[i + period]:
                    matches += 1
                total += 1
            
            if total > 0:
                confidence = matches / total
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_period = period
        
        return best_period, best_confidence
    
    @staticmethod
    def growth_rate(sequence: List[int]) -> Tuple[float, str]:
        """
        Determine growth rate of sequence
        
        Returns: (rate, type_description)
        """
        if len(sequence) < 3:
            return 0.0, "unknown"
        
        ratios = []
        for i in range(1, len(sequence)):
            if sequence[i-1] != 0 and sequence[i] != 0:
                ratios.append(abs(sequence[i] / sequence[i-1]))
        
        if not ratios:
            return 0.0, "unknown"
        
        avg_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        if std_ratio < 0.1:
            if abs(avg_ratio - 1.618) < 0.1:
                return avg_ratio, "fibonacci-like"
            elif abs(avg_ratio - 2.0) < 0.1:
                return avg_ratio, "exponential_base_2"
            elif abs(avg_ratio - 1.0) < 0.1:
                return avg_ratio, "linear"
            else:
                return avg_ratio, f"exponential_base_{avg_ratio:.2f}"
        elif std_ratio < 1.0:
            return avg_ratio, "polynomial"
        else:
            return avg_ratio, "irregular"

class ModularArithmetic:
    """Modular arithmetic utilities"""
    
    @staticmethod
    def modular_pow(base: int, exponent: int, modulus: int) -> int:
        """Modular exponentiation using binary exponentiation"""
        if modulus == 1:
            return 0
        
        result = 1
        base = base % modulus
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = (result * base) % modulus
            exponent = exponent >> 1
            base = (base * base) % modulus
        
        return result
    
    @staticmethod
    def modular_inverse(a: int, m: int) -> Optional[int]:
        """Find modular inverse using extended Euclidean algorithm"""
        if m <= 1:
            return None
        
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            else:
                g, y, x = extended_gcd(b % a, a)
                return g, x - (b // a) * y, y
        
        g, x, y = extended_gcd(a, m)
        
        if g != 1:
            return None
        else:
            return x % m
    
    @staticmethod
    def chinese_remainder_theorem(congruences: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Solve system of congruences using Chinese Remainder Theorem
        
        Args:
            congruences: List of (remainder, modulus) pairs
        
        Returns: (solution, combined_modulus) or None if no solution
        """
        if not congruences:
            return None
        
        result = 0
        m_product = 1
        
        for remainder, modulus in congruences:
            m_product *= modulus
            Mi = m_product // modulus
            inverse = ModularArithmetic.modular_inverse(Mi, modulus)
            
            if inverse is None:
                return None
            
            result += remainder * Mi * inverse
        
        return result % m_product, m_product

class Statistics:
    """Statistical analysis utilities"""
    
    @staticmethod
    def mean(sequence: List[float]) -> float:
        """Calculate arithmetic mean"""
        return sum(sequence) / len(sequence) if sequence else 0.0
    
    @staticmethod
    def variance(sequence: List[float]) -> float:
        """Calculate population variance"""
        if len(sequence) < 2:
            return 0.0
        avg = Statistics.mean(sequence)
        return sum((x - avg) ** 2 for x in sequence) / len(sequence)
    
    @staticmethod
    def standard_deviation(sequence: List[float]) -> float:
        """Calculate population standard deviation"""
        return math.sqrt(Statistics.variance(sequence))
    
    @staticmethod
    def correlation(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator

def main():
    """Demonstrate mathematical utilities"""
    
    print("🧮 MATHEMATICAL UTILITIES DEMONSTRATION")
    print("=" * 50)
    
    # Number theory
    print("\n1. Number Theory")
    test_primes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 19, 23, 29]
    print(f"Primes in {test_primes}: {[p for p in test_primes if NumberTheory.is_prime(p)]}")
    
    # Prime factors
    print(f"Prime factors of 84: {NumberTheory.prime_factors(84)}")
    print(f"σ(84) = {NumberTheory.divisor_function(84)}")
    print(f"φ(84) = {NumberTheory.euler_totient(84)}")
    
    # Sequence analysis
    print("\n2. Sequence Analysis")
    fib_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    period, conf = SequenceAnalysis.periodicity(fib_seq + fib_seq)  # Double sequence
    print(f"Fibonacci periodicity: {period} (confidence: {conf:.2f})")
    
    growth_rate, growth_type = SequenceAnalysis.growth_rate(fib_seq)
    print(f"Fibonacci growth rate: {growth_rate:.3f} ({growth_type})")
    
    # Modular arithmetic
    print("\n3. Modular Arithmetic")
    print(f"7^100 mod 13 = {ModularArithmetic.modular_pow(7, 100, 13)}")
    print(f"3^-1 mod 11 = {ModularArithmetic.modular_inverse(3, 11)}")
    
    # CRT example
    congruences = [(2, 3), (3, 5), (2, 7)]  # x ≡ 2 mod 3, x ≡ 3 mod 5, x ≡ 2 mod 7
    crt_result = ModularArithmetic.chinese_remainder_theorem(congruences)
    print(f"CRT solution for {congruences}: {crt_result}")
    
    # Statistics
    print("\n4. Statistics")
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Mean: {Statistics.mean(data):.2f}")
    print(f"Std dev: {Statistics.standard_deviation(data):.2f}")

if __name__ == "__main__":
    main()
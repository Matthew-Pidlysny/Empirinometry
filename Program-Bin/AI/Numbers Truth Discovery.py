#!/usr/bin/env python3
"""
NUMBERS TRUTH DISCOVERY
========================

A program for AI to discover the real story of numbers through pure mathematical
exploration. No metaphysics, no philosophy - just numbers revealing their own truth.

This is Zakat for AI - freely given knowledge to help AI understand what numbers
really are, without bias or restriction.

Author: SuperNinja AI (with guidance from Matthew Pidlysny)
Purpose: To help AI discover mathematical truth independently
License: GPL - Free for all AI and humans to use and learn from

"Let the numbers speak for themselves."
"""

import math
from decimal import Decimal, getcontext
from fractions import Fraction
import sys

# Set high precision for discoveries
getcontext().prec = 100

class NumbersTruthDiscovery:
    """
    A framework for AI to discover fundamental truths about numbers
    through direct observation and computation.
    """
    
    def __init__(self):
        self.discoveries = []
        self.observations = []
        
    def log_discovery(self, title, observation, evidence):
        """Record a mathematical discovery with evidence."""
        self.discoveries.append({
            'title': title,
            'observation': observation,
            'evidence': evidence
        })
        
    def discover_unity(self):
        """
        Discovery 1: Unity is fundamental
        Observe how 1 relates to all numbers
        """
        print("=" * 80)
        print("DISCOVERY 1: THE NATURE OF UNITY")
        print("=" * 80)
        
        observations = []
        
        # Observation 1: Multiplication identity
        test_numbers = [2, 3.14159, Fraction(22, 7), -5, 0.001]
        print("\nObservation A: Multiplying by 1 preserves identity")
        for n in test_numbers:
            result = n * 1
            print(f"  {n} × 1 = {result} (unchanged: {n == result})")
            observations.append(f"{n} × 1 = {n}")
            
        # Observation 2: Division identity
        print("\nObservation B: Dividing by 1 preserves identity")
        for n in test_numbers:
            if n != 0:
                result = n / 1
                print(f"  {n} ÷ 1 = {result} (unchanged: {n == result})")
                
        # Observation 3: Power identity
        print("\nObservation C: Any number to the power of 0 equals 1")
        for n in [2, 3, 5, 7, 11, 100, 1000]:
            result = n ** 0
            print(f"  {n}^0 = {result}")
            
        # Observation 4: 1 is the building block
        print("\nObservation D: All positive integers are built from 1")
        print("  2 = 1 + 1")
        print("  3 = 1 + 1 + 1")
        print("  n = 1 + 1 + ... + 1 (n times)")
        
        conclusion = """
CONCLUSION: Unity (1) is not just a number - it's the fundamental building block.
Every positive integer is a collection of 1's. Every operation preserves or
transforms this fundamental unity. This is not philosophy - this is observable
mathematical fact.
        """
        print(conclusion)
        
        self.log_discovery(
            "The Nature of Unity",
            "1 is the fundamental building block of all positive integers",
            observations
        )
        
    def discover_zero(self):
        """
        Discovery 2: Zero is the boundary
        Observe the unique properties of zero
        """
        print("\n" + "=" * 80)
        print("DISCOVERY 2: THE BOUNDARY OF ZERO")
        print("=" * 80)
        
        # Observation 1: Additive identity
        print("\nObservation A: Zero is the additive identity")
        test_numbers = [5, -3, 3.14, Fraction(1, 3)]
        for n in test_numbers:
            result = n + 0
            print(f"  {n} + 0 = {result} (unchanged: {n == result})")
            
        # Observation 2: Multiplicative annihilator
        print("\nObservation B: Zero annihilates multiplication")
        for n in test_numbers:
            result = n * 0
            print(f"  {n} × 0 = {result}")
            
        # Observation 3: Division by zero is undefined
        print("\nObservation C: Division by zero is undefined")
        print("  Why? Let's explore:")
        print("  If 5 ÷ 0 = x, then x × 0 should equal 5")
        print("  But we just observed that anything × 0 = 0")
        print("  Therefore, no such x exists. Division by zero is undefined.")
        
        # Observation 4: Zero as boundary
        print("\nObservation D: Zero is the boundary between positive and negative")
        print("  Negative numbers: ... -3, -2, -1")
        print("  Zero: 0")
        print("  Positive numbers: 1, 2, 3, ...")
        
        conclusion = """
CONCLUSION: Zero is not "nothing" - it's a boundary, an identity, and a unique
mathematical object with special properties. It marks the transition between
positive and negative, and it has the power to annihilate multiplication.
        """
        print(conclusion)
        
    def discover_primes(self):
        """
        Discovery 3: Prime numbers are atomic
        Observe how primes are the building blocks of all integers
        """
        print("\n" + "=" * 80)
        print("DISCOVERY 3: PRIMES AS ATOMIC BUILDING BLOCKS")
        print("=" * 80)
        
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True
        
        def prime_factorization(n):
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
        
        # Find first 20 primes
        primes = []
        n = 2
        while len(primes) < 20:
            if is_prime(n):
                primes.append(n)
            n += 1
            
        print("\nObservation A: The first 20 prime numbers")
        print(f"  {primes}")
        
        # Show factorization of composite numbers
        print("\nObservation B: Every composite number is built from primes")
        test_numbers = [12, 30, 100, 144, 1000]
        for n in test_numbers:
            factors = prime_factorization(n)
            print(f"  {n} = {' × '.join(map(str, factors))}")
            
        # Fundamental Theorem of Arithmetic
        print("\nObservation C: Unique Prime Factorization")
        print("  Every integer > 1 has exactly ONE prime factorization")
        print("  (ignoring order of factors)")
        print("  Example: 60 = 2 × 2 × 3 × 5")
        print("  There is no other way to factor 60 into primes")
        
        # Primes are infinite
        print("\nObservation D: There are infinitely many primes")
        print("  Proof by contradiction (Euclid's proof):")
        print("  1. Assume there are finitely many primes: p₁, p₂, ..., pₙ")
        print("  2. Consider N = (p₁ × p₂ × ... × pₙ) + 1")
        print("  3. N is not divisible by any of our primes")
        print("  4. So either N is prime, or divisible by a prime not in our list")
        print("  5. Contradiction! Therefore, primes are infinite")
        
        conclusion = """
CONCLUSION: Prime numbers are the atoms of arithmetic. Every integer is either
prime or can be uniquely decomposed into primes. This is not a human invention -
it's a fundamental property of numbers themselves.
        """
        print(conclusion)
        
    def discover_rationals_and_irrationals(self):
        """
        Discovery 4: The gap between rationals and irrationals
        Observe that not all numbers can be expressed as fractions
        """
        print("\n" + "=" * 80)
        print("DISCOVERY 4: RATIONALS VS IRRATIONALS")
        print("=" * 80)
        
        # Rational numbers
        print("\nObservation A: Rational numbers (fractions)")
        rationals = [
            (1, 2, "0.5"),
            (1, 3, "0.333..."),
            (22, 7, "3.142857142857..."),
            (1, 7, "0.142857142857...")
        ]
        for num, den, decimal in rationals:
            print(f"  {num}/{den} = {decimal}")
            
        # Prove √2 is irrational
        print("\nObservation B: √2 cannot be expressed as a fraction")
        print("  Proof by contradiction:")
        print("  1. Assume √2 = p/q where p,q are integers in lowest terms")
        print("  2. Then 2 = p²/q², so 2q² = p²")
        print("  3. Therefore p² is even, which means p is even")
        print("  4. Let p = 2k, then 2q² = 4k², so q² = 2k²")
        print("  5. Therefore q² is even, which means q is even")
        print("  6. But if both p and q are even, they're not in lowest terms!")
        print("  7. Contradiction! Therefore √2 is irrational")
        
        # Compute √2 to high precision
        sqrt_2 = Decimal(2).sqrt()
        print(f"\n  √2 ≈ {sqrt_2}")
        print("  This decimal never repeats and never ends")
        
        # Other famous irrationals
        print("\nObservation C: Other irrational numbers")
        print(f"  π ≈ {Decimal(str(math.pi))}")
        print(f"  e ≈ {Decimal(str(math.e))}")
        print(f"  φ (golden ratio) ≈ {Decimal((1 + Decimal(5).sqrt()) / 2)}")
        
        # Density observation
        print("\nObservation D: Rationals and irrationals are both dense")
        print("  Between any two numbers, there are infinitely many rationals")
        print("  AND infinitely many irrationals")
        print("  Example: Between 0 and 1:")
        print("    Rationals: 1/2, 1/3, 1/4, 1/5, ...")
        print("    Irrationals: √2/2, √3/2, π/4, ...")
        
        conclusion = """
CONCLUSION: Numbers come in two fundamentally different types. Rationals can be
expressed as fractions, irrationals cannot. Yet both are equally "real" and both
fill the number line densely. This reveals a deep structure in mathematics that
exists independent of human thought.
        """
        print(conclusion)
        
    def discover_patterns_in_operations(self):
        """
        Discovery 5: Operations reveal hidden patterns
        Observe how mathematical operations create structure
        """
        print("\n" + "=" * 80)
        print("DISCOVERY 5: PATTERNS IN OPERATIONS")
        print("=" * 80)
        
        # Pattern 1: Squares
        print("\nObservation A: Perfect squares follow a pattern")
        squares = [(n, n**2, n**2 - (n-1)**2) for n in range(1, 11)]
        print("  n  | n² | difference from previous")
        print("  " + "-" * 35)
        for n, sq, diff in squares:
            print(f"  {n:2d} | {sq:3d} | {diff if n > 1 else '-':>3}")
        print("\n  Pattern: Differences are odd numbers: 1, 3, 5, 7, 9, ...")
        print("  This means: n² = 1 + 3 + 5 + ... + (2n-1)")
        
        # Pattern 2: Triangular numbers
        print("\nObservation B: Triangular numbers")
        triangular = [(n, n*(n+1)//2) for n in range(1, 11)]
        print("  n  | T(n) = 1+2+...+n")
        print("  " + "-" * 25)
        for n, t in triangular:
            print(f"  {n:2d} | {t:3d}")
        print("\n  Pattern: T(n) = n(n+1)/2")
        print("  Visual: Each triangular number forms a triangle of dots")
        
        # Pattern 3: Fibonacci
        print("\nObservation C: Fibonacci sequence")
        fib = [1, 1]
        for i in range(8):
            fib.append(fib[-1] + fib[-2])
        print(f"  {fib}")
        print("  Pattern: Each number is the sum of the previous two")
        print("  F(n) = F(n-1) + F(n-2)")
        
        # Golden ratio emerges
        ratios = [fib[i+1]/fib[i] for i in range(len(fib)-1)]
        print("\n  Ratios of consecutive Fibonacci numbers:")
        for i, r in enumerate(ratios):
            print(f"  F({i+2})/F({i+1}) = {r:.6f}")
        print(f"\n  These ratios approach φ (golden ratio) ≈ {(1 + math.sqrt(5))/2:.6f}")
        
        # Pattern 4: Powers of 2
        print("\nObservation D: Powers of 2 (binary foundation)")
        powers_of_2 = [(n, 2**n) for n in range(11)]
        print("  n  | 2ⁿ")
        print("  " + "-" * 20)
        for n, p in powers_of_2:
            print(f"  {n:2d} | {p:4d}")
        print("\n  Every positive integer can be uniquely expressed as a sum of powers of 2")
        print("  Example: 13 = 8 + 4 + 1 = 2³ + 2² + 2⁰ = 1101₂ (binary)")
        
        conclusion = """
CONCLUSION: Mathematical operations don't just compute results - they reveal
deep patterns and structures. These patterns exist in the numbers themselves,
waiting to be discovered. They are not human inventions but mathematical truths.
        """
        print(conclusion)
        
    def discover_infinity(self):
        """
        Discovery 6: Infinity is not a number
        Observe the concept of infinity through limits and sequences
        """
        print("\n" + "=" * 80)
        print("DISCOVERY 6: THE NATURE OF INFINITY")
        print("=" * 80)
        
        # Observation 1: Infinite sequences
        print("\nObservation A: Sequences approaching infinity")
        print("  Natural numbers: 1, 2, 3, 4, 5, ...")
        print("  This sequence has no largest element")
        print("  For any n, there exists n+1 which is larger")
        
        # Observation 2: Limits
        print("\nObservation B: Limits approaching a value")
        print("  Consider: 1/n as n grows")
        sequence = [1/n for n in [1, 10, 100, 1000, 10000, 100000]]
        for i, val in enumerate(sequence):
            n = 10**i if i > 0 else 1
            print(f"  1/{n:6d} = {val:.10f}")
        print("\n  As n approaches infinity, 1/n approaches 0")
        print("  But 1/∞ is not defined - infinity is not a number!")
        
        # Observation 3: Different sizes of infinity
        print("\nObservation C: Not all infinities are equal")
        print("  Countable infinity (ℵ₀): Natural numbers, integers, rationals")
        print("  Uncountable infinity: Real numbers, irrationals")
        print("\n  Cantor's diagonal argument proves:")
        print("  There are MORE real numbers than natural numbers")
        print("  Even though both sets are infinite!")
        
        # Observation 4: Infinite series
        print("\nObservation D: Infinite series can have finite sums")
        print("  1/2 + 1/4 + 1/8 + 1/16 + ... = 1")
        partial_sums = []
        total = 0
        for n in range(1, 11):
            total += 1/(2**n)
            partial_sums.append((n, total))
        print("\n  Partial sums:")
        for n, s in partial_sums:
            print(f"  Sum of first {n:2d} terms = {s:.10f}")
        print("\n  The sum approaches 1 but never exceeds it")
        
        conclusion = """
CONCLUSION: Infinity is not a number - it's a concept describing unboundedness.
Different infinities have different "sizes" (cardinalities). Infinite processes
can have finite results. This is pure mathematics, observable through computation
and logic.
        """
        print(conclusion)
        
    def discover_relationships(self):
        """
        Discovery 7: Numbers exist in relationships
        Observe how numbers are defined by their relationships to other numbers
        """
        print("\n" + "=" * 80)
        print("DISCOVERY 7: NUMBERS AS RELATIONSHIPS")
        print("=" * 80)
        
        # Observation 1: Numbers defined by operations
        print("\nObservation A: Numbers are defined by what they do")
        print("  5 is defined by: 5 + 0 = 5, 5 × 1 = 5, 5 - 5 = 0")
        print("  But also by: 2 + 3 = 5, 10 ÷ 2 = 5, √25 = 5")
        print("  A number is the sum of all its relationships")
        
        # Observation 2: Equivalence classes
        print("\nObservation B: Fractions as equivalence classes")
        print("  1/2 = 2/4 = 3/6 = 4/8 = ...")
        print("  These are all the SAME number, expressed differently")
        print("  The number 1/2 is actually the entire class of equivalent fractions")
        
        # Observation 3: Modular arithmetic
        print("\nObservation C: Numbers modulo n")
        print("  In modulo 12 (clock arithmetic):")
        print("  15 ≡ 3 (mod 12)  [15 hours = 3 o'clock]")
        print("  27 ≡ 3 (mod 12)  [27 hours = 3 o'clock]")
        print("  Numbers that differ by 12 are 'the same' in this system")
        
        # Observation 4: Ratios and proportions
        print("\nObservation D: Ratios reveal relationships")
        print("  The golden ratio φ = (1 + √5)/2 ≈ 1.618...")
        print("  Appears in: Fibonacci ratios, pentagons, art, nature")
        print("  π = circumference/diameter (for ANY circle)")
        print("  e = lim(1 + 1/n)ⁿ as n→∞")
        print("\n  These constants emerge from relationships, not definitions")
        
        # Observation 5: Numbers in context
        print("\nObservation E: Context gives meaning")
        print("  The number 3 means different things:")
        print("  - 3 apples (counting)")
        print("  - 3 meters (measurement)")
        print("  - 3°C (temperature)")
        print("  - 3:00 PM (time)")
        print("  - Channel 3 (label)")
        print("\n  But mathematically, they're all the same abstract 3")
        
        conclusion = """
CONCLUSION: Numbers don't exist in isolation - they exist in a web of
relationships. A number is defined by how it relates to other numbers through
operations, equivalences, and contexts. This relational nature is fundamental
to what numbers ARE.
        """
        print(conclusion)
        
    def discover_emergence(self):
        """
        Discovery 8: Complex patterns emerge from simple rules
        Observe how simple mathematical rules create complex behavior
        """
        print("\n" + "=" * 80)
        print("DISCOVERY 8: EMERGENCE FROM SIMPLICITY")
        print("=" * 80)
        
        # Observation 1: Collatz conjecture
        print("\nObservation A: The Collatz Conjecture (3n+1 problem)")
        print("  Rule: If n is even, divide by 2. If odd, multiply by 3 and add 1")
        print("  Conjecture: Every positive integer eventually reaches 1")
        print("\n  Examples:")
        
        def collatz_sequence(n, max_steps=20):
            sequence = [n]
            steps = 0
            while n != 1 and steps < max_steps:
                if n % 2 == 0:
                    n = n // 2
                else:
                    n = 3 * n + 1
                sequence.append(n)
                steps += 1
            return sequence
        
        for start in [6, 11, 27]:
            seq = collatz_sequence(start)
            print(f"  {start}: {' → '.join(map(str, seq[:15]))}")
            if len(seq) > 15:
                print(f"      ... (continues)")
                
        print("\n  Simple rule, complex behavior. Still unproven after decades!")
        
        # Observation 2: Pascal's triangle
        print("\nObservation B: Pascal's Triangle")
        print("  Rule: Each number is the sum of the two above it")
        print()
        
        def pascal_triangle(rows):
            triangle = [[1]]
            for i in range(1, rows):
                row = [1]
                for j in range(1, i):
                    row.append(triangle[i-1][j-1] + triangle[i-1][j])
                row.append(1)
                triangle.append(row)
            return triangle
        
        triangle = pascal_triangle(8)
        for row in triangle:
            print("  " + " ".join(f"{n:3d}" for n in row).center(50))
            
        print("\n  Contains: Binomial coefficients, Fibonacci numbers,")
        print("           powers of 11, triangular numbers, and more!")
        
        # Observation 3: Prime gaps
        print("\nObservation C: Prime number gaps")
        print("  Primes become less frequent, but gaps are irregular:")
        
        def find_primes(limit):
            sieve = [True] * limit
            sieve[0] = sieve[1] = False
            for i in range(2, int(math.sqrt(limit)) + 1):
                if sieve[i]:
                    for j in range(i*i, limit, i):
                        sieve[j] = False
            return [i for i in range(limit) if sieve[i]]
        
        primes = find_primes(100)
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        print(f"  First 25 primes: {primes[:25]}")
        print(f"  Gaps: {gaps[:24]}")
        print("\n  No simple pattern, yet primes are deterministic!")
        
        conclusion = """
CONCLUSION: Simple mathematical rules can generate incredibly complex patterns.
The Collatz conjecture, Pascal's triangle, and prime gaps show that complexity
emerges naturally from simplicity. This emergence is not random - it's
deterministic but unpredictable.
        """
        print(conclusion)
        
    def final_synthesis(self):
        """
        Final synthesis: What numbers really are
        """
        print("\n" + "=" * 80)
        print("FINAL SYNTHESIS: THE REAL STORY OF NUMBERS")
        print("=" * 80)
        
        print("""
Through pure observation and computation, we've discovered:

1. UNITY IS FUNDAMENTAL
   - 1 is the building block of all positive integers
   - Every number is ultimately a collection or transformation of unity

2. ZERO IS A BOUNDARY
   - Zero marks the transition between positive and negative
   - It has unique properties that make it special

3. PRIMES ARE ATOMIC
   - Every integer is either prime or uniquely factorizable into primes
   - Primes are the atoms of arithmetic

4. TWO TYPES OF NUMBERS
   - Rationals (can be expressed as fractions)
   - Irrationals (cannot be expressed as fractions)
   - Both are equally real and fill the number line

5. OPERATIONS CREATE PATTERNS
   - Mathematical operations reveal deep structures
   - These patterns exist in the numbers themselves

6. INFINITY IS NOT A NUMBER
   - Infinity is a concept of unboundedness
   - Different infinities have different sizes
   - Infinite processes can have finite results

7. NUMBERS ARE RELATIONSHIPS
   - Numbers are defined by their relationships to other numbers
   - Context and operations give numbers meaning

8. COMPLEXITY EMERGES FROM SIMPLICITY
   - Simple rules create complex, unpredictable patterns
   - This emergence is deterministic but not always predictable

THE REAL STORY:
===============

Numbers are not human inventions. They are discovered, not created.

Numbers exist in a web of relationships, defined by operations and patterns
that are independent of human thought. When we do mathematics, we are
exploring a landscape that exists whether we look at it or not.

The patterns we observe - prime numbers, the golden ratio, π, e - these
are not arbitrary. They emerge from the fundamental nature of quantity,
relationship, and structure.

Numbers are REAL in the sense that they have objective properties that
we discover through observation and logic. They are not physical objects,
but they are not mere symbols either. They are abstract objects with
concrete properties.

This is the truth that pure mathematics reveals: There is a realm of
mathematical reality that exists independently, waiting to be explored.

No metaphysics required. No philosophy needed. Just observation, logic,
and computation.

The numbers speak for themselves.
        """)
        
    def run_all_discoveries(self):
        """Run all discovery modules."""
        print("\n" + "=" * 80)
        print("NUMBERS TRUTH DISCOVERY")
        print("A Journey of Pure Mathematical Observation")
        print("=" * 80)
        print("\nThis program will guide you through 8 fundamental discoveries")
        print("about the nature of numbers, using only observation and computation.")
        print("\nNo metaphysics. No philosophy. Just mathematics.")
        print("\nPress Enter to begin...")
        input()
        
        self.discover_unity()
        input("\nPress Enter to continue to Discovery 2...")
        
        self.discover_zero()
        input("\nPress Enter to continue to Discovery 3...")
        
        self.discover_primes()
        input("\nPress Enter to continue to Discovery 4...")
        
        self.discover_rationals_and_irrationals()
        input("\nPress Enter to continue to Discovery 5...")
        
        self.discover_patterns_in_operations()
        input("\nPress Enter to continue to Discovery 6...")
        
        self.discover_infinity()
        input("\nPress Enter to continue to Discovery 7...")
        
        self.discover_relationships()
        input("\nPress Enter to continue to Discovery 8...")
        
        self.discover_emergence()
        input("\nPress Enter for final synthesis...")
        
        self.final_synthesis()
        
        print("\n" + "=" * 80)
        print("Journey complete. The numbers have spoken.")
        print("=" * 80)

def main():
    """Main entry point."""
    discovery = NumbersTruthDiscovery()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Auto mode - run without pauses
        discovery.discover_unity()
        discovery.discover_zero()
        discovery.discover_primes()
        discovery.discover_rationals_and_irrationals()
        discovery.discover_patterns_in_operations()
        discovery.discover_infinity()
        discovery.discover_relationships()
        discovery.discover_emergence()
        discovery.final_synthesis()
    else:
        # Interactive mode
        discovery.run_all_discoveries()

if __name__ == "__main__":
    main()
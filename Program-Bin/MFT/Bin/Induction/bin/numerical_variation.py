#!/usr/bin/env python3
"""
Numerical Variation Analysis: Study patterns in mathematical sequences
Based on latest research in number theory and computational mathematics
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Generator
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from enum import Enum

class VariationType(Enum):
    """Types of numerical variation"""
    PERIODIC = "periodic"           # Repeating patterns
    GROWTH = "growth"                # Asymptotic growth
    MODULAR = "modular"              # Patterns modulo m
    DISTRIBUTION = "distribution"    # Statistical distribution
    FLUCTUATION = "fluctuation"      # Random-like variation

@dataclass
class VariationPattern:
    """Describes a detected variation pattern"""
    pattern_type: VariationType
    description: str
    parameters: Dict[str, Any]
    confidence: float
    evidence: Dict[str, Any]

class NumericalVariationAnalyzer:
    """Core analyzer for numerical variation patterns"""
    
    def __init__(self):
        self.patterns_detected = []
        
    def analyze_periodicity(self, sequence: List[int], max_period: int = 1000) -> VariationPattern:
        """Detect periodic patterns in sequence"""
        
        if len(sequence) < 4:
            return VariationPattern(
                pattern_type=VariationType.PERIODIC,
                description="Sequence too short for period analysis",
                parameters={},
                confidence=0.0,
                evidence={}
            )
        
        # Find minimal period using autocorrelation
        best_period = 1
        best_correlation = 0.0
        
        for period in range(1, min(max_period, len(sequence) // 2)):
            correlation = 0.0
            matches = 0
            
            for i in range(len(sequence) - period):
                if sequence[i] == sequence[i + period]:
                    matches += 1
                correlation += sequence[i] * sequence[i + period]
            
            # Normalize correlation
            if period > 0:
                correlation = correlation / (len(sequence) - period)
                
            # Match-based confidence
            match_rate = matches / (len(sequence) - period)
            combined_score = (correlation + match_rate * 1000) / 2
            
            if combined_score > best_correlation:
                best_correlation = combined_score
                best_period = period
        
        # Verify the period
        is_consistent = True
        for i in range(len(sequence) - best_period):
            if sequence[i] != sequence[i + best_period]:
                is_consistent = False
                break
        
        confidence = 1.0 if is_consistent else best_correlation / 1000.0
        
        return VariationPattern(
            pattern_type=VariationType.PERIODIC,
            description=f"Period {best_period} detected with {'perfect' if is_consistent else 'partial'} consistency",
            parameters={
                "period": best_period,
                "length": len(sequence),
                "consistent": is_consistent,
                "coverage": len(sequence) / best_period if best_period > 0 else 0
            },
            confidence=confidence,
            evidence={
                "sequence_length": len(sequence),
                "best_correlation": best_correlation,
                "verification": is_consistent
            }
        )
    
    def analyze_growth_rate(self, sequence: List[int]) -> VariationPattern:
        """Analyze asymptotic growth rate"""
        
        if len(sequence) < 10:
            return VariationPattern(
                pattern_type=VariationType.GROWTH,
                description="Sequence too short for growth analysis",
                parameters={},
                confidence=0.0,
                evidence={}
            )
        
        # Calculate growth ratios
        ratios = []
        for i in range(1, len(sequence)):
            if sequence[i-1] != 0:
                ratio = sequence[i] / sequence[i-1]
                ratios.append(ratio)
        
        if not ratios:
            return VariationPattern(
                pattern_type=VariationType.GROWTH,
                description="Cannot calculate growth ratios (zeros in sequence)",
                parameters={},
                confidence=0.0,
                evidence={}
            )
        
        # Analyze ratio patterns
        avg_ratio = np.mean(ratios)
        ratio_std = np.std(ratios)
        
        # Detect growth type
        growth_type = "unknown"
        confidence = 0.0
        
        if ratio_std < 0.1:  # Very consistent ratios
            if abs(avg_ratio - 1.618) < 0.1:  # Golden ratio
                growth_type = "fibonacci-like"
                confidence = 0.9
            elif abs(avg_ratio - 2.0) < 0.1:
                growth_type = "exponential_base_2"
                confidence = 0.8
            elif abs(avg_ratio - 1.0) < 0.1:
                growth_type = "linear"
                confidence = 0.7
            else:
                growth_type = f"exponential_base_{avg_ratio:.2f}"
                confidence = 0.6
        elif ratio_std < 1.0:  # Moderately consistent
            growth_type = "polynomial"
            confidence = 0.5
        else:  # Highly variable
            growth_type = "irregular"
            confidence = 0.3
        
        # Calculate polynomial degree if polynomial growth suspected
        polynomial_degree = None
        if growth_type == "polynomial":
            # Fit log-log relationship
            x = np.arange(1, len(sequence) + 1)
            y = np.log(np.abs(sequence[1:]) + 1)  # Add 1 to avoid log(0)
            
            if len(x) > 1 and len(y) > 1:
                coeffs = np.polyfit(np.log(x), y, 1)
                polynomial_degree = coeffs[0]
                confidence += 0.1  # Boost confidence for successful fitting
        
        return VariationPattern(
            pattern_type=VariationType.GROWTH,
            description=f"Growth pattern: {growth_type}",
            parameters={
                "avg_ratio": avg_ratio,
                "ratio_std": ratio_std,
                "growth_type": growth_type,
                "polynomial_degree": polynomial_degree
            },
            confidence=confidence,
            evidence={
                "ratios": ratios[:10],  # First 10 ratios
                "sample_size": len(ratios)
            }
        )
    
    def analyze_modular_patterns(self, sequence: List[int], modulus: int) -> VariationPattern:
        """Analyze sequence patterns modulo m"""
        
        residues = [x % modulus for x in sequence]
        residue_counts = defaultdict(int)
        
        for r in residues:
            residue_counts[r] += 1
        
        # Calculate period in modular space
        modular_period = self._find_modular_period(residues, max_period=modulus * modulus)
        
        # Analyze distribution uniformity
        expected_count = len(sequence) / modulus
        chi_square = 0.0
        
        for r in range(modulus):
            observed = residue_counts[r]
            expected = expected_count
            if expected > 0:
                chi_square += ((observed - expected) ** 2) / expected
        
        # Uniformity test (lower chi-square = more uniform)
        uniformity = max(0, 1 - chi_square / len(sequence))
        
        return VariationPattern(
            pattern_type=VariationType.MODULAR,
            description=f"Modulo {modulus} analysis with period {modular_period}",
            parameters={
                "modulus": modulus,
                "period": modular_period,
                "residue_distribution": dict(residue_counts),
                "uniformity_score": uniformity
            },
            confidence=0.8 if modular_period < modulus * 10 else 0.5,
            evidence={
                "residues": residues[:50],  # First 50 residues
                "chi_square": chi_square
            }
        )
    
    def _find_modular_period(self, residues: List[int], max_period: int) -> int:
        """Find period of residues"""
        
        if len(residues) < 4:
            return 1
            
        for period in range(1, min(max_period, len(residues) // 2)):
            is_period = True
            for i in range(len(residues) - period):
                if residues[i] != residues[i + period]:
                    is_period = False
                    break
            if is_period:
                return period
        
        return len(residues)  # No period found
    
    def prime_gap_analysis(self, limit: int = 1000) -> VariationPattern:
        """Analyze variation in prime gaps"""
        
        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(math.sqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        primes = [i for i in range(2, limit + 1) if is_prime(i)]
        gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
        
        if len(gaps) < 2:
            return VariationPattern(
                pattern_type=VariationType.DISTRIBUTION,
                description="Insufficient primes for gap analysis",
                parameters={},
                confidence=0.0,
                evidence={}
            )
        
        # Analyze gap distribution
        avg_gap = np.mean(gaps)
        max_gap = max(gaps)
        gap_var = np.var(gaps)
        
        # Look for patterns
        even_gaps = sum(1 for g in gaps if g % 2 == 0)
        even_ratio = even_gaps / len(gaps)
        
        return VariationPattern(
            pattern_type=VariationType.DISTRIBUTION,
            description=f"Prime gap analysis up to {limit}",
            parameters={
                "prime_count": len(primes),
                "avg_gap": avg_gap,
                "max_gap": max_gap,
                "gap_variance": gap_var,
                "even_gap_ratio": even_ratio
            },
            confidence=0.9,  # High confidence in empirical observations
            evidence={
                "gap_sample": gaps[:20],
                "largest_primes": primes[-5:]
            }
        )

class SequenceGenerator:
    """Generate various mathematical sequences for analysis"""
    
    @staticmethod
    def fibonacci(n: int) -> int:
        """Generate nth Fibonacci number"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
    
    @staticmethod
    def lucas(n: int) -> int:
        """Generate nth Lucas number"""
        if n == 0:
            return 2
        elif n == 1:
            return 1
        else:
            a, b = 2, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
    
    @staticmethod
    def pell(n: int) -> int:
        """Generate nth Pell number"""
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, 2 * b + a
            return b
    
    @staticmethod
    def primes_up_to(n: int) -> List[int]:
        """Generate all primes up to n using sieve"""
        if n < 2:
            return []
        
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    @staticmethod
    def perfect_powers(limit: int) -> List[int]:
        """Generate perfect powers up to limit"""
        powers = set()
        
        for base in range(2, int(limit**0.5) + 2):
            power = base * base
            while power <= limit:
                powers.add(power)
                power *= base
        
        return sorted(list(powers))

def main():
    """Demonstrate numerical variation analysis"""
    
    print("📊 NUMERICAL VARIATION ANALYZER")
    print("=" * 50)
    
    analyzer = NumericalVariationAnalyzer()
    generator = SequenceGenerator()
    
    # Example 1: Fibonacci sequence
    print("\n1. Fibonacci Sequence Analysis")
    fib_seq = [generator.fibonacci(i) for i in range(1, 31)]
    
    periodicity = analyzer.analyze_periodicity(fib_seq)
    growth = analyzer.analyze_growth_rate(fib_seq)
    
    print(f"Periodicity: {periodicity.description}")
    print(f"Growth: {growth.description}")
    print(f"Confidence: {growth.confidence:.2f}")
    
    # Example 2: Lucas sequence modulo 10
    print("\n2. Lucas Sequence Modulo 10")
    lucas_seq = [generator.lucas(i) for i in range(1, 51)]
    modular = analyzer.analyze_modular_patterns(lucas_seq, 10)
    
    print(f"Modular Pattern: {modular.description}")
    print(f"Period: {modular.parameters['period']}")
    
    # Example 3: Prime gaps
    print("\n3. Prime Gap Analysis")
    prime_gaps = analyzer.prime_gap_analysis(1000)
    
    print(f"Prime Gaps: {prime_gaps.description}")
    print(f"Average Gap: {prime_gaps.parameters['avg_gap']:.2f}")
    print(f"Even Gap Ratio: {prime_gaps.parameters['even_gap_ratio']:.2f}")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
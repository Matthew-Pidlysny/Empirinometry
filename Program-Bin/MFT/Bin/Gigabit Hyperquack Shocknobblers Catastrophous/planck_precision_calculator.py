#!/usr/bin/env python3
"""
Planck Precision Calculator
============================
Calculates optimal precision for any number based on Planck scale boundaries.

The Planck length (1.616255 × 10^-35 meters) represents the smallest meaningful
length scale in physics. Below this scale, quantum gravitational effects dominate
and continuous numbers lose physical meaning.

This program determines the maximum meaningful precision for any numerical
calculation based on the physical context and Planck scale constraints.
"""

import math
from decimal import Decimal, getcontext

# Physical constants
PLANCK_LENGTH = Decimal('1.616255e-35')  # meters
PLANCK_TIME = Decimal('5.391247e-44')    # seconds
PLANCK_MASS = Decimal('2.176434e-8')     # kilograms
PLANCK_ENERGY = Decimal('1.956e9')       # joules
SPEED_OF_LIGHT = Decimal('299792458')    # m/s
PLANCK_CONSTANT = Decimal('6.62607015e-34')  # J⋅s

class PlanckPrecisionCalculator:
    """Calculate optimal precision based on Planck scale physics."""
    
    def __init__(self):
        """Initialize calculator with high precision."""
        getcontext().prec = 100
        
    def calculate_spatial_precision(self, length_scale):
        """
        Calculate maximum meaningful decimal places for spatial measurements.
        
        Args:
            length_scale: The characteristic length scale of the measurement (meters)
            
        Returns:
            Maximum meaningful decimal places
        """
        length_scale = Decimal(str(length_scale))
        
        if length_scale <= PLANCK_LENGTH:
            return 0  # Below Planck scale, no meaningful precision
        
        # Calculate ratio to Planck length
        ratio = length_scale / PLANCK_LENGTH
        
        # Maximum meaningful digits = log10(ratio)
        max_digits = int(ratio.log10())
        
        return max(0, min(max_digits, 35))  # Cap at 35 (Planck boundary)
    
    def calculate_temporal_precision(self, time_scale):
        """
        Calculate maximum meaningful decimal places for time measurements.
        
        Args:
            time_scale: The characteristic time scale of the measurement (seconds)
            
        Returns:
            Maximum meaningful decimal places
        """
        time_scale = Decimal(str(time_scale))
        
        if time_scale <= PLANCK_TIME:
            return 0
        
        ratio = time_scale / PLANCK_TIME
        max_digits = int(ratio.log10())
        
        return max(0, min(max_digits, 44))  # Cap at 44 (Planck time boundary)
    
    def calculate_mass_precision(self, mass_scale):
        """
        Calculate maximum meaningful decimal places for mass measurements.
        
        Args:
            mass_scale: The characteristic mass scale (kilograms)
            
        Returns:
            Maximum meaningful decimal places
        """
        mass_scale = Decimal(str(mass_scale))
        
        if mass_scale <= PLANCK_MASS:
            return 0
        
        ratio = mass_scale / PLANCK_MASS
        max_digits = int(ratio.log10())
        
        return max(0, min(max_digits, 8))
    
    def calculate_energy_precision(self, energy_scale):
        """
        Calculate maximum meaningful decimal places for energy measurements.
        
        Args:
            energy_scale: The characteristic energy scale (joules)
            
        Returns:
            Maximum meaningful decimal places
        """
        energy_scale = Decimal(str(energy_scale))
        
        if energy_scale <= PLANCK_ENERGY:
            return 0
        
        ratio = energy_scale / PLANCK_ENERGY
        max_digits = int(ratio.log10())
        
        return max(0, min(max_digits, 9))
    
    def calculate_dimensionless_precision(self, context="general"):
        """
        Calculate precision for dimensionless numbers based on physical context.
        
        Args:
            context: Physical context ("general", "quantum", "cosmological", "atomic")
            
        Returns:
            Maximum meaningful decimal places
        """
        context_precisions = {
            "general": 35,        # Planck scale default
            "quantum": 61,        # Quantum measurement limit
            "cosmological": 123,  # Information theoretical limit
            "atomic": 15,         # Atomic scale measurements
            "nuclear": 20,        # Nuclear scale
            "particle": 35        # Particle physics scale
        }
        
        return context_precisions.get(context, 35)
    
    def optimize_calculation_precision(self, number, physical_context):
        """
        Determine optimal precision for a specific number in a physical context.
        
        Args:
            number: The number to analyze
            physical_context: Dict with keys like 'type', 'scale', 'unit'
            
        Returns:
            Dict with precision recommendations
        """
        number = Decimal(str(number))
        context_type = physical_context.get('type', 'dimensionless')
        scale = physical_context.get('scale', 1.0)
        
        if context_type == 'length':
            max_precision = self.calculate_spatial_precision(scale)
        elif context_type == 'time':
            max_precision = self.calculate_temporal_precision(scale)
        elif context_type == 'mass':
            max_precision = self.calculate_mass_precision(scale)
        elif context_type == 'energy':
            max_precision = self.calculate_energy_precision(scale)
        else:
            max_precision = self.calculate_dimensionless_precision(
                physical_context.get('context', 'general')
            )
        
        # Calculate current precision of the number
        number_str = str(number)
        if '.' in number_str:
            current_precision = len(number_str.split('.')[1])
        else:
            current_precision = 0
        
        # Determine if current precision exceeds physical meaning
        exceeds_limit = current_precision > max_precision
        
        # Calculate wasted digits
        wasted_digits = max(0, current_precision - max_precision)
        
        return {
            'number': str(number),
            'current_precision': current_precision,
            'max_meaningful_precision': max_precision,
            'exceeds_physical_limit': exceeds_limit,
            'wasted_digits': wasted_digits,
            'optimal_number': self._truncate_to_precision(number, max_precision),
            'physical_context': context_type,
            'scale': str(scale)
        }
    
    def _truncate_to_precision(self, number, precision):
        """Truncate number to specified precision."""
        if precision == 0:
            return str(int(number))
        
        multiplier = Decimal(10) ** precision
        truncated = int(number * multiplier) / multiplier
        return str(truncated)
    
    def batch_analyze(self, numbers, contexts):
        """
        Analyze multiple numbers with their physical contexts.
        
        Args:
            numbers: List of numbers to analyze
            contexts: List of context dicts (same length as numbers)
            
        Returns:
            List of analysis results
        """
        if len(numbers) != len(contexts):
            raise ValueError("Numbers and contexts must have same length")
        
        results = []
        for num, ctx in zip(numbers, contexts):
            results.append(self.optimize_calculation_precision(num, ctx))
        
        return results
    
    def generate_report(self, analysis_results):
        """Generate human-readable report from analysis results."""
        report = []
        report.append("=" * 80)
        report.append("PLANCK PRECISION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        for i, result in enumerate(analysis_results, 1):
            report.append(f"Analysis #{i}")
            report.append("-" * 80)
            report.append(f"Number: {result['number']}")
            report.append(f"Physical Context: {result['physical_context']}")
            report.append(f"Scale: {result['scale']}")
            report.append(f"Current Precision: {result['current_precision']} decimal places")
            report.append(f"Max Meaningful Precision: {result['max_meaningful_precision']} decimal places")
            report.append(f"Exceeds Physical Limit: {result['exceeds_physical_limit']}")
            report.append(f"Wasted Digits: {result['wasted_digits']}")
            report.append(f"Optimal Number: {result['optimal_number']}")
            report.append("")
        
        report.append("=" * 80)
        report.append("PHYSICAL CONSTANTS REFERENCE")
        report.append("=" * 80)
        report.append(f"Planck Length: {PLANCK_LENGTH} m")
        report.append(f"Planck Time: {PLANCK_TIME} s")
        report.append(f"Planck Mass: {PLANCK_MASS} kg")
        report.append(f"Planck Energy: {PLANCK_ENERGY} J")
        report.append("")
        
        return "\n".join(report)


def main():
    """Demonstrate Planck precision calculator."""
    calc = PlanckPrecisionCalculator()
    
    print("PLANCK PRECISION CALCULATOR")
    print("=" * 80)
    print()
    
    # Example 1: Measuring atomic distances
    print("Example 1: Atomic Distance Measurement")
    print("-" * 80)
    atomic_distance = "1.234567890123456789012345678901234567890"  # 39 decimal places
    result = calc.optimize_calculation_precision(
        atomic_distance,
        {'type': 'length', 'scale': 1e-10}  # Angstrom scale
    )
    print(f"Measuring distance at atomic scale (1 Angstrom = 10^-10 m)")
    print(f"Input: {atomic_distance}")
    print(f"Current precision: {result['current_precision']} decimal places")
    print(f"Max meaningful: {result['max_meaningful_precision']} decimal places")
    print(f"Optimal value: {result['optimal_number']}")
    print(f"Wasted digits: {result['wasted_digits']}")
    print()
    
    # Example 2: Measuring time intervals
    print("Example 2: Time Interval Measurement")
    print("-" * 80)
    time_interval = "0.000000000000000000000000000000000000001"  # 39 decimal places
    result = calc.optimize_calculation_precision(
        time_interval,
        {'type': 'time', 'scale': 1e-39}
    )
    print(f"Measuring time at 10^-39 seconds")
    print(f"Input: {time_interval}")
    print(f"Current precision: {result['current_precision']} decimal places")
    print(f"Max meaningful: {result['max_meaningful_precision']} decimal places")
    print(f"Optimal value: {result['optimal_number']}")
    print()
    
    # Example 3: Pi calculation
    print("Example 3: Pi in Different Contexts")
    print("-" * 80)
    pi_value = "3.14159265358979323846264338327950288419716939937510"
    
    contexts = [
        {'type': 'dimensionless', 'context': 'general'},
        {'type': 'dimensionless', 'context': 'quantum'},
        {'type': 'dimensionless', 'context': 'atomic'}
    ]
    
    for ctx in contexts:
        result = calc.optimize_calculation_precision(pi_value, ctx)
        print(f"Context: {ctx.get('context', 'general')}")
        print(f"  Max precision: {result['max_meaningful_precision']} digits")
        print(f"  Optimal π: {result['optimal_number']}")
        print()
    
    # Example 4: Batch analysis
    print("Example 4: Batch Analysis of Multiple Measurements")
    print("-" * 80)
    
    numbers = [
        "299792458.123456789",  # Speed of light with extra precision
        "6.62607015e-34",       # Planck constant
        "1.616255e-35"          # Planck length
    ]
    
    contexts = [
        {'type': 'length', 'scale': 1.0},  # Meter scale
        {'type': 'energy', 'scale': 1e-34},
        {'type': 'length', 'scale': 1e-35}
    ]
    
    results = calc.batch_analyze(numbers, contexts)
    report = calc.generate_report(results)
    print(report)


if __name__ == "__main__":
    main()
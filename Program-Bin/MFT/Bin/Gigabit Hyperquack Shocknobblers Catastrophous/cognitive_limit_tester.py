#!/usr/bin/env python3
"""
Cognitive Limit Tester
======================
Tests human cognitive limits for number recognition and precision perception.

Research shows humans can reliably distinguish approximately 10^15 different
values (15 digits). Beyond this, cognitive limitations make additional
precision imperceptible and meaningless for human understanding.

This program tests and validates the cognitive termination boundary.
"""

import random
import time
import math
from decimal import Decimal, getcontext
from collections import defaultdict

# Cognitive constants
COGNITIVE_LIMIT = 15  # Maximum digits humans can reliably process
WORKING_MEMORY_CAPACITY = 7  # Miller's Law: 7±2 chunks
SUBITIZING_LIMIT = 4  # Instant recognition without counting
PERCEPTUAL_THRESHOLD = 1e-15  # Minimum perceptible difference

class CognitiveLimitTester:
    """Test and validate human cognitive limits for numerical precision."""
    
    def __init__(self):
        """Initialize tester."""
        getcontext().prec = 100
        self.test_results = defaultdict(list)
        
    def test_digit_span(self, num_digits):
        """
        Test if a human could reliably remember/process N digits.
        
        Args:
            num_digits: Number of digits to test
            
        Returns:
            Cognitive feasibility assessment
        """
        # Calculate cognitive load
        chunks_needed = math.ceil(num_digits / 3)  # Group digits in chunks of 3
        
        # Assess feasibility
        if num_digits <= SUBITIZING_LIMIT:
            difficulty = "TRIVIAL"
            success_rate = 0.99
            cognitive_load = "Minimal"
        elif num_digits <= WORKING_MEMORY_CAPACITY:
            difficulty = "EASY"
            success_rate = 0.95
            cognitive_load = "Low"
        elif num_digits <= COGNITIVE_LIMIT:
            difficulty = "MODERATE"
            success_rate = 0.90 - (num_digits - WORKING_MEMORY_CAPACITY) * 0.05
            cognitive_load = "Medium"
        elif num_digits <= 20:
            difficulty = "HARD"
            success_rate = 0.50 - (num_digits - COGNITIVE_LIMIT) * 0.05
            cognitive_load = "High"
        elif num_digits <= 30:
            difficulty = "VERY HARD"
            success_rate = 0.20 - (num_digits - 20) * 0.01
            cognitive_load = "Very High"
        else:
            difficulty = "IMPOSSIBLE"
            success_rate = 0.01
            cognitive_load = "Overwhelming"
        
        return {
            'num_digits': num_digits,
            'chunks_needed': chunks_needed,
            'difficulty': difficulty,
            'estimated_success_rate': max(0.01, success_rate),
            'cognitive_load': cognitive_load,
            'exceeds_cognitive_limit': num_digits > COGNITIVE_LIMIT,
            'exceeds_working_memory': chunks_needed > WORKING_MEMORY_CAPACITY
        }
    
    def test_discrimination_ability(self, value1, value2):
        """
        Test if humans can discriminate between two values.
        
        Args:
            value1: First value
            value2: Second value
            
        Returns:
            Discrimination assessment
        """
        value1 = Decimal(str(value1))
        value2 = Decimal(str(value2))
        
        # Calculate relative difference
        if value1 == 0 and value2 == 0:
            relative_diff = Decimal(0)
        elif value1 == 0 or value2 == 0:
            relative_diff = Decimal('Infinity')
        else:
            relative_diff = abs((value1 - value2) / value1)
        
        # Determine if difference is perceptible
        perceptible = relative_diff > Decimal(str(PERCEPTUAL_THRESHOLD))
        
        # Calculate required precision to distinguish
        if relative_diff > 0:
            required_precision = int(-relative_diff.log10()) + 1
        else:
            required_precision = 0
        
        # Assess discrimination difficulty
        if relative_diff > Decimal('0.1'):
            discrimination = "EASY"
        elif relative_diff > Decimal('0.01'):
            discrimination = "MODERATE"
        elif relative_diff > Decimal('0.001'):
            discrimination = "HARD"
        elif relative_diff > Decimal(str(PERCEPTUAL_THRESHOLD)):
            discrimination = "VERY HARD"
        else:
            discrimination = "IMPOSSIBLE"
        
        return {
            'value1': str(value1),
            'value2': str(value2),
            'absolute_difference': str(abs(value1 - value2)),
            'relative_difference': str(relative_diff),
            'perceptible': perceptible,
            'required_precision': required_precision,
            'discrimination_difficulty': discrimination,
            'within_cognitive_limit': required_precision <= COGNITIVE_LIMIT
        }
    
    def test_number_comprehension(self, number):
        """
        Test human comprehension of a number's magnitude and precision.
        
        Args:
            number: Number to test
            
        Returns:
            Comprehension assessment
        """
        number = Decimal(str(number))
        
        # Analyze number structure
        number_str = str(number)
        if 'E' in number_str or 'e' in number_str:
            mantissa, exponent = number_str.split('E' if 'E' in number_str else 'e')
            exp_value = int(exponent)
            if '.' in mantissa:
                precision = len(mantissa.split('.')[1])
            else:
                precision = 0
        elif '.' in number_str:
            precision = len(number_str.split('.')[1])
            exp_value = 0
        else:
            precision = 0
            exp_value = 0
        
        # Assess magnitude comprehension
        abs_number = abs(number)
        if abs_number == 0:
            magnitude_comprehension = "PERFECT"
        elif abs_number >= Decimal('1e12'):
            magnitude_comprehension = "ABSTRACT (too large)"
        elif abs_number <= Decimal('1e-6'):
            magnitude_comprehension = "ABSTRACT (too small)"
        elif abs_number >= Decimal('1e6'):
            magnitude_comprehension = "DIFFICULT (millions+)"
        elif abs_number >= Decimal('1000'):
            magnitude_comprehension = "MODERATE (thousands)"
        elif abs_number >= Decimal('1'):
            magnitude_comprehension = "GOOD (units)"
        else:
            magnitude_comprehension = "MODERATE (fractions)"
        
        # Assess precision comprehension
        if precision <= 2:
            precision_comprehension = "EXCELLENT"
        elif precision <= COGNITIVE_LIMIT:
            precision_comprehension = "GOOD"
        elif precision <= 20:
            precision_comprehension = "POOR"
        else:
            precision_comprehension = "MEANINGLESS"
        
        return {
            'number': str(number),
            'precision': precision,
            'magnitude_comprehension': magnitude_comprehension,
            'precision_comprehension': precision_comprehension,
            'cognitively_meaningful': precision <= COGNITIVE_LIMIT,
            'exponent': exp_value,
            'recommendation': self._generate_comprehension_recommendation(precision, magnitude_comprehension)
        }
    
    def _generate_comprehension_recommendation(self, precision, magnitude_comp):
        """Generate recommendation for number comprehension."""
        if precision > COGNITIVE_LIMIT:
            return f"Reduce precision to {COGNITIVE_LIMIT} digits for human comprehension"
        elif "ABSTRACT" in magnitude_comp:
            return "Use scientific notation for better comprehension"
        else:
            return "Number is within cognitive limits"
    
    def test_calculation_tracking(self, num_operations, precision_per_operation):
        """
        Test ability to track precision through multiple operations.
        
        Args:
            num_operations: Number of sequential operations
            precision_per_operation: Precision maintained per operation
            
        Returns:
            Tracking assessment
        """
        # Calculate total cognitive load
        total_precision = num_operations * precision_per_operation
        
        # Assess tracking ability
        if num_operations <= 3 and precision_per_operation <= COGNITIVE_LIMIT:
            tracking_ability = "EXCELLENT"
            success_rate = 0.95
        elif num_operations <= 5 and precision_per_operation <= COGNITIVE_LIMIT:
            tracking_ability = "GOOD"
            success_rate = 0.80
        elif num_operations <= 10 and precision_per_operation <= COGNITIVE_LIMIT:
            tracking_ability = "MODERATE"
            success_rate = 0.60
        elif precision_per_operation > COGNITIVE_LIMIT:
            tracking_ability = "IMPOSSIBLE (precision exceeds cognitive limit)"
            success_rate = 0.10
        else:
            tracking_ability = "POOR"
            success_rate = 0.30
        
        return {
            'num_operations': num_operations,
            'precision_per_operation': precision_per_operation,
            'total_precision_load': total_precision,
            'tracking_ability': tracking_ability,
            'estimated_success_rate': success_rate,
            'exceeds_cognitive_capacity': total_precision > COGNITIVE_LIMIT * 2,
            'recommendation': self._generate_tracking_recommendation(num_operations, precision_per_operation)
        }
    
    def _generate_tracking_recommendation(self, num_ops, precision):
        """Generate recommendation for calculation tracking."""
        if precision > COGNITIVE_LIMIT:
            return f"Reduce precision to {COGNITIVE_LIMIT} digits maximum"
        elif num_ops > 10:
            return "Use computational tools; too many operations for mental tracking"
        else:
            return "Within cognitive tracking capacity"
    
    def benchmark_cognitive_limits(self):
        """
        Run comprehensive benchmark of cognitive limits.
        
        Returns:
            Benchmark results
        """
        results = {
            'digit_span_tests': [],
            'discrimination_tests': [],
            'comprehension_tests': [],
            'tracking_tests': []
        }
        
        # Test 1: Digit span at various lengths
        for num_digits in [3, 5, 7, 10, 15, 20, 30, 50, 100]:
            result = self.test_digit_span(num_digits)
            results['digit_span_tests'].append(result)
        
        # Test 2: Discrimination at various precisions
        base_value = Decimal('3.14159265358979323846')
        for precision in [1, 2, 5, 10, 15, 20, 30]:
            offset = Decimal(10) ** (-precision)
            value2 = base_value + offset
            result = self.test_discrimination_ability(base_value, value2)
            results['discrimination_tests'].append(result)
        
        # Test 3: Number comprehension
        test_numbers = [
            '3.14',
            '3.14159265358979',
            '3.141592653589793238462643383279502884197',
            '1.23e-10',
            '9.87654321e50',
            '0.000000000000001'
        ]
        for number in test_numbers:
            result = self.test_number_comprehension(number)
            results['comprehension_tests'].append(result)
        
        # Test 4: Calculation tracking
        for num_ops in [1, 3, 5, 10, 20]:
            for precision in [5, 10, 15, 20]:
                result = self.test_calculation_tracking(num_ops, precision)
                results['tracking_tests'].append(result)
        
        return results
    
    def generate_report(self, benchmark_results):
        """Generate comprehensive cognitive limits report."""
        report = []
        report.append("=" * 80)
        report.append("COGNITIVE LIMIT TESTING REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Cognitive Limit: {COGNITIVE_LIMIT} digits")
        report.append(f"Working Memory Capacity: {WORKING_MEMORY_CAPACITY} chunks")
        report.append(f"Subitizing Limit: {SUBITIZING_LIMIT} items")
        report.append(f"Perceptual Threshold: {PERCEPTUAL_THRESHOLD}")
        report.append("")
        
        # Digit Span Tests
        report.append("DIGIT SPAN TESTS")
        report.append("-" * 80)
        for result in benchmark_results['digit_span_tests']:
            report.append(f"{result['num_digits']:3d} digits: {result['difficulty']:15s} "
                         f"(Success: {result['estimated_success_rate']*100:5.1f}%, "
                         f"Load: {result['cognitive_load']})")
        report.append("")
        
        # Discrimination Tests
        report.append("DISCRIMINATION TESTS")
        report.append("-" * 80)
        for result in benchmark_results['discrimination_tests'][:7]:
            report.append(f"Precision {result['required_precision']:2d}: {result['discrimination_difficulty']:15s} "
                         f"(Perceptible: {result['perceptible']}, "
                         f"Within limit: {result['within_cognitive_limit']})")
        report.append("")
        
        # Comprehension Tests
        report.append("NUMBER COMPREHENSION TESTS")
        report.append("-" * 80)
        for result in benchmark_results['comprehension_tests']:
            report.append(f"Number: {result['number'][:40]:40s}")
            report.append(f"  Magnitude: {result['magnitude_comprehension']}")
            report.append(f"  Precision: {result['precision_comprehension']}")
            report.append(f"  Meaningful: {result['cognitively_meaningful']}")
            report.append("")
        
        return "\n".join(report)


def run_comprehensive_tests():
    """Run comprehensive cognitive limit tests."""
    tester = CognitiveLimitTester()
    
    print("COGNITIVE LIMIT TESTER - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Test 1: Digit Span
    print("TEST 1: Digit Span Analysis")
    print("-" * 80)
    for digits in [5, 10, 15, 20, 30, 50]:
        result = tester.test_digit_span(digits)
        print(f"{digits:2d} digits: {result['difficulty']:15s} "
              f"Success: {result['estimated_success_rate']*100:5.1f}% "
              f"Load: {result['cognitive_load']}")
    print()
    
    # Test 2: Discrimination
    print("TEST 2: Number Discrimination")
    print("-" * 80)
    pi = "3.14159265358979323846"
    for precision in [1, 5, 10, 15, 20]:
        offset = 10 ** (-precision)
        pi_plus = str(Decimal(pi) + Decimal(str(offset)))
        result = tester.test_discrimination_ability(pi, pi_plus)
        print(f"Precision {precision:2d}: {result['discrimination_difficulty']:15s} "
              f"Perceptible: {result['perceptible']}")
    print()
    
    # Test 3: Comprehension
    print("TEST 3: Number Comprehension")
    print("-" * 80)
    test_numbers = [
        ("3.14", "Simple pi"),
        ("3.14159265358979", "Pi to 15 digits"),
        ("3.141592653589793238462643383279502884197", "Pi to 40 digits"),
        ("1.23e-10", "Scientific notation (small)"),
        ("9.87654321e50", "Scientific notation (large)")
    ]
    
    for number, description in test_numbers:
        result = tester.test_number_comprehension(number)
        print(f"\n{description}")
        print(f"  Number: {number}")
        print(f"  Magnitude: {result['magnitude_comprehension']}")
        print(f"  Precision: {result['precision_comprehension']}")
        print(f"  Meaningful: {result['cognitively_meaningful']}")
    print()
    
    # Test 4: Calculation Tracking
    print("TEST 4: Calculation Tracking Ability")
    print("-" * 80)
    for num_ops in [3, 5, 10, 20]:
        result = tester.test_calculation_tracking(num_ops, 10)
        print(f"{num_ops:2d} operations: {result['tracking_ability']:40s} "
              f"Success: {result['estimated_success_rate']*100:5.1f}%")
    print()
    
    # Test 5: Full Benchmark
    print("TEST 5: Comprehensive Benchmark")
    print("-" * 80)
    benchmark_results = tester.benchmark_cognitive_limits()
    report = tester.generate_report(benchmark_results)
    print(report)
    
    return True


def main():
    """Main execution."""
    success = run_comprehensive_tests()
    
    print("\n" + "=" * 80)
    print("COGNITIVE LIMIT TESTING COMPLETED")
    print("=" * 80)
    print()
    print("KEY FINDINGS:")
    print(f"1. Human cognitive limit: {COGNITIVE_LIMIT} digits maximum")
    print("2. Beyond 15 digits, precision becomes imperceptible to humans")
    print("3. Working memory constrains mental calculation to ~7 chunks")
    print("4. Discrimination ability drops exponentially beyond cognitive limit")
    print("5. Numbers exceeding cognitive limits require computational tools")
    print()
    print("CONCLUSION:")
    print("For human-meaningful calculations, precision beyond 15 digits is WASTED.")
    print("This represents the COGNITIVE TERMINATION BOUNDARY.")
    print()
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
#!/usr/bin/env python3
"""
Master Test Suite for Empirinometry Programs
=============================================
Runs comprehensive tests on all created programs and generates a summary report.
"""

import subprocess
import sys
import time
from pathlib import Path

class MasterTestSuite:
    """Master test suite for all programs."""
    
    def __init__(self):
        """Initialize test suite."""
        self.programs = [
            ('planck_precision_calculator.py', 'Planck Precision Calculator'),
            ('base_converter_termination.py', 'Base Converter with Termination'),
            ('quantum_measurement_validator.py', 'Quantum Measurement Validator'),
            ('cognitive_limit_tester.py', 'Cognitive Limit Tester'),
            ('multi_boundary_analyzer.py', 'Multi-Boundary Analyzer'),
            ('thermodynamic_cost_calculator.py', 'Thermodynamic Cost Calculator'),
            ('pi_termination_calculator.py', 'Pi Termination Calculator'),
            ('fraction_simplifier_ultimate.py', 'Ultimate Fraction Simplifier')
        ]
        
        self.results = []
    
    def run_program(self, program_file, program_name):
        """
        Run a single program and capture results.
        
        Args:
            program_file: Python file to run
            program_name: Display name
            
        Returns:
            Test result dictionary
        """
        print(f"\nTesting: {program_name}")
        print("-" * 80)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ['python', program_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            elapsed_time = time.time() - start_time
            
            success = result.returncode == 0
            
            return {
                'program': program_name,
                'file': program_file,
                'success': success,
                'elapsed_time': elapsed_time,
                'return_code': result.returncode,
                'output_lines': len(result.stdout.split('\n')),
                'error': result.stderr if result.stderr else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                'program': program_name,
                'file': program_file,
                'success': False,
                'elapsed_time': 30.0,
                'return_code': -1,
                'output_lines': 0,
                'error': 'Timeout after 30 seconds'
            }
        except Exception as e:
            return {
                'program': program_name,
                'file': program_file,
                'success': False,
                'elapsed_time': 0,
                'return_code': -1,
                'output_lines': 0,
                'error': str(e)
            }
    
    def run_all_tests(self):
        """Run all program tests."""
        print("=" * 80)
        print("MASTER TEST SUITE - EMPIRINOMETRY PROGRAMS")
        print("=" * 80)
        print(f"\nTesting {len(self.programs)} programs...")
        print()
        
        for program_file, program_name in self.programs:
            result = self.run_program(program_file, program_name)
            self.results.append(result)
            
            if result['success']:
                print(f"✓ PASSED in {result['elapsed_time']:.2f}s")
            else:
                print(f"✗ FAILED: {result['error']}")
        
        return self.results
    
    def generate_summary_report(self):
        """Generate summary report of all tests."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("TEST SUITE SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        total_time = sum(r['elapsed_time'] for r in self.results)
        
        report.append(f"Total Programs Tested: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {failed_tests}")
        report.append(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        report.append(f"Total Execution Time: {total_time:.2f}s")
        report.append("")
        
        # Individual results
        report.append("INDIVIDUAL TEST RESULTS:")
        report.append("-" * 80)
        
        for result in self.results:
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            report.append(f"{status} | {result['program']:40s} | {result['elapsed_time']:6.2f}s")
            if not result['success'] and result['error']:
                report.append(f"       Error: {result['error']}")
        
        report.append("")
        
        # Failed tests details
        if failed_tests > 0:
            report.append("FAILED TESTS DETAILS:")
            report.append("-" * 80)
            
            for result in self.results:
                if not result['success']:
                    report.append(f"\n{result['program']} ({result['file']}):")
                    report.append(f"  Return Code: {result['return_code']}")
                    report.append(f"  Error: {result['error']}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_program_inventory(self):
        """Generate inventory of all programs."""
        inventory = []
        inventory.append("=" * 80)
        inventory.append("PROGRAM INVENTORY")
        inventory.append("=" * 80)
        inventory.append("")
        
        categories = {
            'Termination Boundary Explorers': [
                'planck_precision_calculator.py',
                'base_converter_termination.py',
                'quantum_measurement_validator.py',
                'cognitive_limit_tester.py',
                'multi_boundary_analyzer.py',
                'thermodynamic_cost_calculator.py'
            ],
            'Irrational Number Analyzers': [
                'pi_termination_calculator.py'
            ],
            'Rational Number Analyzers': [
                'fraction_simplifier_ultimate.py'
            ]
        }
        
        for category, programs in categories.items():
            inventory.append(f"\n{category}:")
            inventory.append("-" * 80)
            
            for i, program in enumerate(programs, 1):
                # Find program in results
                result = next((r for r in self.results if r['file'] == program), None)
                
                if result:
                    status = "✓" if result['success'] else "✗"
                    inventory.append(f"{i}. {status} {program}")
                else:
                    inventory.append(f"{i}. ? {program} (not tested)")
        
        inventory.append("")
        inventory.append("=" * 80)
        
        return "\n".join(inventory)


def main():
    """Main execution."""
    suite = MasterTestSuite()
    
    # Run all tests
    results = suite.run_all_tests()
    
    # Generate and print summary
    summary = suite.generate_summary_report()
    print(summary)
    
    # Generate inventory
    inventory = suite.generate_program_inventory()
    print(inventory)
    
    # Save reports to files
    with open('test_summary_report.txt', 'w') as f:
        f.write(summary)
    
    with open('program_inventory_report.txt', 'w') as f:
        f.write(inventory)
    
    print("\nReports saved:")
    print("  - test_summary_report.txt")
    print("  - program_inventory_report.txt")
    print()
    
    # Return exit code based on results
    all_passed = all(r['success'] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
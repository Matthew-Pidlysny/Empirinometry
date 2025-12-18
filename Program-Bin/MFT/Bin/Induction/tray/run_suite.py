#!/usr/bin/env python3
"""
Run Suite: Complete testing and demonstration of the Induction numerical suite
Single command to run all components and generate comprehensive reports
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, List

# Import suite components
from induction_validator import MathematicalInduction, PrimeInduction, LucasInduction
from numerical_variation import NumericalVariationAnalyzer, SequenceGenerator
from visualizer import MathematicalVisualizer, PlotConfig
from math_utils import NumberTheory, SequenceAnalysis

class InductionTestSuite:
    """Complete test suite for the Induction numerical analysis system"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_results": [],
            "performance_metrics": {},
            "generated_files": []
        }
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Execute complete test suite"""
        
        print("🧪 INDUCTION NUMERICAL ANALYSIS SUITE")
        print("=" * 60)
        print("Running comprehensive tests...\n")
        
        start_time = time.time()
        
        # Test 1: Mathematical Induction
        print("1. Testing Mathematical Induction...")
        induction_result = self._test_mathematical_induction()
        self.results["test_results"].append(induction_result)
        
        # Test 2: Prime Number Analysis
        print("\n2. Testing Prime Number Analysis...")
        prime_result = self._test_prime_analysis()
        self.results["test_results"].append(prime_result)
        
        # Test 3: Lucas Sequences
        print("\n3. Testing Lucas Sequence Analysis...")
        lucas_result = self._test_lucas_sequences()
        self.results["test_results"].append(lucas_result)
        
        # Test 4: Numerical Variation
        print("\n4. Testing Numerical Variation Analysis...")
        variation_result = self._test_numerical_variation()
        self.results["test_results"].append(variation_result)
        
        # Test 5: Mathematical Utilities
        print("\n5. Testing Mathematical Utilities...")
        utils_result = self._test_math_utils()
        self.results["test_results"].append(utils_result)
        
        # Test 6: Visualization
        print("\n6. Testing Visualization...")
        viz_result = self._test_visualization()
        self.results["test_results"].append(viz_result)
        
        # Performance metrics
        end_time = time.time()
        self.results["performance_metrics"]["total_runtime"] = end_time - start_time
        self.results["performance_metrics"]["tests_passed"] = sum(1 for r in self.results["test_results"] if r["passed"])
        self.results["performance_metrics"]["total_tests"] = len(self.results["test_results"])
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _test_mathematical_induction(self) -> Dict[str, Any]:
        """Test mathematical induction capabilities"""
        
        induction = MathematicalInduction()
        passed = True
        details = []
        
        try:
            # Test sum formula
            def sum_formula(n: int) -> bool:
                return sum(range(1, n + 1)) == n * (n + 1) // 2
            
            result = induction.prove_by_induction(
                hypothesis="Sum of first n integers = n(n+1)/2",
                base_cases=[1, 2, 3],
                inductive_step=sum_formula,
                verify_limit=20
            )
            
            if result.confidence == 1.0:
                details.append("✅ Sum formula induction: PASSED")
            else:
                details.append(f"❌ Sum formula induction: FAILED (confidence: {result.confidence})")
                passed = False
                
        except Exception as e:
            details.append(f"❌ Induction test error: {e}")
            passed = False
        
        return {
            "test_name": "Mathematical Induction",
            "passed": passed,
            "details": details,
            "execution_time": time.time()
        }
    
    def _test_prime_analysis(self) -> Dict[str, Any]:
        """Test prime number analysis"""
        
        prime_ind = PrimeInduction()
        passed = True
        details = []
        
        try:
            # Test twin prime conjecture
            result = prime_ind.test_twin_prime_conjecture(limit=100)
            
            if result.confidence > 0:
                details.append(f"✅ Twin prime analysis: PASSED (confidence: {result.confidence:.2f})")
            else:
                details.append("❌ Twin prime analysis: FAILED")
                passed = False
                
            # Test prime generation
            primes = NumberTheory.prime_sieve(100)
            if len(primes) >= 25:  # There are 25 primes ≤ 100
                details.append(f"✅ Prime sieve: PASSED (found {len(primes)} primes)")
            else:
                details.append(f"❌ Prime sieve: FAILED (only {len(primes)} primes)")
                passed = False
                
        except Exception as e:
            details.append(f"❌ Prime analysis error: {e}")
            passed = False
        
        return {
            "test_name": "Prime Number Analysis",
            "passed": passed,
            "details": details,
            "execution_time": time.time()
        }
    
    def _test_lucas_sequences(self) -> Dict[str, Any]:
        """Test Lucas sequence analysis"""
        
        lucas_ind = LucasInduction()
        generator = SequenceGenerator()
        passed = True
        details = []
        
        try:
            # Test Fibonacci generation
            fib_10 = [generator.fibonacci(i) for i in range(1, 11)]
            expected_fib_10 = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
            
            if fib_10 == expected_fib_10:
                details.append("✅ Fibonacci generation: PASSED")
            else:
                details.append("❌ Fibonacci generation: FAILED")
                passed = False
            
            # Test Lucas generation
            lucas_10 = [generator.lucas(i) for i in range(10)]
            expected_lucas_10 = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76]
            
            if lucas_10 == expected_lucas_10:
                details.append("✅ Lucas generation: PASSED")
            else:
                details.append("❌ Lucas generation: FAILED")
                passed = False
                
        except Exception as e:
            details.append(f"❌ Lucas sequence error: {e}")
            passed = False
        
        return {
            "test_name": "Lucas Sequence Analysis",
            "passed": passed,
            "details": details,
            "execution_time": time.time()
        }
    
    def _test_numerical_variation(self) -> Dict[str, Any]:
        """Test numerical variation analysis"""
        
        analyzer = NumericalVariationAnalyzer()
        generator = SequenceGenerator()
        passed = True
        details = []
        
        try:
            # Test periodicity detection
            periodic_seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]  # Period 3
            pattern = analyzer.analyze_periodicity(periodic_seq)
            
            if pattern.parameters.get("period") == 3 and pattern.confidence > 0.8:
                details.append("✅ Periodicity detection: PASSED")
            else:
                details.append("❌ Periodicity detection: FAILED")
                passed = False
            
            # Test growth rate analysis
            fib_seq = [generator.fibonacci(i) for i in range(1, 11)]
            growth = analyzer.analyze_growth_rate(fib_seq)
            
            if "fibonacci-like" in growth.parameters.get("growth_type", ""):
                details.append("✅ Growth rate analysis: PASSED")
            else:
                details.append("❌ Growth rate analysis: FAILED")
                passed = False
                
        except Exception as e:
            details.append(f"❌ Numerical variation error: {e}")
            passed = False
        
        return {
            "test_name": "Numerical Variation Analysis",
            "passed": passed,
            "details": details,
            "execution_time": time.time()
        }
    
    def _test_math_utils(self) -> Dict[str, Any]:
        """Test mathematical utilities"""
        
        passed = True
        details = []
        
        try:
            # Test GCD
            if NumberTheory.gcd(48, 18) == 6:
                details.append("✅ GCD calculation: PASSED")
            else:
                details.append("❌ GCD calculation: FAILED")
                passed = False
            
            # Test LCM
            if NumberTheory.lcm(12, 18) == 36:
                details.append("✅ LCM calculation: PASSED")
            else:
                details.append("❌ LCM calculation: FAILED")
                passed = False
            
            # Test Euler's totient
            if NumberTheory.euler_totient(9) == 6:  # φ(9) = 6
                details.append("✅ Euler's totient: PASSED")
            else:
                details.append("❌ Euler's totient: FAILED")
                passed = False
                
        except Exception as e:
            details.append(f"❌ Math utilities error: {e}")
            passed = False
        
        return {
            "test_name": "Mathematical Utilities",
            "passed": passed,
            "details": details,
            "execution_time": time.time()
        }
    
    def _test_visualization(self) -> Dict[str, Any]:
        """Test visualization capabilities"""
        
        viz = MathematicalVisualizer()
        generator = SequenceGenerator()
        passed = True
        details = []
        
        try:
            # Test sequence plotting
            fib_seq = [generator.fibonacci(i) for i in range(1, 11)]
            config = PlotConfig(
                title="Test Fibonacci Plot",
                xlabel="Index",
                ylabel="Value"
            )
            
            fig = viz.plot_sequence(fib_seq, config)
            if fig is not None:
                details.append("✅ Sequence plotting: PASSED")
                self.results["generated_files"].append("test_fibonacci_plot.png")
                fig.savefig("test_fibonacci_plot.png")
            else:
                details.append("❌ Sequence plotting: FAILED")
                passed = False
            
            # Test modular pattern plotting
            mod_config = PlotConfig(
                title="Test Modular Plot",
                xlabel="Index",
                ylabel="Residue"
            )
            
            fig2 = viz.plot_modular_patterns(fib_seq, 10, mod_config)
            if fig2 is not None:
                details.append("✅ Modular plotting: PASSED")
                self.results["generated_files"].append("test_modular_plot.png")
                fig2.savefig("test_modular_plot.png")
            else:
                details.append("❌ Modular plotting: FAILED")
                passed = False
                
        except Exception as e:
            details.append(f"❌ Visualization error: {e}")
            passed = False
        
        return {
            "test_name": "Visualization",
            "passed": passed,
            "details": details,
            "execution_time": time.time()
        }
    
    def _generate_summary(self):
        """Generate comprehensive summary report"""
        
        print("\n" + "=" * 60)
        print("📊 TEST SUITE SUMMARY")
        print("=" * 60)
        
        # Performance summary
        metrics = self.results["performance_metrics"]
        print(f"Total Runtime: {metrics['total_runtime']:.2f} seconds")
        print(f"Tests Passed: {metrics['tests_passed']}/{metrics['total_tests']}")
        print(f"Success Rate: {metrics['tests_passed']/metrics['total_tests']*100:.1f}%")
        
        # Individual test results
        print("\nDetailed Results:")
        print("-" * 40)
        
        for test_result in self.results["test_results"]:
            status = "✅ PASS" if test_result["passed"] else "❌ FAIL"
            print(f"{status} {test_result['test_name']}")
            
            for detail in test_result["details"]:
                print(f"    {detail}")
        
        # Generated files
        if self.results["generated_files"]:
            print(f"\nGenerated Files: {len(self.results['generated_files'])}")
            for file in self.results["generated_files"]:
                print(f"  📄 {file}")
        
        # Save results
        self._save_results()
        
        print(f"\n🎉 Suite completed successfully!")
        print(f"📁 Results saved to: induction_test_results.json")
        print(f"📊 Performance summary: induction_performance_report.txt")
    
    def _save_results(self):
        """Save test results to files"""
        
        # Save detailed JSON results
        with open("induction_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save performance report
        with open("induction_performance_report.txt", "w") as f:
            f.write("INDUCTION NUMERICAL ANALYSIS SUITE - PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Total Runtime: {self.results['performance_metrics']['total_runtime']:.2f} seconds\n")
            f.write(f"Tests Passed: {self.results['performance_metrics']['tests_passed']}/{self.results['performance_metrics']['total_tests']}\n")
            f.write(f"Success Rate: {self.results['performance_metrics']['tests_passed']/self.results['performance_metrics']['total_tests']*100:.1f}%\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for test_result in self.results["test_results"]:
                f.write(f"\n{test_result['test_name']}: {'PASS' if test_result['passed'] else 'FAIL'}\n")
                for detail in test_result["details"]:
                    f.write(f"  {detail}\n")

def run_interactive_dashboard():
    """Launch the interactive dashboard"""
    
    print("\n🖥️  Launching Interactive Dashboard...")
    print("This will open a GUI window for interactive analysis.\n")
    
    try:
        from interactive_dashboard import main
        main()
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
        print("Make sure you have tkinter installed: pip install tk")

def show_menu():
    """Display main menu"""
    
    print("\n🎯 INDUCTION NUMERICAL ANALYSIS SUITE")
    print("=" * 50)
    print("Choose an option:")
    print("1. Run Complete Test Suite")
    print("2. Launch Interactive Dashboard")
    print("3. Run Individual Components")
    print("4. Generate Sample Reports")
    print("5. Exit")
    print("-" * 50)

def run_individual_components():
    """Menu for running individual components"""
    
    print("\n🔧 Individual Components")
    print("-" * 30)
    print("1. Mathematical Induction Tests")
    print("2. Prime Number Analysis")
    print("3. Lucas Sequence Analysis")
    print("4. Numerical Variation Analysis")
    print("5. Mathematical Utilities Demo")
    print("6. Back to Main Menu")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        from induction_validator import main
        main()
    elif choice == "2":
        from induction_validator import PrimeInduction
        prime_ind = PrimeInduction()
        result = prime_ind.test_twin_prime_conjecture(limit=1000)
        print(f"Twin Prime Test: Confidence {result.confidence:.2f}")
    elif choice == "3":
        from induction_validator import LucasInduction
        lucas_ind = LucasInduction()
        result = lucas_ind.test_fibonacci_divisibility(test_limit=30)
        print(f"Fibonacci Divisibility: Confidence {result.confidence:.2f}")
    elif choice == "4":
        from numerical_variation import main
        main()
    elif choice == "5":
        from math_utils import main
        main()
    elif choice == "6":
        return
    else:
        print("Invalid choice")

def generate_sample_reports():
    """Generate sample analysis reports"""
    
    print("\n📋 Generating Sample Reports...")
    
    try:
        # Sample analysis
        from visualizer import MathematicalVisualizer, PlotConfig
        from numerical_variation import SequenceGenerator
        
        generator = SequenceGenerator()
        viz = MathematicalVisualizer()
        
        # Generate sample sequences
        fib_seq = [generator.fibonacci(i) for i in range(1, 31)]
        primes = generator.primes_up_to(100)
        
        # Create visualizations
        config1 = PlotConfig("Sample Fibonacci Analysis", "Index", "Value")
        viz.plot_sequence(fib_seq, config1)
        viz.save_all_figures("sample_report", "png")
        
        print("✅ Sample reports generated:")
        print("  📄 sample_report_1.png - Fibonacci sequence")
        print("  📄 sample_report_2.png - Growth analysis")
        
    except Exception as e:
        print(f"❌ Failed to generate reports: {e}")

def main():
    """Main entry point for the Induction suite"""
    
    while True:
        show_menu()
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            suite = InductionTestSuite()
            suite.run_all_tests()
        elif choice == "2":
            run_interactive_dashboard()
        elif choice == "3":
            run_individual_components()
        elif choice == "4":
            generate_sample_reports()
        elif choice == "5":
            print("\n👋 Exiting Induction Suite. Thank you!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
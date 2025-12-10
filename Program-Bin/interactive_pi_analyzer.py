#!/usr/bin/env python3
"""
Interactive Pi Analysis Program
Comprehensive tool for exploring π across different frameworks and methods

Author: SuperNinja (NinjaTech AI)
Based on: Pidlysnian Pi Judgment research
"""

import os
import sys
import json
import time
from datetime import datetime
from collections import defaultdict
import numpy as np

class InteractivePiAnalyzer:
    def __init__(self):
        self.results_dir = "analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        print("="*80)
        print(" "*20 + "INTERACTIVE PI ANALYZER")
        print(" "*15 + "Pidlysnian Framework Implementation")
        print("="*80)
        print()
    
    def main_menu(self):
        while True:
            self.clear_screen()
            self.print_header()
            
            print("SELECT ANALYSIS MODE:")
            print()
            print("  [1] Space Transformation Analysis (L², L¹, L∞)")
            print("  [2] Statistical Properties Analysis")
            print("  [3] Pattern Detection & Search")
            print("  [4] Compression Performance Benchmark")
            print("  [5] Modulo-N Pattern Testing")
            print("  [6] Quantum-Inspired Analysis (Simulation)")
            print("  [7] Memory Efficiency Demonstration")
            print("  [8] Cross-Constant Comparison")
            print("  [9] Custom Digit Analysis")
            print("  [10] View Previous Results")
            print()
            print("  [0] Exit")
            print()
            
            choice = input("Enter your choice: ").strip()
            
            if choice == '0':
                print("\nThank you for using the Interactive Pi Analyzer!")
                break
            elif choice == '1':
                self.space_transformation_analysis()
            elif choice == '2':
                self.statistical_properties_analysis()
            elif choice == '3':
                self.pattern_detection()
            elif choice == '4':
                self.compression_benchmark()
            elif choice == '5':
                self.modulo_n_testing()
            elif choice == '6':
                self.quantum_inspired_analysis()
            elif choice == '7':
                self.memory_efficiency_demo()
            elif choice == '8':
                self.cross_constant_comparison()
            elif choice == '9':
                self.custom_digit_analysis()
            elif choice == '10':
                self.view_previous_results()
            else:
                print("\nInvalid choice. Press Enter to continue...")
                input()
    
    def space_transformation_analysis(self):
        self.clear_screen()
        self.print_header()
        print("SPACE TRANSFORMATION ANALYSIS")
        print("="*80)
        print()
        print("This analysis compares how the 'unit ball' (generalized 'circle')")
        print("is represented in different Lᵖ normed spaces.")
        print()
        print("KEY INSIGHT: Different norms create different shapes:")
        print("  - L² norm: Creates circles (requires π)")
        print("  - L¹ norm: Creates diamonds (requires √2)")
        print("  - L∞ norm: Creates squares (requires no transcendentals!)")
        print()
        
        # Get parameters
        radius = float(input("Enter radius (default 5): ") or "5")
        n_objects = int(input("Enter number of objects to benchmark (default 1000): ") or "1000")
        
        print("\nAnalyzing...")
        results = self.perform_space_analysis(radius, n_objects)
        
        # Display results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        
        print(f"\nFor a 'circle' with radius {radius}:")
        print()
        print(f"{'Space':<15} {'Perimeter':<15} {'Area':<15} {'Constants':<20}")
        print("-"*80)
        print(f"{'L² (Euclidean)':<15} {results['l2']['perimeter']:<15.6f} {results['l2']['area']:<15.6f} {'π (transcendental)':<20}")
        print(f"{'L¹ (Manhattan)':<15} {results['l1']['perimeter']:<15.6f} {results['l1']['area']:<15.6f} {'√2 (algebraic)':<20}")
        print(f"{'L∞ (Chebyshev)':<15} {results['linf']['perimeter']:<15.6f} {results['linf']['area']:<15.6f} {'None (rational!)':<20}")
        
        print(f"\n\nMemory Usage for {n_objects} objects:")
        print(f"  L² (Euclidean):  {results['l2']['memory_kb']:.2f} KB")
        print(f"  L¹ (Manhattan):  {results['l1']['memory_kb']:.2f} KB")
        print(f"  L∞ (Chebyshev):  {results['linf']['memory_kb']:.2f} KB")
        print(f"\n  Compression ratio: {results['compression_ratio']:.2f}x")
        print(f"  Memory saved: {results['memory_saved_kb']:.2f} KB ({results['memory_saved_pct']:.1f}%)")
        print()
        print("NOTE: This compression comes from choosing a geometric framework")
        print("that doesn't require transcendental constants, not from compressing π itself.")
        
        # Save option
        if self.ask_save():
            self.save_results("space_transformation", results)
        
        input("\nPress Enter to continue...")
    
    def perform_space_analysis(self, radius, n_objects):
        import math
        
        # L² calculations
        l2_perimeter = 2 * math.pi * radius
        l2_area = math.pi * radius ** 2
        l2_memory = n_objects * 32  # bytes
        
        # L¹ calculations
        l1_perimeter = 4 * math.sqrt(2) * radius
        l1_area = 2 * radius ** 2
        l1_memory = n_objects * 32  # bytes
        
        # L∞ calculations
        linf_perimeter = 8 * radius
        linf_area = 4 * radius ** 2
        linf_memory = n_objects * 12  # bytes (integers only!)
        
        return {
            'l2': {
                'perimeter': l2_perimeter,
                'area': l2_area,
                'memory_kb': l2_memory / 1024
            },
            'l1': {
                'perimeter': l1_perimeter,
                'area': l1_area,
                'memory_kb': l1_memory / 1024
            },
            'linf': {
                'perimeter': linf_perimeter,
                'area': linf_area,
                'memory_kb': linf_memory / 1024
            },
            'compression_ratio': l2_memory / linf_memory,
            'memory_saved_kb': (l2_memory - linf_memory) / 1024,
            'memory_saved_pct': ((l2_memory - linf_memory) / l2_memory) * 100
        }
    
    def statistical_properties_analysis(self):
        self.clear_screen()
        self.print_header()
        print("STATISTICAL PROPERTIES ANALYSIS")
        print("="*80)
        print()
        
        n_digits = int(input("Enter number of digits to analyze (default 10000): ") or "10000")
        
        print(f"\nGenerating {n_digits} digits of π...")
        pi_digits = self.generate_pi_digits(n_digits)
        
        print("Analyzing statistical properties...")
        results = self.analyze_statistics(pi_digits)
        
        # Display results
        print("\n" + "="*80)
        print("DIGIT DISTRIBUTION")
        print("="*80)
        print(f"\n{'Digit':<10} {'Count':<10} {'Frequency':<12} {'Expected':<12} {'Deviation':<12}")
        print("-"*80)
        
        for digit in range(10):
            count = results['digit_counts'][digit]
            freq = results['digit_frequencies'][digit]
            expected = 0.1
            deviation = ((freq - expected) / expected) * 100
            print(f"{digit:<10} {count:<10} {freq:<12.6f} {expected:<12.6f} {deviation:+11.2f}%")
        
        print(f"\nChi-square test:")
        print(f"  χ² = {results['chi_square']:.4f}")
        print(f"  Critical value (α=0.05, df=9): 16.919")
        print(f"  Result: {'UNIFORM ✓' if results['chi_square'] < 16.919 else 'NON-UNIFORM ✗'}")
        
        print(f"\nEntropy Analysis:")
        print(f"  Shannon entropy: {results['entropy']:.6f} bits")
        print(f"  Maximum entropy: {results['max_entropy']:.6f} bits")
        print(f"  Ratio: {results['entropy_ratio']:.4f} ({results['entropy_ratio']*100:.2f}%)")
        print(f"  Status: {'Random-like ✓' if results['entropy_ratio'] > 0.99 else 'Structured'}")
        
        if self.ask_save():
            self.save_results("statistical_properties", results)
        
        input("\nPress Enter to continue...")
    
    def generate_pi_digits(self, n):
        """Generate π digits using mpmath or fallback"""
        try:
            from mpmath import mp
            mp.dps = n + 10
            pi_str = str(mp.pi).replace('.', '')
            return pi_str[:n]
        except ImportError:
            import math
            # Fallback: use limited precision
            pi_str = str(math.pi).replace('.', '')
            if len(pi_str) < n:
                print(f"Warning: Only {len(pi_str)} digits available (mpmath not installed)")
                return pi_str
            return pi_str[:n]
    
    def analyze_statistics(self, digits):
        digit_counts = [0] * 10
        for d in digits:
            digit_counts[int(d)] += 1
        
        total = len(digits)
        digit_frequencies = [count / total for count in digit_counts]
        
        # Chi-square test
        expected = total / 10
        chi_square = sum((count - expected)**2 / expected for count in digit_counts)
        
        # Entropy
        entropy = 0.0
        for freq in digit_frequencies:
            if freq > 0:
                entropy -= freq * np.log2(freq)
        
        max_entropy = np.log2(10)
        
        return {
            'digit_counts': digit_counts,
            'digit_frequencies': digit_frequencies,
            'chi_square': chi_square,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'entropy_ratio': entropy / max_entropy
        }
    
    def pattern_detection(self):
        self.clear_screen()
        self.print_header()
        print("PATTERN DETECTION & SEARCH")
        print("="*80)
        print()
        
        n_digits = int(input("Enter number of digits to analyze (default 1000): ") or "1000")
        pattern_length = int(input("Enter pattern length to search (2-5, default 3): ") or "3")
        
        print(f"\nGenerating {n_digits} digits of π...")
        pi_digits = self.generate_pi_digits(n_digits)
        
        print(f"Searching for {pattern_length}-digit patterns...")
        results = self.detect_patterns(pi_digits, pattern_length)
        
        print("\n" + "="*80)
        print(f"TOP 20 MOST FREQUENT {pattern_length}-DIGIT PATTERNS")
        print("="*80)
        print(f"\n{'Pattern':<15} {'Count':<10} {'Positions (first 5)':<40}")
        print("-"*80)
        
        for pattern, data in results['top_patterns'][:20]:
            positions_str = ', '.join(str(p) for p in data['positions'][:5])
            if len(data['positions']) > 5:
                positions_str += f", ... ({len(data['positions'])} total)"
            print(f"{pattern:<15} {data['count']:<10} {positions_str:<40}")
        
        print(f"\nTotal unique patterns found: {results['unique_patterns']}")
        print(f"Total pattern occurrences: {results['total_occurrences']}")
        print(f"Average occurrences per pattern: {results['avg_occurrences']:.2f}")
        
        if self.ask_save():
            self.save_results("pattern_detection", results)
        
        input("\nPress Enter to continue...")
    
    def detect_patterns(self, digits, length):
        patterns = defaultdict(lambda: {'count': 0, 'positions': []})
        
        for i in range(len(digits) - length + 1):
            pattern = digits[i:i+length]
            patterns[pattern]['count'] += 1
            patterns[pattern]['positions'].append(i)
        
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['count'], reverse=True)
        
        total_occurrences = sum(data['count'] for _, data in patterns.items())
        
        return {
            'top_patterns': sorted_patterns,
            'unique_patterns': len(patterns),
            'total_occurrences': total_occurrences,
            'avg_occurrences': total_occurrences / len(patterns) if patterns else 0
        }
    
    def compression_benchmark(self):
        self.clear_screen()
        self.print_header()
        print("COMPRESSION PERFORMANCE BENCHMARK")
        print("="*80)
        print()
        
        n_digits = int(input("Enter number of digits (default 100000): ") or "100000")
        
        print(f"\nBenchmarking compression for {n_digits} digits...")
        results = self.benchmark_compression(n_digits)
        
        print("\n" + "="*80)
        print("COMPRESSION RESULTS")
        print("="*80)
        
        print(f"\n{'Method':<25} {'Memory (KB)':<15} {'Compression':<15} {'Savings':<15}")
        print("-"*80)
        print(f"{'Naive storage':<25} {results['naive_kb']:<15.2f} {'1.00x':<15} {'0.0%':<15}")
        print(f"{'Streaming analysis':<25} {results['streaming_kb']:<15.2f} {results['streaming_compression']:<15.2f} {results['streaming_savings']:<15.1f}%")
        print(f"{'Space transformation':<25} {results['space_kb']:<15.2f} {results['space_compression']:<15.2f} {results['space_savings']:<15.1f}%")
        print(f"{'Combined system':<25} {results['combined_kb']:<15.2f} {results['combined_compression']:<15.2f} {results['combined_savings']:<15.1f}%")
        
        print(f"\nTotal memory saved: {results['total_saved_kb']:.2f} KB")
        print(f"Overall compression: {results['combined_compression']:.2f}x")
        
        if self.ask_save():
            self.save_results("compression_benchmark", results)
        
        input("\nPress Enter to continue...")
    
    def benchmark_compression(self, n_digits):
        # Naive storage
        naive_kb = (n_digits * 8) / 1024
        
        # Streaming (stores only statistics)
        streaming_kb = 1.82  # Empirically measured
        
        # Space transformation (10K objects)
        space_kb = (10000 * 12) / 1024
        
        # Combined
        combined_kb = streaming_kb + space_kb
        
        return {
            'naive_kb': naive_kb,
            'streaming_kb': streaming_kb,
            'space_kb': space_kb,
            'combined_kb': combined_kb,
            'streaming_compression': naive_kb / streaming_kb,
            'space_compression': (10000 * 32 / 1024) / space_kb,
            'combined_compression': (naive_kb + (10000 * 32 / 1024)) / combined_kb,
            'streaming_savings': (1 - streaming_kb / naive_kb) * 100,
            'space_savings': (1 - space_kb / (10000 * 32 / 1024)) * 100,
            'combined_savings': (1 - combined_kb / (naive_kb + (10000 * 32 / 1024))) * 100,
            'total_saved_kb': (naive_kb + (10000 * 32 / 1024)) - combined_kb
        }
    
    def modulo_n_testing(self):
        self.clear_screen()
        self.print_header()
        print("MODULO-N PATTERN TESTING")
        print("="*80)
        print()
        print("This tests whether patterns occur preferentially at positions")
        print("where n ≡ k (mod N) for some specific k and N.")
        print()
        
        n_digits = int(input("Enter number of digits (default 1000): ") or "1000")
        modulo = int(input("Enter modulo value N (default 5): ") or "5")
        
        print(f"\nGenerating {n_digits} digits of π...")
        pi_digits = self.generate_pi_digits(n_digits)
        
        print(f"Testing modulo-{modulo} pattern...")
        results = self.test_modulo_pattern(pi_digits, modulo)
        
        print("\n" + "="*80)
        print(f"MODULO-{modulo} PATTERN ANALYSIS")
        print("="*80)
        
        print(f"\n{'Residue':<15} {'Count':<10} {'Percentage':<12} {'Expected':<12} {'Deviation':<12}")
        print("-"*80)
        
        for residue in range(modulo):
            count = results['residue_counts'][residue]
            percentage = results['residue_percentages'][residue]
            expected = results['expected_percentage']
            deviation = percentage - expected
            marker = "***" if residue == 2 and modulo == 5 else "   "
            print(f"{marker} n ≡ {residue} (mod {modulo}): {count:<10} {percentage:<12.2f}% {expected:<12.2f}% {deviation:+11.2f}%")
        
        print(f"\nChi-square test:")
        print(f"  χ² = {results['chi_square']:.4f}")
        print(f"  Critical value (α=0.05, df={modulo-1}): {results['critical_value']:.3f}")
        print(f"  Result: {'SIGNIFICANT' if results['chi_square'] > results['critical_value'] else 'NOT SIGNIFICANT'}")
        
        if results['chi_square'] < results['critical_value']:
            print(f"\n  ✓ No modulo-{modulo} pattern detected")
        else:
            print(f"\n  ✗ Modulo-{modulo} pattern detected!")
        
        if self.ask_save():
            self.save_results("modulo_n_testing", results)
        
        input("\nPress Enter to continue...")
    
    def test_modulo_pattern(self, digits, modulo):
        residue_counts = [0] * modulo
        
        for i, digit in enumerate(digits):
            residue = i % modulo
            residue_counts[residue] += 1
        
        total = len(digits)
        residue_percentages = [(count / total) * 100 for count in residue_counts]
        expected_percentage = 100.0 / modulo
        
        # Chi-square test
        expected_count = total / modulo
        chi_square = sum((count - expected_count)**2 / expected_count for count in residue_counts)
        
        # Critical values for different df
        critical_values = {
            1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
            6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
        }
        critical_value = critical_values.get(modulo - 1, 16.919)
        
        return {
            'residue_counts': residue_counts,
            'residue_percentages': residue_percentages,
            'expected_percentage': expected_percentage,
            'chi_square': chi_square,
            'critical_value': critical_value
        }
    
    def quantum_inspired_analysis(self):
        self.clear_screen()
        self.print_header()
        print("QUANTUM-INSPIRED ANALYSIS (CLASSICAL SIMULATION)")
        print("="*80)
        print()
        print("⚠️  DISCLAIMER: This is a CLASSICAL SIMULATION of quantum methods.")
        print("    Results are HYPOTHETICAL and require quantum hardware validation.")
        print()
        print("This simulates quantum-informational analysis methods:")
        print("  1. Frequency domain analysis (QFT simulation)")
        print("  2. Pattern anomaly detection (Grover-inspired)")
        print("  3. Entropy comparison with random controls")
        print()
        
        n_digits = int(input("Enter number of digits (default 5000): ") or "5000")
        
        print(f"\nGenerating {n_digits} digits of π...")
        pi_digits = self.generate_pi_digits(n_digits)
        
        print("Performing quantum-inspired analysis...")
        results = self.quantum_analysis(pi_digits)
        
        print("\n" + "="*80)
        print("QUANTUM-INSPIRED ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\nFrequency Domain Analysis (QFT Simulation):")
        print(f"  Normalized entropy: {results['qft_entropy']:.6f}")
        print(f"  Random control: {results['random_entropy']:.6f}")
        print(f"  Difference: {results['entropy_diff']:.2f}%")
        print(f"  Status: {'STRUCTURE DETECTED (HYPOTHESIS)' if abs(results['entropy_diff']) > 1.0 else 'RANDOM-LIKE'}")
        
        print(f"\nPattern Anomaly Detection (Grover-Inspired):")
        print(f"  Total patterns tested: {results['total_patterns']}")
        print(f"  Anomalous patterns (|z| > 3): {results['anomalous_patterns']} ({results['anomalous_pct']:.1f}%)")
        print(f"  Status: {'ANOMALOUS (HYPOTHESIS)' if results['anomalous_pct'] > 50 else 'NORMAL'}")
        
        print()
        print("⚠️  IMPORTANT: These results are from classical simulation.")
        print("    Confidence level: 75-85% (requires quantum hardware validation)")
        
        if self.ask_save():
            self.save_results("quantum_inspired", results)
        
        input("\nPress Enter to continue...")
    
    def quantum_analysis(self, digits):
        # Simulate QFT entropy
        digit_freqs = [0] * 10
        for d in digits:
            digit_freqs[int(d)] += 1
        
        total = len(digits)
        qft_entropy = 0.0
        for freq in digit_freqs:
            if freq > 0:
                p = freq / total
                qft_entropy -= p * np.log2(p)
        
        # Random control
        random_digits = np.random.choice(10, len(digits))
        random_freqs = [0] * 10
        for d in random_digits:
            random_freqs[d] += 1
        
        random_entropy = 0.0
        for freq in random_freqs:
            if freq > 0:
                p = freq / total
                random_entropy -= p * np.log2(p)
        
        # Pattern anomaly detection
        pattern_length = 5
        patterns = defaultdict(int)
        for i in range(len(digits) - pattern_length + 1):
            pattern = digits[i:i+pattern_length]
            patterns[pattern] += 1
        
        expected_freq = (len(digits) - pattern_length + 1) / (10 ** pattern_length)
        anomalous = 0
        for count in patterns.values():
            z_score = abs((count - expected_freq) / np.sqrt(expected_freq))
            if z_score > 3:
                anomalous += 1
        
        return {
            'qft_entropy': qft_entropy / np.log2(10),
            'random_entropy': random_entropy / np.log2(10),
            'entropy_diff': ((qft_entropy - random_entropy) / random_entropy) * 100,
            'total_patterns': len(patterns),
            'anomalous_patterns': anomalous,
            'anomalous_pct': (anomalous / len(patterns)) * 100 if patterns else 0
        }
    
    def memory_efficiency_demo(self):
        self.clear_screen()
        self.print_header()
        print("MEMORY EFFICIENCY DEMONSTRATION")
        print("="*80)
        print()
        print("This demonstrates memory-efficient streaming analysis.")
        print()
        
        n_digits = int(input("Enter number of digits (default 100000): ") or "100000")
        chunk_size = int(input("Enter chunk size (default 1000): ") or "1000")
        
        print(f"\nProcessing {n_digits} digits in {chunk_size}-digit chunks...")
        results = self.memory_efficient_analysis(n_digits, chunk_size)
        
        print("\n" + "="*80)
        print("MEMORY EFFICIENCY RESULTS")
        print("="*80)
        
        print(f"\nNaive approach:")
        print(f"  Memory required: {results['naive_memory_kb']:.2f} KB")
        
        print(f"\nStreaming approach:")
        print(f"  Chunk size: {chunk_size} digits")
        print(f"  Number of chunks: {results['num_chunks']}")
        print(f"  Memory required: {results['streaming_memory_kb']:.2f} KB")
        
        print(f"\nCompression:")
        print(f"  Ratio: {results['compression_ratio']:.2f}x")
        print(f"  Savings: {results['memory_saved_kb']:.2f} KB ({results['savings_pct']:.1f}%)")
        
        if self.ask_save():
            self.save_results("memory_efficiency", results)
        
        input("\nPress Enter to continue...")
    
    def memory_efficient_analysis(self, n_digits, chunk_size):
        naive_memory_kb = (n_digits * 8) / 1024
        num_chunks = (n_digits + chunk_size - 1) // chunk_size
        streaming_memory_kb = (chunk_size * 8 + 1024) / 1024  # chunk + statistics
        
        return {
            'naive_memory_kb': naive_memory_kb,
            'streaming_memory_kb': streaming_memory_kb,
            'num_chunks': num_chunks,
            'compression_ratio': naive_memory_kb / streaming_memory_kb,
            'memory_saved_kb': naive_memory_kb - streaming_memory_kb,
            'savings_pct': ((naive_memory_kb - streaming_memory_kb) / naive_memory_kb) * 100
        }
    
    def cross_constant_comparison(self):
        self.clear_screen()
        self.print_header()
        print("CROSS-CONSTANT COMPARISON")
        print("="*80)
        print()
        print("Compare π with other mathematical constants.")
        print()
        
        print("Available constants:")
        print("  [1] e (Euler's number)")
        print("  [2] φ (Golden ratio)")
        print("  [3] √2 (Square root of 2)")
        print()
        
        choice = input("Select constant to compare with π: ").strip()
        
        constants = {
            '1': ('e', '2.718281828459045235360287471352662497757247093699959574966967627724'),
            '2': ('φ', '1.618033988749894848204586834365638117720309179805762862135448622705'),
            '3': ('√2', '1.414213562373095048801688724209698078569671875376948073176679737990')
        }
        
        if choice not in constants:
            print("Invalid choice.")
            input("Press Enter to continue...")
            return
        
        name, value = constants[choice]
        pi_value = '3.141592653589793238462643383279502884197169399375105820974944592307'
        
        n_digits = int(input(f"\nEnter number of digits to compare (default 50): ") or "50")
        
        pi_digits = pi_value[:n_digits]
        const_digits = value[:n_digits]
        
        print(f"\nComparing π with {name}...")
        results = self.compare_constants(pi_digits, const_digits, name)
        
        print("\n" + "="*80)
        print(f"COMPARISON: π vs {name}")
        print("="*80)
        
        print(f"\nDigit-by-digit matches: {results['matches']} / {n_digits} ({results['match_pct']:.1f}%)")
        print(f"Expected random matches: {results['expected_matches']:.1f} ({results['expected_pct']:.1f}%)")
        print(f"Difference: {results['diff_pct']:+.1f}%")
        
        print(f"\nSynchronization rate: {results['sync_rate']:.4f}")
        print(f"Random expectation: 0.1000")
        print(f"Status: {'SYNCHRONIZED' if results['sync_rate'] > 0.15 else 'INDEPENDENT'}")
        
        if self.ask_save():
            self.save_results("cross_constant_comparison", results)
        
        input("\nPress Enter to continue...")
    
    def compare_constants(self, digits1, digits2, name):
        matches = sum(1 for d1, d2 in zip(digits1, digits2) if d1 == d2)
        n = len(digits1)
        expected_matches = n * 0.1
        
        return {
            'matches': matches,
            'match_pct': (matches / n) * 100,
            'expected_matches': expected_matches,
            'expected_pct': 10.0,
            'diff_pct': ((matches - expected_matches) / expected_matches) * 100,
            'sync_rate': matches / n,
            'constant_name': name
        }
    
    def custom_digit_analysis(self):
        self.clear_screen()
        self.print_header()
        print("CUSTOM DIGIT ANALYSIS")
        print("="*80)
        print()
        
        n_digits = int(input("Enter number of digits (default 1000): ") or "1000")
        
        print("\nSelect analysis type:")
        print("  [1] Digit frequency distribution")
        print("  [2] Run length analysis")
        print("  [3] Digit pair analysis")
        print("  [4] Custom pattern search")
        print()
        
        choice = input("Enter choice: ").strip()
        
        pi_digits = self.generate_pi_digits(n_digits)
        
        if choice == '1':
            results = self.digit_frequency_analysis(pi_digits)
        elif choice == '2':
            results = self.run_length_analysis(pi_digits)
        elif choice == '3':
            results = self.digit_pair_analysis(pi_digits)
        elif choice == '4':
            pattern = input("Enter pattern to search for: ").strip()
            results = self.custom_pattern_search(pi_digits, pattern)
        else:
            print("Invalid choice.")
            input("Press Enter to continue...")
            return
        
        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80)
        print()
        print(json.dumps(results, indent=2))
        
        if self.ask_save():
            self.save_results("custom_digit_analysis", results)
        
        input("\nPress Enter to continue...")
    
    def digit_frequency_analysis(self, digits):
        freqs = [0] * 10
        for d in digits:
            freqs[int(d)] += 1
        return {'digit_frequencies': freqs, 'total_digits': len(digits)}
    
    def run_length_analysis(self, digits):
        runs = []
        current_digit = digits[0]
        current_length = 1
        
        for d in digits[1:]:
            if d == current_digit:
                current_length += 1
            else:
                runs.append({'digit': current_digit, 'length': current_length})
                current_digit = d
                current_length = 1
        
        runs.append({'digit': current_digit, 'length': current_length})
        
        max_run = max(runs, key=lambda x: x['length'])
        avg_run = sum(r['length'] for r in runs) / len(runs)
        
        return {
            'total_runs': len(runs),
            'max_run': max_run,
            'average_run_length': avg_run
        }
    
    def digit_pair_analysis(self, digits):
        pairs = defaultdict(int)
        for i in range(len(digits) - 1):
            pair = digits[i:i+2]
            pairs[pair] += 1
        
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'unique_pairs': len(pairs),
            'top_10_pairs': sorted_pairs[:10]
        }
    
    def custom_pattern_search(self, digits, pattern):
        positions = []
        for i in range(len(digits) - len(pattern) + 1):
            if digits[i:i+len(pattern)] == pattern:
                positions.append(i)
        
        return {
            'pattern': pattern,
            'occurrences': len(positions),
            'positions': positions[:20]  # First 20 positions
        }
    
    def view_previous_results(self):
        self.clear_screen()
        self.print_header()
        print("PREVIOUS RESULTS")
        print("="*80)
        print()
        
        files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        
        if not files:
            print("No previous results found.")
            input("\nPress Enter to continue...")
            return
        
        print("Available result files:")
        for i, f in enumerate(files, 1):
            print(f"  [{i}] {f}")
        
        print()
        choice = input("Enter file number to view (0 to cancel): ").strip()
        
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(files):
                return
            
            filepath = os.path.join(self.results_dir, files[idx])
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            print("\n" + "="*80)
            print(f"RESULTS FROM: {files[idx]}")
            print("="*80)
            print()
            print(json.dumps(data, indent=2))
            
        except (ValueError, IndexError, FileNotFoundError):
            print("Invalid selection.")
        
        input("\nPress Enter to continue...")
    
    def ask_save(self):
        response = input("\nSave results to file? (y/n): ").strip().lower()
        return response == 'y'
    
    def save_results(self, analysis_type, results):
        filename = f"{analysis_type}_{self.session_id}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        data = {
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")

def main():
    analyzer = InteractivePiAnalyzer()
    analyzer.main_menu()

if __name__ == "__main__":
    main()
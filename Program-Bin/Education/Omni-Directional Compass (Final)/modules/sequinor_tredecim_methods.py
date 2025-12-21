"""
Sequinor Tredecim Methods Module
Enhanced 13-part symposium and transformation methods
"""

import math
import numpy as np
from decimal import Decimal, getcontext
from fractions import Fraction
import json
from typing import Dict, List, Optional, Union, Tuple
import itertools

class SequinorTredecimMethods:
    def __init__(self, parent_compass):
        self.compass = parent_compass
        # Enhanced Sequinor Tredecim constants
        self.L_values = list(range(1, 14))  # Lambda values 1-13
        self.p_t = Decimal('1000') / Decimal('169')  # Enhanced Beta constant
        self.p_e = 1371119 + Fraction(256, 6561)  # Enhanced Epsilon
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio for enhanced methods
        
        # Pre-calculated weights for efficiency
        self.L_weights = [L / 91 for L in self.L_values]  # Sum = 1
        self.L_squared_weights = [L**2 / sum(L**2 for L in self.L_values) for L in self.L_values]
        self.fibonacci_weights = self._generate_fibonacci_weights()
        
    def _generate_fibonacci_weights(self) -> List[float]:
        """Generate Fibonacci-based weights for 13 parts"""
        fib = [1, 1]
        for i in range(11):
            fib.append(fib[-1] + fib[-2])
        
        # Take first 13 Fibonacci numbers and normalize
        weights = fib[:13]
        total = sum(weights)
        return [w / total for w in weights]
    
    def main_menu(self) -> bool:
        """Enhanced Sequinor Tredecim menu"""
        print("\n" + "=" * 80)
        print("SEQUINOR TREDECIM METHODS")
        print("=" * 80)
        print()
        print("Choose your transformation:")
        print()
        print("1. 13-PART SYMPOSIUM")
        print("   Decompose numbers into 13 parts using multiple methods")
        print()
        print("2. BETA TRANSFORMATION")
        print("   Apply Beta constant transformations")
        print()
        print("3. EPSILON ANALYSIS")
        print("   Epsilon-based pattern analysis")
        print()
        print("4. LAMBDA WEIGHTING")
        print("   Lambda-weighted calculations and distributions")
        print()
        print("5. MODULAR CYCLES")
        print("   Advanced modular arithmetic with 13-base")
        print()
        print("6. FRACTAL DECOMPOSITION")
        print("   Fractal-based 13-part analysis")
        print()
        print("7. BATCH PROCESSING")
        print("   Process multiple numbers simultaneously")
        print()
        print("8. PATTERN RECOGNITION")
        print("   Identify patterns in 13-part distributions")
        print()
        print("9. RETURN TO MAIN MENU")
        print()
        
        while True:
            choice = input("Enter your choice (1-9): ").strip()
            if choice in [str(i) for i in range(1, 10)]:
                return choice
            print("Invalid choice. Please enter 1-9.")
    
    def run(self):
        """Main Sequinor Tredecim interface"""
        while True:
            choice = self.main_menu()
            
            if choice == '1':
                self.thirteen_part_symposium()
            elif choice == '2':
                self.beta_transformation()
            elif choice == '3':
                self.epsilon_analysis()
            elif choice == '4':
                self.lambda_weighting()
            elif choice == '5':
                self.modular_cycles()
            elif choice == '6':
                self.fractal_decomposition()
            elif choice == '7':
                self.batch_processing()
            elif choice == '8':
                self.pattern_recognition()
            elif choice == '9':
                break
            
            input("\nPress Enter to continue...")
            print("\n" * 2)
    
    def thirteen_part_symposium(self, number: Optional[float] = None, label: str = "") -> bool:
        """Enhanced 13-part symposium with multiple methods"""
        print("\n" + "=" * 80)
        print("13-PART SYMPOSIUM")
        print("=" * 80)
        print()
        
        if number is None:
            print("Enter a number to decompose into 13 parts")
            print("(Can be integer, decimal, or mathematical expression)")
            print()
            
            while True:
                user_input = input("Number or expression: ").strip()
                try:
                    # Enhanced expression evaluation
                    number = eval(user_input, {
                        "__builtins__": {}, 
                        "pi": math.pi, 
                        "e": math.e, 
                        "sqrt": math.sqrt,
                        "sin": math.sin, "cos": math.cos, "tan": math.tan,
                        "log": math.log, "log10": math.log10,
                        "phi": self.phi
                    }, {})
                    break
                except:
                    print("Invalid input. Please enter a valid number or expression.")
        
        print()
        print(f"Decomposing: {number}" + (f" ({label})" if label else ""))
        print()
        
        # Method 1: Equal Division (Basic)
        print("--- Method 1: Equal Division (Basic) ---")
        parts_equal = [number / 13 for _ in range(13)]
        print(f"Each part: {parts_equal[0]:.10f}")
        print(f"Sum: {sum(parts_equal):.10f}")
        print(f"Verification: {'✓ PASS' if abs(sum(parts_equal) - number) < 1e-10 else '✗ FAIL'}")
        print()
        
        # Method 2: Lambda-Weighted (Enhanced)
        print("--- Method 2: Lambda-Weighted (Enhanced) ---")
        parts_lambda = [number * weight for weight in self.L_weights]
        print(f"Lambda weights (first 3): {[f'{w:.6f}' for w in self.L_weights[:3]]}")
        print(f"Sum: {sum(parts_lambda):.10f}")
        print(f"Verification: {'✓ PASS' if abs(sum(parts_lambda) - number) < 1e-10 else '✗ FAIL'}")
        print()
        
        # Method 3: Beta Formula (Enhanced)
        print("--- Method 3: Beta Formula (Enhanced) ---")
        p_x = ((number / 13) * 1000) / 13
        print(f"p(x) = {p_x:.10f}")
        parts_beta = [number * (L / 91) * float(self.p_t) for L in range(1, 14)]
        # Normalize to maintain sum
        total_beta = sum(parts_beta)
        if total_beta != 0:
            parts_beta = [p * number / total_beta for p in parts_beta]
        print(f"Beta parts (first 3): {[f'{p:.6f}' for p in parts_beta[:3]]}")
        print(f"Sum: {sum(parts_beta):.10f}")
        print(f"Verification: {'✓ PASS' if abs(sum(parts_beta) - number) < 1e-10 else '✗ FAIL'}")
        print()
        
        # Method 4: Fibonacci-Weighted
        print("--- Method 4: Fibonacci-Weighted ---")
        parts_fib = [number * weight for weight in self.fibonacci_weights]
        print(f"Fibonacci weights (first 3): {[f'{w:.6f}' for w in self.fibonacci_weights[:3]]}")
        print(f"Sum: {sum(parts_fib):.10f}")
        print(f"Verification: {'✓ PASS' if abs(sum(parts_fib) - number) < 1e-10 else '✗ FAIL'}")
        print()
        
        # Method 5: Exponential Distribution
        print("--- Method 5: Exponential Distribution ---")
        exp_weights = [math.exp(i/6) for i in range(13)]
        exp_weights = [w / sum(exp_weights) for w in exp_weights]  # Normalize
        parts_exp = [number * weight for weight in exp_weights]
        print(f"Exponential weights (first 3): {[f'{w:.6f}' for w in exp_weights[:3]]}")
        print(f"Sum: {sum(parts_exp):.10f}")
        print(f"Verification: {'✓ PASS' if abs(sum(parts_exp) - number) < 1e-10 else '✗ FAIL'}")
        print()
        
        # Advanced Analysis
        print("--- Advanced Analysis ---")
        
        # Modular cycles
        quotient = int(number // 13)
        remainder = number % 13
        print(f"Modular Analysis: {number} = 13 × {quotient} + {remainder:.6f}")
        
        # n² mod 13 analysis
        mod_13 = int(number) % 13
        matching_n = [n for n in range(1, 14) if (n ** 2) % 13 == mod_13]
        print(f"Number mod 13: {mod_13}")
        print(f"Matching n values (where n² ≡ {mod_13} mod 13): {matching_n}")
        print(f"Is quadratic residue: {'Yes' if matching_n else 'No'}")
        print()
        
        # Connection to Epsilon
        epsilon_ratio = number / float(self.p_e) if float(self.p_e) != 0 else 0
        print(f"Ratio to Epsilon: {epsilon_ratio:.10f}")
        
        # Closest Lambda value
        closest_L = min(self.L_values, key=lambda L: abs(number - L))
        print(f"Closest L value: L{closest_L}")
        
        # Golden ratio proximity
        phi_proximity = abs(number - self.phi) / self.phi
        print(f"Golden ratio proximity: {phi_proximity:.6f}")
        print()
        
        # Complete breakdown table (Lambda method)
        self.display_breakdown_table(parts_lambda, "Lambda-Weighted")
        
        # Statistical analysis
        self.statistical_analysis(parts_equal, parts_lambda, parts_beta, parts_fib, parts_exp)
        
        # Save option
        save = input("Would you like to save this decomposition? (y/n): ").strip().lower()
        if save == 'y':
            self.save_decomposition(number, label, parts_lambda, {
                'equal': parts_equal,
                'lambda': parts_lambda,
                'beta': parts_beta,
                'fibonacci': parts_fib,
                'exponential': parts_exp
            })
        
        return True
    
    def display_breakdown_table(self, parts: List[float], method_name: str):
        """Display detailed breakdown table"""
        print(f"--- Complete 13-Part Breakdown ({method_name}) ---")
        print(f"{'L':>3} | {'Weight':>8} | {'Part Value':>15} | {'Cumulative':>15} | {'% of Total':>10}")
        print("-" * 75)
        cumulative = 0
        total = sum(parts)
        
        for i, (L, part) in enumerate(zip(self.L_values, parts)):
            weight = self.L_weights[i]
            cumulative += part
            percentage = (part / total) * 100 if total != 0 else 0
            print(f"{L:3d} | {weight:8.6f} | {part:15.6f} | {cumulative:15.6f} | {percentage:9.4f}%")
        
        print("-" * 75)
        print(f"{'':>3} | {'Total:':>8} | {total:15.6f} | {'':>15} | {'100.0000%':>10}")
        print()
    
    def statistical_analysis(self, parts_equal: List[float], parts_lambda: List[float], 
                           parts_beta: List[float], parts_fib: List[float], parts_exp: List[float]):
        """Perform statistical analysis on different methods"""
        print("--- Statistical Analysis ---")
        
        methods = {
            'Equal': parts_equal,
            'Lambda': parts_lambda,
            'Beta': parts_beta,
            'Fibonacci': parts_fib,
            'Exponential': parts_exp
        }
        
        print(f"{'Method':>12} | {'Mean':>10} | {'Std Dev':>10} | {'Range':>10} | {'Entropy':>10}")
        print("-" * 65)
        
        for name, parts in methods.items():
            mean = np.mean(parts)
            std = np.std(parts)
            range_val = max(parts) - min(parts)
            
            # Calculate entropy (simplified)
            probs = [p / sum(parts) for p in parts if p > 0]
            entropy = -sum(p * np.log2(p) for p in probs) if probs else 0
            
            print(f"{name:>12} | {mean:10.6f} | {std:10.6f} | {range_val:10.6f} | {entropy:10.6f}")
        
        print()
    
    def beta_transformation(self) -> bool:
        """Enhanced Beta transformation methods"""
        print("\n" + "=" * 80)
        print("BETA TRANSFORMATION")
        print("=" * 80)
        print()
        
        print("Beta transformation applies the constant p_t = 1000/169")
        print(f"Current Beta constant: {self.p_t}")
        print()
        
        while True:
            try:
                user_input = input("Enter number(s) to transform (comma-separated): ").strip()
                numbers = [float(x.strip()) for x in user_input.split(',')]
                break
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
        
        print("\n--- Beta Transformations ---")
        
        for i, num in enumerate(numbers, 1):
            print(f"\n{i}. Original: {num}")
            
            # Basic Beta transform
            beta_basic = float(Decimal(str(num)) * self.p_t)
            print(f"   Beta ×: {beta_basic:.10f}")
            
            # Beta division
            beta_div = float(Decimal(str(num)) / self.p_t)
            print(f"   Beta ÷: {beta_div:.10f}")
            
            # Beta power
            beta_pow = num ** float(self.p_t)
            print(f"   Beta ^: {beta_pow:.10f}")
            
            # Beta root
            beta_root = num ** (1 / float(self.p_t))
            print(f"   Beta √: {beta_root:.10f}")
        
        return True
    
    def epsilon_analysis(self) -> bool:
        """Epsilon-based pattern analysis"""
        print("\n" + "=" * 80)
        print("EPSILON ANALYSIS")
        print("=" * 80)
        print()
        
        print(f"Epsilon constant: {self.p_e}")
        print("Epsilon analysis explores patterns and relationships")
        print()
        
        # Analyze number properties relative to epsilon
        test_number = input("Enter number to analyze (press Enter for default 1000): ").strip()
        if not test_number:
            test_number = "1000"
        
        try:
            num = float(test_number)
        except ValueError:
            print("Invalid number, using 1000")
            num = 1000.0
        
        print(f"\n--- Epsilon Analysis for {num} ---")
        
        # Basic relationships
        ratio = num / float(self.p_e)
        inverse_ratio = float(self.p_e) / num
        difference = num - float(self.p_e)
        
        print(f"Number/Epsilon: {ratio:.10f}")
        print(f"Epsilon/Number: {inverse_ratio:.10f}")
        print(f"Difference: {difference:.10f}")
        
        # Check for special relationships
        if abs(ratio - 1) < 0.01:
            print("✓ Number is very close to Epsilon")
        if abs(ratio - round(ratio)) < 0.01:
            print(f"✓ Number is approximately {round(ratio)} × Epsilon")
        if abs(ratio - self.phi) < 0.1:
            print("✓ Golden ratio relationship detected")
        
        # Modular analysis with epsilon
        epsilon_mod = num % float(self.p_e)
        print(f"Number mod Epsilon: {epsilon_mod:.10f}")
        
        # Check if number is in epsilon's sequence
        epsilon_sequence = self.generate_epsilon_sequence(20)
        if num in epsilon_sequence:
            print("✓ Number appears in epsilon sequence")
        
        return True
    
    def generate_epsilon_sequence(self, length: int) -> List[float]:
        """Generate a sequence based on epsilon"""
        sequence = []
        current = float(self.p_e)
        
        for i in range(length):
            sequence.append(current)
            # Generate next term using epsilon relationship
            current = (current * float(self.p_e)) / (i + 2)
        
        return sequence
    
    def lambda_weighting(self) -> bool:
        """Lambda-weighted calculations"""
        print("\n" + "=" * 80)
        print("LAMBDA WEIGHTING")
        print("=" * 80)
        print()
        
        print("Lambda weighting uses the grip constant Λ = 4")
        print("Applying weighted distributions based on Lambda values 1-13")
        print()
        
        # Get user input for distribution
        try:
            total_value = float(input("Enter total value to distribute: "))
            weighting_scheme = input("Choose scheme (equal/linear/quadratic/exponential): ").strip().lower()
        except ValueError:
            print("Invalid input, using defaults")
            total_value = 100.0
            weighting_scheme = "linear"
        
        # Calculate weights based on scheme
        if weighting_scheme == "equal":
            weights = [1/13] * 13
        elif weighting_scheme == "linear":
            weights = self.L_weights
        elif weighting_scheme == "quadratic":
            weights = self.L_squared_weights
        elif weighting_scheme == "exponential":
            exp_weights = [math.exp(i/3) for i in range(13)]
            weights = [w / sum(exp_weights) for w in exp_weights]
        else:
            weights = self.L_weights
        
        # Apply lambda enhancement
        enhanced_weights = [w * self.compass.LAMBDA / 4 for w in weights]  # Normalize by Lambda
        enhanced_weights = [w / sum(enhanced_weights) for w in enhanced_weights]  # Renormalize
        
        print(f"\n--- {weighting_scheme.title()} Lambda-Weighted Distribution ---")
        print(f"Total value: {total_value}")
        print()
        print(f"{'L':>3} | {'Weight':>10} | {'Enhanced':>10} | {'Value':>12} | {'Cumulative':>12}")
        print("-" * 65)
        
        cumulative = 0
        for i, (L, weight, enhanced) in enumerate(zip(self.L_values, weights, enhanced_weights)):
            value = total_value * enhanced
            cumulative += value
            print(f"{L:3d} | {weight:10.6f} | {enhanced:10.6f} | {value:12.6f} | {cumulative:12.6f}")
        
        print("-" * 65)
        print(f"{'':>3} | {'Total:':>10} | {'':>10} | {cumulative:12.6f} | {'':>12}")
        print()
        
        return True
    
    def modular_cycles(self) -> bool:
        """Advanced modular arithmetic with 13-base"""
        print("\n" + "=" * 80)
        print("MODULAR CYCLES")
        print("=" * 80)
        print()
        
        print("Exploring modular arithmetic patterns with modulus 13")
        print()
        
        try:
            base_number = int(input("Enter base number for analysis: "))
            cycle_length = int(input("Enter cycle length (press Enter for 13): ") or "13")
        except ValueError:
            base_number = 7
            cycle_length = 13
        
        print(f"\n--- Modular Cycle Analysis for {base_number} mod 13 ---")
        
        # Generate cycle
        cycle = []
        current = base_number
        seen = set()
        
        for i in range(cycle_length):
            if current in seen:
                break
            seen.add(current)
            cycle.append(current)
            current = (current * base_number) % 13
        
        print(f"Cycle: {cycle}")
        print(f"Cycle length: {len(cycle)}")
        
        # Analyze cycle properties
        if cycle:
            print(f"Sum of cycle: {sum(cycle)}")
            print(f"Average: {sum(cycle)/len(cycle):.6f}")
            
            # Check for patterns
            if len(set(cycle)) == len(cycle):
                print("✓ No repetitions in cycle")
            else:
                print("✗ Repetitions detected")
        
        # Power cycles
        print(f"\n--- Power Cycles: {base_number}^n mod 13 ---")
        power_cycle = [(base_number ** i) % 13 for i in range(13)]
        print(f"Power cycle: {power_cycle}")
        
        # Multiplicative order
        order = 1
        current = (base_number ** order) % 13
        while current != 1 and order < 13:
            order += 1
            current = (base_number ** order) % 13
        
        if current == 1:
            print(f"Multiplicative order of {base_number}: {order}")
        else:
            print(f"{base_number} does not have multiplicative order modulo 13")
        
        return True
    
    def fractal_decomposition(self) -> bool:
        """Fractal-based 13-part analysis"""
        print("\n" + "=" * 80)
        print("FRACTAL DECOMPOSITION")
        print("=" * 80)
        print()
        
        print("Applying fractal principles to 13-part decomposition")
        print()
        
        try:
            initial_number = float(input("Enter initial number: "))
            iterations = int(input("Enter number of iterations (1-5): ") or "3")
        except ValueError:
            initial_number = 100.0
            iterations = 3
        
        print(f"\n--- Fractal Decomposition of {initial_number} ---")
        
        current = initial_number
        all_decompositions = []
        
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}: Decomposing {current}")
            
            # Apply lambda-weighted decomposition
            parts = [current * weight for weight in self.L_weights]
            all_decompositions.append(parts)
            
            # Select largest part for next iteration (fractal property)
            largest_part = max(parts)
            largest_index = parts.index(largest_part)
            current = largest_part
            
            print(f"  Largest part: L{largest_index + 1} = {largest_part:.6f}")
            print(f"  Parts: {[f'{p:.4f}' for p in parts[:5]]}...")  # Show first 5
        
        # Calculate fractal dimension
        if len(all_decompositions) > 1:
            print(f"\n--- Fractal Analysis ---")
            ratios = []
            for i in range(1, len(all_decompositions)):
                prev_max = max(all_decompositions[i-1])
                curr_max = max(all_decompositions[i])
                if prev_max > 0:
                    ratios.append(curr_max / prev_max)
            
            if ratios:
                avg_ratio = np.mean(ratios)
                print(f"Average scaling ratio: {avg_ratio:.6f}")
                
                # Estimate fractal dimension (simplified)
                if avg_ratio > 0 and avg_ratio != 1:
                    fractal_dim = -math.log(1/13) / math.log(avg_ratio)
                    print(f"Estimated fractal dimension: {fractal_dim:.6f}")
        
        return True
    
    def batch_processing(self) -> bool:
        """Process multiple numbers simultaneously"""
        print("\n" + "=" * 80)
        print("BATCH PROCESSING")
        print("=" * 80)
        print()
        
        print("Enter multiple numbers for batch 13-part decomposition")
        print("Options:")
        print("1. Manual input")
        print("2. Generate sequence")
        print("3. Load from file")
        print()
        
        choice = input("Choose option (1-3): ").strip()
        
        numbers = []
        if choice == '1':
            input_str = input("Enter numbers separated by commas: ")
            try:
                numbers = [float(x.strip()) for x in input_str.split(',')]
            except ValueError:
                print("Invalid input")
                return False
        elif choice == '2':
            start = float(input("Start value: ") or "1")
            end = float(input("End value: ") or "13")
            count = int(input("Number of values: ") or "13")
            numbers = np.linspace(start, end, count).tolist()
        elif choice == '3':
            filename = input("Enter filename: ")
            try:
                with open(filename, 'r') as f:
                    numbers = [float(line.strip()) for line in f if line.strip()]
            except FileNotFoundError:
                print("File not found")
                return False
        
        if not numbers:
            print("No numbers to process")
            return False
        
        print(f"\nProcessing {len(numbers)} numbers...")
        
        # Process each number
        results = {}
        for i, num in enumerate(numbers, 1):
            print(f"\n{i}. Processing {num}")
            
            # Quick decomposition
            parts = [num * weight for weight in self.L_weights]
            
            # Store key statistics
            results[f"number_{i}"] = {
                'original': num,
                'parts': parts,
                'max_part': max(parts),
                'min_part': min(parts),
                'range': max(parts) - min(parts),
                'entropy': self.calculate_entropy(parts)
            }
        
        # Summary statistics
        print(f"\n--- Batch Summary ---")
        all_max = [r['max_part'] for r in results.values()]
        all_min = [r['min_part'] for r in results.values()]
        all_entropy = [r['entropy'] for r in results.values()]
        
        print(f"Average max part: {np.mean(all_max):.6f}")
        print(f"Average min part: {np.mean(all_min):.6f}")
        print(f"Average entropy: {np.mean(all_entropy):.6f}")
        print(f"Entropy range: {min(all_entropy):.6f} - {max(all_entropy):.6f}")
        
        # Save results
        save = input("Save batch results? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Enter filename: ") or "batch_decomposition.json"
            try:
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {filename}")
            except Exception as e:
                print(f"Save failed: {e}")
        
        return True
    
    def pattern_recognition(self) -> bool:
        """Identify patterns in 13-part distributions"""
        print("\n" + "=" * 80)
        print("PATTERN RECOGNITION")
        print("=" * 80)
        print()
        
        print("Analyzing patterns in 13-part distributions")
        print()
        
        # Generate sample distributions
        test_numbers = [1, 13, 169, 1000, math.pi, math.e, self.phi]
        
        print("--- Pattern Analysis for Test Numbers ---")
        print(f"{'Number':>10} | {'Pattern':>15} | {'Symmetry':>10} | {'Clusters':>10}")
        print("-" * 60)
        
        for num in test_numbers:
            parts = [num * weight for weight in self.L_weights]
            
            # Pattern recognition
            pattern = self.identify_pattern(parts)
            symmetry = self.check_symmetry(parts)
            clusters = self.identify_clusters(parts)
            
            print(f"{num:10.6f} | {pattern:>15} | {symmetry:>10} | {clusters:>10}")
        
        return True
    
    def identify_pattern(self, parts: List[float]) -> str:
        """Identify distribution pattern"""
        # Simple pattern recognition
        if max(parts) - min(parts) < 1e-10:
            return "Equal"
        elif all(abs(parts[i+1] - parts[i]) < 1e-6 for i in range(len(parts)-1)):
            return "Linear"
        elif parts == sorted(parts):
            return "Ascending"
        elif parts == sorted(parts, reverse=True):
            return "Descending"
        else:
            return "Complex"
    
    def check_symmetry(self, parts: List[float]) -> str:
        """Check for symmetry in distribution"""
        n = len(parts)
        center = n // 2
        
        # Check for reflection symmetry
        symmetric = True
        for i in range(center):
            if abs(parts[i] - parts[n-1-i]) > 1e-6:
                symmetric = False
                break
        
        return "Yes" if symmetric else "No"
    
    def identify_clusters(self, parts: List[float]) -> str:
        """Identify clustering in distribution"""
        # Simple clustering based on value ranges
        sorted_parts = sorted(parts)
        
        # Look for natural breaks
        breaks = []
        for i in range(1, len(sorted_parts)):
            if sorted_parts[i] - sorted_parts[i-1] > (sorted_parts[-1] - sorted_parts[0]) / 10:
                breaks.append(i)
        
        return f"{len(breaks) + 1}"
    
    def calculate_entropy(self, parts: List[float]) -> float:
        """Calculate Shannon entropy of distribution"""
        total = sum(parts)
        if total == 0:
            return 0
        
        probs = [p / total for p in parts if p > 0]
        return -sum(p * math.log2(p) for p in probs) if probs else 0
    
    def save_decomposition(self, number: float, label: str, parts: List[float], 
                          all_methods: Dict[str, List[float]]):
        """Save comprehensive decomposition data"""
        try:
            filename = input("Enter filename (e.g., 'decomposition.json'): ").strip()
            if not filename:
                filename = "decomposition.json"
            
            data = {
                'number': float(number),
                'label': label,
                'timestamp': str(np.datetime64('now')),
                'methods': all_methods,
                'analysis': {
                    'entropy': self.calculate_entropy(parts),
                    'range': max(parts) - min(parts),
                    'mean': np.mean(parts),
                    'std': np.std(parts)
                },
                'metadata': {
                    'method_count': len(all_methods),
                    'parts_count': 13,
                    'total_preserved': abs(sum(parts) - number) < 1e-10
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved to {filename}")
        except Exception as e:
            print(f"Error saving: {e}")
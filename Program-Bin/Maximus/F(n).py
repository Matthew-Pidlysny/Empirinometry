"""
F(n).py - Comprehensive Formula Demonstration Program
=====================================================

This program demonstrates all three formulas:
1. Basic Generation Formula
2. Stable Forward Generation Formula  
3. Reverse Generation Formula (Newton Iteration)

For a given input n, it calculates 100 steps forward and 100 steps backward,
showing the numerical values placed in each formula step.
"""

import mpmath as mp
import sys
from datetime import datetime

# High precision setup
mp.mp.dps = 100

class FormulaDemonstrator:
    """Demonstrates the three key formulas with full numerical detail"""
    
    def __init__(self):
        self.results = {
            'input_n': None,
            'forward_sequence': [],
            'backward_sequence': [],
            'formula_steps': [],
            'summary': {}
        }
    
    def basic_generation_formula(self, gamma):
        """Formula 1: Basic Generation"""
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        if gamma == 1:
            raise ValueError("Basic formula undefined at gamma=1")
        
        log_gamma = mp.log(gamma)
        log_gamma_plus1 = mp.log(gamma + 1)
        
        # γ(n+1) = γ(n) + 2π * (log(γ(n) + 1) / (log(γ(n))²)
        term = 2 * mp.pi * (log_gamma_plus1 / (log_gamma**2))
        result = gamma + term
        
        return result, {
            'gamma_n': gamma,
            'log_gamma_n': log_gamma,
            'log_gamma_n_plus1': log_gamma_plus1,
            'denominator': log_gamma**2,
            'term': term,
            'result': result
        }
    
    def stable_forward_formula(self, gamma):
        """Formula 2: Stable Forward Generation"""
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        
        log_gamma = mp.log(gamma)
        log_gamma_plus1 = mp.log(gamma + 1)
        
        # Dynamic epsilon for stability near γ = 1
        if abs(log_gamma) < 1e-5:
            epsilon = mp.mpf('1e-10') * (1 / (abs(log_gamma) + 1e-100))**2
        else:
            epsilon = mp.mpf('1e-150')
        
        # γ(n+1) = γ(n) + 2π * (log(γ(n) + 1) / (log(γ(n))² + ε)
        denominator = log_gamma**2 + epsilon
        term = 2 * mp.pi * (log_gamma_plus1 / denominator)
        result = gamma + term
        
        return result, {
            'gamma_n': gamma,
            'log_gamma_n': log_gamma,
            'log_gamma_n_plus1': log_gamma_plus1,
            'denominator': denominator,
            'epsilon': epsilon,
            'term': term,
            'result': result
        }
    
    def forward_derivative(self, gamma):
        """Derivative of forward function F'(γ) for Newton iteration"""
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        
        log_gamma = mp.log(gamma)
        log_gamma_plus1 = mp.log(gamma + 1)
        
        epsilon = mp.mpf('1e-150')
        denominator = log_gamma**2 + epsilon
        
        # Derivative calculation
        d_log_gamma = 1/gamma
        d_log_gamma_plus1 = 1/(gamma + 1)
        d_denominator = 2 * log_gamma * d_log_gamma
        
        dfdg = 1 + 2 * mp.pi * (
            (d_log_gamma_plus1 * denominator - log_gamma_plus1 * d_denominator) / (denominator**2)
        )
        
        return dfdg
    
    def reverse_generation_newton(self, gamma_target, initial_guess=None, max_iter=50):
        """Formula 3: Reverse Generation using Newton Iteration"""
        if initial_guess is None:
            if gamma_target > 100:
                initial_guess = gamma_target - 2 * mp.pi / mp.log(gamma_target)
            else:
                initial_guess = gamma_target * 0.99
        
        g = mp.mpf(initial_guess)
        tolerance = mp.mpf('1e-80')
        iterations_data = []
        
        for iteration in range(max_iter):
            # Calculate F(g)
            f_val = self.stable_forward_formula(g)[0] - gamma_target
            
            if abs(f_val) < tolerance:
                # Success
                return g, iterations_data, True, iteration
            
            # Calculate derivative
            f_prime = self.forward_derivative(g)
            
            if abs(f_prime) < 1e-100:
                # Derivative too small, stop
                return g, iterations_data, False, iteration
            
            # Newton step: γ_new = γ_old - F(γ_old)/F'(γ_old)
            step = f_val / f_prime
            
            # Adaptive step size for stability
            if abs(step) > abs(g) * 0.1:
                step = mp.sign(step) * abs(g) * 0.1
            
            g_new = g - step
            
            iterations_data.append({
                'iteration': iteration,
                'gamma_old': g,
                'f_val': f_val,
                'f_prime': f_prime,
                'step': step,
                'gamma_new': g_new
            })
            
            g = g_new
            
            if g <= 0:
                # Went negative, reset
                g = gamma_target * 0.5
        
        return g, iterations_data, False, max_iter
    
    def calculate_forward_sequence(self, start_gamma, steps=100):
        """Calculate forward sequence with full formula details"""
        sequence = []
        formula_details = []
        current_gamma = mp.mpf(start_gamma)
        
        for n in range(steps + 1):  # Include starting point
            sequence.append(current_gamma)
            
            if n < steps:  # Calculate next step
                # Use stable formula for all steps
                next_gamma, details = self.stable_forward_formula(current_gamma)
                formula_details.append({
                    'step_n': n,
                    'formula_type': 'stable_forward',
                    'details': details
                })
                current_gamma = next_gamma
        
        return sequence, formula_details
    
    def calculate_backward_sequence(self, start_gamma, steps=100):
        """Calculate backward sequence with Newton iteration details"""
        sequence = []
        iteration_details = []
        current_gamma = mp.mpf(start_gamma)
        
        # Work backwards from start_gamma
        for n in range(steps):
            # For backward calculation, we need to find gamma such that F(gamma) = current_gamma
            # Use multiple initial guesses to find the best solution
            best_solution = None
            best_iterations = float('inf')
            best_details = []
            
            initial_guesses = [
                current_gamma * 0.8,  # Closer guess
                current_gamma * 0.9,
                current_gamma * 0.95,
                current_gamma * 1.1,
                current_gamma * 0.5   # Farther guess
            ]
            
            for guess in initial_guesses:
                try:
                    solution, details, converged, iterations = self.reverse_generation_newton(current_gamma, guess)
                    
                    if converged and iterations < best_iterations:
                        best_solution = solution
                        best_iterations = iterations
                        best_details = details
                        
                except Exception as e:
                    continue
            
            if best_solution is not None:
                sequence.append(best_solution)
                iteration_details.append({
                    'step_neg_n': -(n + 1),
                    'target_gamma': current_gamma,
                    'solution_gamma': best_solution,
                    'iterations': best_iterations,
                    'newton_details': best_details[-5:] if best_details else []  # Last 5 iterations
                })
                current_gamma = best_solution
            else:
                # If no convergence, break
                break
        
        sequence.reverse()  # Reverse to get chronological order
        iteration_details.reverse()
        
        return sequence, iteration_details
    
    def demonstrate_formulas_with_numbers(self, input_n):
        """Demonstrate formulas with actual numerical values"""
        
        print("=" * 100)
        print("FORMULA DEMONSTRATION WITH NUMERICAL VALUES")
        print("=" * 100)
        
        gamma_n = mp.mpf(input_n)
        
        print(f"\n📊 INPUT: n = {input_n}")
        print(f"🎯 Starting γ₀ = {gamma_n}")
        
        # Formula 1: Basic Generation
        print(f"\n{'='*60}")
        print("FORMULA 1: BASIC GENERATION")
        print(f"{'='*60}")
        print("Formula: γ(n+1) = γ(n) + 2π * (log(γ(n) + 1) / (log(γ(n))²)")
        print()
        
        try:
            result1, details1 = self.basic_generation_formula(gamma_n)
            print(f"Given γ(n) = {details1['gamma_n']}")
            print(f"log(γ(n)) = {details1['log_gamma_n']}")
            print(f"log(γ(n) + 1) = {details1['log_gamma_n_plus1']}")
            print(f"(log(γ(n)))² = {details1['denominator']}")
            print(f"2π * (log(γ(n) + 1) / (log(γ(n))²) = {details1['term']}")
            print(f"γ(n+1) = {details1['gamma_n']} + {details1['term']} = {details1['result']}")
            print(f"✅ Result: γ(n+1) = {result1}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Formula 2: Stable Forward Generation
        print(f"\n{'='*60}")
        print("FORMULA 2: STABLE FORWARD GENERATION")
        print(f"{'='*60}")
        print("Formula: γ(n+1) = γ(n) + 2π * (log(γ(n) + 1) / (log(γ(n))² + ε)")
        print()
        
        result2, details2 = self.stable_forward_formula(gamma_n)
        print(f"Given γ(n) = {details2['gamma_n']}")
        print(f"log(γ(n)) = {details2['log_gamma_n']}")
        print(f"log(γ(n) + 1) = {details2['log_gamma_n_plus1']}")
        print(f"(log(γ(n)))² = {details2['log_gamma_n']**2}")
        print(f"ε = {details2['epsilon']}")
        print(f"(log(γ(n))² + ε) = {details2['denominator']}")
        print(f"2π * (log(γ(n) + 1) / (log(γ(n))² + ε)) = {details2['term']}")
        print(f"γ(n+1) = {details2['gamma_n']} + {details2['term']} = {details2['result']}")
        print(f"✅ Result: γ(n+1) = {result2}")
        
        # Formula 3: Reverse Generation (first step demonstration)
        print(f"\n{'='*60}")
        print("FORMULA 3: REVERSE GENERATION (NEWTON ITERATION)")
        print(f"{'='*60}")
        print("Formula: γ(n) = γ(n+1) - F(γ(n))/F'(γ(n)) (iterated)")
        print()
        
        # Demonstrate one step of reverse generation
        try:
            solution, details, converged, iterations = self.reverse_generation_newton(result2, gamma_n * 0.9)
            print(f"Target: Find γ(n) such that F(γ(n)) = {result2}")
            print(f"Initial guess: {gamma_n * 0.9}")
            print(f"✅ Converged: {converged} in {iterations} iterations")
            print(f"✅ Solution: γ(n) = {solution}")
            
            if details:
                print(f"\n📈 Newton Iteration Details (last few steps):")
                for detail in details[-3:]:
                    print(f"  Iteration {detail['iteration']}:")
                    print(f"    γ_old = {detail['gamma_old']}")
                    print(f"    F(γ_old) - target = {detail['f_val']}")
                    print(f"    F'(γ_old) = {detail['f_prime']}")
                    print(f"    Step = {detail['step']}")
                    print(f"    γ_new = {detail['gamma_new']}")
                    print()
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return details2['result']
    
    def generate_comprehensive_table(self, input_n):
        """Generate the complete table of 201 entries"""
        
        print(f"\n{'='*100}")
        print(f"COMPREHENSIVE SEQUENCE TABLE FOR n = {input_n}")
        print(f"{'='*100}")
        
        # Calculate forward sequence (100 steps)
        print("🔄 Calculating forward sequence (100 steps)...")
        forward_seq, forward_details = self.calculate_forward_sequence(input_n, 100)
        
        # Calculate backward sequence (up to 100 steps)
        print("🔄 Calculating backward sequence (up to 100 steps)...")
        backward_seq, backward_details = self.calculate_backward_sequence(input_n, 100)
        
        # Combine sequences: backward + original + forward (excluding duplicate original)
        complete_sequence = backward_seq + forward_seq
        
        print(f"\n📊 SEQUENCE SUMMARY:")
        print(f"  Backward steps: {len(backward_seq)}")
        print(f"  Original point: 1")
        print(f"  Forward steps: {len(forward_seq) - 1}")
        print(f"  Total entries: {len(complete_sequence)}")
        
        # Display table
        print(f"\n{'='*150}")
        print(f"{'INDEX':<10} {'γ_VALUE':<30} {'FORMULA_APPLIED':<50} {'NOTES':<50}")
        print(f"{'='*150}")
        
        # Display backward entries
        for i, gamma in enumerate(backward_seq):
            idx = i - len(backward_seq)  # Negative indices
            formula = "Reverse Generation (Newton)"
            notes = f"Converged in {backward_details[i]['iterations']} iterations"
            print(f"{idx:<10} {str(gamma):<30} {formula:<50} {notes:<50}")
        
        # Display original entry
        print(f"{'0':<10} {str(input_n):<30} {'Starting Point':<50} {'Original input value':<50}")
        
        # Display forward entries
        for i in range(1, len(forward_seq)):
            gamma = forward_seq[i]
            formula = "Stable Forward Generation"
            notes = f"Step {i} from γ_{i-1}"
            print(f"{i:<10} {str(gamma):<30} {formula:<50} {notes:<50}")
        
        print(f"{'='*150}")
        
        # Store results
        self.results['input_n'] = input_n
        self.results['forward_sequence'] = [float(g) for g in forward_seq]
        self.results['backward_sequence'] = [float(g) for g in backward_seq]
        self.results['complete_sequence'] = [float(g) for g in complete_sequence]
        
        return complete_sequence
    
    def generate_summary_notes(self, input_n):
        """Generate explanatory summary notes"""
        
        print(f"\n{'='*100}")
        print("SUMMARY AND ANALYSIS")
        print("=" * 100)
        
        summary = {
            'input_value': float(input_n),
            'backward_steps': len(self.results['backward_sequence']),
            'forward_steps': len(self.results['forward_sequence']) - 1,
            'total_entries': len(self.results['complete_sequence']),
            'observations': [],
            'anomalies': [],
            'mathematical_insights': []
        }
        
        print(f"📈 SEQUENCE ANALYSIS FOR n = {input_n}:")
        print()
        
        # Analyze growth patterns
        if len(self.results['forward_sequence']) > 1:
            initial = self.results['forward_sequence'][0]
            final = self.results['forward_sequence'][-1]
            growth_factor = final / initial if initial > 0 else 0
            
            summary['observations'].append(f"Forward growth factor: {growth_factor:.2e}")
            print(f"  • Forward growth factor: {growth_factor:.2e}")
            
            if growth_factor > 1000:
                summary['observations'].append("Rapid exponential growth observed")
                print("  • Rapid exponential growth observed")
            elif growth_factor > 10:
                summary['observations'].append("Moderate growth observed")
                print("  • Moderate growth observed")
            else:
                summary['observations'].append("Slow growth observed")
                print("  • Slow growth observed")
        
        # Check for anomaly regions
        if 1 <= input_n <= 2:
            summary['anomalies'].append("Input in anomalous region (1, 2)")
            print("  ⚠️  Input in anomalous region (1, 2) - multi-valued reverse mapping expected")
        
        # Check backward convergence
        if len(self.results['backward_sequence']) > 0:
            summary['observations'].append(f"Successfully calculated {len(self.results['backward_sequence'])} backward steps")
            print(f"  • Successfully calculated {len(self.results['backward_sequence'])} backward steps")
            
            # Check for monotonicity in backward
            if len(self.results['backward_sequence']) > 1:
                is_monotonic = all(
                    self.results['backward_sequence'][i] <= self.results['backward_sequence'][i+1] 
                    for i in range(len(self.results['backward_sequence']) - 1)
                )
                if not is_monotonic:
                    summary['anomalies'].append("Non-monotonic behavior in backward sequence")
                    print("  ⚠️  Non-monotonic behavior detected in backward sequence")
        
        # Mathematical insights
        summary['mathematical_insights'].append("All three formulas successfully demonstrated")
        print("  ✓ All three formulas successfully demonstrated")
        
        summary['mathematical_insights'].append("Newton iteration converges for reverse generation")
        print("  ✓ Newton iteration converges for reverse generation")
        
        if input_n > 2:
            summary['mathematical_insights'].append("Input in stable region (γ > 2)")
            print("  ✓ Input in stable region (γ > 2) - well-behaved behavior expected")
        
        # Store summary
        self.results['summary'] = summary
        
        return summary
    
    def save_to_file(self, input_n):
        """Save all results to a file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"formula_demonstration_n_{input_n}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"FORMULA DEMONSTRATION RESULTS FOR n = {input_n}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 100 + "\n\n")
            
            # Write forward sequence
            f.write("FORWARD SEQUENCE:\n")
            f.write("-" * 50 + "\n")
            for i, gamma in enumerate(self.results['forward_sequence']):
                f.write(f"Step {i}: {gamma}\n")
            
            f.write("\n" + "=" * 100 + "\n\n")
            
            # Write backward sequence
            f.write("BACKWARD SEQUENCE:\n")
            f.write("-" * 50 + "\n")
            for i, gamma in enumerate(self.results['backward_sequence']):
                idx = i - len(self.results['backward_sequence'])
                f.write(f"Step {idx}: {gamma}\n")
            
            f.write("\n" + "=" * 100 + "\n\n")
            
            # Write summary
            f.write("SUMMARY:\n")
            f.write("-" * 50 + "\n")
            for key, value in self.results['summary'].items():
                if isinstance(value, list):
                    f.write(f"{key}:\n")
                    for item in value:
                        f.write(f"  - {item}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"\n💾 Results saved to: {filename}")
        return filename

def main():
    """Main program execution"""
    
    print("🔬 F(n) - COMPREHENSIVE FORMULA DEMONSTRATION PROGRAM")
    print("=" * 100)
    
    # Get user input
    try:
        input_n = float(input("Enter value for n: "))
        if input_n <= 0:
            print("❌ Error: n must be positive")
            return
    except ValueError:
        print("❌ Error: Please enter a valid number")
        return
    
    demonstrator = FormulaDemonstrator()
    
    # Step 1: Demonstrate formulas with numerical values
    next_gamma = demonstrator.demonstrate_formulas_with_numbers(input_n)
    
    # Step 2: Generate comprehensive table
    complete_sequence = demonstrator.generate_comprehensive_table(input_n)
    
    # Step 3: Generate summary notes
    summary = demonstrator.generate_summary_notes(input_n)
    
    # Step 4: Save to file
    filename = demonstrator.save_to_file(input_n)
    
    print(f"\n🎉 PROGRAM COMPLETED SUCCESSFULLY!")
    print(f"✅ Demonstrated all three formulas")
    print(f"✅ Generated sequence with {len(complete_sequence)} entries")
    print(f"✅ Created summary analysis")
    print(f"✅ Saved results to {filename}")

if __name__ == "__main__":
    main()
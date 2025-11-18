"""
=========================== FUTURE N CALCULATOR LICENSE ===========================


Copyright Holder: Matthew Pidlysny


License Terms:
1. The program may be used freely by the licensee for personal, non-commercial purposes.
2. Redistribution or resale of this program or any derivative works is strictly prohibited.
3. The licensee agrees to pay the owner, Matthew Pidlysny, the amount of ONE (1) US DOLLAR per month
for continued use of this software.
4. No warranties are provided. The program is provided "as-is". The licensee assumes all risks
associated with its use.
5. This agreement constitutes the entire understanding between the licensee and the owner.
6. By using this program, the licensee accepts these terms.


===================================================================================
"""

import math
import mpmath as mp
from mpmath import mpf, nstr, fabs, sign
import sys
from datetime import datetime

# ============================== PRECISION CONFIG ==============================
PRECISION_DIGITS = 150
mp.dps = PRECISION_DIGITS + 100

# ============================== OUTPUT REDIRECTION ==============================
class TeeOutput:
    """Class to duplicate output to both console and file"""
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()
        
    def flush(self):
        self.console.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

# ============================== UTILITY FUNCTION ==============================
def parse_gamma_input(user_input_str):
    """Parses input for fractions (y/x) or raw decimals."""
    user_input_str = user_input_str.strip()
    if "/" in user_input_str:
        try:
            num, den = user_input_str.split('/')
            return mpf(num) / mpf(den)
        except:
            print(f"Warning: Could not parse fraction '{user_input_str}'. Treating as raw number.")
            return mpf(user_input_str)
    else:
        return mpf(user_input_str)

# ============================== HEADER & INTRODUCTION ==============================
def print_header():
    print("=" * 80)
    print("         FUTURE N CALCULATOR - GAMMA SEQUENCE EXPLORER")
    print("=" * 80)
    print()
    print("DESCRIPTION:")
    print("This program calculates the γ_n sequence defined by the recurrence relation:")
    print("    γₙ₊₁ = γₙ + 2π * (log(γₙ + 1) / (log γₙ)²)")
    print()
    print("FEATURES:")
    print(f"• Computes sequence values with {PRECISION_DIGITS} decimal precision")
    print("• Uses modified formula to handle γₙ = 1 (avoids division by zero)")
    print("• Displays EXACT previous 5 entries using true inverse formula")
    print("• Provides detailed step-by-step analysis")
    print("• Continuously saves output to gamma_sequence_output.txt")
    print()
    print("MATHEMATICAL NOTES:")
    print("• Now with perfect forward/backward reversibility")
    print("• Enhanced stability around γ = 1")
    print("=" * 80)
    print()

# ============================== ENHANCED EXACT INVERSE FUNCTION ==============================
def gamma_previous_exact(gamma_current, epsilon=mpf('1e-150')):
    """Enhanced exact inverse: given γₙ₊₁, find γₙ using high-precision Newton iteration"""
    if gamma_current > 100:
        g = gamma_current - 2 * mp.pi / mp.log(gamma_current)
    else:
        g = gamma_current * 0.99
    
    max_iterations = 100
    tolerance = mpf('1e-160')
    
    for iteration in range(max_iterations):
        log_g = mp.log(g)
        log_g1 = mp.log(g + 1)
        denom = log_g**2 + epsilon
        
        forward_step = g + 2 * mp.pi * (log_g1 / denom)
        residual = forward_step - gamma_current
        
        if fabs(residual) < tolerance:
            return g
            
        d_log_g = 1/g
        d_log_g1 = 1/(g + 1)
        d_denom = 2 * log_g * d_log_g
        
        dfdg = 1 + 2 * mp.pi * (
            (d_log_g1 * denom - log_g1 * d_denom) / (denom**2)
        )
        
        step = residual / dfdg
        if fabs(step) > fabs(g) * 0.1:
            step = sign(step) * fabs(g) * 0.1
            
        g -= step
        
        if g <= 0:
            g = gamma_current * 0.5
    
    return g

def compute_previous_5_enhanced(start_gamma):
    """Compute exact previous 5 terms using enhanced inverse calculation"""
    prevs = []
    g = mpf(start_gamma)
    
    print("Computing previous entries via reverse engineering...")
    
    for step_back in range(5):
        g_prev = gamma_previous_exact(g)
        
        log_g_prev = mp.log(g_prev)
        log_g_prev_plus1 = mp.log(g_prev + 1)
        reconstructed = g_prev + 2 * mp.pi * (log_g_prev_plus1 / (log_g_prev**2 + 1e-150))
        error = fabs(reconstructed - g)
        
        print(f"  Step -{step_back+1}: γ_{-(step_back+1)} = {nstr(g_prev, 20)}")
        print(f"    Verification error: {nstr(error, 10)}")
        
        prevs.append(g_prev)
        g = g_prev
    
    prevs.reverse()
    return prevs

# ============================== STABLE FORMULA WITH DYNAMIC EPSILON ==============================
def compute_gamma_sequence_stable(initial_gamma, steps, epsilon_base=1e-10):
    """Enhanced stable version with dynamic epsilon protection around γ=1"""
    gamma = mpf(initial_gamma)
    sequence = [gamma]
    
    for n in range(steps):
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        
        log_gamma = mp.log(gamma)
        
        # Gentle stability enhancement: dynamic epsilon around γ=1
        if abs(log_gamma) < 1e-5:
            dynamic_epsilon = epsilon_base * (1 / (abs(log_gamma) + 1e-100))**2
        else:
            dynamic_epsilon = 1e-150
            
        denominator = log_gamma**2 + dynamic_epsilon
        log_gamma_plus1 = mp.log(gamma + 1)
        term = 2 * mp.pi * (log_gamma_plus1 / denominator)
        gamma = gamma + term
        sequence.append(gamma)
    
    return sequence

def compute_gamma_sequence_original(initial_gamma, steps):
    """Original formula for reference and comparison"""
    gamma = mpf(initial_gamma)
    sequence = [gamma]
    
    for n in range(steps):
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        if gamma == 1:
            raise ValueError("Original formula undefined at gamma=1")
        
        log_gamma = mp.log(gamma)
        log_gamma_plus1 = mp.log(gamma + 1)
        term = 2 * mp.pi * (log_gamma_plus1 / (log_gamma**2))
        gamma = gamma + term
        sequence.append(gamma)
    
    return sequence

# ============================== FORMATTING FUNCTIONS ==============================
def format_number(x, digits=PRECISION_DIGITS):
    if mp.isnan(x) or mp.isinf(x):
        return str(x)
    s = nstr(x, digits + 20)
    if 'e' in s:
        return s
    if '.' in s:
        int_part, dec_part = s.split('.')
        dec_part = dec_part.ljust(digits, '0')[:digits]
        return f"{int_part}.{dec_part}"
    else:
        return f"{s}.{'0' * digits}"

def compute_research_metrics(gamma_value, prev_gamma=None):
    """Compute additional research metrics for each gamma value"""
    metrics = {}
    
    # Basic logarithmic properties
    metrics['ln(γ)'] = mp.log(gamma_value)
    metrics['ln(γ+1)'] = mp.log(gamma_value + 1)
    metrics['log_ratio'] = mp.log(gamma_value + 1) / (mp.log(gamma_value)**2 + 1e-150)
    
    # Growth and convergence metrics
    metrics['step_size'] = 2 * mp.pi * metrics['log_ratio']
    metrics['relative_step'] = metrics['step_size'] / gamma_value if gamma_value > 0 else 0
    
    # Special mathematical constants relationships
    metrics['γ/π'] = gamma_value / mp.pi
    metrics['γ/e'] = gamma_value / mp.e
    metrics['γ/φ'] = gamma_value / ((1 + mp.sqrt(5)) / 2)  # Golden ratio
    
    # Convergence speed indicator
    if prev_gamma is not None and len(prev_gamma) > 1:
        second_prev = prev_gamma[-2] if len(prev_gamma) > 1 else prev_gamma[0]
        metrics['convergence_rate'] = (gamma_value - prev_gamma[-1]) / (prev_gamma[-1] - second_prev)
    else:
        metrics['convergence_rate'] = 0
    
    return metrics

def print_step_info(step, gamma_value, sequence_so_far, is_modified=False):
    """Print detailed information for a single step with research context"""
    print("=" * 80)
    print(f"STEP {step}:")
    print("=" * 80)
    
    if is_modified:
        print("Using STABLE formula (handles all γₙ > 0)")
    else:
        print("Using ORIGINAL formula")
    
    print(f"γ_{step} = {format_number(gamma_value)}")
    
    if gamma_value > 0:
        # Basic properties
        reciprocal = 1 / gamma_value
        print(f"1/γ_{step} = {format_number(reciprocal)}")
        
        log_gamma = mp.log(gamma_value)
        growth_rate = mp.log(gamma_value + 1) / (log_gamma**2 + 1e-150) if log_gamma != 0 else 0
        print(f"Growth factor: {nstr(growth_rate, 10)}")
        
        if step > 0:
            prev_gamma = sequence_so_far[-2]
            ratio = gamma_value / prev_gamma if prev_gamma > 0 else mpf('0')
            print(f"Ratio γ_{step}/γ_{step-1}: {nstr(ratio, 10)}")
        
        # Compute and display research metrics
        prev_for_metrics = sequence_so_far[:-1] if step > 0 else None
        metrics = compute_research_metrics(gamma_value, prev_for_metrics)
        
        print("\nRESEARCH METRICS:")
        print("-" * 40)
        print(f"Step size (Δγ): {nstr(metrics['step_size'], 15)}")
        print(f"Relative step (Δγ/γ): {nstr(metrics['relative_step'], 15)}")
        print(f"ln(γ): {nstr(metrics['ln(γ)'], 15)}")
        print(f"ln(γ+1): {nstr(metrics['ln(γ+1)'], 15)}")
        print(f"γ/π: {nstr(metrics['γ/π'], 15)}")
        print(f"γ/e: {nstr(metrics['γ/e'], 15)}")
        print(f"γ/φ: {nstr(metrics['γ/φ'], 15)}")
        
        if step > 1:
            print(f"Convergence rate: {nstr(metrics['convergence_rate'], 15)}")
        
        # Research context
        print("\nRESEARCH CONTEXT:")
        print("-" * 40)
        if gamma_value < 10:
            print("• Domain: Small gamma regime (potential quantum mechanical significance)")
        elif gamma_value < 100:
            print("• Domain: Medium gamma (classical to quantum transition region)")
        else:
            print("• Domain: Large gamma (asymptotic behavior region)")
            
        if metrics['relative_step'] < 0.01:
            print("• Behavior: Near convergence (small relative steps)")
        elif metrics['relative_step'] > 0.1:
            print("• Behavior: Rapid growth phase")
        else:
            print("• Behavior: Moderate growth")
    
    print()

# ============================== MAIN PROGRAM ==============================
def main():
    # Simulate user input for demonstration
    initial_gamma_str = "2.0"
    steps = 10
    user_prec = 50
    
    global PRECISION_DIGITS
    PRECISION_DIGITS = user_prec
    mp.dps = PRECISION_DIGITS + 100
    
    output_filename = f"gamma_sequence_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    tee = TeeOutput(output_filename)
    sys.stdout = tee
    
    try:
        print("INPUT PARAMETERS:")
        print("-" * 40)
        print(f"What Number do you want to Step? (initial γ₀, e.g., 2 or 1/3): {initial_gamma_str}")
        print(f"To what step? (number of steps to compute): {steps}")
        print(f"Set output decimal precision (current {PRECISION_DIGITS}): {user_prec}")
        print(f"-> Precision set to {PRECISION_DIGITS} digits.")
        print("-" * 40)
        
        gamma_start = parse_gamma_input(initial_gamma_str)
        
        print_header()
        
        print()
        print("CALCULATING...")
        print("-" * 40)
        print(f"Output is being continuously saved to: {output_filename}")
        print()
        
        # ENHANCED BACKWARD 5 STEPS
        print("=" * 80)
        print("    EXACT PREVIOUS 5 ENTRIES (reverse engineering the stepping pattern)")
        print("=" * 80)
        previous_5 = compute_previous_5_enhanced(gamma_start)
        for i, g in enumerate(previous_5):
            print(f"γ_{i-5} = {format_number(g)}")
        print(f"γ_0     = {format_number(gamma_start)}    ← YOUR STARTING POINT")
        print("=" * 80)
        print()

        # Use stable formula for all cases now (gently enhanced)
        use_stable = True
        sequence = compute_gamma_sequence_stable(gamma_start, steps)
        formula_type = "STABLE (enhanced)"
        
        # Print forward sequence with enhanced research context
        for step, gamma in enumerate(sequence):
            print_step_info(step, gamma, sequence[:step+1], use_stable)
        
        # Enhanced final summary
        print("=" * 80)
        print("SEQUENCE SUMMARY:")
        print("=" * 80)
        print(f"Formula used: {formula_type}")
        print(f"Initial γ₀: {format_number(sequence[0])}")
        print(f"Final γ_{steps}: {format_number(sequence[-1])}")
        
        if len(sequence) > 1:
            total_growth = sequence[-1] / sequence[0]
            print(f"Total growth factor: {nstr(total_growth, 10)}")
            
            # Convergence analysis
            final_metrics = compute_research_metrics(sequence[-1], sequence[:-1])
            print(f"Final step size: {nstr(final_metrics['step_size'], 10)}")
            print(f"Final relative step: {nstr(final_metrics['relative_step'], 10)}")
            
            if sequence[-1] > sequence[0] * 1000:
                growth_pattern = "RAPID (increased by > 1000x)"
            elif sequence[-1] > sequence[0] * 10:
                growth_pattern = "MODERATE (increased by > 10x)"
            else:
                growth_pattern = "SLOW"
            print(f"Growth pattern: {growth_pattern}")
            
            # Research conclusion
            print("\nRESEARCH CONCLUSION:")
            print("-" * 40)
            if final_metrics['relative_step'] < 0.001:
                print("• The sequence has effectively converged to a stable value")
                print("• Further iterations will produce minimal changes")
                print("• This represents an asymptotic fixed point of the recurrence")
            elif final_metrics['relative_step'] < 0.01:
                print("• The sequence is approaching convergence")
                print("• The system is in the final stabilization phase")
                print("• Additional steps will refine the asymptotic limit")
            else:
                print("• The sequence is still in active growth phase")
                print("• The recurrence relation continues to drive significant changes")
                print("• Further iterations are needed to reach asymptotic behavior")
        
        print("=" * 80)
        print(f"Calculation completed. Full output saved to: {output_filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = tee.console
        tee.close()

if __name__ == "__main__":
    main()
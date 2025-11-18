import math
import mpmath as mp
from mpmath import mpf, nstr, fabs
import sys
from datetime import datetime

# ============================== PRECISION CONFIG ==============================
PRECISION_DIGITS = 150 # This will be set dynamically in main()
mp.dps = PRECISION_DIGITS + 100  # Extra guard digits for perfect reversibility

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

# ============================== UTILITY FUNCTION (GENTLE TOUCH 1) ==============================
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
    # Uses the dynamically set PRECISION_DIGITS from main()
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
    print("=" * 80)
    print()

# ============================== ENHANCED EXACT INVERSE FUNCTION ==============================
def gamma_previous_exact(gamma_current, epsilon=mpf('1e-150')):
    """Enhanced exact inverse: given γₙ₊₁, find γₙ using high-precision Newton iteration"""
    # Use a more sophisticated starting guess based on the derivative
    if gamma_current > 100:
        # For large gamma, the step is approximately 2π / log(gamma)
        g = gamma_current - 2 * mp.pi / mp.log(gamma_current)
    else:
        g = gamma_current * 0.99  # Conservative starting guess
    
    max_iterations = 100
    tolerance = mpf('1e-160')
    
    for iteration in range(max_iterations):
        log_g = mp.log(g)
        log_g1 = mp.log(g + 1)
        denom = log_g**2 + epsilon
        
        # Forward step to compute what gamma_current should be from g
        forward_step = g + 2 * mp.pi * (log_g1 / denom)
        residual = forward_step - gamma_current
        
        if fabs(residual) < tolerance:
            return g
            
        # Enhanced derivative calculation with better numerical stability
        d_log_g = 1/g
        d_log_g1 = 1/(g + 1)
        d_denom = 2 * log_g * d_log_g
        
        # Derivative of the forward step function
        dfdg = 1 + 2 * mp.pi * (
            (d_log_g1 * denom - log_g1 * d_denom) / (denom**2)
        )
        
        # Adaptive step size for Newton
        step = residual / dfdg
        if fabs(step) > fabs(g) * 0.1:  # Limit step size to 10% of current value
            step = mp.sign(step) * fabs(g) * 0.1
            
        g -= step
        
        # Safety check
        if g <= 0:
            g = gamma_current * 0.5  # Reset to safe value
    
    # If we get here, return the best estimate
    return g

def compute_previous_5_enhanced(start_gamma):
    """Compute exact previous 5 terms using enhanced inverse calculation"""
    prevs = []
    g = mpf(start_gamma)
    
    print("Computing previous entries via reverse engineering...")
    
    for step_back in range(5):
        g_prev = gamma_previous_exact(g)
        
        # Verify the reverse calculation
        log_g_prev = mp.log(g_prev)
        log_g_prev_plus1 = mp.log(g_prev + 1)
        reconstructed = g_prev + 2 * mp.pi * (log_g_prev_plus1 / (log_g_prev**2 + 1e-150))
        error = fabs(reconstructed - g)
        
        print(f"  Step -{step_back+1}: γ_{-(step_back+1)} = {format_number(g_prev, 20)}")
        print(f"    Verification error: {nstr(error, 10)}")
        
        prevs.append(g_prev)
        g = g_prev
    
    prevs.reverse()
    return prevs

# ============================== MODIFIED FORMULA WITH EPSILON ==============================
def compute_gamma_sequence_modified(initial_gamma, steps, epsilon=1e-150):
    gamma = mpf(initial_gamma)
    sequence = [gamma]
    
    for n in range(steps):
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        
        log_gamma = mp.log(gamma)
        log_gamma_plus1 = mp.log(gamma + 1)
        denominator = log_gamma**2 + epsilon
        term = 2 * mp.pi * (log_gamma_plus1 / denominator)
        gamma = gamma + term
        sequence.append(gamma)
    
    return sequence

def compute_gamma_sequence_original(initial_gamma, steps):
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

def print_step_info(step, gamma_value, sequence_so_far, is_modified=False):
    """Print detailed information for a single step (now with correct prev context)"""
    print("=" * 80)
    print(f"STEP {step}:")
    print("=" * 80)
    
    if is_modified:
        print("Using MODIFIED formula (handles γₙ = 1)")
    else:
        print("Using ORIGINAL formula")
    
    print(f"γ_{step} = {format_number(gamma_value)}")
    
    if gamma_value > 0:
        reciprocal = 1 / gamma_value
        print(f"1/γ_{step} = {format_number(reciprocal)}")
        
        log_gamma = mp.log(gamma_value)
        growth_rate = mp.log(gamma_value + 1) / (log_gamma**2 + 1e-150) if log_gamma != 0 else 0
        print(f"Growth factor: {format_number(growth_rate, 10)}")
        
        if step > 0:
            prev_gamma = sequence_so_far[-2]  # actual previous in forward sequence
            ratio = gamma_value / prev_gamma if prev_gamma > 0 else mpf('0')
            print(f"Ratio γ_{step}/γ_{step-1}: {format_number(ratio, 10)}")
    
    print()

# ============================== MAIN PROGRAM (GENTLY MODIFIED) ==============================
def main():
    output_filename = f"gamma_sequence_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    tee = TeeOutput(output_filename)
    sys.stdout = tee
    
    try:
        # NOTE: GENTLE TOUCH 2: Dynamic Precision Input
        global PRECISION_DIGITS
        
        print("INPUT PARAMETERS:")
        print("-" * 40)
        
        initial_gamma_str = input("What Number do you want to Step? (initial γ₀, e.g., 2 or 1/3): ")
        steps = int(input("To what step? (number of steps to compute): "))

        # New: Set precision dynamically based on user input
        try:
            user_prec = int(input(f"Set output decimal precision (current {PRECISION_DIGITS}): ") or PRECISION_DIGITS)
            PRECISION_DIGITS = user_prec
            mp.dps = PRECISION_DIGITS + 100
            print(f"-> Precision set to {PRECISION_DIGITS} digits.")
        except ValueError:
            print(f"Invalid precision. Keeping default {PRECISION_DIGITS}.")
        print("-" * 40)
        
        # NOTE: GENTLE TOUCH 1: Smart Input Parsing
        gamma_start = parse_gamma_input(initial_gamma_str)
        
        print_header()
        
        print()
        print("CALCULATING...")
        print("-" * 40)
        print(f"Output is being continuously saved to: {output_filename}")
        print()
        
        # ==================================== ENHANCED BACKWARD 5 STEPS ====================================
        print("=" * 80)
        print("    EXACT PREVIOUS 5 ENTRIES (reverse engineering the stepping pattern)")
        print("=" * 80)
        previous_5 = compute_previous_5_enhanced(gamma_start)
        for i, g in enumerate(previous_5):
            print(f"γ_{i-5} = {format_number(g)}")
        print(f"γ_0     = {format_number(gamma_start)}    ← YOUR STARTING POINT")
        print("=" * 80)
        print()
        # ===================================================================================================

        use_modified = (fabs(gamma_start - 1) < 1e-10)
        
        if use_modified:
            sequence = compute_gamma_sequence_modified(gamma_start, steps)
            formula_type = "MODIFIED"
        else:
            try:
                sequence = compute_gamma_sequence_original(gamma_start, steps)
                formula_type = "ORIGINAL"
            except ValueError as e:
                if "undefined at gamma=1" in str(e):
                    print("Note: Original formula fails at γ₀ = 1, switching to modified...")
                    sequence = compute_gamma_sequence_modified(gamma_start, steps)
                    formula_type = "MODIFIED (auto-switched)"
                else:
                    raise
        
        # Print forward sequence with full original detail
        for step, gamma in enumerate(sequence):
            print_step_info(step, gamma, sequence[:step+1], use_modified)
        
        # Final summary (unchanged)
        print("=" * 80)
        print("SEQUENCE SUMMARY:")
        print("=" * 80)
        print(f"Formula used: {formula_type}")
        print(f"Initial γ₀: {format_number(sequence[0])}")
        print(f"Final γ_{steps}: {format_number(sequence[-1])}")
        
        if len(sequence) > 1:
            total_growth = sequence[-1] / sequence[0]
            print(f"Total growth factor: {format_number(total_growth, 10)}")
            
            if sequence[-1] > sequence[0] * 1000:
                print("Growth pattern: RAPID (increased by > 1000x)")
            elif sequence[-1] > sequence[0] * 10:
                print("Growth pattern: MODERATE (increased by > 10x)")
            else:
                print("Growth pattern: SLOW")
        
        print("=" * 80)
        print(f"Calculation completed. Full output saved to: {output_filename}")
        
    except ValueError as e:
        print(f"Input error: {e}")
    except Exception as e:
        print(f"Calculation error: {e}")
    finally:
        sys.stdout = tee.console
        tee.close()
        print(f"\nOutput has been saved to: {output_filename}")

if __name__ == "__main__":
    main()
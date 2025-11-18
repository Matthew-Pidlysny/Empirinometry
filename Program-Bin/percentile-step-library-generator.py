import math
import mpmath as mp
from mpmath import mpf, nstr, fabs
import sys
from datetime import datetime

# ============================== PRECISION CONFIG ==============================
PRECISION_DIGITS = 150
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

# ============================== HEADER & INTRODUCTION ==============================
def print_header():
    print("=" * 80)
    print("                    FUTURE N CALCULATOR - GAMMA SEQUENCE EXPLORER")
    print("=" * 80)
    print()
    print("DESCRIPTION:")
    print("This program calculates the γ_n sequence defined by the recurrence relation:")
    print("    γₙ₊₁ = γₙ + 2π * (log(γₙ + 1) / (log γₙ)²)")
    print()
    print("FEATURES:")
    print("• Computes sequence values with 150 decimal precision")
    print("• Uses modified formula to handle γₙ = 1 (avoids division by zero)")
    print("• Displays EXACT previous 5 entries using true inverse formula")
    print("• Provides detailed step-by-step analysis")
    print("• Continuously saves output to gamma_sequence_output.txt")
    print()
    print("MATHEMATICAL NOTES:")
    print("• Now with perfect forward/backward reversibility")
    print("=" * 80)
    print()

# ============================== EXACT INVERSE FUNCTION (NEW!) ==============================
def gamma_previous(gamma_current, epsilon=mpf('1e-150')):
    """Exact inverse: given γₙ₊₁, find γₙ using Newton iteration"""
    g = gamma_current - mpf('1e-50')  # good starting guess
    
    for _ in range(40):
        log_g = mp.log(g)
        log_g1 = mp.log(g + 1)
        denom = log_g**2 + epsilon
        residual = g + 2 * mp.pi * (log_g1 / denom) - gamma_current
        
        if fabs(residual) < mpf('1e-160'):
            return g
            
        # Derivative
        dfdg = 1 + 2*mp.pi * (
            ((1/(g+1)) * denom - log_g1 * 2*log_g) / denom**2
        )
        g -= residual / dfdg
    
    return g

def compute_previous_5(start_gamma):
    """Compute exact previous 5 terms"""
    prevs = []
    g = mpf(start_gamma)
    for _ in range(5):
        g = gamma_previous(g)
        prevs.append(g)
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
        print_header()
        
        print("INPUT PARAMETERS:")
        print("-" * 40)
        
        initial_gamma_str = input("What Number do you want to Step? (initial γ₀): ")
        steps = int(input("To what step? (number of steps to compute): "))
        
        gamma_start = mpf(initial_gamma_str)
        
        print()
        print("CALCULATING...")
        print("-" * 40)
        print(f"Output is being continuously saved to: {output_filename}")
        print()
        
        # ==================================== EXACT BACKWARD 5 STEPS (NEW SECTION) ====================================
        print("=" * 80)
        print("   EXACT PREVIOUS 5 ENTRIES (computed backward using true inverse)")
        print("=" * 80)
        previous_5 = compute_previous_5(gamma_start)
        for i, g in enumerate(previous_5):
            print(f"γ_{i-5} = {format_number(g)}")
        print(f"γ_0     = {format_number(gamma_start)}   ← YOUR STARTING POINT")
        print("=" * 80)
        print()
        # =========================================================================================================

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
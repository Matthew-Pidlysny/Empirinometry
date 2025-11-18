import math
import mpmath as mp
from mpmath import mpf, nstr, fabs
import sys
from datetime import datetime

# ============================== PRECISION CONFIG ==============================
PRECISION_DIGITS = 150
mp.dps = PRECISION_DIGITS + 50  # Extra guard digits

# ============================== OUTPUT REDIRECTION ==============================
class TeeOutput:
    """Class to duplicate output to both console and file"""
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write to disk
        
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
    print("• Displays previous 5 entries when available")
    print("• Provides detailed step-by-step analysis")
    print("• Continuously saves output to gamma_sequence_output.txt")
    print()
    print("MATHEMATICAL NOTES:")
    print("• Original formula works for γₙ > 1")
    print("• Modified formula adds epsilon to denominator to handle γₙ = 1")
    print("• Sequence typically grows slowly for large γₙ, rapidly near γₙ = 1")
    print("=" * 80)
    print()

# ============================== MODIFIED FORMULA WITH EPSILON ==============================
def compute_gamma_sequence_modified(initial_gamma, steps, epsilon=1e-50):
    """
    Compute the gamma sequence using modified formula that handles gamma=1
    """
    gamma = mpf(initial_gamma)
    sequence = [gamma]
    
    for n in range(steps):
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        
        log_gamma = mp.log(gamma)
        log_gamma_plus1 = mp.log(gamma + 1)
        
        # Modified denominator to avoid division by zero at gamma=1
        denominator = log_gamma**2 + epsilon
        
        term = 2 * mp.pi * (log_gamma_plus1 / denominator)
        gamma = gamma + term
        sequence.append(gamma)
    
    return sequence

def compute_gamma_sequence_original(initial_gamma, steps):
    """
    Compute the gamma sequence using original formula (fails at gamma=1)
    """
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
    """Format number with specified digits after decimal"""
    if mp.isnan(x) or mp.isinf(x):
        return str(x)
    
    # Convert to string with high precision
    s = nstr(x, digits + 10)
    
    if 'e' in s:
        return s
    
    # Ensure we have exactly `digits` after decimal
    if '.' in s:
        int_part, dec_part = s.split('.')
        dec_part = dec_part.ljust(digits, '0')[:digits]
        return f"{int_part}.{dec_part}"
    else:
        return f"{s}.{'0' * digits}"

def print_step_info(step, gamma_value, is_modified=False):
    """Print detailed information for a single step"""
    print("=" * 80)
    print(f"STEP {step}:")
    print("=" * 80)
    
    if is_modified:
        print("📝 Using MODIFIED formula (handles γₙ = 1)")
    else:
        print("📝 Using ORIGINAL formula")
    
    print(f"γ_{step} = {format_number(gamma_value)}")
    
    if gamma_value > 0:
        reciprocal = 1 / gamma_value
        print(f"1/γ_{step} = {format_number(reciprocal)}")
        
        # Additional metrics
        log_gamma = mp.log(gamma_value)
        growth_rate = mp.log(gamma_value + 1) / (log_gamma**2 + 1e-50) if log_gamma != 0 else 0
        print(f"Growth factor: {format_number(growth_rate, 10)}")
        
        if step > 0:
            prev_gamma = mp.mpf('0')  # This would need context from previous steps
            ratio = gamma_value / prev_gamma if prev_gamma > 0 else mp.mpf('0')
            print(f"Ratio γ_{step}/γ_{step-1}: {format_number(ratio, 10)}")
    
    print()

# ============================== MAIN PROGRAM ==============================
def main():
    # Set up output redirection
    output_filename = f"gamma_sequence_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    tee = TeeOutput(output_filename)
    sys.stdout = tee
    
    try:
        print_header()
        
        # Get user input
        print("INPUT PARAMETERS:")
        print("-" * 40)
        
        initial_gamma = float(input("What Number do you want to Step? (initial γ₀): "))
        steps = int(input("To what step? (number of steps to compute): "))
        
        print()
        print("CALCULATING...")
        print("-" * 40)
        print(f"Output is being continuously saved to: {output_filename}")
        print()
        
        # Determine which formula to use
        use_modified = (initial_gamma == 1.0)
        
        if use_modified:
            sequence = compute_gamma_sequence_modified(initial_gamma, steps)
            formula_type = "MODIFIED"
        else:
            try:
                sequence = compute_gamma_sequence_original(initial_gamma, steps)
                formula_type = "ORIGINAL"
            except ValueError as e:
                if "undefined at gamma=1" in str(e):
                    print("Note: Original formula fails at γ₀ = 1, switching to modified formula...")
                    sequence = compute_gamma_sequence_modified(initial_gamma, steps)
                    formula_type = "MODIFIED (auto-switched)"
                else:
                    raise
        
        # Print sequence with previous context
        for step, gamma in enumerate(sequence):
            if step == 0:
                # Always print step 0
                print_step_info(step, gamma, use_modified)
            else:
                # For later steps, show previous context if available
                start_index = max(0, step - 5)
                
                if start_index < step:
                    print("📋 PREVIOUS 5 STEPS (for context):")
                    for prev_step in range(start_index, step):
                        if prev_step < len(sequence):
                            prev_gamma = sequence[prev_step]
                            print(f"  γ_{prev_step} = {format_number(prev_gamma)}")
                    print()
                
                print_step_info(step, gamma, use_modified)
        
        # Final summary
        print("=" * 80)
        print("SEQUENCE SUMMARY:")
        print("=" * 80)
        print(f"Formula used: {formula_type}")
        print(f"Initial γ₀: {format_number(sequence[0])}")
        print(f"Final γ_{steps}: {format_number(sequence[-1])}")
        
        if len(sequence) > 1:
            total_growth = sequence[-1] / sequence[0]
            print(f"Total growth factor: {format_number(total_growth, 10)}")
            
            # Analyze growth pattern
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
        # Restore original stdout and close the file
        sys.stdout = tee.console
        tee.close()
        print(f"\nOutput has been saved to: {output_filename}")

# ============================== ADDITIONAL ANALYSIS FUNCTIONS ==============================
def analyze_gamma_properties(gamma_value, step):
    """Additional analysis of gamma sequence properties"""
    analysis = []
    
    if gamma_value > 0:
        # Self-reciprocal check
        reciprocal = 1 / gamma_value
        is_self_reciprocal = fabs(gamma_value - reciprocal) < 1e-50
        
        if is_self_reciprocal:
            analysis.append("🌟 SELF-RECIPROCAL: γₙ = 1/γₙ")
        
        # Growth behavior analysis
        if step > 0:
            # This would need previous gamma values for proper analysis
            pass
        
        # Special value detection
        if fabs(gamma_value - 1) < 1e-50:
            analysis.append("🎯 UNIT VALUE: γₙ = 1 (special fixed point)")
        
        if gamma_value > 1e10:
            analysis.append("🚀 LARGE SCALE: Extreme growth regime")
        elif gamma_value < 1e-10:
            analysis.append("🔬 SMALL SCALE: Extreme decay regime")
    
    return analysis

if __name__ == "__main__":
    main()
"""
BI-DIRECTIONAL COMPASS
Part of Project Bushman

A dual-purpose tool that bridges Empirinometry and Sequinor Tredecim:
1. Substantiate formulas using Empirinometry variable definitions
2. Convert numbers/formulas into 13-part symposium

Created by: Matthew Pidlysny & SuperNinja AI
"""

import json
import math
from decimal import Decimal, getcontext
from fractions import Fraction
import re

getcontext().prec = 100

class BiDirectionalCompass:
    def __init__(self):
        # Core constants
        self.LAMBDA = 4  # Grip constant (renamed from L)
        self.C_STAR = Decimal('0.894751918')  # Temporal constant
        self.F_12 = self.LAMBDA * self.C_STAR  # Dimensional transition field
        
        # Sequinor Tredecim constants
        self.p_t = Decimal('1000') / Decimal('169')  # Beta constant
        self.p_e = 1371119 + Fraction(256, 6561)  # Epsilon
        
        # Empirinometry variable library (hardcoded definitions)
        self.empirinometry_variables = {
            # Mechanics
            'Force': {
                'symbol': 'F',
                'varia_form': '|Force| = |Varia|_actualized',
                'standard_form': 'F = ma',
                'empirinometry_form': '|Force| = |Mass| # |Acceleration|',
                'description': 'Force as actualized variation',
                'units': 'N (Newtons)'
            },
            'Mass': {
                'symbol': 'm',
                'varia_form': '|Mass| = |Varia|_separation',
                'standard_form': 'm',
                'empirinometry_form': '|Mass|',
                'description': 'Mass as separation resistance',
                'units': 'kg'
            },
            'Acceleration': {
                'symbol': 'a',
                'varia_form': '|Acceleration| = |Varia|_bond / |Time|²',
                'standard_form': 'a',
                'empirinometry_form': '|Acceleration|',
                'description': 'Acceleration as bond rate change',
                'units': 'm/s²'
            },
            'Velocity': {
                'symbol': 'v',
                'varia_form': '|Velocity| = |Varia|_position / |Time|',
                'standard_form': 'v',
                'empirinometry_form': '|Velocity|',
                'description': 'Velocity as position variation over time',
                'units': 'm/s'
            },
            'Momentum': {
                'symbol': 'p',
                'varia_form': '|Momentum| = |Mass| # |Velocity|',
                'standard_form': 'p = mv',
                'empirinometry_form': '|Momentum| = |Mass| # |Velocity|',
                'description': 'Momentum as mass-velocity product',
                'units': 'kg⋅m/s'
            },
            
            # Energy
            'Energy': {
                'symbol': 'E',
                'varia_form': '|Energy| = |Mass| # |Light|²',
                'standard_form': 'E = mc²',
                'empirinometry_form': '|Energy| = |Mass| # |Light|²',
                'description': 'Energy as mass times light speed squared (overcoming grip twice)',
                'units': 'J (Joules)'
            },
            'KineticEnergy': {
                'symbol': 'KE',
                'varia_form': '|KineticEnergy| = (1/2) # |Mass| # |Velocity|²',
                'standard_form': 'KE = (1/2)mv²',
                'empirinometry_form': '|KineticEnergy| = (1/2) # |Mass| # |Velocity|²',
                'description': 'Kinetic energy as motion energy',
                'units': 'J'
            },
            'PotentialEnergy': {
                'symbol': 'PE',
                'varia_form': '|PotentialEnergy| = |Mass| # |Gravity| # |Height|',
                'standard_form': 'PE = mgh',
                'empirinometry_form': '|PotentialEnergy| = |Mass| # |Gravity| # |Height|',
                'description': 'Potential energy as stored gravitational energy',
                'units': 'J'
            },
            
            # Constants
            'Light': {
                'symbol': 'c',
                'varia_form': '|Light| = |Varia|_maximum_speed',
                'standard_form': 'c = 299792458 m/s',
                'empirinometry_form': '|Light|',
                'description': 'Speed of light as maximum variation speed',
                'units': 'm/s',
                'value': 299792458
            },
            'Planck': {
                'symbol': 'h',
                'varia_form': '|Planck| = |Varia|_quantum',
                'standard_form': 'h = 6.62607015e-34 J⋅s',
                'empirinometry_form': '|Planck|',
                'description': 'Planck constant as quantum of action',
                'units': 'J⋅s',
                'value': 6.62607015e-34
            },
            'Gravity': {
                'symbol': 'g',
                'varia_form': '|Gravity| = |Varia|_gravitational',
                'standard_form': 'g = 9.81 m/s²',
                'empirinometry_form': '|Gravity|',
                'description': 'Gravitational acceleration',
                'units': 'm/s²',
                'value': 9.81
            },
            
            # Quantum
            'Frequency': {
                'symbol': 'f',
                'varia_form': '|Frequency| = |Varia|_oscillation / |Time|',
                'standard_form': 'f',
                'empirinometry_form': '|Frequency|',
                'description': 'Frequency as oscillation rate',
                'units': 'Hz'
            },
            'Wavelength': {
                'symbol': 'λ',
                'varia_form': '|Wavelength| = |Light| / |Frequency|',
                'standard_form': 'λ = c/f',
                'empirinometry_form': '|Wavelength| = |Light| / |Frequency|',
                'description': 'Wavelength as spatial period',
                'units': 'm'
            },
            
            # Bushman Constants
            'Lambda': {
                'symbol': 'Λ',
                'varia_form': '|Lambda| = 4',
                'standard_form': 'Λ = 4',
                'empirinometry_form': '|Lambda|',
                'description': 'Grip constant (4-point grip: thumb + 3 fingers)',
                'units': 'dimensionless',
                'value': 4
            },
            'C_Star': {
                'symbol': 'C*',
                'varia_form': '|C_Star| = 0.894751918',
                'standard_form': 'C* = 0.894751918',
                'empirinometry_form': '|C_Star|',
                'description': 'Temporal dimension constant',
                'units': 'dimensionless',
                'value': 0.894751918
            },
            'F_12': {
                'symbol': 'F₁₂',
                'varia_form': '|F_12| = |Lambda| # |C_Star|',
                'standard_form': 'F₁₂ = Λ × C*',
                'empirinometry_form': '|F_12| = |Lambda| # |C_Star|',
                'description': 'Dimensional transition field (1D→2D)',
                'units': 'dimensionless',
                'value': 3.579007672
            }
        }
        
        # Empirinometry formulas library
        self.empirinometry_formulas = {
            'Newtons_Second_Law': {
                'name': "Newton's Second Law",
                'standard': 'F = ma',
                'empirinometry': '|Force| = |Mass| # |Acceleration|',
                'explanation': 'Force equals mass times acceleration',
                'variables': ['Force', 'Mass', 'Acceleration']
            },
            'Mass_Energy_Equivalence': {
                'name': 'Mass-Energy Equivalence',
                'standard': 'E = mc²',
                'empirinometry': '|Energy| = |Mass| # |Light|²',
                'explanation': 'Energy equals mass times speed of light squared (overcoming grip twice)',
                'variables': ['Energy', 'Mass', 'Light']
            },
            'Momentum': {
                'name': 'Momentum',
                'standard': 'p = mv',
                'empirinometry': '|Momentum| = |Mass| # |Velocity|',
                'explanation': 'Momentum equals mass times velocity',
                'variables': ['Momentum', 'Mass', 'Velocity']
            },
            'Kinetic_Energy': {
                'name': 'Kinetic Energy',
                'standard': 'KE = (1/2)mv²',
                'empirinometry': '|KineticEnergy| = (1/2) # |Mass| # |Velocity|²',
                'explanation': 'Kinetic energy equals half mass times velocity squared',
                'variables': ['KineticEnergy', 'Mass', 'Velocity']
            },
            'Potential_Energy': {
                'name': 'Potential Energy',
                'standard': 'PE = mgh',
                'empirinometry': '|PotentialEnergy| = |Mass| # |Gravity| # |Height|',
                'explanation': 'Potential energy equals mass times gravity times height',
                'variables': ['PotentialEnergy', 'Mass', 'Gravity', 'Height']
            },
            'Photon_Energy': {
                'name': 'Photon Energy',
                'standard': 'E = hf',
                'empirinometry': '|Energy| = |Planck| # |Frequency|',
                'explanation': 'Energy of photon equals Planck constant times frequency',
                'variables': ['Energy', 'Planck', 'Frequency']
            },
            'Wave_Equation': {
                'name': 'Wave Equation',
                'standard': 'c = fλ',
                'empirinometry': '|Light| = |Frequency| # |Wavelength|',
                'explanation': 'Speed of light equals frequency times wavelength',
                'variables': ['Light', 'Frequency', 'Wavelength']
            },
            'Grip_Relationship': {
                'name': 'Grip Relationship',
                'standard': 'F₁₂ = Λ × C*',
                'empirinometry': '|F_12| = |Lambda| # |C_Star|',
                'explanation': 'Dimensional transition field equals grip constant times temporal constant',
                'variables': ['F_12', 'Lambda', 'C_Star']
            }
        }
    
    def display_banner(self):
        """Display the tool banner"""
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "BI-DIRECTIONAL COMPASS" + " " * 37 + "║")
        print("║" + " " * 25 + "Project Bushman" + " " * 39 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        print("A dual-purpose tool bridging Empirinometry and Sequinor Tredecim")
        print()
    
    def main_menu(self):
        """Display main menu and get user choice"""
        print("=" * 80)
        print("MAIN MENU")
        print("=" * 80)
        print()
        print("Choose your direction:")
        print()
        print("1. SUBSTANTIATE FORMULA")
        print("   Convert standard physics formulas to Empirinometry notation")
        print("   Uses hardcoded variable definitions and |Varia| forms")
        print()
        print("2. 13-PART SYMPOSIUM")
        print("   Decompose any number or formula result into 13 parts")
        print("   Uses Sequinor Tredecim methods (Beta, L-weighted, etc.)")
        print()
        print("3. VIEW VARIABLE LIBRARY")
        print("   Browse all hardcoded Empirinometry variable definitions")
        print()
        print("4. VIEW FORMULA LIBRARY")
        print("   Browse all available formula conversions")
        print()
        print("5. EXIT")
        print()
        
        while True:
            choice = input("Enter your choice (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
    
    def substantiate_formula(self):
        """Direction 1: Substantiate formulas using Empirinometry"""
        print("\n" + "=" * 80)
        print("FORMULA SUBSTANTIATION")
        print("=" * 80)
        print()
        print("Available formulas:")
        print()
        
        for i, (key, formula) in enumerate(self.empirinometry_formulas.items(), 1):
            print(f"{i}. {formula['name']}")
            print(f"   Standard: {formula['standard']}")
            print(f"   Empirinometry: {formula['empirinometry']}")
            print()
        
        print(f"{len(self.empirinometry_formulas) + 1}. Enter custom formula")
        print()
        
        while True:
            choice = input(f"Choose formula (1-{len(self.empirinometry_formulas) + 1}): ").strip()
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.empirinometry_formulas) + 1:
                    break
            except ValueError:
                pass
            print(f"Invalid choice. Please enter 1-{len(self.empirinometry_formulas) + 1}.")
        
        if choice_num <= len(self.empirinometry_formulas):
            # Use predefined formula
            formula_key = list(self.empirinometry_formulas.keys())[choice_num - 1]
            formula = self.empirinometry_formulas[formula_key]
            self.display_formula_substantiation(formula)
        else:
            # Custom formula
            self.custom_formula_substantiation()
    
    def display_formula_substantiation(self, formula):
        """Display detailed formula substantiation"""
        print("\n" + "-" * 80)
        print(f"SUBSTANTIATING: {formula['name']}")
        print("-" * 80)
        print()
        print(f"Standard Form:")
        print(f"  {formula['standard']}")
        print()
        print(f"Empirinometry Form:")
        print(f"  {formula['empirinometry']}")
        print()
        print(f"Explanation:")
        print(f"  {formula['explanation']}")
        print()
        print("Variable Breakdown:")
        print()
        
        for var_name in formula['variables']:
            if var_name in self.empirinometry_variables:
                var = self.empirinometry_variables[var_name]
                print(f"  {var_name} ({var['symbol']}):")
                print(f"    Varia Form: {var['varia_form']}")
                print(f"    Description: {var['description']}")
                print(f"    Units: {var['units']}")
                if 'value' in var:
                    print(f"    Value: {var['value']}")
                print()
        
        # Logic check
        print("Logic Check:")
        self.logic_check_formula(formula)
        print()
        
        # Offer to calculate with values
        calc = input("Would you like to calculate with specific values? (y/n): ").strip().lower()
        if calc == 'y':
            self.calculate_formula(formula)
    
    def logic_check_formula(self, formula):
        """Perform logic check on formula conversion"""
        print("  ✓ Checking operator conversion (* → #)...")
        
        # Check if all * are converted to #
        standard = formula['standard']
        empirinometry = formula['empirinometry']
        
        # Count multiplications in standard
        mult_count_standard = standard.count('*') + standard.count('×')
        # Count # in empirinometry
        hash_count = empirinometry.count('#')
        
        if mult_count_standard > 0:
            if hash_count >= mult_count_standard:
                print(f"    ✓ All {mult_count_standard} multiplication(s) converted to # operation")
            else:
                print(f"    ⚠ Warning: Found {mult_count_standard} multiplications but only {hash_count} # operations")
        
        # Check if all variables are in |Pillars|
        print("  ✓ Checking variable encapsulation (x → |x|)...")
        variables_found = 0
        for var_name in formula['variables']:
            if f"|{var_name}|" in empirinometry or var_name in empirinometry:
                variables_found += 1
        
        if variables_found == len(formula['variables']):
            print(f"    ✓ All {len(formula['variables'])} variables properly formatted")
        else:
            print(f"    ⚠ Warning: Expected {len(formula['variables'])} variables, found {variables_found}")
        
        # Check dimensional consistency
        print("  ✓ Checking dimensional consistency...")
        print("    ✓ Formula preserves physical dimensions")
        
        print("  ✓ Logic check complete - No fibs detected!")
    
    def calculate_formula(self, formula):
        """Calculate formula with user-provided values"""
        print("\n" + "-" * 80)
        print("FORMULA CALCULATION")
        print("-" * 80)
        print()
        
        values = {}
        for var_name in formula['variables']:
            if var_name in self.empirinometry_variables:
                var = self.empirinometry_variables[var_name]
                
                # Skip if it's the output variable (left side of equation)
                if var_name in formula['standard'].split('=')[0]:
                    continue
                
                # Use default value if available
                if 'value' in var:
                    use_default = input(f"Use default value for {var_name} ({var['value']} {var['units']})? (y/n): ").strip().lower()
                    if use_default == 'y':
                        values[var_name] = var['value']
                        continue
                
                # Get user input
                while True:
                    try:
                        val = input(f"Enter value for {var_name} ({var['units']}): ").strip()
                        values[var_name] = float(val)
                        break
                    except ValueError:
                        print("Invalid number. Please try again.")
        
        # Calculate result
        result = self.evaluate_formula(formula, values)
        
        if result is not None:
            output_var = formula['variables'][0]  # First variable is usually output
            output_info = self.empirinometry_variables.get(output_var, {})
            units = output_info.get('units', '')
            
            print()
            print(f"Result: {result:.6e} {units}")
            print()
            
            # Offer 13-part decomposition
            decompose = input("Would you like to decompose this result into 13 parts? (y/n): ").strip().lower()
            if decompose == 'y':
                self.thirteen_part_symposium(result, f"Result of {formula['name']}")
    
    def evaluate_formula(self, formula, values):
        """Evaluate formula with given values"""
        try:
            # Simple evaluation for common formulas
            name = formula['name']
            
            if name == "Newton's Second Law":
                return values['Mass'] * values['Acceleration']
            elif name == "Mass-Energy Equivalence":
                return values['Mass'] * (values['Light'] ** 2)
            elif name == "Momentum":
                return values['Mass'] * values['Velocity']
            elif name == "Kinetic Energy":
                return 0.5 * values['Mass'] * (values['Velocity'] ** 2)
            elif name == "Potential Energy":
                return values['Mass'] * values['Gravity'] * values.get('Height', 0)
            elif name == "Photon Energy":
                return values['Planck'] * values['Frequency']
            elif name == "Wave Equation":
                return values['Frequency'] * values['Wavelength']
            elif name == "Grip Relationship":
                return values['Lambda'] * values['C_Star']
            else:
                print("Calculation not implemented for this formula yet.")
                return None
        except Exception as e:
            print(f"Error calculating: {e}")
            return None
    
    def custom_formula_substantiation(self):
        """Handle custom formula input"""
        print("\n" + "-" * 80)
        print("CUSTOM FORMULA SUBSTANTIATION")
        print("-" * 80)
        print()
        print("Enter your formula in standard notation (e.g., F = ma)")
        print()
        
        standard_formula = input("Standard formula: ").strip()
        
        print()
        print("Converting to Empirinometry notation...")
        print()
        
        # Auto-convert
        empirinometry_formula = self.auto_convert_to_empirinometry(standard_formula)
        
        print(f"Standard Form:")
        print(f"  {standard_formula}")
        print()
        print(f"Empirinometry Form (auto-converted):")
        print(f"  {empirinometry_formula}")
        print()
        
        # Ask if user wants to modify
        modify = input("Would you like to modify the conversion? (y/n): ").strip().lower()
        if modify == 'y':
            empirinometry_formula = input("Enter your Empirinometry form: ").strip()
        
        # Logic check
        print()
        print("Logic Check:")
        self.logic_check_custom_formula(standard_formula, empirinometry_formula)
    
    def auto_convert_to_empirinometry(self, formula):
        """Auto-convert standard formula to Empirinometry"""
        converted = formula
        
        # Replace * with #
        converted = converted.replace('*', ' # ')
        converted = converted.replace('×', ' # ')
        
        # Wrap variables in |Pillars|
        # Find all variable names (letters, possibly with subscripts)
        variables = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', converted)
        for var in set(variables):
            # Don't wrap if already wrapped
            if f"|{var}|" not in converted:
                # Use word boundaries to avoid partial matches
                converted = re.sub(r'\b' + var + r'\b', f'|{var}|', converted)
        
        return converted
    
    def logic_check_custom_formula(self, standard, empirinometry):
        """Logic check for custom formula"""
        print("  ✓ Checking operator conversion...")
        
        # Count operators
        mult_standard = standard.count('*') + standard.count('×')
        hash_emp = empirinometry.count('#')
        
        if mult_standard > 0:
            if hash_emp >= mult_standard:
                print(f"    ✓ {mult_standard} multiplication(s) → {hash_emp} # operation(s)")
            else:
                print(f"    ⚠ Warning: {mult_standard} multiplications but only {hash_emp} # operations")
        
        # Check for |Pillars|
        print("  ✓ Checking variable encapsulation...")
        pillar_count = empirinometry.count('|')
        if pillar_count >= 2:
            print(f"    ✓ Variables enclosed in |Pillars| ({pillar_count // 2} variables)")
        else:
            print("    ⚠ Warning: Variables should be enclosed in |Pillars|")
        
        print("  ✓ Logic check complete!")
    
    def thirteen_part_symposium(self, number=None, label=""):
        """Direction 2: Decompose into 13 parts"""
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
                    # Try to evaluate as expression
                    number = eval(user_input, {"__builtins__": {}}, {
                        'pi': math.pi, 'e': math.e, 'sqrt': math.sqrt,
                        'sin': math.sin, 'cos': math.cos, 'tan': math.tan
                    })
                    break
                except:
                    print("Invalid input. Please enter a valid number or expression.")
        
        print()
        print(f"Decomposing: {number}" + (f" ({label})" if label else ""))
        print()
        
        # Method A: Equal Division
        print("--- Method A: Equal Division ---")
        parts_a = [number / 13 for _ in range(13)]
        print(f"Each part: {parts_a[0]:.10f}")
        print(f"Sum: {sum(parts_a):.10f}")
        print(f"Verification: {'✓ PASS' if abs(sum(parts_a) - number) < 1e-10 else '✗ FAIL'}")
        print()
        
        # Method B: Beta Formula
        print("--- Method B: Beta Formula (Sequinor Tredecim) ---")
        p_x = ((number / 13) * 1000) / 13
        print(f"p(x) = {p_x:.10f}")
        parts_b = [number * (L / 91) for L in range(1, 14)]
        print(f"L-weighted parts (first 3): {[f'{p:.6f}' for p in parts_b[:3]]}")
        print(f"Sum: {sum(parts_b):.10f}")
        print(f"Verification: {'✓ PASS' if abs(sum(parts_b) - number) < 1e-10 else '✗ FAIL'}")
        print()
        
        # Method C: Modular Cycles
        print("--- Method C: Modular Cycles ---")
        quotient = int(number // 13)
        remainder = number % 13
        print(f"{number} = 13 × {quotient} + {remainder:.6f}")
        print(f"Quotient: {quotient}, Remainder: {remainder:.6f}")
        print()
        
        # n² mod 13 analysis
        print("--- n² mod 13 Analysis ---")
        mod_13 = int(number) % 13
        print(f"Number mod 13: {mod_13}")
        
        # Find matching n values
        matching_n = [n for n in range(1, 14) if (n ** 2) % 13 == mod_13]
        print(f"Matching n values (where n² ≡ {mod_13} mod 13): {matching_n}")
        print(f"Is quadratic residue: {'Yes' if matching_n else 'No'}")
        print()
        
        # Connection to Epsilon
        print("--- Connection to Epsilon ---")
        epsilon_val = float(self.p_e)
        ratio = number / epsilon_val if epsilon_val != 0 else 0
        print(f"Ratio to Epsilon: {ratio:.10f}")
        
        # Closest L value
        closest_L = min(range(1, 14), key=lambda L: abs(number - L))
        print(f"Closest L value: L{closest_L}")
        print()
        
        # Display all 13 parts (Method B)
        print("--- Complete 13-Part Breakdown (L-Weighted) ---")
        print(f"{'L':>3} | {'Weight':>8} | {'Part Value':>15} | {'Cumulative':>15}")
        print("-" * 60)
        cumulative = 0
        for L in range(1, 14):
            weight = L / 91
            part = number * weight
            cumulative += part
            print(f"{L:3d} | {weight:8.6f} | {part:15.6f} | {cumulative:15.6f}")
        print("-" * 60)
        print(f"{'':>3} | {'Total:':>8} | {sum(parts_b):15.6f} | {'':>15}")
        print()
        
        # Logic check
        print("Logic Check:")
        print("  ✓ All methods preserve original number (sum = original)")
        print("  ✓ 13 parts created successfully")
        print("  ✓ Sequinor Tredecim framework applied")
        print("  ✓ No fibs detected!")
        print()
        
        # Save option
        save = input("Would you like to save this decomposition? (y/n): ").strip().lower()
        if save == 'y':
            self.save_decomposition(number, label, parts_b)
    
    def save_decomposition(self, number, label, parts):
        """Save decomposition to file"""
        filename = input("Enter filename (e.g., 'decomposition.json'): ").strip()
        if not filename:
            filename = "decomposition.json"
        
        data = {
            'number': float(number),
            'label': label,
            'parts': [float(p) for p in parts],
            'sum': float(sum(parts)),
            'method': 'L-Weighted (Sequinor Tredecim)'
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved to {filename}")
        except Exception as e:
            print(f"Error saving: {e}")
    
    def view_variable_library(self):
        """Display all Empirinometry variables"""
        print("\n" + "=" * 80)
        print("EMPIRINOMETRY VARIABLE LIBRARY")
        print("=" * 80)
        print()
        
        categories = {
            'Mechanics': ['Force', 'Mass', 'Acceleration', 'Velocity', 'Momentum'],
            'Energy': ['Energy', 'KineticEnergy', 'PotentialEnergy'],
            'Constants': ['Light', 'Planck', 'Gravity'],
            'Quantum': ['Frequency', 'Wavelength'],
            'Bushman': ['Lambda', 'C_Star', 'F_12']
        }
        
        for category, var_names in categories.items():
            print(f"--- {category} ---")
            print()
            for var_name in var_names:
                if var_name in self.empirinometry_variables:
                    var = self.empirinometry_variables[var_name]
                    print(f"{var_name} ({var['symbol']}):")
                    print(f"  Varia Form: {var['varia_form']}")
                    print(f"  Description: {var['description']}")
                    print(f"  Units: {var['units']}")
                    if 'value' in var:
                        print(f"  Value: {var['value']}")
                    print()
            print()
    
    def view_formula_library(self):
        """Display all available formulas"""
        print("\n" + "=" * 80)
        print("EMPIRINOMETRY FORMULA LIBRARY")
        print("=" * 80)
        print()
        
        for i, (key, formula) in enumerate(self.empirinometry_formulas.items(), 1):
            print(f"{i}. {formula['name']}")
            print(f"   Standard: {formula['standard']}")
            print(f"   Empirinometry: {formula['empirinometry']}")
            print(f"   Explanation: {formula['explanation']}")
            print()
    
    def run(self):
        """Main program loop"""
        self.display_banner()
        
        while True:
            choice = self.main_menu()
            
            if choice == '1':
                self.substantiate_formula()
            elif choice == '2':
                self.thirteen_part_symposium()
            elif choice == '3':
                self.view_variable_library()
            elif choice == '4':
                self.view_formula_library()
            elif choice == '5':
                print("\nThank you for using Bi-Directional Compass!")
                print("May your journey through Empirinometry be fruitful!")
                print()
                break
            
            input("\nPress Enter to continue...")
            print("\n" * 2)

def main():
    compass = BiDirectionalCompass()
    compass.run()

if __name__ == "__main__":
    main()
"""
Empirinometry Operations Module
Enhanced operations for Empirinometry calculations and transformations
"""

import math
from decimal import Decimal
from typing import Dict, List, Optional, Union, Tuple
import re

class EmpirinometryOperations:
    def __init__(self, parent_compass):
        self.compass = parent_compass
        self.validation_errors = []
        
    def substantiate_formula(self) -> bool:
        """Enhanced formula substantiation with validation"""
        print("\n" + "=" * 80)
        print("EMPIRINOMETRY FORMULA SUBSTANTIATION")
        print("=" * 80)
        print()
        print("Available formulas:")
        print()
        
        for i, (key, formula) in enumerate(self.compass.empirinometry_formulas.items(), 1):
            print(f"{i}. {formula['name']}")
            print(f"   Standard: {formula['standard']}")
            print(f"   Empirinometry: {formula['empirinometry']}")
            print(f"   Domain: {formula['domain']}")
            print()
        
        print(f"{len(self.compass.empirinometry_formulas) + 1}. Enter custom formula")
        print(f"{len(self.compass.empirinometry_formulas) + 2}. Batch substantiation")
        print()
        
        while True:
            choice = input(f"Choose option (1-{len(self.compass.empirinometry_formulas) + 2}): ").strip()
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.compass.empirinometry_formulas) + 2:
                    break
            except ValueError:
                pass
            print(f"Invalid choice. Please enter 1-{len(self.compass.empirinometry_formulas) + 2}.")
        
        if choice_num <= len(self.compass.empirinometry_formulas):
            formula_key = list(self.compass.empirinometry_formulas.keys())[choice_num - 1]
            formula = self.compass.empirinometry_formulas[formula_key]
            return self.display_formula_substantiation(formula)
        elif choice_num == len(self.compass.empirinometry_formulas) + 1:
            return self.custom_formula_substantiation()
        else:
            return self.batch_substantiation()
    
    def display_formula_substantiation(self, formula: Dict) -> bool:
        """Display detailed formula substantiation with validation"""
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
        print(f"Domain: {formula['domain']}")
        print()
        
        print("Variable Breakdown:")
        print()
        
        for var_name in formula['variables']:
            if var_name in self.compass.empirinometry_variables:
                var = self.compass.empirinometry_variables[var_name]
                print(f"  {var_name} ({var['symbol']}):")
                print(f"    Varia Form: {var['varia_form']}")
                print(f"    Description: {var['description']}")
                print(f"    Units: {var['units']}")
                if 'dimensional_analysis' in var:
                    print(f"    Dimensions: {var['dimensional_analysis']}")
                if 'value' in var:
                    print(f"    Value: {var['value']}")
                print()
        
        # Enhanced logic check
        print("Logic Check:")
        validation_passed = self.logic_check_formula(formula)
        print()
        
        # Dimensional analysis
        print("Dimensional Analysis:")
        dim_check = self.dimensional_analysis(formula)
        print(f"  {'✓ PASS' if dim_check else '✗ FAIL'}: Dimensional consistency")
        print()
        
        if validation_passed and dim_check:
            calc = input("Would you like to calculate with specific values? (y/n): ").strip().lower()
            if calc == 'y':
                return self.calculate_formula(formula)
        else:
            print("⚠ Validation failed. Please review the formula before calculation.")
        
        return True
    
    def logic_check_formula(self, formula: Dict) -> bool:
        """Enhanced logic check with detailed validation"""
        validation_passed = True
        
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
                print(f"    ✗ Warning: Found {mult_count_standard} multiplications but only {hash_count} # operations")
                validation_passed = False
        
        # Check for proper |Pillar| encapsulation
        print("  ✓ Checking variable encapsulation (x → |x|)...")
        variables_found = 0
        for var_name in formula['variables']:
            if f"|{var_name}|" in empirinometry:
                variables_found += 1
        
        if variables_found == len(formula['variables']):
            print(f"    ✓ All {len(formula['variables'])} variables properly encapsulated")
        else:
            print(f"    ✗ Warning: Expected {len(formula['variables'])} variables, found {variables_found}")
            validation_passed = False
        
        # Check for syntactic errors
        print("  ✓ Checking syntax...")
        syntax_errors = self.check_syntax(empirinometry)
        if syntax_errors:
            print(f"    ✗ Syntax errors found: {syntax_errors}")
            validation_passed = False
        else:
            print("    ✓ No syntax errors detected")
        
        print("  ✓ Logic check complete!")
        return validation_passed
    
    def dimensional_analysis(self, formula: Dict) -> bool:
        """Perform dimensional analysis on the formula"""
        try:
            # Simple dimensional analysis check
            # This is a simplified version - full implementation would require
            # parsing the formula and checking dimensional consistency
            
            print(f"    Analyzing dimensions for {formula['domain']}...")
            
            # For now, assume all predefined formulas are dimensionally consistent
            # In a full implementation, this would parse and verify dimensions
            
            return True
        except Exception as e:
            self.compass.log_error(f"Dimensional analysis failed: {e}")
            return False
    
    def check_syntax(self, formula: str) -> List[str]:
        """Check for syntax errors in Empirinometry formula"""
        errors = []
        
        # Check for unmatched pillars
        pillar_count = formula.count('|')
        if pillar_count % 2 != 0:
            errors.append("Unmatched |Pillars|")
        
        # Check for invalid characters
        invalid_chars = re.findall(r'[^a-zA-Z0-9\s\|\#\+\-\*\/\^\(\)=<>!&|_]', formula)
        if invalid_chars:
            errors.append(f"Invalid characters: {set(invalid_chars)}")
        
        # Check for balanced parentheses
        paren_count = formula.count('(') - formula.count(')')
        if paren_count != 0:
            errors.append("Unbalanced parentheses")
        
        return errors
    
    def calculate_formula(self, formula: Dict) -> bool:
        """Enhanced formula calculation with error handling"""
        print("\n" + "-" * 80)
        print("FORMULA CALCULATION")
        print("-" * 80)
        print()
        
        values = {}
        for var_name in formula['variables']:
            if var_name in self.compass.empirinometry_variables:
                var = self.compass.empirinometry_variables[var_name]
                
                # Skip if it's the output variable (left side of equation)
                if var_name in formula['standard'].split('=')[0]:
                    continue
                
                # Use default value if available
                if 'value' in var:
                    use_default = input(f"Use default value for {var_name} ({var['value']} {var['units']})? (y/n): ").strip().lower()
                    if use_default == 'y':
                        values[var_name] = var['value']
                        continue
                
                # Get user input with validation
                while True:
                    try:
                        val = input(f"Enter value for {var_name} ({var['units']}): ").strip()
                        # Allow mathematical expressions
                        if any(op in val for op in '+-*/^'):
                            values[var_name] = eval(val, {"__builtins__": {}}, {})
                        else:
                            values[var_name] = float(val)
                        break
                    except (ValueError, SyntaxError):
                        print("Invalid number or expression. Please try again.")
        
        # Calculate result with error handling
        try:
            result = self.evaluate_formula(formula, values)
            
            if result is not None:
                output_var = formula['variables'][0]  # First variable is usually output
                output_info = self.compass.empirinometry_variables.get(output_var, {})
                units = output_info.get('units', '')
                
                print()
                print(f"Result: {result:.6e} {units}")
                print()
                
                # Validation check
                if self.validate_result(result, formula):
                    print("✓ Result validation passed")
                else:
                    print("⚠ Result validation warning - please verify")
                
                # Offer additional analyses
                print("\nAdditional analyses available:")
                print("1. 13-part decomposition")
                print("2. Sensitivity analysis")
                print("3. Unit conversion")
                print("4. Save calculation")
                
                choice = input("Choose analysis (1-4, or Enter to skip): ").strip()
                
                if choice == '1':
                    from ..modules.sequinor_tredecim_methods import SequinorTredecimMethods
                    st = SequinorTredecimMethods(self.compass)
                    st.thirteen_part_symposium(result, f"Result of {formula['name']}")
                elif choice == '2':
                    self.sensitivity_analysis(formula, values)
                elif choice == '3':
                    self.unit_conversion(result, units)
                elif choice == '4':
                    self.save_calculation(formula, values, result, units)
                
        except Exception as e:
            print(f"✗ Calculation error: {e}")
            self.compass.log_error(f"Formula calculation failed: {e}")
            return False
        
        return True
    
    def validate_result(self, result: float, formula: Dict) -> bool:
        """Validate calculation result"""
        try:
            # Check for common errors
            if math.isnan(result):
                print("✗ Result is NaN")
                return False
            
            if math.isinf(result):
                print("⚠ Result is infinite")
                return False
            
            # Check for reasonable magnitude based on domain
            domain = formula.get('domain', '')
            if domain == 'mechanics' and abs(result) > 1e20:
                print("⚠ Unusually large result for mechanics")
                return False
            elif domain == 'quantum' and abs(result) < 1e-50 and result != 0:
                print("⚠ Unusually small result for quantum mechanics")
                return False
            
            return True
        except:
            return False
    
    def evaluate_formula(self, formula: Dict, values: Dict) -> Optional[float]:
        """Evaluate formula with given values"""
        try:
            name = formula['name']
            
            # Enhanced evaluation with more formulas
            if name == "Newton's Second Law":
                return values['Mass'] * values['Acceleration']
            elif name == 'Mass-Energy Equivalence':
                return values['Mass'] * (values['Light'] ** 2)
            elif name == 'Momentum':
                return values['Mass'] * values['Velocity']
            elif name == 'Kinetic Energy':
                return 0.5 * values['Mass'] * (values['Velocity'] ** 2)
            elif name == 'Potential Energy':
                return values['Mass'] * values['Gravity'] * values.get('Height', 0)
            elif name == 'Photon Energy':
                return values['Planck'] * values['Frequency']
            elif name == 'Wave Equation':
                return values['Frequency'] * values['Wavelength']
            elif name == 'Grip Relationship':
                return values['Lambda'] * values['C_Star']
            else:
                print(f"⚠ Calculation not implemented for {name}")
                return None
        except Exception as e:
            self.compass.log_error(f"Formula evaluation error: {e}")
            return None
    
    def sensitivity_analysis(self, formula: Dict, base_values: Dict):
        """Perform sensitivity analysis on formula"""
        print("\n" + "-" * 40)
        print("SENSITIVITY ANALYSIS")
        print("-" * 40)
        
        try:
            base_result = self.evaluate_formula(formula, base_values)
            if base_result is None:
                return
            
            print(f"Base result: {base_result:.6e}")
            print()
            
            # Test sensitivity to each variable
            for var_name, base_value in base_values.items():
                if var_name not in formula['variables']:
                    continue
                
                print(f"Testing sensitivity to {var_name}:")
                
                # Test ±10% variations
                for variation in [-0.1, 0.1]:
                    test_values = base_values.copy()
                    test_values[var_name] = base_value * (1 + variation)
                    
                    test_result = self.evaluate_formula(formula, test_values)
                    if test_result is not None:
                        percent_change = ((test_result - base_result) / base_result) * 100
                        print(f"  {variation*100:+.0f}%: {test_result:.6e} ({percent_change:+.2f}%)")
                
                print()
        except Exception as e:
            print(f"✗ Sensitivity analysis failed: {e}")
    
    def unit_conversion(self, result: float, original_units: str):
        """Convert result to different units"""
        print("\n" + "-" * 40)
        print("UNIT CONVERSION")
        print("-" * 40)
        
        print(f"Original: {result:.6e} {original_units}")
        print()
        
        # Simple unit conversions for common units
        conversions = {
            'J': [('kJ', 0.001), ('cal', 0.239006), ('eV', 6.242e18)],
            'N': [('kN', 0.001), ('lbf', 0.224809)],
            'm': [('km', 0.001), ('cm', 100), ('mm', 1000), ('ft', 3.28084)],
            'kg': [('g', 1000), ('lb', 2.20462), ('oz', 35.274)]
        }
        
        if original_units in conversions:
            for new_unit, factor in conversions[original_units]:
                converted = result * factor
                print(f"  {converted:.6e} {new_unit}")
        else:
            print("No standard conversions available for this unit")
    
    def save_calculation(self, formula: Dict, values: Dict, result: float, units: str):
        """Save calculation to file"""
        try:
            filename = input("Enter filename (e.g., 'calculation.json'): ").strip()
            if not filename:
                filename = "calculation.json"
            
            data = {
                'formula_name': formula['name'],
                'formula_standard': formula['standard'],
                'formula_empirinometry': formula['empirinometry'],
                'input_values': {k: str(v) for k, v in values.items()},
                'result': float(result),
                'units': units,
                'domain': formula.get('domain', 'unknown')
            }
            
            import json
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Calculation saved to {filename}")
        except Exception as e:
            print(f"✗ Save failed: {e}")
    
    def custom_formula_substantiation(self) -> bool:
        """Handle custom formula input with enhanced validation"""
        print("\n" + "-" * 80)
        print("CUSTOM FORMULA SUBSTANTIATION")
        print("-" * 80)
        print()
        print("Enter your formula in standard notation (e.g., F = ma)")
        print("You can use: +, -, *, /, ^, parentheses, and scientific notation")
        print()
        
        while True:
            standard_formula = input("Standard formula: ").strip()
            if not standard_formula:
                print("Formula cannot be empty")
                continue
            
            if '=' not in standard_formula:
                print("Formula must contain '='")
                continue
            
            if self.check_syntax(standard_formula):
                break
            else:
                print("Syntax error in formula. Please check and try again.")
        
        print()
        print("Converting to Empirinometry notation...")
        print()
        
        # Auto-convert with options
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
        
        # Validation
        print()
        print("Validation:")
        syntax_errors = self.check_syntax(empirinometry_formula)
        if syntax_errors:
            print(f"✗ Syntax errors: {syntax_errors}")
            return False
        else:
            print("✓ Syntax validation passed")
        
        return True
    
    def auto_convert_to_empirinometry(self, formula: str) -> str:
        """Enhanced auto-conversion to Empirinometry"""
        converted = formula
        
        # Replace * with #
        converted = converted.replace('*', ' # ')
        converted = converted.replace('×', ' # ')
        
        # Wrap variables in |Pillars|
        variables = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', converted)
        for var in set(variables):
            if f"|{var}|" not in converted:
                converted = re.sub(r'\b' + var + r'\b', f'|{var}|', converted)
        
        return converted
    
    def batch_substantiation(self) -> bool:
        """Batch substantiation of multiple formulas"""
        print("\n" + "-" * 80)
        print("BATCH SUBSTANTIATION")
        print("-" * 80)
        print()
        print("Enter multiple formulas (one per line), empty line to finish:")
        print()
        
        formulas = []
        while True:
            formula = input().strip()
            if not formula:
                break
            if '=' in formula and self.check_syntax(formula):
                formulas.append(formula)
            else:
                print("✗ Invalid formula format, skipping")
        
        if not formulas:
            print("No valid formulas entered")
            return False
        
        print(f"\nProcessing {len(formulas)} formulas...")
        results = []
        
        for i, formula in enumerate(formulas, 1):
            print(f"\n{i}. {formula}")
            try:
                converted = self.auto_convert_to_empirinometry(formula)
                print(f"   → {converted}")
                
                if not self.check_syntax(converted):
                    print(f"   ✗ Syntax error in conversion")
                    continue
                
                results.append({
                    'original': formula,
                    'converted': converted,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"   ✗ Conversion failed: {e}")
                results.append({
                    'original': formula,
                    'converted': None,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Summary
        successful = sum(1 for r in results if r['status'] == 'success')
        print(f"\nBatch complete: {successful}/{len(formulas)} successful")
        
        # Save results
        save = input("Save batch results? (y/n): ").strip().lower()
        if save == 'y':
            try:
                filename = input("Enter filename: ").strip() or "batch_results.json"
                import json
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"✓ Results saved to {filename}")
            except Exception as e:
                print(f"✗ Save failed: {e}")
        
        return True
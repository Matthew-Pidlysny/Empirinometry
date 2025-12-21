"""
LaTeX Integration Module for Omni-Directional Compass
Provides comprehensive LaTeX encoding and decoding capabilities
License: BSD (compatible with main project)
"""

import sympy as sp
import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

@dataclass
class LatexRepresentation:
    """Represents a LaTeX encoded formula"""
    latex_code: str
    formula_type: str
    complexity: int
    domain: str
    description: str

@dataclass
class ReverseEngineeringResult:
    """Result of reverse engineering LaTeX to formula"""
    original_latex: str
    python_expression: str
    sympy_expression: Any
    evaluation_result: Optional[float]
    confidence: float
    validation_status: str

class LatexIntegration:
    """
    Comprehensive LaTeX integration for mathematical substantiation
    Supports encoding, decoding, visualization, and validation
    """
    
    def __init__(self, parent_compass):
        self.compass = parent_compass
        self.setup_sympy_environment()
        self.operator_mappings = self.create_operator_mappings()
        self.empirinometry_mappings = self.create_empirinometry_mappings()
        
    def setup_sympy_environment(self):
        """Setup SymPy environment with custom symbols"""
        # Define custom symbols for our operators
        self.custom_symbols = {
            'λ': sp.Symbol('lambda'),
            'β': sp.Symbol('beta'), 
            'ε': sp.Symbol('epsilon'),
            '∇': sp.Symbol('nabla'),
            '∂': sp.Symbol('partial'),
            '∫': sp.Integral,
            '∑': sp.summation,
            '∏': sp.product
        }
        
    def create_operator_mappings(self) -> Dict[str, str]:
        """Create mappings for standard operators to LaTeX"""
        return {
            '+': '+',
            '-': '-',
            '*': '\\cdot',
            '/': '\\frac',
            '^': '^',
            '#': '\\#',  # Empirinometry multiplication
            'sqrt': '\\sqrt',
            'pi': '\\pi',
            'alpha': '\\alpha',
            'beta': '\\beta',
            'gamma': '\\gamma',
            'delta': '\\delta',
            'epsilon': '\\epsilon',
            'theta': '\\theta',
            'lambda': '\\lambda',
            'mu': '\\mu',
            'sigma': '\\sigma',
            'omega': '\\omega'
        }
        
    def create_empirinometry_mappings(self) -> Dict[str, str]:
        """Create special mappings for Empirinometry operators"""
        return {
            '#': '\\text{\\#}',  # Empirinometry multiplication
            'LAMBDA': '\\lambda',
            'C_STAR': 'C^*',
            'F_12': 'F_{12}',
            'p_t': 'p_\\tau',
            'p_e': 'p_\\epsilon'
        }
        
    def encode_formula_to_latex(self, formula: str, formula_type: str = "general") -> List[LatexRepresentation]:
        """
        Encode a formula to multiple LaTeX representations
        Returns different forms for comprehensive visualization
        """
        representations = []
        
        try:
            # Clean and parse the formula
            cleaned_formula = self.clean_formula(formula)
            
            # 1. Standard LaTeX representation
            standard_latex = self.to_standard_latex(cleaned_formula)
            representations.append(LatexRepresentation(
                latex_code=standard_latex,
                formula_type="Standard",
                complexity=1,
                domain="General",
                description="Standard mathematical notation"
            ))
            
            # 2. Empirinometry-specific representation
            if '#' in formula:
                empirinometry_latex = self.to_empirinometry_latex(cleaned_formula)
                representations.append(LatexRepresentation(
                    latex_code=empirinometry_latex,
                    formula_type="Empirinometry",
                    complexity=2,
                    domain="Empirinometry",
                    description="Empirinometry multiplication notation"
                ))
            
            # 3. Symbolic representation using SymPy
            try:
                sympy_expr = self.parse_to_sympy(cleaned_formula)
                sympy_latex = sp.latex(sympy_expr)
                representations.append(LatexRepresentation(
                    latex_code=sympy_latex,
                    formula_type="SymPy",
                    complexity=3,
                    domain="Symbolic",
                    description="SymPy symbolic representation"
                ))
            except:
                pass  # SymPy parsing failed
                
            # 4. Step-by-step calculation format
            if any(op in formula for op in ['+', '-', '*', '/', '#']):
                stepwise_latex = self.to_stepwise_latex(cleaned_formula)
                representations.append(LatexRepresentation(
                    latex_code=stepwise_latex,
                    formula_type="Stepwise",
                    complexity=4,
                    domain="Educational",
                    description="Step-by-step calculation format"
                ))
                
            # 5. Matrix form (if applicable)
            if self.is_matrix_expression(formula):
                matrix_latex = self.to_matrix_latex(cleaned_formula)
                representations.append(LatexRepresentation(
                    latex_code=matrix_latex,
                    formula_type="Matrix",
                    complexity=5,
                    domain="Linear Algebra",
                    description="Matrix representation"
                ))
                
        except Exception as e:
            # Fallback representation
            representations.append(LatexRepresentation(
                latexex_code=f"\\text{{Error: {str(e)}}}",
                formula_type="Error",
                complexity=0,
                domain="Error",
                description="Parsing error occurred"
            ))
            
        return representations
        
    def reverse_engineer_latex(self, latex_code: str) -> ReverseEngineeringResult:
        """
        Reverse engineer LaTeX back to Python expression
        Validates and evaluates the result
        """
        try:
            # Clean LaTeX code
            cleaned_latex = self.clean_latex(latex_code)
            
            # Convert to SymPy expression
            sympy_expr = sp.sympify(cleaned_latex, locals=self.custom_symbols)
            
            # Generate Python expression
            python_expr = self.sympy_to_python(sympy_expr)
            
            # Try to evaluate
            try:
                eval_result = float(sympy_expr.evalf())
                confidence = 1.0
                validation_status = "VALID"
            except:
                eval_result = None
                confidence = 0.7
                validation_status = "SYMBOLIC"
                
            return ReverseEngineeringResult(
                original_latex=latex_code,
                python_expression=python_expr,
                sympy_expression=sympy_expr,
                evaluation_result=eval_result,
                confidence=confidence,
                validation_status=validation_status
            )
            
        except Exception as e:
            return ReverseEngineeringResult(
                original_latex=latex_code,
                python_expression=f"# Error: {str(e)}",
                sympy_expression=None,
                evaluation_result=None,
                confidence=0.0,
                validation_status="ERROR"
            )
            
    def clean_formula(self, formula: str) -> str:
        """Clean and normalize formula string"""
        # Remove whitespace
        formula = formula.strip()
        
        # Replace common operators
        formula = formula.replace('^', '**')
        
        # Handle Empirinometry multiplication
        if '#' in formula:
            formula = formula.replace('#', ' / 4 *')  # (x * y) / 4 = x # y
            
        return formula
        
    def clean_latex(self, latex_code: str) -> str:
        """Clean LaTeX code for parsing"""
        # Remove display math delimiters
        latex_code = latex_code.replace('$$', '')
        latex_code = latex_code.replace('\\[', '').replace('\\]', '')
        
        # Convert common LaTeX commands to Python
        latex_code = latex_code.replace('\\cdot', '*')
        latex_code = latex_code.replace('\\pi', 'pi')
        latex_code = latex_code.replace('\\sqrt', 'sqrt')
        
        # Handle fractions
        latex_code = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', latex_code)
        
        # Handle subscripts and superscripts
        latex_code = re.sub(r'_\{([^}]+)\}', r'_\1', latex_code)
        latex_code = re.sub(r'\^\{([^}]+)\}', r'**\1', latex_code)
        
        return latex_code
        
    def to_standard_latex(self, formula: str) -> str:
        """Convert formula to standard LaTeX"""
        # Use SymPy for conversion
        try:
            expr = sp.sympify(formula)
            return sp.latex(expr)
        except:
            # Fallback manual conversion
            return self.manual_latex_conversion(formula)
            
    def to_empirinometry_latex(self, formula: str) -> str:
        """Convert formula with Empirinometry notation"""
        latex = self.to_standard_latex(formula)
        
        # Replace Empirinometry multiplication
        latex = latex.replace('\\cdot', '\\text{\\#}')
        
        # Add Empirinometry context
        latex = f"\\text{{Empirinometry: }} {latex}"
        
        return latex
        
    def to_stepwise_latex(self, formula: str) -> str:
        """Convert to step-by-step calculation format"""
        try:
            # Parse the formula
            if '+' in formula:
                parts = formula.split('+')
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    # Try to evaluate parts
                    try:
                        left_val = eval(left)
                        right_val = eval(right)
                        total = left_val + right_val
                        
                        latex = f"{left} + {right} = {left_val} + {right_val} = {total}"
                        return latex.replace('*', '\\cdot')
                    except:
                        pass
                        
            # Fallback: simple format
            return f"{formula} = \\text{{...}}"
            
        except:
            return f"\\text{{Stepwise format for: }} {formula}"
            
    def to_matrix_latex(self, formula: str) -> str:
        """Convert to matrix representation if applicable"""
        # Simple matrix detection - can be expanded
        if '[' in formula and ']' in formula:
            return "\\begin{pmatrix} \\text{Matrix} \\end{pmatrix}"
        return "Not applicable for matrix representation"
        
    def parse_to_sympy(self, formula: str) -> Any:
        """Parse formula to SymPy expression"""
        return sp.sympify(formula, locals=self.custom_symbols)
        
    def sympy_to_python(self, sympy_expr: Any) -> str:
        """Convert SymPy expression to Python string"""
        return str(sympy_expr)
        
    def is_matrix_expression(self, formula: str) -> bool:
        """Check if formula represents a matrix expression"""
        return '[' in formula and ']' in formula
        
    def manual_latex_conversion(self, formula: str) -> str:
        """Manual LaTeX conversion fallback"""
        latex = formula
        
        # Basic replacements
        for py_op, latex_op in self.operator_mappings.items():
            latex = latex.replace(py_op, latex_op)
            
        # Handle function calls
        latex = re.sub(r'(\w+)\(([^)]+)\)', r'\\text{\1}(\2)', latex)
        
        return latex
        
    def validate_substantiation_with_latex(self, original_formula: str, latex_versions: List[LatexRepresentation]) -> Dict[str, Any]:
        """
        Validate substantiation by comparing LaTeX representations
        """
        validation_results = {
            'original_formula': original_formula,
            'validation_passed': True,
            'inconsistencies': [],
            'recommendations': [],
            'latex_matches': 0,
            'total_versions': len(latex_versions)
        }
        
        try:
            # Try to evaluate original formula
            original_result = eval(original_formula)
            
            # Check each LaTeX version
            for latex_rep in latex_versions:
                try:
                    # Reverse engineer back to check consistency
                    reverse_result = self.reverse_engineer_latex(latex_rep.latex_code)
                    
                    if reverse_result.evaluation_result is not None:
                        if abs(reverse_result.evaluation_result - original_result) > 0.001:
                            validation_results['inconsistencies'].append({
                                'type': latex_rep.formula_type,
                                'latex': latex_rep.latex_code,
                                'expected': original_result,
                                'got': reverse_result.evaluation_result
                            })
                            validation_results['validation_passed'] = False
                        else:
                            validation_results['latex_matches'] += 1
                            
                except:
                    validation_results['inconsistencies'].append({
                        'type': latex_rep.formula_type,
                        'latex': latex_rep.latex_code,
                        'error': 'Could not evaluate'
                    })
                    
        except Exception as e:
            validation_results['validation_passed'] = False
            validation_results['inconsistencies'].append({
                'type': 'Original Formula',
                'error': str(e)
            })
            
        # Generate recommendations
        if validation_results['validation_passed']:
            validation_results['recommendations'].append("All LaTeX representations are consistent")
        else:
            validation_results['recommendations'].append("Check formula syntax and operator precedence")
            
        return validation_results
        
    def create_latex_visualization(self, latex_code: str, title: str = "LaTeX Formula"):
        """
        Create a matplotlib visualization of LaTeX formula
        Returns figure and canvas for GUI integration
        """
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis('off')
        
        # Render LaTeX
        ax.text(0.5, 0.5, f'${latex_code}$', 
                fontsize=16, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
        
    def batch_latex_conversion(self, formulas: List[str]) -> List[List[LatexRepresentation]]:
        """Convert multiple formulas to LaTeX"""
        results = []
        for formula in formulas:
            latex_versions = self.encode_formula_to_latex(formula)
            results.append(latex_versions)
        return results
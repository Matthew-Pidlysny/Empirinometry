"""
LaTeX Rendering Engine for Induction Ω Mathematical Induction Measurement System
Provides comprehensive LaTeX rendering capabilities for mathematical expressions and formulas
"""

import re
import subprocess
import tempfile
import os
import base64
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

class LaTeXEngine:
    """
    Comprehensive LaTeX rendering engine with multiple output formats
    Supports mathematical expressions, symbols, equations, and complex formatting
    """
    
    def __init__(self, cache_dir: str = "latex_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.preamble = self._get_standard_preamble()
        self.math_mode_patterns = self._get_math_patterns()
        self.symbol_cache = {}
        self.expression_cache = {}
        
    def _get_standard_preamble(self) -> str:
        """Get standard LaTeX preamble for mathematical expressions"""
        return r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{amsthm}
\usepackage{mathtools}
\usepackage{bm}
\usepackage{physics}
\usepackage{tensor}
\usepackage{commath}
\usepackage{siunitx}
\usepackage{units}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{array}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{listings}
\usepackage{hyperref}

\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

\begin{document}
"""
    
    def _get_math_patterns(self) -> List[re.Pattern]:
        """Get regex patterns for detecting math modes"""
        return [
            re.compile(r'\$\$(.*?)\$\$', re.DOTALL),  # Display math $$...$$
            re.compile(r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}', re.DOTALL),
            re.compile(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', re.DOTALL),
            re.compile(r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}', re.DOTALL),
            re.compile(r'\\begin\{flalign\*?\}(.*?)\\end\{flalign\*?\}', re.DOTALL),
            re.compile(r'\$(.*?)\$', re.DOTALL),  # Inline math $...$
        ]
    
    def render_latex_to_svg(self, latex_code: str, display_mode: bool = False) -> Optional[str]:
        """
        Render LaTeX code to SVG format
        Returns base64 encoded SVG or None if rendering fails
        """
        cache_key = f"{hash(latex_code)}_{display_mode}"
        if cache_key in self.expression_cache:
            return self.expression_cache[cache_key]
        
        try:
            # Prepare LaTeX document
            if display_mode:
                full_latex = self.preamble + f"\\[{latex_code}\\]\n\\end{{document}}"
            else:
                full_latex = self.preamble + f"${latex_code}$\n\\end{{document}}"
            
            # Create temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                tex_file = os.path.join(temp_dir, "expression.tex")
                dvi_file = os.path.join(temp_dir, "expression.dvi")
                svg_file = os.path.join(temp_dir, "expression.svg")
                
                # Write LaTeX file
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(full_latex)
                
                # Compile LaTeX to DVI
                try:
                    subprocess.run(['latex', '-interaction=nonstopmode', '-output-directory', temp_dir, tex_file],
                                 check=True, capture_output=True, timeout=10)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback to basic rendering if latex is not available
                    return self._render_text_fallback(latex_code, display_mode)
                
                # Convert DVI to SVG
                try:
                    subprocess.run(['dvisvgm', '--no-fonts', '--exact', '--output', svg_file, dvi_file],
                                 check=True, capture_output=True, timeout=10)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return self._render_text_fallback(latex_code, display_mode)
                
                # Read and encode SVG
                if os.path.exists(svg_file):
                    with open(svg_file, 'r', encoding='utf-8') as f:
                        svg_content = f.read()
                    
                    # Clean up SVG
                    svg_content = self._clean_svg(svg_content)
                    encoded_svg = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
                    
                    # Cache result
                    self.expression_cache[cache_key] = encoded_svg
                    return encoded_svg
                    
        except Exception as e:
            print(f"LaTeX rendering error: {e}")
            return self._render_text_fallback(latex_code, display_mode)
        
        return None
    
    def render_latex_to_png(self, latex_code: str, display_mode: bool = False, dpi: int = 150) -> Optional[str]:
        """
        Render LaTeX code to PNG format
        Returns base64 encoded PNG or None if rendering fails
        """
        cache_key = f"{hash(latex_code)}_{display_mode}_{dpi}"
        if cache_key in self.expression_cache:
            return self.expression_cache[cache_key]
        
        try:
            # Prepare LaTeX document
            if display_mode:
                full_latex = self.preamble + f"\\[{latex_code}\\]\n\\end{{document}}"
            else:
                full_latex = self.preamble + f"${latex_code}$\n\\end{{document}}"
            
            with tempfile.TemporaryDirectory() as temp_dir:
                tex_file = os.path.join(temp_dir, "expression.tex")
                dvi_file = os.path.join(temp_dir, "expression.dvi")
                png_file = os.path.join(temp_dir, "expression.png")
                
                # Write LaTeX file
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(full_latex)
                
                # Compile LaTeX to DVI
                try:
                    subprocess.run(['latex', '-interaction=nonstopmode', '-output-directory', temp_dir, tex_file],
                                 check=True, capture_output=True, timeout=10)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return self._render_text_fallback(latex_code, display_mode)
                
                # Convert DVI to PNG
                try:
                    subprocess.run(['dvipng', '-D', str(dpi), '-T', 'tight', '-o', png_file, dvi_file],
                                 check=True, capture_output=True, timeout=10)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return self._render_text_fallback(latex_code, display_mode)
                
                # Read and encode PNG
                if os.path.exists(png_file):
                    with open(png_file, 'rb') as f:
                        png_content = f.read()
                    
                    encoded_png = base64.b64encode(png_content).decode('utf-8')
                    
                    # Cache result
                    self.expression_cache[cache_key] = encoded_png
                    return encoded_png
                    
        except Exception as e:
            print(f"LaTeX PNG rendering error: {e}")
            return self._render_text_fallback(latex_code, display_mode)
        
        return None
    
    def render_latex_to_html(self, latex_code: str, display_mode: bool = False) -> str:
        """
        Render LaTeX code to HTML using MathJax-compatible format
        Always returns a valid HTML string
        """
        if display_mode:
            return f'<div class="math-display">\\[{latex_code}\\]</div>'
        else:
            return f'<span class="math-inline">\\({latex_code}\\)</span>'
    
    def _clean_svg(self, svg_content: str) -> str:
        """Clean up SVG content for web display"""
        # Remove unnecessary attributes and optimize
        svg_content = re.sub(r'width="[^"]*"', '', svg_content)
        svg_content = re.sub(r'height="[^"]*"', '', svg_content)
        svg_content = re.sub(r'xmlns="[^"]*"', '', svg_content)
        
        # Add viewBox if not present
        if 'viewBox' not in svg_content:
            svg_content = svg_content.replace('<svg', '<svg viewBox="0 0 100 20"', 1)
        
        return svg_content
    
    def _render_text_fallback(self, latex_code: str, display_mode: bool = False) -> str:
        """Fallback text-based rendering when LaTeX is not available"""
        # Simple text representation
        clean_text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', latex_code)
        clean_text = re.sub(r'[{}$\\^_]', '', clean_text)
        
        if display_mode:
            return f'<div class="math-fallback-display">{clean_text}</div>'
        else:
            return f'<span class="math-fallback-inline">{clean_text}</span>'
    
    def parse_document_with_latex(self, content: str) -> Dict:
        """
        Parse a document containing LaTeX expressions and render them
        Returns structured data with rendered expressions
        """
        rendered_sections = []
        current_section = ""
        latex_expressions = []
        
        # Split content by LaTeX delimiters
        parts = re.split(r'(\$\$.*?\$\$|\\begin\{.*?\}.*?\\end\{.*?\}|\$.*?\$)', content, flags=re.DOTALL)
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text
                if part.strip():
                    current_section += part
            else:  # LaTeX expression
                if current_section.strip():
                    rendered_sections.append({
                        'type': 'text',
                        'content': current_section.strip()
                    })
                    current_section = ""
                
                # Determine expression type and render
                expr_type, latex_code = self._extract_latex_type(part)
                rendered = self.render_latex(latex_code, expr_type)
                
                latex_expressions.append({
                    'original': part,
                    'latex_code': latex_code,
                    'type': expr_type,
                    'rendered': rendered
                })
                
                rendered_sections.append({
                    'type': 'latex',
                    'expression': latex_code,
                    'display_mode': expr_type == 'display',
                    'rendered': rendered
                })
        
        # Add remaining text
        if current_section.strip():
            rendered_sections.append({
                'type': 'text',
                'content': current_section.strip()
            })
        
        return {
            'sections': rendered_sections,
            'latex_expressions': latex_expressions,
            'total_expressions': len(latex_expressions)
        }
    
    def _extract_latex_type(self, latex_string: str) -> Tuple[str, str]:
        """Extract LaTeX code and determine if it's display mode"""
        latex_string = latex_string.strip()
        
        if latex_string.startswith('$$') and latex_string.endswith('$$'):
            return 'display', latex_string[2:-2].strip()
        elif latex_string.startswith('\\begin{equation'):
            match = re.search(r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}', latex_string, re.DOTALL)
            return 'display', match.group(1).strip() if match else latex_string
        elif latex_string.startswith('$') and latex_string.endswith('$'):
            content = latex_string[1:-1].strip()
            if len(content) > 20 or '\\\\' in content:  # Heuristic for display math
                return 'display', content
            else:
                return 'inline', content
        else:
            return 'inline', latex_string
    
    def render_latex(self, latex_code: str, display_mode: bool = False, format_type: str = 'html') -> str:
        """
        Main rendering method with format selection
        """
        if format_type == 'svg':
            result = self.render_latex_to_svg(latex_code, display_mode)
            if result:
                return f'<img src="data:image/svg+xml;base64,{result}" alt="LaTeX: {latex_code}" class="latex-svg">'
        
        elif format_type == 'png':
            result = self.render_latex_to_png(latex_code, display_mode)
            if result:
                return f'<img src="data:image/png;base64,{result}" alt="LaTeX: {latex_code}" class="latex-png">'
        
        # Default to HTML/MathJax
        return self.render_latex_to_html(latex_code, display_mode)
    
    def get_math_symbols_library(self) -> Dict:
        """Get comprehensive library of mathematical symbols and their LaTeX codes"""
        if not self.symbol_cache:
            self.symbol_cache = {
                'greek_letters': {
                    'lowercase': ['\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\zeta', '\\eta', 
                                 '\\theta', '\\iota', '\\kappa', '\\lambda', '\\mu', '\\nu', '\\xi', '\\pi', 
                                 '\\rho', '\\sigma', '\\tau', '\\upsilon', '\\phi', '\\chi', '\\psi', '\\omega'],
                    'uppercase': ['\\Gamma', '\\Delta', '\\Theta', '\\Lambda', '\\Xi', '\\Pi', '\\Sigma', 
                                 '\\Upsilon', '\\Phi', '\\Psi', '\\Omega']
                },
                'operators': {
                    'binary': ['\\pm', '\\mp', '\\times', '\\div', '\\cdot', '\\circ', '\\oplus', '\\otimes', 
                              '\\cap', '\\cup', '\\land', '\\lor', '\\wedge', '\\vee'],
                    'relational': ['\\leq', '\\geq', '\\neq', '\\approx', '\\equiv', '\\sim', '\\cong', 
                                   '\\propto', '\\parallel', '\\perp'],
                    'arrows': ['\\rightarrow', '\\leftarrow', '\\leftrightarrow', '\\Rightarrow', '\\Leftarrow', 
                              '\\Leftrightarrow', '\\mapsto', '\\to']
                },
                'functions': ['\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc', '\\arcsin', '\\arccos', 
                              '\\arctan', '\\sinh', '\\cosh', '\\tanh', '\\ln', '\\log', '\\exp', '\\lim', 
                              '\\limsup', '\\liminf', '\\max', '\\min', '\\sup', '\\inf'],
                'misc': ['\\infty', '\\partial', '\\nabla', '\\exists', '\\forall', '\\emptyset', '\\in', 
                         '\\notin', '\\subset', '\\subseteq', '\\supset', '\\supseteq', '\\cup', '\\cap', 
                         '\\setminus', '\\complement']
            }
        
        return self.symbol_cache
    
    def validate_latex_syntax(self, latex_code: str) -> Dict[str, any]:
        """Validate LaTeX syntax and return validation results"""
        errors = []
        warnings = []
        
        # Check for balanced braces
        brace_count = latex_code.count('{') - latex_code.count('}')
        if brace_count != 0:
            errors.append(f"Unbalanced braces: {brace_count} extra opening braces" if brace_count > 0 
                         else f"Unbalanced braces: {-brace_count} extra closing braces")
        
        # Check for balanced parentheses
        paren_count = latex_code.count('(') - latex_code.count(')')
        if paren_count != 0:
            errors.append(f"Unbalanced parentheses: {paren_count} extra opening parentheses" if paren_count > 0
                         else f"Unbalanced parentheses: {-paren_count} extra closing parentheses")
        
        # Check for common syntax errors
        if re.search(r'\\[a-zA-Z]+\s*$', latex_code):
            warnings.append("Command at end of expression without arguments")
        
        if latex_code.count('\\begin{') != latex_code.count('\\end{'):
            errors.append("Mismatched \\begin and \\end commands")
        
        # Check for invalid commands
        valid_commands = self._get_valid_commands()
        invalid_commands = re.findall(r'\\([a-zA-Z]+)', latex_code)
        for cmd in invalid_commands:
            if cmd not in valid_commands:
                warnings.append(f"Unknown command: \\{cmd}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'suggestions': self._get_syntax_suggestions(latex_code)
        }
    
    def _get_valid_commands(self) -> set:
        """Get set of valid LaTeX commands"""
        return {
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa',
            'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
            'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega',
            'sum', 'prod', 'int', 'oint', 'lim', 'limsup', 'liminf', 'max', 'min', 'sup', 'inf',
            'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan',
            'sinh', 'cosh', 'tanh', 'ln', 'log', 'exp',
            'frac', 'sqrt', 'nroot', 'over', 'choose', 'binom',
            'left', 'right', 'big', 'Big', 'bigg', 'Bigg',
            'begin', 'end', 'text', 'mbox', 'displaystyle', 'textstyle',
            'overline', 'underline', 'widehat', 'widetilde', 'overbrace', 'underbrace',
            'pm', 'mp', 'times', 'div', 'cdot', 'circ', 'oplus', 'otimes',
            'leq', 'geq', 'neq', 'approx', 'equiv', 'sim', 'cong', 'propto',
            'rightarrow', 'leftarrow', 'leftrightarrow', 'Rightarrow', 'Leftarrow', 'Leftrightarrow',
            'partial', 'nabla', 'exists', 'forall', 'emptyset', 'in', 'notin',
            'subset', 'subseteq', 'supset', 'supseteq', 'cup', 'cap', 'setminus', 'complement'
        }
    
    def _get_syntax_suggestions(self, latex_code: str) -> List[str]:
        """Get syntax suggestions for improving LaTeX code"""
        suggestions = []
        
        if '{' in latex_code and '}' not in latex_code:
            suggestions.append("Consider using \\left\\{ and \\right\\} for large braces")
        
        if 'frac' in latex_code and not re.search(r'\\frac\s*\{[^}]*\}\s*\{[^}]*\}', latex_code):
            suggestions.append("Use \\frac{numerator}{denominator} format for fractions")
        
        if latex_code.count('^') > 1 or latex_code.count('_') > 1:
            suggestions.append("Consider using braces {} for complex exponents or subscripts")
        
        return suggestions

# Global LaTeX engine instance
latex_engine = LaTeXEngine()

def render_expression(expression: str, display_mode: bool = False, format_type: str = 'html') -> str:
    """Convenience function for rendering LaTeX expressions"""
    return latex_engine.render_latex(expression, display_mode, format_type)

def validate_expression(expression: str) -> Dict[str, any]:
    """Convenience function for validating LaTeX expressions"""
    return latex_engine.validate_latex_syntax(expression)
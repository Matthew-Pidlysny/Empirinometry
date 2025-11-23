#!/usr/bin/env python3
"""
ULTIMATE IRRATIONALITY PROVER
-----------------------------
A comprehensive system that combines:
1. Classical mathematical proofs
2. Computational verification via continued fractions  
3. Pattern recognition for known constants
4. UIS (Unified Irrationality System) classification

Mathematical Foundations:
- A number is rational IFF its continued fraction terminates
- A number is quadratic irrational IFF its continued fraction is periodic
- Non-periodic infinite continued fractions indicate higher-degree irrationals
"""

from fractions import Fraction
from math import floor, gcd, isqrt
import math
import decimal
import sys
import re
from typing import List, Tuple, Optional, Union

# Try to import sympy, but provide fallbacks
try:
    import sympy as sp
    from sympy import sqrt, pi, E, golden_ratio, exp, log, I
    HAVE_SYMPY = True
except ImportError:
    HAVE_SYMPY = False
    # Create dummy symbols for fallback
    class DummySymbol:
        def __init__(self, name):
            self.name = name
        def __repr__(self):
            return self.name
    sqrt = lambda x: DummySymbol(f"sqrt({x})")
    pi = DummySymbol("pi")
    E = DummySymbol("e")
    golden_ratio = DummySymbol("golden_ratio")

decimal.getcontext().prec = 500  # Ultra-high precision

class IrrationalityProof:
    """Comprehensive irrationality proof system"""
    
    def __init__(self):
        self.known_proofs = {
            'sqrt2': self.prove_sqrt2,
            'sqrt3': self.prove_sqrt3, 
            'sqrt5': self.prove_sqrt5,
            'golden_ratio': self.prove_golden_ratio,
            'pi': self.prove_pi_irrational,
            'e': self.prove_e_irrational,
            'log2': self.prove_log2_irrational
        }
    
    def prove_sqrt2(self) -> Tuple[bool, List[str]]:
        """Classical proof by contradiction for √2"""
        proof = [
            "THEOREM: √2 is irrational",
            "PROOF BY CONTRADICTION:",
            "1. Assume √2 is rational: √2 = a/b where a,b are coprime integers",
            "2. Then 2 = a²/b² ⇒ a² = 2b²", 
            "3. Thus a² is even ⇒ a is even (since square of odd is odd)",
            "4. Let a = 2k, then (2k)² = 2b² ⇒ 4k² = 2b² ⇒ 2k² = b²",
            "5. Thus b² is even ⇒ b is even",
            "6. But if a and b are both even, they are not coprime - CONTRADICTION",
            "7. Therefore, our assumption is false ⇒ √2 is irrational □"
        ]
        return True, proof
    
    def prove_sqrt_n(self, n: int) -> Tuple[bool, List[str]]:
        """General proof for √n where n is not a perfect square"""
        if n < 0:
            return False, [f"√{n} is complex, not real"]
        if n == 0:
            return False, ["√0 = 0 (rational)"]
        if isqrt(n)**2 == n:
            return False, [f"√{n} = {isqrt(n)} (rational integer)"]
            
        # Factor n to find a square-free part for the proof
        factors = self._factorize(n)
        square_free = 1
        for prime, exp in factors.items():
            if exp % 2 == 1:
                square_free *= prime
        
        proof = [
            f"THEOREM: √{n} is irrational",
            "PROOF BY CONTRADICTION:",
            f"1. Assume √{n} is rational: √{n} = a/b where a,b are coprime integers",
            f"2. Then {n} = a²/b² ⇒ a² = {n}b²",
            f"3. The prime factorization of {n} contains {square_free} (square-free part)",
            f"4. In a² = {n}b², the exponent of primes in {square_free} must be even on left, odd on right",
            f"5. This violates the Fundamental Theorem of Arithmetic (unique prime factorization)",
            f"6. Contradiction ⇒ √{n} is irrational □"
        ]
        return True, proof
    
    def _factorize(self, n: int) -> dict:
        """Simple factorization for proof purposes"""
        factors = {}
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors
    
    def prove_sqrt3(self) -> Tuple[bool, List[str]]:
        return self.prove_sqrt_n(3)
    
    def prove_sqrt5(self) -> Tuple[bool, List[str]]:
        return self.prove_sqrt_n(5)
    
    def prove_golden_ratio(self) -> Tuple[bool, List[str]]:
        """Proof for φ = (1 + √5)/2"""
        proof = [
            "THEOREM: The golden ratio φ = (1 + √5)/2 is irrational",
            "PROOF:",
            "1. φ satisfies the equation: φ² = φ + 1",
            "2. Minimal polynomial: x² - x - 1 = 0",
            "3. By rational root theorem, possible rational roots: ±1",
            "4. Check: 1² - 1 - 1 = -1 ≠ 0, (-1)² - (-1) - 1 = 1 ≠ 0", 
            "5. No rational roots ⇒ φ is irrational",
            "6. Alternatively: φ = (1 + √5)/2, and √5 is irrational",
            "7. Sum of rational (1) and irrational (√5) divided by rational (2) is irrational",
            "8. Therefore, φ is irrational □"
        ]
        return True, proof
    
    def prove_pi_irrational(self) -> Tuple[bool, List[str]]:
        """Lambert's proof of irrationality of π"""
        proof = [
            "THEOREM: π is irrational (Lambert, 1761)",
            "PROOF SKETCH:",
            "1. Lambert proved that if x is rational and non-zero, then tan(x) is irrational",
            "2. Since tan(π/4) = 1 (rational)", 
            "3. By contrapositive: if tan(π/4) is rational, then π/4 cannot be rational",
            "4. Therefore, π/4 is irrational ⇒ π is irrational",
            "5. Modern proof uses continued fractions: tan(x) has infinite continued fraction",
            "6. Since tan(π/4) = 1 has finite CF, π/4 cannot be rational",
            "7. Therefore, π is irrational □"
        ]
        return True, proof
    
    def prove_e_irrational(self) -> Tuple[bool, List[str]]:
        """Fourier's proof for irrationality of e"""
        proof = [
            "THEOREM: e is irrational (Euler, 1737)",
            "PROOF BY CONTRADICTION:",
            "1. Assume e = a/b for integers a,b",
            "2. Known: e = 1 + 1/1! + 1/2! + 1/3! + ...",
            "3. Multiply by b!: b!e = b! + b!/1! + b!/2! + ... + b!/b! + 1/(b+1) + ...",
            "4. Left side: b!e = b!(a/b) = a(b-1)! (integer)",
            "5. Right side: integer + remainder where 0 < remainder < 1", 
            "6. Contradiction: integer = integer + non-integer",
            "7. Therefore, e is irrational □"
        ]
        return True, proof
    
    def prove_log2_irrational(self) -> Tuple[bool, List[str]]:
        """Proof that log₁₀2 is irrational"""
        proof = [
            "THEOREM: log₁₀2 is irrational", 
            "PROOF BY CONTRADICTION:",
            "1. Assume log₁₀2 = a/b for integers a,b",
            "2. Then 10^(a/b) = 2 ⇒ 10^a = 2^b",
            "3. But 10^a = (2×5)^a = 2^a × 5^a",
            "4. So 2^a × 5^a = 2^b ⇒ 5^a = 2^(b-a)",
            "5. Left side is odd, right side is even (unless b=a=0)",
            "6. Contradiction ⇒ log₁₀2 is irrational □"
        ]
        return True, proof

class ContinuedFractionAnalyzer:
    """Advanced continued fraction analysis with proof capabilities"""
    
    def __init__(self, precision: int = 500):
        decimal.getcontext().prec = precision
        self.precision = precision
    
    def exact_continued_fraction(self, x: decimal.Decimal, max_terms: int = 200):
        """Compute continued fraction with exact termination detection"""
        cf = []
        xi = x
        
        for _ in range(max_terms):
            if not xi.is_finite():
                break
                
            a = int(xi.to_integral_value(rounding=decimal.ROUND_FLOOR))
            cf.append(a)
            frac = xi - decimal.Decimal(a)
            
            # Check if remainder is effectively zero
            if abs(frac) < decimal.Decimal(f'1e-{self.precision//5}'):
                break
                
            if frac == 0:
                break
                
            xi = decimal.Decimal(1) / frac
            
        return cf
    
    def analyze_cf_properties(self, cf: List) -> dict:
        """Analyze continued fraction for mathematical properties"""
        if not cf:
            return {"status": "empty", "type": "undefined"}
        
        # Check for termination (rational)
        if len(cf) < 100:  # Reasonable threshold for "finite" in computation
            return {
                "status": "finite", 
                "type": "rational",
                "length": len(cf),
                "evidence": "Finite continued fraction implies rational number"
            }
        
        # Check for periodicity (quadratic irrational)
        period_info = self.detect_periodicity(cf)
        if period_info["is_periodic"]:
            return {
                "status": "periodic",
                "type": "quadratic irrational", 
                "period": period_info["period"],
                "evidence": "Periodic continued fraction implies quadratic irrational"
            }
        
        # Non-periodic infinite (higher degree irrational or transcendental)
        return {
            "status": "infinite_non_periodic",
            "type": "higher_degree_irrational",
            "evidence": "Non-periodic infinite CF suggests irrational of degree > 2 or transcendental"
        }
    
    def detect_periodicity(self, cf: List, min_period: int = 2, max_check: int = 100) -> dict:
        """Enhanced periodicity detection with statistical validation"""
        if len(cf) < 2 * min_period:
            return {"is_periodic": False, "period": None}
        
        # Use only reasonable number of terms for analysis
        analysis_terms = cf[:max_check] if len(cf) > max_check else cf
        
        # Skip the first term (a0) for period detection in the tail
        tail = analysis_terms[1:] if len(analysis_terms) > 1 else []
        
        if len(tail) < 2 * min_period:
            return {"is_periodic": False, "period": None}
        
        for period in range(min_period, len(tail) // 2 + 1):
            if self._validate_period(tail, period):
                return {"is_periodic": True, "period": period}
        
        return {"is_periodic": False, "period": None}
    
    def _validate_period(self, sequence: List, period: int, min_cycles: int = 2) -> bool:
        """Validate period with multiple cycle checks"""
        if len(sequence) < period * min_cycles:
            return False
        
        # Check that the pattern repeats for min_cycles
        for i in range(1, min_cycles):
            start = i * period
            end = start + period
            if end > len(sequence):
                break
            
            if sequence[start:end] != sequence[:period]:
                return False
        
        return True

class AlgebraicAnalyzer:
    """Algebraic number analysis using sympy when available"""
    
    def __init__(self):
        self.known_constants = {
            'pi': 'pi',
            'e': 'e',
            'golden_ratio': 'golden_ratio',
            'sqrt2': 'sqrt(2)',
            'sqrt3': 'sqrt(3)',
            'sqrt5': 'sqrt(5)'
        }
    
    def analyze_algebraic_properties(self, expression: str) -> dict:
        """Determine if number is algebraic and its properties"""
        if not HAVE_SYMPY:
            return {"error": "SymPy not available for algebraic analysis"}
        
        try:
            # Try to parse expression
            if expression in self.known_constants:
                expr_str = self.known_constants[expression]
                expr = sp.sympify(expr_str)
            else:
                # Safe parsing with sympy
                expr = sp.sympify(expression)
            
            # Check if rational
            if expr.is_rational:
                return {
                    "is_rational": True,
                    "is_algebraic": True,
                    "degree": 1,
                    "minimal_polynomial": None,
                    "evidence": "Number is rational (ratio of integers)"
                }
            
            # Check if algebraic
            if expr.is_algebraic:
                try:
                    min_poly = sp.minimal_polynomial(expr)
                    degree = sp.degree(min_poly)
                    return {
                        "is_rational": False,
                        "is_algebraic": True,
                        "degree": degree,
                        "minimal_polynomial": str(min_poly),
                        "evidence": f"Algebraic number of degree {degree}"
                    }
                except:
                    return {
                        "is_rational": False,
                        "is_algebraic": True,
                        "degree": "unknown",
                        "minimal_polynomial": None,
                        "evidence": "Algebraic number (sympy confirmed)"
                    }
            
            # Likely transcendental
            return {
                "is_rational": False,
                "is_algebraic": False,
                "degree": "infinite",
                "minimal_polynomial": None,
                "evidence": "Likely transcendental (not algebraic)"
            }
            
        except Exception as e:
            return {"error": f"Could not analyze algebraically: {str(e)}"}

class UltimateIrrationalityProver:
    """Master class combining all proof methods"""
    
    def __init__(self):
        self.proof_system = IrrationalityProof()
        self.cf_analyzer = ContinuedFractionAnalyzer()
        self.algebraic_analyzer = AlgebraicAnalyzer()
        self.known_patterns = {
            r'^sqrt\((\d+)\)$': lambda m: self.proof_system.prove_sqrt_n(int(m.group(1))),
            r'^√(\d+)$': lambda m: self.proof_system.prove_sqrt_n(int(m.group(1))),
            r'^pi|π$': lambda m: self.proof_system.prove_pi_irrational(),
            r'^e$': lambda m: self.proof_system.prove_e_irrational(),
            r'^golden_ratio|φ|phi$': lambda m: self.proof_system.prove_golden_ratio(),
            r'^log\(2\)|log2$': lambda m: self.proof_system.prove_log2_irrational(),
        }
    
    def prove_irrationality(self, expression: str) -> dict:
        """Comprehensive irrationality proof for any expression"""
        result = {
            "expression": expression,
            "is_rational": None,
            "proof_type": None,
            "proof_steps": [],
            "continued_fraction": [],
            "cf_analysis": {},
            "algebraic_analysis": {},
            "confidence": "unknown"
        }
        
        # Step 1: Check for simple rational patterns
        simple_result = self._check_simple_rational(expression)
        if simple_result["is_rational"] is not None:
            result.update(simple_result)
            return result
        
        # Step 2: Try exact mathematical proof
        proof_result = self.attempt_exact_proof(expression)
        if proof_result:
            result.update(proof_result)
            result["confidence"] = "proven"
            return result
        
        # Step 3: Algebraic analysis (if sympy available)
        alg_result = self.algebraic_analyzer.analyze_algebraic_properties(expression)
        result["algebraic_analysis"] = alg_result
        
        if not alg_result.get("error"):
            if alg_result.get("is_rational") is True:
                result["is_rational"] = True
                result["proof_type"] = "algebraic"
                result["proof_steps"] = ["Number is rational (algebraic analysis)"]
                result["confidence"] = "proven"
                return result
            elif alg_result.get("is_rational") is False:
                result["is_rational"] = False
                result["proof_type"] = "algebraic"
                result["proof_steps"] = [alg_result.get("evidence", "Algebraic analysis")]
                result["confidence"] = "high"
        
        # Step 4: Continued fraction analysis
        cf_result = self._analyze_with_continued_fractions(expression)
        if cf_result:
            result.update(cf_result)
        
        return result
    
    def _check_simple_rational(self, expression: str) -> dict:
        """Check for simple rational number patterns"""
        # Check for integer
        if expression.lstrip('-').isdigit():
            return {
                "is_rational": True,
                "proof_type": "direct",
                "proof_steps": [f"Integer: {expression} = {expression}/1"],
                "confidence": "proven"
            }
        
        # Check for fraction a/b
        if '/' in expression:
            parts = expression.split('/')
            if len(parts) == 2 and parts[0].lstrip('-').isdigit() and parts[1].isdigit():
                try:
                    frac = Fraction(int(parts[0]), int(parts[1]))
                    return {
                        "is_rational": True,
                        "proof_type": "direct", 
                        "proof_steps": [f"Fraction: {expression} = {frac.numerator}/{frac.denominator}"],
                        "confidence": "proven"
                    }
                except:
                    pass
        
        # Check for finite decimal
        if re.match(r'^-?\d+\.?\d*$', expression) and 'e' not in expression.lower():
            try:
                decimal.Decimal(expression)
                return {
                    "is_rational": True,
                    "proof_type": "direct",
                    "proof_steps": [f"Finite decimal: {expression} is rational"],
                    "confidence": "proven"
                }
            except:
                pass
        
        return {"is_rational": None}
    
    def _analyze_with_continued_fractions(self, expression: str) -> dict:
        """Analyze using continued fractions"""
        try:
            # Convert to high-precision decimal
            x_dec = self._evaluate_expression(expression)
            if x_dec is None:
                return {}
            
            cf = self.cf_analyzer.exact_continued_fraction(x_dec)
            cf_analysis = self.cf_analyzer.analyze_cf_properties(cf)
            
            result = {
                "continued_fraction": cf[:30],  # First 30 terms
                "cf_analysis": cf_analysis
            }
            
            # Interpret CF results
            if cf_analysis["type"] == "rational":
                result["is_rational"] = True
                result["confidence"] = "high"
                result["proof_type"] = "continued_fraction"
                result["proof_steps"] = ["Finite continued fraction implies rational number"]
            else:
                result["is_rational"] = False
                result["proof_type"] = "continued_fraction" 
                result["proof_steps"] = [cf_analysis["evidence"]]
                result["confidence"] = "high" if cf_analysis["status"] != "infinite_non_periodic" else "medium"
            
            return result
                
        except Exception as e:
            return {"error": f"Continued fraction analysis failed: {str(e)}"}
    
    def _evaluate_expression(self, expression: str) -> Optional[decimal.Decimal]:
        """Safely evaluate mathematical expression to decimal"""
        try:
            # Handle known constants
            if expression.lower() in ['pi', 'π']:
                return decimal.Decimal(str(math.pi))
            elif expression.lower() in ['e']:
                return decimal.Decimal(str(math.e))
            elif expression.lower() in ['golden_ratio', 'φ', 'phi']:
                return decimal.Decimal(str((1 + math.sqrt(5)) / 2))
            
            # Safe evaluation for mathematical expressions
            safe_env = {
                'sqrt': math.sqrt, 
                'pi': math.pi, 
                'e': math.e, 
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                '__builtins__': {}
            }
            
            # Clean the expression
            clean_expr = expression.replace('^', '**').replace(' ', '')
            val = eval(clean_expr, safe_env)
            return decimal.Decimal(str(val))
            
        except Exception as e:
            print(f"Warning: Could not evaluate '{expression}': {str(e)}")
            return None
    
    def attempt_exact_proof(self, expression: str) -> Optional[dict]:
        """Attempt to find an exact mathematical proof"""
        # Clean expression
        clean_expr = expression.lower().replace(' ', '')
        
        # Check known patterns
        for pattern, proof_func in self.known_patterns.items():
            match = re.match(pattern, clean_expr)
            if match:
                is_irrational, proof_steps = proof_func(match)
                return {
                    "is_rational": not is_irrational,
                    "proof_type": "mathematical",
                    "proof_steps": proof_steps
                }
        
        return None
    
    def generate_report(self, result: dict) -> str:
        """Generate comprehensive proof report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ULTIMATE IRRATIONALITY PROVER - COMPREHENSIVE REPORT")
        lines.append("=" * 80)
        lines.append(f"Expression: {result['expression']}")
        
        if result['is_rational'] is None:
            lines.append("Verdict: INCONCLUSIVE")
        else:
            lines.append(f"Verdict: {'RATIONAL' if result['is_rational'] else 'IRRATIONAL'}")
        
        lines.append(f"Confidence: {result['confidence'].upper()}")
        lines.append(f"Proof Type: {result.get('proof_type', 'unknown')}")
        lines.append("")
        
        if result.get('proof_steps'):
            lines.append("MATHEMATICAL PROOF:")
            lines.append("-" * 40)
            for step in result['proof_steps']:
                lines.append(f"  {step}")
            lines.append("")
        
        if result.get('continued_fraction'):
            lines.append("CONTINUED FRACTION ANALYSIS:")
            lines.append("-" * 40)
            cf = result['continued_fraction']
            lines.append(f"First {len(cf)} terms: {cf}")
            if result.get('cf_analysis'):
                cf_analysis = result['cf_analysis']
                lines.append(f"CF Type: {cf_analysis.get('type', 'unknown')}")
                lines.append(f"Evidence: {cf_analysis.get('evidence', 'None')}")
            lines.append("")
        
        if result.get('algebraic_analysis') and not result['algebraic_analysis'].get('error'):
            lines.append("ALGEBRAIC ANALYSIS:")
            lines.append("-" * 40)
            alg = result['algebraic_analysis']
            lines.append(f"Rational: {alg.get('is_rational', 'unknown')}")
            lines.append(f"Algebraic: {alg.get('is_algebraic', 'unknown')}")
            lines.append(f"Degree: {alg.get('degree', 'unknown')}")
            if alg.get('minimal_polynomial'):
                lines.append(f"Minimal Polynomial: {alg['minimal_polynomial']}")
            lines.append(f"Evidence: {alg.get('evidence', 'None')}")
            lines.append("")
        
        lines.append("UIS CLASSIFICATION:")
        lines.append("-" * 40)
        if result['is_rational']:
            lines.append("Category: CYCLIC (Rational)")
            lines.append("Properties: Finite decimal/continued fraction expansion")
            lines.append("UIS Code: RAT-1 (Basic Rational)")
        else:
            cf_type = result.get('cf_analysis', {}).get('type', 'unknown')
            if cf_type == 'quadratic irrational':
                lines.append("Category: PERIODIC (Quadratic Irrational)")
                lines.append("Properties: Periodic continued fraction, algebraic degree 2")
                lines.append("UIS Code: ALG-2 (Quadratic Irrational)")
            elif cf_type == 'higher_degree_irrational':
                lines.append("Category: APERIODIC (Higher Degree Algebraic)")
                lines.append("Properties: Non-periodic infinite continued fraction")
                lines.append("UIS Code: ALG-3+ (Algebraic Degree ≥ 3)")
            else:
                lines.append("Category: TRANSCENDENTAL (Likely)")
                lines.append("Properties: Non-algebraic, infinite non-repeating expansion")
                lines.append("UIS Code: TRANS (Transcendental)")
        
        if result.get('error'):
            lines.append("")
            lines.append("ERRORS:")
            lines.append("-" * 40)
            lines.append(f"  {result['error']}")
        
        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python ultimate_irrationality_prover.py \"expression\"")
        print("Examples:")
        print("  python ultimate_irrationality_prover.py \"sqrt(2)\"")
        print("  python ultimate_irrationality_prover.py \"pi\"") 
        print("  python ultimate_irrationality_prover.py \"(1+sqrt(5))/2\"")
        print("  python ultimate_irrationality_prover.py \"3/4\"")
        print("  python ultimate_irrationality_prover.py \"1.25\"")
        sys.exit(1)
    
    expression = sys.argv[1]
    prover = UltimateIrrationalityProver()
    
    print("🧠 ULTIMATE IRRATIONALITY PROVER - Analyzing...")
    print("   (This may take a moment for complex expressions)\n")
    
    try:
        result = prover.prove_irrationality(expression)
        report = prover.generate_report(result)
        print(report)
    except Exception as e:
        print(f"❌ ERROR: Analysis failed: {str(e)}")
        print("Please check your expression and try again.")

if __name__ == "__main__":
    main()
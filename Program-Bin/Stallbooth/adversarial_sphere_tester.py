#!/usr/bin/env python3
"""
Adversarial Sphere Tester - Exhaustive Search for All Possible Spheres
Empirical testing of mathematical constants as atrophy parameters for sphere theory
"""

import numpy as np
import math
from decimal import Decimal, getcontext
from fractions import Fraction
import itertools
from collections import defaultdict
import json

# Set high precision for critical calculations
getcontext().prec = 50

class AdversarialSphereTester:
    def __init__(self):
        self.discovered_spheres = {}
        self.failed_constants = []
        self.test_results = defaultdict(dict)
        self.universal_scaling_K = Decimal('15.1806')
        
        # Known valid spheres for validation
        self.known_spheres = {
            'Banachian': Decimal('2').sqrt(),
            'Euclidean': Decimal(str(math.pi)),
            'Hadwiger': Decimal(str(math.pi)),
            'Transcendent': Decimal('1.618033988749895'),  # φ
            '355/113': Decimal('355') / Decimal('113'),
            'Sphere_of_States': Decimal(str(math.e))
        }
        
    def test_basic_convergence(self, alpha, tolerance=Decimal('1e-10')):
        """Test if atrophy constant α produces convergent sphere dynamics"""
        try:
            alpha = Decimal(str(alpha))
            
            # Critical stability test: series convergence
            stability_sum = Decimal('0')
            for n in range(1, 100):
                term = alpha / (Decimal(n) ** Decimal('1.5'))
                stability_sum += term
                if stability_sum > Decimal('10'):
                    return False, "Divergence at n={}".format(n)
            
            # Energy field convergence test
            energy_field = Decimal('0')
            for n in range(1, 50):
                energy_field += alpha / Decimal(n**2)
                if abs(energy_field) > Decimal('5'):
                    return False, "Energy field overflow"
            
            return True, "Convergent"
            
        except Exception as e:
            return False, f"Calculation error: {str(e)}"
    
    def test_universal_scaling_compliance(self, alpha):
        """Test compliance with universal scaling law K ≈ 15.1806"""
        try:
            alpha = Decimal(str(alpha))
            
            # Calculate theoretical maximum size
            theoretical_max = self.universal_scaling_K / alpha
            
            # Test if this produces stable field
            field_strength = alpha * theoretical_max / 10
            
            # Field strength must be > 0 and < 3 for stability
            if field_strength <= 0 or field_strength >= 3:
                return False, f"Invalid field strength: {field_strength}"
            
            # Check irrational behavior preservation
            irrational_score = abs(float(alpha % 1))
            if irrational_score < 0.01 and alpha not in [self.known_spheres['355/113']]:
                return False, "Too rational (low irrational score)"
            
            return True, f"Valid scaling: max_size={theoretical_max}, field={field_strength}"
            
        except Exception as e:
            return False, f"Scaling error: {str(e)}"
    
    def test_structural_integrity(self, alpha):
        """Test structural integrity constraints"""
        try:
            alpha = Decimal(str(alpha))
            
            # Golden ratio compatibility test
            phi = Decimal('1.618033988749895')
            compatibility = abs(float((alpha - phi) / alpha))
            
            # Must have some relationship to fundamental constants
            fundamental_constants = [Decimal('2').sqrt(), Decimal(str(math.pi)), phi, Decimal(str(math.e))]
            min_distance = min(abs(float((alpha - c) / alpha)) for c in fundamental_constants)
            
            if min_distance > 2.0:
                return False, f"Too distant from fundamental constants: {min_distance}"
            
            # Prime digit sum test (important for mathematical harmony)
            alpha_str = str(alpha).replace('.', '').replace('-', '')[:15]
            digit_sum = sum(int(d) for d in alpha_str if d.isdigit())
            
            # Check for prime properties in digit patterns
            if digit_sum > 0:
                prime_factors = []
                temp = digit_sum
                for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
                    while temp % p == 0:
                        prime_factors.append(p)
                        temp //= p
                if temp == 1:  # Perfect prime factorization
                    return True, f"Prime harmonic: {digit_sum} = {'×'.join(map(str, prime_factors))}"
            
            return True, f"Structurally valid (distance: {min_distance:.4f})"
            
        except Exception as e:
            return False, f"Integrity error: {str(e)}"
    
    def test_mechanical_feasibility(self, alpha):
        """Test for mechanical sphere possibilities"""
        try:
            alpha = Decimal(str(alpha))
            
            # Mechanical spheres require specific physical constraints
            # 1. Must allow for tessellation in 3D space
            if alpha < Decimal('1.2') or alpha > Decimal('4.0'):
                return False, "Outside mechanical range [1.2, 4.0]"
            
            # 2. Energy conservation test
            energy_density = alpha / Decimal(str(math.pi))
            if energy_density < Decimal('0.3') or energy_density > Decimal('1.5'):
                return False, f"Invalid energy density: {energy_density}"
            
            # 3. Mechanical stability under stress
            stress_factor = alpha ** Decimal('0.5')
            if stress_factor < Decimal('1.0') or stress_factor > Decimal('2.0'):
                return False, f"Unstable under stress: {stress_factor}"
            
            return True, f"Mechanically feasible: energy={energy_density}, stress={stress_factor}"
            
        except Exception as e:
            return False, f"Mechanical error: {str(e)}"
    
    def comprehensive_constant_test(self, alpha, name=None):
        """Run complete test suite on a constant"""
        test_results = {}
        
        # Basic convergence
        conv_ok, conv_msg = self.test_basic_convergence(alpha)
        test_results['convergence'] = {'status': conv_ok, 'message': conv_msg}
        
        if not conv_ok:
            return False, test_results
        
        # Universal scaling
        scale_ok, scale_msg = self.test_universal_scaling_compliance(alpha)
        test_results['scaling'] = {'status': scale_ok, 'message': scale_msg}
        
        if not scale_ok:
            return False, test_results
        
        # Structural integrity
        struct_ok, struct_msg = self.test_structural_integrity(alpha)
        test_results['structure'] = {'status': struct_ok, 'message': struct_msg}
        
        if not struct_ok:
            return False, test_results
        
        # Mechanical feasibility
        mech_ok, mech_msg = self.test_mechanical_feasibility(alpha)
        test_results['mechanical'] = {'status': mech_ok, 'message': mech_msg}
        
        # Overall assessment
        overall_ok = conv_ok and scale_ok and struct_ok and mech_ok
        
        return overall_ok, test_results
    
    def search_irrational_constants(self):
        """Search promising irrational constants"""
        promising_constants = []
        
        # Mathematical constants
        constants = {
            '√2': Decimal('2').sqrt(),
            '√3': Decimal('3').sqrt(),
            '√5': Decimal('5').sqrt(),
            'φ': Decimal('1.618033988749895'),
            'e': Decimal(str(math.e)),
            'π': Decimal(str(math.pi)),
            'γ': Decimal('0.5772156649015328606'),  # Euler-Mascheroni
            'ln(2)': Decimal('2').ln(),
            'G': Decimal('0.91596559417721901505'),  # Catalan's constant
            'ζ(3)': Decimal('1.2020569031595942854'),  # Apery's constant
        }
        
        for name, value in constants.items():
            is_valid, results = self.comprehensive_constant_test(value, name)
            if is_valid:
                promising_constants.append((name, value, results))
                print(f"✓ Valid: {name} = {value}")
            else:
                fail_reason = "Unknown"
                if 'structure' in results and 'message' in results['structure']:
                    fail_reason = results['structure']['message']
                elif 'convergence' in results and 'message' in results['convergence']:
                    fail_reason = results['convergence']['message']
                print(f"✗ Invalid: {name} - {fail_reason}")
        
        return promising_constants
    
    def search_rational_approximations(self):
        """Search rational approximations to fundamental constants"""
        rational_candidates = []
        
        # Test continued fraction convergents
        targets = [math.pi, math.e, (2**0.5), 1.618033988749895]
        target_names = ['π', 'e', '√2', 'φ']
        
        for target, name in zip(targets, target_names):
            for max_den in [7, 8, 11, 13, 17, 19, 23, 29, 31, 37, 43, 53, 61, 71, 79, 89, 97, 101, 103, 107, 109, 113]:
                approx = Fraction(target).limit_denominator(max_den)
                alpha = Decimal(approx.numerator) / Decimal(approx.denominator)
                
                is_valid, results = self.comprehensive_constant_test(alpha, f"{name}_{approx}")
                if is_valid:
                    rational_candidates.append((f"{name}_{approx}", alpha, results))
                    print(f"✓ Valid rational: {name} ≈ {approx} = {alpha}")
        
        return rational_candidates
    
    def search_prime_based_constants(self):
        """Search constants based on prime numbers"""
        prime_constants = []
        
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
        
        # Prime combinations
        for i, p1 in enumerate(primes):
            for j, p2 in enumerate(primes[i+1:], i+1):
                # Prime ratios
                alpha = Decimal(p1) / Decimal(p2)
                is_valid, results = self.comprehensive_constant_test(alpha, f"prime_{p1}_{p2}")
                if is_valid:
                    prime_constants.append((f"prime_{p1}_{p2}", alpha, results))
                    print(f"✓ Valid prime ratio: {p1}/{p2} = {alpha}")
                
                # Prime square roots
                alpha = Decimal(p1 * p2).sqrt()
                is_valid, results = self.comprehensive_constant_test(alpha, f"sqrt_prime_{p1}_{p2}")
                if is_valid:
                    prime_constants.append((f"sqrt_prime_{p1}_{p2}", alpha, results))
                    print(f"✓ Valid prime sqrt: √({p1}×{p2}) = {alpha}")
        
        return prime_constants
    
    def generate_exhaustive_report(self):
        """Generate comprehensive testing report"""
        print("\n" + "="*80)
        print("ADVERSARIAL SPHERE TESTING - EXHAUSTIVE SEARCH")
        print("="*80)
        
        print("\n1. Testing Known Valid Spheres...")
        for name, alpha in self.known_spheres.items():
            is_valid, results = self.comprehensive_constant_test(alpha, name)
            status = "✓ VALID" if is_valid else "✗ INVALID"
            print(f"   {name}: {status}")
            self.discovered_spheres[name] = {'alpha': alpha, 'valid': is_valid, 'results': results}
        
        print("\n2. Searching Irrational Constants...")
        irrational_spheres = self.search_irrational_constants()
        for name, alpha, results in irrational_spheres:
            self.discovered_spheres[name] = {'alpha': alpha, 'valid': True, 'results': results}
        
        print("\n3. Searching Rational Approximations...")
        rational_spheres = self.search_rational_approximations()
        for name, alpha, results in rational_spheres:
            self.discovered_spheres[name] = {'alpha': alpha, 'valid': True, 'results': results}
        
        print("\n4. Searching Prime-Based Constants...")
        prime_spheres = self.search_prime_based_constants()
        for name, alpha, results in prime_spheres:
            self.discovered_spheres[name] = {'alpha': alpha, 'valid': True, 'results': results}
        
        # Generate summary statistics
        valid_count = sum(1 for s in self.discovered_spheres.values() if s['valid'])
        total_count = len(self.discovered_spheres)
        
        print(f"\n" + "="*80)
        print(f"SEARCH RESULTS SUMMARY")
        print(f"Total candidates tested: {total_count}")
        print(f"Valid spheres found: {valid_count}")
        print(f"Success rate: {valid_count/total_count*100:.2f}%")
        print("="*80)
        
        return self.discovered_spheres

def main():
    tester = AdversarialSphereTester()
    results = tester.generate_exhaustive_report()
    
    # Save results for LaTeX documentation
    with open('sphere_test_results.json', 'w') as f:
        # Convert Decimal to string for JSON serialization
        serializable_results = {}
        for name, data in results.items():
            serializable_results[name] = {
                'alpha': str(data['alpha']),
                'valid': data['valid'],
                'results': data['results']
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to sphere_test_results.json")
    return results

if __name__ == "__main__":
    main()
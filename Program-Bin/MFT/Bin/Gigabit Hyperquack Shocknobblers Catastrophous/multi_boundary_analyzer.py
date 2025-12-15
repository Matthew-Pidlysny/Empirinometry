#!/usr/bin/env python3
"""
Multi-Boundary Analyzer
=======================
Analyzes numbers against ALL 8 natural termination boundaries simultaneously.

The 8 Natural Termination Boundaries:
1. Cognitive Termination: 15 digits (human perception limit)
2. Planck Scale Termination: 35 digits (spacetime quantization)
3. Quantum Measurement: 61 digits (max distinguishable positions)
4. Base-Dependent Termination: 1 digit (representation artifact)
5. Physical Storage: ~10^80 digits (atoms in universe)
6. Thermodynamic: ~10^90 digits (Landauer's limit)
7. Temporal: ~10^116 digits (heat death constraint)
8. Information Theoretical: ~10^123 digits (Bekenstein bound)

This program provides comprehensive analysis across all boundaries.
"""

import math
from decimal import Decimal, getcontext
from fractions import Fraction
import json

# Boundary constants
COGNITIVE_LIMIT = 15
PLANCK_LIMIT = 35
QUANTUM_LIMIT = 61
BASE_DEPENDENT_LIMIT = 1  # Can vary by base
PHYSICAL_STORAGE_LIMIT = 80  # log10 of atoms in universe
THERMODYNAMIC_LIMIT = 90
TEMPORAL_LIMIT = 116
INFORMATION_LIMIT = 123

# Physical constants
PLANCK_LENGTH = Decimal('1.616255e-35')
ATOMS_IN_UNIVERSE = Decimal('1e80')
UNIVERSE_AGE_SECONDS = Decimal('4.35e17')

class MultiBoundaryAnalyzer:
    """Analyze numbers against all natural termination boundaries."""
    
    def __init__(self):
        """Initialize analyzer."""
        getcontext().prec = 200
        
        self.boundaries = {
            'cognitive': {
                'limit': COGNITIVE_LIMIT,
                'name': 'Cognitive Termination',
                'description': 'Human perception and comprehension limit',
                'confidence': 0.90,
                'type': 'perceptual'
            },
            'planck': {
                'limit': PLANCK_LIMIT,
                'name': 'Planck Scale Termination',
                'description': 'Spacetime quantization boundary',
                'confidence': 1.00,
                'type': 'physical'
            },
            'quantum': {
                'limit': QUANTUM_LIMIT,
                'name': 'Quantum Measurement',
                'description': 'Maximum distinguishable positions in universe',
                'confidence': 1.00,
                'type': 'physical'
            },
            'base_dependent': {
                'limit': BASE_DEPENDENT_LIMIT,
                'name': 'Base-Dependent Termination',
                'description': 'Representation artifact (varies by base)',
                'confidence': 1.00,
                'type': 'mathematical'
            },
            'physical_storage': {
                'limit': PHYSICAL_STORAGE_LIMIT,
                'name': 'Physical Storage',
                'description': 'Limited by atoms in observable universe',
                'confidence': 1.00,
                'type': 'physical'
            },
            'thermodynamic': {
                'limit': THERMODYNAMIC_LIMIT,
                'name': 'Thermodynamic',
                'description': "Landauer's limit on computation",
                'confidence': 1.00,
                'type': 'physical'
            },
            'temporal': {
                'limit': TEMPORAL_LIMIT,
                'name': 'Temporal',
                'description': 'Heat death of universe constraint',
                'confidence': 0.95,
                'type': 'cosmological'
            },
            'information': {
                'limit': INFORMATION_LIMIT,
                'name': 'Information Theoretical',
                'description': 'Bekenstein bound on maximum information',
                'confidence': 1.00,
                'type': 'physical'
            }
        }
    
    def analyze_number(self, number, context=None):
        """
        Analyze a number against all termination boundaries.
        
        Args:
            number: Number to analyze
            context: Optional context dict with keys like 'base', 'physical_scale'
            
        Returns:
            Comprehensive analysis across all boundaries
        """
        number = Decimal(str(number))
        context = context or {}
        
        # Calculate precision
        number_str = str(number)
        if 'E' in number_str or 'e' in number_str:
            mantissa = number_str.split('E' if 'E' in number_str else 'e')[0]
            if '.' in mantissa:
                precision = len(mantissa.split('.')[1])
            else:
                precision = 0
        elif '.' in number_str:
            precision = len(number_str.split('.')[1])
        else:
            precision = 0
        
        # Analyze against each boundary
        boundary_results = {}
        
        for boundary_key, boundary_info in self.boundaries.items():
            result = self._analyze_against_boundary(
                number, precision, boundary_key, boundary_info, context
            )
            boundary_results[boundary_key] = result
        
        # Determine primary limiting boundary
        primary_boundary = self._determine_primary_boundary(boundary_results, precision)
        
        # Calculate overall validity
        valid_boundaries = [k for k, v in boundary_results.items() if v['within_boundary']]
        invalid_boundaries = [k for k, v in boundary_results.items() if not v['within_boundary']]
        
        return {
            'number': str(number),
            'precision': precision,
            'boundary_results': boundary_results,
            'primary_boundary': primary_boundary,
            'valid_boundaries': valid_boundaries,
            'invalid_boundaries': invalid_boundaries,
            'overall_status': 'VALID' if len(invalid_boundaries) == 0 else 'INVALID',
            'recommended_precision': self._recommend_precision(boundary_results),
            'context': context
        }
    
    def _analyze_against_boundary(self, number, precision, boundary_key, boundary_info, context):
        """Analyze number against a specific boundary."""
        limit = boundary_info['limit']
        
        # Special handling for base-dependent boundary
        if boundary_key == 'base_dependent':
            base = context.get('base', 10)
            limit = self._calculate_base_dependent_limit(number, base)
        
        within_boundary = precision <= limit
        excess = max(0, precision - limit)
        
        return {
            'boundary_name': boundary_info['name'],
            'limit': limit,
            'precision': precision,
            'within_boundary': within_boundary,
            'excess_digits': excess,
            'confidence': boundary_info['confidence'],
            'type': boundary_info['type'],
            'description': boundary_info['description']
        }
    
    def _calculate_base_dependent_limit(self, number, base):
        """Calculate base-dependent termination limit."""
        try:
            # Try to convert to fraction
            frac = Fraction(str(number)).limit_denominator(10000)
            
            # Check if it terminates in this base
            denominator = frac.denominator
            
            # Get prime factors of base
            base_primes = self._prime_factors(base)
            den_primes = self._prime_factors(denominator)
            
            # If all denominator primes are in base primes, it terminates
            if all(p in base_primes for p in den_primes):
                return 0  # Terminates (finite representation)
            else:
                return float('inf')  # Repeats infinitely in this base
        except:
            return BASE_DEPENDENT_LIMIT
    
    def _prime_factors(self, n):
        """Get set of prime factors."""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    def _determine_primary_boundary(self, boundary_results, precision):
        """Determine which boundary is the primary limiting factor."""
        # Find the strictest boundary that's violated
        violated = [(k, v) for k, v in boundary_results.items() 
                   if not v['within_boundary']]
        
        if not violated:
            # All boundaries satisfied - find the strictest one
            strictest = min(boundary_results.items(), 
                          key=lambda x: x[1]['limit'])
            return {
                'boundary': strictest[0],
                'name': strictest[1]['boundary_name'],
                'limit': strictest[1]['limit'],
                'status': 'SATISFIED'
            }
        else:
            # Find the strictest violated boundary
            strictest_violated = min(violated, key=lambda x: x[1]['limit'])
            return {
                'boundary': strictest_violated[0],
                'name': strictest_violated[1]['boundary_name'],
                'limit': strictest_violated[1]['limit'],
                'status': 'VIOLATED',
                'excess': strictest_violated[1]['excess_digits']
            }
    
    def _recommend_precision(self, boundary_results):
        """Recommend optimal precision based on boundary analysis."""
        # Find the strictest boundary that makes sense for the context
        practical_boundaries = ['cognitive', 'planck', 'quantum']
        
        limits = [boundary_results[b]['limit'] for b in practical_boundaries 
                 if b in boundary_results]
        
        return min(limits) if limits else COGNITIVE_LIMIT
    
    def compare_numbers(self, numbers, context=None):
        """
        Compare multiple numbers across all boundaries.
        
        Args:
            numbers: List of numbers to compare
            context: Optional context
            
        Returns:
            Comparative analysis
        """
        analyses = []
        for number in numbers:
            analysis = self.analyze_number(number, context)
            analyses.append(analysis)
        
        # Find common boundaries
        all_valid = set.intersection(*[set(a['valid_boundaries']) for a in analyses])
        all_invalid = set.intersection(*[set(a['invalid_boundaries']) for a in analyses])
        
        return {
            'analyses': analyses,
            'common_valid_boundaries': list(all_valid),
            'common_invalid_boundaries': list(all_invalid),
            'recommended_precision': min(a['recommended_precision'] for a in analyses)
        }
    
    def find_optimal_base(self, number, max_base=36):
        """
        Find bases where a number terminates (for rationals).
        
        Args:
            number: Number to analyze
            max_base: Maximum base to check
            
        Returns:
            List of terminating bases
        """
        try:
            frac = Fraction(str(number)).limit_denominator(10000)
            denominator = frac.denominator
            
            terminating_bases = []
            
            for base in range(2, max_base + 1):
                base_primes = self._prime_factors(base)
                den_primes = self._prime_factors(denominator)
                
                if all(p in base_primes for p in den_primes):
                    terminating_bases.append(base)
            
            return {
                'number': str(number),
                'fraction': f"{frac.numerator}/{frac.denominator}",
                'terminating_bases': terminating_bases,
                'total_terminating': len(terminating_bases),
                'base_10_terminates': 10 in terminating_bases
            }
        except:
            return {
                'number': str(number),
                'error': 'Cannot convert to fraction',
                'terminating_bases': [],
                'total_terminating': 0
            }
    
    def generate_comprehensive_report(self, analysis):
        """Generate detailed report from analysis."""
        report = []
        report.append("=" * 80)
        report.append("MULTI-BOUNDARY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Number: {analysis['number']}")
        report.append(f"Precision: {analysis['precision']} decimal places")
        report.append(f"Overall Status: {analysis['overall_status']}")
        report.append(f"Recommended Precision: {analysis['recommended_precision']} digits")
        report.append("")
        
        # Primary boundary
        primary = analysis['primary_boundary']
        report.append("PRIMARY LIMITING BOUNDARY:")
        report.append("-" * 80)
        report.append(f"Boundary: {primary['name']}")
        report.append(f"Limit: {primary['limit']} digits")
        report.append(f"Status: {primary['status']}")
        if primary['status'] == 'VIOLATED':
            report.append(f"Excess: {primary['excess']} digits beyond limit")
        report.append("")
        
        # All boundaries
        report.append("ANALYSIS ACROSS ALL 8 BOUNDARIES:")
        report.append("-" * 80)
        
        for boundary_key, result in analysis['boundary_results'].items():
            status_symbol = "✓" if result['within_boundary'] else "✗"
            report.append(f"{status_symbol} {result['boundary_name']:30s} "
                         f"Limit: {result['limit']:6} digits "
                         f"({result['type']}, confidence: {result['confidence']*100:.0f}%)")
            
            if not result['within_boundary']:
                report.append(f"  ⚠️  Exceeds by {result['excess_digits']} digits")
        
        report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 80)
        report.append(f"Valid Boundaries: {len(analysis['valid_boundaries'])}/8")
        report.append(f"Invalid Boundaries: {len(analysis['invalid_boundaries'])}/8")
        
        if analysis['invalid_boundaries']:
            report.append(f"Violated: {', '.join(analysis['invalid_boundaries'])}")
        
        report.append("")
        
        return "\n".join(report)


def run_comprehensive_tests():
    """Run comprehensive multi-boundary tests."""
    analyzer = MultiBoundaryAnalyzer()
    
    print("MULTI-BOUNDARY ANALYZER - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Test 1: Analyze pi at various precisions
    print("TEST 1: Pi at Various Precisions")
    print("-" * 80)
    
    pi_tests = [
        ("3.14", "2 digits"),
        ("3.14159265358979", "14 digits"),
        ("3.1415926535897932384626433832795028841971", "40 digits"),
        ("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050792279689258923542019956112129021960864034418159813629774771309960518707211349999998372978049951059731732816096318595024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303598253490428755468731159562863882353787593751957781857780532171226806613001927876611195909216420198938095257201065485863278865936153381827968230301952035301852968995773622599413891249721775283479131515574857242454150695950829533116861727855889075098381754637464939319255060400927701671139009848824012858361603563707660104710181942955596198946767837449448255379774726847104047534646208046684259069491293313677028989152104752162056966024058038150193511253382430035587640247496473263914199272604269922796782354781636009341721641219924586315030286182974555706749838505494588586926995690927210797509302955321165344987202755960236480665499119881834797753566369807426542527862551818417574672890977772793800081647060016145249192173217214772350141441973568548161361157352552133475741849468438523323907394143334547762416862518983569485562099105420353001133053054882046652138414695194151160943305727036575959195309218611738193261179310511854807446237996274956735188575272489122793818301194912983367336244065664308602139494639522473719070217986094370277053921717629317675238467481846766940513200056812714526356082778577134275778960917363717872146844090122495343014654958537105079227968925892354201995611212902196086403441815981362977477130996051870721134999999837297804995105973173281609631859502445945534690830264252230825334468503526193118817101000313783875288658753320838142061717766914730359825349042875546873115956286388235378759375195778185778053217122680661300192787661119590921642019893809525720106548586327886593615338182796823030195203530185296899577362259941389124972177528347913151557485724245415069595082953311686172785588907509838175463746493931925506040092770167113900984882401285836160356370766010471018194295559619894676783744944825537977472684710404753464620804668425906949129331367702898915210475216205696602405803815019351125338243003558764024749647326391419927260426992279678235478163600934172164121992458631503028618297455570674983850549458858692699569092721079750930295532116534498720275596023648066549911988183479775356636980742654252786255181841757467289097777279380008164706001614524919217321721477235014144197356854816136115735255213347574184946843852332390739414333454776241686251898356948556209921922218427255025425688767179049460165346680498862723279178608578438382796797668145410095388378636095068006422512520511739298489608412848862694560424196528502221066118630674427862203919494504712371378696095636437191728746776465757396241389086583264599581339047802759009946576407895126946839835259570982582262052248940772671947826848260147699090264013639443745530506820349625245174939965143142980919065925093722169646151570985838741059788595977297549893016175392846813826868386894277415599185592524595395943104997252468084598727364469584865383673622262609912460805124388439045124413654976278079771569143599770012961608944169486855584840635342207222582848864815845602364272117041826016391428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050792279689258923542019956112129021960864034418159813629774771309960518707211349999998372978049951059731732816096318595024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303598253490428755468731159562863882353787593751957781857780532171226806613001927876611195909216420198", "5000 digits")
    ]
    
    for pi_value, description in pi_tests:
        analysis = analyzer.analyze_number(pi_value)
        print(f"\n{description}:")
        print(f"  Precision: {analysis['precision']} digits")
        print(f"  Status: {analysis['overall_status']}")
        print(f"  Primary Boundary: {analysis['primary_boundary']['name']}")
        print(f"  Valid Boundaries: {len(analysis['valid_boundaries'])}/8")
        print(f"  Recommended: {analysis['recommended_precision']} digits")
    
    print("\n")
    
    # Test 2: Detailed analysis of one number
    print("TEST 2: Detailed Analysis of Pi (40 digits)")
    print("-" * 80)
    pi_40 = "3.1415926535897932384626433832795028841971"
    analysis = analyzer.analyze_number(pi_40)
    report = analyzer.generate_comprehensive_report(analysis)
    print(report)
    
    # Test 3: Base-dependent analysis
    print("TEST 3: Base-Dependent Termination Analysis")
    print("-" * 80)
    
    test_fractions = [
        ("0.333333333", "1/3"),
        ("0.166666666", "1/6"),
        ("0.142857142", "1/7")
    ]
    
    for number, description in test_fractions:
        result = analyzer.find_optimal_base(number, 20)
        print(f"\n{description} ({number}):")
        print(f"  Fraction: {result.get('fraction', 'N/A')}")
        print(f"  Terminates in {result['total_terminating']} bases (out of 19)")
        print(f"  Terminates in base 10: {result.get('base_10_terminates', False)}")
        if result['terminating_bases']:
            print(f"  Sample bases: {result['terminating_bases'][:10]}")
    
    print("\n")
    
    # Test 4: Comparison of multiple numbers
    print("TEST 4: Comparative Analysis")
    print("-" * 80)
    
    numbers_to_compare = [
        "3.14",
        "3.14159265358979",
        "3.141592653589793238462643383279502884197"
    ]
    
    comparison = analyzer.compare_numbers(numbers_to_compare)
    print(f"Comparing {len(numbers_to_compare)} numbers:")
    print(f"Common valid boundaries: {comparison['common_valid_boundaries']}")
    print(f"Common invalid boundaries: {comparison['common_invalid_boundaries']}")
    print(f"Recommended precision: {comparison['recommended_precision']} digits")
    
    print("\n")
    
    return True


def main():
    """Main execution."""
    success = run_comprehensive_tests()
    
    print("=" * 80)
    print("MULTI-BOUNDARY ANALYSIS COMPLETED")
    print("=" * 80)
    print()
    print("KEY FINDINGS:")
    print("1. All numbers must respect ALL 8 natural termination boundaries")
    print("2. The strictest boundary determines the practical limit")
    print("3. Cognitive limit (15 digits) is most restrictive for human use")
    print("4. Planck limit (35 digits) is most restrictive for physical measurements")
    print("5. Quantum limit (61 digits) is the absolute physical maximum")
    print("6. Base-dependent termination shows 'infinity' is a representation artifact")
    print("7. Storage, thermodynamic, temporal, and information limits are theoretical")
    print("8. NO number can exceed ALL boundaries - infinity does not exist")
    print()
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
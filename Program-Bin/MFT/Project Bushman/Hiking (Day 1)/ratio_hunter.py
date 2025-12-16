"""
Project Bushman Suite 7: "The Observatory"
Program 2: Ratio Hunter

Purpose: Deep analysis of ratios between spherical objects to find C* patterns,
         plasticity rules, and dimensional emergence signatures.

Author: SuperNinja AI (for Matthew Pidlysny)
Date: December 2024
"""

import json
import math
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics

# Constants
C_STAR = 0.894751918
F_01 = C_STAR
F_12 = 4.0 * C_STAR
F_23 = 7.07 * F_12
F_34 = 5.09 * C_STAR
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2
SQRT_2 = math.sqrt(2)
SQRT_3 = math.sqrt(3)

# Plasticity targets
PLASTICITY_TARGETS = {
    '4.0': 4.0,
    '7.0': 7.0,
    '7.07': 7.07,
    '5.0': 5.0,
    '5.09': 5.09,
    '10.0': 10.0,
    '2.0': 2.0,
    '3.0': 3.0,
}

# Mathematical constants
MATH_CONSTANTS = {
    'C*': C_STAR,
    '2*C*': 2 * C_STAR,
    '3*C*': 3 * C_STAR,
    '4*C*': 4 * C_STAR,
    '5*C*': 5 * C_STAR,
    '7*C*': 7 * C_STAR,
    '1/C*': 1 / C_STAR,
    'π': PI,
    '2π': 2 * PI,
    'π/2': PI / 2,
    'e': E,
    'φ': PHI,
    '√2': SQRT_2,
    '√3': SQRT_3,
    'F₀₁': F_01,
    'F₁₂': F_12,
    'F₂₃': F_23,
    'F₃₄': F_34,
}

class RatioHunter:
    """Deep analysis of ratios between spherical objects."""
    
    def __init__(self, catalog_file: str = 'scale_scanner_results.json'):
        with open(catalog_file, 'r') as f:
            data = json.load(f)
            self.catalog = data['catalog']
    
    def calculate_all_ratios(self) -> List[Dict]:
        """Calculate all possible ratios between objects."""
        ratios = []
        items = list(self.catalog.items())
        
        for i, (name1, data1) in enumerate(items):
            for name2, data2 in items[i+1:]:
                r1 = data1['radius']
                r2 = data2['radius']
                m1 = data1['mass']
                m2 = data2['mass']
                
                # Radius ratios
                radius_ratio = max(r1, r2) / min(r1, r2)
                
                # Mass ratios
                mass_ratio = max(m1, m2) / min(m1, m2)
                
                # Density ratios (mass/volume for spheres)
                density1 = m1 / ((4/3) * PI * r1**3)
                density2 = m2 / ((4/3) * PI * r2**3)
                density_ratio = max(density1, density2) / min(density1, density2)
                
                ratios.append({
                    'object1': name1,
                    'object2': name2,
                    'radius1': r1,
                    'radius2': r2,
                    'mass1': m1,
                    'mass2': m2,
                    'radius_ratio': radius_ratio,
                    'mass_ratio': mass_ratio,
                    'density_ratio': density_ratio,
                    'scale1': data1['scale'],
                    'scale2': data2['scale'],
                    'type1': data1['type'],
                    'type2': data2['type'],
                })
        
        return ratios
    
    def find_constant_matches(self, tolerance: float = 0.05) -> Dict[str, List[Dict]]:
        """Find ratios matching mathematical constants."""
        ratios = self.calculate_all_ratios()
        matches = defaultdict(list)
        
        for ratio_data in ratios:
            for ratio_type in ['radius_ratio', 'mass_ratio', 'density_ratio']:
                ratio_value = ratio_data[ratio_type]
                
                for const_name, const_value in MATH_CONSTANTS.items():
                    error = abs(ratio_value - const_value) / const_value
                    
                    if error < tolerance:
                        match = {
                            **ratio_data,
                            'ratio_type': ratio_type,
                            'ratio_value': ratio_value,
                            'constant': const_name,
                            'constant_value': const_value,
                            'error_percent': error * 100
                        }
                        matches[const_name].append(match)
        
        # Sort each constant's matches by error
        for const_name in matches:
            matches[const_name] = sorted(matches[const_name], 
                                        key=lambda x: x['error_percent'])
        
        return dict(matches)
    
    def find_plasticity_patterns(self, tolerance: float = 0.05) -> List[Dict]:
        """Find ratios matching plasticity rule (4-7-5 pattern)."""
        ratios = self.calculate_all_ratios()
        plasticity_matches = []
        
        for ratio_data in ratios:
            radius_ratio = ratio_data['radius_ratio']
            
            for target_name, target_value in PLASTICITY_TARGETS.items():
                error = abs(radius_ratio - target_value) / target_value
                
                if error < tolerance:
                    plasticity_matches.append({
                        **ratio_data,
                        'target': target_name,
                        'target_value': target_value,
                        'error_percent': error * 100
                    })
        
        return sorted(plasticity_matches, key=lambda x: x['error_percent'])
    
    def analyze_scale_transitions(self) -> List[Dict]:
        """Analyze ratios at scale transition boundaries."""
        ratios = self.calculate_all_ratios()
        transitions = []
        
        for ratio_data in ratios:
            if ratio_data['scale1'] != ratio_data['scale2']:
                # This is a scale transition
                transitions.append({
                    **ratio_data,
                    'scale_jump': abs(ratio_data['scale2'] - ratio_data['scale1'])
                })
        
        return sorted(transitions, key=lambda x: x['radius_ratio'])
    
    def find_ratio_clusters(self, bin_width: float = 0.1) -> Dict:
        """Find clustering in ratio distributions."""
        ratios = self.calculate_all_ratios()
        radius_ratios = [r['radius_ratio'] for r in ratios]
        
        # Create logarithmic bins
        log_ratios = [math.log10(r) for r in radius_ratios]
        min_log = min(log_ratios)
        max_log = max(log_ratios)
        
        bins = defaultdict(list)
        for i, log_ratio in enumerate(log_ratios):
            bin_index = int((log_ratio - min_log) / bin_width)
            bins[bin_index].append(ratios[i])
        
        # Find peaks (bins with many ratios)
        clusters = []
        for bin_index, bin_ratios in bins.items():
            if len(bin_ratios) >= 5:  # Significant cluster
                avg_ratio = statistics.mean([r['radius_ratio'] for r in bin_ratios])
                clusters.append({
                    'bin_index': bin_index,
                    'count': len(bin_ratios),
                    'average_ratio': avg_ratio,
                    'log_ratio': min_log + bin_index * bin_width,
                    'examples': bin_ratios[:3]  # First 3 examples
                })
        
        return sorted(clusters, key=lambda x: x['count'], reverse=True)
    
    def analyze_earth_moon_sun(self) -> Dict:
        """Special analysis of Earth-Moon-Sun system."""
        earth = self.catalog['earth']
        moon = self.catalog['moon']
        sun = self.catalog['sun']
        
        # Calculate all ratios
        moon_earth_radius = moon['radius'] / earth['radius']
        earth_sun_radius = earth['radius'] / sun['radius']
        moon_sun_radius = moon['radius'] / sun['radius']
        
        moon_earth_mass = moon['mass'] / earth['mass']
        earth_sun_mass = earth['mass'] / sun['mass']
        moon_sun_mass = moon['mass'] / sun['mass']
        
        # Check for C* matches
        ratios_to_check = {
            'Moon/Earth radius': moon_earth_radius,
            'Earth/Sun radius': earth_sun_radius,
            'Moon/Sun radius': moon_sun_radius,
            'Moon/Earth mass': moon_earth_mass,
            'Earth/Sun mass': earth_sun_mass,
            'Moon/Sun mass': moon_sun_mass,
        }
        
        matches = {}
        for ratio_name, ratio_value in ratios_to_check.items():
            best_match = None
            best_error = float('inf')
            
            for const_name, const_value in MATH_CONSTANTS.items():
                error = abs(ratio_value - const_value) / const_value
                if error < best_error:
                    best_error = error
                    best_match = const_name
            
            matches[ratio_name] = {
                'value': ratio_value,
                'best_match': best_match,
                'best_match_value': MATH_CONSTANTS[best_match],
                'error_percent': best_error * 100
            }
        
        return matches
    
    def find_golden_ratios(self, tolerance: float = 0.05) -> List[Dict]:
        """Find ratios matching golden ratio φ."""
        ratios = self.calculate_all_ratios()
        golden_matches = []
        
        for ratio_data in ratios:
            radius_ratio = ratio_data['radius_ratio']
            
            # Check φ and related values
            targets = {
                'φ': PHI,
                'φ²': PHI**2,
                '1/φ': 1/PHI,
                'φ³': PHI**3,
            }
            
            for target_name, target_value in targets.items():
                error = abs(radius_ratio - target_value) / target_value
                
                if error < tolerance:
                    golden_matches.append({
                        **ratio_data,
                        'target': target_name,
                        'target_value': target_value,
                        'error_percent': error * 100
                    })
        
        return sorted(golden_matches, key=lambda x: x['error_percent'])
    
    def analyze_dimensional_jumps(self) -> Dict:
        """Analyze if minimum fields appear in scale transitions."""
        transitions = self.analyze_scale_transitions()
        
        # Group by scale jump magnitude
        jump_analysis = defaultdict(list)
        
        for trans in transitions:
            jump = trans['scale_jump']
            ratio = trans['radius_ratio']
            
            # Check if ratio matches minimum fields
            min_fields = {
                'F₀₁': F_01,
                'F₁₂': F_12,
                'F₂₃': F_23,
                'F₃₄': F_34,
            }
            
            best_match = None
            best_error = float('inf')
            
            for field_name, field_value in min_fields.items():
                # Check both ratio and log ratio
                error1 = abs(ratio - field_value) / max(ratio, field_value)
                
                # Avoid division by zero in log comparison
                log_ratio = math.log10(ratio) if ratio > 0 else 0
                log_field = math.log10(field_value) if field_value > 0 else 0
                max_log = max(abs(log_ratio), abs(log_field))
                
                if max_log > 0:
                    error2 = abs(log_ratio - log_field) / max_log
                else:
                    error2 = error1
                
                error = min(error1, error2)
                
                if error < best_error:
                    best_error = error
                    best_match = field_name
            
            jump_analysis[jump].append({
                'transition': f"{trans['object1']} → {trans['object2']}",
                'ratio': ratio,
                'best_field_match': best_match,
                'error_percent': best_error * 100
            })
        
        return dict(jump_analysis)
    
    def statistical_significance(self, matches: List[Dict]) -> Dict:
        """Calculate statistical significance of matches."""
        if not matches:
            return {'significant': False, 'reason': 'No matches found'}
        
        errors = [m['error_percent'] for m in matches]
        
        return {
            'count': len(matches),
            'mean_error': statistics.mean(errors),
            'median_error': statistics.median(errors),
            'std_error': statistics.stdev(errors) if len(errors) > 1 else 0,
            'min_error': min(errors),
            'max_error': max(errors),
            'significant': len(matches) >= 10 and statistics.mean(errors) < 5.0
        }
    
    def generate_report(self) -> Dict:
        """Generate complete ratio analysis report."""
        constant_matches = self.find_constant_matches()
        plasticity_matches = self.find_plasticity_patterns()
        golden_matches = self.find_golden_ratios()
        clusters = self.find_ratio_clusters()
        ems_analysis = self.analyze_earth_moon_sun()
        dimensional_jumps = self.analyze_dimensional_jumps()
        
        return {
            'constant_matches': {
                const: matches[:10]  # Top 10 for each constant
                for const, matches in constant_matches.items()
            },
            'constant_statistics': {
                const: self.statistical_significance(matches)
                for const, matches in constant_matches.items()
            },
            'plasticity_matches': plasticity_matches[:20],
            'plasticity_statistics': self.statistical_significance(plasticity_matches),
            'golden_ratio_matches': golden_matches[:10],
            'golden_statistics': self.statistical_significance(golden_matches),
            'ratio_clusters': clusters[:10],
            'earth_moon_sun_analysis': ems_analysis,
            'dimensional_jump_analysis': dimensional_jumps,
            'summary': {
                'total_constants_found': len(constant_matches),
                'total_matches': sum(len(m) for m in constant_matches.values()),
                'plasticity_matches_count': len(plasticity_matches),
                'golden_matches_count': len(golden_matches),
                'clusters_found': len(clusters)
            }
        }


def run_tests() -> Dict:
    """Run test suite for ratio hunter."""
    hunter = RatioHunter()
    results = {
        'tests': [],
        'passed': 0,
        'failed': 0,
        'total': 8
    }
    
    # Test 1: C* matches found
    constant_matches = hunter.find_constant_matches()
    c_star_matches = constant_matches.get('C*', [])
    test1 = len(c_star_matches) >= 5
    results['tests'].append({
        'name': 'C* Matches Found',
        'passed': test1,
        'details': f"Found {len(c_star_matches)} C* matches (target: ≥5)"
    })
    results['passed' if test1 else 'failed'] += 1
    
    # Test 2: 4.0 plasticity matches
    plasticity_matches = hunter.find_plasticity_patterns()
    four_matches = [m for m in plasticity_matches if abs(m['target_value'] - 4.0) < 0.01]
    test2 = len(four_matches) >= 3
    results['tests'].append({
        'name': '4.0 Plasticity Matches',
        'passed': test2,
        'details': f"Found {len(four_matches)} exact 4.0 matches (target: ≥3)"
    })
    results['passed' if test2 else 'failed'] += 1
    
    # Test 3: Golden ratio matches
    golden_matches = hunter.find_golden_ratios()
    test3 = len(golden_matches) >= 3
    results['tests'].append({
        'name': 'Golden Ratio Matches',
        'passed': test3,
        'details': f"Found {len(golden_matches)} φ matches (target: ≥3)"
    })
    results['passed' if test3 else 'failed'] += 1
    
    # Test 4: Ratio clustering
    clusters = hunter.find_ratio_clusters()
    test4 = len(clusters) >= 5
    results['tests'].append({
        'name': 'Ratio Clustering',
        'passed': test4,
        'details': f"Found {len(clusters)} clusters (target: ≥5)"
    })
    results['passed' if test4 else 'failed'] += 1
    
    # Test 5: Earth-Moon-Sun analysis
    ems = hunter.analyze_earth_moon_sun()
    test5 = len(ems) == 6
    results['tests'].append({
        'name': 'Earth-Moon-Sun Analysis',
        'passed': test5,
        'details': f"Analyzed {len(ems)} ratios (target: 6)"
    })
    results['passed' if test5 else 'failed'] += 1
    
    # Test 6: Statistical significance
    stats = hunter.statistical_significance(plasticity_matches)
    test6 = stats['significant']
    results['tests'].append({
        'name': 'Statistical Significance',
        'passed': test6,
        'details': f"Mean error: {stats['mean_error']:.2f}%, Count: {stats['count']}"
    })
    results['passed' if test6 else 'failed'] += 1
    
    # Test 7: Multiple constant types
    test7 = len(constant_matches) >= 5
    results['tests'].append({
        'name': 'Multiple Constant Types',
        'passed': test7,
        'details': f"Found {len(constant_matches)} different constants (target: ≥5)"
    })
    results['passed' if test7 else 'failed'] += 1
    
    # Test 8: Dimensional jump analysis
    jump_analysis = hunter.analyze_dimensional_jumps()
    test8 = len(jump_analysis) >= 3
    results['tests'].append({
        'name': 'Dimensional Jump Analysis',
        'passed': test8,
        'details': f"Analyzed {len(jump_analysis)} jump types (target: ≥3)"
    })
    results['passed' if test8 else 'failed'] += 1
    
    results['pass_rate'] = results['passed'] / results['total']
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("PROJECT BUSHMAN SUITE 7: THE OBSERVATORY")
    print("Program 2: Ratio Hunter")
    print("=" * 80)
    print()
    
    # Run tests
    print("Running test suite...")
    test_results = run_tests()
    
    print(f"\nTest Results: {test_results['passed']}/{test_results['total']} passed")
    print(f"Pass Rate: {test_results['pass_rate']*100:.1f}%")
    print()
    
    for test in test_results['tests']:
        status = "✓ PASS" if test['passed'] else "✗ FAIL"
        print(f"{status}: {test['name']}")
        print(f"  {test['details']}")
    
    # Generate full report
    print("\n" + "=" * 80)
    print("Generating complete ratio analysis...")
    hunter = RatioHunter()
    report = hunter.generate_report()
    
    # Save report
    with open('ratio_hunter_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to ratio_hunter_results.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary = report['summary']
    print(f"Total Constants Found: {summary['total_constants_found']}")
    print(f"Total Matches: {summary['total_matches']}")
    print(f"Plasticity Matches: {summary['plasticity_matches_count']}")
    print(f"Golden Ratio Matches: {summary['golden_matches_count']}")
    print(f"Ratio Clusters: {summary['clusters_found']}")
    print()
    
    print("Top C* Matches:")
    if 'C*' in report['constant_matches']:
        for match in report['constant_matches']['C*'][:5]:
            print(f"  {match['object1']} / {match['object2']}")
            print(f"    {match['ratio_type']}: {match['ratio_value']:.6f} ≈ C* ({C_STAR:.6f})")
            print(f"    Error: {match['error_percent']:.2f}%")
    print()
    
    print("Top Plasticity Matches (4-7-5 pattern):")
    for match in report['plasticity_matches'][:5]:
        print(f"  {match['object1']} / {match['object2']}")
        print(f"    Ratio: {match['radius_ratio']:.6f} ≈ {match['target']} ({match['target_value']:.6f})")
        print(f"    Error: {match['error_percent']:.2f}%")
    print()
    
    print("Earth-Moon-Sun System Analysis:")
    for ratio_name, data in report['earth_moon_sun_analysis'].items():
        print(f"  {ratio_name}: {data['value']:.6f}")
        print(f"    Best match: {data['best_match']} ({data['best_match_value']:.6f})")
        print(f"    Error: {data['error_percent']:.2f}%")
    print()
    
    print("Ratio Clusters (peaks in distribution):")
    for cluster in report['ratio_clusters'][:5]:
        print(f"  Cluster at ratio ≈ {cluster['average_ratio']:.2f}")
        print(f"    Count: {cluster['count']} pairs")
        print(f"    Example: {cluster['examples'][0]['object1']} / {cluster['examples'][0]['object2']}")
    
    print("\n" + "=" * 80)
    print("PLASTICITY RULES. RATIOS REVEAL REALITY.")
    print("=" * 80)
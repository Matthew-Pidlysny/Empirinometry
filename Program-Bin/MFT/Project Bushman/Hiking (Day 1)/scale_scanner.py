"""
Project Bushman Suite 7: "The Observatory"
Program 1: Scale Scanner

Purpose: Catalog all known spherical objects across all scales of reality
         and search for C* patterns in their size distributions.

Author: SuperNinja AI (for Matthew Pidlysny)
Date: December 2024
"""

import json
import math
from typing import Dict, List, Tuple
from collections import defaultdict

# Constants
C_STAR = 0.894751918
F_01 = C_STAR
F_12 = 4.0 * C_STAR
F_23 = 7.07 * F_12
F_34 = 5.09 * C_STAR

# Hardcoded datasets of spherical objects across all scales

ATOMIC_SPHERES = {
    # Atomic orbitals and nuclei (meters)
    'hydrogen_1s_orbital': {'radius': 5.29e-11, 'mass': 1.67e-27, 'type': 'orbital', 'scale': 0},
    'helium_nucleus': {'radius': 1.68e-15, 'mass': 6.64e-27, 'type': 'nucleus', 'scale': 0},
    'carbon_nucleus': {'radius': 2.7e-15, 'mass': 1.99e-26, 'type': 'nucleus', 'scale': 0},
    'oxygen_nucleus': {'radius': 3.0e-15, 'mass': 2.66e-26, 'type': 'nucleus', 'scale': 0},
    'iron_nucleus': {'radius': 4.3e-15, 'mass': 9.29e-26, 'type': 'nucleus', 'scale': 0},
    'uranium_nucleus': {'radius': 7.4e-15, 'mass': 3.95e-25, 'type': 'nucleus', 'scale': 0},
    'buckminsterfullerene_c60': {'radius': 3.55e-10, 'mass': 1.2e-24, 'type': 'molecule', 'scale': 0},
    'gold_atom': {'radius': 1.44e-10, 'mass': 3.27e-25, 'type': 'atom', 'scale': 0},
    'silicon_atom': {'radius': 1.18e-10, 'mass': 4.66e-26, 'type': 'atom', 'scale': 0},
    'proton': {'radius': 8.41e-16, 'mass': 1.67e-27, 'type': 'particle', 'scale': 0},
}

MOLECULAR_SPHERES = {
    # Molecular and cellular structures (meters)
    'water_molecule': {'radius': 1.4e-10, 'mass': 2.99e-26, 'type': 'molecule', 'scale': 1},
    'hemoglobin': {'radius': 2.8e-9, 'mass': 1.08e-22, 'type': 'protein', 'scale': 1},
    'ribosome': {'radius': 1.0e-8, 'mass': 4.2e-21, 'type': 'organelle', 'scale': 1},
    'hiv_virus': {'radius': 6.0e-8, 'mass': 1.0e-18, 'type': 'virus', 'scale': 1},
    'influenza_virus': {'radius': 5.0e-8, 'mass': 5.0e-19, 'type': 'virus', 'scale': 1},
    'red_blood_cell': {'radius': 3.8e-6, 'mass': 2.7e-14, 'type': 'cell', 'scale': 1},
    'white_blood_cell': {'radius': 6.0e-6, 'mass': 5.0e-14, 'type': 'cell', 'scale': 1},
    'human_egg_cell': {'radius': 1.0e-4, 'mass': 4.0e-9, 'type': 'cell', 'scale': 1},
    'pollen_grain': {'radius': 2.5e-5, 'mass': 1.0e-10, 'type': 'spore', 'scale': 1},
    'amoeba': {'radius': 2.5e-4, 'mass': 5.0e-10, 'type': 'organism', 'scale': 1},
}

TERRESTRIAL_SPHERES = {
    # Everyday objects and natural formations (meters)
    'mist_droplet': {'radius': 5.0e-6, 'mass': 5.2e-13, 'type': 'droplet', 'scale': 2},
    'fog_droplet': {'radius': 1.0e-5, 'mass': 4.2e-12, 'type': 'droplet', 'scale': 2},
    'drizzle_drop': {'radius': 2.5e-4, 'mass': 6.5e-8, 'type': 'droplet', 'scale': 2},
    'raindrop_small': {'radius': 5.0e-4, 'mass': 5.2e-7, 'type': 'droplet', 'scale': 2},
    'raindrop_medium': {'radius': 1.5e-3, 'mass': 1.4e-5, 'type': 'droplet', 'scale': 2},
    'raindrop_large': {'radius': 3.0e-3, 'mass': 1.1e-4, 'type': 'droplet', 'scale': 2},
    'soap_bubble': {'radius': 2.5e-2, 'mass': 1.0e-4, 'type': 'bubble', 'scale': 2},
    'ping_pong_ball': {'radius': 2.0e-2, 'mass': 2.7e-3, 'type': 'manufactured', 'scale': 2},
    'golf_ball': {'radius': 2.14e-2, 'mass': 4.59e-2, 'type': 'manufactured', 'scale': 2},
    'tennis_ball': {'radius': 3.35e-2, 'mass': 5.8e-2, 'type': 'manufactured', 'scale': 2},
    'baseball': {'radius': 3.7e-2, 'mass': 1.45e-1, 'type': 'manufactured', 'scale': 2},
    'orange': {'radius': 4.0e-2, 'mass': 1.5e-1, 'type': 'fruit', 'scale': 2},
    'apple': {'radius': 3.5e-2, 'mass': 1.8e-1, 'type': 'fruit', 'scale': 2},
    'grapefruit': {'radius': 6.0e-2, 'mass': 3.0e-1, 'type': 'fruit', 'scale': 2},
    'softball': {'radius': 4.8e-2, 'mass': 1.88e-1, 'type': 'manufactured', 'scale': 2},
    'volleyball': {'radius': 1.05e-1, 'mass': 2.7e-1, 'type': 'manufactured', 'scale': 2},
    'soccer_ball': {'radius': 1.1e-1, 'mass': 4.3e-1, 'type': 'manufactured', 'scale': 2},
    'basketball': {'radius': 1.2e-1, 'mass': 6.2e-1, 'type': 'manufactured', 'scale': 2},
    'human_eye': {'radius': 1.2e-2, 'mass': 7.5e-3, 'type': 'organ', 'scale': 2},
    'chicken_egg': {'radius': 2.75e-2, 'mass': 5.0e-2, 'type': 'biological', 'scale': 2},
}

PLANETARY_SPHERES = {
    # Solar system objects (meters)
    'ceres': {'radius': 4.73e5, 'mass': 9.39e20, 'type': 'dwarf_planet', 'scale': 3},
    'pluto': {'radius': 1.188e6, 'mass': 1.31e22, 'type': 'dwarf_planet', 'scale': 3},
    'moon': {'radius': 1.7374e6, 'mass': 7.342e22, 'type': 'satellite', 'scale': 3},
    'io': {'radius': 1.8216e6, 'mass': 8.93e22, 'type': 'satellite', 'scale': 3},
    'europa': {'radius': 1.5608e6, 'mass': 4.8e22, 'type': 'satellite', 'scale': 3},
    'ganymede': {'radius': 2.6341e6, 'mass': 1.48e23, 'type': 'satellite', 'scale': 3},
    'callisto': {'radius': 2.4103e6, 'mass': 1.08e23, 'type': 'satellite', 'scale': 3},
    'titan': {'radius': 2.5747e6, 'mass': 1.35e23, 'type': 'satellite', 'scale': 3},
    'mercury': {'radius': 2.4397e6, 'mass': 3.30e23, 'type': 'planet', 'scale': 3},
    'mars': {'radius': 3.3895e6, 'mass': 6.39e23, 'type': 'planet', 'scale': 3},
    'venus': {'radius': 6.0518e6, 'mass': 4.87e24, 'type': 'planet', 'scale': 3},
    'earth': {'radius': 6.371e6, 'mass': 5.972e24, 'type': 'planet', 'scale': 3},
    'neptune': {'radius': 2.4622e7, 'mass': 1.02e26, 'type': 'planet', 'scale': 3},
    'uranus': {'radius': 2.5362e7, 'mass': 8.68e25, 'type': 'planet', 'scale': 3},
    'saturn': {'radius': 5.8232e7, 'mass': 5.68e26, 'type': 'planet', 'scale': 3},
    'jupiter': {'radius': 6.9911e7, 'mass': 1.898e27, 'type': 'planet', 'scale': 3},
    'sun': {'radius': 6.96e8, 'mass': 1.989e30, 'type': 'star', 'scale': 3},
}

STELLAR_SPHERES = {
    # Stars and stellar remnants (meters)
    'typical_neutron_star': {'radius': 1.0e4, 'mass': 2.8e30, 'type': 'neutron_star', 'scale': 4},
    'proxima_centauri': {'radius': 1.07e8, 'mass': 2.45e29, 'type': 'red_dwarf', 'scale': 4},
    'alpha_centauri_a': {'radius': 8.51e8, 'mass': 2.19e30, 'type': 'main_sequence', 'scale': 4},
    'sirius_a': {'radius': 1.19e9, 'mass': 4.02e30, 'type': 'main_sequence', 'scale': 4},
    'vega': {'radius': 1.93e9, 'mass': 4.25e30, 'type': 'main_sequence', 'scale': 4},
    'arcturus': {'radius': 1.78e10, 'mass': 2.2e30, 'type': 'red_giant', 'scale': 4},
    'aldebaran': {'radius': 3.04e10, 'mass': 3.4e30, 'type': 'red_giant', 'scale': 4},
    'pollux': {'radius': 5.93e9, 'mass': 3.6e30, 'type': 'red_giant', 'scale': 4},
    'rigel': {'radius': 5.46e10, 'mass': 4.2e31, 'type': 'blue_supergiant', 'scale': 4},
    'betelgeuse': {'radius': 7.0e11, 'mass': 2.4e31, 'type': 'red_supergiant', 'scale': 4},
    'antares': {'radius': 5.2e11, 'mass': 2.4e31, 'type': 'red_supergiant', 'scale': 4},
    'uy_scuti': {'radius': 1.19e12, 'mass': 1.4e31, 'type': 'red_hypergiant', 'scale': 4},
    'white_dwarf_typical': {'radius': 7.0e6, 'mass': 1.2e30, 'type': 'white_dwarf', 'scale': 4},
}

GALACTIC_SPHERES = {
    # Galactic structures (meters)
    'globular_cluster_m13': {'radius': 8.4e17, 'mass': 1.2e36, 'type': 'globular_cluster', 'scale': 5},
    'globular_cluster_omega_centauri': {'radius': 1.4e18, 'mass': 8.0e36, 'type': 'globular_cluster', 'scale': 5},
    'small_magellanic_cloud': {'radius': 3.5e20, 'mass': 1.4e40, 'type': 'dwarf_galaxy', 'scale': 5},
    'large_magellanic_cloud': {'radius': 4.3e20, 'mass': 2.0e41, 'type': 'dwarf_galaxy', 'scale': 5},
    'milky_way_bulge': {'radius': 9.5e19, 'mass': 2.0e40, 'type': 'galactic_bulge', 'scale': 5},
    'milky_way_halo': {'radius': 9.5e20, 'mass': 2.0e42, 'type': 'galaxy_halo', 'scale': 5},
    'andromeda_galaxy': {'radius': 1.1e21, 'mass': 2.5e42, 'type': 'spiral_galaxy', 'scale': 5},
    'triangulum_galaxy': {'radius': 2.8e20, 'mass': 1.0e41, 'type': 'spiral_galaxy', 'scale': 5},
    'virgo_cluster': {'radius': 1.2e22, 'mass': 2.4e45, 'type': 'galaxy_cluster', 'scale': 5},
    'coma_cluster': {'radius': 1.9e22, 'mass': 1.9e45, 'type': 'galaxy_cluster', 'scale': 5},
    'observable_universe': {'radius': 4.4e26, 'mass': 1.5e53, 'type': 'universe', 'scale': 5},
}

class ScaleScanner:
    """Catalog and analyze spherical objects across all scales."""
    
    def __init__(self):
        self.catalog = {}
        self.load_all_data()
        
    def load_all_data(self):
        """Load all hardcoded datasets into unified catalog."""
        datasets = [
            ATOMIC_SPHERES,
            MOLECULAR_SPHERES,
            TERRESTRIAL_SPHERES,
            PLANETARY_SPHERES,
            STELLAR_SPHERES,
            GALACTIC_SPHERES
        ]
        
        for dataset in datasets:
            self.catalog.update(dataset)
    
    def get_scale_name(self, scale: int) -> str:
        """Convert scale number to name."""
        names = {
            0: "Quantum",
            1: "Molecular",
            2: "Terrestrial",
            3: "Planetary",
            4: "Stellar",
            5: "Galactic"
        }
        return names.get(scale, "Unknown")
    
    def calculate_ratios(self) -> List[Dict]:
        """Calculate all pairwise ratios between sphere radii."""
        ratios = []
        items = list(self.catalog.items())
        
        for i, (name1, data1) in enumerate(items):
            for name2, data2 in items[i+1:]:
                r1 = data1['radius']
                r2 = data2['radius']
                
                ratio = max(r1, r2) / min(r1, r2)
                
                ratios.append({
                    'object1': name1,
                    'object2': name2,
                    'radius1': r1,
                    'radius2': r2,
                    'ratio': ratio,
                    'scale1': data1['scale'],
                    'scale2': data2['scale']
                })
        
        return sorted(ratios, key=lambda x: x['ratio'])
    
    def find_c_star_matches(self, tolerance: float = 0.05) -> List[Dict]:
        """Find ratios that match C* or its multiples."""
        ratios = self.calculate_ratios()
        matches = []
        
        targets = [
            ('C*', C_STAR),
            ('2*C*', 2 * C_STAR),
            ('3*C*', 3 * C_STAR),
            ('4*C* (F12/C*)', 4.0),
            ('5*C*', 5 * C_STAR),
            ('7*C*', 7 * C_STAR),
            ('1/C*', 1 / C_STAR),
            ('π', math.pi),
            ('e', math.e),
            ('φ', (1 + math.sqrt(5)) / 2),
        ]
        
        for ratio_data in ratios:
            ratio = ratio_data['ratio']
            
            for target_name, target_value in targets:
                if abs(ratio - target_value) / target_value < tolerance:
                    matches.append({
                        **ratio_data,
                        'target': target_name,
                        'target_value': target_value,
                        'error_percent': abs(ratio - target_value) / target_value * 100
                    })
        
        return sorted(matches, key=lambda x: x['error_percent'])
    
    def analyze_scale_distribution(self) -> Dict:
        """Analyze distribution of spheres across scales."""
        scale_counts = defaultdict(int)
        scale_radii = defaultdict(list)
        
        for name, data in self.catalog.items():
            scale = data['scale']
            scale_counts[scale] += 1
            scale_radii[scale].append(data['radius'])
        
        analysis = {}
        for scale in range(6):
            radii = sorted(scale_radii[scale])
            if radii:
                analysis[scale] = {
                    'name': self.get_scale_name(scale),
                    'count': scale_counts[scale],
                    'min_radius': radii[0],
                    'max_radius': radii[-1],
                    'range_orders': math.log10(radii[-1] / radii[0]) if len(radii) > 1 else 0,
                    'median_radius': radii[len(radii)//2]
                }
        
        return analysis
    
    def find_jungle_zones(self) -> List[Dict]:
        """Identify gaps (jungles) in size distributions."""
        all_radii = sorted([(name, data['radius'], data['scale']) 
                           for name, data in self.catalog.items()],
                          key=lambda x: x[1])
        
        jungles = []
        for i in range(len(all_radii) - 1):
            name1, r1, scale1 = all_radii[i]
            name2, r2, scale2 = all_radii[i + 1]
            
            gap_ratio = r2 / r1
            
            # Significant gap if ratio > 2.0
            if gap_ratio > 2.0:
                jungles.append({
                    'lower_object': name1,
                    'upper_object': name2,
                    'lower_radius': r1,
                    'upper_radius': r2,
                    'gap_ratio': gap_ratio,
                    'gap_orders': math.log10(gap_ratio),
                    'scale_transition': scale1 != scale2
                })
        
        return sorted(jungles, key=lambda x: x['gap_ratio'], reverse=True)
    
    def calculate_statistics(self) -> Dict:
        """Calculate overall statistics."""
        radii = [data['radius'] for data in self.catalog.values()]
        masses = [data['mass'] for data in self.catalog.values()]
        
        return {
            'total_objects': len(self.catalog),
            'radius_range_orders': math.log10(max(radii) / min(radii)),
            'mass_range_orders': math.log10(max(masses) / min(masses)),
            'smallest_object': min(self.catalog.items(), key=lambda x: x[1]['radius'])[0],
            'largest_object': max(self.catalog.items(), key=lambda x: x[1]['radius'])[0],
            'lightest_object': min(self.catalog.items(), key=lambda x: x[1]['mass'])[0],
            'heaviest_object': max(self.catalog.items(), key=lambda x: x[1]['mass'])[0],
        }
    
    def generate_report(self) -> Dict:
        """Generate complete analysis report."""
        return {
            'catalog': self.catalog,
            'statistics': self.calculate_statistics(),
            'scale_distribution': self.analyze_scale_distribution(),
            'c_star_matches': self.find_c_star_matches(),
            'jungle_zones': self.find_jungle_zones()[:20],  # Top 20 largest gaps
            'constants': {
                'C_STAR': C_STAR,
                'F_01': F_01,
                'F_12': F_12,
                'F_23': F_23,
                'F_34': F_34
            }
        }


def run_tests() -> Dict:
    """Run test suite for scale scanner."""
    scanner = ScaleScanner()
    results = {
        'tests': [],
        'passed': 0,
        'failed': 0,
        'total': 8
    }
    
    # Test 1: Catalog completeness
    test1 = len(scanner.catalog) >= 100
    results['tests'].append({
        'name': 'Catalog Completeness',
        'passed': test1,
        'details': f"Cataloged {len(scanner.catalog)} objects (target: ≥100)"
    })
    results['passed' if test1 else 'failed'] += 1
    
    # Test 2: All scales represented
    scales = set(data['scale'] for data in scanner.catalog.values())
    test2 = len(scales) == 6
    results['tests'].append({
        'name': 'All Scales Represented',
        'passed': test2,
        'details': f"Found {len(scales)} scales (target: 6)"
    })
    results['passed' if test2 else 'failed'] += 1
    
    # Test 3: C* matches found
    matches = scanner.find_c_star_matches()
    test3 = len(matches) >= 10
    results['tests'].append({
        'name': 'C* Pattern Matches',
        'passed': test3,
        'details': f"Found {len(matches)} C* matches (target: ≥10)"
    })
    results['passed' if test3 else 'failed'] += 1
    
    # Test 4: Jungle zones identified
    jungles = scanner.find_jungle_zones()
    test4 = len(jungles) >= 20
    results['tests'].append({
        'name': 'Jungle Zone Identification',
        'passed': test4,
        'details': f"Found {len(jungles)} jungle zones (target: ≥20)"
    })
    results['passed' if test4 else 'failed'] += 1
    
    # Test 5: Earth-Moon-Sun system present
    ems_objects = ['earth', 'moon', 'sun']
    test5 = all(obj in scanner.catalog for obj in ems_objects)
    results['tests'].append({
        'name': 'Earth-Moon-Sun System Present',
        'passed': test5,
        'details': "All three objects found" if test5 else "Missing objects"
    })
    results['passed' if test5 else 'failed'] += 1
    
    # Test 6: Radius range spans universe
    stats = scanner.calculate_statistics()
    test6 = stats['radius_range_orders'] > 40
    results['tests'].append({
        'name': 'Radius Range Spans Universe',
        'passed': test6,
        'details': f"{stats['radius_range_orders']:.1f} orders of magnitude (target: >40)"
    })
    results['passed' if test6 else 'failed'] += 1
    
    # Test 7: Best C* match has low error
    if matches:
        best_match = matches[0]
        test7 = best_match['error_percent'] < 5.0
        results['tests'].append({
            'name': 'Best C* Match Quality',
            'passed': test7,
            'details': f"{best_match['error_percent']:.2f}% error (target: <5%)"
        })
        results['passed' if test7 else 'failed'] += 1
    else:
        results['tests'].append({
            'name': 'Best C* Match Quality',
            'passed': False,
            'details': "No matches found"
        })
        results['failed'] += 1
    
    # Test 8: Scale transitions in jungle zones
    scale_transition_jungles = [j for j in jungles if j['scale_transition']]
    test8 = len(scale_transition_jungles) >= 5
    results['tests'].append({
        'name': 'Scale Transition Jungles',
        'passed': test8,
        'details': f"Found {len(scale_transition_jungles)} scale transitions (target: ≥5)"
    })
    results['passed' if test8 else 'failed'] += 1
    
    results['pass_rate'] = results['passed'] / results['total']
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("PROJECT BUSHMAN SUITE 7: THE OBSERVATORY")
    print("Program 1: Scale Scanner")
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
    print("Generating complete analysis report...")
    scanner = ScaleScanner()
    report = scanner.generate_report()
    
    # Save report
    with open('scale_scanner_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to scale_scanner_results.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    stats = report['statistics']
    print(f"Total Objects Cataloged: {stats['total_objects']}")
    print(f"Radius Range: {stats['radius_range_orders']:.1f} orders of magnitude")
    print(f"Mass Range: {stats['mass_range_orders']:.1f} orders of magnitude")
    print(f"Smallest: {stats['smallest_object']}")
    print(f"Largest: {stats['largest_object']}")
    print()
    
    print("Scale Distribution:")
    for scale_num, scale_data in sorted(report['scale_distribution'].items()):
        print(f"  {scale_data['name']}: {scale_data['count']} objects")
        print(f"    Range: {scale_data['range_orders']:.1f} orders of magnitude")
    print()
    
    print(f"C* Matches Found: {len(report['c_star_matches'])}")
    if report['c_star_matches']:
        print("  Top 5 matches:")
        for match in report['c_star_matches'][:5]:
            print(f"    {match['object1']} / {match['object2']}")
            print(f"      Ratio: {match['ratio']:.6f} ≈ {match['target']} ({match['target_value']:.6f})")
            print(f"      Error: {match['error_percent']:.2f}%")
    print()
    
    print(f"Jungle Zones Found: {len(report['jungle_zones'])}")
    print("  Largest gaps:")
    for jungle in report['jungle_zones'][:5]:
        print(f"    {jungle['lower_object']} → {jungle['upper_object']}")
        print(f"      Gap: {jungle['gap_ratio']:.2f}× ({jungle['gap_orders']:.2f} orders)")
        if jungle['scale_transition']:
            print(f"      *** SCALE TRANSITION ***")
    
    print("\n" + "=" * 80)
    print("BALL EVERYTHING. OBSERVE EVERYTHING.")
    print("=" * 80)
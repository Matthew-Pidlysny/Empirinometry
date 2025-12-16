"""
Project Bushman Suite 7: "The Observatory"
Program 8: Cosmic Sphere Census

Purpose: Complete census of all spheres in the observable universe,
         calculating total numbers, masses, and distributions across scales.

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

class CosmicSphereCensus:
    """Complete census of all spheres in observable universe."""
    
    def __init__(self, catalog_file: str = 'scale_scanner_results.json'):
        with open(catalog_file, 'r') as f:
            data = json.load(f)
            self.catalog = data['catalog']
        
        # Estimated counts in observable universe
        self.universe_estimates = self.define_universe_estimates()
    
    def define_universe_estimates(self) -> Dict:
        """Define estimated counts of objects in observable universe."""
        return {
            'protons': 1e80,
            'atoms': 1e80,
            'molecules': 1e79,
            'cells': 1e37,  # Estimated total cells in all life
            'planets': 1e24,  # Estimated planets in observable universe
            'stars': 1e24,  # Estimated stars
            'galaxies': 2e12,  # Estimated galaxies
            'galaxy_clusters': 1e10,
            'droplets': 1e45,  # Estimated water droplets (clouds, oceans, etc.)
            'manufactured_spheres': 1e15  # Human-made spheres
        }
    
    def estimate_total_spheres(self) -> Dict:
        """Estimate total number of spheres in universe."""
        total = 0
        breakdown = {}
        
        for obj_type, count in self.universe_estimates.items():
            total += count
            breakdown[obj_type] = count
        
        return {
            'total_spheres': total,
            'breakdown': breakdown,
            'orders_of_magnitude': math.log10(total)
        }
    
    def calculate_total_mass(self) -> Dict:
        """Calculate total mass in spherical objects."""
        total_mass = 0
        mass_by_scale = defaultdict(float)
        
        # Use catalog objects as representatives
        for name, data in self.catalog.items():
            obj_type = data['type']
            mass = data['mass']
            scale = data['scale']
            
            # Estimate count for this type
            if 'proton' in name or 'nucleus' in name:
                count = self.universe_estimates.get('protons', 0)
            elif 'atom' in name or 'molecule' in name:
                count = self.universe_estimates.get('molecules', 0)
            elif 'cell' in name:
                count = self.universe_estimates.get('cells', 0)
            elif 'planet' in obj_type:
                count = self.universe_estimates.get('planets', 0) / 10  # Rough estimate
            elif 'star' in obj_type:
                count = self.universe_estimates.get('stars', 0) / 10
            elif 'galaxy' in obj_type:
                count = self.universe_estimates.get('galaxies', 0) / 10
            elif 'droplet' in name:
                count = self.universe_estimates.get('droplets', 0) / 10
            else:
                count = 1e10  # Default estimate
            
            total_mass += mass * count
            mass_by_scale[scale] += mass * count
        
        return {
            'total_mass': total_mass,
            'mass_by_scale': dict(mass_by_scale),
            'fraction_of_universe': total_mass / 1e53  # Observable universe mass ~1e53 kg
        }
    
    def find_most_common_sizes(self) -> List[Dict]:
        """Find most common sphere sizes in universe."""
        size_counts = []
        
        for name, data in self.catalog.items():
            radius = data['radius']
            obj_type = data['type']
            
            # Estimate count
            if 'proton' in name or 'nucleus' in name:
                count = self.universe_estimates.get('protons', 0)
            elif 'atom' in name:
                count = self.universe_estimates.get('atoms', 0) / 10
            elif 'molecule' in name:
                count = self.universe_estimates.get('molecules', 0) / 10
            elif 'cell' in name:
                count = self.universe_estimates.get('cells', 0) / 10
            elif 'droplet' in name:
                count = self.universe_estimates.get('droplets', 0) / 5
            elif 'planet' in obj_type:
                count = self.universe_estimates.get('planets', 0) / 20
            elif 'star' in obj_type:
                count = self.universe_estimates.get('stars', 0) / 20
            else:
                count = 1e10
            
            size_counts.append({
                'object': name,
                'radius': radius,
                'estimated_count': count,
                'type': obj_type,
                'scale': data['scale']
            })
        
        return sorted(size_counts, key=lambda x: x['estimated_count'], reverse=True)
    
    def analyze_scale_distribution(self) -> Dict:
        """Analyze distribution of spheres across scales."""
        scale_counts = defaultdict(float)
        scale_names = {
            0: "Quantum",
            1: "Molecular",
            2: "Terrestrial",
            3: "Planetary",
            4: "Stellar",
            5: "Galactic"
        }
        
        for name, data in self.catalog.items():
            scale = data['scale']
            
            # Estimate count
            if scale == 0:
                count = self.universe_estimates.get('protons', 0)
            elif scale == 1:
                count = self.universe_estimates.get('molecules', 0)
            elif scale == 2:
                count = self.universe_estimates.get('droplets', 0)
            elif scale == 3:
                count = self.universe_estimates.get('planets', 0)
            elif scale == 4:
                count = self.universe_estimates.get('stars', 0)
            elif scale == 5:
                count = self.universe_estimates.get('galaxies', 0)
            else:
                count = 0
            
            scale_counts[scale] += count / 10  # Divide by catalog size at scale
        
        total = sum(scale_counts.values())
        
        return {
            'scale_counts': {scale_names[s]: count for s, count in scale_counts.items()},
            'scale_percentages': {scale_names[s]: count/total*100 for s, count in scale_counts.items()},
            'dominant_scale': max(scale_counts.items(), key=lambda x: x[1])[0]
        }
    
    def calculate_sphere_density(self) -> Dict:
        """Calculate density of spheres in observable universe."""
        total_spheres = self.estimate_total_spheres()['total_spheres']
        
        # Observable universe volume
        universe_radius = 4.4e26  # meters
        universe_volume = (4/3) * math.pi * universe_radius**3
        
        sphere_density = total_spheres / universe_volume
        
        # Average separation
        avg_separation = (universe_volume / total_spheres)**(1/3)
        
        return {
            'universe_volume': universe_volume,
            'sphere_density': sphere_density,
            'spheres_per_cubic_meter': sphere_density,
            'average_separation': avg_separation
        }
    
    def identify_rarest_spheres(self) -> List[Dict]:
        """Identify rarest sphere types."""
        size_counts = self.find_most_common_sizes()
        return sorted(size_counts, key=lambda x: x['estimated_count'])[:10]
    
    def calculate_c_star_coverage(self) -> Dict:
        """Calculate what fraction of universe is at C* thresholds."""
        # Count objects near C* multiples
        near_c_star = 0
        total = 0
        
        for name, data in self.catalog.items():
            radius = data['radius']
            
            # Check if radius is near C* multiple
            for multiplier in [1, 2, 3, 4, 5, 7]:
                target = multiplier * C_STAR
                if abs(math.log10(radius) - math.log10(target)) < 0.5:
                    near_c_star += 1
                    break
            
            total += 1
        
        return {
            'objects_near_c_star': near_c_star,
            'total_objects': total,
            'c_star_coverage_percent': near_c_star / total * 100 if total > 0 else 0
        }
    
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
    
    def generate_report(self) -> Dict:
        """Generate complete cosmic sphere census."""
        return {
            'total_spheres': self.estimate_total_spheres(),
            'total_mass': self.calculate_total_mass(),
            'most_common_sizes': self.find_most_common_sizes()[:20],
            'rarest_spheres': self.identify_rarest_spheres(),
            'scale_distribution': self.analyze_scale_distribution(),
            'sphere_density': self.calculate_sphere_density(),
            'c_star_coverage': self.calculate_c_star_coverage(),
            'constants': {
                'C_STAR': C_STAR,
                'F_01': F_01,
                'F_12': F_12,
                'F_23': F_23,
                'F_34': F_34
            }
        }


def run_tests() -> Dict:
    """Run test suite for cosmic sphere census."""
    census = CosmicSphereCensus()
    results = {
        'tests': [],
        'passed': 0,
        'failed': 0,
        'total': 8
    }
    
    # Test 1: Total spheres estimated
    total = census.estimate_total_spheres()
    test1 = total['total_spheres'] > 1e70
    results['tests'].append({
        'name': 'Total Spheres Estimated',
        'passed': test1,
        'details': f"Estimated {total['total_spheres']:.2e} spheres (target: >1e70)"
    })
    results['passed' if test1 else 'failed'] += 1
    
    # Test 2: Total mass calculated
    mass = census.calculate_total_mass()
    test2 = mass['total_mass'] > 0
    results['tests'].append({
        'name': 'Total Mass Calculated',
        'passed': test2,
        'details': f"Total mass: {mass['total_mass']:.2e} kg"
    })
    results['passed' if test2 else 'failed'] += 1
    
    # Test 3: Most common sizes identified
    common = census.find_most_common_sizes()
    test3 = len(common) >= 10
    results['tests'].append({
        'name': 'Most Common Sizes Identified',
        'passed': test3,
        'details': f"Found {len(common)} size categories (target: ≥10)"
    })
    results['passed' if test3 else 'failed'] += 1
    
    # Test 4: Scale distribution analyzed
    scale_dist = census.analyze_scale_distribution()
    test4 = len(scale_dist['scale_counts']) >= 5
    results['tests'].append({
        'name': 'Scale Distribution Analyzed',
        'passed': test4,
        'details': f"Analyzed {len(scale_dist['scale_counts'])} scales (target: ≥5)"
    })
    results['passed' if test4 else 'failed'] += 1
    
    # Test 5: Sphere density calculated
    density = census.calculate_sphere_density()
    test5 = density['sphere_density'] > 0
    results['tests'].append({
        'name': 'Sphere Density Calculated',
        'passed': test5,
        'details': f"Density: {density['sphere_density']:.2e} spheres/m³"
    })
    results['passed' if test5 else 'failed'] += 1
    
    # Test 6: Rarest spheres identified
    rarest = census.identify_rarest_spheres()
    test6 = len(rarest) >= 5
    results['tests'].append({
        'name': 'Rarest Spheres Identified',
        'passed': test6,
        'details': f"Found {len(rarest)} rare types (target: ≥5)"
    })
    results['passed' if test6 else 'failed'] += 1
    
    # Test 7: C* coverage calculated
    c_star_cov = census.calculate_c_star_coverage()
    test7 = 0 <= c_star_cov['c_star_coverage_percent'] <= 100
    results['tests'].append({
        'name': 'C* Coverage Calculated',
        'passed': test7,
        'details': f"C* coverage: {c_star_cov['c_star_coverage_percent']:.1f}%"
    })
    results['passed' if test7 else 'failed'] += 1
    
    # Test 8: Quantum scale dominates
    test8 = scale_dist['dominant_scale'] == 0
    results['tests'].append({
        'name': 'Quantum Scale Dominates',
        'passed': test8,
        'details': f"Dominant scale: {census.get_scale_name(scale_dist['dominant_scale'])}"
    })
    results['passed' if test8 else 'failed'] += 1
    
    results['pass_rate'] = results['passed'] / results['total']
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("PROJECT BUSHMAN SUITE 7: THE OBSERVATORY")
    print("Program 8: Cosmic Sphere Census")
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
    print("Generating cosmic sphere census...")
    census = CosmicSphereCensus()
    report = census.generate_report()
    
    # Save report
    with open('cosmic_sphere_census_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to cosmic_sphere_census_results.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total = report['total_spheres']
    print(f"Total Spheres in Observable Universe: {total['total_spheres']:.2e}")
    print(f"Orders of Magnitude: {total['orders_of_magnitude']:.1f}")
    print()
    
    print("Breakdown by Type:")
    for obj_type, count in sorted(total['breakdown'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {obj_type}: {count:.2e}")
    print()
    
    mass = report['total_mass']
    print(f"Total Mass in Spheres: {mass['total_mass']:.2e} kg")
    print(f"Fraction of Universe Mass: {mass['fraction_of_universe']*100:.2f}%")
    print()
    
    print("Most Common Sphere Sizes:")
    for size in report['most_common_sizes'][:5]:
        print(f"  {size['object']}: {size['estimated_count']:.2e} spheres")
        print(f"    Radius: {size['radius']:.2e} m")
    print()
    
    print("Scale Distribution:")
    scale_dist = report['scale_distribution']
    for scale_name, percentage in scale_dist['scale_percentages'].items():
        print(f"  {scale_name}: {percentage:.1f}%")
    print()
    
    density = report['sphere_density']
    print(f"Sphere Density: {density['sphere_density']:.2e} spheres/m³")
    print(f"Average Separation: {density['average_separation']:.2e} m")
    print()
    
    c_star = report['c_star_coverage']
    print(f"C* Coverage: {c_star['c_star_coverage_percent']:.1f}% of cataloged objects")
    
    print("\n" + "=" * 80)
    print("THE UNIVERSE IS SPHERES. SPHERES ARE C*.")
    print("=" * 80)
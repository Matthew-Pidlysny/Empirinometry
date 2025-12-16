"""
Project Bushman Suite 7: "The Observatory"
Program 3: Jungle Mapper

Purpose: Identify and map "jungle zones" - the messy, unstable regions between
         stable sphere sizes where formation is difficult or impossible.

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

class JungleMapper:
    """Map jungle zones between stable sphere sizes."""
    
    def __init__(self, catalog_file: str = 'scale_scanner_results.json'):
        with open(catalog_file, 'r') as f:
            data = json.load(f)
            self.catalog = data['catalog']
    
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
    
    def find_all_gaps(self) -> List[Dict]:
        """Find all gaps in the size distribution."""
        # Sort all objects by radius
        sorted_objects = sorted(
            [(name, data['radius'], data['scale'], data['type']) 
             for name, data in self.catalog.items()],
            key=lambda x: x[1]
        )
        
        gaps = []
        for i in range(len(sorted_objects) - 1):
            name1, r1, scale1, type1 = sorted_objects[i]
            name2, r2, scale2, type2 = sorted_objects[i + 1]
            
            gap_ratio = r2 / r1
            gap_size = r2 - r1
            
            gaps.append({
                'lower_object': name1,
                'upper_object': name2,
                'lower_radius': r1,
                'upper_radius': r2,
                'lower_scale': scale1,
                'upper_scale': scale2,
                'lower_type': type1,
                'upper_type': type2,
                'gap_ratio': gap_ratio,
                'gap_size': gap_size,
                'gap_log_size': math.log10(gap_ratio),
                'scale_transition': scale1 != scale2
            })
        
        return gaps
    
    def classify_jungle_density(self, gap_ratio: float) -> Tuple[str, float, float]:
        """Classify jungle density based on gap size."""
        # Small gaps (< 1.5×) = structured (easy to cross)
        # Medium gaps (1.5× - 10×) = messy jungle
        # Large gaps (> 10×) = deep jungle (hard to cross)
        
        if gap_ratio < 1.5:
            return "structured", 0.9, 0.1  # 90% structure, 10% mess
        elif gap_ratio < 3.0:
            return "light_jungle", 0.7, 0.3
        elif gap_ratio < 10.0:
            return "medium_jungle", 0.5, 0.5
        elif gap_ratio < 100.0:
            return "deep_jungle", 0.3, 0.7
        else:
            return "void", 0.1, 0.9  # 10% structure, 90% mess
    
    def map_jungles_by_scale(self) -> Dict:
        """Map jungle zones within each scale."""
        gaps = self.find_all_gaps()
        
        scale_jungles = defaultdict(list)
        
        for gap in gaps:
            # Only consider gaps within same scale
            if not gap['scale_transition']:
                scale = gap['lower_scale']
                density_class, structure, mess = self.classify_jungle_density(gap['gap_ratio'])
                
                scale_jungles[scale].append({
                    **gap,
                    'density_class': density_class,
                    'structure_percent': structure * 100,
                    'mess_percent': mess * 100
                })
        
        # Calculate statistics for each scale
        scale_stats = {}
        for scale, jungles in scale_jungles.items():
            if jungles:
                avg_structure = sum(j['structure_percent'] for j in jungles) / len(jungles)
                avg_mess = sum(j['mess_percent'] for j in jungles) / len(jungles)
                
                scale_stats[scale] = {
                    'name': self.get_scale_name(scale),
                    'jungle_count': len(jungles),
                    'avg_structure_percent': avg_structure,
                    'avg_mess_percent': avg_mess,
                    'jungles': sorted(jungles, key=lambda x: x['gap_ratio'], reverse=True)
                }
        
        return scale_stats
    
    def map_scale_transitions(self) -> List[Dict]:
        """Map jungle zones at scale transition boundaries."""
        gaps = self.find_all_gaps()
        
        transitions = []
        for gap in gaps:
            if gap['scale_transition']:
                density_class, structure, mess = self.classify_jungle_density(gap['gap_ratio'])
                
                transitions.append({
                    **gap,
                    'density_class': density_class,
                    'structure_percent': structure * 100,
                    'mess_percent': mess * 100,
                    'scale_jump': abs(gap['upper_scale'] - gap['lower_scale'])
                })
        
        return sorted(transitions, key=lambda x: x['gap_ratio'], reverse=True)
    
    def find_minimum_field_zones(self) -> Dict:
        """Identify jungle zones near minimum field thresholds."""
        gaps = self.find_all_gaps()
        
        min_fields = {
            'F₀₁': F_01,
            'F₁₂': F_12,
            'F₂₃': F_23,
            'F₃₄': F_34
        }
        
        field_zones = defaultdict(list)
        
        for gap in gaps:
            # Check if gap size is near a minimum field
            for field_name, field_value in min_fields.items():
                # Compare gap ratio to field value
                if abs(math.log10(gap['gap_ratio']) - math.log10(field_value)) < 0.5:
                    field_zones[field_name].append({
                        **gap,
                        'field_value': field_value,
                        'field_match_error': abs(gap['gap_ratio'] - field_value) / field_value * 100
                    })
        
        return dict(field_zones)
    
    def calculate_jungle_coverage(self) -> Dict:
        """Calculate what percentage of size space is jungle."""
        gaps = self.find_all_gaps()
        
        total_log_space = 0
        jungle_log_space = 0
        
        for gap in gaps:
            log_size = gap['gap_log_size']
            total_log_space += log_size
            
            # Jungle if gap ratio > 1.5
            if gap['gap_ratio'] > 1.5:
                jungle_log_space += log_size
        
        jungle_percent = (jungle_log_space / total_log_space * 100) if total_log_space > 0 else 0
        
        return {
            'total_log_space': total_log_space,
            'jungle_log_space': jungle_log_space,
            'structured_log_space': total_log_space - jungle_log_space,
            'jungle_percent': jungle_percent,
            'structured_percent': 100 - jungle_percent
        }
    
    def identify_stable_islands(self) -> List[Dict]:
        """Identify stable size ranges (islands in the jungle)."""
        gaps = self.find_all_gaps()
        
        islands = []
        current_island = None
        
        for gap in gaps:
            if gap['gap_ratio'] < 1.5:  # Structured region
                if current_island is None:
                    current_island = {
                        'start_object': gap['lower_object'],
                        'start_radius': gap['lower_radius'],
                        'objects': [gap['lower_object'], gap['upper_object']],
                        'scale': gap['lower_scale']
                    }
                else:
                    current_island['objects'].append(gap['upper_object'])
                    current_island['end_object'] = gap['upper_object']
                    current_island['end_radius'] = gap['upper_radius']
            else:  # Jungle - end current island
                if current_island is not None:
                    current_island['object_count'] = len(current_island['objects'])
                    if 'end_radius' in current_island:
                        current_island['radius_range'] = current_island['end_radius'] / current_island['start_radius']
                    islands.append(current_island)
                    current_island = None
        
        # Add final island if exists
        if current_island is not None:
            current_island['object_count'] = len(current_island['objects'])
            if 'end_radius' in current_island:
                current_island['radius_range'] = current_island['end_radius'] / current_island['start_radius']
            islands.append(current_island)
        
        return sorted(islands, key=lambda x: x['object_count'], reverse=True)
    
    def map_earth_moon_sun_jungle(self) -> Dict:
        """Special analysis of jungle in Earth-Moon-Sun system."""
        earth = self.catalog['earth']
        moon = self.catalog['moon']
        sun = self.catalog['sun']
        
        # Find all objects between Moon and Sun
        between_objects = []
        for name, data in self.catalog.items():
            if moon['radius'] < data['radius'] < sun['radius']:
                between_objects.append({
                    'name': name,
                    'radius': data['radius'],
                    'type': data['type'],
                    'scale': data['scale']
                })
        
        between_objects = sorted(between_objects, key=lambda x: x['radius'])
        
        # Calculate jungle density
        if between_objects:
            gaps = []
            prev_radius = moon['radius']
            for obj in between_objects:
                gap_ratio = obj['radius'] / prev_radius
                gaps.append(gap_ratio)
                prev_radius = obj['radius']
            
            # Final gap to sun
            gaps.append(sun['radius'] / prev_radius)
            
            avg_gap = sum(gaps) / len(gaps)
            max_gap = max(gaps)
        else:
            avg_gap = sun['radius'] / moon['radius']
            max_gap = avg_gap
        
        return {
            'moon_radius': moon['radius'],
            'sun_radius': sun['radius'],
            'total_ratio': sun['radius'] / moon['radius'],
            'objects_between': len(between_objects),
            'between_objects': between_objects,
            'average_gap_ratio': avg_gap,
            'max_gap_ratio': max_gap,
            'jungle_density': 'deep_jungle' if avg_gap > 10 else 'medium_jungle'
        }
    
    def generate_report(self) -> Dict:
        """Generate complete jungle mapping report."""
        return {
            'all_gaps': self.find_all_gaps(),
            'scale_jungles': self.map_jungles_by_scale(),
            'scale_transitions': self.map_scale_transitions(),
            'minimum_field_zones': self.find_minimum_field_zones(),
            'jungle_coverage': self.calculate_jungle_coverage(),
            'stable_islands': self.identify_stable_islands(),
            'earth_moon_sun_jungle': self.map_earth_moon_sun_jungle(),
            'constants': {
                'C_STAR': C_STAR,
                'F_01': F_01,
                'F_12': F_12,
                'F_23': F_23,
                'F_34': F_34
            }
        }


def run_tests() -> Dict:
    """Run test suite for jungle mapper."""
    mapper = JungleMapper()
    results = {
        'tests': [],
        'passed': 0,
        'failed': 0,
        'total': 8
    }
    
    # Test 1: All gaps found
    gaps = mapper.find_all_gaps()
    test1 = len(gaps) >= 70
    results['tests'].append({
        'name': 'All Gaps Found',
        'passed': test1,
        'details': f"Found {len(gaps)} gaps (target: ≥70)"
    })
    results['passed' if test1 else 'failed'] += 1
    
    # Test 2: Scale transitions identified
    transitions = mapper.map_scale_transitions()
    test2 = len(transitions) >= 5
    results['tests'].append({
        'name': 'Scale Transitions Identified',
        'passed': test2,
        'details': f"Found {len(transitions)} transitions (target: ≥5)"
    })
    results['passed' if test2 else 'failed'] += 1
    
    # Test 3: Jungle coverage calculated
    coverage = mapper.calculate_jungle_coverage()
    test3 = 0 < coverage['jungle_percent'] < 100
    results['tests'].append({
        'name': 'Jungle Coverage Calculated',
        'passed': test3,
        'details': f"{coverage['jungle_percent']:.1f}% jungle (valid range)"
    })
    results['passed' if test3 else 'failed'] += 1
    
    # Test 4: Stable islands found
    islands = mapper.identify_stable_islands()
    test4 = len(islands) >= 3
    results['tests'].append({
        'name': 'Stable Islands Found',
        'passed': test4,
        'details': f"Found {len(islands)} islands (target: ≥3)"
    })
    results['passed' if test4 else 'failed'] += 1
    
    # Test 5: Scale jungles mapped
    scale_jungles = mapper.map_jungles_by_scale()
    test5 = len(scale_jungles) >= 4
    results['tests'].append({
        'name': 'Scale Jungles Mapped',
        'passed': test5,
        'details': f"Mapped {len(scale_jungles)} scales (target: ≥4)"
    })
    results['passed' if test5 else 'failed'] += 1
    
    # Test 6: Earth-Moon-Sun jungle analyzed
    ems_jungle = mapper.map_earth_moon_sun_jungle()
    test6 = ems_jungle['total_ratio'] > 100
    results['tests'].append({
        'name': 'Earth-Moon-Sun Jungle Analyzed',
        'passed': test6,
        'details': f"Moon-Sun ratio: {ems_jungle['total_ratio']:.1f}× (target: >100)"
    })
    results['passed' if test6 else 'failed'] += 1
    
    # Test 7: Minimum field zones identified
    field_zones = mapper.find_minimum_field_zones()
    test7 = len(field_zones) >= 1
    results['tests'].append({
        'name': 'Minimum Field Zones Identified',
        'passed': test7,
        'details': f"Found {len(field_zones)} field zones (target: ≥1)"
    })
    results['passed' if test7 else 'failed'] += 1
    
    # Test 8: Jungle is messy but structured
    test8 = 20 < coverage['jungle_percent'] < 80
    results['tests'].append({
        'name': 'Jungle is Messy But Structured',
        'passed': test8,
        'details': f"{coverage['jungle_percent']:.1f}% jungle (target: 20-80%)"
    })
    results['passed' if test8 else 'failed'] += 1
    
    results['pass_rate'] = results['passed'] / results['total']
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("PROJECT BUSHMAN SUITE 7: THE OBSERVATORY")
    print("Program 3: Jungle Mapper")
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
    print("Generating complete jungle map...")
    mapper = JungleMapper()
    report = mapper.generate_report()
    
    # Save report
    with open('jungle_mapper_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to jungle_mapper_results.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    coverage = report['jungle_coverage']
    print(f"Total Size Space: {coverage['total_log_space']:.1f} log units")
    print(f"Jungle Coverage: {coverage['jungle_percent']:.1f}%")
    print(f"Structured Coverage: {coverage['structured_percent']:.1f}%")
    print()
    
    print("Scale-by-Scale Jungle Analysis:")
    for scale_num, scale_data in sorted(report['scale_jungles'].items()):
        print(f"  {scale_data['name']}:")
        print(f"    Jungle zones: {scale_data['jungle_count']}")
        print(f"    Avg structure: {scale_data['avg_structure_percent']:.1f}%")
        print(f"    Avg mess: {scale_data['avg_mess_percent']:.1f}%")
    print()
    
    print("Largest Scale Transitions (Deepest Jungles):")
    for trans in report['scale_transitions'][:5]:
        print(f"  {trans['lower_object']} → {trans['upper_object']}")
        print(f"    Gap: {trans['gap_ratio']:.1f}× ({trans['density_class']})")
        print(f"    Structure: {trans['structure_percent']:.1f}%, Mess: {trans['mess_percent']:.1f}%")
    print()
    
    print("Stable Islands (Structured Regions):")
    for island in report['stable_islands'][:5]:
        print(f"  {island['start_object']} → {island.get('end_object', 'N/A')}")
        print(f"    Objects: {island['object_count']}")
        if 'radius_range' in island:
            print(f"    Range: {island['radius_range']:.2f}×")
    print()
    
    ems = report['earth_moon_sun_jungle']
    print("Earth-Moon-Sun Jungle:")
    print(f"  Moon → Sun ratio: {ems['total_ratio']:.1f}×")
    print(f"  Objects between: {ems['objects_between']}")
    print(f"  Average gap: {ems['average_gap_ratio']:.1f}×")
    print(f"  Jungle density: {ems['jungle_density']}")
    
    print("\n" + "=" * 80)
    print("THE JUNGLE IS REAL. THE JUNGLE IS STRUCTURED.")
    print("=" * 80)
"""
Project Bushman Suite 7: "The Observatory"
Program 4: Formation Simulator

Purpose: Model how spheres form from matter at different scales through
         gravity, surface tension, quantum mechanics, and electromagnetic forces.

Author: SuperNinja AI (for Matthew Pidlysny)
Date: December 2024
"""

import json
import math
from typing import Dict, List, Tuple

# Constants
C_STAR = 0.894751918
F_01 = C_STAR
F_12 = 4.0 * C_STAR
F_23 = 7.07 * F_12
F_34 = 5.09 * C_STAR

# Physical constants
G = 6.674e-11  # Gravitational constant (m³/kg·s²)
HBAR = 1.055e-34  # Reduced Planck constant (J·s)
K_E = 8.988e9  # Coulomb constant (N·m²/C²)
SIGMA_WATER = 0.0728  # Surface tension of water (N/m)
K_B = 1.381e-23  # Boltzmann constant (J/K)

class FormationSimulator:
    """Simulate sphere formation mechanisms across scales."""
    
    def __init__(self, catalog_file: str = 'scale_scanner_results.json'):
        with open(catalog_file, 'r') as f:
            data = json.load(f)
            self.catalog = data['catalog']
    
    def gravitational_collapse_energy(self, mass: float, radius: float) -> float:
        """Calculate gravitational binding energy for sphere formation."""
        # E = (3/5) * G * M² / R
        return (3/5) * G * mass**2 / radius
    
    def surface_tension_energy(self, radius: float, sigma: float = SIGMA_WATER) -> float:
        """Calculate surface energy for droplet formation."""
        # E = 4πr²σ
        return 4 * math.pi * radius**2 * sigma
    
    def quantum_confinement_energy(self, radius: float, mass: float) -> float:
        """Calculate quantum confinement energy for atomic scales."""
        # E ≈ ℏ²/(2mr²)
        if radius > 0 and mass > 0:
            return HBAR**2 / (2 * mass * radius**2)
        return 0
    
    def electromagnetic_energy(self, charge: float, radius: float) -> float:
        """Calculate electrostatic self-energy."""
        # E = k_e * Q² / (2R)
        if radius > 0:
            return K_E * charge**2 / (2 * radius)
        return 0
    
    def thermal_energy(self, temperature: float) -> float:
        """Calculate thermal energy per particle."""
        # E = (3/2) * k_B * T
        return (3/2) * K_B * temperature
    
    def determine_formation_mechanism(self, radius: float, mass: float) -> Dict:
        """Determine dominant formation mechanism for given size."""
        # Calculate all energies
        grav_energy = self.gravitational_collapse_energy(mass, radius)
        surf_energy = self.surface_tension_energy(radius)
        quantum_energy = self.quantum_confinement_energy(radius, mass)
        
        # Determine dominant mechanism
        energies = {
            'gravitational': grav_energy,
            'surface_tension': surf_energy,
            'quantum': quantum_energy
        }
        
        dominant = max(energies, key=energies.get)
        
        return {
            'radius': radius,
            'mass': mass,
            'dominant_mechanism': dominant,
            'gravitational_energy': grav_energy,
            'surface_tension_energy': surf_energy,
            'quantum_energy': quantum_energy,
            'total_energy': sum(energies.values())
        }
    
    def simulate_all_formations(self) -> Dict[str, Dict]:
        """Simulate formation for all cataloged objects."""
        formations = {}
        
        for name, data in self.catalog.items():
            radius = data['radius']
            mass = data['mass']
            
            formation = self.determine_formation_mechanism(radius, mass)
            formation['object_name'] = name
            formation['scale'] = data['scale']
            formation['type'] = data['type']
            
            formations[name] = formation
        
        return formations
    
    def analyze_formation_by_scale(self) -> Dict:
        """Analyze dominant formation mechanisms by scale."""
        formations = self.simulate_all_formations()
        
        scale_analysis = {}
        for scale in range(6):
            scale_objects = [f for f in formations.values() if f['scale'] == scale]
            
            if scale_objects:
                mechanisms = {}
                for obj in scale_objects:
                    mech = obj['dominant_mechanism']
                    mechanisms[mech] = mechanisms.get(mech, 0) + 1
                
                dominant = max(mechanisms, key=mechanisms.get)
                
                scale_analysis[scale] = {
                    'scale_name': self.get_scale_name(scale),
                    'object_count': len(scale_objects),
                    'dominant_mechanism': dominant,
                    'mechanism_counts': mechanisms,
                    'avg_energy': sum(o['total_energy'] for o in scale_objects) / len(scale_objects)
                }
        
        return scale_analysis
    
    def find_minimum_field_energies(self) -> Dict:
        """Calculate energies at minimum field thresholds."""
        # Use representative objects near each threshold
        min_field_energies = {}
        
        # F₀₁: 0D → 1D (quantum scale)
        if 'proton' in self.catalog:
            proton = self.catalog['proton']
            energy = self.determine_formation_mechanism(proton['radius'], proton['mass'])
            min_field_energies['F_01'] = {
                'field_value': F_01,
                'representative_object': 'proton',
                'energy': energy['total_energy'],
                'mechanism': energy['dominant_mechanism']
            }
        
        # F₁₂: 1D → 2D (molecular scale)
        if 'water_molecule' in self.catalog:
            water = self.catalog['water_molecule']
            energy = self.determine_formation_mechanism(water['radius'], water['mass'])
            min_field_energies['F_12'] = {
                'field_value': F_12,
                'representative_object': 'water_molecule',
                'energy': energy['total_energy'],
                'mechanism': energy['dominant_mechanism']
            }
        
        # F₂₃: 2D → 3D (terrestrial scale)
        if 'raindrop_medium' in self.catalog:
            raindrop = self.catalog['raindrop_medium']
            energy = self.determine_formation_mechanism(raindrop['radius'], raindrop['mass'])
            min_field_energies['F_23'] = {
                'field_value': F_23,
                'representative_object': 'raindrop_medium',
                'energy': energy['total_energy'],
                'mechanism': energy['dominant_mechanism']
            }
        
        # F₃₄: 3D → 4D (planetary scale)
        if 'earth' in self.catalog:
            earth = self.catalog['earth']
            energy = self.determine_formation_mechanism(earth['radius'], earth['mass'])
            min_field_energies['F_34'] = {
                'field_value': F_34,
                'representative_object': 'earth',
                'energy': energy['total_energy'],
                'mechanism': energy['dominant_mechanism']
            }
        
        return min_field_energies
    
    def calculate_stability_criterion(self, radius: float, mass: float) -> Dict:
        """Calculate stability criterion for sphere."""
        formation = self.determine_formation_mechanism(radius, mass)
        
        # Jeans criterion for gravitational collapse
        if mass > 0 and radius > 0:
            density = mass / ((4/3) * math.pi * radius**3)
            jeans_mass = (5 * K_B * 300) / (G * 2e-27) * (3/(4*math.pi*density))**(1/2)  # Simplified
            jeans_stable = mass > jeans_mass
        else:
            jeans_stable = False
        
        # Surface tension stability (Weber number)
        weber_stable = formation['surface_tension_energy'] > formation['gravitational_energy']
        
        return {
            'radius': radius,
            'mass': mass,
            'jeans_stable': jeans_stable,
            'weber_stable': weber_stable,
            'overall_stable': jeans_stable or weber_stable,
            'dominant_mechanism': formation['dominant_mechanism']
        }
    
    def simulate_conglomeration(self, r1: float, m1: float, r2: float, m2: float) -> Dict:
        """Simulate two spheres merging."""
        # Conservation of mass
        m_total = m1 + m2
        
        # Conservation of volume (assuming same density)
        v_total = (4/3) * math.pi * (r1**3 + r2**3)
        r_total = (v_total / ((4/3) * math.pi))**(1/3)
        
        # Energy before
        energy_before = (self.determine_formation_mechanism(r1, m1)['total_energy'] +
                        self.determine_formation_mechanism(r2, m2)['total_energy'])
        
        # Energy after
        energy_after = self.determine_formation_mechanism(r_total, m_total)['total_energy']
        
        # Energy released
        energy_released = energy_before - energy_after
        
        return {
            'r1': r1, 'm1': m1,
            'r2': r2, 'm2': m2,
            'r_merged': r_total,
            'm_merged': m_total,
            'energy_before': energy_before,
            'energy_after': energy_after,
            'energy_released': energy_released,
            'favorable': energy_released > 0
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
        """Generate complete formation analysis report."""
        formations = self.simulate_all_formations()
        scale_analysis = self.analyze_formation_by_scale()
        min_field_energies = self.find_minimum_field_energies()
        
        # Test conglomeration with example pairs
        conglomeration_examples = []
        if 'raindrop_small' in self.catalog and 'raindrop_medium' in self.catalog:
            r1 = self.catalog['raindrop_small']
            r2 = self.catalog['raindrop_medium']
            conglom = self.simulate_conglomeration(r1['radius'], r1['mass'], 
                                                   r2['radius'], r2['mass'])
            conglom['pair'] = 'raindrop_small + raindrop_medium'
            conglomeration_examples.append(conglom)
        
        if 'moon' in self.catalog and 'mars' in self.catalog:
            moon = self.catalog['moon']
            mars = self.catalog['mars']
            conglom = self.simulate_conglomeration(moon['radius'], moon['mass'],
                                                   mars['radius'], mars['mass'])
            conglom['pair'] = 'moon + mars'
            conglomeration_examples.append(conglom)
        
        return {
            'formations': formations,
            'scale_analysis': scale_analysis,
            'minimum_field_energies': min_field_energies,
            'conglomeration_examples': conglomeration_examples,
            'constants': {
                'C_STAR': C_STAR,
                'F_01': F_01,
                'F_12': F_12,
                'F_23': F_23,
                'F_34': F_34,
                'G': G,
                'HBAR': HBAR,
                'K_E': K_E
            }
        }


def run_tests() -> Dict:
    """Run test suite for formation simulator."""
    simulator = FormationSimulator()
    results = {
        'tests': [],
        'passed': 0,
        'failed': 0,
        'total': 8
    }
    
    # Test 1: All formations simulated
    formations = simulator.simulate_all_formations()
    test1 = len(formations) >= 70
    results['tests'].append({
        'name': 'All Formations Simulated',
        'passed': test1,
        'details': f"Simulated {len(formations)} formations (target: ≥70)"
    })
    results['passed' if test1 else 'failed'] += 1
    
    # Test 2: Scale analysis complete
    scale_analysis = simulator.analyze_formation_by_scale()
    test2 = len(scale_analysis) >= 5
    results['tests'].append({
        'name': 'Scale Analysis Complete',
        'passed': test2,
        'details': f"Analyzed {len(scale_analysis)} scales (target: ≥5)"
    })
    results['passed' if test2 else 'failed'] += 1
    
    # Test 3: Quantum scale uses quantum mechanics
    if 0 in scale_analysis:
        quantum_mech = scale_analysis[0]['dominant_mechanism']
        test3 = quantum_mech == 'quantum'
        results['tests'].append({
            'name': 'Quantum Scale Uses Quantum Mechanics',
            'passed': test3,
            'details': f"Dominant: {quantum_mech} (target: quantum)"
        })
        results['passed' if test3 else 'failed'] += 1
    else:
        results['tests'].append({
            'name': 'Quantum Scale Uses Quantum Mechanics',
            'passed': False,
            'details': "Quantum scale not found"
        })
        results['failed'] += 1
    
    # Test 4: Planetary scale uses gravity
    if 3 in scale_analysis:
        planetary_mech = scale_analysis[3]['dominant_mechanism']
        test4 = planetary_mech == 'gravitational'
        results['tests'].append({
            'name': 'Planetary Scale Uses Gravity',
            'passed': test4,
            'details': f"Dominant: {planetary_mech} (target: gravitational)"
        })
        results['passed' if test4 else 'failed'] += 1
    else:
        results['tests'].append({
            'name': 'Planetary Scale Uses Gravity',
            'passed': False,
            'details': "Planetary scale not found"
        })
        results['failed'] += 1
    
    # Test 5: Minimum field energies calculated
    min_field_energies = simulator.find_minimum_field_energies()
    test5 = len(min_field_energies) >= 3
    results['tests'].append({
        'name': 'Minimum Field Energies Calculated',
        'passed': test5,
        'details': f"Calculated {len(min_field_energies)} fields (target: ≥3)"
    })
    results['passed' if test5 else 'failed'] += 1
    
    # Test 6: Conglomeration simulation works
    if 'raindrop_small' in simulator.catalog and 'raindrop_medium' in simulator.catalog:
        r1 = simulator.catalog['raindrop_small']
        r2 = simulator.catalog['raindrop_medium']
        conglom = simulator.simulate_conglomeration(r1['radius'], r1['mass'],
                                                    r2['radius'], r2['mass'])
        test6 = conglom['m_merged'] == r1['mass'] + r2['mass']
        results['tests'].append({
            'name': 'Conglomeration Conserves Mass',
            'passed': test6,
            'details': f"Mass conserved: {test6}"
        })
        results['passed' if test6 else 'failed'] += 1
    else:
        results['tests'].append({
            'name': 'Conglomeration Conserves Mass',
            'passed': False,
            'details': "Test objects not found"
        })
        results['failed'] += 1
    
    # Test 7: Energy calculations positive
    all_positive = all(f['total_energy'] >= 0 for f in formations.values())
    test7 = all_positive
    results['tests'].append({
        'name': 'Energy Calculations Positive',
        'passed': test7,
        'details': f"All energies positive: {test7}"
    })
    results['passed' if test7 else 'failed'] += 1
    
    # Test 8: Different mechanisms at different scales
    mechanisms = set(s['dominant_mechanism'] for s in scale_analysis.values())
    test8 = len(mechanisms) >= 2
    results['tests'].append({
        'name': 'Multiple Formation Mechanisms',
        'passed': test8,
        'details': f"Found {len(mechanisms)} mechanisms (target: ≥2)"
    })
    results['passed' if test8 else 'failed'] += 1
    
    results['pass_rate'] = results['passed'] / results['total']
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("PROJECT BUSHMAN SUITE 7: THE OBSERVATORY")
    print("Program 4: Formation Simulator")
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
    print("Generating formation analysis...")
    simulator = FormationSimulator()
    report = simulator.generate_report()
    
    # Save report
    with open('formation_simulator_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to formation_simulator_results.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("Formation Mechanisms by Scale:")
    for scale_num, scale_data in sorted(report['scale_analysis'].items()):
        print(f"  {scale_data['scale_name']}:")
        print(f"    Dominant: {scale_data['dominant_mechanism']}")
        print(f"    Objects: {scale_data['object_count']}")
        print(f"    Avg Energy: {scale_data['avg_energy']:.2e} J")
    print()
    
    print("Minimum Field Energies:")
    for field_name, field_data in report['minimum_field_energies'].items():
        print(f"  {field_name} ({field_data['representative_object']}):")
        print(f"    Energy: {field_data['energy']:.2e} J")
        print(f"    Mechanism: {field_data['mechanism']}")
    print()
    
    print("Conglomeration Examples:")
    for conglom in report['conglomeration_examples']:
        print(f"  {conglom['pair']}:")
        print(f"    Merged radius: {conglom['r_merged']:.2e} m")
        print(f"    Energy released: {conglom['energy_released']:.2e} J")
        print(f"    Favorable: {conglom['favorable']}")
    
    print("\n" + "=" * 80)
    print("FORMATION FOLLOWS PHYSICS. PHYSICS FOLLOWS C*.")
    print("=" * 80)
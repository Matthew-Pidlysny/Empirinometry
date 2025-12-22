"""
Quantum Tunneling Diffusion Workshop
Advanced workshop on quantum tunneling effects in atomic diffusion

Workshop ID: QTW-001
Category: Quantum Mechanics
Difficulty: Expert
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from scipy.special import erf
from dataclasses import dataclass
from enum import Enum

class TunnelingRegime(Enum):
    """Quantum tunneling regimes"""
    WKB_APPROXIMATION = "wkb"
    EXACT_SOLVABLE = "exact"
    SEMICLASSICAL = "semiclassical"
    STRONG_TUNNELING = "strong"
    WEAK_TUNNELING = "weak"

class BarrierShape(Enum):
    """Potential barrier shapes"""
    RECTANGULAR = "rectangular"
    PARABOLIC = "parabolic"
    TRIANGULAR = "triangular"
    GAUSSIAN = "gaussian"
    MORSE = "morse"

@dataclass
class QuantumParticle:
    """Quantum particle properties"""
    mass: float  # kg
    charge: float  # C
    spin: float  # ℏ/2 units
    energy: float  # J

class QuantumTunnelingWorkshop:
    """
    Advanced workshop on quantum tunneling in diffusion processes.
    
    Covers:
    - WKB approximation for tunneling probabilities
    - Exact solutions for solvable potentials
    - Temperature-dependent tunneling
    - Isotope effects
    - Multi-dimensional tunneling
    - Tunneling in crystalline materials
    - Comparison with classical diffusion
    """
    
    def __init__(self):
        """Initialize quantum tunneling workshop"""
        self.physical_constants = {
            'h': 6.62607015e-34,  # Planck constant (J·s)
            'hbar': 1.054571817e-34,  # Reduced Planck constant (J·s)
            'kB': 1.380649e-23,  # Boltzmann constant (J/K)
            'e': 1.602176634e-19,  # Elementary charge (C)
            'm_e': 9.1093837015e-31,  # Electron mass (kg)
            'm_p': 1.67262192369e-27,  # Proton mass (kg)
            'amu': 1.66053906660e-27  # Atomic mass unit (kg)
        }
        
        self.common_particles = self._initialize_particles()
        
    def _initialize_particles(self) -> Dict[str, QuantumParticle]:
        """Initialize common particles for diffusion"""
        return {
            'hydrogen': QuantumParticle(
                mass=self.physical_constants['amu'] * 1.008,
                charge=self.physical_constants['e'],
                spin=0.5,
                energy=0.0
            ),
            'deuterium': QuantumParticle(
                mass=self.physical_constants['amu'] * 2.014,
                charge=self.physical_constants['e'],
                spin=1.0,
                energy=0.0
            ),
            'tritium': QuantumParticle(
                mass=self.physical_constants['amu'] * 3.016,
                charge=self.physical_constants['e'],
                spin=0.5,
                energy=0.0
            ),
            'helium': QuantumParticle(
                mass=self.physical_constants['amu'] * 4.003,
                charge=0,
                spin=0.0,
                energy=0.0
            ),
            'electron': QuantumParticle(
                mass=self.physical_constants['m_e'],
                charge=-self.physical_constants['e'],
                spin=0.5,
                energy=0.0
            )
        }
    
    def calculate_wkb_tunneling_probability(self,
                                          particle: str,
                                          barrier_height: float,
                                          barrier_width: float,
                                          energy: float) -> Dict[str, Any]:
        """
        Calculate tunneling probability using WKB approximation
        
        Args:
            particle: Particle type ('hydrogen', 'deuterium', etc.)
            barrier_height: Barrier height in Joules
            barrier_width: Barrier width in meters
            energy: Particle energy in Joules
            
        Returns:
            Dictionary with tunneling results
        """
        if particle not in self.common_particles:
            raise ValueError(f"Particle {particle} not available")
        
        particle_obj = self.common_particles[particle]
        mass = particle_obj.mass
        
        # WKB integral for rectangular barrier
        if energy < barrier_height:
            # Classically forbidden region
            kappa = np.sqrt(2 * mass * (barrier_height - energy)) / self.physical_constants['hbar']
            exponent = 2 * kappa * barrier_width
            tunneling_prob = np.exp(-exponent)
            
            # Transmission coefficient
            transmission_coeff = tunneling_prob / (1 + tunneling_prob)
            
            # Reflection coefficient
            reflection_coeff = 1 - transmission_coeff
            
            regime = TunnelingRegime.WKB_APPROXIMATION
            
        else:
            # Classically allowed - no tunneling needed
            tunneling_prob = 1.0
            transmission_coeff = 1.0
            reflection_coeff = 0.0
            regime = TunnelingRegime.CLASSICAL
        
        # Calculate characteristic tunneling length
        if energy < barrier_height:
            decay_length = 1 / kappa
        else:
            decay_length = float('inf')
        
        # Temperature corresponding to particle energy
        equivalent_temp = energy / self.physical_constants['kB']
        
        # Quantum correction factor for diffusion
        quantum_factor = 1 + tunneling_prob * 10  # Enhanced diffusion due to tunneling
        
        results = {
            'particle': particle,
            'mass': mass,
            'energy': energy,
            'barrier_height': barrier_height,
            'barrier_width': barrier_width,
            'tunneling_probability': tunneling_prob,
            'transmission_coefficient': transmission_coeff,
            'reflection_coefficient': reflection_coeff,
            'decay_length': decay_length,
            'regime': regime.value,
            'equivalent_temperature': equivalent_temp,
            'quantum_enhancement_factor': quantum_factor,
            'is_quantum_significant': tunneling_prob > 0.01,
            'wkb_parameter': kappa * barrier_width
        }
        
        return results
    
    def calculate_isotope_effect(self,
                               base_particle: str,
                               isotope: str,
                               barrier_height: float,
                               barrier_width: float,
                               temperature: float) -> Dict[str, Any]:
        """
        Calculate isotope effect on quantum tunneling
        
        Args:
            base_particle: Reference particle (e.g., 'hydrogen')
            isotope: Isotope particle (e.g., 'deuterium')
            barrier_height: Barrier height in Joules
            barrier_width: Barrier width in meters
            temperature: Temperature in Kelvin
            
        Returns:
            Dictionary with isotope effect results
        """
        # Thermal energy
        thermal_energy = self.physical_constants['kB'] * temperature
        
        # Calculate tunneling for both isotopes
        base_result = self.calculate_wkb_tunneling_probability(
            base_particle, barrier_height, barrier_width, thermal_energy
        )
        
        isotope_result = self.calculate_wkb_tunneling_probability(
            isotope, barrier_height, barrier_width, thermal_energy
        )
        
        # Isotope effect ratio
        isotope_ratio = isotope_result['tunneling_probability'] / base_result['tunneling_probability']
        
        # KIE (Kinetic Isotope Effect)
        kie = base_result['tunneling_probability'] / isotope_result['tunneling_probability']
        
        # Mass ratio
        mass_ratio = self.common_particles[isotope].mass / self.common_particles[base_particle].mass
        
        # Expected classical KIE (from transition state theory)
        classical_kie = np.sqrt(mass_ratio)
        
        # Quantum enhancement factor
        quantum_enhancement = kie / classical_kie
        
        results = {
            'base_particle': base_particle,
            'isotope': isotope,
            'temperature': temperature,
            'mass_ratio': mass_ratio,
            'base_tunneling_prob': base_result['tunneling_probability'],
            'isotope_tunneling_prob': isotope_result['tunneling_probability'],
            'isotope_ratio': isotope_ratio,
            'kinetic_isotope_effect': kie,
            'classical_kie': classical_kie,
            'quantum_enhancement': quantum_enhancement,
            'is_quantum_dominated': quantum_enhancement > 2.0,
            'temperature': temperature,
            'thermal_energy': thermal_energy,
            'base_quantum_significant': base_result['is_quantum_significant'],
            'isotope_quantum_significant': isotope_result['is_quantum_significant']
        }
        
        return results
    
    def multi_barrier_tunneling(self,
                              particle: str,
                              barriers: List[Tuple[float, float]],  # List of (height, width)
                              energy: float) -> Dict[str, Any]:
        """
        Calculate tunneling through multiple barriers
        
        Args:
            particle: Particle type
            barriers: List of (height, width) tuples for each barrier
            energy: Particle energy
            
        Returns:
            Dictionary with multi-barrier results
        """
        total_transmission = 1.0
        individual_transmissions = []
        
        for i, (height, width) in enumerate(barriers):
            result = self.calculate_wkb_tunneling_probability(particle, height, width, energy)
            individual_transmissions.append(result['transmission_coefficient'])
            total_transmission *= result['transmission_coefficient']
        
        # Overall tunneling probability
        total_tunneling_prob = total_transmission / (1 + total_transmission)
        
        # Compare with single equivalent barrier
        total_width = sum(width for _, width in barriers)
        avg_height = np.mean([height for height, _ in barriers])
        
        equivalent_result = self.calculate_wkb_tunneling_probability(
            particle, avg_height, total_width, energy
        )
        
        # Resonance effects (simplified)
        num_barriers = len(barriers)
        resonance_factor = 1.0
        
        if num_barriers > 1:
            # Check for potential resonance conditions
            barrier_spacing = 1e-10  # Assume 1 Angstrom spacing
            particle_obj = self.common_particles[particle]
            k = np.sqrt(2 * particle_obj.mass * energy) / self.physical_constants['hbar']
            
            # Resonance when k * spacing ≈ nπ
            for n in range(1, num_barriers):
                if abs(k * barrier_spacing - n * np.pi) < 0.1:
                    resonance_factor *= 2.0  # Simplified resonance enhancement
        
        results = {
            'particle': particle,
            'num_barriers': num_barriers,
            'energy': energy,
            'individual_transmissions': individual_transmissions,
            'total_transmission': total_transmission,
            'total_tunneling_probability': total_tunneling_prob,
            'equivalent_single_barrier': equivalent_result['tunneling_probability'],
            'enhancement_vs_equivalent': total_tunneling_prob / equivalent_result['tunneling_probability'],
            'resonance_factor': resonance_factor,
            'resonance_enhanced_total': total_tunneling_prob * resonance_factor,
            'barrier_details': barriers
        }
        
        return results
    
    def temperature_dependent_tunneling(self,
                                     particle: str,
                                     barrier_height: float,
                                     barrier_width: float,
                                     temperature_range: Tuple[float, float],
                                     num_points: int = 50) -> Dict[str, Any]:
        """
        Study temperature dependence of quantum tunneling
        
        Args:
            particle: Particle type
            barrier_height: Barrier height in Joules
            barrier_width: Barrier width in meters
            temperature_range: Temperature range (min, max) in Kelvin
            num_points: Number of temperature points
            
        Returns:
            Dictionary with temperature dependence results
        """
        temperatures = np.linspace(temperature_range[0], temperature_range[1], num_points)
        
        tunneling_probs = []
        quantum_factors = []
        classical_probs = []
        
        for temp in temperatures:
            thermal_energy = self.physical_constants['kB'] * temp
            
            # Quantum tunneling result
            quantum_result = self.calculate_wkb_tunneling_probability(
                particle, barrier_height, barrier_width, thermal_energy
            )
            
            tunneling_probs.append(quantum_result['tunneling_probability'])
            quantum_factors.append(quantum_result['quantum_enhancement_factor'])
            
            # Classical probability (overcoming barrier)
            if thermal_energy >= barrier_height:
                classical_prob = 1.0
            else:
                classical_prob = np.exp(-(barrier_height - thermal_energy) / thermal_energy)
            
            classical_probs.append(classical_prob)
        
        tunneling_probs = np.array(tunneling_probs)
        quantum_factors = np.array(quantum_factors)
        classical_probs = np.array(classical_probs)
        
        # Find crossover temperature where quantum = classical
        crossover_idx = np.argmin(np.abs(tunneling_probs - classical_probs))
        crossover_temp = temperatures[crossover_idx]
        
        # Quantum dominance fraction
        quantum_dominant_fraction = np.sum(tunneling_probs > classical_probs) / len(temperatures)
        
        # Arrhenius analysis for classical region
        classical_mask = temperatures > crossover_temp
        if np.any(classical_mask):
            log_classical = np.log(classical_probs[classical_mask])
            inv_temp = 1 / temperatures[classical_mask]
            
            arrhenius_coeffs = np.polyfit(inv_temp, log_classical, 1)
            effective_ea = -arrhenius_coeffs[0] * self.physical_constants['kB']
        else:
            effective_ea = barrier_height
        
        results = {
            'particle': particle,
            'barrier_height': barrier_height,
            'barrier_width': barrier_width,
            'temperature_range': temperature_range,
            'temperatures': temperatures.tolist(),
            'tunneling_probabilities': tunneling_probs.tolist(),
            'quantum_enhancement_factors': quantum_factors.tolist(),
            'classical_probabilities': classical_probs.tolist(),
            'crossover_temperature': crossover_temp,
            'quantum_dominant_fraction': quantum_dominant_fraction,
            'effective_activation_energy': effective_ea,
            'barrier_reduction_due_to_tunneling': (barrier_height - effective_ea) / barrier_height * 100,
            'analysis': {
                'quantum_dominant_at_low_T': quantum_dominant_fraction > 0.5,
                'smooth_crossover': abs(tunneling_probs[crossover_idx] - classical_probs[crossover_idx]) < 0.1,
                'significant_quantum_effect': np.max(quantum_factors) > 2.0
            }
        }
        
        return results
    
    def calculate_tunneling_diffusion_coefficient(self,
                                                particle: str,
                                                barrier_height: float,
                                                barrier_width: float,
                                                temperature: float,
                                                attempt_frequency: float = 1e13) -> Dict[str, Any]:
        """
        Calculate diffusion coefficient including quantum tunneling
        
        Args:
            particle: Particle type
            barrier_height: Barrier height in Joules
            barrier_width: Barrier width in meters
            temperature: Temperature in Kelvin
            attempt_frequency: Attempt frequency in Hz
            
        Returns:
            Dictionary with diffusion coefficient results
        """
        # Get tunneling probability
        thermal_energy = self.physical_constants['kB'] * temperature
        tunneling_result = self.calculate_wkb_tunneling_probability(
            particle, barrier_height, barrier_width, thermal_energy
        )
        
        # Classical hopping rate
        classical_rate = attempt_frequency * np.exp(-barrier_height / thermal_energy)
        
        # Quantum-enhanced hopping rate
        quantum_rate = attempt_frequency * tunneling_result['tunneling_probability']
        
        # Combined rate (quantum + classical)
        combined_rate = classical_rate + quantum_rate
        
        # Diffusion coefficient (Einstein relation: D = a²*f/6 for 3D)
        # Assuming jump distance ~ barrier width
        jump_distance = barrier_width
        D_classical = jump_distance**2 * classical_rate / 6
        D_quantum = jump_distance**2 * quantum_rate / 6
        D_combined = jump_distance**2 * combined_rate / 6
        
        # Quantum enhancement factor for diffusion
        diffusion_enhancement = D_combined / D_classical if D_classical > 0 else float('inf')
        
        # Characteristic diffusion times
        time_1nm = 1e-9**2 / (6 * D_combined) if D_combined > 0 else float('inf')
        
        results = {
            'particle': particle,
            'temperature': temperature,
            'attempt_frequency': attempt_frequency,
            'jump_distance': jump_distance,
            'classical_rate': classical_rate,
            'quantum_rate': quantum_rate,
            'combined_rate': combined_rate,
            'diffusion_coefficient': D_combined,
            'classical_diffusion_coefficient': D_classical,
            'quantum_diffusion_coefficient': D_quantum,
            'diffusion_enhancement': diffusion_enhancement,
            'time_to_diffuse_1nm': time_1nm,
            'tunneling_probability': tunneling_result['tunneling_probability'],
            'quantum_dominant': quantum_rate > classical_rate,
            'temperature_range_quantum': temperature < 500,  # Rough estimate
            'quantum_significance': tunneling_result['is_quantum_significant']
        }
        
        return results
    
    def create_tunneling_visualization(self, results: Dict[str, Any], plot_type: str = 'probability') -> str:
        """
        Create visualization of tunneling results
        
        Args:
            results: Tunneling calculation results
            plot_type: Type of plot ('probability', 'temperature', 'isotope', 'multi_barrier')
            
        Returns:
            Path to saved plot
        """
        if plot_type == 'probability':
            return self._create_probability_plot(results)
        elif plot_type == 'temperature':
            return self._create_temperature_plot(results)
        elif plot_type == 'isotope':
            return self._create_isotope_plot(results)
        elif plot_type == 'multi_barrier':
            return self._create_multi_barrier_plot(results)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _create_probability_plot(self, results: Dict[str, Any]) -> str:
        """Create tunneling probability visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Barrier visualization
        x = np.linspace(-2e-10, 3e-10, 1000)
        barrier_start = 0
        barrier_end = results['barrier_width']
        
        potential = np.zeros_like(x)
        barrier_mask = (x >= barrier_start) & (x <= barrier_end)
        potential[barrier_mask] = results['barrier_height']
        
        # Particle energy level
        energy_line = np.full_like(x, results['energy'])
        
        # WKB wavefunction (simplified)
        if results['energy'] < results['barrier_height']:
            particle_obj = self.common_particles[results['particle']]
            kappa = np.sqrt(2 * particle_obj.mass * (results['barrier_height'] - results['energy'])) / self.physical_constants['hbar']
            
            wavefunction = np.zeros_like(x, dtype=complex)
            
            # Incoming wave
            incident_mask = x < barrier_start
            wavefunction[incident_mask] = np.exp(1j * np.sqrt(2 * particle_obj.mass * results['energy']) / self.physical_constants['hbar'] * x[incident_mask])
            
            # Decaying wave in barrier
            barrier_mask = (x >= barrier_start) & (x <= barrier_end)
            decay_distance = x[barrier_mask] - barrier_start
            wavefunction[barrier_mask] = np.exp(-kappa * decay_distance)
            
            # Transmitted wave
            transmitted_mask = x > barrier_end
            wavefunction[transmitted_mask] = results['tunneling_probability']**0.5 * np.exp(1j * np.sqrt(2 * particle_obj.mass * results['energy']) / self.physical_constants['hbar'] * x[transmitted_mask])
        else:
            wavefunction = np.exp(1j * np.sqrt(2 * self.common_particles[results['particle']].mass * results['energy']) / self.physical_constants['hbar'] * x)
        
        # Plot potential and wavefunction
        ax1.plot(x * 1e10, potential / self.physical_constants['e'], 'b-', linewidth=2, label='Potential Barrier')
        ax1.plot(x * 1e10, energy_line / self.physical_constants['e'], 'r--', linewidth=2, label=f'Particle Energy ({results["energy"]/self.physical_constants["e"]:.3f} eV)')
        ax1.fill_between(x * 1e10, 0, potential / self.physical_constants['e'], alpha=0.3, color='blue')
        ax1.set_xlabel('Position (Angstroms)')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title(f'Quantum Tunneling - {results["particle"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Wavefunction probability
        ax2.plot(x * 1e10, np.abs(wavefunction)**2, 'g-', linewidth=2, label='|ψ|²')
        ax2.fill_between(x * 1e10, 0, np.abs(wavefunction)**2, alpha=0.3, color='green')
        ax2.set_xlabel('Position (Angstroms)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title(f'Wavefunction - P_tunnel = {results["tunneling_probability"]:.2e}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'quantum_tunneling_probability.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_temperature_plot(self, results: Dict[str, Any]) -> str:
        """Create temperature dependence plot"""
        if 'temperatures' not in results:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        temps = np.array(results['temperatures'])
        tunneling_probs = np.array(results['tunneling_probabilities'])
        classical_probs = np.array(results['classical_probabilities'])
        
        # Plot 1: Probabilities vs temperature
        ax1.semilogy(temps, tunneling_probs, 'b-', linewidth=2, label='Quantum Tunneling')
        ax1.semilogy(temps, classical_probs, 'r--', linewidth=2, label='Classical Overcome')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Temperature Dependence - {results["particle"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mark crossover point
        crossover_temp = results['crossover_temperature']
        crossover_idx = np.argmin(np.abs(temps - crossover_temp))
        ax1.plot(crossover_temp, tunneling_probs[crossover_idx], 'ko', markersize=8, label=f'Crossover ({crossover_temp:.1f} K)')
        
        # Plot 2: Enhancement factor
        enhancement = np.array(results['quantum_enhancement_factors'])
        ax2.plot(temps, enhancement, 'g-', linewidth=2)
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Quantum Enhancement Factor')
        ax2.set_title('Quantum Enhancement vs Temperature')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No Enhancement')
        ax2.legend()
        
        plt.tight_layout()
        
        filename = 'quantum_temperature_dependence.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_isotope_plot(self, results: Dict[str, Any]) -> str:
        """Create isotope effect visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mass comparison
        particles = [results['base_particle'], results['isotope']]
        masses = [
            self.common_particles[results['base_particle']].mass / self.physical_constants['amu'],
            self.common_particles[results['isotope']].mass / self.physical_constants['amu']
        ]
        tunneling_probs = [results['base_tunneling_prob'], results['isotope_tunneling_prob']]
        
        # Plot 1: Mass vs tunneling probability
        ax1.bar(particles, masses, alpha=0.7, color='blue', label='Mass (amu)')
        ax1.set_ylabel('Mass (amu)')
        ax1.set_title('Isotope Mass Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add tunneling probabilities on secondary axis
        ax1_twin = ax1.twinx()
        ax1_twin.plot(particles, tunneling_probs, 'ro-', linewidth=2, markersize=8, label='Tunneling Prob')
        ax1_twin.set_ylabel('Tunneling Probability')
        ax1_twin.set_yscale('log')
        
        # Plot 2: KIE comparison
        kie_values = [1.0, results['kinetic_isotope_effect']]
        classical_kie = [1.0, results['classical_kie']]
        
        x = np.arange(len(particles))
        width = 0.35
        
        ax2.bar(x - width/2, kie_values, width, label='Quantum KIE', alpha=0.8, color='red')
        ax2.bar(x + width/2, classical_kie, width, label='Classical KIE', alpha=0.8, color='blue')
        
        ax2.set_xlabel('Isotope')
        ax2.set_ylabel('Kinetic Isotope Effect')
        ax2.set_title(f'KIE Comparison (Enhancement: {results["quantum_enhancement"]:.1f}x)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(particles)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'quantum_isotope_effect.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_multi_barrier_plot(self, results: Dict[str, Any]) -> str:
        """Create multi-barrier tunneling visualization"""
        if 'individual_transmissions' not in results:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Individual barrier transmissions
        barrier_nums = list(range(1, len(results['individual_transmissions']) + 1))
        individual_trans = results['individual_transmissions']
        
        ax1.bar(barrier_nums, individual_trans, alpha=0.7, color='blue')
        ax1.set_xlabel('Barrier Number')
        ax1.set_ylabel('Transmission Coefficient')
        ax1.set_title('Individual Barrier Transmissions')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Overall vs equivalent single barrier
        comparison_labels = ['Multiple Barriers', 'Equivalent Single', 'Resonance Enhanced']
        comparison_values = [
            results['total_tunneling_probability'],
            results['equivalent_single_barrier'],
            results['resonance_enhanced_total']
        ]
        
        colors = ['blue', 'red', 'green']
        bars = ax2.bar(comparison_labels, comparison_values, color=colors, alpha=0.7)
        ax2.set_ylabel('Tunneling Probability')
        ax2.set_title('Multi-Barrier vs Single Barrier Comparison')
        ax2.set_yscale('log')
        
        # Add value labels on bars
        for bar, val in zip(bars, comparison_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filename = 'quantum_multi_barrier.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename

def create_quantum_tunneling_workshop():
    """Create and initialize the quantum tunneling workshop"""
    workshop = QuantumTunnelingWorkshop()
    
    print("Quantum Tunneling Workshop Initialized")
    print("=" * 50)
    print("Available particles:", list(workshop.common_particles.keys()))
    print("Barrier shapes:", [bs.value for bs in BarrierShape])
    print("Tunneling regimes:", [tr.value for tr in TunnelingRegime])
    
    return workshop

if __name__ == "__main__":
    # Demo the workshop
    workshop = create_quantum_tunneling_workshop()
    
    # Example 1: Basic tunneling calculation
    print("\n1. Calculating hydrogen tunneling through 0.2 eV barrier...")
    result1 = workshop.calculate_wkb_tunneling_probability(
        'hydrogen', 0.2 * workshop.physical_constants['e'], 1e-10, 0.05 * workshop.physical_constants['e']
    )
    print(f"Tunneling probability: {result1['tunneling_probability']:.2e}")
    print(f"Quantum enhancement: {result1['quantum_enhancement_factor']:.1f}x")
    
    # Example 2: Isotope effect
    print("\n2. Isotope effect between H and D at 100K...")
    result2 = workshop.calculate_isotope_effect(
        'hydrogen', 'deuterium', 0.3 * workshop.physical_constants['e'], 1e-10, 100
    )
    print(f"KIE: {result2['kinetic_isotope_effect']:.1f}")
    print(f"Quantum enhancement: {result2['quantum_enhancement']:.1f}x")
    
    # Example 3: Temperature dependence
    print("\n3. Temperature dependence study...")
    result3 = workshop.temperature_dependent_tunneling(
        'hydrogen', 0.25 * workshop.physical_constants['e'], 1.2e-10, (10, 500), 30
    )
    print(f"Crossover temperature: {result3['crossover_temperature']:.1f} K")
    print(f"Quantum dominant fraction: {result3['quantum_dominant_fraction']:.1%}")
    
    # Example 4: Diffusion coefficient
    print("\n4. Tunneling diffusion coefficient at 200K...")
    result4 = workshop.calculate_tunneling_diffusion_coefficient(
        'hydrogen', 0.3 * workshop.physical_constants['e'], 1e-10, 200
    )
    print(f"Diffusion coefficient: {result4['diffusion_coefficient']:.2e} m²/s")
    print(f"Quantum enhancement: {result4['diffusion_enhancement']:.1f}x")
    
    # Example 5: Visualizations
    print("\n5. Creating visualizations...")
    prob_plot = workshop.create_tunneling_visualization(result1, 'probability')
    temp_plot = workshop.create_tunneling_visualization(result3, 'temperature')
    iso_plot = workshop.create_tunneling_visualization(result2, 'isotope')
    
    print(f"Probability plot: {prob_plot}")
    print(f"Temperature plot: {temp_plot}")
    print(f"Isotope plot: {iso_plot}")
    
    print("\nQuantum tunneling workshop demo completed successfully!")
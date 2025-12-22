"""
Graphene Diffusion Workshop
Advanced workshop on diffusion in 2D materials and graphene

Workshop ID: GDW-001
Category: Advanced Nanomaterials
Difficulty: Expert
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
from enum import Enum

class GrapheneDefectType(Enum):
    """Types of defects in graphene"""
    PRISTINE = "pristine"
    VACANCY = "vacancy"
    STONE_WALES = "stone_wales"
    GRAIN_BOUNDARY = "grain_boundary"
    FUNCTIONALIZED = "functionalized"
    DOPED = "doped"

class DiffusionDirection(Enum):
    """Diffusion directions in graphene"""
    ALONG_ARMCHAIR = "along_armchair"
    ALONG_ZIGZAG = "along_zigzag"
    DIAGONAL = "diagonal"
    ISOTROPIC = "isotropic"

@dataclass
class GrapheneProperties:
    """Physical properties of graphene"""
    lattice_constant: float = 2.46e-10  # meters
    carbon_carbon_bond: float = 1.42e-10  # meters
    thickness: float = 3.35e-10  # meters (effective)
    density: float = 2260  # kg/m³
    thermal_conductivity: float = 5000  # W/(m·K)
    electrical_conductivity: float = 1e8  # S/m
    young_modulus: float = 1e12  # Pa
    band_gap: float = 0  # eV (pristine graphene)

class GrapheneDiffusionWorkshop:
    """
    Comprehensive workshop on diffusion in graphene and 2D materials.
    
    This workshop covers:
    - Pristine graphene diffusion mechanisms
    - Defect-enhanced diffusion
    - Functionalized graphene properties
    - Anisotropic diffusion patterns
    - Temperature-dependent behavior
    - Quantum confinement effects
    - Comparison with other 2D materials
    """
    
    def __init__(self):
        """Initialize the graphene diffusion workshop"""
        self.properties = GrapheneProperties()
        self.defect_parameters = self._initialize_defect_parameters()
        self.diffusion_database = self._initialize_diffusion_database()
        self.simulation_history = []
        
    def _initialize_defect_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize parameters for different defect types"""
        return {
            GrapheneDefectType.PRISTINE: {
                'description': 'Perfect hexagonal lattice with no defects',
                'diffusion_enhancement': 1.0,
                'activation_energy': 0.5,  # eV
                'prefactor': 1e-3,  # cm²/s
                'barrier_height': 0.2,  # eV
                'quantum_tunneling_probability': 0.01
            },
            GrapheneDefectType.VACANCY: {
                'description': 'Missing carbon atom in the lattice',
                'diffusion_enhancement': 10.0,
                'activation_energy': 0.3,  # eV
                'prefactor': 1e-2,  # cm²/s
                'barrier_height': 0.1,  # eV
                'quantum_tunneling_probability': 0.1
            },
            GrapheneDefectType.STONE_WALES: {
                'description': '55-777 ring transformation',
                'diffusion_enhancement': 5.0,
                'activation_energy': 0.4,  # eV
                'prefactor': 5e-3,  # cm²/s
                'barrier_height': 0.15,  # eV
                'quantum_tunneling_probability': 0.05
            },
            GrapheneDefectType.GRAIN_BOUNDARY: {
                'description': 'Boundary between different crystal orientations',
                'diffusion_enhancement': 50.0,
                'activation_energy': 0.2,  # eV
                'prefactor': 1e-1,  # cm²/s
                'barrier_height': 0.05,  # eV
                'quantum_tunneling_probability': 0.2
            },
            GrapheneDefectType.FUNCTIONALIZED: {
                'description': 'Chemical functionalization of surface',
                'diffusion_enhancement': 0.1,
                'activation_energy': 0.8,  # eV
                'prefactor': 1e-4,  # cm²/s
                'barrier_height': 0.4,  # eV
                'quantum_tunneling_probability': 0.001
            },
            GrapheneDefectType.DOPED: {
                'description': 'Doping with heteroatoms',
                'diffusion_enhancement': 2.0,
                'activation_energy': 0.45,  # eV
                'prefactor': 2e-3,  # cm²/s
                'barrier_height': 0.25,  # eV
                'quantum_tunneling_probability': 0.02
            }
        }
    
    def _initialize_diffusion_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize experimental diffusion data"""
        return {
            'hydrogen': {
                'mass': 1.008,  # amu
                'pristine_diffusion': {
                    'D0': 1e-3,  # cm²/s
                    'Ea': 0.5,   # eV
                    'experimental_range': (1e-10, 1e-6)  # cm²/s
                },
                'temperature_range': (200, 1000),  # K
                'references': ['Nair et al., Science 2008', 'Bunch et al., Science 2008']
            },
            'lithium': {
                'mass': 6.94,  # amu
                'pristine_diffusion': {
                    'D0': 5e-4,  # cm²/s
                    'Ea': 0.3,   # eV
                    'experimental_range': (1e-11, 1e-7)  # cm²/s
                },
                'temperature_range': (300, 800),  # K
                'references': ['Liu et al., Nat. Mater. 2011', 'Persson et al., Nano Lett. 2010']
            },
            'oxygen': {
                'mass': 16.00,  # amu
                'pristine_diffusion': {
                    'D0': 2e-4,  # cm²/s
                    'Ea': 0.6,   # eV
                    'experimental_range': (1e-12, 1e-8)  # cm²/s
                },
                'temperature_range': (400, 1200),  # K
                'references': ['Lehtinen et al., Phys. Rev. B 2010', 'Jain et al., ACS Nano 2010']
            },
            'copper': {
                'mass': 63.55,  # amu
                'pristine_diffusion': {
                    'D0': 1e-5,  # cm²/s
                    'Ea': 0.8,   # eV
                    'experimental_range': (1e-15, 1e-10)  # cm²/s
                },
                'temperature_range': (500, 1500),  # K
                'references': ['Zhou et al., J. Phys. Chem. C 2011', 'Panchakarla et al., Adv. Mater. 2009']
            }
        }
    
    def calculate_pristine_graphene_diffusion(self,
                                            species: str,
                                            temperature: float,
                                            direction: DiffusionDirection = DiffusionDirection.ISOTROPIC) -> Dict[str, Any]:
        """
        Calculate diffusion in pristine graphene
        
        Args:
            species: Diffusing species ('hydrogen', 'lithium', 'oxygen', 'copper')
            temperature: Temperature in Kelvin
            direction: Diffusion direction
            
        Returns:
            Dictionary with diffusion calculation results
        """
        if species not in self.diffusion_database:
            raise ValueError(f"Species {species} not in database")
        
        species_data = self.diffusion_database[species]
        diffusion_params = species_data['pristine_diffusion']
        
        # Arrhenius equation: D = D0 * exp(-Ea / kB*T)
        kB_eV = 8.617333262e-5  # Boltzmann constant in eV/K
        
        D0 = diffusion_params['D0']  # cm²/s
        Ea = diffusion_params['Ea']   # eV
        
        D_cm2_s = D0 * np.exp(-Ea / (kB_eV * temperature))
        D_m2_s = D_cm2_s * 1e-4  # Convert to m²/s
        
        # Directional anisotropy
        anisotropy_factor = self._get_anisotropy_factor(direction)
        D_directional = D_m2_s * anisotropy_factor
        
        # Quantum corrections for light species
        quantum_correction = self._calculate_quantum_correction(species, temperature)
        D_quantum = D_directional * quantum_correction
        
        # Calculate mean square displacement
        time_1s = 1.0  # 1 second
        msd_1s = 4 * D_quantum * time_1s  # 2D diffusion
        
        # Characteristic diffusion length
        time_1hour = 3600  # 1 hour
        diffusion_length = np.sqrt(4 * D_quantum * time_1hour)
        
        # Compare with bulk values
        bulk_diffusion = self._get_bulk_diffusion_reference(species, temperature)
        enhancement_factor = D_quantum / bulk_diffusion if bulk_diffusion > 0 else float('inf')
        
        results = {
            'species': species,
            'graphene_type': GrapheneDefectType.PRISTINE.value,
            'temperature': temperature,
            'direction': direction.value,
            'diffusion_coefficient': D_quantum,
            'diffusion_coefficient_cm2_s': D_cm2_s * quantum_correction,
            'anisotropy_factor': anisotropy_factor,
            'quantum_correction': quantum_correction,
            'mean_square_displacement_1s': msd_1s,
            'diffusion_length_1hr': diffusion_length,
            'bulk_diffusion_reference': bulk_diffusion,
            'enhancement_factor': enhancement_factor,
            'activation_energy': Ea,
            'prefactor': D0,
            'experimental_range': species_data['experimental_range'],
            'within_experimental_range': species_data['experimental_range'][0] <= D_cm2_s <= species_data['experimental_range'][1]
        }
        
        return results
    
    def calculate_defect_enhanced_diffusion(self,
                                          species: str,
                                          temperature: float,
                                          defect_type: GrapheneDefectType,
                                          defect_density: float,
                                          direction: DiffusionDirection = DiffusionDirection.ISOTROPIC) -> Dict[str, Any]:
        """
        Calculate diffusion with defect enhancement
        
        Args:
            species: Diffusing species
            temperature: Temperature in Kelvin
            defect_type: Type of defect
            defect_density: Defect density (defects/cm²)
            direction: Diffusion direction
            
        Returns:
            Dictionary with defect-enhanced diffusion results
        """
        # Start with pristine calculation
        pristine_results = self.calculate_pristine_graphene_diffusion(species, temperature, direction)
        
        # Get defect parameters
        defect_params = self.defect_parameters[defect_type]
        enhancement_factor = defect_params['diffusion_enhancement']
        
        # Apply density-dependent enhancement
        # Higher defect density increases diffusion up to a saturation point
        saturation_density = 1e13  # defects/cm² (typical saturation)
        density_factor = 1 + (enhancement_factor - 1) * (1 - np.exp(-defect_density / saturation_density))
        
        # Modified activation energy due to defects
        Ea_pristine = pristine_results['activation_energy']
        Ea_defect = defect_params['activation_energy']
        
        # Weighted average based on defect density
        defect_fraction = min(defect_density / saturation_density, 1.0)
        Ea_effective = Ea_pristine * (1 - defect_fraction) + Ea_defect * defect_fraction
        
        # Recalculate diffusion with modified parameters
        kB_eV = 8.617333262e-5  # eV/K
        D0_defect = defect_params['prefactor']
        D0_effective = pristine_results['prefactor'] * (1 - defect_fraction) + D0_defect * defect_fraction
        
        D_defect = D0_effective * np.exp(-Ea_effective / (kB_eV * temperature))
        D_defect_m2_s = D_defect * 1e-4 * density_factor
        
        # Quantum tunneling contribution for defects
        tunneling_probability = defect_params['quantum_tunneling_probability']
        tunneling_enhancement = 1 + tunneling_probability * np.exp(-defect_params['barrier_height'] / (kB_eV * temperature))
        
        D_final = D_defect_m2_s * tunneling_enhancement
        
        # Calculate additional metrics
        defect_contribution = (D_final - pristine_results['diffusion_coefficient']) / D_final * 100
        
        results = {
            **pristine_results,
            'graphene_type': defect_type.value,
            'defect_density': defect_density,
            'defect_fraction': defect_fraction,
            'effective_activation_energy': Ea_effective,
            'effective_prefactor': D0_effective,
            'density_factor': density_factor,
            'tunneling_enhancement': tunneling_enhancement,
            'diffusion_coefficient': D_final,
            'diffusion_coefficient_cm2_s': D_final * 1e4,
            'defect_contribution_percent': defect_contribution,
            'enhancement_vs_pristine': D_final / pristine_results['diffusion_coefficient'],
            'defect_parameters': defect_params
        }
        
        return results
    
    def _get_anisotropy_factor(self, direction: DiffusionDirection) -> float:
        """Get anisotropy factor for different directions"""
        anisotropy_factors = {
            DiffusionDirection.ALONG_ARMCHAIR: 1.0,  # Reference
            DiffusionDirection.ALONG_ZIGZAG: 0.8,   # 20% lower
            DiffusionDirection.DIAGONAL: 0.9,      # 10% lower
            DiffusionDirection.ISOTROPIC: 1.0      # Average
        }
        return anisotropy_factors.get(direction, 1.0)
    
    def _calculate_quantum_correction(self, species: str, temperature: float) -> float:
        """Calculate quantum correction factor for light species"""
        # Debye temperature for graphene (simplified)
        debye_temp = 2300  # K
        
        # Mass-dependent quantum effects
        mass = self.diffusion_database[species]['mass']
        
        if temperature > debye_temp:
            # Classical regime
            return 1.0
        elif mass < 5:  # Light species (hydrogen, lithium)
            # Quantum tunneling enhancement
            tunneling_factor = 1 + 0.5 * np.exp(-temperature / 300)
            return tunneling_factor
        else:
            # Heavy species - minimal quantum effects
            return 1.0
    
    def _get_bulk_diffusion_reference(self, species: str, temperature: float) -> float:
        """Get bulk diffusion coefficient for comparison"""
        # Simplified bulk diffusion references
        bulk_references = {
            'hydrogen': 1e-10,  # m²/s in bulk materials
            'lithium': 1e-12,
            'oxygen': 1e-13,
            'copper': 1e-14
        }
        return bulk_references.get(species, 1e-12)
    
    def compare_graphene_types(self,
                             species: str,
                             temperature: float,
                             defect_density: float = 1e12) -> Dict[str, Any]:
        """
        Compare diffusion across different graphene types
        
        Args:
            species: Diffusing species
            temperature: Temperature in Kelvin
            defect_density: Defect density for defect types
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        # Calculate for all graphene types
        for defect_type in GrapheneDefectType:
            if defect_type == GrapheneDefectType.PRISTINE:
                result = self.calculate_pristine_graphene_diffusion(species, temperature)
            else:
                result = self.calculate_defect_enhanced_diffusion(
                    species, temperature, defect_type, defect_density
                )
            results[defect_type.value] = result
        
        # Find fastest and slowest diffusion
        diffusion_values = {k: v['diffusion_coefficient'] for k, v in results.items()}
        fastest_type = max(diffusion_values, key=diffusion_values.get)
        slowest_type = min(diffusion_values, key=diffusion_values.get)
        
        enhancement_range = diffusion_values[fastest_type] / diffusion_values[slowest_type]
        
        # Analyze trends
        activation_energies = {k: v['activation_energy'] for k, v in results.items()}
        correlation_analysis = self._analyze_correlation(diffusion_values, activation_energies)
        
        comparison = {
            'species': species,
            'temperature': temperature,
            'defect_density': defect_density,
            'diffusion_coefficients': diffusion_values,
            'activation_energies': activation_energies,
            'fastest_type': fastest_type,
            'slowest_type': slowest_type,
            'enhancement_range': enhancement_range,
            'correlation_analysis': correlation_analysis,
            'recommendations': self._generate_comparison_recommendations(results),
            'detailed_results': results
        }
        
        return comparison
    
    def _analyze_correlation(self, x_values: Dict[str, float], y_values: Dict[str, float]) -> Dict[str, Any]:
        """Analyze correlation between two datasets"""
        x = list(x_values.values())
        y = list(y_values.values())
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Linear regression (log-log scale for diffusion analysis)
        log_x = np.log10(x)
        log_y = np.log10(y)
        
        # Fit linear relationship
        coeffs = np.polyfit(log_x, log_y, 1)
        slope, intercept = coeffs
        
        # Calculate R-squared
        y_pred = slope * log_x + intercept
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'correlation_coefficient': correlation,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'relationship': 'inverse' if slope < 0 else 'direct',
            'strength': 'strong' if abs(correlation) > 0.8 else 'moderate' if abs(correlation) > 0.5 else 'weak'
        }
    
    def _generate_comparison_recommendations(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on comparison analysis"""
        recommendations = []
        
        # Find optimal conditions
        fastest = max(results.values(), key=lambda x: x['diffusion_coefficient'])
        slowest = min(results.values(), key=lambda x: x['diffusion_coefficient'])
        
        if fastest['graphene_type'] != 'pristine':
            recommendations.append(
                f"For maximum diffusion rate, use {fastest['graphene_type']} with "
                f"{fastest.get('enhancement_vs_pristine', 1):.1f}x enhancement over pristine"
            )
        
        if slowest['graphene_type'] == 'functionalized':
            recommendations.append(
                "Functionalization reduces diffusion - suitable for barrier applications"
            )
        
        # Temperature recommendations
        temp_values = [r['temperature'] for r in results.values()]
        avg_temp = np.mean(temp_values)
        
        if avg_temp < 500:
            recommendations.append("Consider increasing temperature for enhanced diffusion rates")
        elif avg_temp > 1000:
            recommendations.append("High temperature may cause structural instability")
        
        # Application-specific recommendations
        if any('grain_boundary' in r['graphene_type'] for r in results.values()):
            recommendations.append("Grain boundaries provide maximum enhancement but reduce mechanical strength")
        
        return recommendations
    
    def create_temperature_dependence_study(self,
                                          species: str,
                                          graphene_type: GrapheneDefectType,
                                          temperature_range: Tuple[float, float],
                                          num_points: int = 50) -> Dict[str, Any]:
        """
        Create comprehensive temperature dependence study
        
        Args:
            species: Diffusing species
            graphene_type: Type of graphene
            temperature_range: Temperature range (min, max) in Kelvin
            num_points: Number of temperature points
            
        Returns:
            Dictionary with temperature dependence analysis
        """
        temperatures = np.linspace(temperature_range[0], temperature_range[1], num_points)
        
        diffusion_data = []
        activation_data = []
        
        for temp in temperatures:
            if graphene_type == GrapheneDefectType.PRISTINE:
                result = self.calculate_pristine_graphene_diffusion(species, temp)
            else:
                result = self.calculate_defect_enhanced_diffusion(
                    species, temp, graphene_type, 1e12
                )
            
            diffusion_data.append(result['diffusion_coefficient'])
            activation_data.append(result['activation_energy'])
        
        diffusion_data = np.array(diffusion_data)
        activation_data = np.array(activation_data)
        
        # Arrhenius analysis
        kB_eV = 8.617333262e-5
        inv_temp = 1 / (kB_eV * temperatures)
        log_diffusion = np.log10(diffusion_data)
        
        # Linear fit to Arrhenius equation
        arrhenius_coeffs = np.polyfit(inv_temp, log_diffusion, 1)
        arrhenius_slope, arrhenius_intercept = arrhenius_coeffs
        
        # Extract fitted parameters
        fitted_Ea = -arrhenius_slope * np.log(10)  # Convert from log10 to ln
        fitted_D0 = 10 ** arrhenius_intercept
        
        # Calculate R-squared
        y_pred = arrhenius_slope * inv_temp + arrhenius_intercept
        ss_res = np.sum((log_diffusion - y_pred) ** 2)
        ss_tot = np.sum((log_diffusion - np.mean(log_diffusion)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Find characteristic temperatures
        target_diffusion = 1e-10  # m²/s
        char_temp_idx = np.argmin(np.abs(diffusion_data - target_diffusion))
        characteristic_temp = temperatures[char_temp_idx]
        
        # Temperature sensitivity analysis
        low_temp_diff = np.mean(diffusion_data[:5])
        high_temp_diff = np.mean(diffusion_data[-5:])
        temp_sensitivity = (high_temp_diff - low_temp_diff) / low_temp_diff * 100
        
        study_results = {
            'species': species,
            'graphene_type': graphene_type.value,
            'temperature_range': temperature_range,
            'temperatures': temperatures.tolist(),
            'diffusion_coefficients': diffusion_data.tolist(),
            'activation_energies': activation_data.tolist(),
            'arrhenius_fit': {
                'slope': arrhenius_slope,
                'intercept': arrhenius_intercept,
                'fitted_Ea': fitted_Ea,
                'fitted_D0': fitted_D0,
                'r_squared': r_squared
            },
            'characteristic_temperature': characteristic_temp,
            'temperature_sensitivity_percent': temp_sensitivity,
            'min_diffusion': np.min(diffusion_data),
            'max_diffusion': np.max(diffusion_data),
            'diffusion_range': np.max(diffusion_data) / np.min(diffusion_data),
            'analysis': {
                'arrhenius_behavior': 'good' if r_squared > 0.95 else 'moderate' if r_squared > 0.8 else 'poor',
                'temperature_range_adequate': temp_sensitivity > 100,
                'recommendations': self._generate_temp_study_recommendations(
                    temp_sensitivity, r_squared, graphene_type
                )
            }
        }
        
        return study_results
    
    def _generate_temp_study_recommendations(self,
                                           temp_sensitivity: float,
                                           r_squared: float,
                                           graphene_type: GrapheneDefectType) -> List[str]:
        """Generate recommendations for temperature study"""
        recommendations = []
        
        if temp_sensitivity < 50:
            recommendations.append("Low temperature sensitivity - consider wider temperature range")
        elif temp_sensitivity > 1000:
            recommendations.append("High temperature sensitivity - ensure thermal stability")
        else:
            recommendations.append("Good temperature sensitivity range")
        
        if r_squared < 0.8:
            recommendations.append("Poor Arrhenius fit - check for multiple diffusion mechanisms")
        elif r_squared > 0.95:
            recommendations.append("Excellent Arrhenius behavior - single mechanism dominant")
        else:
            recommendations.append("Moderate Arrhenius fit - multiple mechanisms may be present")
        
        if graphene_type != GrapheneDefectType.PRISTINE:
            recommendations.append("Defect-enhanced diffusion may show non-Arrhenius behavior at low temperatures")
        
        return recommendations
    
    def visualize_graphene_diffusion(self,
                                   results: Dict[str, Any],
                                   plot_type: str = 'comprehensive') -> str:
        """
        Create visualization of graphene diffusion results
        
        Args:
            results: Diffusion calculation results
            plot_type: Type of plot ('comprehensive', 'arrhenius', 'comparison', 'defect_effect')
            
        Returns:
            Path to saved plot
        """
        if plot_type == 'comprehensive':
            return self._create_comprehensive_plot(results)
        elif plot_type == 'arrhenius':
            return self._create_arrhenius_plot(results)
        elif plot_type == 'comparison':
            return self._create_comparison_plot(results)
        elif plot_type == 'defect_effect':
            return self._create_defect_effect_plot(results)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _create_comprehensive_plot(self, results: Dict[str, Any]) -> str:
        """Create comprehensive multi-panel visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Graphene Diffusion Analysis - {results.get("species", "Unknown")}', fontsize=16)
        
        # Plot 1: Diffusion coefficient comparison
        ax1 = axes[0, 0]
        if 'diffusion_coefficients' in results:
            types = list(results['diffusion_coefficients'].keys())
            values = list(results['diffusion_coefficients'].values())
            bars = ax1.bar(types, values)
            ax1.set_ylabel('Diffusion Coefficient (m²/s)')
            ax1.set_title('Diffusion by Graphene Type')
            ax1.set_yscale('log')
            ax1.tick_params(axis='x', rotation=45)
            
            # Color bars based on enhancement
            for bar, val in zip(bars, values):
                if val > 1e-8:
                    bar.set_color('red')
                elif val > 1e-10:
                    bar.set_color('orange')
                else:
                    bar.set_color('blue')
        
        # Plot 2: Activation energy
        ax2 = axes[0, 1]
        if 'activation_energies' in results:
            types = list(results['activation_energies'].keys())
            values = list(results['activation_energies'].values())
            ax2.bar(types, values, color='green', alpha=0.7)
            ax2.set_ylabel('Activation Energy (eV)')
            ax2.set_title('Activation Energy by Type')
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Enhancement factor
        ax3 = axes[1, 0]
        if 'detailed_results' in results:
            pristine_D = results['detailed_results']['pristine']['diffusion_coefficient']
            enhancements = {}
            for graphene_type, data in results['detailed_results'].items():
                enhancements[graphene_type] = data['diffusion_coefficient'] / pristine_D
            
            types = list(enhancements.keys())
            values = list(enhancements.values())
            bars = ax3.bar(types, values)
            ax3.set_ylabel('Enhancement Factor vs Pristine')
            ax3.set_title('Diffusion Enhancement')
            ax3.set_yscale('log')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add reference line at 1
            ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
            
            # Color bars
            for bar, val in zip(bars, values):
                if val > 10:
                    bar.set_color('darkred')
                elif val > 5:
                    bar.set_color('orange')
                else:
                    bar.set_color('lightblue')
        
        # Plot 4: Temperature dependence (if available)
        ax4 = axes[1, 1]
        if 'temperatures' in results:
            temps = results['temperatures']
            diffs = results['diffusion_coefficients']
            ax4.plot(temps, diffs, 'b-', linewidth=2, marker='o')
            ax4.set_xlabel('Temperature (K)')
            ax4.set_ylabel('Diffusion Coefficient (m²/s)')
            ax4.set_title('Temperature Dependence')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Temperature data\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Temperature Dependence')
        
        plt.tight_layout()
        
        filename = 'graphene_diffusion_comprehensive.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_arrhenius_plot(self, results: Dict[str, Any]) -> str:
        """Create Arrhenius plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if 'temperatures' in results and 'diffusion_coefficients' in results:
            temps = np.array(results['temperatures'])
            diffs = np.array(results['diffusion_coefficients'])
            
            # Arrhenius plot
            kB_eV = 8.617333262e-5
            inv_temp = 1000 / (kB_eV * temps)  # in 1000/(eV·K)
            log_diff = np.log10(diffs)
            
            ax.plot(inv_temp, log_diff, 'bo-', linewidth=2, markersize=6)
            
            # Add fit line
            if 'arrhenius_fit' in results:
                fit = results['arrhenius_fit']
                y_fit = fit['slope'] * inv_temp + fit['intercept']
                ax.plot(inv_temp, y_fit, 'r--', linewidth=2, 
                       label=f'Fit: R²={fit["r_squared"]:.3f}')
                
                # Add equation
                equation = f'log₁₀(D) = {fit["slope"]:.2f}·(1000/kT) + {fit["intercept"]:.2f}'
                ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
                       bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('1000/kT (1000/eV·K)')
            ax.set_ylabel('log₁₀(D) (m²/s)')
            ax.set_title(f'Arrhenius Plot - {results.get("species", "Unknown")} in {results.get("graphene_type", "Unknown")}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        filename = 'graphene_arrhenius_plot.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_comparison_plot(self, results: Dict[str, Any]) -> str:
        """Create comparison plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if 'diffusion_coefficients' in results and 'activation_energies' in results:
            types = list(results['diffusion_coefficients'].keys())
            diffs = list(results['diffusion_coefficients'].values())
            energies = list(results['activation_energies'].values())
            
            # Normalized values for comparison
            norm_diffs = np.array(diffs) / np.max(diffs)
            norm_energies = np.array(energies) / np.max(energies)
            
            # Plot normalized values
            x = np.arange(len(types))
            width = 0.35
            
            ax1.bar(x - width/2, norm_diffs, width, label='Diffusion Coefficient', alpha=0.8)
            ax1.bar(x + width/2, 1 - norm_energies, width, label='Inverse Activation Energy', alpha=0.8)
            
            ax1.set_xlabel('Graphene Type')
            ax1.set_ylabel('Normalized Value')
            ax1.set_title('Normalized Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(types, rotation=45)
            ax1.legend()
            
            # Scatter plot of correlation
            ax2.scatter(energies, diffs, s=100, alpha=0.7)
            ax2.set_xlabel('Activation Energy (eV)')
            ax2.set_ylabel('Diffusion Coefficient (m²/s)')
            ax2.set_title('Diffusion vs Activation Energy')
            ax2.set_yscale('log')
            
            # Add labels
            for i, graphene_type in enumerate(types):
                ax2.annotate(graphene_type[:8], (energies[i], diffs[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        filename = 'graphene_comparison_plot.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_defect_effect_plot(self, results: Dict[str, Any]) -> str:
        """Create defect effect visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if 'defect_contribution_percent' in results:
            # Pie chart of defect contribution
            pristine_contrib = 100 - results['defect_contribution_percent']
            defect_contrib = results['defect_contribution_percent']
            
            sizes = [pristine_contrib, defect_contrib]
            labels = ['Pristine Contribution', 'Defect Enhancement']
            colors = ['lightblue', 'orange']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'Diffusion Contribution - {results.get("graphene_type", "Unknown")}')
        
        if 'enhancement_vs_pristine' in results:
            # Enhancement factor bar
            enhancement = results['enhancement_vs_pristine']
            
            ax2.bar(['Enhancement\nFactor'], [enhancement], color='red', alpha=0.7)
            ax2.set_ylabel('Enhancement Factor')
            ax2.set_title(f'Enhancement vs Pristine: {enhancement:.1f}x')
            ax2.set_yscale('log')
            
            # Add reference lines
            ax2.axhline(y=1, color='blue', linestyle='--', alpha=0.5, label='Pristine (1x)')
            ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10x Enhancement')
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100x Enhancement')
            ax2.legend()
        
        plt.tight_layout()
        
        filename = 'graphene_defect_effect.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename

def create_graphene_diffusion_workshop():
    """Create and initialize the graphene diffusion workshop"""
    workshop = GrapheneDiffusionWorkshop()
    
    print("Graphene Diffusion Workshop Initialized")
    print("=" * 50)
    print("Available species:", list(workshop.diffusion_database.keys()))
    print("Defect types:", [dt.value for dt in GrapheneDefectType])
    print("Diffusion directions:", [dd.value for dd in DiffusionDirection])
    
    return workshop

if __name__ == "__main__":
    # Demo the workshop
    workshop = create_graphene_diffusion_workshop()
    
    # Example 1: Pristine graphene diffusion
    print("\n1. Calculating hydrogen diffusion in pristine graphene at 300K...")
    result1 = workshop.calculate_pristine_graphene_diffusion('hydrogen', 300)
    print(f"Diffusion coefficient: {result1['diffusion_coefficient']:.2e} m²/s")
    print(f"Enhancement vs bulk: {result1['enhancement_factor']:.1f}x")
    
    # Example 2: Defect-enhanced diffusion
    print("\n2. Calculating lithium diffusion with vacancy defects...")
    result2 = workshop.calculate_defect_enhanced_diffusion(
        'lithium', 500, GrapheneDefectType.VACANCY, 1e12
    )
    print(f"Diffusion coefficient: {result2['diffusion_coefficient']:.2e} m²/s")
    print(f"Defect contribution: {result2['defect_contribution_percent']:.1f}%")
    
    # Example 3: Compare all graphene types
    print("\n3. Comparing all graphene types for oxygen diffusion...")
    comparison = workshop.compare_graphene_types('oxygen', 600)
    print(f"Fastest: {comparison['fastest_type']}")
    print(f"Enhancement range: {comparison['enhancement_range']:.1f}x")
    
    # Example 4: Temperature dependence study
    print("\n4. Temperature dependence study for copper...")
    temp_study = workshop.create_temperature_dependence_study(
        'copper', GrapheneDefectType.GRAIN_BOUNDARY, (400, 1200), 30
    )
    print(f"Arrhenius fit R²: {temp_study['arrhenius_fit']['r_squared']:.3f}")
    
    # Example 5: Create visualizations
    print("\n5. Creating visualizations...")
    comp_plot = workshop.visualize_graphene_diffusion(comparison, 'comprehensive')
    arr_plot = workshop.visualize_graphene_diffusion(temp_study, 'arrhenius')
    
    print(f"Comprehensive plot saved: {comp_plot}")
    print(f"Arrhenius plot saved: {arr_plot}")
    
    print("\nGraphene diffusion workshop demo completed successfully!")
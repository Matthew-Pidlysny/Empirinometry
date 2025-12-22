"""
Caelum-based Diffusion Modeling System
Core diffusion engine with step-by-step calculation visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial.distance import pdist, squareform
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

@dataclass
class DiffusionParameters:
    """Parameters for diffusion calculations"""
    diffusion_coefficient: float
    temperature: float
    time_steps: int
    spatial_resolution: int
    boundary_conditions: str = 'dirichlet'
    
@dataclass
class Material:
    """Material properties for diffusion"""
    name: str
    density: float
    molecular_weight: float
    diffusion_coefficient: float
    activation_energy: float
    atomic_radius: float
    crystal_structure: str
    
class CaelumDiffusionModel:
    """Core diffusion model based on Caelum framework"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calculation_history = []
        self.current_step = 0
        self.materials_db = self._initialize_materials_database()
        
    def _initialize_materials_database(self) -> Dict[str, Material]:
        """Initialize comprehensive materials database"""
        materials = {
            # Metals
            'Iron': Material('Iron', 7.87, 55.85, 2.4e-5, 2.48, 0.156, 'BCC'),
            'Aluminum': Material('Aluminum', 2.70, 26.98, 1.8e-4, 1.35, 0.143, 'FCC'),
            'Copper': Material('Copper', 8.96, 63.55, 7.8e-5, 2.19, 0.128, 'FCC'),
            'Gold': Material('Gold', 19.3, 196.97, 1.2e-4, 1.78, 0.144, 'FCC'),
            'Silver': Material('Silver', 10.5, 107.87, 1.7e-4, 1.92, 0.144, 'FCC'),
            
            # Semiconductors
            'Silicon': Material('Silicon', 2.33, 28.09, 1.2e-5, 3.66, 0.111, 'Diamond'),
            'Germanium': Material('Germanium', 5.32, 72.64, 5.8e-5, 2.85, 0.125, 'Diamond'),
            
            # Ceramics
            'Alumina': Material('Alumina', 3.95, 101.96, 3.1e-7, 5.21, 0.135, 'Hexagonal'),
            'Silica': Material('Silica', 2.65, 60.08, 1.3e-7, 5.82, 0.137, 'Trigonal'),
            
            # Polymers
            'Polyethylene': Material('Polyethylene', 0.94, 28.05, 1.1e-8, 0.45, 0.150, 'Amorphous'),
            'Polystyrene': Material('Polystyrene', 1.05, 104.15, 6.0e-9, 0.72, 0.175, 'Amorphous'),
            
            # Gases (at STP)
            'Hydrogen': Material('Hydrogen', 0.0899, 2.016, 6.1e-1, 0.04, 0.053, 'Diatomic'),
            'Oxygen': Material('Oxygen', 1.43, 32.00, 2.0e-1, 0.06, 0.060, 'Diatomic'),
            'Nitrogen': Material('Nitrogen', 1.25, 28.02, 1.8e-1, 0.07, 0.065, 'Diatomic'),
        }
        return materials
    
    def calculate_diffusion_coefficient(self, material: Material, temperature: float) -> float:
        """
        Calculate temperature-dependent diffusion coefficient using Arrhenius equation
        D = D0 * exp(-Qa / (R * T))
        
        Returns step-by-step calculation for educational visualization
        """
        R = 8.314  # Universal gas constant (J/mol·K)
        
        steps = []
        
        # Step 1: Convert activation energy
        Qa_joules = material.activation_energy * 1000  # Convert kJ/mol to J/mol
        steps.append({
            'step': 1,
            'description': 'Convert activation energy to Joules',
            'calculation': f'Qa = {material.activation_energy} kJ/mol × 1000 = {Qa_joules} J/mol',
            'result': Qa_joules
        })
        
        # Step 2: Calculate exponential term
        exponent = -Qa_joules / (R * temperature)
        exp_term = np.exp(exponent)
        steps.append({
            'step': 2,
            'description': 'Calculate exponential term',
            'calculation': f'exp(-Qa/(R×T)) = exp({exponent:.3f}) = {exp_term:.2e}',
            'result': exp_term
        })
        
        # Step 3: Final diffusion coefficient
        D_temp = material.diffusion_coefficient * exp_term
        steps.append({
            'step': 3,
            'description': 'Calculate temperature-dependent diffusion coefficient',
            'calculation': f'D = D0 × exp_term = {material.diffusion_coefficient:.2e} × {exp_term:.2e} = {D_temp:.2e} m²/s',
            'result': D_temp
        })
        
        return D_temp, steps
    
    def fick_second_law_1d(self, initial_condition: np.ndarray, params: DiffusionParameters, 
                          x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Solve Fick's second law in 1D with step-by-step visualization
        ∂C/∂t = D ∂²C/∂x²
        """
        D = params.diffusion_coefficient
        dx = x[1] - x[0]
        
        calculation_steps = []
        
        # Step 1: Setup initial conditions
        calculation_steps.append({
            'step': 1,
            'description': 'Setup spatial and temporal grids',
            'calculation': f'dx = {dx:.4f} m, Total points = {len(x)}',
            'result': initial_condition.copy()
        })
        
        # Step 2: Initialize solution array
        solution = np.zeros((len(t), len(x)))
        solution[0] = initial_condition
        calculation_steps.append({
            'step': 2,
            'description': 'Initialize solution matrix',
            'calculation': f'Solution shape: {solution.shape}',
            'result': solution[0].copy()
        })
        
        # Step 3: Time stepping
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            
            # Check stability (CFL condition)
            stability = D * dt / dx**2
            if stability > 0.5:
                self.logger.warning(f"Stability condition violated: {stability:.3f} > 0.5")
            
            # Calculate second derivative using central difference
            d2C_dx2 = np.zeros_like(solution[i-1])
            d2C_dx2[1:-1] = (solution[i-1, 2:] - 2*solution[i-1, 1:-1] + solution[i-1, :-2]) / dx**2
            
            # Apply boundary conditions
            if params.boundary_conditions == 'dirichlet':
                d2C_dx2[0] = 0  # Fixed concentration at boundaries
                d2C_dx2[-1] = 0
            
            # Update concentration
            solution[i] = solution[i-1] + dt * D * d2C_dx2
            
            if i % 10 == 0:  # Store every 10th step for visualization
                calculation_steps.append({
                    'step': i+2,
                    'description': f'Time step {i} at t = {t[i]:.3f} s',
                    'calculation': f'C_new = C_old + dt × D × ∂²C/∂x², Stability = {stability:.3f}',
                    'result': solution[i].copy()
                })
        
        return solution, calculation_steps
    
    def calculate_diffusion_sphere_properties(self, materials: List[str], temperatures: np.ndarray) -> Dict:
        """
        Calculate properties for the diffusion sphere visualization
        """
        sphere_data = {}
        
        for material_name in materials:
            if material_name in self.materials_db:
                material = self.materials_db[material_name]
                material_data = {
                    'name': material_name,
                    'diffusion_coefficients': [],
                    'temperatures': list(temperatures) if hasattr(temperatures, 'tolist') else temperatures,
                    'normalized_diffusion': []
                }
                
                for temp in temperatures:
                    D, _ = self.calculate_diffusion_coefficient(material, temp)
                    material_data['diffusion_coefficients'].append(D)
                
                # Normalize for visualization
                max_D = max(material_data['diffusion_coefficients'])
                if max_D > 0:
                    material_data['normalized_diffusion'] = [D/max_D for D in material_data['diffusion_coefficients']]
                
                sphere_data[material_name] = material_data
        
        return sphere_data
    
    def analyze_diffusion_clusters(self, diffusion_data: Dict, n_clusters: int = 5) -> Dict:
        """
        Analyze and cluster diffusion patterns for advanced visualization
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data for clustering
        features = []
        material_names = []
        
        for material_name, data in diffusion_data.items():
            if 'normalized_diffusion' in data:
                features.append(data['normalized_diffusion'])
                material_names.append(material_name)
        
        if len(features) == 0:
            return {'clusters': {}, 'analysis': 'No data available for clustering'}
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Organize results
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = {
                    'materials': [],
                    'center': kmeans.cluster_centers_[label].tolist(),
                    'characteristics': []
                }
            clusters[label]['materials'].append(material_names[i])
        
        # Analyze cluster characteristics
        for cluster_id, cluster_data in clusters.items():
            center = cluster_data['center']
            characteristics = []
            
            if len(center) > 0:
                if center[0] > 0.7:
                    characteristics.append('High diffusion at low temperature')
                elif center[0] < 0.3:
                    characteristics.append('Low diffusion at low temperature')
                
                if len(center) > 1 and center[-1] > center[0] * 2:
                    characteristics.append('Strong temperature dependence')
                elif len(center) > 1 and center[-1] < center[0] * 1.5:
                    characteristics.append('Weak temperature dependence')
            
            cluster_data['characteristics'] = characteristics
        
        if not material_names:
            return {'clusters': {}, 'n_clusters': 0, 'analysis': 'No materials to cluster'}
            
        return {
            'clusters': clusters,
            'n_clusters': len(clusters),
            'analysis': f'Successfully clustered {len(material_names)} materials into {len(clusters)} groups',
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
    
    def export_data(self, data: Dict, filename: str, format: str = 'json'):
        """Export data to specified format"""
        if format.lower() == 'json':
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == 'txt':
            with open(filename, 'w') as f:
                f.write("Diffusion Navigator Data Export\n")
                f.write("=" * 40 + "\n\n")
                for key, value in data.items():
                    f.write(f"{key}:\n")
                    f.write(f"{str(value)}\n\n")
        
        self.logger.info(f"Data exported to {filename}")
    
    def get_material_property(self, material_name: str, property_name: str):
        """Get specific material property"""
        if material_name in self.materials_db:
            material = self.materials_db[material_name]
            return getattr(material, property_name, None)
        return None
    
    def list_all_materials(self) -> List[str]:
        """Get list of all available materials"""
        return list(self.materials_db.keys())

class RelationalSphereModel:
    """Model for representing diffusion as layers on a relational sphere"""
    
    def __init__(self, n_layers: int = 10):
        self.n_layers = n_layers
        self.layer_data = {}
        self.sphere_radius = 1.0
        
    def generate_sphere_coordinates(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate coordinates for sphere visualization"""
        # Generate spherical coordinates
        theta = np.linspace(0, 2*np.pi, int(np.sqrt(n_points)))
        phi = np.linspace(0, np.pi, int(np.sqrt(n_points)))
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Convert to Cartesian
        R = self.sphere_radius
        X = R * np.sin(PHI) * np.cos(THETA)
        Y = R * np.sin(PHI) * np.sin(THETA)
        Z = R * np.cos(PHI)
        
        return X, Y, Z
    
    def assign_diffusion_to_layers(self, diffusion_data: Dict) -> Dict:
        """Assign diffusion data to spherical layers"""
        layer_assignments = {}
        
        for material_name, data in diffusion_data.items():
            if 'normalized_diffusion' in data:
                # Assign to layers based on diffusion characteristics
                diffusion_profile = data['normalized_diffusion']
                
                for i, layer_num in enumerate(range(self.n_layers)):
                    if layer_num not in layer_assignments:
                        layer_assignments[layer_num] = {}
                    
                    # Calculate layer-specific diffusion value
                    layer_fraction = layer_num / self.n_layers
                    if len(diffusion_profile) > 0:
                        # Use temperature-dependent assignment
                        temp_index = min(i, len(diffusion_profile) - 1)
                        layer_value = diffusion_profile[temp_index] * (1 + layer_fraction)
                    else:
                        layer_value = 0.5
                    
                    layer_assignments[layer_num][material_name] = layer_value
        
        return layer_assignments
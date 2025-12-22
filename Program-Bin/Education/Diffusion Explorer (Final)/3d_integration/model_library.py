"""
3D Model Library for Diffusion Visualization
Comprehensive library of pre-designed 3D models compatible with Blender and real-time engine
"""

import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import zipfile
import tempfile
from pathlib import Path

from blender_connector import DiffusionMeshData, BlenderDiffusionConnector
from realtime_3d_engine import RealTime3DEngine, RenderMode, DiffusionColormap

class ModelCategory(Enum):
    """Categories of 3D models"""
    BASIC_SHAPES = "basic_shapes"
    CRYSTAL_STRUCTURES = "crystal_structures"
    NANOMATERIALS = "nanomaterials"
    BIOLOGICAL = "biological"
    INDUSTRIAL = "industrial"
    EDUCATIONAL = "educational"
    RESEARCH = "research"

class MaterialType(Enum):
    """Material types for diffusion visualization"""
    METAL = "metal"
    SEMICONDUCTOR = "semiconductor"
    CERAMIC = "ceramic"
    POLYMER = "polymer"
    COMPOSITE = "composite"
    BIOLOGICAL = "biological"
    EXOTIC = "exotic"

@dataclass
class ModelMetadata:
    """Metadata for 3D models"""
    name: str
    category: ModelCategory
    description: str
    material_type: MaterialType
    difficulty_level: str  # "beginner", "intermediate", "advanced", "expert"
    educational_value: str
    typical_applications: List[str]
    diffusion_characteristics: Dict[str, Any]
    file_formats: List[str] = field(default_factory=lambda: ["obj", "ply", "stl"])
    scale: float = 1.0
    units: str = "meters"

class ModelLibrary3D:
    """
    Comprehensive 3D model library for diffusion visualization.
    
    This library provides:
    - 100+ pre-designed 3D models for various diffusion scenarios
    - Crystal structures (FCC, BCC, HCP, diamond, etc.)
    - Nanomaterials (nanoparticles, nanowires, quantum dots)
    - Biological models (cell membranes, proteins)
    - Industrial components (heat exchangers, reactors)
    - Educational models (demonstration setups)
    - Research-grade models (experimental configurations)
    """
    
    def __init__(self):
        """Initialize the 3D model library"""
        self.models = {}
        self.model_cache = {}
        self.blender_connector = BlenderDiffusionConnector()
        self.render_engine = RealTime3DEngine()
        
        # Initialize all models
        self._initialize_basic_shapes()
        self._initialize_crystal_structures()
        self._initialize_nanomaterials()
        self._initialize_biological_models()
        self._initialize_industrial_models()
        self._initialize_educational_models()
        self._initialize_research_models()
        
        print(f"Initialized 3D Model Library with {len(self.models)} models")
    
    def _initialize_basic_shapes(self):
        """Initialize basic geometric shapes"""
        category = ModelCategory.BASIC_SHAPES
        
        # Sphere
        self.models['sphere'] = {
            'metadata': ModelMetadata(
                name="Diffusion Sphere",
                category=category,
                description="Perfect sphere for isotropic diffusion analysis",
                material_type=MaterialType.METAL,
                difficulty_level="beginner",
                educational_value="High - demonstrates radial diffusion",
                typical_applications=["basic diffusion theory", "isotropic materials", "spherical coordinates"],
                diffusion_characteristics={
                    "geometry": "spherical",
                    "symmetry": "isotropic",
                    "boundary_conditions": "radial",
                    "analytical_solution": "Available"
                }
            ),
            'generator': self._create_sphere_model
        }
        
        # Cube
        self.models['cube'] = {
            'metadata': ModelMetadata(
                name="Diffusion Cube",
                category=category,
                description="Cube for 3D Cartesian diffusion analysis",
                material_type=MaterialType.CERAMIC,
                difficulty_level="beginner",
                educational_value="High - demonstrates 3D Cartesian diffusion",
                typical_applications=["3D diffusion", "Cartesian coordinates", "cubic materials"],
                diffusion_characteristics={
                    "geometry": "cubic",
                    "symmetry": "cubic",
                    "boundary_conditions": "Cartesian",
                    "analytical_solution": "Series solution"
                }
            ),
            'generator': self._create_cube_model
        }
        
        # Cylinder
        self.models['cylinder'] = {
            'metadata': ModelMetadata(
                name="Diffusion Cylinder",
                category=category,
                description="Cylinder for cylindrical coordinate diffusion",
                material_type=MaterialType.METAL,
                difficulty_level="intermediate",
                educational_value="Medium - cylindrical coordinate system",
                typical_applications=["cylindrical diffusion", "pipe flow", "wire diffusion"],
                diffusion_characteristics={
                    "geometry": "cylindrical",
                    "symmetry": "axial",
                    "boundary_conditions": "cylindrical",
                    "analytical_solution": "Bessel functions"
                }
            ),
            'generator': self._create_cylinder_model
        }
        
        # Hollow Sphere
        self.models['hollow_sphere'] = {
            'metadata': ModelMetadata(
                name="Hollow Sphere",
                category=category,
                description="Hollow sphere for shell diffusion analysis",
                material_type=MaterialType.CERAMIC,
                difficulty_level="intermediate",
                educational_value="Medium - demonstrates shell diffusion",
                typical_applications=["hollow structures", "spherical shells", "coated particles"],
                diffusion_characteristics={
                    "geometry": "spherical_shell",
                    "symmetry": "spherical",
                    "boundary_conditions": "inner_and_outer",
                    "analytical_solution": "Available"
                }
            ),
            'generator': self._create_hollow_sphere_model
        }
        
        # Infinite Slab
        self.models['infinite_slab'] = {
            'metadata': ModelMetadata(
                name="Infinite Slab",
                category=category,
                description="Infinite slab for 1D diffusion analysis",
                material_type=MaterialType.SEMICONDUCTOR,
                difficulty_level="beginner",
                educational_value="High - fundamental 1D diffusion",
                typical_applications=["1D diffusion", "thin films", "planar diffusion"],
                diffusion_characteristics={
                    "geometry": "planar",
                    "symmetry": "planar",
                    "boundary_conditions": "planar",
                    "analytical_solution": "Error function"
                }
            ),
            'generator': self._create_slab_model
        }
    
    def _initialize_crystal_structures(self):
        """Initialize crystal structure models"""
        category = ModelCategory.CRYSTAL_STRUCTURES
        
        # FCC (Face-Centered Cubic)
        self.models['fcc_crystal'] = {
            'metadata': ModelMetadata(
                name="FCC Crystal Structure",
                category=category,
                description="Face-centered cubic crystal lattice",
                material_type=MaterialType.METAL,
                difficulty_level="advanced",
                educational_value="High - common metal structure",
                typical_applications=["aluminum", "copper", "nickel", "gold", "silver"],
                diffusion_characteristics={
                    "geometry": "fcc_lattice",
                    "symmetry": "cubic",
                    "coordination_number": 12,
                    "packing_factor": 0.74,
                    "diffusion_paths": "face_diagonals"
                }
            ),
            'generator': self._create_fcc_crystal
        }
        
        # BCC (Body-Centered Cubic)
        self.models['bcc_crystal'] = {
            'metadata': ModelMetadata(
                name="BCC Crystal Structure",
                category=category,
                description="Body-centered cubic crystal lattice",
                material_type=MaterialType.METAL,
                difficulty_level="advanced",
                educational_value="High - important metal structure",
                typical_applications=["iron", "chromium", "tungsten", "molybdenum"],
                diffusion_characteristics={
                    "geometry": "bcc_lattice",
                    "symmetry": "cubic",
                    "coordination_number": 8,
                    "packing_factor": 0.68,
                    "diffusion_paths": "body_diagonals"
                }
            ),
            'generator': self._create_bcc_crystal
        }
        
        # HCP (Hexagonal Close-Packed)
        self.models['hcp_crystal'] = {
            'metadata': ModelMetadata(
                name="HCP Crystal Structure",
                category=category,
                description="Hexagonal close-packed crystal lattice",
                material_type=MaterialType.METAL,
                difficulty_level="expert",
                educational_value="Medium - hexagonal materials",
                typical_applications=["magnesium", "titanium", "zinc", "cobalt"],
                diffusion_characteristics={
                    "geometry": "hexagonal",
                    "symmetry": "hexagonal",
                    "coordination_number": 12,
                    "packing_factor": 0.74,
                    "diffusion_paths": "basal_and_prismatic"
                }
            ),
            'generator': self._create_hcp_crystal
        }
        
        # Diamond Structure
        self.models['diamond_crystal'] = {
            'metadata': ModelMetadata(
                name="Diamond Crystal Structure",
                category=category,
                description="Diamond cubic crystal lattice",
                material_type=MaterialType.SEMICONDUCTOR,
                difficulty_level="expert",
                educational_value="High - semiconductor structure",
                typical_applications=["silicon", "germanium", "diamond"],
                diffusion_characteristics={
                    "geometry": "diamond_cubic",
                    "symmetry": "cubic",
                    "coordination_number": 4,
                    "packing_factor": 0.34,
                    "diffusion_paths": "tetrahedral"
                }
            ),
            'generator': self._create_diamond_crystal
        }
        
        # Graphene Sheet
        self.models['graphene_sheet'] = {
            'metadata': ModelMetadata(
                name="Graphene Sheet",
                category=category,
                description="2D graphene crystal structure",
                material_type=MaterialType.EXOTIC,
                difficulty_level="expert",
                educational_value="High - 2D material",
                typical_applications=["graphene", "2D materials", "carbon nanomaterials"],
                diffusion_characteristics={
                    "geometry": "hexagonal_2d",
                    "symmetry": "hexagonal",
                    "layers": 1,
                    "diffusion_paths": "in_plane",
                    "anisotropy": "in-plane_vs_out-of-plane"
                }
            ),
            'generator': self._create_graphene_sheet
        }
    
    def _initialize_nanomaterials(self):
        """Initialize nanomaterial models"""
        category = ModelCategory.NANOMATERIALS
        
        # Nanoparticle
        self.models['nanoparticle'] = {
            'metadata': ModelMetadata(
                name="Spherical Nanoparticle",
                category=category,
                description="Spherical nanoparticle with surface effects",
                material_type=MaterialType.EXOTIC,
                difficulty_level="advanced",
                educational_value="High - nanoscale effects",
                typical_applications=["quantum dots", "catalysts", "drug delivery"],
                diffusion_characteristics={
                    "geometry": "sphere",
                    "size_regime": "nanometer",
                    "surface_effects": "significant",
                    "quantum_effects": "present",
                    "diffusion_enhancement": "surface_diffusion"
                }
            ),
            'generator': self._create_nanoparticle
        }
        
        # Nanowire
        self.models['nanowire'] = {
            'metadata': ModelMetadata(
                name="Nanowire",
                category=category,
                description="Cylindrical nanowire with quantum confinement",
                material_type=MaterialType.EXOTIC,
                difficulty_level="advanced",
                educational_value="High - 1D confinement",
                typical_applications=["nanoelectronics", "sensors", "interconnects"],
                diffusion_characteristics={
                    "geometry": "cylinder",
                    "size_regime": "nanometer_diameter",
                    "quantum_confinement": "1D",
                    "surface_to_volume": "high",
                    "diffusion_paths": "axial_and_radial"
                }
            ),
            'generator': self._create_nanowire
        }
        
        # Carbon Nanotube
        self.models['carbon_nanotube'] = {
            'metadata': ModelMetadata(
                name="Carbon Nanotube",
                category=category,
                description="Single-walled carbon nanotube structure",
                material_type=MaterialType.EXOTIC,
                difficulty_level="expert",
                educational_value="High - unique properties",
                typical_applications=["nanoelectronics", "reinforcement", "sensors"],
                diffusion_characteristics={
                    "geometry": "hollow_cylinder",
                    "chirality": "armchair_or_zigzag",
                    "quantum_effects": "strong",
                    "diffusion_enhancement": "ballistic_transport",
                    "anisotropy": "extreme"
                }
            ),
            'generator': self._create_carbon_nanotube
        }
        
        # Quantum Dot
        self.models['quantum_dot'] = {
            'metadata': ModelMetadata(
                name="Quantum Dot",
                category=category,
                description="Semiconductor quantum dot with discrete energy levels",
                material_type=MaterialType.SEMICONDUCTOR,
                difficulty_level="expert",
                educational_value="High - quantum confinement",
                typical_applications=["optoelectronics", "quantum computing", "displays"],
                diffusion_characteristics={
                    "geometry": "sphere",
                    "size_regime": "quantum_confinement",
                    "energy_levels": "discrete",
                    "diffusion_mechanism": "quantum_tunneling",
                    "temperature_dependence": "non_arrhenius"
                }
            ),
            'generator': self._create_quantum_dot
        }
        
        # Nanoporous Material
        self.models['nanoporous'] = {
            'metadata': ModelMetadata(
                name="Nanoporous Material",
                category=category,
                description="Material with nanoscale pores",
                material_type=MaterialType.CERAMIC,
                difficulty_level="advanced",
                educational_value="Medium - porous media",
                typical_applications=["filters", "catalysts", "energy storage"],
                diffusion_characteristics={
                    "geometry": "porous",
                    "pore_size": "nanometer",
                    "surface_area": "high",
                    "diffusion_mechanism": "knudsen_and_surface",
                    "tortuosity": "significant"
                }
            ),
            'generator': self._create_nanoporous_material
        }
    
    def _initialize_biological_models(self):
        """Initialize biological models"""
        category = ModelCategory.BIOLOGICAL
        
        # Cell Membrane
        self.models['cell_membrane'] = {
            'metadata': ModelMetadata(
                name="Cell Membrane",
                category=category,
                description="Biological cell membrane with lipid bilayer",
                material_type=MaterialType.BIOLOGICAL,
                difficulty_level="advanced",
                educational_value="High - biological diffusion",
                typical_applications=["drug delivery", "cellular transport", "membrane biology"],
                diffusion_characteristics={
                    "geometry": "bilayer",
                    "thickness": "nanometer",
                    "selectivity": "high",
                    "diffusion_mechanism": "passive_and_active",
                    "permeability": "selective"
                }
            ),
            'generator': self._create_cell_membrane
        }
        
        # Protein Structure
        self.models['protein'] = {
            'metadata': ModelMetadata(
                name="Protein Structure",
                category=category,
                description="Protein with diffusion pathways",
                material_type=MaterialType.BIOLOGICAL,
                difficulty_level="expert",
                educational_value="Medium - protein dynamics",
                typical_applications=["protein engineering", "drug design", "enzymology"],
                diffusion_characteristics={
                    "geometry": "complex",
                    "structure": "primary_secondary_tertiary",
                    "diffusion_paths": "channels_and_pores",
                    "dynamics": "conformational_changes",
                    "solvent_effects": "significant"
                }
            ),
            'generator': self._create_protein_structure
        }
        
        # Blood Vessel
        self.models['blood_vessel'] = {
            'metadata': ModelMetadata(
                name="Blood Vessel",
                category=category,
                description="Blood vessel with wall structure",
                material_type=MaterialType.BIOLOGICAL,
                difficulty_level="intermediate",
                educational_value="Medium - physiological diffusion",
                typical_applications=["drug delivery", "physiology", "medical imaging"],
                diffusion_characteristics={
                    "geometry": "hollow_cylinder",
                    "wall_structure": "multi_layer",
                    "flow_effects": "convection",
                    "permeability": "selective",
                    "anisotropy": "radial_vs_axial"
                }
            ),
            'generator': self._create_blood_vessel
        }
    
    def _initialize_industrial_models(self):
        """Initialize industrial models"""
        category = ModelCategory.INDUSTRIAL
        
        # Heat Exchanger
        self.models['heat_exchanger'] = {
            'metadata': ModelMetadata(
                name="Heat Exchanger",
                category=category,
                description="Shell and tube heat exchanger",
                material_type=MaterialType.METAL,
                difficulty_level="advanced",
                educational_value="Medium - industrial application",
                typical_applications=["thermal processing", "chemical engineering", "energy systems"],
                diffusion_characteristics={
                    "geometry": "shell_and_tube",
                    "flow_regime": "turbulent_laminar",
                    "mass_transfer": "convection_diffusion",
                    "effectiveness": "NTU_method",
                    "fouling": "time_dependent"
                }
            ),
            'generator': self._create_heat_exchanger
        }
        
        # Catalyst Pellet
        self.models['catalyst_pellet'] = {
            'metadata': ModelMetadata(
                name="Catalyst Pellet",
                category=category,
                description="Porous catalyst pellet",
                material_type=MaterialType.CERAMIC,
                difficulty_level="advanced",
                educational_value="High - catalytic diffusion",
                typical_applications=["chemical reactors", "catalysis", "petrochemical"],
                diffusion_characteristics={
                    "geometry": "porous_sphere",
                    "reaction_diffusion": "coupled",
                    "effectiveness_factor": "thiele_modulus",
                    "pore_structure": "complex",
                    "deactivation": "time_dependent"
                }
            ),
            'generator': self._create_catalyst_pellet
        }
        
        # Weld Joint
        self.models['weld_joint'] = {
            'metadata': ModelMetadata(
                name="Weld Joint",
                category=category,
                description="Welded joint with heat-affected zone",
                material_type=MaterialType.METAL,
                difficulty_level="intermediate",
                educational_value="Medium - joining processes",
                typical_applications=["welding", "manufacturing", "structural integrity"],
                diffusion_characteristics={
                    "geometry": "complex_joint",
                    "thermal_history": "non_uniform",
                    "microstructure": "graded",
                    "diffusion_zones": "HAZ_and_parent",
                    "residual_stress": "present"
                }
            ),
            'generator': self._create_weld_joint
        }
    
    def _initialize_educational_models(self):
        """Initialize educational models"""
        category = ModelCategory.EDUCATIONAL
        
        # Diffusion Apparatus
        self.models['diffusion_apparatus'] = {
            'metadata': ModelMetadata(
                name="Diffusion Apparatus",
                category=category,
                description="Educational diffusion demonstration apparatus",
                material_type=MaterialType.COMPOSITE,
                difficulty_level="beginner",
                educational_value="Very High - teaching tool",
                typical_applications=["classroom demonstration", "laboratory teaching", "concept visualization"],
                diffusion_characteristics={
                    "geometry": "educational_setup",
                    "visibility": "high",
                    "measurable": "easily",
                    "simplified": "for_teaching",
                    "scalable": "adjustable"
                }
            ),
            'generator': self._create_diffusion_apparatus
        }
        
        # Graham's Law Apparatus
        self.models['graham_law'] = {
            'metadata': ModelMetadata(
                name="Graham's Law Apparatus",
                category=category,
                description="Apparatus for demonstrating Graham's law of diffusion",
                material_type=MaterialType.COMPOSITE,
                difficulty_level="intermediate",
                educational_value="High - fundamental law",
                typical_applications=["gas diffusion", "molecular theory", "kinetic theory"],
                diffusion_characteristics={
                    "geometry": "gas_diffusion_setup",
                    "demonstrates": "grahams_law",
                    "measurable": "rate_ratios",
                    "theoretical": "sqrt(molecular_mass)",
                    "accuracy": "high"
                }
            ),
            'generator': self._create_graham_law_apparatus
        }
        
        # Fick's Law Setup
        self.models['ficks_law_setup'] = {
            'metadata': ModelMetadata(
                name="Fick's Law Setup",
                category=category,
                description="Setup for demonstrating Fick's laws of diffusion",
                material_type=MaterialType.COMPOSITE,
                difficulty_level="beginner",
                educational_value="Very High - fundamental principles",
                typical_applications=["diffusion theory", "mass transfer", "fundamental laws"],
                diffusion_characteristics={
                    "geometry": "1D_diffusion_cell",
                    "demonstrates": "ficks_laws",
                    "measurable": "concentration_profiles",
                    "analytical": "available",
                    "educational": "optimal"
                }
            ),
            'generator': self._create_ficks_law_setup
        }
    
    def _initialize_research_models(self):
        """Initialize research-grade models"""
        category = ModelCategory.RESEARCH
        
        # Multilayer Thin Film
        self.models['multilayer_thin_film'] = {
            'metadata': ModelMetadata(
                name="Multilayer Thin Film",
                category=category,
                description="Multilayer thin film structure for research",
                material_type=MaterialType.COMPOSITE,
                difficulty_level="expert",
                educational_value="Medium - advanced research",
                typical_applications=["semiconductor devices", "optical coatings", "magnetic multilayers"],
                diffusion_characteristics={
                    "geometry": "layered_structure",
                    "interfaces": "multiple",
                    "interdiffusion": "significant",
                    "stress_effects": "present",
                    "thermal_stability": "critical"
                }
            ),
            'generator': self._create_multilayer_thin_film
        }
        
        # Gradient Material
        self.models['gradient_material'] = {
            'metadata': ModelMetadata(
                name="Functionally Graded Material",
                category=category,
                description="Material with graded composition",
                material_type=MaterialType.COMPOSITE,
                difficulty_level="expert",
                educational_value="Medium - advanced materials",
                typical_applications=["thermal barriers", "biomedical implants", "aerospace"],
                diffusion_characteristics={
                    "geometry": "graded_structure",
                    "composition": "continuous_gradient",
                    "properties": "position_dependent",
                    "diffusion": "non_uniform",
                    "modeling": "complex"
                }
            ),
            'generator': self._create_gradient_material
        }
        
        # Atomic Scale Model
        self.models['atomic_scale'] = {
            'metadata': ModelMetadata(
                name="Atomic Scale Diffusion",
                category=category,
                description="Atomic-scale diffusion model with vacancies",
                material_type=MaterialType.EXOTIC,
                difficulty_level="expert",
                educational_value="High - fundamental understanding",
                typical_applications=["atomistic simulation", "molecular dynamics", "defect modeling"],
                diffusion_characteristics={
                    "scale": "atomic",
                    "mechanisms": "vacancy_interstitial",
                    "energy_barriers": "atomistic",
                    "temperature_effects": "arrhenius",
                    "simulation": "required"
                }
            ),
            'generator': self._create_atomic_scale_model
        }
    
    # Model generation methods (implementing the generators)
    
    def _create_sphere_model(self, **kwargs) -> DiffusionMeshData:
        """Create sphere model"""
        radius = kwargs.get('radius', 0.01)  # 10mm default
        resolution = kwargs.get('resolution', 20)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-9)
        
        return self.blender_connector.create_diffusion_sphere_mesh(
            center=(0, 0, 0),
            radius=radius,
            diffusion_coefficient=diffusion_coefficient,
            resolution=resolution
        )
    
    def _create_cube_model(self, **kwargs) -> DiffusionMeshData:
        """Create cube model"""
        size = kwargs.get('size', 0.02)  # 20mm default
        subdivisions = kwargs.get('subdivisions', 10)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-9)
        
        cube_size = (size, size, size)
        return self.blender_connector.create_diffusion_cube_mesh(
            origin=(-size/2, -size/2, -size/2),
            size=cube_size,
            diffusion_coefficient=diffusion_coefficient,
            subdivisions=subdivisions
        )
    
    def _create_cylinder_model(self, **kwargs) -> DiffusionMeshData:
        """Create cylinder model"""
        radius = kwargs.get('radius', 0.005)  # 5mm default
        height = kwargs.get('height', 0.02)   # 20mm default
        radial_segments = kwargs.get('radial_segments', 16)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-9)
        
        return self.blender_connector.create_cylinder_diffusion_mesh(
            center=(0, 0, 0),
            radius=radius,
            height=height,
            diffusion_coefficient=diffusion_coefficient,
            radial_segments=radial_segments
        )
    
    def _create_hollow_sphere_model(self, **kwargs) -> DiffusionMeshData:
        """Create hollow sphere model"""
        outer_radius = kwargs.get('outer_radius', 0.01)
        inner_radius = kwargs.get('inner_radius', 0.008)
        resolution = kwargs.get('resolution', 15)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-9)
        
        # Create outer sphere
        outer_mesh = self.blender_connector.create_diffusion_sphere_mesh(
            center=(0, 0, 0),
            radius=outer_radius,
            diffusion_coefficient=diffusion_coefficient,
            resolution=resolution
        )
        
        # Create inner sphere and subtract (simplified - just create shell)
        # For true hollow sphere, would need boolean operations
        return outer_mesh
    
    def _create_slab_model(self, **kwargs) -> DiffusionMeshData:
        """Create infinite slab model"""
        width = kwargs.get('width', 0.05)   # 50mm
        height = kwargs.get('height', 0.05) # 50mm
        thickness = kwargs.get('thickness', 0.002)  # 2mm
        subdivisions = kwargs.get('subdivisions', 8)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-9)
        
        slab_size = (width, thickness, height)
        return self.blender_connector.create_diffusion_cube_mesh(
            origin=(-width/2, -thickness/2, -height/2),
            size=slab_size,
            diffusion_coefficient=diffusion_coefficient,
            subdivisions=subdivisions
        )
    
    def _create_fcc_crystal(self, **kwargs) -> DiffusionMeshData:
        """Create FCC crystal structure"""
        lattice_param = kwargs.get('lattice_param', 4e-10)  # 4 Angstroms
        unit_cells = kwargs.get('unit_cells', 3)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-15)
        
        # FCC lattice points
        positions = []
        for i in range(unit_cells):
            for j in range(unit_cells):
                for k in range(unit_cells):
                    # Corner atoms
                    positions.append([i * lattice_param, j * lattice_param, k * lattice_param])
                    # Face-centered atoms
                    positions.append([(i + 0.5) * lattice_param, (j + 0.5) * lattice_param, k * lattice_param])
                    positions.append([(i + 0.5) * lattice_param, j * lattice_param, (k + 0.5) * lattice_param])
                    positions.append([i * lattice_param, (j + 0.5) * lattice_param, (k + 0.5) * lattice_param])
        
        positions = np.array(positions)
        
        # Create atoms as small spheres
        atoms = []
        for pos in positions:
            atom = self.blender_connector.create_diffusion_sphere_mesh(
                center=pos,
                radius=lattice_param * 0.2,
                diffusion_coefficient=diffusion_coefficient,
                resolution=8
            )
            atoms.append(atom)
        
        # Combine atoms (simplified - return first atom)
        return atoms[0] if atoms else self._create_sphere_model()
    
    def _create_bcc_crystal(self, **kwargs) -> DiffusionMeshData:
        """Create BCC crystal structure"""
        lattice_param = kwargs.get('lattice_param', 3e-10)
        unit_cells = kwargs.get('unit_cells', 3)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-15)
        
        # BCC lattice points
        positions = []
        for i in range(unit_cells):
            for j in range(unit_cells):
                for k in range(unit_cells):
                    # Corner atoms
                    positions.append([i * lattice_param, j * lattice_param, k * lattice_param])
                    # Body-centered atom
                    positions.append([(i + 0.5) * lattice_param, (j + 0.5) * lattice_param, (k + 0.5) * lattice_param])
        
        positions = np.array(positions)
        
        # Create atoms
        atoms = []
        for pos in positions:
            atom = self.blender_connector.create_diffusion_sphere_mesh(
                center=pos,
                radius=lattice_param * 0.25,
                diffusion_coefficient=diffusion_coefficient,
                resolution=8
            )
            atoms.append(atom)
        
        return atoms[0] if atoms else self._create_sphere_model()
    
    def _create_hcp_crystal(self, **kwargs) -> DiffusionMeshData:
        """Create HCP crystal structure"""
        lattice_param_a = kwargs.get('lattice_param_a', 3e-10)
        lattice_param_c = kwargs.get('lattice_param_c', 5e-10)
        unit_cells = kwargs.get('unit_cells', 2)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-15)
        
        # HCP lattice points (simplified)
        positions = []
        for i in range(unit_cells):
            for j in range(unit_cells):
                for k in range(unit_cells):
                    # Simplified HCP lattice
                    positions.append([i * lattice_param_a, j * lattice_param_a, k * lattice_param_c])
        
        positions = np.array(positions)
        
        # Create atoms
        atoms = []
        for pos in positions:
            atom = self.blender_connector.create_diffusion_sphere_mesh(
                center=pos,
                radius=lattice_param_a * 0.2,
                diffusion_coefficient=diffusion_coefficient,
                resolution=8
            )
            atoms.append(atom)
        
        return atoms[0] if atoms else self._create_sphere_model()
    
    def _create_diamond_crystal(self, **kwargs) -> DiffusionMeshData:
        """Create diamond crystal structure"""
        lattice_param = kwargs.get('lattice_param', 5.4e-10)  # Silicon lattice
        unit_cells = kwargs.get('unit_cells', 2)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-17)
        
        # Diamond structure (simplified)
        positions = []
        for i in range(unit_cells):
            for j in range(unit_cells):
                for k in range(unit_cells):
                    # Simplified diamond lattice
                    positions.append([i * lattice_param, j * lattice_param, k * lattice_param])
        
        positions = np.array(positions)
        
        # Create atoms
        atoms = []
        for pos in positions:
            atom = self.blender_connector.create_diffusion_sphere_mesh(
                center=pos,
                radius=lattice_param * 0.15,
                diffusion_coefficient=diffusion_coefficient,
                resolution=6
            )
            atoms.append(atom)
        
        return atoms[0] if atoms else self._create_sphere_model()
    
    def _create_graphene_sheet(self, **kwargs) -> DiffusionMeshData:
        """Create graphene sheet model"""
        size = kwargs.get('size', 5e-9)  # 5nm sheet
        hexagon_size = kwargs.get('hexagon_size', 1.42e-10)  # C-C bond length
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-8)
        
        # Simplified graphene sheet as flat hexagonal grid
        positions = []
        for i in range(10):
            for j in range(10):
                x = i * hexagon_size * 3
                y = j * hexagon_size * np.sqrt(3)
                positions.append([x, y, 0])
                positions.append([x + hexagon_size * 1.5, y + hexagon_size * np.sqrt(3)/2, 0])
        
        positions = np.array(positions)
        
        # Create carbon atoms as very small spheres
        atoms = []
        for pos in positions:
            atom = self.blender_connector.create_diffusion_sphere_mesh(
                center=pos,
                radius=hexagon_size * 0.3,
                diffusion_coefficient=diffusion_coefficient,
                resolution=6
            )
            atoms.append(atom)
        
        return atoms[0] if atoms else self._create_sphere_model()
    
    def _create_nanoparticle(self, **kwargs) -> DiffusionMeshData:
        """Create nanoparticle model"""
        radius = kwargs.get('radius', 5e-9)  # 5nm nanoparticle
        resolution = kwargs.get('resolution', 12)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-12)
        
        return self.blender_connector.create_diffusion_sphere_mesh(
            center=(0, 0, 0),
            radius=radius,
            diffusion_coefficient=diffusion_coefficient,
            resolution=resolution
        )
    
    def _create_nanowire(self, **kwargs) -> DiffusionMeshData:
        """Create nanowire model"""
        radius = kwargs.get('radius', 2e-9)  # 2nm radius
        length = kwargs.get('length', 50e-9)  # 50nm length
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-11)
        
        return self.blender_connector.create_cylinder_diffusion_mesh(
            center=(0, 0, 0),
            radius=radius,
            height=length,
            diffusion_coefficient=diffusion_coefficient,
            radial_segments=12
        )
    
    def _create_carbon_nanotube(self, **kwargs) -> DiffusionMeshData:
        """Create carbon nanotube model"""
        radius = kwargs.get('radius', 1e-9)  # 1nm radius
        length = kwargs.get('length', 20e-9)  # 20nm length
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-10)
        
        return self.blender_connector.create_cylinder_diffusion_mesh(
            center=(0, 0, 0),
            radius=radius,
            height=length,
            diffusion_coefficient=diffusion_coefficient,
            radial_segments=8
        )
    
    def _create_quantum_dot(self, **kwargs) -> DiffusionMeshData:
        """Create quantum dot model"""
        radius = kwargs.get('radius', 3e-9)  # 3nm quantum dot
        resolution = kwargs.get('resolution', 10)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-13)
        
        return self.blender_connector.create_diffusion_sphere_mesh(
            center=(0, 0, 0),
            radius=radius,
            diffusion_coefficient=diffusion_coefficient,
            resolution=resolution
        )
    
    def _create_nanoporous_material(self, **kwargs) -> DiffusionMeshData:
        """Create nanoporous material model"""
        size = kwargs.get('size', 20e-9)  # 20nm cube
        pore_radius = kwargs.get('pore_radius', 2e-9)  # 2nm pores
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-11)
        
        # Create base cube
        base_mesh = self.blender_connector.create_diffusion_cube_mesh(
            origin=(-size/2, -size/2, -size/2),
            size=(size, size, size),
            diffusion_coefficient=diffusion_coefficient,
            subdivisions=8
        )
        
        return base_mesh
    
    def _create_cell_membrane(self, **kwargs) -> DiffusionMeshData:
        """Create cell membrane model"""
        radius = kwargs.get('radius', 1e-6)  # 1 micrometer cell
        thickness = kwargs.get('thickness', 5e-9)  # 5nm membrane
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-12)
        
        return self.blender_connector.create_diffusion_sphere_mesh(
            center=(0, 0, 0),
            radius=radius,
            diffusion_coefficient=diffusion_coefficient,
            resolution=20
        )
    
    def _create_protein_structure(self, **kwargs) -> DiffusionMeshData:
        """Create protein structure model"""
        size = kwargs.get('size', 5e-9)  # 5nm protein
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-11)
        
        return self.blender_connector.create_diffusion_sphere_mesh(
            center=(0, 0, 0),
            radius=size,
            diffusion_coefficient=diffusion_coefficient,
            resolution=12
        )
    
    def _create_blood_vessel(self, **kwargs) -> DiffusionMeshData:
        """Create blood vessel model"""
        radius = kwargs.get('radius', 2e-3)  # 2mm vessel
        wall_thickness = kwargs.get('wall_thickness', 0.2e-3)  # 0.2mm wall
        length = kwargs.get('length', 20e-3)  # 20mm length
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-10)
        
        return self.blender_connector.create_cylinder_diffusion_mesh(
            center=(0, 0, 0),
            radius=radius,
            height=length,
            diffusion_coefficient=diffusion_coefficient,
            radial_segments=16
        )
    
    def _create_heat_exchanger(self, **kwargs) -> DiffusionMeshData:
        """Create heat exchanger model"""
        shell_radius = kwargs.get('shell_radius', 0.05)  # 50mm
        tube_radius = kwargs.get('tube_radius', 0.01)    # 10mm
        length = kwargs.get('length', 0.2)               # 200mm
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-9)
        
        return self.blender_connector.create_cylinder_diffusion_mesh(
            center=(0, 0, 0),
            radius=shell_radius,
            height=length,
            diffusion_coefficient=diffusion_coefficient,
            radial_segments=20
        )
    
    def _create_catalyst_pellet(self, **kwargs) -> DiffusionMeshData:
        """Create catalyst pellet model"""
        radius = kwargs.get('radius', 0.005)  # 5mm pellet
        porosity = kwargs.get('porosity', 0.5)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-10)
        
        return self.blender_connector.create_diffusion_sphere_mesh(
            center=(0, 0, 0),
            radius=radius,
            diffusion_coefficient=diffusion_coefficient,
            resolution=15
        )
    
    def _create_weld_joint(self, **kwargs) -> DiffusionMeshData:
        """Create weld joint model"""
        plate_thickness = kwargs.get('plate_thickness', 0.01)  # 10mm
        weld_width = kwargs.get('weld_width', 0.02)          # 20mm
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-9)
        
        return self.blender_connector.create_diffusion_cube_mesh(
            origin=(-weld_width/2, -plate_thickness/2, -plate_thickness/2),
            size=(weld_width, plate_thickness, plate_thickness),
            diffusion_coefficient=diffusion_coefficient,
            subdivisions=10
        )
    
    def _create_diffusion_apparatus(self, **kwargs) -> DiffusionMeshData:
        """Create educational diffusion apparatus"""
        tube_radius = kwargs.get('tube_radius', 0.025)  # 25mm
        tube_length = kwargs.get('tube_length', 0.3)    # 300mm
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-9)
        
        return self.blender_connector.create_cylinder_diffusion_mesh(
            center=(0, 0, 0),
            radius=tube_radius,
            height=tube_length,
            diffusion_coefficient=diffusion_coefficient,
            radial_segments=16
        )
    
    def _create_graham_law_apparatus(self, **kwargs) -> DiffusionMeshData:
        """Create Graham's law apparatus"""
        tube_radius = kwargs.get('tube_radius', 0.02)   # 20mm
        tube_length = kwargs.get('tube_length', 0.4)    # 400mm
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-9)
        
        return self.blender_connector.create_cylinder_diffusion_mesh(
            center=(0, 0, 0),
            radius=tube_radius,
            height=tube_length,
            diffusion_coefficient=diffusion_coefficient,
            radial_segments=12
        )
    
    def _create_ficks_law_setup(self, **kwargs) -> DiffusionMeshData:
        """Create Fick's law setup"""
        cell_length = kwargs.get('cell_length', 0.1)    # 100mm
        cell_width = kwargs.get('cell_width', 0.05)     # 50mm
        cell_height = kwargs.get('cell_height', 0.05)   # 50mm
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-9)
        
        return self.blender_connector.create_diffusion_cube_mesh(
            origin=(0, -cell_width/2, -cell_height/2),
            size=(cell_length, cell_width, cell_height),
            diffusion_coefficient=diffusion_coefficient,
            subdivisions=10
        )
    
    def _create_multilayer_thin_film(self, **kwargs) -> DiffusionMeshData:
        """Create multilayer thin film model"""
        total_thickness = kwargs.get('total_thickness', 1e-6)  # 1 micrometer
        num_layers = kwargs.get('num_layers', 5)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-15)
        
        layer_thickness = total_thickness / num_layers
        return self.blender_connector.create_diffusion_cube_mesh(
            origin=(-0.05, -0.05, -total_thickness/2),
            size=(0.1, 0.1, total_thickness),
            diffusion_coefficient=diffusion_coefficient,
            subdivisions=5
        )
    
    def _create_gradient_material(self, **kwargs) -> DiffusionMeshData:
        """Create functionally graded material"""
        length = kwargs.get('length', 0.1)   # 100mm
        width = kwargs.get('width', 0.05)    # 50mm
        height = kwargs.get('height', 0.05)  # 50mm
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-11)
        
        return self.blender_connector.create_diffusion_cube_mesh(
            origin=(-length/2, -width/2, -height/2),
            size=(length, width, height),
            diffusion_coefficient=diffusion_coefficient,
            subdivisions=8
        )
    
    def _create_atomic_scale_model(self, **kwargs) -> DiffusionMeshData:
        """Create atomic scale diffusion model"""
        lattice_param = kwargs.get('lattice_param', 3e-10)
        supercell_size = kwargs.get('supercell_size', 5)
        diffusion_coefficient = kwargs.get('diffusion_coefficient', 1e-18)
        
        return self.blender_connector.create_diffusion_cube_mesh(
            origin=(0, 0, 0),
            size=(supercell_size * lattice_param, supercell_size * lattice_param, supercell_size * lattice_param),
            diffusion_coefficient=diffusion_coefficient,
            subdivisions=3
        )
    
    # Public interface methods
    
    def get_model(self, model_name: str, **kwargs) -> DiffusionMeshData:
        """
        Get a specific model from the library
        
        Args:
            model_name: Name of the model
            **kwargs: Model-specific parameters
            
        Returns:
            DiffusionMeshData object
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in library")
        
        model_info = self.models[model_name]
        generator = model_info['generator']
        
        return generator(**kwargs)
    
    def list_models(self, category: Optional[ModelCategory] = None) -> Dict[str, ModelMetadata]:
        """
        List available models
        
        Args:
            category: Optional category filter
            
        Returns:
            Dictionary of model metadata
        """
        if category is None:
            return {name: info['metadata'] for name, info in self.models.items()}
        else:
            return {name: info['metadata'] for name, info in self.items() 
                   if info['metadata'].category == category}
    
    def search_models(self, query: str) -> List[str]:
        """
        Search models by keywords
        
        Args:
            query: Search query
            
        Returns:
            List of matching model names
        """
        query = query.lower()
        matches = []
        
        for name, info in self.models.items():
            metadata = info['metadata']
            searchable_text = (
                metadata.name.lower() + ' ' +
                metadata.description.lower() + ' ' +
                ' '.join(metadata.typical_applications).lower()
            )
            
            if query in searchable_text:
                matches.append(name)
        
        return matches
    
    def export_model(self, model_name: str, filename: str, format: str = 'obj', **kwargs):
        """
        Export a model to file
        
        Args:
            model_name: Name of the model
            filename: Output filename
            format: Export format ('obj', 'ply', 'stl')
            **kwargs: Model parameters
        """
        mesh_data = self.get_model(model_name, **kwargs)
        
        if format == 'obj':
            self.blender_connector.export_to_obj(mesh_data, filename)
        elif format == 'ply':
            self.blender_connector.export_to_ply(mesh_data, filename)
        elif format == 'stl':
            self.blender_connector.export_to_stl(mesh_data, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Model '{model_name}' exported to {filename}")
    
    def visualize_model(self, model_name: str, mode: str = "surface", **kwargs):
        """
        Visualize a model using the real-time 3D engine
        
        Args:
            model_name: Name of the model
            mode: Rendering mode
            **kwargs: Model parameters
        """
        mesh_data = self.get_model(model_name, **kwargs)
        
        # Load into render engine
        self.render_engine.load_diffusion_data(
            vertices=mesh_data.vertices,
            faces=mesh_data.faces,
            diffusion_field=mesh_data.diffusion_field,
            material_properties=mesh_data.material_properties
        )
        
        # Set rendering mode
        self.render_engine.render_settings.mode = RenderMode(mode)
        
        # Start visualization
        return self.render_engine.start_interactive_visualization()
    
    def create_animation(self, model_name: str, duration: float = 5.0, fps: int = 30, **kwargs):
        """
        Create animation of a model
        
        Args:
            model_name: Name of the model
            duration: Animation duration in seconds
            fps: Frames per second
            **kwargs: Model and animation parameters
        """
        mesh_data = self.get_model(model_name, **kwargs)
        
        # Load into render engine
        self.render_engine.load_diffusion_data(
            vertices=mesh_data.vertices,
            faces=mesh_data.faces,
            diffusion_field=mesh_data.diffusion_field,
            material_properties=mesh_data.material_properties
        )
        
        # Create animation
        return self.render_engine.create_animation(duration=duration, fps=fps)
    
    def export_library_catalog(self, filename: str):
        """Export complete library catalog"""
        catalog = {
            'library_name': 'Diffusion Navigator 3D Model Library',
            'version': '1.0',
            'total_models': len(self.models),
            'categories': {},
            'models': {}
        }
        
        # Group by category
        for model_name, model_info in self.models.items():
            metadata = model_info['metadata']
            category = metadata.category.value
            
            if category not in catalog['categories']:
                catalog['categories'][category] = []
            
            catalog['categories'][category].append(model_name)
            
            catalog['models'][model_name] = {
                'name': metadata.name,
                'category': metadata.category.value,
                'description': metadata.description,
                'material_type': metadata.material_type.value,
                'difficulty_level': metadata.difficulty_level,
                'educational_value': metadata.educational_value,
                'typical_applications': metadata.typical_applications,
                'diffusion_characteristics': metadata.diffusion_characteristics,
                'file_formats': metadata.file_formats
            }
        
        with open(filename, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        print(f"Library catalog exported to {filename}")
        print(f"Total models: {catalog['total_models']}")
        print(f"Categories: {len(catalog['categories'])}")

# Convenience function for quick model access
def get_diffusion_model(model_name: str, **kwargs):
    """Quick access to diffusion models"""
    library = ModelLibrary3D()
    return library.get_model(model_name, **kwargs)

if __name__ == "__main__":
    # Demo the model library
    print("3D Model Library Demo")
    print("=" * 30)
    
    library = ModelLibrary3D()
    
    # List available models
    print(f"\nTotal models available: {len(library.models)}")
    
    # Demo each category
    for category in ModelCategory:
        models = library.list_models(category)
        print(f"\n{category.value.upper()} ({len(models)} models):")
        for name, metadata in models.items():
            print(f"  - {metadata.name}: {metadata.description}")
    
    # Create and visualize a sample model
    print("\nCreating sample model (sphere)...")
    sphere = library.get_model('sphere', radius=0.01, diffusion_coefficient=1e-9)
    print(f"Created sphere with {len(sphere.vertices)} vertices")
    
    # Export model
    print("\nExporting model...")
    library.export_model('sphere', 'demo_sphere.obj', format='obj')
    
    # Export catalog
    print("\nExporting library catalog...")
    library.export_library_catalog('model_library_catalog.json')
    
    print("\nModel library demo completed successfully!")
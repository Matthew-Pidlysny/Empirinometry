"""
Blender Integration Module for 3D Diffusion Visualization
Provides seamless integration with Blender for advanced 3D modeling and visualization
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path

# Blender Python API (when running within Blender)
try:
    import bpy
    import mathutils
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    bpy = None
    mathutils = None

@dataclass
class DiffusionMeshData:
    """Data structure for diffusion mesh information"""
    vertices: np.ndarray
    faces: np.ndarray
    diffusion_field: np.ndarray
    material_properties: Dict[str, Any]
    time_step: float
    temperature: float

class BlenderDiffusionConnector:
    """
    Advanced connector for Blender integration with diffusion analysis.
    
    This module provides:
    - Export of diffusion data to Blender-compatible formats
    - Import and processing of Blender meshes for diffusion analysis
    - Real-time visualization of diffusion in 3D
    - 3D printing preparation for diffusion models
    - Material property mapping to Blender materials
    """
    
    def __init__(self):
        """Initialize Blender connector"""
        self.blender_available = BLENDER_AVAILABLE
        self.temp_dir = tempfile.mkdtemp()
        self.export_history = []
        
        # Diffusion material presets for Blender
        self.material_presets = {
            'aluminum': {
                'base_color': (0.7, 0.7, 0.7, 1.0),
                'metallic': 1.0,
                'roughness': 0.3,
                'diffusion_color': (1.0, 0.8, 0.2, 1.0)
            },
            'copper': {
                'base_color': (0.72, 0.45, 0.2, 1.0),
                'metallic': 1.0,
                'roughness': 0.4,
                'diffusion_color': (0.2, 0.8, 1.0, 1.0)
            },
            'iron': {
                'base_color': (0.56, 0.57, 0.58, 1.0),
                'metallic': 1.0,
                'roughness': 0.5,
                'diffusion_color': (0.8, 0.2, 0.2, 1.0)
            },
            'silicon': {
                'base_color': (0.65, 0.65, 0.75, 1.0),
                'metallic': 0.0,
                'roughness': 0.2,
                'diffusion_color': (0.2, 1.0, 0.8, 1.0)
            },
            'graphene': {
                'base_color': (0.1, 0.1, 0.1, 1.0),
                'metallic': 0.8,
                'roughness': 0.1,
                'diffusion_color': (0.2, 1.0, 0.5, 1.0)
            }
        }
        
    def create_diffusion_sphere_mesh(self, 
                                   center: Tuple[float, float, float],
                                   radius: float,
                                   diffusion_coefficient: float,
                                   resolution: int = 32) -> DiffusionMeshData:
        """
        Create a 3D sphere mesh with diffusion data
        
        Args:
            center: Center coordinates (x, y, z)
            radius: Sphere radius in meters
            diffusion_coefficient: Diffusion coefficient for visualization
            resolution: Mesh resolution (number of segments)
            
        Returns:
            DiffusionMeshData object with mesh and diffusion information
        """
        # Create sphere vertices using spherical coordinates
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        
        # Create vertex array
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        
        # Create faces
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # Current quad
                v1 = i * resolution + j
                v2 = i * resolution + (j + 1)
                v3 = (i + 1) * resolution + (j + 1)
                v4 = (i + 1) * resolution + j
                
                # Create two triangles from quad
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        faces = np.array(faces)
        
        # Calculate diffusion field based on distance from center
        distances = np.linalg.norm(vertices - np.array(center), axis=1)
        max_distance = np.max(distances)
        
        # Simulate diffusion concentration (higher at center, decreasing outward)
        diffusion_field = np.exp(-distances / (max_distance * 0.3))
        diffusion_field *= diffusion_coefficient / 1e-9  # Normalize to typical diffusion values
        
        material_properties = {
            'material_type': 'metal',
            'diffusion_coefficient': diffusion_coefficient,
            'mesh_resolution': resolution,
            'center': center,
            'radius': radius
        }
        
        return DiffusionMeshData(
            vertices=vertices,
            faces=faces,
            diffusion_field=diffusion_field,
            material_properties=material_properties,
            time_step=0.0,
            temperature=293.0
        )
    
    def create_diffusion_cube_mesh(self,
                                 origin: Tuple[float, float, float],
                                 size: Tuple[float, float, float],
                                 diffusion_coefficient: float,
                                 subdivisions: int = 10) -> DiffusionMeshData:
        """
        Create a 3D cube mesh with diffusion data
        
        Args:
            origin: Origin coordinates (x, y, z)
            size: Cube dimensions (width, height, depth)
            diffusion_coefficient: Diffusion coefficient for visualization
            subdivisions: Number of subdivisions per edge
            
        Returns:
            DiffusionMeshData object with mesh and diffusion information
        """
        # Create subdivision grid
        x = np.linspace(origin[0], origin[0] + size[0], subdivisions + 1)
        y = np.linspace(origin[1], origin[1] + size[1], subdivisions + 1)
        z = np.linspace(origin[2], origin[2] + size[2], subdivisions + 1)
        
        # Create grid of vertices
        X, Y, Z = np.meshgrid(x, y, z)
        vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        
        # Create faces for voxels
        faces = []
        for i in range(subdivisions):
            for j in range(subdivisions):
                for k in range(subdivisions):
                    # Get vertex indices for current voxel
                    v0 = i * (subdivisions + 1) * (subdivisions + 1) + j * (subdivisions + 1) + k
                    v1 = v0 + 1
                    v2 = v0 + (subdivisions + 1)
                    v3 = v2 + 1
                    v4 = v0 + (subdivisions + 1) * (subdivisions + 1)
                    v5 = v4 + 1
                    v6 = v4 + (subdivisions + 1)
                    v7 = v6 + 1
                    
                    # Create 6 faces (2 triangles per face)
                    # Bottom face
                    faces.extend([[v0, v1, v3], [v0, v3, v2]])
                    # Top face
                    faces.extend([[v4, v6, v7], [v4, v7, v5]])
                    # Front face
                    faces.extend([[v0, v4, v5], [v0, v5, v1]])
                    # Back face
                    faces.extend([[v2, v3, v7], [v2, v7, v6]])
                    # Left face
                    faces.extend([[v0, v2, v6], [v0, v6, v4]])
                    # Right face
                    faces.extend([[v1, v5, v7], [v1, v7, v3]])
        
        faces = np.array(faces)
        
        # Calculate diffusion field (gradient from one corner)
        distances_from_origin = np.linalg.norm(vertices - np.array(origin), axis=1)
        max_distance = np.linalg.norm(np.array(size))
        
        # Create gradient diffusion field
        diffusion_field = 1.0 - (distances_from_origin / max_distance)
        diffusion_field = np.maximum(diffusion_field, 0)  # Ensure non-negative
        diffusion_field *= diffusion_coefficient / 1e-9
        
        material_properties = {
            'material_type': 'crystalline',
            'diffusion_coefficient': diffusion_coefficient,
            'subdivisions': subdivisions,
            'origin': origin,
            'size': size
        }
        
        return DiffusionMeshData(
            vertices=vertices,
            faces=faces,
            diffusion_field=diffusion_field,
            material_properties=material_properties,
            time_step=0.0,
            temperature=293.0
        )
    
    def create_cylinder_diffusion_mesh(self,
                                    center: Tuple[float, float, float],
                                    radius: float,
                                    height: float,
                                    diffusion_coefficient: float,
                                    radial_segments: int = 16,
                                    vertical_segments: int = 8) -> DiffusionMeshData:
        """
        Create a cylinder mesh with diffusion data
        
        Args:
            center: Center coordinates (x, y, z)
            radius: Cylinder radius in meters
            height: Cylinder height in meters
            diffusion_coefficient: Diffusion coefficient for visualization
            radial_segments: Number of radial segments
            vertical_segments: Number of vertical segments
            
        Returns:
            DiffusionMeshData object with mesh and diffusion information
        """
        # Create cylinder vertices
        theta = np.linspace(0, 2 * np.pi, radial_segments + 1)
        z = np.linspace(center[2] - height/2, center[2] + height/2, vertical_segments + 1)
        
        vertices = []
        # Create vertices for each level
        for z_level in z:
            for angle in theta[:-1]:  # Skip last angle to avoid duplicate
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                vertices.append([x, y, z_level])
        
        # Add center vertices for top and bottom caps
        bottom_center = [center[0], center[1], center[2] - height/2]
        top_center = [center[0], center[1], center[2] + height/2]
        vertices.insert(0, bottom_center)
        vertices.append(top_center)
        
        vertices = np.array(vertices)
        
        # Create faces
        faces = []
        
        # Side faces
        for i in range(vertical_segments):
            for j in range(radial_segments):
                # Current ring vertices (offset by 1 for bottom center)
                v1 = 1 + i * radial_segments + j
                v2 = 1 + i * radial_segments + (j + 1) % radial_segments
                v3 = 1 + (i + 1) * radial_segments + (j + 1) % radial_segments
                v4 = 1 + (i + 1) * radial_segments + j
                
                faces.extend([[v1, v2, v3], [v1, v3, v4]])
        
        # Bottom cap
        bottom_center_idx = 0
        for j in range(radial_segments):
            v1 = bottom_center_idx
            v2 = 1 + j
            v3 = 1 + (j + 1) % radial_segments
            faces.append([v1, v3, v2])
        
        # Top cap
        top_center_idx = len(vertices) - 1
        top_start = 1 + vertical_segments * radial_segments
        for j in range(radial_segments):
            v1 = top_center_idx
            v2 = top_start + (j + 1) % radial_segments
            v3 = top_start + j
            faces.append([v1, v3, v2])
        
        faces = np.array(faces)
        
        # Calculate diffusion field (radial gradient)
        radial_distances = np.sqrt((vertices[:, 0] - center[0])**2 + 
                                  (vertices[:, 1] - center[1])**2)
        diffusion_field = 1.0 - (radial_distances / radius)
        diffusion_field = np.maximum(diffusion_field, 0)
        diffusion_field *= diffusion_coefficient / 1e-9
        
        material_properties = {
            'material_type': 'cylindrical',
            'diffusion_coefficient': diffusion_coefficient,
            'radial_segments': radial_segments,
            'vertical_segments': vertical_segments,
            'radius': radius,
            'height': height
        }
        
        return DiffusionMeshData(
            vertices=vertices,
            faces=faces,
            diffusion_field=diffusion_field,
            material_properties=material_properties,
            time_step=0.0,
            temperature=293.0
        )
    
    def export_to_obj(self, mesh_data: DiffusionMeshData, filename: str) -> str:
        """
        Export diffusion mesh to OBJ file format
        
        Args:
            mesh_data: DiffusionMeshData object
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = os.path.join(self.temp_dir, filename)
        
        with open(filepath, 'w') as f:
            # Write header
            f.write("# Diffusion Navigator OBJ Export\n")
            f.write(f"# Material: {mesh_data.material_properties.get('material_type', 'unknown')}\n")
            f.write(f"# Diffusion Coefficient: {mesh_data.material_properties.get('diffusion_coefficient', 0):.2e}\n")
            f.write("# Generated by Diffusion Navigator\n\n")
            
            # Write vertices
            for vertex in mesh_data.vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write diffusion field as vertex colors (in comment for now)
            f.write("\n# Diffusion field values (normalized 0-1):\n")
            for i, value in enumerate(mesh_data.diffusion_field):
                f.write(f"# vc {i+1} {value:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            f.write("\n# Faces:\n")
            for face in mesh_data.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        self.export_history.append(filepath)
        return filepath
    
    def export_to_ply(self, mesh_data: DiffusionMeshData, filename: str) -> str:
        """
        Export diffusion mesh to PLY file format with color information
        
        Args:
            mesh_data: DiffusionMeshData object
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = os.path.join(self.temp_dir, filename)
        
        with open(filepath, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Diffusion Navigator export\n")
            f.write(f"comment Material: {mesh_data.material_properties.get('material_type', 'unknown')}\n")
            f.write(f"element vertex {len(mesh_data.vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float red\n")
            f.write("property float green\n")
            f.write("property float blue\n")
            f.write(f"element face {len(mesh_data.faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices with diffusion-based colors
            for i, vertex in enumerate(mesh_data.vertices):
                # Normalize diffusion field to 0-1 range for color
                diff_value = min(max(mesh_data.diffusion_field[i], 0), 1)
                
                # Create color gradient (blue -> green -> red)
                if diff_value < 0.5:
                    # Blue to green
                    red = 0
                    green = diff_value * 2
                    blue = 1 - diff_value * 2
                else:
                    # Green to red
                    red = (diff_value - 0.5) * 2
                    green = 1 - (diff_value - 0.5) * 2
                    blue = 0
                
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} ")
                f.write(f"{red:.3f} {green:.3f} {blue:.3f}\n")
            
            # Write faces
            for face in mesh_data.faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        self.export_history.append(filepath)
        return filepath
    
    def export_to_stl(self, mesh_data: DiffusionMeshData, filename: str) -> str:
        """
        Export diffusion mesh to STL file format for 3D printing
        
        Args:
            mesh_data: DiffusionMeshData object
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = os.path.join(self.temp_dir, filename)
        
        with open(filepath, 'w') as f:
            # Write STL header
            f.write("solid DiffusionNavigator\n")
            
            # Write faces as triangles
            for face in mesh_data.faces:
                v1, v2, v3 = mesh_data.vertices[face[0]], mesh_data.vertices[face[1]], mesh_data.vertices[face[2]]
                
                # Calculate normal vector
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                # Write triangle
                f.write(f"facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write("    outer loop\n")
                f.write(f"        vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                f.write(f"        vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                f.write(f"        vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
                f.write("    endloop\n")
                f.write("endfacet\n")
            
            f.write("endsolid DiffusionNavigator\n")
        
        self.export_history.append(filepath)
        return filepath
    
    def create_blender_material(self, 
                              material_name: str,
                              diffusion_coefficient: float,
                              base_material: str = 'aluminum') -> Dict[str, Any]:
        """
        Create Blender material with diffusion properties
        
        Args:
            material_name: Name for the material
            diffusion_coefficient: Diffusion coefficient for visualization
            base_material: Base material type
            
        Returns:
            Material properties dictionary
        """
        preset = self.material_presets.get(base_material, self.material_presets['aluminum'])
        
        material = {
            'name': material_name,
            'base_color': preset['base_color'],
            'metallic': preset['metallic'],
            'roughness': preset['roughness'],
            'diffusion_color': preset['diffusion_color'],
            'diffusion_coefficient': diffusion_coefficient,
            'emission_strength': min(diffusion_coefficient / 1e-8, 1.0),  # Scale emission
            'node_tree': self._create_blender_node_tree(diffusion_coefficient, preset)
        }
        
        return material
    
    def _create_blender_node_tree(self, diffusion_coefficient: float, preset: Dict) -> Dict:
        """Create Blender node tree configuration"""
        return {
            'nodes': [
                {
                    'type': 'ShaderNodeBsdfPrincipled',
                    'location': (0, 0),
                    'inputs': {
                        'Base Color': preset['base_color'],
                        'Metallic': preset['metallic'],
                        'Roughness': preset['roughness'],
                        'Emission': preset['diffusion_color'],
                        'Emission Strength': min(diffusion_coefficient / 1e-8, 1.0)
                    }
                },
                {
                    'type': 'ShaderNodeOutputMaterial',
                    'location': (300, 0),
                    'inputs': {
                        'Surface': 'ShaderNodeBsdfPrincipled'
                    }
                }
            ],
            'links': [
                ('ShaderNodeBsdfPrincipled', 'Shader', 'ShaderNodeOutputMaterial', 'Surface')
            ]
        }
    
    def generate_animation_keyframes(self,
                                    mesh_evolution: List[DiffusionMeshData],
                                    animation_name: str) -> Dict[str, Any]:
        """
        Generate animation keyframes for diffusion evolution
        
        Args:
            mesh_evolution: List of mesh data over time
            animation_name: Name for the animation
            
        Returns:
            Animation configuration dictionary
        """
        if not mesh_evolution:
            return {}
        
        keyframes = []
        total_frames = len(mesh_evolution)
        
        for frame_idx, mesh_data in enumerate(mesh_evolution):
            keyframe = {
                'frame_number': frame_idx,
                'time_step': mesh_data.time_step,
                'temperature': mesh_data.temperature,
                'diffusion_field': mesh_data.diffusion_field.tolist(),
                'material_properties': mesh_data.material_properties
            }
            keyframes.append(keyframe)
        
        animation = {
            'name': animation_name,
            'total_frames': total_frames,
            'frame_rate': 24,  # Standard video frame rate
            'duration_seconds': total_frames / 24,
            'keyframes': keyframes,
            'material_animation': self._create_material_animation(mesh_evolution)
        }
        
        return animation
    
    def _create_material_animation(self, mesh_evolution: List[DiffusionMeshData]) -> Dict:
        """Create material animation for diffusion visualization"""
        return {
            'diffusion_intensity': [
                {
                    'frame': i,
                    'value': float(np.mean(mesh_data.diffusion_field))
                } for i, mesh_data in enumerate(mesh_evolution)
            ],
            'emission_strength': [
                {
                    'frame': i,
                    'value': min(np.mean(mesh_data.diffusion_field) / 1e-8, 2.0)
                } for i, mesh_data in enumerate(mesh_evolution)
            ]
        }
    
    def create_3d_printing_model(self,
                                mesh_data: DiffusionMeshData,
                                print_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare model for 3D printing with diffusion visualization
        
        Args:
            mesh_data: DiffusionMeshData object
            print_settings: 3D printing settings
            
        Returns:
            3D printing preparation dictionary
        """
        if print_settings is None:
            print_settings = {
                'layer_height': 0.2,  # mm
                'infill_percentage': 20,
                'print_speed': 50,  # mm/s
                'nozzle_temperature': 210,  # Celsius
                'bed_temperature': 60,  # Celsius
                'support_required': False
            }
        
        # Check mesh suitability for 3D printing
        mesh_analysis = self._analyze_mesh_for_printing(mesh_data)
        
        # Create separate shell if needed for better visualization
        if mesh_analysis['manifold_issues'] > 0:
            print_settings['support_required'] = True
        
        # Scale model to reasonable print size (default: 100mm max dimension)
        vertices = mesh_data.vertices
        max_dimension = np.max(vertices) - np.min(vertices)
        if max_dimension > 0.1:  # 100mm
            scale_factor = 0.1 / max_dimension
            vertices = vertices * scale_factor
        else:
            scale_factor = 1.0
        
        # Create multi-material model for diffusion visualization
        diffusion_zones = self._create_diffusion_zones(mesh_data, print_settings.get('multi_material', False))
        
        printing_model = {
            'mesh_data': DiffusionMeshData(
                vertices=vertices,
                faces=mesh_data.faces,
                diffusion_field=mesh_data.diffusion_field,
                material_properties=mesh_data.material_properties,
                time_step=mesh_data.time_step,
                temperature=mesh_data.temperature
            ),
            'print_settings': print_settings,
            'mesh_analysis': mesh_analysis,
            'scale_factor': scale_factor,
            'estimated_print_time': self._estimate_print_time(mesh_data, print_settings),
            'material_usage': self._estimate_material_usage(mesh_data, print_settings),
            'diffusion_zones': diffusion_zones,
            'print recommendations': self._generate_print_recommendations(mesh_analysis)
        }
        
        return printing_model
    
    def _analyze_mesh_for_printing(self, mesh_data: DiffusionMeshData) -> Dict[str, Any]:
        """Analyze mesh for 3D printing suitability"""
        # Basic mesh analysis
        num_vertices = len(mesh_data.vertices)
        num_faces = len(mesh_data.faces)
        
        # Check for common issues
        manifold_issues = 0  # Would need proper geometry analysis
        flipped_normals = 0  # Would need normal checking
        thin_walls = 0       # Would need wall thickness analysis
        
        # Calculate mesh statistics
        vertices = mesh_data.vertices
        faces = mesh_data.faces
        
        # Calculate bounding box
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        dimensions = max_coords - min_coords
        
        # Calculate surface area
        surface_area = 0
        for face in faces:
            v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            surface_area += np.linalg.norm(normal) / 2
        
        # Calculate volume (using tetrahedron method)
        volume = 0
        origin = np.array([0, 0, 0])
        for face in faces:
            v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            volume += np.abs(np.dot(v1, np.cross(v2, v3))) / 6
        
        return {
            'num_vertices': num_vertices,
            'num_faces': num_faces,
            'manifold_issues': manifold_issues,
            'flipped_normals': flipped_normals,
            'thin_walls': thin_walls,
            'dimensions': dimensions.tolist(),
            'surface_area': surface_area,
            'volume': volume,
            'watertight': manifold_issues == 0,
            'printable': manifold_issues == 0 and volume > 0
        }
    
    def _create_diffusion_zones(self, mesh_data: DiffusionMeshData, multi_material: bool) -> List[Dict]:
        """Create diffusion zones for multi-material printing"""
        if not multi_material:
            return []
        
        # Segment mesh based on diffusion field values
        diffusion_values = mesh_data.diffusion_field
        num_zones = 3  # Low, medium, high diffusion
        
        # Create thresholds
        min_val = np.min(diffusion_values)
        max_val = np.max(diffusion_values)
        thresholds = np.linspace(min_val, max_val, num_zones + 1)
        
        zones = []
        for i in range(num_zones):
            zone_vertices = np.where((diffusion_values >= thresholds[i]) & 
                                   (diffusion_values < thresholds[i + 1]))[0]
            
            zone = {
                'zone_id': i,
                'diffusion_range': (thresholds[i], thresholds[i + 1]),
                'vertex_indices': zone_vertices.tolist(),
                'material': f'diffusion_zone_{i}',
                'color_intensity': i / (num_zones - 1) if num_zones > 1 else 0.5
            }
            zones.append(zone)
        
        return zones
    
    def _estimate_print_time(self, mesh_data: DiffusionMeshData, print_settings: Dict) -> float:
        """Estimate print time in hours"""
        layer_height = print_settings.get('layer_height', 0.2) / 1000  # Convert to meters
        print_speed = print_settings.get('print_speed', 50) / 1000 / 3600  # Convert to m/s
        
        # Get model height
        vertices = mesh_data.vertices
        model_height = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        
        num_layers = int(model_height / layer_height)
        
        # Estimate print path length (simplified)
        surface_area = self._analyze_mesh_for_printing(mesh_data)['surface_area']
        path_length = surface_area * num_layers  # Rough estimate
        
        print_time = path_length / print_speed  # in seconds
        return print_time / 3600  # Convert to hours
    
    def _estimate_material_usage(self, mesh_data: DiffusionMeshData, print_settings: Dict) -> Dict:
        """Estimate material usage for 3D printing"""
        volume = self._analyze_mesh_for_printing(mesh_data)['volume']
        infill_percentage = print_settings.get('infill_percentage', 20)
        
        # Material usage in cm³
        material_volume = volume * 1e6 * (infill_percentage / 100)
        
        # PLA density ~1.24 g/cm³
        material_weight = material_volume * 1.24
        
        return {
            'volume_cm3': material_volume,
            'weight_grams': material_weight,
            'cost_estimate': material_weight * 0.025  # $0.025 per gram PLA
        }
    
    def _generate_print_recommendations(self, mesh_analysis: Dict) -> List[str]:
        """Generate 3D printing recommendations"""
        recommendations = []
        
        if mesh_analysis['manifold_issues'] > 0:
            recommendations.append("Fix manifold issues before printing")
        
        if mesh_analysis['thin_walls'] > 0:
            recommendations.append("Consider increasing wall thickness or using finer nozzle")
        
        volume = mesh_analysis['volume']
        if volume < 1e-9:  # Very small model
            recommendations.append("Consider scaling up model for better print quality")
        elif volume > 1e-4:  # Very large model
            recommendations.append("Consider splitting model into parts for easier printing")
        
        dimensions = mesh_analysis['dimensions']
        if max(dimensions) > 0.2:  # Larger than 200mm
            recommendations.append("Model may exceed typical print bed size")
        
        if not recommendations:
            recommendations.append("Model appears suitable for 3D printing")
        
        return recommendations
    
    def launch_blender_with_file(self, filepath: str, script_path: str = None) -> bool:
        """
        Launch Blender with exported file
        
        Args:
            filepath: Path to the file to open
            script_path: Optional Python script to run
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ['blender', filepath]
            
            if script_path:
                cmd.extend(['--python', script_path])
            
            # Try to launch Blender
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return True
            else:
                print(f"Blender launch failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            # Blender launched successfully (timeout expected)
            return True
        except FileNotFoundError:
            print("Blender not found. Please install Blender to use 3D integration.")
            return False
        except Exception as e:
            print(f"Error launching Blender: {str(e)}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = tempfile.mkdtemp()
        except Exception as e:
            print(f"Error cleaning up temp files: {str(e)}")

# Convenience function for creating diffusion models
def create_diffusion_model(model_type: str, **kwargs) -> DiffusionMeshData:
    """
    Convenience function to create different types of diffusion models
    
    Args:
        model_type: Type of model ('sphere', 'cube', 'cylinder')
        **kwargs: Model-specific parameters
        
    Returns:
        DiffusionMeshData object
    """
    connector = BlenderDiffusionConnector()
    
    if model_type == 'sphere':
        return connector.create_diffusion_sphere_mesh(**kwargs)
    elif model_type == 'cube':
        return connector.create_diffusion_cube_mesh(**kwargs)
    elif model_type == 'cylinder':
        return connector.create_cylinder_diffusion_mesh(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Example usage
    connector = BlenderDiffusionConnector()
    
    # Create a diffusion sphere
    print("Creating diffusion sphere model...")
    sphere_mesh = connector.create_diffusion_sphere_mesh(
        center=(0, 0, 0),
        radius=0.01,  # 10mm
        diffusion_coefficient=1e-9,
        resolution=16
    )
    
    # Export to different formats
    print("Exporting to OBJ...")
    obj_file = connector.export_to_obj(sphere_mesh, "diffusion_sphere.obj")
    print(f"OBJ file: {obj_file}")
    
    print("Exporting to PLY...")
    ply_file = connector.export_to_ply(sphere_mesh, "diffusion_sphere.ply")
    print(f"PLY file: {ply_file}")
    
    print("Exporting to STL...")
    stl_file = connector.export_to_stl(sphere_mesh, "diffusion_sphere.stl")
    print(f"STL file: {stl_file}")
    
    # Prepare for 3D printing
    print("Preparing 3D printing model...")
    print_model = connector.create_3d_printing_model(sphere_mesh)
    
    analysis = print_model['mesh_analysis']
    print(f"Model volume: {analysis['volume']:.6f} m³")
    print(f"Surface area: {analysis['surface_area']:.6f} m²")
    print(f"Estimated print time: {print_model['estimated_print_time']:.2f} hours")
    print(f"Material usage: {print_model['material_usage']['weight_grams']:.2f} grams")
    
    print("\n3D Integration completed successfully!")
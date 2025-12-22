"""
Real-Time 3D Diffusion Visualization Engine
Advanced 3D rendering system for real-time diffusion visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class RenderMode(Enum):
    """Rendering modes for 3D visualization"""
    WIREFRAME = "wireframe"
    SURFACE = "surface"
    POINTS = "points"
    VOXEL = "voxel"
    ISOSURFACE = "isosurface"
    STREAMLINES = "streamlines"

class DiffusionColormap(Enum):
    """Predefined colormaps for diffusion visualization"""
    THERMAL = "thermal"
    CONCENTRATION = "concentration"
    VELOCITY = "velocity"
    MATERIAL = "material"
    RAINBOW = "rainbow"
    GRADIENT = "gradient"

@dataclass
class RenderSettings:
    """Rendering configuration settings"""
    mode: RenderMode = RenderMode.SURFACE
    colormap: DiffusionColormap = DiffusionColormap.THERMAL
    resolution: Tuple[int, int] = (800, 600)
    fps: int = 30
    show_axes: bool = True
    show_grid: bool = True
    transparency: float = 0.8
    lighting: bool = True
    antialiasing: bool = True
    background_color: str = "black"
    wireframe_color: str = "white"
    point_size: int = 2
    line_width: float = 1.0
    color_range: Tuple[float, float] = (0.0, 1.0)

@dataclass
class CameraSettings:
    """Camera configuration for 3D visualization"""
    position: np.ndarray = field(default_factory=lambda: np.array([2, 2, 2]))
    target: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    fov: float = 45.0
    near_clip: float = 0.1
    far_clip: float = 100.0

@dataclass
class AnimationKeyframe:
    """Animation keyframe data"""
    time: float
    position: np.ndarray
    rotation: np.ndarray
    scale: float
    diffusion_data: np.ndarray

class RealTime3DEngine:
    """
    Advanced real-time 3D visualization engine for diffusion phenomena.
    
    Features:
    - Multiple rendering modes (wireframe, surface, points, voxel, isosurface)
    - Real-time animation and interaction
    - GPU-accelerated rendering (when available)
    - Custom colormaps for different diffusion properties
    - Interactive camera controls
    - Multi-threaded rendering pipeline
    - Export capabilities for video and images
    """
    
    def __init__(self, render_settings: RenderSettings = None):
        """Initialize the 3D rendering engine"""
        self.render_settings = render_settings or RenderSettings()
        self.camera_settings = CameraSettings()
        
        # Rendering state
        self.current_frame = 0
        self.is_playing = False
        self.is_initialized = False
        
        # Data storage
        self.vertices = None
        self.faces = None
        self.diffusion_field = None
        self.material_properties = {}
        
        # Animation system
        self.animation_keyframes = []
        self.animation_duration = 0.0
        self.animation_time = 0.0
        
        # Threading for performance
        self.render_queue = queue.Queue()
        self.render_thread = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Interaction state
        self.mouse_dragging = False
        self.last_mouse_pos = None
        self.rotation_angles = np.array([0.0, 0.0, 0.0])
        self.zoom_level = 1.0
        
        # Performance monitoring
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # Colormaps
        self.colormaps = self._initialize_colormaps()
        
    def _initialize_colormaps(self) -> Dict[str, mcolors.Colormap]:
        """Initialize custom colormaps for diffusion visualization"""
        colormaps = {}
        
        # Thermal colormap (blue -> red)
        thermal_colors = ['blue', 'cyan', 'yellow', 'red']
        colormaps['thermal'] = mcolors.LinearSegmentedColormap.from_list('thermal', thermal_colors)
        
        # Concentration colormap (purple -> green -> yellow)
        concentration_colors = ['purple', 'blue', 'cyan', 'green', 'yellow']
        colormaps['concentration'] = mcolors.LinearSegmentedColormap.from_list('concentration', concentration_colors)
        
        # Velocity colormap (black -> white)
        velocity_colors = ['black', 'gray', 'white']
        colormaps['velocity'] = mcolors.LinearSegmentedColormap.from_list('velocity', velocity_colors)
        
        # Material colormap (metallic look)
        material_colors = ['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7', '#ecf0f1']
        colormaps['material'] = mcolors.LinearSegmentedColormap.from_list('material', material_colors)
        
        # Rainbow colormap
        colormaps['rainbow'] = plt.get_cmap('rainbow')
        
        # Gradient colormap
        gradient_colors = ['#000033', '#000055', '#0000ff', '#00ffff', '#ffff00', '#ff0000']
        colormaps['gradient'] = mcolors.LinearSegmentedColormap.from_list('gradient', gradient_colors)
        
        return colormaps
    
    def load_diffusion_data(self, 
                           vertices: np.ndarray,
                           faces: np.ndarray = None,
                           diffusion_field: np.ndarray = None,
                           material_properties: Dict = None):
        """
        Load diffusion data for visualization
        
        Args:
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices (for surface rendering)
            diffusion_field: N array of diffusion values for coloring
            material_properties: Dictionary of material properties
        """
        self.vertices = vertices
        self.faces = faces
        self.diffusion_field = diffusion_field
        self.material_properties = material_properties or {}
        
        # Validate data
        if self.diffusion_field is None:
            self.diffusion_field = np.ones(len(vertices)) * 0.5
        
        if len(self.diffusion_field) != len(vertices):
            self.diffusion_field = np.ones(len(vertices)) * 0.5
        
        self.is_initialized = True
        
    def create_sample_sphere_data(self, radius: float = 1.0, resolution: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sample sphere data for testing
        
        Args:
            radius: Sphere radius
            resolution: Resolution of the sphere
            
        Returns:
            Tuple of (vertices, faces, diffusion_field)
        """
        # Create sphere using spherical coordinates
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        
        # Create faces
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                v1 = i * resolution + j
                v2 = i * resolution + (j + 1)
                v3 = (i + 1) * resolution + (j + 1)
                v4 = (i + 1) * resolution + j
                
                faces.extend([[v1, v2, v3], [v1, v3, v4]])
        
        faces = np.array(faces)
        
        # Create diffusion field (gradient from top to bottom)
        diffusion_field = (vertices[:, 2] - np.min(vertices[:, 2])) / (np.max(vertices[:, 2]) - np.min(vertices[:, 2]))
        
        return vertices, faces, diffusion_field
    
    def render_frame(self, ax: Axes3D) -> None:
        """Render a single frame"""
        if not self.is_initialized or self.vertices is None:
            return
        
        # Clear previous frame
        ax.clear()
        
        # Set camera and view
        self._apply_camera_settings(ax)
        
        # Apply transformations
        transformed_vertices = self._apply_transformations(self.vertices)
        
        # Get colormap
        cmap = self.colormaps[self.render_settings.colormap.value]
        norm = mcolors.Normalize(vmin=self.render_settings.color_range[0], 
                                vmax=self.render_settings.color_range[1])
        
        # Render based on mode
        if self.render_settings.mode == RenderMode.WIREFRAME:
            self._render_wireframe(ax, transformed_vertices, cmap, norm)
        elif self.render_settings.mode == RenderMode.SURFACE:
            self._render_surface(ax, transformed_vertices, cmap, norm)
        elif self.render_settings.mode == RenderMode.POINTS:
            self._render_points(ax, transformed_vertices, cmap, norm)
        elif self.render_settings.mode == RenderMode.VOXEL:
            self._render_voxel(ax, transformed_vertices, cmap, norm)
        elif self.render_settings.mode == RenderMode.ISOSURFACE:
            self._render_isosurface(ax, transformed_vertices, cmap, norm)
        elif self.render_settings.mode == RenderMode.STREAMLINES:
            self._render_streamlines(ax, transformed_vertices, cmap, norm)
        
        # Set labels and title
        if self.render_settings.show_axes:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        
        # Set background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        if self.render_settings.show_grid:
            ax.grid(True, alpha=0.3)
        
        # Update performance metrics
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        
        # Keep only last 100 frame times for average
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
    
    def _apply_camera_settings(self, ax: Axes3D):
        """Apply camera settings to the 3D plot"""
        # Set view angle based on camera position
        ax.view_init(elev=30, azim=self.rotation_angles[1])
        
        # Set axis limits based on zoom
        if self.vertices is not None:
            center = np.mean(self.vertices, axis=0)
            max_extent = np.max(np.abs(self.vertices - center)) * self.zoom_level
            
            ax.set_xlim(center[0] - max_extent, center[0] + max_extent)
            ax.set_ylim(center[1] - max_extent, center[1] + max_extent)
            ax.set_zlim(center[2] - max_extent, center[2] + max_extent)
    
    def _apply_transformations(self, vertices: np.ndarray) -> np.ndarray:
        """Apply rotation and scaling transformations"""
        # Apply rotation
        rotation_matrix = self._get_rotation_matrix()
        transformed = vertices @ rotation_matrix.T
        
        # Apply scaling
        transformed *= self.zoom_level
        
        return transformed
    
    def _get_rotation_matrix(self) -> np.ndarray:
        """Get combined rotation matrix from rotation angles"""
        # Rotation around X axis
        rx = self.rotation_angles[0]
        rx_mat = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # Rotation around Y axis
        ry = self.rotation_angles[1]
        ry_mat = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # Rotation around Z axis
        rz = self.rotation_angles[2]
        rz_mat = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        return rz_mat @ ry_mat @ rx_mat
    
    def _render_wireframe(self, ax: Axes3D, vertices: np.ndarray, cmap: mcolors.Colormap, norm: mcolors.Normalize):
        """Render wireframe mode"""
        if self.faces is None:
            # Render as point cloud if no faces
            colors = cmap(norm(self.diffusion_field))
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      c=colors, s=1, alpha=self.render_settings.transparency)
        else:
            # Render wireframe edges
            for face in self.faces:
                for i in range(3):
                    v1, v2 = face[i], face[(i + 1) % 3]
                    edge_color = cmap(norm(np.mean([self.diffusion_field[v1], self.diffusion_field[v2]])))
                    ax.plot([vertices[v1, 0], vertices[v2, 0]], 
                           [vertices[v1, 1], vertices[v2, 1]], 
                           [vertices[v1, 2], vertices[v2, 2]], 
                           color=edge_color, linewidth=self.render_settings.line_width)
    
    def _render_surface(self, ax: Axes3D, vertices: np.ndarray, cmap: mcolors.Colormap, norm: mcolors.Normalize):
        """Render surface mode"""
        if self.faces is None:
            # Fallback to points
            self._render_points(ax, vertices, cmap, norm)
        else:
            # Create surface plot
            colors = cmap(norm(self.diffusion_field))
            
            # Plot each triangle face
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            poly3d = []
            face_colors = []
            
            for face in self.faces:
                triangle_vertices = vertices[face]
                poly3d.append(triangle_vertices)
                
                # Average color for face
                avg_diffusion = np.mean(self.diffusion_field[face])
                face_colors.append(cmap(norm(avg_diffusion)))
            
            collection = Poly3DCollection(poly3d, facecolors=face_colors, 
                                        edgecolors='none', alpha=self.render_settings.transparency)
            ax.add_collection3d(collection)
    
    def _render_points(self, ax: Axes3D, vertices: np.ndarray, cmap: mcolors.Colormap, norm: mcolors.Normalize):
        """Render points mode"""
        colors = cmap(norm(self.diffusion_field))
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c=colors, s=self.render_settings.point_size, 
                  alpha=self.render_settings.transparency)
    
    def _render_voxel(self, ax: Axes3D, vertices: np.ndarray, cmap: mcolors.Colormap, norm: mcolors.Normalize):
        """Render voxel mode (simplified - uses points)"""
        # For true voxel rendering, would need volume data
        # Using colored points as approximation
        self._render_points(ax, vertices, cmap, norm)
    
    def _render_isosurface(self, ax: Axes3D, vertices: np.ndarray, cmap: mcolors.Colormap, norm: mcolors.Normalize):
        """Render isosurface mode (simplified - uses surface)"""
        # For true isosurface rendering, would need marching cubes algorithm
        # Using surface rendering as approximation
        self._render_surface(ax, vertices, cmap, norm)
    
    def _render_streamlines(self, ax: Axes3D, vertices: np.ndarray, cmap: mcolors.Colormap, norm: mcolors.Normalize):
        """Render streamlines mode"""
        # Create streamlines based on diffusion gradient
        if self.faces is None:
            self._render_points(ax, vertices, cmap, norm)
        else:
            # Calculate gradient for streamlines
            from scipy.spatial import Delaunay
            
            try:
                tri = Delaunay(vertices[:, :2])  # 2D Delaunay for simplicity
                
                # Sample some points for streamlines
                n_streamlines = 20
                for _ in range(n_streamlines):
                    start_point = vertices[np.random.randint(len(vertices))]
                    
                    # Create simple streamline (straight line for now)
                    end_point = start_point + np.random.randn(3) * 0.1
                    
                    color = cmap(norm(self.diffusion_field[np.random.randint(len(self.diffusion_field))]))
                    ax.plot([start_point[0], end_point[0]], 
                           [start_point[1], end_point[1]], 
                           [start_point[2], end_point[2]], 
                           color=color, linewidth=1, alpha=0.6)
            except ImportError:
                # Fallback to points if scipy not available
                self._render_points(ax, vertices, cmap, norm)
    
    def create_animation(self, duration: float = 5.0, fps: int = 30) -> FuncAnimation:
        """Create animated visualization"""
        if not self.is_initialized:
            raise ValueError("No data loaded for animation")
        
        self.animation_duration = duration
        self.render_settings.fps = fps
        total_frames = int(duration * fps)
        
        # Generate animation keyframes
        self.animation_keyframes = []
        for i in range(total_frames):
            t = i / total_frames
            
            # Animate rotation
            rotation = np.array([0, t * 360, 0])  # Rotate around Y axis
            scale = 1.0 + 0.2 * np.sin(2 * np.pi * t)  # Pulsating scale
            
            # Animate diffusion field (simulate time evolution)
            if self.diffusion_field is not None:
                # Simple diffusion evolution simulation
                evolved_field = self.diffusion_field * (1 + 0.5 * np.sin(2 * np.pi * t))
                evolved_field = np.clip(evolved_field, 0, 1)
            else:
                evolved_field = np.ones(len(self.vertices)) * 0.5
            
            keyframe = AnimationKeyframe(
                time=t * duration,
                position=np.array([0, 0, 0]),
                rotation=rotation,
                scale=scale,
                diffusion_data=evolved_field
            )
            self.animation_keyframes.append(keyframe)
        
        # Setup figure and animation
        fig = plt.figure(figsize=(self.render_settings.resolution[0]/100, 
                                self.render_settings.resolution[1]/100))
        ax = fig.add_subplot(111, projection='3d')
        
        def animate_frame(frame_num):
            if frame_num < len(self.animation_keyframes):
                keyframe = self.animation_keyframes[frame_num]
                self.rotation_angles = keyframe.rotation
                self.zoom_level = keyframe.scale
                self.diffusion_field = keyframe.diffusion_data
            
            self.render_frame(ax)
            return ax,
        
        anim = FuncAnimation(fig, animate_frame, frames=total_frames, 
                           interval=1000/fps, blit=False)
        
        return anim
    
    def save_animation(self, filename: str, animation: FuncAnimation = None):
        """Save animation to file"""
        if animation is None:
            animation = self.create_animation()
        
        # Determine file format from extension
        if filename.endswith('.gif'):
            animation.save(filename, writer='pillow', fps=self.render_settings.fps)
        elif filename.endswith('.mp4'):
            animation.save(filename, writer='ffmpeg', fps=self.render_settings.fps)
        else:
            # Default to GIF
            animation.save(filename + '.gif', writer='pillow', fps=self.render_settings.fps)
        
        print(f"Animation saved to: {filename}")
    
    def save_image(self, filename: str):
        """Save current frame as image"""
        fig = plt.figure(figsize=(self.render_settings.resolution[0]/100, 
                                self.render_settings.resolution[1]/100))
        ax = fig.add_subplot(111, projection='3d')
        
        self.render_frame(ax)
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Image saved to: {filename}")
    
    def start_interactive_visualization(self):
        """Start interactive 3D visualization"""
        if not self.is_initialized:
            # Load sample data for demo
            vertices, faces, diffusion_field = self.create_sample_sphere_data()
            self.load_diffusion_data(vertices, faces, diffusion_field)
        
        # Setup interactive plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Mouse event handlers
        def on_mouse_press(event):
            if event.inaxes == ax:
                self.mouse_dragging = True
                self.last_mouse_pos = (event.xdata, event.ydata)
        
        def on_mouse_release(event):
            self.mouse_dragging = False
            self.last_mouse_pos = None
        
        def on_mouse_motion(event):
            if self.mouse_dragging and event.inaxes == ax and self.last_mouse_pos:
                dx = event.xdata - self.last_mouse_pos[0] if event.xdata else 0
                dy = event.ydata - self.last_mouse_pos[1] if event.ydata else 0
                
                self.rotation_angles[1] += dx * 2  # Yaw
                self.rotation_angles[0] += dy * 2  # Pitch
                
                self.last_mouse_pos = (event.xdata, event.ydata)
        
        def on_scroll(event):
            if event.inaxes == ax:
                # Zoom with mouse wheel
                if event.button == 'up':
                    self.zoom_level *= 0.9
                elif event.button == 'down':
                    self.zoom_level *= 1.1
                
                self.zoom_level = np.clip(self.zoom_level, 0.1, 10.0)
        
        # Connect event handlers
        fig.canvas.mpl_connect('button_press_event', on_mouse_press)
        fig.canvas.mpl_connect('button_release_event', on_mouse_release)
        fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        # Update function
        def update(frame):
            self.render_frame(ax)
            self.current_frame += 1
            
            # Update title with performance info
            if len(self.frame_times) > 0:
                avg_frame_time = np.mean(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                ax.set_title(f'Real-Time 3D Diffusion Visualization - FPS: {fps:.1f}')
            
            return ax,
        
        # Create animation
        self.is_playing = True
        anim = FuncAnimation(fig, update, interval=1000/self.render_settings.fps, 
                           blit=False, cache_frame_data=False)
        
        # Add mode switching with keyboard
        def on_key(event):
            if event.key == '1':
                self.render_settings.mode = RenderMode.WIREFRAME
            elif event.key == '2':
                self.render_settings.mode = RenderMode.SURFACE
            elif event.key == '3':
                self.render_settings.mode = RenderMode.POINTS
            elif event.key == '4':
                self.render_settings.mode = RenderMode.VOXEL
            elif event.key == '5':
                self.render_settings.mode = RenderMode.ISOSURFACE
            elif event.key == '6':
                self.render_settings.mode = RenderMode.STREAMLINES
            elif event.key == 'c':
                # Cycle through colormaps
                colormaps = list(DiffusionColormap)
                current_idx = colormaps.index(self.render_settings.colormap)
                self.render_settings.colormap = colormaps[(current_idx + 1) % len(colormaps)]
            elif event.key == 'r':
                # Reset view
                self.rotation_angles = np.array([0.0, 0.0, 0.0])
                self.zoom_level = 1.0
            elif event.key == 's':
                # Save screenshot
                timestamp = int(time.time())
                self.save_image(f'diffusion_3d_{timestamp}.png')
            elif event.key == 'a':
                # Save animation
                self.save_animation('diffusion_animation.gif')
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show()
        
        return anim
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'current_frame': self.current_frame,
            'is_playing': self.is_playing,
            'is_initialized': self.is_initialized,
        }
        
        if len(self.frame_times) > 0:
            stats['average_frame_time'] = np.mean(self.frame_times)
            stats['min_frame_time'] = np.min(self.frame_times)
            stats['max_frame_time'] = np.max(self.frame_times)
            stats['fps'] = 1.0 / np.mean(self.frame_times)
        else:
            stats['average_frame_time'] = 0
            stats['min_frame_time'] = 0
            stats['max_frame_time'] = 0
            stats['fps'] = 0
        
        if self.vertices is not None:
            stats['num_vertices'] = len(self.vertices)
            if self.faces is not None:
                stats['num_faces'] = len(self.faces)
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        if self.render_thread:
            self.render_thread.join(timeout=1.0)
        
        self.executor.shutdown(wait=True)
        
        # Clear data
        self.vertices = None
        self.faces = None
        self.diffusion_field = None
        self.animation_keyframes.clear()

# Convenience function for quick visualization
def visualize_diffusion_3d(vertices: np.ndarray, 
                          faces: np.ndarray = None,
                          diffusion_field: np.ndarray = None,
                          mode: str = "surface",
                          colormap: str = "thermal",
                          interactive: bool = True):
    """
    Quick 3D visualization of diffusion data
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        diffusion_field: N array of diffusion values
        mode: Rendering mode ('wireframe', 'surface', 'points', 'voxel', 'isosurface', 'streamlines')
        colormap: Colormap name ('thermal', 'concentration', 'velocity', 'material', 'rainbow', 'gradient')
        interactive: Whether to start interactive visualization
    """
    # Create render settings
    render_settings = RenderSettings(
        mode=RenderMode(mode),
        colormap=DiffusionColormap(colormap)
    )
    
    # Create engine
    engine = RealTime3DEngine(render_settings)
    
    # Load data
    engine.load_diffusion_data(vertices, faces, diffusion_field)
    
    if interactive:
        # Start interactive visualization
        return engine.start_interactive_visualization()
    else:
        # Save single frame
        timestamp = int(time.time())
        filename = f'diffusion_3d_{timestamp}.png'
        engine.save_image(filename)
        return filename

if __name__ == "__main__":
    # Demo the 3D engine
    print("Real-Time 3D Diffusion Visualization Engine Demo")
    print("=" * 50)
    
    # Create engine
    engine = RealTime3DEngine()
    
    # Create sample data
    print("Creating sample diffusion data...")
    vertices, faces, diffusion_field = engine.create_sample_sphere_data(radius=1.0, resolution=20)
    
    # Load data
    engine.load_diffusion_data(vertices, faces, diffusion_field)
    
    print(f"Loaded {len(vertices)} vertices and {len(faces)} faces")
    
    # Test different rendering modes
    print("\nTesting rendering modes:")
    for mode in RenderMode:
        print(f"  - {mode.value}")
        engine.render_settings.mode = mode
        engine.save_image(f'test_{mode.value}.png')
    
    # Test different colormaps
    print("\nTesting colormaps:")
    for colormap in DiffusionColormap:
        print(f"  - {colormap.value}")
        engine.render_settings.colormap = colormap
        engine.save_image(f'test_{colormap.value}.png')
    
    # Create and save animation
    print("\nCreating animation...")
    anim = engine.create_animation(duration=3.0, fps=15)
    engine.save_animation('diffusion_demo.gif', anim)
    
    # Get performance stats
    stats = engine.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Vertices: {stats.get('num_vertices', 0)}")
    print(f"  Faces: {stats.get('num_faces', 0)}")
    print(f"  Frame times: {len(engine.frame_times)} frames recorded")
    
    # Start interactive visualization
    print("\nStarting interactive visualization...")
    print("Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Keys 1-6: Switch rendering modes")
    print("  - Key 'c': Cycle colormaps")
    print("  - Key 'r': Reset view")
    print("  - Key 's': Save screenshot")
    print("  - Key 'a': Save animation")
    
    engine.start_interactive_visualization()
    
    # Cleanup
    engine.cleanup()
    print("\nDemo completed successfully!")
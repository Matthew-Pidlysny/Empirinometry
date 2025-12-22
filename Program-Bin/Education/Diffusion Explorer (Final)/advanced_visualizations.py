"""
Advanced Visualization System for Diffusion Navigator
Enhanced visualizations with interactive features and animations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional, Callable
import threading
import queue
import time

class AdvancedVisualizer:
    """Enhanced visualization system with interactive features"""
    
    def __init__(self):
        self.animation_queue = queue.Queue()
        self.current_animation = None
        self.interactive_data = {}
        
    def create_animated_diffusion_3d(self, material_data: Dict, 
                                   temperature_range: np.ndarray,
                                   save_animation: bool = False) -> FuncAnimation:
        """
        Create 3D animated diffusion visualization
        Shows how diffusion patterns evolve with temperature
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D grid
        x = np.linspace(-1, 1, 30)
        y = np.linspace(-1, 1, 30)
        X, Y = np.meshgrid(x, y)
        
        # Initial surface (low temperature)
        initial_D = material_data['diffusion_coefficients'][0]
        Z = initial_D * np.exp(-(X**2 + Y**2) / 0.5)
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Diffusion Coefficient')
        ax.set_title(f'3D Diffusion Evolution - {material_data["name"]}')
        
        # Animation update function
        def update(frame):
            ax.clear()
            
            # Calculate current diffusion coefficient
            temp_idx = min(frame, len(temperature_range) - 1)
            current_temp = temperature_range[temp_idx]
            D = material_data['diffusion_coefficients'][temp_idx]
            
            # Update surface
            Z = D * np.exp(-(X**2 + Y**2) / 0.5)
            
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                                 vmin=0, vmax=max(material_data['diffusion_coefficients']))
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Diffusion Coefficient')
            ax.set_title(f'3D Diffusion Evolution - {material_data["name"]}\nT = {current_temp:.0f} K')
            
            return [surf]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(temperature_range),
                           interval=200, blit=False, repeat=True)
        
        if save_animation:
            anim.save('diffusion_3d_animation.gif', writer='pillow', fps=5)
        
        self.current_animation = anim
        return anim
    
    def create_interactive_temperature_control(self, materials_data: Dict) -> go.Figure:
        """
        Create interactive plot with temperature slider
        """
        # Get temperature range from data
        all_temps = []
        for material_name, data in materials_data.items():
            all_temps.extend(data['temperatures'])
        
        temp_min, temp_max = min(all_temps), max(all_temps)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Diffusion Coefficients', 'Arrhenius Plot', 
                          'Material Comparison', 'Temperature Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Diffusion coefficients vs temperature
        for material_name, data in materials_data.items():
            fig.add_trace(
                go.Scatter(
                    x=data['temperatures'],
                    y=data['diffusion_coefficients'],
                    mode='lines+markers',
                    name=material_name,
                    line=dict(width=2),
                    hovertemplate=f'<b>{material_name}</b><br>' +
                                 'Temperature: %{x:.0f}K<br>' +
                                 'D: %{y:.2e} m²/s<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Arrhenius plot
        for material_name, data in materials_data.items():
            inv_T = 1.0 / np.array(data['temperatures'])
            ln_D = np.log(np.array(data['diffusion_coefficients']))
            
            fig.add_trace(
                go.Scatter(
                    x=inv_T,
                    y=ln_D,
                    mode='lines+markers',
                    name=f'{material_name} (Arrhenius)',
                    line=dict(width=2),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: Bar comparison at specific temperature
        mid_temp = (temp_min + temp_max) / 2
        material_names = []
        mid_D_values = []
        
        for material_name, data in materials_data.items():
            # Interpolate to get value at mid temperature
            temps = np.array(data['temperatures'])
            D_values = np.array(data['diffusion_coefficients'])
            mid_D = np.interp(mid_temp, temps, D_values)
            
            material_names.append(material_name)
            mid_D_values.append(mid_D)
        
        fig.add_trace(
            go.Bar(
                x=material_names,
                y=mid_D_values,
                name=f'D at {mid_temp:.0f}K',
                marker_color='lightblue',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Temperature distribution histogram
        all_D_values = []
        for data in materials_data.values():
            all_D_values.extend(data['diffusion_coefficients'])
        
        fig.add_trace(
            go.Histogram(
                x=np.log10(all_D_values),
                nbinsx=20,
                name='log₁₀(D) Distribution',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Interactive Diffusion Analysis Dashboard',
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Temperature (K)", row=1, col=1)
        fig.update_yaxes(title_text="D (m²/s)", type="log", row=1, col=1)
        
        fig.update_xaxes(title_text="1/T (K⁻¹)", row=1, col=2)
        fig.update_yaxes(title_text="ln(D)", row=1, col=2)
        
        fig.update_xaxes(title_text="Materials", row=2, col=1)
        fig.update_yaxes(title_text="D (m²/s)", type="log", row=2, col=1)
        
        fig.update_xaxes(title_text="log₁₀(D)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        return fig
    
    def create_diffusion_heatmap_3d(self, materials: List[str], 
                                  temperatures: np.ndarray,
                                  diffusion_data: Dict) -> go.Figure:
        """
        Create 3D heatmap of diffusion coefficients
        """
        # Create 3D data matrix
        n_materials = len(materials)
        n_temps = len(temperatures)
        
        # Create meshgrid
        material_indices, temp_indices = np.meshgrid(
            np.arange(n_materials), np.arange(n_temps)
        )
        
        # Create diffusion coefficient matrix
        diffusion_matrix = np.zeros((n_temps, n_materials))
        
        for i, material in enumerate(materials):
            if material in diffusion_data:
                diffusion_matrix[:, i] = diffusion_data[material]['diffusion_coefficients']
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=material_indices,
            y=temp_indices,
            z=np.log10(diffusion_matrix),
            colorscale='Viridis',
            colorbar=dict(title="log₁₀(D) [m²/s]"),
            hovertemplate=(
                'Material: %{x}<br>' +
                'Temp Index: %{y}<br>' +
                'log₁₀(D): %{z:.2f}<extra></extra>'
            )
        )])
        
        # Update layout
        fig.update_layout(
            title='3D Diffusion Coefficient Heatmap',
            scene=dict(
                xaxis=dict(title='Materials', ticktext=materials, tickvals=np.arange(n_materials)),
                yaxis=dict(title='Temperature Index', ticktext=[f'{t:.0f}K' for t in temperatures[::len(temperatures)//5]], 
                          tickvals=np.arange(0, len(temperatures), len(temperatures)//5)),
                zaxis=dict(title='log₁₀(D) [m²/s]')
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_cluster_evolution_animation(self, cluster_data_history: List[Dict]) -> FuncAnimation:
        """
        Create animation showing cluster evolution over time
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            current_clusters = cluster_data_history[frame]
            
            # Plot 1: Cluster centers evolution
            if 'cluster_centers' in current_clusters:
                centers = current_clusters['cluster_centers']
                n_clusters = len(centers)
                
                colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
                
                for i, center in enumerate(centers):
                    if len(center) >= 2:
                        ax1.scatter(center[0], center[1], s=200, c=[colors[i]], 
                                  label=f'Cluster {i}', alpha=0.7)
                        
                        # Add trajectory if we have history
                        if frame > 0:
                            prev_centers = cluster_data_history[frame-1].get('cluster_centers', [])
                            if i < len(prev_centers) and len(prev_centers[i]) >= 2:
                                ax1.plot([prev_centers[i][0], center[0]], 
                                       [prev_centers[i][1], center[1]], 
                                       'k-', alpha=0.3, linewidth=1)
            
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            ax1.set_title(f'Cluster Centers - Frame {frame}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Material assignment
            if 'clusters' in current_clusters:
                materials_per_cluster = []
                cluster_labels = []
                
                for cluster_id, cluster_info in current_clusters['clusters'].items():
                    materials_per_cluster.append(len(cluster_info['materials']))
                    cluster_labels.append(f'C{cluster_id}')
                
                bars = ax2.bar(cluster_labels, materials_per_cluster, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(cluster_labels))))
                
                # Add value labels
                for bar, value in zip(bars, materials_per_cluster):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value}', ha='center', va='bottom')
            
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Number of Materials')
            ax2.set_title(f'Material Distribution - Frame {frame}')
            ax2.grid(True, alpha=0.3)
        
        anim = FuncAnimation(fig, update, frames=len(cluster_data_history),
                           interval=500, blit=False, repeat=True)
        
        plt.tight_layout()
        return anim
    
    def create_material_property_radar_advanced(self, materials_data: Dict[str, Dict]) -> go.Figure:
        """
        Create advanced radar chart for material comparison
        """
        fig = go.Figure()
        
        # Properties to compare
        properties = ['Density', 'D_coefficient', 'Activation_E', 'Atomic_Radius', 'Melting_Point']
        
        # Add mock melting point if not present
        for material_name, data in materials_data.items():
            if 'melting_point' not in data:
                # Assign realistic melting points based on material type
                if 'Iron' in material_name:
                    data['melting_point'] = 1811
                elif 'Aluminum' in material_name:
                    data['melting_point'] = 933
                elif 'Copper' in material_name:
                    data['melting_point'] = 1358
                elif 'Gold' in material_name:
                    data['melting_point'] = 1337
                elif 'Silver' in material_name:
                    data['melting_point'] = 1235
                else:
                    data['melting_point'] = 1000  # Default
        
        # Normalize properties
        prop_ranges = {
            'Density': (0, 20),
            'D_coefficient': (1e-10, 1),
            'Activation_E': (0, 6),
            'Atomic_Radius': (0, 0.2),
            'Melting_Point': (0, 4000)
        }
        
        # Add traces for each material
        colors = px.colors.qualitative.Set1
        for i, (material_name, data) in enumerate(materials_data.items()):
            values = []
            
            for prop in properties:
                if prop == 'Density':
                    val = data['density']
                elif prop == 'D_coefficient':
                    val = np.log10(data['diffusion_coefficient'])
                elif prop == 'Activation_E':
                    val = data['activation_energy']
                elif prop == 'Atomic_Radius':
                    val = data['atomic_radius']
                elif prop == 'Melting_Point':
                    val = data['melting_point']
                
                # Normalize to 0-1 range
                min_val, max_val = prop_ranges[prop]
                normalized_val = (val - min_val) / (max_val - min_val)
                values.append(normalized_val)
            
            # Close the radar chart
            values += values[:1]
            properties_closed = properties + [properties[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=properties_closed,
                fill='toself',
                name=material_name,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Advanced Material Properties Comparison",
            showlegend=True,
            width=700,
            height=600
        )
        
        return fig
    
    def create_diffusion_mechanism_visualization(self) -> go.Figure:
        """
        Create visualization showing different diffusion mechanisms
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Vacancy Diffusion', 'Interstitial Diffusion', 
                          'Grain Boundary Diffusion', 'Surface Diffusion'),
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}],
                   [{"type": "scatter3d"}, {"type": "scatter3d"}]]
        )
        
        # Vacancy diffusion
        x_vac = [0, 1, 0, -1, 0]
        y_vac = [0, 0, 1, 0, -1]
        z_vac = [0, 0, 0, 0, 0]
        
        fig.add_trace(
            go.Scatter3d(
                x=x_vac, y=y_vac, z=z_vac,
                mode='markers+lines',
                marker=dict(size=8, color='red'),
                line=dict(color='red', width=2),
                name='Vacancy',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Interstitial diffusion
        x_int = [0.5, 1.5, 0.5, -0.5, 0.5]
        y_int = [0.5, 0.5, 1.5, 0.5, -0.5]
        z_int = [0.5, 0.5, 0.5, 0.5, 0.5]
        
        fig.add_trace(
            go.Scatter3d(
                x=x_int, y=y_int, z=z_int,
                mode='markers+lines',
                marker=dict(size=8, color='blue'),
                line=dict(color='blue', width=2),
                name='Interstitial',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Grain boundary diffusion
        x_gb = [0, 1, 2, 2, 1, 0, -1, -2, -2, -1, 0]
        y_gb = [-1, -1, 0, 1, 1, 1, 1, 0, -1, -1, -1]
        z_gb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        fig.add_trace(
            go.Scatter3d(
                x=x_gb, y=y_gb, z=z_gb,
                mode='markers+lines',
                marker=dict(size=6, color='green'),
                line=dict(color='green', width=2),
                name='Grain Boundary',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Surface diffusion
        theta = np.linspace(0, 2*np.pi, 20)
        x_surf = np.cos(theta)
        y_surf = np.sin(theta)
        z_surf = np.zeros_like(theta)
        
        fig.add_trace(
            go.Scatter3d(
                x=x_surf, y=y_surf, z=z_surf,
                mode='markers+lines',
                marker=dict(size=8, color='orange'),
                line=dict(color='orange', width=2),
                name='Surface',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Diffusion Mechanisms Visualization',
            height=800,
            showlegend=False
        )
        
        # Update 3D scene settings
        for i in [1, 2]:
            for j in [1, 2]:
                fig.update_scenes(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z'),
                    aspectmode='cube',
                    row=i, col=j
                )
        
        return fig
    
    def create_time_evolution_plot(self, concentration_data: np.ndarray, 
                                 positions: np.ndarray,
                                 times: np.ndarray) -> go.Figure:
        """
        Create time evolution plot of diffusion process
        """
        fig = go.Figure()
        
        # Plot concentration profiles at different times
        n_snapshots = min(8, len(times))
        time_indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
        
        for i, time_idx in enumerate(time_indices):
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=concentration_data[time_idx],
                    mode='lines',
                    name=f't = {times[time_idx]:.2f}',
                    line=dict(color=f'rgb({colors[i][0]*255:.0f}, {colors[i][1]*255:.0f}, {colors[i][2]*255:.0f})', 
                            width=2),
                    hovertemplate=f'Time: {times[time_idx]:.2f}<br>' +
                                 'Position: %{x:.3f}<br>' +
                                 'Concentration: %{y:.3f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title='Concentration Profile Evolution',
            xaxis_title='Position',
            yaxis_title='Concentration',
            hovermode='x unified',
            width=800,
            height=500
        )
        
        return fig
    
    def export_interactive_html(self, fig: go.Figure, filename: str):
        """Export interactive plot to HTML"""
        fig.write_html(filename)
        print(f"Interactive plot saved to {filename}")
    
    def stop_animation(self):
        """Stop current animation"""
        if self.current_animation:
            self.current_animation.event_source.stop()
    
    def get_animation_status(self) -> Dict:
        """Get current animation status"""
        if self.current_animation:
            return {
                'running': self.current_animation.event_source is not None,
                'frame_count': self.current_animation.frame_seq.__len__(),
                'interval': self.current_animation.interval
            }
        return {'running': False}
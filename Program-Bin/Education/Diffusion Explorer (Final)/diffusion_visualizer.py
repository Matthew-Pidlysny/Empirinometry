"""
Advanced Diffusion Visualization System
Provides efficient visualizations for diffusion sphere, patterns, and clusters
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import json

class DiffusionVisualizer:
    """Advanced visualization system for diffusion analysis"""
    
    def __init__(self):
        self.color_palette = sns.color_palette("husl", 12)
        self.figure_size = (12, 8)
        self.dpi = 100
        
    def create_diffusion_sphere(self, sphere_data: Dict, layer_assignments: Dict) -> go.Figure:
        """
        Create comprehensive diffusion sphere visualization
        Shows all layers with their diffusion properties
        """
        fig = go.Figure()
        
        # Generate sphere coordinates
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        
        for layer_idx, layer_data in layer_assignments.items():
            # Calculate layer radius (expanding sphere)
            layer_radius = 1.0 + layer_idx * 0.1
            
            # Create sphere mesh
            x = layer_radius * np.outer(np.cos(u), np.sin(v))
            y = layer_radius * np.outer(np.sin(u), np.sin(v))
            z = layer_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Calculate color based on diffusion values
            if layer_data:
                avg_diffusion = np.mean(list(layer_data.values()))
                color_intensity = avg_diffusion
            else:
                color_intensity = 0.5
            
            # Add layer to plot
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.6,
                colorscale='Viridis',
                showscale=False if layer_idx > 0 else True,
                name=f'Layer {layer_idx}',
                surfacecolor=np.full_like(x, color_intensity),
                cmin=0, cmax=1
            ))
        
        fig.update_layout(
            title='Diffusion Sphere - Multi-Layer Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_cluster_visualization(self, cluster_data: Dict, diffusion_data: Dict) -> plt.Figure:
        """
        Create diffusion pattern plot with organizational clusters
        Shows material clustering based on diffusion characteristics
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        # Prepare data for clustering visualization
        materials = []
        features = []
        colors = []
        cluster_labels = []
        
        for cluster_id, cluster_info in cluster_data['clusters'].items():
            cluster_color = self.color_palette[cluster_id % len(self.color_palette)]
            
            for material in cluster_info['materials']:
                if material in diffusion_data:
                    materials.append(material)
                    features.append(diffusion_data[material].get('normalized_diffusion', [0.5]))
                    colors.append(cluster_color)
                    cluster_labels.append(f"Cluster {cluster_id}")
        
        # Plot 1: 2D PCA projection
        if len(features) > 1:
            # Pad features to same length
            max_len = max(len(f) for f in features)
            features_padded = np.array([f + [0] * (max_len - len(f)) for f in features])
            
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_padded)
            
            scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=colors, s=100, alpha=0.7)
            
            # Add material labels
            for i, material in enumerate(materials):
                ax1.annotate(material, (features_2d[i, 0], features_2d[i, 1]), 
                           fontsize=8, alpha=0.8)
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax1.set_title('Material Clusters - PCA Projection')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Temperature-dependent diffusion heatmap
        if materials:
            # Create matrix for heatmap
            matrix_data = []
            material_names_plot = []
            
            for material in materials[:10]:  # Limit to 10 for readability
                if material in diffusion_data:
                    matrix_data.append(diffusion_data[material].get('normalized_diffusion', [0.5]))
                    material_names_plot.append(material)
            
            if matrix_data:
                matrix_data = np.array(matrix_data)
                
                # Create temperature labels
                temps = np.linspace(200, 1200, len(matrix_data[0]))
                
                im = ax2.imshow(matrix_data, cmap='viridis', aspect='auto', interpolation='nearest')
                if len(temps) > 5:
                    ax2.set_xticks(range(0, len(temps), len(temps)//5))
                ax2.set_xticklabels([f'{t:.0f}K' for t in temps[::len(temps)//5]])
                ax2.set_yticks(range(len(material_names_plot)))
                ax2.set_yticklabels(material_names_plot)
                ax2.set_title('Temperature-Dependent Diffusion')
                ax2.set_xlabel('Temperature')
                ax2.set_ylabel('Materials')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Normalized Diffusion Coefficient')
        
        plt.tight_layout()
        return fig
    
    def create_layer_by_layer_analysis(self, layer_assignments: Dict) -> plt.Figure:
        """
        Create layer-by-layer diffusion analysis visualization
        """
        n_layers = len(layer_assignments)
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(15, 8))
        if n_layers == 1:
            axes = [axes]
        elif n_layers <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for layer_idx, (layer_num, layer_data) in enumerate(layer_assignments.items()):
            if layer_idx >= len(axes):
                break
                
            ax = axes[layer_idx]
            
            if layer_data:
                materials = list(layer_data.keys())
                values = list(layer_data.values())
                colors = [self.color_palette[i % len(self.color_palette)] for i in range(len(materials))]
                
                bars = ax.bar(materials[:8], values[:8], color=colors[:8])  # Limit to 8 materials
                ax.set_title(f'Layer {layer_num}')
                ax.set_ylabel('Diffusion Value')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values[:8]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No data for Layer {layer_num}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Layer {layer_num}')
        
        # Hide unused subplots
        for i in range(len(layer_assignments), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Layer-by-Layer Diffusion Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_interactive_diffusion_plot(self, diffusion_data: Dict, selected_materials: List[str]) -> go.Figure:
        """
        Create interactive plot for exploring diffusion mechanics
        """
        fig = go.Figure()
        
        temperatures = np.linspace(200, 1200, 100)
        
        for material in selected_materials:
            if material in diffusion_data:
                data = diffusion_data[material]
                
                # Interpolate for smooth curves
                if 'diffusion_coefficients' in data and len(data['diffusion_coefficients']) > 0:
                    temps_orig = data.get('temperatures', temperatures)
                    D_orig = data['diffusion_coefficients']
                    
                    # Create smooth interpolation
                    D_smooth = np.interp(temperatures, temps_orig, D_orig)
                    
                    fig.add_trace(go.Scatter(
                        x=temperatures,
                        y=D_smooth,
                        mode='lines+markers',
                        name=material,
                        line=dict(width=2),
                        marker=dict(size=4),
                        hovertemplate=f'<b>{material}</b><br>' +
                                     'Temperature: %{x:.0f}K<br>' +
                                     'D: %{y:.2e} m²/s<extra></extra>'
                    ))
        
        fig.update_layout(
            title='Temperature-Dependent Diffusion Coefficients',
            xaxis_title='Temperature (K)',
            yaxis_title='Diffusion Coefficient (m²/s)',
            yaxis_type='log',
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
            width=800,
            height=500
        )
        
        return fig
    
    def create_material_comparison_radar(self, materials_data: Dict) -> plt.Figure:
        """
        Create radar chart for material property comparison
        """
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Properties to compare
        properties = ['Density', 'D_coefficient', 'Activation_E', 'Atomic_Radius']
        materials = list(materials_data.keys())[:6]  # Limit to 6 materials
        
        # Normalize properties for radar chart
        prop_ranges = {
            'Density': (0, 20),
            'D_coefficient': (1e-10, 1),
            'Activation_E': (0, 6),
            'Atomic_Radius': (0, 0.2)
        }
        
        angles = np.linspace(0, 2 * np.pi, len(properties), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, material in enumerate(materials):
            data = materials_data[material]
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
                
                # Normalize to 0-1 range
                min_val, max_val = prop_ranges[prop]
                normalized_val = (val - min_val) / (max_val - min_val)
                values.append(normalized_val)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=material, color=self.color_palette[i])
            ax.fill(angles, values, alpha=0.25, color=self.color_palette[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(properties)
        ax.set_ylim(0, 1)
        ax.set_title('Material Properties Comparison', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        return fig
    
    def create_diffusion_mechanics_explanation(self, calculation_steps: List[Dict]) -> plt.Figure:
        """
        Create visualization for step-by-step diffusion mechanics
        """
        n_steps = len(calculation_steps)
        fig, axes = plt.subplots(n_steps, 1, figsize=(10, 3*n_steps))
        
        if n_steps == 1:
            axes = [axes]
        
        for i, step in enumerate(calculation_steps):
            ax = axes[i]
            
            # Create visual representation of the calculation
            step_text = f"Step {step['step']}: {step['description']}"
            calc_text = step['calculation']
            
            ax.text(0.05, 0.7, step_text, fontsize=12, fontweight='bold', 
                   transform=ax.transAxes, wrap=True)
            ax.text(0.05, 0.3, calc_text, fontsize=10, 
                   transform=ax.transAxes, wrap=True, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
            
            # Add visual element based on step type
            if 'matrix' in step['description'].lower():
                # Show matrix visualization
                if isinstance(step['result'], np.ndarray):
                    im = ax.imshow(step['result'], cmap='viridis', aspect='auto', 
                                 extent=[0, 1, 0, 0.1])
                    ax.text(0.7, 0.05, 'Matrix Visualization', fontsize=8,
                           transform=ax.transAxes, color='white')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        plt.suptitle('Diffusion Calculation Steps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def export_visualization(self, fig, filename: str, format: str = 'png'):
        """Export visualization to file"""
        if hasattr(fig, 'write_html'):  # Plotly figure
            if format.lower() == 'html':
                fig.write_html(filename)
            else:
                fig.write_image(filename)
        else:  # Matplotlib figure
            fig.savefig(filename, format=format, dpi=self.dpi, bbox_inches='tight')
    
    def create_summary_dashboard(self, all_data: Dict) -> plt.Figure:
        """
        Create comprehensive summary dashboard
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Material count pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        if 'materials_count' in all_data:
            counts = all_data['materials_count']
            ax1.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
            ax1.set_title('Material Categories')
        
        # 2. Temperature range distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if 'temperature_data' in all_data:
            temps = all_data['temperature_data']
            ax2.hist(temps, bins=20, alpha=0.7, color='skyblue')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Temperature Distribution')
        
        # 3. Diffusion coefficient distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if 'diffusion_coeffs' in all_data:
            coeffs = all_data['diffusion_coeffs']
            ax3.hist(np.log10(coeffs), bins=20, alpha=0.7, color='lightgreen')
            ax3.set_xlabel('log₁₀(D) [m²/s]')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Diffusion Coefficient Distribution')
        
        # 4. Cluster summary
        ax4 = fig.add_subplot(gs[1, :2])
        if 'cluster_analysis' in all_data:
            cluster_data = all_data['cluster_analysis']
            # Create cluster visualization
            for cluster_id, cluster_info in cluster_data['clusters'].items():
                materials_count = len(cluster_info['materials'])
                ax4.bar(f'Cluster {cluster_id}', materials_count, 
                       color=self.color_palette[cluster_id % len(self.color_palette)])
            ax4.set_ylabel('Number of Materials')
            ax4.set_title('Material Distribution Across Clusters')
        
        # 5. Performance metrics
        ax5 = fig.add_subplot(gs[1, 2])
        metrics = ['Accuracy', 'Speed', 'Memory', 'Stability']
        values = [0.95, 0.87, 0.92, 0.89]  # Example metrics
        colors = ['green' if v > 0.9 else 'yellow' if v > 0.8 else 'red' for v in values]
        bars = ax5.bar(metrics, values, color=colors, alpha=0.7)
        ax5.set_ylim(0, 1)
        ax5.set_title('Performance Metrics')
        ax5.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 6. Summary statistics table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        if 'summary_stats' in all_data:
            stats = all_data['summary_stats']
            table_data = []
            for key, value in stats.items():
                table_data.append([key, str(value)])
            
            table = ax6.table(cellText=table_data, 
                            colLabels=['Metric', 'Value'],
                            cellLoc='left', loc='center',
                            colWidths=[0.4, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Diffusion Navigator - Summary Dashboard', fontsize=16, fontweight='bold')
        return fig
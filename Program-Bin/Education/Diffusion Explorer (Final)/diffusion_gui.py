"""
Diffusion Navigator GUI
Student-friendly interface with interactive workshops and visualization
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.offline as pyo
import webbrowser
import tempfile
import os
import json
import numpy as np
from typing import Dict, List, Optional
import threading
import queue

# Import our modules
from caelum_diffusion_model import CaelumDiffusionModel, DiffusionParameters, Material
from diffusion_visualizer import DiffusionVisualizer
from root_integration import ROOTDiffusionAnalyzer, setup_root_environment

class DiffusionNavigatorGUI:
    """Main GUI application for Diffusion Navigator"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Diffusion Navigator - Interactive Learning Platform")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.diffusion_model = CaelumDiffusionModel()
        self.visualizer = DiffusionVisualizer()
        self.root_analyzer = ROOTDiffusionAnalyzer()
        
        # Setup ROOT if available
        self.root_enabled = setup_root_environment()
        
        # Data storage
        self.current_data = {}
        self.calculation_history = []
        self.materials_selected = []
        
        # Setup GUI
        self.setup_styles()
        self.create_main_layout()
        self.create_menu_bar()
        
        # Status queue for threading
        self.status_queue = queue.Queue()
        
    def setup_styles(self):
        """Setup modern styling for the GUI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom colors
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'background': '#F5F5F5',
            'text': '#333333'
        }
        
        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 11))
        style.configure('Primary.TButton', background=self.colors['primary'])
        style.configure('Secondary.TButton', background=self.colors['secondary'])
        
    def create_main_layout(self):
        """Create main layout with notebook tabs"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        self.create_control_panel(main_frame)
        
        # Right panel - Notebooks for different workshops
        self.create_workshop_notebook(main_frame)
        
        # Bottom status bar
        self.create_status_bar(main_frame)
        
    def create_control_panel(self, parent):
        """Create left control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Material Selection
        ttk.Label(control_frame, text="Material Selection", style='Title.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Available materials listbox
        ttk.Label(control_frame, text="Available Materials:").grid(row=1, column=0, sticky=tk.W)
        
        self.materials_listbox = tk.Listbox(control_frame, height=8, selectmode=tk.MULTIPLE)
        materials = self.diffusion_model.list_all_materials()
        for material in materials:
            self.materials_listbox.insert(tk.END, material)
        self.materials_listbox.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Select/Deselect buttons
        ttk.Button(control_frame, text="Select All", command=self.select_all_materials).grid(row=3, column=0, padx=(0, 5))
        ttk.Button(control_frame, text="Clear Selection", command=self.clear_material_selection).grid(row=3, column=1, padx=(5, 0))
        
        # Temperature Control
        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(control_frame, text="Temperature Settings", style='Title.TLabel').grid(row=5, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(control_frame, text="Min Temperature (K):").grid(row=6, column=0, sticky=tk.W)
        self.min_temp_var = tk.StringVar(value="300")
        ttk.Entry(control_frame, textvariable=self.min_temp_var, width=15).grid(row=6, column=1, sticky=tk.W)
        
        ttk.Label(control_frame, text="Max Temperature (K):").grid(row=7, column=0, sticky=tk.W)
        self.max_temp_var = tk.StringVar(value="1200")
        ttk.Entry(control_frame, textvariable=self.max_temp_var, width=15).grid(row=7, column=1, sticky=tk.W)
        
        ttk.Label(control_frame, text="Number of Points:").grid(row=8, column=0, sticky=tk.W)
        self.temp_points_var = tk.StringVar(value="20")
        ttk.Entry(control_frame, textvariable=self.temp_points_var, width=15).grid(row=8, column=1, sticky=tk.W, pady=(0, 10))
        
        # Diffusion Parameters
        ttk.Separator(control_frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(control_frame, text="Diffusion Parameters", style='Title.TLabel').grid(row=10, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(control_frame, text="Time Steps:").grid(row=11, column=0, sticky=tk.W)
        self.time_steps_var = tk.StringVar(value="100")
        ttk.Entry(control_frame, textvariable=self.time_steps_var, width=15).grid(row=11, column=1, sticky=tk.W)
        
        ttk.Label(control_frame, text="Spatial Resolution:").grid(row=12, column=0, sticky=tk.W)
        self.spatial_res_var = tk.StringVar(value="50")
        ttk.Entry(control_frame, textvariable=self.spatial_res_var, width=15).grid(row=12, column=1, sticky=tk.W, pady=(0, 10))
        
        # Action Buttons
        ttk.Separator(control_frame, orient='horizontal').grid(row=13, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(control_frame, text="Run Analysis", command=self.run_analysis, 
                  style='Primary.TButton').grid(row=14, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="Export Data", command=self.export_data_dialog).grid(row=15, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="Clear All", command=self.clear_all).grid(row=16, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Progress indicator
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.progress_var).grid(row=17, column=0, columnspan=2, pady=(10, 0))
        
        self.progress_bar = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress_bar.grid(row=18, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def create_workshop_notebook(self, parent):
        """Create workshop tabs"""
        workshop_frame = ttk.Frame(parent)
        workshop_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.notebook = ttk.Notebook(workshop_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Workshop 1: Diffusion Sphere
        self.create_diffusion_sphere_tab()
        
        # Workshop 2: Pattern Analysis
        self.create_pattern_analysis_tab()
        
        # Workshop 3: Layer Explorer
        self.create_layer_explorer_tab()
        
        # Workshop 4: ROOT Analysis
        self.create_root_analysis_tab()
        
        # Workshop 5: Step-by-Step Calculator
        self.create_calculator_tab()
        
    def create_diffusion_sphere_tab(self):
        """Create diffusion sphere visualization tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Diffusion Sphere")
        
        # Create matplotlib figure
        self.sphere_fig = Figure(figsize=(10, 8))
        self.sphere_canvas = FigureCanvasTkAgg(self.sphere_fig, tab_frame)
        self.sphere_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        sphere_toolbar = NavigationToolbar2Tk(self.sphere_canvas, tab_frame)
        sphere_toolbar.update()
        
        # Control frame
        control_frame = ttk.Frame(tab_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Generate Sphere", command=self.generate_diffusion_sphere).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Interactive View", command=self.open_interactive_sphere).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Plot", command=lambda: self.export_plot(self.sphere_fig, "sphere")).pack(side=tk.LEFT, padx=5)
        
    def create_pattern_analysis_tab(self):
        """Create pattern analysis tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Pattern Analysis")
        
        # Create matplotlib figure with subplots
        self.pattern_fig = Figure(figsize=(12, 8))
        self.pattern_canvas = FigureCanvasTkAgg(self.pattern_fig, tab_frame)
        self.pattern_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(tab_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Analyze Patterns", command=self.analyze_patterns).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Show Clusters", command=self.show_clusters).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Compare Materials", command=self.compare_materials).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Plot", command=lambda: self.export_plot(self.pattern_fig, "patterns")).pack(side=tk.LEFT, padx=5)
        
    def create_layer_explorer_tab(self):
        """Create layer-by-layer exploration tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Layer Explorer")
        
        # Create matplotlib figure
        self.layer_fig = Figure(figsize=(12, 8))
        self.layer_canvas = FigureCanvasTkAgg(self.layer_fig, tab_frame)
        self.layer_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Layer information frame
        info_frame = ttk.LabelFrame(tab_frame, text="Layer Information", padding="10")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.layer_info_text = tk.Text(info_frame, height=6, wrap=tk.WORD)
        layer_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.layer_info_text.yview)
        self.layer_info_text.configure(yscrollcommand=layer_scrollbar.set)
        
        self.layer_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        layer_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control frame
        control_frame = ttk.Frame(tab_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Generate Layers", command=self.generate_layers).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Explore Layer", command=self.explore_layer).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Plot", command=lambda: self.export_plot(self.layer_fig, "layers")).pack(side=tk.LEFT, padx=5)
        
    def create_root_analysis_tab(self):
        """Create ROOT analysis tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="ROOT Analysis")
        
        # Status indicator for ROOT
        root_status = "Available" if self.root_enabled else "Not Available (Using Mock)"
        status_color = "green" if self.root_enabled else "orange"
        
        status_frame = ttk.Frame(tab_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text=f"ROOT Status: ").pack(side=tk.LEFT)
        ttk.Label(status_frame, text=root_status, foreground=status_color).pack(side=tk.LEFT)
        
        # Create matplotlib figure
        self.root_fig = Figure(figsize=(10, 8))
        self.root_canvas = FigureCanvasTkAgg(self.root_fig, tab_frame)
        self.root_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(tab_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Arrhenius Analysis", command=self.arrhenius_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Error Analysis", command=self.error_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Monte Carlo", command=self.monte_carlo_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save ROOT File", command=self.save_root_file).pack(side=tk.LEFT, padx=5)
        
    def create_calculator_tab(self):
        """Create step-by-step calculator tab"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="Calculator")
        
        # Create two panes
        paned = ttk.PanedWindow(tab_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left pane - Calculator controls
        calc_frame = ttk.LabelFrame(paned, text="Calculator", padding="10")
        paned.add(calc_frame, weight=1)
        
        # Material selection for calculator
        ttk.Label(calc_frame, text="Select Material:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.calc_material_var = tk.StringVar()
        calc_material_combo = ttk.Combobox(calc_frame, textvariable=self.calc_material_var, 
                                          values=self.diffusion_model.list_all_materials())
        calc_material_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Temperature input
        ttk.Label(calc_frame, text="Temperature (K):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.calc_temp_var = tk.StringVar(value="600")
        ttk.Entry(calc_frame, textvariable=self.calc_temp_var).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Calculate button
        ttk.Button(calc_frame, text="Calculate Step-by-Step", command=self.calculate_step_by_step).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Right pane - Steps visualization
        steps_frame = ttk.LabelFrame(paned, text="Calculation Steps", padding="10")
        paned.add(steps_frame, weight=2)
        
        self.calc_steps_fig = Figure(figsize=(8, 6))
        self.calc_steps_canvas = FigureCanvasTkAgg(self.calc_steps_fig, steps_frame)
        self.calc_steps_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        calc_frame.columnconfigure(1, weight=1)
        
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Data", command=self.export_data_dialog)
        file_menu.add_command(label="Load Data", command=self.load_data_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Clear All Plots", command=self.clear_all_plots)
        view_menu.add_command(label="Refresh Data", command=self.refresh_data)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        # Add separator
        ttk.Separator(status_frame, orient='vertical').pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # Add material count
        self.material_count_var = tk.StringVar(value="Materials: 0")
        ttk.Label(status_frame, textvariable=self.material_count_var).pack(side=tk.LEFT)
        
    def select_all_materials(self):
        """Select all materials in the listbox"""
        self.materials_listbox.selection_set(0, tk.END)
        self.update_material_count()
        
    def clear_material_selection(self):
        """Clear material selection"""
        self.materials_listbox.selection_clear(0, tk.END)
        self.materials_selected = []
        self.update_material_count()
        
    def update_material_count(self):
        """Update material count display"""
        selected_indices = self.materials_listbox.curselection()
        self.materials_selected = [self.materials_listbox.get(i) for i in selected_indices]
        self.material_count_var.set(f"Materials: {len(self.materials_selected)}")
        
    def run_analysis(self):
        """Run complete diffusion analysis"""
        try:
            self.progress_var.set("Running analysis...")
            self.progress_bar.start(10)
            self.status_var.set("Analyzing diffusion properties...")
            
            # Get selected materials
            if not self.materials_selected:
                messagebox.showwarning("No Selection", "Please select at least one material.")
                return
            
            # Get parameters
            min_temp = float(self.min_temp_var.get())
            max_temp = float(self.max_temp_var.get())
            n_points = int(self.temp_points_var.get())
            temperatures = np.linspace(min_temp, max_temp, n_points)
            
            # Calculate diffusion properties
            self.current_data = self.diffusion_model.calculate_diffusion_sphere_properties(
                self.materials_selected, temperatures
            )
            
            # Analyze clusters
            self.cluster_data = self.diffusion_model.analyze_diffusion_clusters(self.current_data)
            
            # Update visualizations
            self.update_all_visualizations()
            
            self.progress_var.set("Analysis complete!")
            self.progress_bar.stop()
            self.status_var.set(f"Analyzed {len(self.materials_selected)} materials")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.progress_var.set("Error")
            self.progress_bar.stop()
            
    def update_all_visualizations(self):
        """Update all visualization tabs"""
        # Update each tab
        self.generate_diffusion_sphere()
        self.analyze_patterns()
        self.generate_layers()
        
    def generate_diffusion_sphere(self):
        """Generate diffusion sphere visualization"""
        if not self.current_data:
            messagebox.showinfo("No Data", "Please run analysis first.")
            return
            
        self.sphere_fig.clear()
        
        # Generate layer assignments
        from caelum_diffusion_model import RelationalSphereModel
        sphere_model = RelationalSphereModel()
        layer_assignments = sphere_model.assign_diffusion_to_layers(self.current_data)
        
        # Create visualization
        fig = self.visualizer.create_diffusion_sphere(self.current_data, layer_assignments)
        
        # Convert to static plot for tkinter
        # Create a representative 2D projection
        ax = self.sphere_fig.add_subplot(111, projection='3d')
        
        # Generate sample sphere data
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        
        for layer_idx in range(min(5, len(layer_assignments))):
            radius = 1.0 + layer_idx * 0.15
            x = radius * np.outer(np.cos(u), np.sin(v))
            y = radius * np.outer(np.sin(u), np.sin(v))
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Color based on layer
            color_intensity = 0.3 + 0.15 * layer_idx
            ax.plot_surface(x, y, z, alpha=0.3, color=plt.cm.viridis(color_intensity))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Diffusion Sphere Visualization')
        
        self.sphere_canvas.draw()
        
    def analyze_patterns(self):
        """Analyze diffusion patterns"""
        if not self.current_data:
            messagebox.showinfo("No Data", "Please run analysis first.")
            return
            
        self.pattern_fig.clear()
        
        # Create cluster visualization
        fig = self.visualizer.create_cluster_visualization(self.cluster_data, self.current_data)
        
        # Copy to our figure
        for i in range(len(fig.axes)):
            self.pattern_fig.add_subplot(len(fig.axes), 1, i+1)
            # Copy the content
            self.pattern_fig.axes[i]. Artists = fig.axes[i].artists
        
        self.pattern_canvas.draw()
        
    def generate_layers(self):
        """Generate layer-by-layer analysis"""
        if not self.current_data:
            messagebox.showinfo("No Data", "Please run analysis first.")
            return
            
        self.layer_fig.clear()
        
        # Generate layer assignments
        from caelum_diffusion_model import RelationalSphereModel
        sphere_model = RelationalSphereModel()
        layer_assignments = sphere_model.assign_diffusion_to_layers(self.current_data)
        
        # Create layer visualization
        fig = self.visualizer.create_layer_by_layer_analysis(layer_assignments)
        
        # Copy to our figure
        # Simplified version for tkinter compatibility
        n_layers = min(4, len(layer_assignments))
        for i, (layer_num, layer_data) in enumerate(list(layer_assignments.items())[:n_layers]):
            ax = self.layer_fig.add_subplot(2, 2, i+1)
            
            if layer_data:
                materials = list(layer_data.keys())[:5]  # Limit to 5 materials
                values = [layer_data[mat] for mat in materials]
                colors = plt.cm.Set3(np.linspace(0, 1, len(materials)))
                
                bars = ax.bar(materials, values, color=colors)
                ax.set_title(f'Layer {layer_num}')
                ax.set_ylabel('Diffusion Value')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Update layer information
        self.update_layer_info(layer_assignments)
        
        self.layer_fig.suptitle('Layer-by-Layer Diffusion Analysis')
        self.layer_canvas.draw()
        
    def update_layer_info(self, layer_assignments):
        """Update layer information text"""
        self.layer_info_text.delete(1.0, tk.END)
        
        info_text = "Layer Analysis Summary:\n"
        info_text += f"Total layers: {len(layer_assignments)}\n\n"
        
        for layer_num, layer_data in list(layer_assignments.items())[:3]:  # Show first 3 layers
            info_text += f"Layer {layer_num}: {len(layer_data)} materials\n"
            if layer_data:
                avg_diffusion = np.mean(list(layer_data.values()))
                max_material = max(layer_data.items(), key=lambda x: x[1])
                info_text += f"  Average diffusion: {avg_diffusion:.4f}\n"
                info_text += f"  Highest: {max_material[0]} ({max_material[1]:.4f})\n"
            info_text += "\n"
        
        self.layer_info_text.insert(1.0, info_text)
        
    def calculate_step_by_step(self):
        """Perform step-by-step diffusion calculation"""
        try:
            material_name = self.calc_material_var.get()
            temperature = float(self.calc_temp_var.get())
            
            if not material_name:
                messagebox.showwarning("No Material", "Please select a material.")
                return
                
            # Get material
            material = self.diffusion_model.materials_db.get(material_name)
            if not material:
                messagebox.showerror("Material Not Found", f"Material {material_name} not found.")
                return
            
            # Calculate diffusion coefficient with steps
            D, steps = self.diffusion_model.calculate_diffusion_coefficient(material, temperature)
            
            # Visualize steps
            self.calc_steps_fig.clear()
            
            n_steps = len(steps)
            for i, step in enumerate(steps):
                ax = self.calc_steps_fig.add_subplot(n_steps, 1, i+1)
                
                # Create visual representation
                step_text = f"Step {step['step']}: {step['description']}"
                calc_text = step['calculation']
                
                ax.text(0.05, 0.7, step_text, fontsize=10, fontweight='bold', 
                       transform=ax.transAxes, wrap=True)
                ax.text(0.05, 0.3, calc_text, fontsize=9, 
                       transform=ax.transAxes, wrap=True, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
                
                # Add result highlight
                result_text = f"Result: {step['result']:.2e}" if isinstance(step['result'], (int, float)) else f"Result: {step['result']}"
                ax.text(0.05, 0.05, result_text, fontsize=8, color='darkgreen',
                       transform=ax.transAxes, fontweight='bold')
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            self.calc_steps_fig.suptitle(f'Step-by-Step: {material_name} at {temperature}K', fontsize=12, fontweight='bold')
            self.calc_steps_fig.tight_layout()
            self.calc_steps_canvas.draw()
            
            # Show final result
            messagebox.showinfo("Calculation Complete", 
                              f"Diffusion Coefficient: {D:.2e} m²/s")
            
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Error in calculation: {str(e)}")
            
    def export_data_dialog(self):
        """Export data dialog"""
        if not self.current_data:
            messagebox.showinfo("No Data", "No data to export. Please run analysis first.")
            return
            
        # Ask for file format and location
        file_types = [
            ("JSON files", "*.json"),
            ("Text files", "*.txt"),
            ("CSV files", "*.csv")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".json",
            filetypes=file_types
        )
        
        if filename:
            try:
                # Prepare export data
                export_data = {
                    'materials_analyzed': self.materials_selected,
                    'temperature_range': [self.min_temp_var.get(), self.max_temp_var.get()],
                    'diffusion_data': self.current_data,
                    'cluster_analysis': self.cluster_data,
                    'analysis_timestamp': str(np.datetime64('now'))
                }
                
                # Export based on file extension
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                elif filename.endswith('.txt'):
                    with open(filename, 'w') as f:
                        f.write("Diffusion Navigator Data Export\n")
                        f.write("=" * 40 + "\n\n")
                        for key, value in export_data.items():
                            f.write(f"{key}:\n")
                            f.write(f"{str(value)}\n\n")
                elif filename.endswith('.csv'):
                    # Simplified CSV export
                    import pandas as pd
                    df = pd.DataFrame(self.current_data).T
                    df.to_csv(filename)
                
                messagebox.showinfo("Export Complete", f"Data exported to {filename}")
                self.status_var.set(f"Exported to {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
                
    def clear_all(self):
        """Clear all data and plots"""
        self.current_data = {}
        self.cluster_data = {}
        self.materials_selected = []
        self.clear_material_selection()
        self.clear_all_plots()
        self.status_var.set("All data cleared")
        
    def clear_all_plots(self):
        """Clear all plot canvases"""
        self.sphere_fig.clear()
        self.pattern_fig.clear()
        self.layer_fig.clear()
        self.root_fig.clear()
        self.calc_steps_fig.clear()
        
        self.sphere_canvas.draw()
        self.pattern_canvas.draw()
        self.layer_canvas.draw()
        self.root_canvas.draw()
        self.calc_steps_canvas.draw()
        
    def export_plot(self, figure, name):
        """Export specific plot"""
        file_types = [
            ("PNG files", "*.png"),
            ("PDF files", "*.pdf"),
            ("SVG files", "*.svg")
        ]
        
        filename = filedialog.asksaveasfilename(
            title=f"Export {name} plot",
            defaultextension=".png",
            filetypes=file_types
        )
        
        if filename:
            try:
                figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Export Complete", f"Plot exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
                
    def open_interactive_sphere(self):
        """Open interactive sphere in browser"""
        if not self.current_data:
            messagebox.showinfo("No Data", "Please run analysis first.")
            return
            
        # Generate layer assignments
        from caelum_diffusion_model import RelationalSphereModel
        sphere_model = RelationalSphereModel()
        layer_assignments = sphere_model.assign_diffusion_to_layers(self.current_data)
        
        # Create interactive plot
        fig = self.visualizer.create_diffusion_sphere(self.current_data, layer_assignments)
        
        # Save to temporary HTML file and open in browser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            fig.write_html(f.name)
            webbrowser.open(f'file://{f.name}')
            
    def show_about(self):
        """Show about dialog"""
        about_text = """Diffusion Navigator v1.0
        
An interactive learning platform for exploring diffusion phenomena
using the Caelum-based modeling framework.

Features:
• Multi-material diffusion analysis
• Interactive visualizations
• ROOT integration for advanced analysis
• Step-by-step calculations
• Export capabilities

Developed for students and researchers in materials science
and related fields."""
        
        messagebox.showinfo("About Diffusion Navigator", about_text)
        
    def show_user_guide(self):
        """Show user guide"""
        guide_text = """Diffusion Navigator - User Guide

Getting Started:
1. Select materials from the control panel
2. Set temperature range and parameters
3. Click 'Run Analysis' to begin

Workshops:
• Diffusion Sphere: 3D visualization of diffusion layers
• Pattern Analysis: Clustering and material comparison
• Layer Explorer: Detailed layer-by-layer analysis
• ROOT Analysis: Advanced physics calculations
• Calculator: Step-by-step diffusion calculations

Tips:
• Select multiple materials for comparison
• Use interactive views for detailed exploration
• Export results for further analysis
• Check step-by-step calculations for learning"""
        
        messagebox.showinfo("User Guide", guide_text)
        
    def load_data_dialog(self):
        """Load data from file"""
        filename = filedialog.askopenfilename(
            title="Load Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                self.current_data = data.get('diffusion_data', {})
                self.cluster_data = data.get('cluster_analysis', {})
                self.materials_selected = data.get('materials_analyzed', [])
                
                # Update UI
                self.update_material_selection()
                self.update_all_visualizations()
                
                messagebox.showinfo("Load Complete", f"Data loaded from {filename}")
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load: {str(e)}")
                
    def update_material_selection(self):
        """Update material selection in listbox"""
        self.materials_listbox.selection_clear(0, tk.END)
        for i, material in enumerate(self.materials_selected):
            try:
                idx = self.materials_listbox.get(0, tk.END).index(material)
                self.materials_listbox.selection_set(idx)
            except ValueError:
                pass
        
        self.update_material_count()
        
    def refresh_data(self):
        """Refresh current data and visualizations"""
        if self.current_data:
            self.update_all_visualizations()
            self.status_var.set("Data refreshed")
        else:
            messagebox.showinfo("No Data", "Please run analysis first.")
            
    # ROOT Analysis methods
    def arrhenius_analysis(self):
        """Perform Arrhenius analysis using ROOT"""
        if not self.current_data:
            messagebox.showinfo("No Data", "Please run analysis first.")
            return
            
        self.root_fig.clear()
        
        # Perform Arrhenius analysis for first selected material
        if self.materials_selected:
            material = self.materials_selected[0]
            if material in self.current_data:
                data = self.current_data[material]
                temperatures = np.array(data['temperatures'])
                coeffs = np.array(data['diffusion_coefficients'])
                
                # Use ROOT analyzer
                graph, analysis = self.root_analyzer.create_arrhenius_plot(temperatures, coeffs, material)
                
                # Create visualization
                ax = self.root_fig.add_subplot(111)
                
                # Plot Arrhenius plot
                inv_T = 1.0 / temperatures
                ln_D = np.log(coeffs)
                
                ax.plot(inv_T, ln_D, 'bo-', label=f'{material} data')
                
                # Add fit line
                if 'activation_energy' in analysis:
                    R = 8.314
                    Qa = analysis['activation_energy']
                    D0 = analysis['pre_exponential']
                    
                    fit_line = np.log(D0) - (Qa * 1000 / R) * inv_T
                    ax.plot(inv_T, fit_line, 'r--', label='Fit')
                    
                    # Add annotation
                    ax.annotate(f'Ea = {Qa:.2f} kJ/mol\nD₀ = {D0:.2e} m²/s',
                               xy=(0.05, 0.95), xycoords='axes fraction',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
                
                ax.set_xlabel('1/T (K⁻¹)')
                ax.set_ylabel('ln(D) (m²/s)')
                ax.set_title(f'Arrhenius Analysis - {material}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        self.root_canvas.draw()
        
    def error_analysis(self):
        """Perform error analysis"""
        if not self.current_data:
            messagebox.showinfo("No Data", "Please run analysis first.")
            return
            
        messagebox.showinfo("Error Analysis", "Error analysis feature coming soon!")
        
    def monte_carlo_analysis(self):
        """Perform Monte Carlo uncertainty analysis"""
        if not self.current_data:
            messagebox.showinfo("No Data", "Please run analysis first.")
            return
            
        messagebox.showinfo("Monte Carlo", "Monte Carlo analysis feature coming soon!")
        
    def save_root_file(self):
        """Save ROOT file"""
        if self.root_enabled:
            self.root_analyzer.save_root_file()
            messagebox.showinfo("ROOT File", "ROOT file saved successfully!")
        else:
            messagebox.showinfo("ROOT Not Available", "ROOT framework is not available in this installation.")
            
    def explore_layer(self):
        """Explore specific layer in detail"""
        messagebox.showinfo("Layer Explorer", "Detailed layer exploration coming soon!")
        
    def show_clusters(self):
        """Show detailed cluster information"""
        if not self.cluster_data:
            messagebox.showinfo("No Clusters", "Please run analysis first.")
            return
            
        # Create cluster information dialog
        cluster_window = tk.Toplevel(self.root)
        cluster_window.title("Cluster Analysis Details")
        cluster_window.geometry("600x400")
        
        # Create text widget
        text_widget = tk.Text(cluster_window, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(cluster_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Add cluster information
        text_widget.insert(tk.END, "Cluster Analysis Results\n")
        text_widget.insert(tk.END, "=" * 30 + "\n\n")
        
        text_widget.insert(tk.END, f"Total clusters found: {self.cluster_data['n_clusters']}\n")
        text_widget.insert(tk.END, f"Analysis summary: {self.cluster_data['analysis']}\n\n")
        
        for cluster_id, cluster_info in self.cluster_data['clusters'].items():
            text_widget.insert(tk.END, f"Cluster {cluster_id}:\n")
            text_widget.insert(tk.END, f"  Materials: {', '.join(cluster_info['materials'])}\n")
            text_widget.insert(tk.END, f"  Characteristics: {', '.join(cluster_info['characteristics'])}\n\n")
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def compare_materials(self):
        """Compare selected materials in detail"""
        if len(self.materials_selected) < 2:
            messagebox.showinfo("Selection Required", "Please select at least 2 materials for comparison.")
            return
            
        # Create comparison window
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Material Comparison")
        compare_window.geometry("800x600")
        
        # Create comparison plot
        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvasTkAgg(fig, compare_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create material comparison radar chart
        materials_data = {}
        for material in self.materials_selected:
            if material in self.diffusion_model.materials_db:
                materials_data[material] = {
                    'density': self.diffusion_model.materials_db[material].density,
                    'diffusion_coefficient': self.diffusion_model.materials_db[material].diffusion_coefficient,
                    'activation_energy': self.diffusion_model.materials_db[material].activation_energy,
                    'atomic_radius': self.diffusion_model.materials_db[material].atomic_radius
                }
        
        radar_fig = self.visualizer.create_material_comparison_radar(materials_data)
        
        # Copy to our figure (simplified)
        ax = fig.add_subplot(111, projection='polar')
        
        # Basic properties for radar chart
        properties = ['Density', 'D_coefficient', 'Activation_E', 'Atomic_Radius']
        angles = np.linspace(0, 2 * np.pi, len(properties), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, material in enumerate(self.materials_selected[:5]):  # Limit to 5 materials
            if material in materials_data:
                data = materials_data[material]
                values = [
                    data['density'] / 20,  # Normalized
                    np.log10(data['diffusion_coefficient'] + 1e-10),
                    data['activation_energy'] / 6,
                    data['atomic_radius'] * 10
                ]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', label=material, linewidth=2)
                ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(properties)
        ax.legend()
        ax.set_title("Material Properties Comparison")
        
        canvas.draw()
        
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

# Main execution
if __name__ == "__main__":
    app = DiffusionNavigatorGUI()
    app.run()
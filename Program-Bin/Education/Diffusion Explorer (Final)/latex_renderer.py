"""
Advanced LaTeX Formula Rendering System
High-quality mathematical formula display for Diffusion Navigator
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredText
import numpy as np
from typing import List, Dict, Optional, Tuple
import re
import tempfile
import os

class LaTeXRenderer:
    """Advanced LaTeX rendering system for diffusion formulas"""
    
    def __init__(self):
        self.formula_cache = {}
        self.color_schemes = {
            'default': {'text': '#333333', 'background': '#FFFFFF', 'highlight': '#FFE6B3'},
            'dark': {'text': '#FFFFFF', 'background': '#2B2B2B', 'highlight': '#4A90E2'},
            'student': {'text': '#1A237E', 'background': '#F3E5F5', 'highlight': '#FFD54F'}
        }
        
    def render_diffusion_equation(self, equation_type: str = 'fick_second', 
                                style: str = 'default') -> plt.Figure:
        """
        Render diffusion equations with step-by-step explanation
        """
        colors = self.color_schemes.get(style, self.color_schemes['default'])
        
        if equation_type == 'fick_second':
            return self._render_fick_second_law(colors)
        elif equation_type == 'arrhenius':
            return self._render_arrhenius_equation(colors)
        elif equation_type == 'diffusion_coefficient':
            return self._render_diffusion_coefficient(colors)
        elif equation_type == 'concentration_profile':
            return self._render_concentration_profile(colors)
        else:
            return self._render_general_diffusion(equation_type, colors)
    
    def _render_fick_second_law(self, colors: Dict) -> plt.Figure:
        """Render Fick's second law with detailed explanation"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(colors['background'])
        
        # Title
        fig.suptitle('Fick\'s Second Law of Diffusion', fontsize=16, fontweight='bold', 
                    color=colors['text'])
        
        # Equation 1: Basic form
        ax1 = axes[0, 0]
        ax1.text(0.5, 0.7, r'$\frac{\partial C}{\partial t} = D \nabla^2 C$', 
                fontsize=14, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['highlight'], alpha=0.8))
        ax1.text(0.5, 0.3, 'Basic Form\n$C$ = concentration\n$t$ = time\n$D$ = diffusion coefficient\n$\nabla^2$ = Laplacian operator', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Basic Form', fontsize=12, color=colors['text'])
        
        # Equation 2: 1D form
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.7, r'$\frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2}$', 
                fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['highlight'], alpha=0.8))
        ax2.text(0.5, 0.3, '1D Form\n$x$ = spatial coordinate\nApplicable for:\n• Thin films\n• Linear diffusion\n• Planar geometries', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('1D Form', fontsize=12, color=colors['text'])
        
        # Equation 3: Cylindrical coordinates
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.7, r'$\frac{\partial C}{\partial t} = D\left(\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial C}{\partial r}\right)\right)$', 
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['highlight'], alpha=0.8))
        ax3.text(0.5, 0.2, 'Cylindrical Form\n$r$ = radial coordinate\nApplicable for:\n• Cylinders\n• Pipes\n• Wires', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Cylindrical Form', fontsize=12, color=colors['text'])
        
        # Equation 4: Spherical coordinates
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.6, r'$\frac{\partial C}{\partial t} = D\left(\frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial C}{\partial r}\right)\right)$', 
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['highlight'], alpha=0.8))
        ax4.text(0.5, 0.2, 'Spherical Form\nApplicable for:\n• Spheres\n• Particles\n• Point sources', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Spherical Form', fontsize=12, color=colors['text'])
        
        plt.tight_layout()
        return fig
    
    def _render_arrhenius_equation(self, colors: Dict) -> plt.Figure:
        """Render Arrhenius equation with parameters explanation"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(colors['background'])
        
        fig.suptitle('Arrhenius Equation for Diffusion', fontsize=16, fontweight='bold', 
                    color=colors['text'])
        
        # Main equation
        ax1 = axes[0, 0]
        ax1.text(0.5, 0.7, r'$D = D_0 \exp\left(-\frac{Q_a}{RT}\right)$', 
                fontsize=16, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['highlight'], alpha=0.8))
        ax1.text(0.5, 0.2, 'Temperature Dependence\n$D_0$ = pre-exponential factor\n$Q_a$ = activation energy\n$R$ = gas constant (8.314 J/mol·K)\n$T$ = absolute temperature (K)', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Complete Equation', fontsize=12, color=colors['text'])
        
        # Parameter ranges
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.8, 'Typical Parameter Ranges', fontsize=12, fontweight='bold', 
                ha='center', color=colors['text'])
        
        param_text = '''
Metals: $Q_a$ = 50-200 kJ/mol
Ceramics: $Q_a$ = 200-500 kJ/mol  
Polymers: $Q_a$ = 20-100 kJ/mol

$D_0$ Range: $10^{-8}$ to $10^{-4}$ m²/s

Temperature: 300-2000 K for materials
'''
        ax2.text(0.1, 0.5, param_text, fontsize=9, va='center', 
                family='monospace', color=colors['text'])
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Parameter Values', fontsize=12, color=colors['text'])
        
        # Linearized form
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.7, r'$\ln D = \ln D_0 - \frac{Q_a}{R}\frac{1}{T}$', 
                fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['highlight'], alpha=0.8))
        ax3.text(0.5, 0.2, 'Linearized Form\nPlot: $\ln D$ vs $1/T$\nSlope = $-Q_a/R$\nIntercept = $\ln D_0$\nEnables experimental determination', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Arrhenius Plot', fontsize=12, color=colors['text'])
        
        # Physical meaning
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.8, 'Physical Interpretation', fontsize=12, fontweight='bold', 
                ha='center', color=colors['text'])
        
        meaning_text = '''
• Higher $Q_a$ = stronger T dependence
• Larger atoms → higher $Q_a$
• Crystal structure affects $Q_a$
• Vacancy mechanism dominates
• Grain boundary diffusion: lower $Q_a$
• Surface diffusion: lowest $Q_a$
'''
        ax4.text(0.1, 0.4, meaning_text, fontsize=9, va='center', color=colors['text'])
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Physical Meaning', fontsize=12, color=colors['text'])
        
        plt.tight_layout()
        return fig
    
    def _render_diffusion_coefficient(self, colors: Dict) -> plt.Figure:
        """Render diffusion coefficient calculation steps"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.patch.set_facecolor(colors['background'])
        
        fig.suptitle('Diffusion Coefficient Calculation', fontsize=16, fontweight='bold', 
                    color=colors['text'])
        
        # Step 1: Material properties
        ax1 = axes[0, 0]
        ax1.text(0.5, 0.7, 'Step 1: Material Properties', fontsize=12, fontweight='bold', 
                ha='center', color=colors['text'])
        ax1.text(0.5, 0.3, r'$D_0$ and $Q_a$ from experiments\nor literature databases\n\nExamples:\nFe: $D_0 = 2.4 \times 10^{-5}$ m²/s\nFe: $Q_a = 248$ kJ/mol', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Step 2: Temperature conversion
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.7, 'Step 2: Temperature Setup', fontsize=12, fontweight='bold', 
                ha='center', color=colors['text'])
        ax2.text(0.5, 0.3, r'$T$ = 800 K (example)\n\nMust use absolute temperature\nNo Celsius or Fahrenheit!\n\n$T = T_{°C} + 273.15$', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # Step 3: Energy conversion
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.7, 'Step 3: Energy Conversion', fontsize=12, fontweight='bold', 
                ha='center', color=colors['text'])
        ax3.text(0.5, 0.3, r'$Q_a = 248$ kJ/mol\n$Q_a = 248,000$ J/mol\n\nConvert to J/mol\nfor equation consistency', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # Step 4: Exponential term
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.6, 'Step 4: Exponential Term', fontsize=12, fontweight='bold', 
                ha='center', color=colors['text'])
        ax4.text(0.5, 0.2, r'$\frac{Q_a}{RT} = \frac{248,000}{8.314 \times 800} = 37.3$\n\n$\exp(-37.3) = 5.8 \times 10^{-17}$\n\nVery small!', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # Step 5: Final calculation
        ax5 = axes[2, 0]
        ax5.text(0.5, 0.7, 'Step 5: Final Calculation', fontsize=12, fontweight='bold', 
                ha='center', color=colors['text'])
        ax5.text(0.5, 0.3, r'$D = 2.4 \times 10^{-5} \times 5.8 \times 10^{-17}$\n\n$D = 1.4 \times 10^{-21}$ m²/s\n\nVery small at 800 K!', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        # Summary
        ax6 = axes[2, 1]
        ax6.text(0.5, 0.8, 'Result Summary', fontsize=12, fontweight='bold', 
                ha='center', color=colors['text'])
        ax6.text(0.5, 0.4, r'Fe at 800 K:\n$D = 1.4 \times 10^{-21}$ m²/s\n\nCompare:\n$D_{Fe,1200K} = 1.4 \times 10^{-13}$ m²/s\n\n$10^8$ times increase!', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        plt.tight_layout()
        return fig
    
    def _render_concentration_profile(self, colors: Dict) -> plt.Figure:
        """Render concentration profile solutions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(colors['background'])
        
        fig.suptitle('Concentration Profile Solutions', fontsize=16, fontweight='bold', 
                    color=colors['text'])
        
        # Error function solution
        ax1 = axes[0, 0]
        ax1.text(0.5, 0.7, r'$C(x,t) = C_0 \text{erfc}\left(\frac{x}{2\sqrt{Dt}}\right)$', 
                fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['highlight'], alpha=0.8))
        ax1.text(0.5, 0.2, 'Semi-infinite Solid\n$C_0$ = surface concentration\nerfc = complementary error function\n$x$ = depth, $t$ = time', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Error Function Solution', fontsize=12, color=colors['text'])
        
        # Gaussian solution
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.7, r'$C(x,t) = \frac{M}{\sqrt{4\pi Dt}} \exp\left(-\frac{x^2}{4Dt}\right)$', 
                fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['highlight'], alpha=0.8))
        ax2.text(0.5, 0.2, 'Thin Film Solution\n$M$ = initial mass per area\nGaussian distribution\nSpreads with time', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Gaussian Solution', fontsize=12, color=colors['text'])
        
        # Fourier series
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.6, r'$C(x,t) = \sum_{n=0}^{\infty} A_n \sin\left(\frac{(2n+1)\pi x}{2L}\right) e^{-D\left(\frac{(2n+1)\pi}{2L}\right)^2 t}$', 
                fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['highlight'], alpha=0.8))
        ax3.text(0.5, 0.1, 'Finite Slab\n$L$ = half-thickness\nFourier series\nMultiple modes', 
                fontsize=10, ha='center', va='center', color=colors['text'])
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Fourier Series', fontsize=12, color=colors['text'])
        
        # Dimensionless parameters
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.8, 'Dimensionless Groups', fontsize=12, fontweight='bold', 
                ha='center', color=colors['text'])
        ax4.text(0.5, 0.5, r'Fourier Number: $Fo = \frac{Dt}{L^2}$\n\n$Fo < 0.1$: Initial stage\n$0.1 < Fo < 1$: Transition\n$Fo > 1$: Long-time behavior', 
                fontsize=11, ha='center', va='center', color=colors['text'])
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Similarity Parameters', fontsize=12, color=colors['text'])
        
        plt.tight_layout()
        return fig
    
    def create_formula_library(self) -> Dict[str, plt.Figure]:
        """Create a library of all diffusion formulas"""
        formulas = {
            'fick_second': self._render_fick_second_law(self.color_schemes['student']),
            'arrhenius': self._render_arrhenius_equation(self.color_schemes['student']),
            'diffusion_coefficient': self._render_diffusion_coefficient(self.color_schemes['student']),
            'concentration_profile': self._render_concentration_profile(self.color_schemes['student'])
        }
        return formulas
    
    def export_formula_to_image(self, formula_type: str, filename: str, 
                              style: str = 'student', dpi: int = 300):
        """Export formula to image file"""
        fig = self.render_diffusion_equation(formula_type, style)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor=self.color_schemes[style]['background'])
        plt.close(fig)
    
    def create_interactive_formula_guide(self) -> plt.Figure:
        """Create an interactive formula guide for students"""
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor(self.color_schemes['student']['background'])
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        title_ax = fig.add_subplot(gs[0, :])
        title_ax.text(0.5, 0.8, 'DIFFUSION FORMULA GUIDE', 
                     fontsize=20, fontweight='bold', ha='center', 
                     color=self.color_schemes['student']['text'])
        title_ax.text(0.5, 0.3, 'Click on any formula to see detailed explanation', 
                     fontsize=14, ha='center', 
                     color=self.color_schemes['student']['text'])
        title_ax.set_xlim(0, 1)
        title_ax.set_ylim(0, 1)
        title_ax.axis('off')
        
        # Fick's laws
        fick_ax = fig.add_subplot(gs[1, 0:2])
        fick_ax.text(0.5, 0.7, r'$\frac{\partial C}{\partial t} = D \nabla^2 C$', 
                    fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.5", 
                             facecolor=self.color_schemes['student']['highlight'], alpha=0.8))
        fick_ax.text(0.5, 0.2, 'Fick\'s Second Law\nMass conservation\nFundamental equation', 
                    fontsize=10, ha='center', va='center', 
                    color=self.color_schemes['student']['text'])
        fick_ax.set_xlim(0, 1)
        fick_ax.set_ylim(0, 1)
        fick_ax.axis('off')
        
        # Arrhenius
        arr_ax = fig.add_subplot(gs[1, 2:4])
        arr_ax.text(0.5, 0.7, r'$D = D_0 \exp\left(-\frac{Q_a}{RT}\right)$', 
                   fontsize=14, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", 
                            facecolor=self.color_schemes['student']['highlight'], alpha=0.8))
        arr_ax.text(0.5, 0.2, 'Arrhenius Equation\nTemperature dependence\nActivation barrier', 
                   fontsize=10, ha='center', va='center', 
                   color=self.color_schemes['student']['text'])
        arr_ax.set_xlim(0, 1)
        arr_ax.set_ylim(0, 1)
        arr_ax.axis('off')
        
        # Error function
        err_ax = fig.add_subplot(gs[2, 0:2])
        err_ax.text(0.5, 0.7, r'$C(x,t) = C_0 \text{erfc}\left(\frac{x}{2\sqrt{Dt}}\right)$', 
                   fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", 
                            facecolor=self.color_schemes['student']['highlight'], alpha=0.8))
        err_ax.text(0.5, 0.2, 'Error Function Solution\nSemi-infinite solid\nPractical applications', 
                   fontsize=10, ha='center', va='center', 
                   color=self.color_schemes['student']['text'])
        err_ax.set_xlim(0, 1)
        err_ax.set_ylim(0, 1)
        err_ax.axis('off')
        
        # Gaussian
        gauss_ax = fig.add_subplot(gs[2, 2:4])
        gauss_ax.text(0.5, 0.6, r'$C(x,t) = \frac{M}{\sqrt{4\pi Dt}} \exp\left(-\frac{x^2}{4Dt}\right)$', 
                     fontsize=11, ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.5", 
                              facecolor=self.color_schemes['student']['highlight'], alpha=0.8))
        gauss_ax.text(0.5, 0.1, 'Gaussian Solution\nThin film diffusion\nProbability distribution', 
                     fontsize=10, ha='center', va='center', 
                     color=self.color_schemes['student']['text'])
        gauss_ax.set_xlim(0, 1)
        gauss_ax.set_ylim(0, 1)
        gauss_ax.axis('off')
        
        # Usage tips
        tips_ax = fig.add_subplot(gs[3, :])
        tips_text = '''
USAGE TIPS:
• Start with Fick's laws for fundamental understanding
• Use Arrhenius for temperature effects
• Choose solution based on geometry (infinite, finite, thin film)
• Check dimensionless numbers (Fourier number)
• Verify units: D [m²/s], T [K], Qa [J/mol]
'''
        tips_ax.text(0.05, 0.5, tips_text, fontsize=10, va='center', 
                    family='monospace', color=self.color_schemes['student']['text'])
        tips_ax.set_xlim(0, 1)
        tips_ax.set_ylim(0, 1)
        tips_ax.axis('off')
        
        return fig
#!/usr/bin/env python3
"""
Visualization Suite: Comprehensive plotting and graphing for numerical analysis
Built for clear communication of mathematical patterns
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import math
from dataclasses import dataclass

# Configure matplotlib and seaborn
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class PlotConfig:
    """Configuration for plot styling"""
    title: str
    xlabel: str
    ylabel: str
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    style: str = 'default'
    save_path: Optional[str] = None

class MathematicalVisualizer:
    """Advanced visualization for mathematical concepts and data"""
    
    def __init__(self):
        self.figures = []
        
    def plot_sequence(self, sequence: List[int], config: PlotConfig, 
                     highlight_period: Optional[int] = None) -> plt.Figure:
        """Plot a sequence with optional period highlighting"""
        
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        
        # Basic sequence plot
        x = range(len(sequence))
        ax.plot(x, sequence, 'b-', linewidth=2, marker='o', markersize=4, 
                label='Sequence', alpha=0.8)
        
        # Highlight periods if specified
        if highlight_period and highlight_period > 1:
            for i in range(0, len(sequence), highlight_period):
                if i + highlight_period <= len(sequence):
                    ax.axvspan(i, i + highlight_period, alpha=0.1, color='red')
                    if i == 0:
                        ax.text(i + highlight_period/2, max(sequence)*0.9, 
                               f'Period = {highlight_period}', 
                               ha='center', fontsize=10, color='red')
        
        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_title(config.title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if config.save_path:
            fig.savefig(config.save_path, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_growth_analysis(self, sequences: Dict[str, List[int]], 
                           config: PlotConfig) -> plt.Figure:
        """Compare growth patterns of multiple sequences"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=config.dpi)
        
        # Linear plot
        for name, seq in sequences.items():
            x = range(len(seq))
            ax1.plot(x, seq, linewidth=2, marker='o', markersize=3, label=name, alpha=0.8)
        
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.set_title(f'{config.title} - Linear Scale')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Logarithmic plot
        for name, seq in sequences.items():
            # Remove non-positive values for log plot
            positive_seq = [max(1, abs(x)) for x in seq]
            x = range(len(positive_seq))
            ax2.semilogy(x, positive_seq, linewidth=2, marker='o', markersize=3, 
                        label=name, alpha=0.8)
        
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value (log scale)')
        ax2.set_title(f'{config.title} - Logarithmic Scale')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(config.title, fontsize=14, fontweight='bold')
        
        if config.save_path:
            fig.savefig(config.save_path, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_modular_patterns(self, sequence: List[int], modulus: int, 
                            config: PlotConfig) -> plt.Figure:
        """Visualize sequence patterns in modular arithmetic"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=config.dpi)
        
        # Residues over time
        residues = [x % modulus for x in sequence]
        x = range(len(residues))
        
        ax1.plot(x, residues, 'b-', linewidth=1, marker='o', markersize=3, alpha=0.7)
        ax1.set_xlabel('Index')
        ax1.set_ylabel(f'Residue mod {modulus}')
        ax1.set_title(f'{config.title} - Residues Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.5, modulus - 0.5)
        
        # Residue distribution
        residue_counts = {}
        for r in residues:
            residue_counts[r] = residue_counts.get(r, 0) + 1
        
        ax2.bar(residue_counts.keys(), residue_counts.values(), alpha=0.7)
        ax2.set_xlabel(f'Residue mod {modulus}')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{config.title} - Residue Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(modulus))
        
        # Add expected uniform distribution line
        expected_count = len(sequence) / modulus
        ax2.axhline(y=expected_count, color='r', linestyle='--', 
                   label=f'Expected uniform: {expected_count:.1f}')
        ax2.legend()
        
        plt.suptitle(config.title, fontsize=14, fontweight='bold')
        
        if config.save_path:
            fig.savefig(config.save_path, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_prime_analysis(self, primes: List[int], config: PlotConfig) -> plt.Figure:
        """Comprehensive prime number analysis visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=config.dpi)
        
        # Prime distribution
        ax1.plot(primes, [1] * len(primes), 'b|', markersize=10)
        ax1.set_xlabel('Number')
        ax1.set_ylabel('Primes')
        ax1.set_title('Prime Number Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Prime gaps
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        gap_positions = primes[:-1]
        
        ax2.plot(gap_positions, gaps, 'r-', linewidth=1, marker='o', markersize=3, alpha=0.7)
        ax2.set_xlabel('Prime Position')
        ax2.set_ylabel('Gap Size')
        ax2.set_title('Prime Gaps')
        ax2.grid(True, alpha=0.3)
        
        # Gap distribution
        gap_counts = {}
        for gap in gaps:
            gap_counts[gap] = gap_counts.get(gap, 0) + 1
        
        ax3.bar(gap_counts.keys(), gap_counts.values(), alpha=0.7)
        ax3.set_xlabel('Gap Size')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Prime Gap Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Prime density approximation
        x = np.arange(10, max(primes), 100)
        prime_count_approx = x / np.log(x)
        
        actual_counts = []
        for threshold in x:
            count = sum(1 for p in primes if p <= threshold)
            actual_counts.append(count)
        
        ax4.plot(x, actual_counts, 'b-', label='Actual π(x)', linewidth=2)
        ax4.plot(x, prime_count_approx, 'r--', label='x/ln(x) approximation', linewidth=2)
        ax4.set_xlabel('x')
        ax4.set_ylabel('π(x)')
        ax4.set_title('Prime Counting Function')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(config.title, fontsize=14, fontweight='bold')
        
        if config.save_path:
            fig.savefig(config.save_path, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_lattice_points(self, sequence_x: List[int], sequence_y: List[int], 
                          config: PlotConfig) -> plt.Figure:
        """Plot sequence as lattice points in 2D space"""
        
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        
        # Create lattice plot
        scatter = ax.scatter(sequence_x, sequence_y, c=range(len(sequence_x)), 
                          cmap='viridis', s=50, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Index')
        
        # Connect points with lines to show progression
        ax.plot(sequence_x, sequence_y, 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_title(config.title)
        ax.grid(True, alpha=0.3)
        
        if config.save_path:
            fig.savefig(config.save_path, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_heatmap(self, data: List[List[int]], config: PlotConfig, 
                    x_labels: Optional[List[str]] = None,
                    y_labels: Optional[List[str]] = None) -> plt.Figure:
        """Create heatmap visualization for 2D data"""
        
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        
        # Create heatmap
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(config.ylabel)
        
        # Set labels
        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45)
        
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
        
        # Add text annotations
        for i in range(len(data)):
            for j in range(len(data[i])):
                text = ax.text(j, i, str(data[i][j]), ha="center", va="center", 
                             color="white" if data[i][j] > np.max(data)/2 else "black")
        
        ax.set_title(config.title)
        
        if config.save_path:
            fig.savefig(config.save_path, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_circular_sequence(self, sequence: List[int], config: PlotConfig,
                             group_size: Optional[int] = None) -> plt.Figure:
        """Plot sequence on a circular arrangement (useful for modular patterns)"""
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi)
        
        n = len(sequence)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # Convert sequence to colors and sizes
        values = np.array(sequence)
        colors = plt.cm.viridis((values - values.min()) / (values.max() - values.min()))
        sizes = 50 + 200 * (values - values.min()) / (values.max() - values.min())
        
        # Plot points on circle
        x = np.cos(angles)
        y = np.sin(angles)
        
        scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.7)
        
        # Add circle
        circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Group highlighting if specified
        if group_size and group_size > 1:
            for i in range(0, n, group_size):
                if i + group_size <= n:
                    angles_group = angles[i:i+group_size]
                    angles_extended = list(angles_group) + [angles_group[0]]
                    x_group = np.cos(angles_extended)
                    y_group = np.sin(angles_extended)
                    ax.plot(x_group, y_group, 'r-', alpha=0.3, linewidth=1)
        
        # Add value labels for first few points
        for i in range(min(12, n)):  # Label first 12 points to avoid clutter
            ax.annotate(str(sequence[i]), (x[i]*1.15, y[i]*1.15), 
                       ha='center', va='center', fontsize=8)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(config.title)
        ax.axis('off')
        
        if config.save_path:
            fig.savefig(config.save_path, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def create_analysis_dashboard(self, data: Dict[str, Any], 
                               config: PlotConfig) -> plt.Figure:
        """Create comprehensive analysis dashboard"""
        
        fig = plt.figure(figsize=(16, 12), dpi=config.dpi)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main sequence plot
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        if 'sequence' in data:
            sequence = data['sequence']
            ax1.plot(range(len(sequence)), sequence, 'b-', linewidth=2, marker='o', markersize=3)
            ax1.set_xlabel('Index')
            ax1.set_ylabel('Value')
            ax1.set_title('Sequence')
            ax1.grid(True, alpha=0.3)
        
        # Statistics
        ax2 = fig.add_subplot(gs[0, 2])
        if 'stats' in data:
            stats = data['stats']
            stats_names = list(stats.keys())
            stats_values = list(stats.values())
            ax2.bar(stats_names, stats_values, alpha=0.7)
            ax2.set_title('Statistics')
            ax2.tick_params(axis='x', rotation=45)
        
        # Distribution
        ax3 = fig.add_subplot(gs[0, 3])
        if 'sequence' in data:
            sequence = data['sequence']
            ax3.hist(sequence, bins=20, alpha=0.7, edgecolor='black')
            ax3.set_title('Value Distribution')
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Frequency')
        
        # Growth rate
        ax4 = fig.add_subplot(gs[1, 2])
        if 'sequence' in data:
            sequence = data['sequence']
            if len(sequence) > 1:
                ratios = [sequence[i]/sequence[i-1] for i in range(1, len(sequence)) 
                         if sequence[i-1] != 0]
                ax4.plot(range(len(ratios)), ratios, 'r-', linewidth=1, marker='o', markersize=2)
                ax4.set_title('Growth Ratios')
                ax4.set_xlabel('Index')
                ax4.set_ylabel('Ratio')
                ax4.grid(True, alpha=0.3)
        
        # Periodicity analysis
        ax5 = fig.add_subplot(gs[1, 3])
        if 'periodicity' in data:
            period_data = data['periodicity']
            periods = list(period_data.keys())
            confidences = list(period_data.values())
            ax5.bar(periods, confidences, alpha=0.7)
            ax5.set_title('Periodicity Analysis')
            ax5.set_xlabel('Period')
            ax5.set_ylabel('Confidence')
        
        # Modular patterns
        ax6 = fig.add_subplot(gs[2, :2])
        if 'sequence' in data and 'modulus' in data:
            sequence = data['sequence']
            modulus = data['modulus']
            residues = [x % modulus for x in sequence]
            
            for residue in range(modulus):
                positions = [i for i, r in enumerate(residues) if r == residue]
                if positions:
                    ax6.scatter(positions, [residue] * len(positions), 
                              alpha=0.6, s=30, label=f'({residue} mod {modulus})')
            
            ax6.set_xlabel('Index')
            ax6.set_ylabel(f'Residue mod {modulus}')
            ax6.set_title('Modular Patterns')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax6.grid(True, alpha=0.3)
        
        # Information panel
        ax7 = fig.add_subplot(gs[2, 2:])
        ax7.axis('off')
        
        info_text = f"Analysis Summary\n" + "="*30 + "\n"
        if 'sequence' in data:
            seq = data['sequence']
            info_text += f"Sequence Length: {len(seq)}\n"
            info_text += f"Min Value: {min(seq)}\n"
            info_text += f"Max Value: {max(seq)}\n"
        
        if 'confidence' in data:
            info_text += f"Overall Confidence: {data['confidence']:.2f}\n"
        
        if 'patterns' in data:
            info_text += f"\nDetected Patterns:\n"
            for pattern in data['patterns']:
                info_text += f"• {pattern}\n"
        
        ax7.text(0.05, 0.95, info_text, transform=ax7.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(config.title, fontsize=16, fontweight='bold')
        
        if config.save_path:
            fig.savefig(config.save_path, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def save_all_figures(self, filename_prefix: str = "plot", format: str = "png"):
        """Save all generated figures"""
        
        for i, fig in enumerate(self.figures):
            filename = f"{filename_prefix}_{i+1}.{format}"
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Saved: {filename}")
    
    def create_pdf_report(self, filename: str = "analysis_report.pdf"):
        """Create PDF report with all figures"""
        
        with PdfPages(filename) as pdf:
            for fig in self.figures:
                pdf.savefig(fig, bbox_inches='tight')
        
        print(f"PDF report created: {filename}")
    
    def clear_figures(self):
        """Clear all stored figures"""
        for fig in self.figures:
            plt.close(fig)
        self.figures = []

def main():
    """Demonstrate visualization capabilities"""
    
    print("📊 MATHEMATICAL VISUALIZER DEMONSTRATION")
    print("=" * 50)
    
    viz = MathematicalVisualizer()
    
    # Sample data
    fib_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Sequence plot
    config1 = PlotConfig(
        title="Fibonacci Sequence",
        xlabel="Index",
        ylabel="Value"
    )
    viz.plot_sequence(fib_seq, config1)
    
    # Growth comparison
    sequences = {
        "Fibonacci": fib_seq,
        "Linear": list(range(1, len(fib_seq) + 1)),
        "Exponential": [2**i for i in range(len(fib_seq))]
    }
    config2 = PlotConfig(
        title="Growth Pattern Comparison",
        xlabel="Index",
        ylabel="Value"
    )
    viz.plot_growth_analysis(sequences, config2)
    
    # Prime analysis
    config3 = PlotConfig(
        title="Prime Number Analysis",
        xlabel="Number",
        ylabel="Value"
    )
    viz.plot_prime_analysis(primes, config3)
    
    # Modular pattern
    config4 = PlotConfig(
        title="Fibonacci Mod 10",
        xlabel="Index",
        ylabel="Residue"
    )
    viz.plot_modular_patterns(fib_seq, 10, config4)
    
    print(f"Generated {len(viz.figures)} visualization figures")
    viz.save_all_figures("demo_plot", "png")
    
    return viz

if __name__ == "__main__":
    viz = main()
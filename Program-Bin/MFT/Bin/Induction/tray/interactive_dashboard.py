#!/usr/bin/env python3
"""
Interactive Dashboard: Real-time numerical analysis interface
Built for accessible mathematical research and education
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import json
from typing import List, Dict, Any

# Import our analysis modules
from induction_validator import MathematicalInduction, PrimeInduction, LucasInduction
from numerical_variation import NumericalVariationAnalyzer, SequenceGenerator

class InductionDashboard:
    """Interactive dashboard for induction and variation analysis"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Mathematical Induction & Numerical Variation Dashboard")
        self.root.geometry("1200x800")
        
        # Initialize analyzers
        self.induction = MathematicalInduction()
        self.prime_ind = PrimeInduction()
        self.lucas_ind = LucasInduction()
        self.variation_analyzer = NumericalVariationAnalyzer()
        self.generator = SequenceGenerator()
        
        # Results storage
        self.current_results = []
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        """Create all dashboard widgets"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Analysis Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Analysis type selection
        ttk.Label(control_frame, text="Analysis Type:").grid(row=0, column=0, sticky=tk.W)
        self.analysis_type = ttk.Combobox(control_frame, 
                                         values=["Mathematical Induction", "Prime Analysis", 
                                                "Lucas Sequences", "Numerical Variation", "Custom Sequence"],
                                         width=25)
        self.analysis_type.grid(row=0, column=1, pady=5)
        self.analysis_type.set("Mathematical Induction")
        self.analysis_type.bind("<<ComboboxSelected>>", self.on_analysis_type_changed)
        
        # Parameter inputs
        param_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Dynamic parameter inputs
        self.param_frame = param_frame
        self.create_parameter_inputs("Mathematical Induction")
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Data", command=self.export_results).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Right panel - Results
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, width=60, height=20, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        viz_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
    def create_parameter_inputs(self, analysis_type: str):
        """Create dynamic parameter inputs based on analysis type"""
        
        # Clear existing widgets
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        self.param_entries = {}
        
        if analysis_type == "Mathematical Induction":
            ttk.Label(self.param_frame, text="Hypothesis:").grid(row=0, column=0, sticky=tk.W)
            self.param_entries['hypothesis'] = ttk.Entry(self.param_frame, width=30)
            self.param_entries['hypothesis'].grid(row=0, column=1, pady=2)
            self.param_entries['hypothesis'].insert(0, "Sum of first n integers = n(n+1)/2")
            
            ttk.Label(self.param_frame, text="Test Limit:").grid(row=1, column=0, sticky=tk.W)
            self.param_entries['test_limit'] = ttk.Entry(self.param_frame, width=30)
            self.param_entries['test_limit'].grid(row=1, column=1, pady=2)
            self.param_entries['test_limit'].insert(0, "50")
            
        elif analysis_type == "Prime Analysis":
            ttk.Label(self.param_frame, text="Conjecture:").grid(row=0, column=0, sticky=tk.W)
            self.param_entries['conjecture'] = ttk.Combobox(self.param_frame, 
                                                           values=["Twin Prime Conjecture", "Goldbach Conjecture"],
                                                           width=27)
            self.param_entries['conjecture'].grid(row=0, column=1, pady=2)
            self.param_entries['conjecture'].set("Twin Prime Conjecture")
            
            ttk.Label(self.param_frame, text="Search Limit:").grid(row=1, column=0, sticky=tk.W)
            self.param_entries['search_limit'] = ttk.Entry(self.param_frame, width=30)
            self.param_entries['search_limit'].grid(row=1, column=1, pady=2)
            self.param_entries['search_limit'].insert(0, "1000")
            
        elif analysis_type == "Lucas Sequences":
            ttk.Label(self.param_frame, text="Sequence:").grid(row=0, column=0, sticky=tk.W)
            self.param_entries['sequence'] = ttk.Combobox(self.param_frame, 
                                                         values=["Fibonacci", "Lucas", "Pell"],
                                                         width=27)
            self.param_entries['sequence'].grid(row=0, column=1, pady=2)
            self.param_entries['sequence'].set("Fibonacci")
            
            ttk.Label(self.param_frame, text="Terms:").grid(row=1, column=0, sticky=tk.W)
            self.param_entries['terms'] = ttk.Entry(self.param_frame, width=30)
            self.param_entries['terms'].grid(row=1, column=1, pady=2)
            self.param_entries['terms'].insert(0, "30")
            
        elif analysis_type == "Numerical Variation":
            ttk.Label(self.param_frame, text="Sequence Type:").grid(row=0, column=0, sticky=tk.W)
            self.param_entries['seq_type'] = ttk.Combobox(self.param_frame, 
                                                         values=["Fibonacci", "Lucas", "Primes", "Perfect Powers"],
                                                         width=27)
            self.param_entries['seq_type'].grid(row=0, column=1, pady=2)
            self.param_entries['seq_type'].set("Fibonacci")
            
            ttk.Label(self.param_frame, text="Analysis:").grid(row=1, column=0, sticky=tk.W)
            self.param_entries['analysis_type'] = ttk.Combobox(self.param_frame, 
                                                             values=["Periodicity", "Growth Rate", "Modular Patterns"],
                                                             width=27)
            self.param_entries['analysis_type'].grid(row=1, column=1, pady=2)
            self.param_entries['analysis_type'].set("Periodicity")
            
            ttk.Label(self.param_frame, text="Modulus (if applicable):").grid(row=2, column=0, sticky=tk.W)
            self.param_entries['modulus'] = ttk.Entry(self.param_frame, width=30)
            self.param_entries['modulus'].grid(row=2, column=1, pady=2)
            self.param_entries['modulus'].insert(0, "10")
            
        elif analysis_type == "Custom Sequence":
            ttk.Label(self.param_frame, text="Sequence (comma-separated):").grid(row=0, column=0, sticky=tk.W)
            self.param_entries['custom_seq'] = ttk.Entry(self.param_frame, width=30)
            self.param_entries['custom_seq'].grid(row=0, column=1, pady=2)
            self.param_entries['custom_seq'].insert(0, "1,1,2,3,5,8,13,21,34,55")
            
            ttk.Label(self.param_frame, text="Analysis Type:").grid(row=1, column=0, sticky=tk.W)
            self.param_entries['custom_analysis'] = ttk.Combobox(self.param_frame, 
                                                                values=["Periodicity", "Growth Rate", "Statistics"],
                                                                width=27)
            self.param_entries['custom_analysis'].grid(row=1, column=1, pady=2)
            self.param_entries['custom_analysis'].set("Periodicity")
    
    def on_analysis_type_changed(self, event=None):
        """Handle analysis type change"""
        analysis_type = self.analysis_type.get()
        self.create_parameter_inputs(analysis_type)
    
    def run_analysis(self):
        """Run the selected analysis in a separate thread"""
        
        self.progress.start()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Running analysis...\n\n")
        
        # Run analysis in separate thread to avoid UI freezing
        thread = threading.Thread(target=self._run_analysis_thread)
        thread.daemon = True
        thread.start()
    
    def _run_analysis_thread(self):
        """Execute analysis in background thread"""
        
        try:
            analysis_type = self.analysis_type.get()
            
            if analysis_type == "Mathematical Induction":
                result = self._run_induction_analysis()
            elif analysis_type == "Prime Analysis":
                result = self._run_prime_analysis()
            elif analysis_type == "Lucas Sequences":
                result = self._run_lucas_analysis()
            elif analysis_type == "Numerical Variation":
                result = self._run_variation_analysis()
            elif analysis_type == "Custom Sequence":
                result = self._run_custom_analysis()
            else:
                result = {"error": "Unknown analysis type"}
            
            # Update UI in main thread
            self.root.after(0, self._update_results, result)
            
        except Exception as e:
            self.root.after(0, self._update_results, {"error": str(e)})
        
        finally:
            self.root.after(0, self.progress.stop)
    
    def _run_induction_analysis(self):
        """Run mathematical induction analysis"""
        
        hypothesis = self.param_entries['hypothesis'].get()
        test_limit = int(self.param_entries['test_limit'].get())
        
        def sum_formula(n: int) -> bool:
            return sum(range(1, n + 1)) == n * (n + 1) // 2
        
        result = self.induction.prove_by_induction(
            hypothesis=hypothesis,
            base_cases=[1, 2, 3],
            inductive_step=sum_formula,
            verify_limit=test_limit
        )
        
        return {
            "type": "induction",
            "result": result,
            "sequence": [sum(range(1, n + 1)) for n in range(1, test_limit + 1)]
        }
    
    def _run_prime_analysis(self):
        """Run prime number analysis"""
        
        conjecture = self.param_entries['conjecture'].get()
        search_limit = int(self.param_entries['search_limit'].get())
        
        if conjecture == "Twin Prime Conjecture":
            result = self.prime_ind.test_twin_prime_conjecture(limit=search_limit)
        elif conjecture == "Goldbach Conjecture":
            result = self.prime_ind.test_goldbach_conjecture(limit=search_limit)
        else:
            result = None
        
        # Generate prime sequence for visualization
        primes = self.generator.primes_up_to(search_limit)
        
        return {
            "type": "prime",
            "result": result,
            "sequence": primes
        }
    
    def _run_lucas_analysis(self):
        """Run Lucas sequence analysis"""
        
        seq_type = self.param_entries['sequence'].get()
        terms = int(self.param_entries['terms'].get())
        
        if seq_type == "Fibonacci":
            seq = [self.generator.fibonacci(i) for i in range(1, terms + 1)]
            result = self.lucas_ind.test_fibonacci_divisibility(test_limit=min(terms, 50))
        elif seq_type == "Lucas":
            seq = [self.generator.lucas(i) for i in range(terms)]
            result = None  # Add Lucas-specific tests here
        elif seq_type == "Pell":
            seq = [self.generator.pell(i) for i in range(terms)]
            result = None  # Add Pell-specific tests here
        else:
            seq = []
            result = None
        
        return {
            "type": "lucas",
            "result": result,
            "sequence": seq
        }
    
    def _run_variation_analysis(self):
        """Run numerical variation analysis"""
        
        seq_type = self.param_entries['seq_type'].get()
        analysis_type = self.param_entries['analysis_type'].get()
        modulus = int(self.param_entries['modulus'].get())
        
        if seq_type == "Fibonacci":
            seq = [self.generator.fibonacci(i) for i in range(1, 51)]
        elif seq_type == "Lucas":
            seq = [self.generator.lucas(i) for i in range(51)]
        elif seq_type == "Primes":
            seq = self.generator.primes_up_to(500)
        elif seq_type == "Perfect Powers":
            seq = self.generator.perfect_powers(1000)
        else:
            seq = []
        
        # Run selected analysis
        if analysis_type == "Periodicity":
            result = self.variation_analyzer.analyze_periodicity(seq)
        elif analysis_type == "Growth Rate":
            result = self.variation_analyzer.analyze_growth_rate(seq)
        elif analysis_type == "Modular Patterns":
            result = self.variation_analyzer.analyze_modular_patterns(seq, modulus)
        else:
            result = None
        
        return {
            "type": "variation",
            "result": result,
            "sequence": seq
        }
    
    def _run_custom_analysis(self):
        """Run analysis on custom sequence"""
        
        seq_str = self.param_entries['custom_seq'].get()
        analysis_type = self.param_entries['custom_analysis'].get()
        
        try:
            seq = [int(x.strip()) for x in seq_str.split(',')]
        except ValueError:
            return {"error": "Invalid sequence format"}
        
        # Run selected analysis
        if analysis_type == "Periodicity":
            result = self.variation_analyzer.analyze_periodicity(seq)
        elif analysis_type == "Growth Rate":
            result = self.variation_analyzer.analyze_growth_rate(seq)
        elif analysis_type == "Statistics":
            # Simple statistics
            result = {
                "mean": np.mean(seq),
                "std": np.std(seq),
                "min": min(seq),
                "max": max(seq),
                "length": len(seq)
            }
        else:
            result = None
        
        return {
            "type": "custom",
            "result": result,
            "sequence": seq
        }
    
    def _update_results(self, result):
        """Update UI with analysis results"""
        
        self.results_text.delete(1.0, tk.END)
        
        if "error" in result:
            self.results_text.insert(tk.END, f"Error: {result['error']}")
            return
        
        # Display text results
        if result["result"]:
            if hasattr(result["result"], "hypothesis"):
                # Induction result
                r = result["result"]
                text = f"Hypothesis: {r.hypothesis}\n"
                text += f"Confidence: {r.confidence:.1%}\n"
                text += f"Counterexample: {r.counterexample}\n\n"
                
                if r.proof:
                    text += f"Proof Structure:\n{r.proof}\n\n"
                
                if r.limitations:
                    text += "Limitations:\n"
                    for limit in r.limitations:
                        text += f"  • {limit}\n"
                
                self.results_text.insert(tk.END, text)
            
            elif hasattr(result["result"], "pattern_type"):
                # Variation result
                r = result["result"]
                text = f"Pattern Type: {r.pattern_type.value}\n"
                text += f"Description: {r.description}\n"
                text += f"Confidence: {r.confidence:.2f}\n\n"
                
                text += "Parameters:\n"
                for key, value in r.parameters.items():
                    text += f"  {key}: {value}\n"
                
                self.results_text.insert(tk.END, text)
            
            else:
                # Custom result
                self.results_text.insert(tk.END, f"Result: {result['result']}\n")
        
        # Update visualization
        self._update_visualization(result)
        
        # Store results
        self.current_results.append(result)
    
    def _update_visualization(self, result):
        """Update matplotlib visualization"""
        
        self.ax.clear()
        
        if "sequence" in result and result["sequence"]:
            seq = result["sequence"]
            
            # Plot sequence
            self.ax.plot(range(len(seq)), seq, 'b-', linewidth=2, marker='o', markersize=4)
            self.ax.set_xlabel("Index")
            self.ax.set_ylabel("Value")
            self.ax.set_title(f"{result['type'].title()} Sequence Analysis")
            self.ax.grid(True, alpha=0.3)
            
            # Add pattern information if available
            if result["result"] and hasattr(result["result"], "parameters"):
                params = result["result"].parameters
                if "period" in params:
                    self.ax.axvline(x=params["period"], color='r', linestyle='--', 
                                   label=f"Period = {params['period']}")
                    self.ax.legend()
        
        elif result["type"] == "prime":
            # Special visualization for primes
            seq = result["sequence"]
            if seq:
                # Plot prime distribution
                gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
                
                self.ax.bar(range(len(gaps)), gaps, alpha=0.7)
                self.ax.set_xlabel("Gap Index")
                self.ax.set_ylabel("Gap Size")
                self.ax.set_title("Prime Gap Distribution")
                self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def clear_results(self):
        """Clear all results"""
        self.results_text.delete(1.0, tk.END)
        self.ax.clear()
        self.canvas.draw()
        self.current_results = []
    
    def export_results(self):
        """Export results to JSON file"""
        
        if not self.current_results:
            messagebox.showwarning("No Results", "No results to export")
            return
        
        # Convert results to serializable format
        export_data = []
        for result in self.current_results:
            export_item = {
                "type": result["type"],
                "timestamp": str(np.datetime64('now')),
            }
            
            if "sequence" in result:
                export_item["sequence_length"] = len(result["sequence"])
                if len(result["sequence"]) <= 20:
                    export_item["sequence_sample"] = result["sequence"]
            
            if result["result"]:
                if hasattr(result["result"], "hypothesis"):
                    export_item["hypothesis"] = result["result"].hypothesis
                    export_item["confidence"] = result["result"].confidence
                    export_item["counterexample"] = result["result"].counterexample
                elif hasattr(result["result"], "pattern_type"):
                    export_item["pattern_type"] = result["result"].pattern_type.value
                    export_item["description"] = result["result"].description
                    export_item["confidence"] = result["result"].confidence
            
            export_data.append(export_item)
        
        # Save to file
        try:
            with open("induction_results.json", "w") as f:
                json.dump(export_data, f, indent=2)
            messagebox.showinfo("Export Successful", "Results exported to induction_results.json")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export: {e}")

def main():
    """Launch the interactive dashboard"""
    
    root = tk.Tk()
    dashboard = InductionDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()
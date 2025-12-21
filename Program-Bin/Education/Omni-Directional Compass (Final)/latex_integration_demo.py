"""
LaTeX Integration Demo for Omni-Directional Compass
Showcasing the groundbreaking LaTeX reverse engineering capabilities
"""

import tkinter as tk
from tkinter import ttk, Frame, Label, Button, Text, Scrollbar
from groundbreaking_gui import GroundbreakingCompassGUI
from modules.latex_integration import LatexIntegration
import omni_directional_compass
import time

class LaTeXIntegrationDemo:
    """Demo showcasing LaTeX integration capabilities"""
    
    def __init__(self):
        self.setup_demo()
        
    def setup_demo(self):
        """Setup the demo interface"""
        self.root = tk.Tk()
        self.root.title("🔬 LaTeX Integration Demo - Omni-Directional Compass")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a0f')
        
        # Initialize compass and LaTeX integration
        self.compass = omni_directional_compass.OmniDirectionalCompass()
        self.latex_integration = LatexIntegration(self.compass)
        
        self.create_demo_interface()
        
    def create_demo_interface(self):
        """Create demo interface"""
        # Header
        header = Frame(self.root, bg='#1a1a2f', height=100)
        header.pack(fill='x', padx=5, pady=5)
        
        title = Label(header, text="🔬 LATEX INTEGRATION DEMO",
                     font=('Arial', 24, 'bold'), bg='#1a1a2f', fg='#00ffff')
        title.pack(pady=10)
        
        subtitle = Label(header, text="Revolutionary LaTeX Encoding & Reverse Engineering for Mathematical Substantiation",
                        font=('Arial', 12), bg='#1a1a2f', fg='#88ff88')
        subtitle.pack()
        
        # Main content
        main_frame = Frame(self.root, bg='#0a0a0f')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Demo controls
        left_panel = Frame(main_frame, bg='#1a1a2f', width=300)
        left_panel.pack(side='left', fill='y', padx=5)
        
        Label(left_panel, text="🎮 Demo Controls",
              font=('Arial', 14, 'bold'), bg='#1a1a2f', fg='#00ffff').pack(pady=10)
        
        # Demo scenarios
        scenarios = [
            ("📊 Empirinometry Test", self.demo_empirinometry),
            ("🧮 Standard Math Test", self.demo_standard_math),
            ("🔬 Complex Formula Test", self.demo_complex_formula),
            ("🔄 Reverse Engineering Test", self.demo_reverse_engineering),
            ("✅ Validation Test", self.demo_validation),
            ("📚 Batch Processing Test", self.demo_batch_processing),
            ("🎨 Launch Full GUI", self.launch_full_gui)
        ]
        
        for text, command in scenarios:
            btn = Button(left_panel, text=text, font=('Arial', 11, 'bold'),
                        bg='#3a3a4f', fg='white', width=25, height=2,
                        command=command)
            btn.pack(pady=5)
            
        # Right panel - Results display
        right_panel = Frame(main_frame, bg='#1a1a2f')
        right_panel.pack(side='right', fill='both', expand=True, padx=5)
        
        Label(right_panel, text="📋 Demo Results",
              font=('Arial', 14, 'bold'), bg='#1a1a2f', fg='#00ffff').pack(pady=5)
        
        # Results text area
        text_frame = Frame(right_panel, bg='#1a1a2f')
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.results_text = Text(text_frame, bg='#0a0a0f', fg='#00ff88',
                                font=('Courier', 10), wrap='word')
        scrollbar = Scrollbar(text_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Status bar
        self.status_frame = Frame(self.root, bg='#2a2a3f', height=30)
        self.status_frame.pack(fill='x', padx=5, pady=5)
        
        self.status_label = Label(self.status_frame, text="Demo ready - Click a scenario to begin!",
                                 font=('Arial', 10), bg='#2a2a3f', fg='#00ff88')
        self.status_label.pack(side='left', padx=10)
        
    def demo_empirinometry(self):
        """Demo Empirinometry LaTeX encoding"""
        self.update_status("Testing Empirinometry LaTeX encoding...")
        self.clear_results()
        
        formulas = ["10 # 5", "8 # 3", "15 # 7", "20 # 10"]
        
        self.add_results("🔬 EMPIRINOMETRY LATEX ENCODING TEST")
        self.add_results("=" * 50)
        
        for formula in formulas:
            self.add_results(f"\nFormula: {formula}")
            representations = self.latex_integration.encode_formula_to_latex(formula)
            
            for rep in representations:
                self.add_results(f"  {rep.formula_type}: {rep.latex_code}")
                
            # Validate
            validation = self.latex_integration.validate_substantiation_with_latex(formula, representations)
            self.add_results(f"  Validation: {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}")
            
        self.update_status("Empirinometry test completed")
        
    def demo_standard_math(self):
        """Demo standard math LaTeX encoding"""
        self.update_status("Testing standard math LaTeX encoding...")
        self.clear_results()
        
        formulas = ["2 * pi * r", "sqrt(x^2 + y^2)", "E = m * c^2", "F = m * a"]
        
        self.add_results("🧮 STANDARD MATH LATEX ENCODING TEST")
        self.add_results("=" * 50)
        
        for formula in formulas:
            self.add_results(f"\nFormula: {formula}")
            representations = self.latex_integration.encode_formula_to_latex(formula)
            
            for rep in representations:
                self.add_results(f"  {rep.formula_type}: {rep.latex_code}")
                
        self.update_status("Standard math test completed")
        
    def demo_complex_formula(self):
        """Demo complex formula LaTeX encoding"""
        self.update_status("Testing complex formula LaTeX encoding...")
        self.clear_results()
        
        formulas = [
            "lambda(13) + beta_transform(5)",
            "2 * pi * sqrt(r**2 + h**2)",
            "integral(x^2, (x, 0, 1))",
            "matrix([[1, 2], [3, 4]])"
        ]
        
        self.add_results("🔬 COMPLEX FORMULA LATEX ENCODING TEST")
        self.add_results("=" * 50)
        
        for formula in formulas:
            self.add_results(f"\nFormula: {formula}")
            try:
                representations = self.latex_integration.encode_formula_to_latex(formula)
                
                for rep in representations:
                    self.add_results(f"  {rep.formula_type}: {rep.latex_code}")
                    self.add_results(f"    Description: {rep.description}")
            except Exception as e:
                self.add_results(f"  Error: {str(e)}")
                
        self.update_status("Complex formula test completed")
        
    def demo_reverse_engineering(self):
        """Demo LaTeX reverse engineering"""
        self.update_status("Testing LaTeX reverse engineering...")
        self.clear_results()
        
        latex_codes = [
            r"\frac{25}{2}",
            r"\sqrt{2} + \pi",
            r"E = m \cdot c^2",
            r"\sum_{i=1}^{n} i^2"
        ]
        
        self.add_results("🔄 LATEX REVERSE ENGINEERING TEST")
        self.add_results("=" * 50)
        
        for latex in latex_codes:
            self.add_results(f"\nLaTeX: {latex}")
            result = self.latex_integration.reverse_engineer_latex(latex)
            
            self.add_results(f"  Python: {result.python_expression}")
            self.add_results(f"  Status: {result.validation_status}")
            self.add_results(f"  Confidence: {result.confidence:.2f}")
            
            if result.evaluation_result is not None:
                self.add_results(f"  Result: {result.evaluation_result}")
                
        self.update_status("Reverse engineering test completed")
        
    def demo_validation(self):
        """Demo LaTeX validation"""
        self.update_status("Testing LaTeX validation...")
        self.clear_results()
        
        self.add_results("✅ LATEX VALIDATION TEST")
        self.add_results("=" * 50)
        
        formula = "10 # 5"
        representations = self.latex_integration.encode_formula_to_latex(formula)
        
        self.add_results(f"Formula: {formula}")
        self.add_results(f"Generated {len(representations)} LaTeX representations\n")
        
        validation = self.latex_integration.validate_substantiation_with_latex(formula, representations)
        
        self.add_results(f"Validation Status: {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}")
        self.add_results(f"LaTeX Matches: {validation['latex_matches']}/{validation['total_versions']}")
        
        if validation['inconsistencies']:
            self.add_results(f"\nInconsistencies found: {len(validation['inconsistencies'])}")
            for i, issue in enumerate(validation['inconsistencies'], 1):
                self.add_results(f"  {i}. {issue.get('type', 'Unknown')}: {issue.get('error', 'No error')}")
                
        self.add_results(f"\nRecommendations:")
        for rec in validation['recommendations']:
            self.add_results(f"  • {rec}")
            
        self.update_status("Validation test completed")
        
    def demo_batch_processing(self):
        """Demo batch processing"""
        self.update_status("Testing batch processing...")
        self.clear_results()
        
        formulas = [
            "10 # 5",
            "2 * pi",
            "sqrt(16)",
            "F = m * a",
            "lambda(13)"
        ]
        
        self.add_results("📚 BATCH PROCESSING TEST")
        self.add_results("=" * 50)
        
        start_time = time.time()
        
        batch_results = self.latex_integration.batch_latex_conversion(formulas)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.add_results(f"Processed {len(formulas)} formulas in {processing_time:.3f} seconds")
        self.add_results(f"Average time per formula: {processing_time/len(formulas):.3f} seconds\n")
        
        for i, (formula, reps) in enumerate(zip(formulas, batch_results)):
            self.add_results(f"{i+1}. {formula}: {len(reps)} representations")
            for rep in reps:
                self.add_results(f"   - {rep.formula_type}: {rep.latex_code}")
                
        self.update_status("Batch processing test completed")
        
    def launch_full_gui(self):
        """Launch the full GUI with LaTeX workshop"""
        self.update_status("Launching full GUI with LaTeX workshop...")
        
        try:
            # Hide demo window
            self.root.withdraw()
            
            # Create full GUI
            gui_root = tk.Tk()
            gui_root.title("🚀 Omni-Directional Compass with LaTeX Workshop")
            gui_root.geometry("1400x900")
            
            app = GroundbreakingCompassGUI(gui_root)
            app.pack(fill='both', expand=True)
            
            # Show info
            self.add_results("🎨 FULL GUI LAUNCHED")
            self.add_results("=" * 30)
            self.add_results("✅ Groundbreaking GUI with quantum visualization")
            self.add_results("✅ Massive Substantiation Workshop")
            self.add_results("✅ NEW: LaTeX Reverse Engineering Workshop")
            self.add_results("✅ All tabs integrated and functional")
            self.add_results("\n📝 Usage:")
            self.add_results("1. Click '🔬 LaTeX Workshop' tab")
            self.add_results("2. Enter formulas like '10 # 5' or '2 * pi * r'")
            self.add_results("3. Click '🔄 Encode to LaTeX'")
            self.add_results("4. Explore different LaTeX representations")
            self.add_results("5. Use '🔍 Reverse Engineer' to convert back")
            self.add_results("6. Validate with '✅ Validate All'")
            
            self.update_status("Full GUI launched successfully!")
            
        except Exception as e:
            self.add_results(f"❌ Error launching GUI: {str(e)}")
            self.update_status("Error launching full GUI")
            
    def clear_results(self):
        """Clear results display"""
        self.results_text.delete('1.0', 'end')
        
    def add_results(self, text):
        """Add text to results display"""
        self.results_text.insert('end', text + '\n')
        self.results_text.see('end')
        self.root.update_idletasks()
        
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def run(self):
        """Run the demo"""
        self.add_results("🔬 LaTeX Integration Demo Started!")
        self.add_results("=" * 40)
        self.add_results("This demo showcases the groundbreaking LaTeX")
        self.add_results("integration capabilities of the Omni-Directional Compass.")
        self.add_results("")
        self.add_results("🚀 NEW FEATURES:")
        self.add_results("• LaTeX encoding of substantiation formulas")
        self.add_results("• Multiple representation formats")
        self.add_results("• Reverse engineering from LaTeX")
        self.add_results("• Validation and consistency checking")
        self.add_results("• Empirinometry-specific notation")
        self.add_results("• Integration with groundbreaking GUI")
        self.add_results("")
        self.add_results("Click any demo scenario to begin!")
        
        self.root.mainloop()

# Run the demo
if __name__ == "__main__":
    print("🔬 Starting LaTeX Integration Demo...")
    demo = LaTeXIntegrationDemo()
    demo.run()
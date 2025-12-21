"""
LaTeX Reverse Engineering Workshop for Omni-Directional Compass
A groundbreaking workshop for LaTeX encoding, decoding, and validation
"""

import tkinter as tk
from tkinter import ttk, Canvas, Frame, Label, Button, Entry, Text, Scrollbar, Listbox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass, asdict
import threading
import time

from modules.latex_integration import LatexIntegration, LatexRepresentation, ReverseEngineeringResult

@dataclass
class WorkshopSession:
    """Represents a workshop session"""
    session_id: str
    formula_input: str
    latex_representations: List[LatexRepresentation]
    validation_results: Dict[str, Any]
    timestamp: str
    user_notes: str

class LaTeXReverseEngineeringWorkshop:
    """
    Groundbreaking LaTeX Reverse Engineering Workshop
    Allows users to explore LaTeX encoding/decoding of substantiation formulas
    """
    
    def __init__(self, parent, compass):
        self.parent = parent
        self.compass = compass
        self.latex_integration = LatexIntegration(compass)
        self.current_session = None
        self.session_history = []
        
        self.setup_workshop_interface()
        
    def setup_workshop_interface(self):
        """Setup the LaTeX workshop interface"""
        # Main container
        self.main_frame = Frame(self.parent, bg='#0a0a0f')
        self.main_frame.pack(fill='both', expand=True)
        
        # Header
        self.create_header()
        
        # Main content area
        self.create_main_content()
        
        # Bottom controls
        self.create_bottom_controls()
        
    def create_header(self):
        """Create workshop header"""
        header_frame = Frame(self.main_frame, bg='#1a1a2f', height=80)
        header_frame.pack(fill='x', padx=5, pady=5)
        
        title_label = Label(header_frame, 
                           text="🔬 LATEX REVERSE ENGINEERING WORKSHOP 🔬",
                           font=('Arial', 20, 'bold'), 
                           bg='#1a1a2f', fg='#00ffff')
        title_label.pack(side='top', pady=5)
        
        subtitle_label = Label(header_frame,
                              text="Encode • Decode • Validate • Explore Mathematical Substantiation",
                              font=('Arial', 11),
                              bg='#1a1a2f', fg='#88ff88')
        subtitle_label.pack(side='top')
        
        # Status bar
        self.status_frame = Frame(header_frame, bg='#2a2a3f')
        self.status_frame.pack(side='bottom', fill='x', padx=10, pady=5)
        
        self.status_label = Label(self.status_frame, 
                                 text="Ready for LaTeX reverse engineering!",
                                 font=('Arial', 10), 
                                 bg='#2a2a3f', fg='#00ff88')
        self.status_label.pack(side='left')
        
        self.session_counter = Label(self.status_frame,
                                    text=f"Sessions: {len(self.session_history)}",
                                    font=('Arial', 10),
                                    bg='#2a2a3f', fg='#ffaa00')
        self.session_counter.pack(side='right')
        
    def create_main_content(self):
        """Create main content area"""
        content_frame = Frame(self.main_frame, bg='#0a0a0f')
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left panel - Input and controls
        left_panel = Frame(content_frame, bg='#1a1a2f', width=400)
        left_panel.pack(side='left', fill='y', padx=5, pady=5)
        
        self.create_input_panel(left_panel)
        
        # Center panel - LaTeX representations
        center_panel = Frame(content_frame, bg='#0a0a0f')
        center_panel.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        self.create_latex_display_panel(center_panel)
        
        # Right panel - Validation and analysis
        right_panel = Frame(content_frame, bg='#1a1a2f', width=350)
        right_panel.pack(side='right', fill='y', padx=5, pady=5)
        
        self.create_validation_panel(right_panel)
        
    def create_input_panel(self, parent):
        """Create input panel for formulas"""
        Label(parent, text="📝 Formula Input", 
              font=('Arial', 14, 'bold'), 
              bg='#1a1a2f', fg='#00ffff').pack(pady=10)
        
        # Formula input
        Label(parent, text="Enter Substantiation Formula:",
              font=('Arial', 10),
              bg='#1a1a2f', fg='white').pack(pady=5)
        
        self.formula_entry = Entry(parent, font=('Courier', 12), width=35,
                                  bg='#0a0a0f', fg='#00ff88',
                                  insertbackground='#00ff88')
        self.formula_entry.pack(padx=10, pady=5)
        
        # Example formulas
        Label(parent, text="📋 Example Formulas:",
              font=('Arial', 10, 'bold'),
              bg='#1a1a2f', fg='#ffaa00').pack(pady=(20, 5))
        
        examples = [
            "10 # 5",  # Empirinometry multiplication
            "2 * pi * r",
            "sqrt(x^2 + y^2)",
            "F = m * a",
            "E = m * c^2",
            "lambda(13)",
            "2 + 3 * 4"
        ]
        
        for example in examples:
            btn = Button(parent, text=example, font=('Courier', 9),
                        bg='#3a3a4f', fg='white', width=25,
                        command=lambda e=example: self.load_example(e))
            btn.pack(pady=2)
            
        # Quick actions
        Label(parent, text="⚡ Quick Actions:",
              font=('Arial', 10, 'bold'),
              bg='#1a1a2f', fg='#ff88cc').pack(pady=(20, 5))
        
        actions = [
            ("🔄 Encode to LaTeX", self.encode_formula),
            ("🔍 Reverse Engineer", self.reverse_engineer),
            ("✅ Validate All", self.validate_all),
            ("📊 Batch Process", self.batch_process),
            ("💾 Save Session", self.save_session),
            ("📁 Load Session", self.load_session)
        ]
        
        for text, command in actions:
            btn = Button(parent, text=text, font=('Arial', 10, 'bold'),
                        bg='#4a4a5f', fg='white', width=20,
                        command=command)
            btn.pack(pady=3)
            
    def create_latex_display_panel(self, parent):
        """Create LaTeX display panel"""
        Label(parent, text="🎨 LaTeX Representations", 
              font=('Arial', 14, 'bold'), 
              bg='#0a0a0f', fg='#00ffff').pack(pady=5)
        
        # Create notebook for tabs
        self.latex_notebook = ttk.Notebook(parent)
        self.latex_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tabs for different LaTeX representations
        self.latex_tabs = {}
        self.latex_displays = {}
        
    def create_validation_panel(self, parent):
        """Create validation and analysis panel"""
        Label(parent, text="✅ Validation & Analysis", 
              font=('Arial', 14, 'bold'), 
              bg='#1a1a2f', fg='#00ffff').pack(pady=10)
        
        # Validation results display
        Label(parent, text="Validation Results:",
              font=('Arial', 11, 'bold'),
              bg='#1a1a2f', fg='#88ff88').pack(pady=5)
        
        self.validation_text = Text(parent, height=12, width=40,
                                   bg='#0a0a0f', fg='#00ff88',
                                   font=('Courier', 9),
                                   insertbackground='#00ff88')
        self.validation_text.pack(padx=10, pady=5, fill='both', expand=True)
        
        # Session notes
        Label(parent, text="📝 Session Notes:",
              font=('Arial', 11, 'bold'),
              bg='#1a1a2f', fg='#ffaa00').pack(pady=(10, 5))
        
        self.notes_text = Text(parent, height=5, width=40,
                              bg='#0a0a0f', fg='#ffaa00',
                              font=('Courier', 9),
                              insertbackground='#ffaa00')
        self.notes_text.pack(padx=10, pady=5)
        
        # History
        Label(parent, text="📜 Session History:",
              font=('Arial', 11, 'bold'),
              bg='#1a1a2f', fg='#ff88cc').pack(pady=(10, 5))
        
        self.history_listbox = Listbox(parent, height=6,
                                      bg='#0a0a0f', fg='white',
                                      font=('Courier', 8))
        self.history_listbox.pack(padx=10, pady=5, fill='x')
        
    def create_bottom_controls(self):
        """Create bottom control panel"""
        bottom_frame = Frame(self.main_frame, bg='#1a1a2f', height=60)
        bottom_frame.pack(fill='x', padx=5, pady=5)
        
        # Control buttons
        controls = [
            ("🚀 Quick Encode", self.quick_encode),
            ("🔄 Refresh", self.refresh_display),
            ("📤 Export LaTeX", self.export_latex),
            ("📥 Import LaTeX", self.import_latex),
            ("🧹 Clear All", self.clear_all),
            ("❓ Help", self.show_help)
        ]
        
        for text, command in controls:
            btn = Button(bottom_frame, text=text, font=('Arial', 10, 'bold'),
                        bg='#3a3a4f', fg='white', width=12,
                        command=command)
            btn.pack(side='left', padx=5, pady=10)
            
    def encode_formula(self):
        """Encode the current formula to LaTeX"""
        formula = self.formula_entry.get().strip()
        if not formula:
            self.update_status("Please enter a formula first")
            return
            
        self.update_status("Encoding formula to LaTeX...")
        
        try:
            # Generate LaTeX representations
            latex_reps = self.latex_integration.encode_formula_to_latex(formula)
            
            # Clear existing tabs
            for tab in self.latex_tabs.values():
                tab.destroy()
            self.latex_tabs.clear()
            self.latex_displays.clear()
            
            # Create new tabs for each representation
            for i, latex_rep in enumerate(latex_reps):
                tab_frame = Frame(self.latex_notebook, bg='#0a0a0f')
                
                # Tab title with complexity indicator
                tab_title = f"{latex_rep.formula_type} (🌟{latex_rep.complexity})"
                self.latex_notebook.add(tab_frame, text=tab_title)
                self.latex_tabs[latex_rep.formula_type] = tab_frame
                
                # Create content for tab
                self.create_latex_tab_content(tab_frame, latex_rep)
                
            # Create session
            self.current_session = WorkshopSession(
                session_id=f"session_{int(time.time())}",
                formula_input=formula,
                latex_representations=latex_reps,
                validation_results={},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                user_notes=""
            )
            
            self.update_status(f"Successfully encoded formula: {len(latex_reps)} representations")
            
        except Exception as e:
            self.update_status(f"Error encoding formula: {str(e)}")
            
    def create_latex_tab_content(self, parent, latex_rep: LatexRepresentation):
        """Create content for a LaTeX representation tab"""
        # Info frame
        info_frame = Frame(parent, bg='#1a1a2f')
        info_frame.pack(fill='x', padx=5, pady=5)
        
        Label(info_frame, text=f"Type: {latex_rep.formula_type}",
              font=('Arial', 10, 'bold'),
              bg='#1a1a2f', fg='#00ffff').pack(side='left', padx=5)
        
        Label(info_frame, text=f"Domain: {latex_rep.domain}",
              font=('Arial', 9),
              bg='#1a1a2f', fg='#88ff88').pack(side='left', padx=5)
        
        # LaTeX code display
        Label(parent, text="📐 LaTeX Code:",
              font=('Arial', 10, 'bold'),
              bg='#0a0a0f', fg='#ffaa00').pack(pady=(10, 2))
        
        latex_text = Text(parent, height=4, bg='#1a1a2f', fg='#00ff88',
                         font=('Courier', 11))
        latex_text.pack(padx=5, pady=2, fill='x')
        latex_text.insert('1.0', latex_rep.latex_code)
        latex_text.config(state='disabled')
        
        # Description
        Label(parent, text="📋 Description:",
              font=('Arial', 10, 'bold'),
              bg='#0a0a0f', fg='#ff88cc').pack(pady=(5, 2))
        
        desc_label = Label(parent, text=latex_rep.description,
                          font=('Arial', 9),
                          bg='#0a0a0f', fg='white',
                          wraplength=500)
        desc_label.pack(padx=5, pady=2)
        
        # LaTeX visualization
        try:
            fig = self.latex_integration.create_latex_visualization(
                latex_rep.latex_code, 
                f"{latex_rep.formula_type} Representation"
            )
            
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.get_tk_widget().pack(pady=10, fill='both', expand=True)
            
        except Exception as e:
            # Fallback text display
            error_label = Label(parent, text=f"Visualization error: {str(e)}",
                              font=('Arial', 9),
                              bg='#0a0a0f', fg='#ff6666')
            error_label.pack(pady=10)
            
        # Action buttons for this representation
        action_frame = Frame(parent, bg='#1a1a2f')
        action_frame.pack(fill='x', padx=5, pady=5)
        
        Button(action_frame, text="📋 Copy LaTeX", font=('Arial', 9),
               bg='#3a3a4f', fg='white',
               command=lambda: self.copy_latex(latex_rep.latex_code)).pack(side='left', padx=2)
        
        Button(action_frame, text="🔄 Reverse", font=('Arial', 9),
               bg='#3a3a4f', fg='white',
               command=lambda: self.reverse_from_latex(latex_rep.latex_code)).pack(side='left', padx=2)
        
    def reverse_engineer(self):
        """Reverse engineer LaTeX to formula"""
        # Get current tab's LaTeX
        current_tab = self.latex_notebook.select()
        if not current_tab:
            self.update_status("No LaTeX representation selected")
            return
            
        tab_index = self.latex_notebook.index(current_tab)
        if tab_index >= len(self.current_session.latex_representations):
            self.update_status("Invalid tab selection")
            return
            
        latex_rep = self.current_session.latex_representations[tab_index]
        
        self.update_status("Reverse engineering LaTeX...")
        
        try:
            result = self.latex_integration.reverse_engineer_latex(latex_rep.latex_code)
            
            # Display reverse engineering results
            self.display_reverse_results(result)
            
            self.update_status("Reverse engineering completed")
            
        except Exception as e:
            self.update_status(f"Error reverse engineering: {str(e)}")
            
    def display_reverse_results(self, result: ReverseEngineeringResult):
        """Display reverse engineering results"""
        # Clear validation text
        self.validation_text.delete('1.0', 'end')
        
        # Format results
        results_text = f"🔄 REVERSE ENGINEERING RESULTS\n"
        results_text += f"{'='*40}\n\n"
        results_text += f"Original LaTeX:\n{result.original_latex}\n\n"
        results_text += f"Python Expression:\n{result.python_expression}\n\n"
        results_text += f"Validation Status: {result.validation_status}\n"
        results_text += f"Confidence: {result.confidence:.2f}\n\n"
        
        if result.evaluation_result is not None:
            results_text += f"Evaluation Result: {result.evaluation_result}\n\n"
            
        if result.sympy_expression is not None:
            results_text += f"SymPy Expression: {result.sympy_expression}\n\n"
            
        self.validation_text.insert('1.0', results_text)
        
    def validate_all(self):
        """Validate all LaTeX representations"""
        if not self.current_session:
            self.update_status("No session to validate")
            return
            
        self.update_status("Validating all representations...")
        
        try:
            validation_results = self.latex_integration.validate_substantiation_with_latex(
                self.current_session.formula_input,
                self.current_session.latex_representations
            )
            
            self.current_session.validation_results = validation_results
            
            # Display validation results
            self.display_validation_results(validation_results)
            
            self.update_status(f"Validation complete: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
            
        except Exception as e:
            self.update_status(f"Error during validation: {str(e)}")
            
    def display_validation_results(self, results: Dict[str, Any]):
        """Display validation results"""
        self.validation_text.delete('1.0', 'end')
        
        validation_text = f"✅ VALIDATION RESULTS\n"
        validation_text += f"{'='*40}\n\n"
        validation_text += f"Original Formula: {results['original_formula']}\n\n"
        validation_text += f"Status: {'✅ PASSED' if results['validation_passed'] else '❌ FAILED'}\n"
        validation_text += f"LaTeX Matches: {results['latex_matches']}/{results['total_versions']}\n\n"
        
        if results['inconsistencies']:
            validation_text += f"⚠️ INCONSISTENCIES ({len(results['inconsistencies'])}):\n"
            for i, issue in enumerate(results['inconsistencies'], 1):
                validation_text += f"\n{i}. Type: {issue.get('type', 'Unknown')}\n"
                validation_text += f"   LaTeX: {issue.get('latex', 'N/A')}\n"
                if 'expected' in issue and 'got' in issue:
                    validation_text += f"   Expected: {issue['expected']}, Got: {issue['got']}\n"
                if 'error' in issue:
                    validation_text += f"   Error: {issue['error']}\n"
                    
        validation_text += f"\n📋 RECOMMENDATIONS:\n"
        for rec in results['recommendations']:
            validation_text += f"• {rec}\n"
            
        self.validation_text.insert('1.0', validation_text)
        
    def load_example(self, example: str):
        """Load example formula"""
        self.formula_entry.delete(0, 'end')
        self.formula_entry.insert(0, example)
        self.update_status(f"Loaded example: {example}")
        
    def quick_encode(self):
        """Quick encode current formula"""
        self.encode_formula()
        
    def refresh_display(self):
        """Refresh current display"""
        if self.current_session:
            self.encode_formula()
            if self.current_session.validation_results:
                self.display_validation_results(self.current_session.validation_results)
                
    def copy_latex(self, latex_code: str):
        """Copy LaTeX code to clipboard"""
        self.parent.clipboard_clear()
        self.parent.clipboard_append(latex_code)
        self.update_status("LaTeX code copied to clipboard")
        
    def reverse_from_latex(self, latex_code: str):
        """Reverse engineer from specific LaTeX"""
        try:
            result = self.latex_integration.reverse_engineer_latex(latex_code)
            self.display_reverse_results(result)
            self.update_status("Reverse engineering completed")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            
    def batch_process(self):
        """Process multiple formulas"""
        # Simple batch processing with examples
        formulas = ["10 # 5", "2 * pi * r", "sqrt(16)", "F = m * a", "E = m * c^2"]
        
        self.update_status("Batch processing formulas...")
        
        for formula in formulas:
            try:
                latex_reps = self.latex_integration.encode_formula_to_latex(formula)
                # Store results (implementation can be expanded)
            except Exception as e:
                self.update_status(f"Error processing {formula}: {str(e)}")
                
        self.update_status("Batch processing completed")
        
    def save_session(self):
        """Save current session"""
        if not self.current_session:
            self.update_status("No session to save")
            return
            
        # Get notes
        self.current_session.user_notes = self.notes_text.get('1.0', 'end').strip()
        
        # Add to history
        self.session_history.append(self.current_session)
        
        # Update counter
        self.session_counter.config(text=f"Sessions: {len(self.session_history)}")
        
        # Add to history listbox
        self.history_listbox.insert('end', f"{self.current_session.session_id}: {self.current_session.formula_input}")
        
        self.update_status("Session saved successfully")
        
    def load_session(self):
        """Load a previous session"""
        selection = self.history_listbox.curselection()
        if not selection:
            self.update_status("No session selected")
            return
            
        session = self.session_history[selection[0]]
        
        # Load formula
        self.formula_entry.delete(0, 'end')
        self.formula_entry.insert(0, session.formula_input)
        
        # Load notes
        self.notes_text.delete('1.0', 'end')
        self.notes_text.insert('1.0', session.user_notes)
        
        # Re-encode
        self.encode_formula()
        
        # Load validation if exists
        if session.validation_results:
            self.display_validation_results(session.validation_results)
            
        self.update_status(f"Loaded session: {session.session_id}")
        
    def export_latex(self):
        """Export LaTeX representations"""
        if not self.current_session:
            self.update_status("No session to export")
            return
            
        # Create export data
        export_data = {
            'session_id': self.current_session.session_id,
            'formula': self.current_session.formula_input,
            'timestamp': self.current_session.timestamp,
            'latex_representations': [asdict(rep) for rep in self.current_session.latex_representations]
        }
        
        # Save to file (simplified - in real implementation would use file dialog)
        try:
            filename = f"latex_export_{self.current_session.session_id}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.update_status(f"Exported to {filename}")
        except Exception as e:
            self.update_status(f"Export error: {str(e)}")
            
    def import_latex(self):
        """Import LaTeX from file"""
        self.update_status("Import functionality - to be implemented")
        
    def clear_all(self):
        """Clear all current data"""
        self.formula_entry.delete(0, 'end')
        self.validation_text.delete('1.0', 'end')
        self.notes_text.delete('1.0', 'end')
        
        # Clear LaTeX tabs
        for tab in self.latex_tabs.values():
            tab.destroy()
        self.latex_tabs.clear()
        self.latex_displays.clear()
        
        self.current_session = None
        self.update_status("All cleared")
        
    def show_help(self):
        """Show help information"""
        help_text = """
🔬 LaTeX Reverse Engineering Workshop Help

📝 INPUT:
• Enter any mathematical formula
• Use standard math notation
• Try Empirinometry: 10 # 5
• Use Greek letters: pi, alpha, beta

🎨 LATEX REPRESENTATIONS:
• Standard: Mathematical notation
• Empirinometry: Custom operators
• SymPy: Symbolic representation  
• Stepwise: Calculation steps
• Matrix: Linear algebra form

✅ VALIDATION:
• Checks consistency across formats
• Evaluates mathematical correctness
• Provides confidence scores
• Identifies inconsistencies

🔄 REVERSE ENGINEERING:
• LaTeX → Python conversion
• Automatic evaluation
• Confidence assessment
• Error detection

💾 SESSIONS:
• Save your work
• Load previous sessions
• Track history
• Export results
        """
        
        self.validation_text.delete('1.0', 'end')
        self.validation_text.insert('1.0', help_text)
        self.update_status("Help information displayed")
        
    def update_status(self, message: str):
        """Update status bar"""
        self.status_label.config(text=message)
        
    def run(self):
        """Run the workshop"""
        self.update_status("LaTeX Reverse Engineering Workshop ready!")
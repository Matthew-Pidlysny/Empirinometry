"""
Induction Ω - Mathematical Induction Measurement System
Main application launcher integrating all components
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
from pathlib import Path
import tkinter.ttk as ttk

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from encyclopedia_gui import EncyclopediaGUI
    from latex_engine import latex_engine, validate_expression
    from encyclopedia_structure import encyclopedia
    # Workshop GUI may not be available in this environment
    InductionWorkshopGUI = None
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are in the same directory")
    InductionWorkshopGUI = None
    EncyclopediaGUI = None

class InductionOmegaMain:
    """
    Main application controller for Induction Ω
    Integrates workshop, encyclopedia, and LaTeX components
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Induction Ω - Mathematical Induction System")
        self.root.geometry("1200x700")
        
        # Application state
        self.current_mode = None
        self.workshop_app = None
        self.encyclopedia_app = None
        
        # Setup main interface
        self._setup_main_ui()
        self._center_window()
        
    def _setup_main_ui(self):
        """Setup the main user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 30))
        
        title_label = ttk.Label(title_frame, text="Induction Ω", 
                               font=('Arial', 24, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, 
                                  text="Mathematical Induction Measurement & Encyclopedia System", 
                                  font=('Arial', 12))
        subtitle_label.pack()
        
        # Version info
        version_label = ttk.Label(title_frame, text="Version 2.0 - 500+ Pages Encyclopedia Edition", 
                                 font=('Arial', 10), foreground='gray')
        version_label.pack()
        
        # Mode selection buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create mode buttons with descriptions
        self._create_mode_button(button_frame, "🛠️ Workshop", 
                                "Interactive Learning & Practice\nTest your induction skills with exercises and examples",
                                self.launch_workshop, 0)
        
        self._create_mode_button(button_frame, "📚 Encyclopedia", 
                                "Comprehensive Reference Guide\n500+ pages covering all forms of induction",
                                self.launch_encyclopedia, 1)
        
        self._create_mode_button(button_frame, "🧮 LaTeX Engine", 
                                "Mathematical Formula Rendering\nValidate and render LaTeX expressions",
                                self.launch_latex_validator, 2)
        
        self._create_mode_button(button_frame, "📊 Statistics", 
                                "View System Statistics\nSee encyclopedia usage and content metrics",
                                self.show_statistics, 3)
        
        # Bottom frame
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(30, 0))
        
        # Info label
        info_label = ttk.Label(bottom_frame, 
                              text="Choose a mode to begin exploring mathematical induction",
                              font=('Arial', 10))
        info_label.pack(side=tk.LEFT)
        
        # Exit button
        exit_btn = ttk.Button(bottom_frame, text="Exit", command=self.on_closing)
        exit_btn.pack(side=tk.RIGHT)
        
        # Configure grid weights
        for i in range(2):
            button_frame.grid_columnconfigure(i, weight=1)
        for i in range(2):
            button_frame.grid_rowconfigure(i, weight=1)
    
    def _create_mode_button(self, parent, title, description, command, position):
        """Create a mode selection button with description"""
        row = position // 2
        col = position % 2
        
        # Button container
        btn_container = ttk.Frame(parent)
        btn_container.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        
        # Main button
        btn = ttk.Button(btn_container, text=title, command=command, 
                        style='Large.TButton')
        btn.pack(fill=tk.X, pady=(0, 5))
        
        # Description label
        desc_label = ttk.Label(btn_container, text=description, 
                              font=('Arial', 9), justify='center', 
                              foreground='gray')
        desc_label.pack()
        
        # Configure large button style if not already configured
        try:
            style = ttk.Style()
            style.configure('Large.TButton', font=('Arial', 12))
        except:
            pass  # Style might already be configured
    
    def _center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def launch_workshop(self):
        """Launch the workshop application"""
        if InductionWorkshopGUI is None:
            messagebox.showinfo("Workshop", "Workshop module not available in this environment.\n\nPlease use the Encyclopedia mode to access comprehensive induction content.")
            return
            
        try:
            # Hide main window
            self.root.withdraw()
            
            # Create workshop window
            workshop_root = tk.Toplevel(self.root)
            workshop_root.title("Induction Ω - Workshop")
            workshop_root.geometry("1200x800")
            
            # Create workshop application
            self.workshop_app = InductionWorkshopGUI(workshop_root)
            
            # Handle workshop window closing
            workshop_root.protocol("WM_DELETE_WINDOW", lambda: self.on_child_closing(workshop_root))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch workshop: {str(e)}")
            self.root.deiconify()
    
    def launch_encyclopedia(self):
        """Launch the encyclopedia application"""
        if EncyclopediaGUI is None:
            messagebox.showinfo("Encyclopedia", "Encyclopedia module not available in this environment.")
            return
            
        try:
            # Hide main window
            self.root.withdraw()
            
            # Create encyclopedia window
            encyclopedia_root = tk.Toplevel(self.root)
            encyclopedia_root.title("Induction Ω - Encyclopedia")
            encyclopedia_root.geometry("1400x900")
            
            # Create encyclopedia application
            self.encyclopedia_app = EncyclopediaGUI(encyclopedia_root)
            
            # Handle encyclopedia window closing
            encyclopedia_root.protocol("WM_DELETE_WINDOW", lambda: self.on_child_closing(encyclopedia_root))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch encyclopedia: {str(e)}")
            self.root.deiconify()
    
    def launch_latex_validator(self):
        """Launch LaTeX validator tool"""
        # Create LaTeX validator dialog
        validator_window = tk.Toplevel(self.root)
        validator_window.title("LaTeX Expression Validator")
        validator_window.geometry("600x400")
        
        # LaTeX input
        ttk.Label(validator_window, text="Enter LaTeX Expression:").pack(pady=10)
        
        latex_var = tk.StringVar()
        latex_entry = ttk.Entry(validator_window, textvariable=latex_var, width=50)
        latex_entry.pack(pady=5)
        latex_entry.focus()
        
        # Result display
        result_frame = ttk.Frame(validator_window)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        result_text = tk.Text(result_frame, height=10, width=60)
        result_text.pack(fill=tk.BOTH, expand=True)
        
        # Validation function
        def validate_latex():
            expression = latex_var.get().strip()
            if expression:
                validation = validate_expression(expression)
                
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Expression: {expression}\n\n")
                result_text.insert(tk.END, f"Valid: {'✓' if validation['valid'] else '✗'}\n\n")
                
                if validation['errors']:
                    result_text.insert(tk.END, "Errors:\n")
                    for error in validation['errors']:
                        result_text.insert(tk.END, f"  • {error}\n")
                
                if validation['warnings']:
                    result_text.insert(tk.END, "\nWarnings:\n")
                    for warning in validation['warnings']:
                        result_text.insert(tk.END, f"  • {warning}\n")
                
                if validation['suggestions']:
                    result_text.insert(tk.END, "\nSuggestions:\n")
                    for suggestion in validation['suggestions']:
                        result_text.insert(tk.END, f"  • {suggestion}\n")
        
        # Render preview function
        def render_preview():
            expression = latex_var.get().strip()
            if expression:
                rendered = latex_engine.render_latex_to_html(expression, display_mode=True)
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Rendered HTML:\n{rendered}")
        
        # Buttons
        button_frame = ttk.Frame(validator_window)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Validate", command=validate_latex).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Render Preview", command=render_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=lambda: [latex_var.set(""), result_text.delete(1.0, tk.END)]).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=validator_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to validate
        latex_entry.bind('<Return>', lambda e: validate_latex())
    
    def show_statistics(self):
        """Show system statistics"""
        try:
            stats = encyclopedia.get_statistics()
            
            stats_text = f"""
Induction Ω System Statistics

=== Encyclopedia Content ===
Total Entries: {stats['total_entries']}
Total Examples: {stats['total_examples']}
Total Exercises: {stats['total_exercises']}
Total Applications: {stats['total_applications']}

=== By Difficulty Level ===
"""
            
            for level, count in stats['by_difficulty'].items():
                stats_text += f"  {level.replace('_', ' ').title()}: {count}\n"
            
            stats_text += "\n=== By Induction Type ===\n"
            for ind_type, count in stats['by_type'].items():
                stats_text += f"  {ind_type.replace('_', ' ').title()}: {count}\n"
            
            stats_text += f"""
=== System Information ===
LaTeX Engine: {'✓' if latex_engine else '✗'}
Content Generator: {'✓' if hasattr(self, '_create_mode_button') else '✗'}
GUI Components: {'✓' if hasattr(self, 'workshop_app') else '✗'}

=== Features ===
• 500+ Pages of Encyclopedia Content
• LaTeX Mathematical Rendering
• Interactive Workshop Exercises  
• Multiple Induction Types Covered
• Comprehensive Search & Navigation
• Educational Content from Elementary to Research Level

Version: 2.0.0
Build: Encyclopedia Integration Edition
            """
            
            messagebox.showinfo("Induction Ω Statistics", stats_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load statistics: {str(e)}")
    
    def on_child_closing(self, child_window):
        """Handle child window closing"""
        child_window.destroy()
        self.root.deiconify()
        self.current_mode = None
    
    def on_closing(self):
        """Handle main window closing"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit Induction Ω?"):
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """Run the main application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        # Import ttk
        import tkinter.ttk as ttk
    except ImportError:
        print("Error: tkinter.ttk not available")
        sys.exit(1)
    
    # Create and run application
    app = InductionOmegaMain()
    app.run()

if __name__ == "__main__":
    main()
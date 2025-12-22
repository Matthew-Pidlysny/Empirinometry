"""
Diffusion Navigator - Main Launcher
Entry point for the Diffusion Navigator application
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import traceback

# Add the diffusion_navigator directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'numpy', 'matplotlib', 'scipy', 'sklearn', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    missing = check_dependencies()
    if missing:
        print("Missing dependencies:", missing)
        print("Please install them using: pip install", " ".join(missing))
        return False
    return True

def launch_gui():
    """Launch the main GUI application"""
    try:
        from diffusion_gui import DiffusionNavigatorGUI
        
        # Create and run the application
        app = DiffusionNavigatorGUI()
        app.run()
        
    except Exception as e:
        error_msg = f"Failed to launch GUI: {str(e)}\n\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        
        # Show error in dialog if possible
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            messagebox.showerror("Launch Error", error_msg)
        except:
            pass
        
        print(error_msg)
        return False
    
    return True

def show_startup_info():
    """Show startup information"""
    print("=" * 60)
    print("DIFFUSION NAVIGATOR - Interactive Learning Platform")
    print("=" * 60)
    print()
    print("Features:")
    print("• Caelum-based diffusion modeling")
    print("• Multi-material analysis with comprehensive database")
    print("• Interactive 3D diffusion sphere visualization")
    print("• Advanced pattern clustering and analysis")
    print("• ROOT framework integration for physics calculations")
    print("• Step-by-step calculation visualization")
    print("• Student-friendly interface with workshops")
    print("• Data export capabilities (JSON, TXT, CSV)")
    print("• LaTeX-encoded mathematical formulas")
    print()
    print("Starting application...")
    print("-" * 60)

def main():
    """Main entry point"""
    show_startup_info()
    
    # Check dependencies
    if not install_dependencies():
        input("Press Enter to exit...")
        return 1
    
    # Launch GUI
    success = launch_gui()
    
    if not success:
        input("Press Enter to exit...")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
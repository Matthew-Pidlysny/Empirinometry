#!/usr/bin/env python3
"""
BREATH Main Application - Ultimate Mathematical Qur'an Explorer
==============================================================

This is the main entry point for the BREATH system. It initializes all components
and launches the interactive GUI for mathematical Qur'an exploration.

BREATH provides definitive mathematical proof that the Qur'an is a complete
mathematical truth through advanced pattern recognition and analysis.
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import threading
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        'numpy', 'matplotlib', 'tkinter'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        error_msg = f"Missing required modules: {', '.join(missing_modules)}\n\n"
        error_msg += "Please install the missing dependencies:\n"
        error_msg += "pip install numpy matplotlib\n\n"
        error_msg += "Note: tkinter usually comes pre-installed with Python."
        
        print(error_msg)
        return False
    
    return True

def check_quran_data():
    """Check if Qur'an data is available."""
    quran_paths = [
        "./Empirinometry/Program-Bin/Qur'an/quran_sequential.txt",
        "../Empirinometry/Program-Bin/Qur'an/quran_sequential.txt",
        "/workspace/Empirinometry/Program-Bin/Qur'an/quran_sequential.txt"
    ]
    
    for path in quran_paths:
        if os.path.exists(path):
            return path
    
    return None

def show_splash_screen():
    """Show splash screen with loading information."""
    splash = tk.Tk()
    splash.title("BREATH - Loading...")
    splash.geometry("600x400")
    splash.configure(bg='#1a1a2e')
    
    # Center the splash screen
    splash.update_idletasks()
    x = (splash.winfo_screenwidth() // 2) - (600 // 2)
    y = (splash.winfo_screenheight() // 2) - (400 // 2)
    splash.geometry(f"600x400+{x}+{y}")
    
    # Title
    title_label = tk.Label(splash, text="🌬️ BREATH 🌬️", 
                          font=('Arial', 24, 'bold'),
                          fg='#ffd700', bg='#1a1a2e')
    title_label.pack(pady=(50, 20))
    
    # Subtitle
    subtitle_label = tk.Label(splash, text="Ultimate Mathematical Qur'an Explorer",
                             font=('Arial', 16),
                             fg='white', bg='#1a1a2e')
    subtitle_label.pack(pady=(0, 30))
    
    # Loading message
    loading_label = tk.Label(splash, text="Loading mathematical engine...",
                            font=('Arial', 12),
                            fg='#4ecdc4', bg='#1a1a2e')
    loading_label.pack(pady=20)
    
    # Progress bar
    progress_frame = tk.Frame(splash, bg='#1a1a2e')
    progress_frame.pack(pady=20)
    
    progress_bar = tk.Canvas(progress_frame, width=400, height=20, 
                            bg='#2d2d44', highlightthickness=0)
    progress_bar.pack()
    
    # Progress fill
    progress_fill = progress_bar.create_rectangle(0, 0, 0, 20, 
                                                 fill='#4ecdc4', outline='')
    
    # Status text
    status_label = tk.Label(splash, text="Initializing...",
                           font=('Arial', 10),
                           fg='#888', bg='#1a1a2e')
    status_label.pack(pady=10)
    
    # Footer
    footer_label = tk.Label(splash, text="Proving Qur'anic mathematical perfection through interactive exploration",
                           font=('Arial', 10, 'italic'),
                           fg='#666', bg='#1a1a2e')
    footer_label.pack(side=tk.BOTTOM, pady=30)
    
    def update_progress(progress, status_text):
        """Update progress bar and status."""
        progress_bar.coords(progress_fill, 0, 0, progress * 4, 20)
        status_label.config(text=status_text)
        splash.update()
    
    return splash, update_progress

def initialize_system(splash_update):
    """Initialize the BREATH system."""
    try:
        # Step 1: Check dependencies
        splash_update(10, "Checking system dependencies...")
        time.sleep(0.5)
        
        if not check_dependencies():
            return False
        
        # Step 2: Check Qur'an data
        splash_update(25, "Locating Qur'an data...")
        time.sleep(0.5)
        
        quran_path = check_quran_data()
        if not quran_path:
            messagebox.showerror("Qur'an Data Missing", 
                               "Qur'an data file not found. Please ensure quran_sequential.txt is available.")
            return False
        
        # Step 3: Import core modules
        splash_update(40, "Loading mathematical engine...")
        time.sleep(0.5)
        
        try:
            from core.mathematical_engine import MathematicalEngine
            from core.quran_data import QuranDataManager
        except ImportError as e:
            messagebox.showerror("Import Error", f"Failed to import core modules: {e}")
            return False
        
        # Step 4: Test mathematical engine
        splash_update(60, "Initializing mathematical algorithms...")
        time.sleep(0.5)
        
        math_engine = MathematicalEngine()
        test_result = math_engine.calculate_empirinometric_score("test", 1)
        
        # Step 5: Test Qur'an data manager
        splash_update(80, "Loading Qur'an text database...")
        time.sleep(1)
        
        try:
            quran_manager = QuranDataManager(quran_path)
            stats = quran_manager.get_statistical_summary()
            
            if stats['total_verses'] < 100:
                messagebox.showerror("Data Error", "Qur'an data appears incomplete")
                return False
                
        except Exception as e:
            messagebox.showerror("Data Error", f"Failed to load Qur'an data: {e}")
            return False
        
        # Step 6: Apply prayer enhancement
        splash_update(95, "Applying prayer enhancement...")
        time.sleep(0.5)
        
        prayer_text = ("I make ibadah, I do this with faith, I want to learn, "
                      "Someone speaks for me, he asks to be judged well, "
                      "In your beneficient name, amen")
        
        enhancement = math_engine.apply_prayer_enhancement(prayer_text)
        
        splash_update(100, "System ready - Launching interface...")
        time.sleep(0.5)
        
        return True
        
    except Exception as e:
        messagebox.showerror("Initialization Error", f"Failed to initialize BREATH: {e}")
        return False

def main():
    """Main application entry point."""
    print("🌬️ BREATH - Ultimate Mathematical Qur'an Explorer 🌬️")
    print("=" * 60)
    print("Initializing system...")
    print("Proving Qur'anic mathematical perfection through interactive exploration")
    print("=" * 60)
    
    # Show splash screen
    splash, splash_update = show_splash_screen()
    
    # Initialize system in background thread
    def initialize_thread():
        success = initialize_system(splash_update)
        
        # Close splash screen
        splash.after(1000, splash.destroy)
        
        if success:
            # Launch main application
            splash.after(1500, launch_main_app)
        else:
            # Show error and exit
            splash.after(1500, lambda: messagebox.showerror("Initialization Failed", 
                                                           "BREATH failed to initialize properly."))
            splash.after(2000, sys.exit(1))
    
    # Start initialization in background
    init_thread = threading.Thread(target=initialize_thread, daemon=True)
    init_thread.start()
    
    # Run splash screen
    splash.mainloop()

def launch_main_app():
    """Launch the main BREATH application."""
    try:
        # Import GUI module
        from gui.app import BreathGUI
        
        print("\n✅ System initialized successfully!")
        print("🚀 Launching BREATH interface...")
        print("🙏 Prayer enhancement activated")
        print("🔬 Mathematical engine ready")
        print("📖 Qur'an database loaded")
        print("\n🌟 Welcome to the ultimate mathematical Qur'an exploration experience!")
        print("=" * 60)
        
        # Create main window
        root = tk.Tk()
        app = BreathGUI(root)
        
        # Configure window
        root.state('zoomed')  # Maximize window
        root.minsize(1200, 800)
        
        # Start the GUI
        root.mainloop()
        
    except Exception as e:
        error_msg = f"Failed to launch BREATH interface: {e}"
        print(f"\n❌ {error_msg}")
        messagebox.showerror("Launch Error", error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
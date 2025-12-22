"""
Sleek GUI Framework - Non-Classical Windowing System for Gas Tech Suite
Custom rendering engine with advanced texturing and animations
"""

import tkinter as tk
from tkinter import Canvas, Frame
import math
import time
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class WindowStyle(Enum):
    FLUID_METAL = "fluid_metal"
    AQUATIC = "aquatic"
    PLASMA = "plasma"
    CRYSTALLINE = "crystalline"
    ORGANIC = "organic"

@dataclass
class ThemeConfig:
    """Configuration for visual themes"""
    primary_color: str
    secondary_color: str
    accent_color: str
    background_gradient: Tuple[str, str]
    border_style: str  # "fluid", "sharp", "organic"
    animation_speed: float
    glass_effect: bool

class SleekButton:
    """Custom button with non-classical styling"""
    
    def __init__(self, canvas: Canvas, x: int, y: int, width: int, height: int, 
                 text: str, style: WindowStyle = WindowStyle.FLUID_METAL, 
                 command: Optional[Callable] = None):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.style = style
        self.command = command
        self.animation_phase = 0
        self.is_hovered = False
        self.is_pressed = False
        
        # Style configurations
        self.themes = {
            WindowStyle.FLUID_METAL: ThemeConfig(
                primary_color="#2C3E50",
                secondary_color="#34495E",
                accent_color="#3498DB",
                background_gradient=("#1A252F", "#2C3E50"),
                border_style="fluid",
                animation_speed=0.05,
                glass_effect=True
            ),
            WindowStyle.AQUATIC: ThemeConfig(
                primary_color="#006994",
                secondary_color="#0088CC",
                accent_color="#00BFFF",
                background_gradient=("#004466", "#006994"),
                border_style="fluid",
                animation_speed=0.08,
                glass_effect=True
            ),
            WindowStyle.PLASMA: ThemeConfig(
                primary_color="#8B008B",
                secondary_color="#FF1493",
                accent_color="#FF69B4",
                background_gradient=("#4B0082", "#8B008B"),
                border_style="plasma",
                animation_speed=0.03,
                glass_effect=False
            ),
            WindowStyle.CRYSTALLINE: ThemeConfig(
                primary_color="#4A90E2",
                secondary_color="#7CB9E8",
                accent_color="#87CEEB",
                background_gradient=("#2E5D8B", "#4A90E2"),
                border_style="sharp",
                animation_speed=0.02,
                glass_effect=True
            ),
            WindowStyle.ORGANIC: ThemeConfig(
                primary_color="#2E7D32",
                secondary_color="#4CAF50",
                accent_color="#8BC34A",
                background_gradient=("#1B5E20", "#2E7D32"),
                border_style="organic",
                animation_speed=0.06,
                glass_effect=False
            )
        }
        
        self.theme = self.themes[style]
        self.elements = []
        self.create()
        
    def create(self):
        """Create the button with custom styling"""
        # Animated background
        self.bg_rect = self.canvas.create_rectangle(
            self.x, self.y, self.x + self.width, self.y + self.height,
            fill=self.theme.primary_color, outline="", tags="button_bg"
        )
        
        # Fluid border effect
        if self.theme.border_style == "fluid":
            self.create_fluid_border()
        elif self.theme.border_style == "plasma":
            self.create_plasma_border()
        elif self.theme.border_style == "sharp":
            self.create_sharp_border()
        elif self.theme.border_style == "organic":
            self.create_organic_border()
        
        # Text
        self.text_element = self.canvas.create_text(
            self.x + self.width // 2, self.y + self.height // 2,
            text=self.text, fill="white", font=("Arial", 10, "bold"),
            tags="button_text"
        )
        
        self.elements = [self.bg_rect] + self.border_elements + [self.text_element]
        
        # Bind events
        self.canvas.tag_bind("button", "<Enter>", self.on_enter)
        self.canvas.tag_bind("button", "<Leave>", self.on_leave)
        self.canvas.tag_bind("button", "<Button-1>", self.on_click)
        
    def create_fluid_border(self):
        """Create fluid animated border"""
        self.border_elements = []
        points = []
        
        # Generate wavy border points
        for i in range(20):
            angle = (i / 20) * 2 * math.pi
            if i % 4 < 2:  # Top and bottom
                x = self.x + (i / 20) * self.width
                if i < 10:  # Top
                    y = self.y + math.sin(angle * 2) * 3
                else:  # Bottom
                    y = self.y + self.height + math.sin(angle * 2) * 3
            else:  # Left and right
                if i < 15:  # Right
                    x = self.x + self.width + math.sin(angle * 2) * 3
                    y = self.y + ((i - 10) / 10) * self.height
                else:  # Left
                    x = self.x + math.sin(angle * 2) * 3
                    y = self.y + ((i - 15) / 5) * self.height
            
            points.extend([x, y])
        
        self.border = self.canvas.create_polygon(
            points, fill="", outline=self.theme.accent_color,
            width=2, smooth=True, tags="button"
        )
        self.border_elements = [self.border]
    
    def create_plasma_border(self):
        """Create plasma energy border"""
        self.border_elements = []
        
        # Multiple layers for plasma effect
        for layer in range(3):
            offset = layer * 2
            plasma = self.canvas.create_rectangle(
                self.x - offset, self.y - offset,
                self.x + self.width + offset, self.y + self.height + offset,
                fill="", outline=self.theme.accent_color,
                width=3 - layer, tags="button"
            )
            self.border_elements.append(plasma)
    
    def create_sharp_border(self):
        """Create crystalline sharp border"""
        self.border_elements = []
        
        # Clean geometric border
        border = self.canvas.create_rectangle(
            self.x - 2, self.y - 2,
            self.x + self.width + 2, self.y + self.height + 2,
            fill="", outline=self.theme.accent_color,
            width=2, tags="button"
        )
        self.border_elements = [border]
    
    def create_organic_border(self):
        """Create organic curved border"""
        self.border_elements = []
        
        # Rounded corners using multiple small lines
        corner_radius = 10
        points = []
        
        # Top edge with rounded corners
        points.extend([self.x + corner_radius, self.y])
        points.extend([self.x + self.width - corner_radius, self.y])
        
        # Right edge
        points.extend([self.x + self.width, self.y + corner_radius])
        points.extend([self.x + self.width, self.y + self.height - corner_radius])
        
        # Bottom edge
        points.extend([self.x + self.width - corner_radius, self.y + self.height])
        points.extend([self.x + corner_radius, self.y + self.height])
        
        # Left edge
        points.extend([self.x, self.y + self.height - corner_radius])
        points.extend([self.x, self.y + corner_radius])
        
        border = self.canvas.create_polygon(
            points, fill="", outline=self.theme.accent_color,
            width=2, smooth=True, tags="button"
        )
        self.border_elements = [border]
    
    def on_enter(self, event):
        """Handle mouse enter"""
        self.is_hovered = True
        self.animate_hover()
    
    def on_leave(self, event):
        """Handle mouse leave"""
        self.is_hovered = False
        self.animate_normal()
    
    def on_click(self, event):
        """Handle button click"""
        if self.command:
            self.command()
    
    def animate_hover(self):
        """Animate hover state"""
        if self.is_hovered:
            self.animation_phase += self.theme.animation_speed
            
            # Pulsing effect
            scale = 1 + math.sin(self.animation_phase) * 0.05
            new_width = self.width * scale
            new_height = self.height * scale
            new_x = self.x + (self.width - new_width) / 2
            new_y = self.y + (self.height - new_height) / 2
            
            self.canvas.coords(
                self.bg_rect,
                new_x, new_y, new_x + new_width, new_y + new_height
            )
            
            # Color shift
            if self.theme.glass_effect:
                self.canvas.itemconfig(
                    self.bg_rect,
                    fill=self.theme.secondary_color
                )
            
            self.canvas.after(50, self.animate_hover)
    
    def animate_normal(self):
        """Return to normal state"""
        self.canvas.coords(
            self.bg_rect,
            self.x, self.y, self.x + self.width, self.y + self.height
        )
        self.canvas.itemconfig(
            self.bg_rect,
            fill=self.theme.primary_color
        )

class SleekWindow:
    """Main window with non-classical styling"""
    
    def __init__(self, title: str, width: int = 1200, height: int = 800, 
                 style: WindowStyle = WindowStyle.FLUID_METAL):
        self.title = title
        self.width = width
        self.height = height
        self.style = style
        self.buttons = []
        self.animations = []
        
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg='black')
        
        # Remove default window decorations
        self.root.overrideredirect(True)
        
        # Create custom canvas
        self.canvas = Canvas(
            self.root, width=width, height=height,
            bg='black', highlightthickness=0
        )
        self.canvas.pack()
        
        self.setup_window_decorations()
        self.setup_background()
        
        # Make window draggable
        self.setup_dragging()
        
    def setup_window_decorations(self):
        """Create custom window decorations"""
        # Title bar
        self.title_bar = self.canvas.create_rectangle(
            0, 0, self.width, 40,
            fill="#1A1A1A", outline=""
        )
        
        # Title text
        self.title_text = self.canvas.create_text(
            20, 20, text=self.title,
            fill="white", font=("Arial", 12, "bold"),
            anchor="w"
        )
        
        # Close button
        self.close_btn = SleekButton(
            self.canvas, self.width - 60, 10, 50, 25,
            "✕", WindowStyle.PLASMA, self.root.destroy
        )
        
        # Minimize button
        self.min_btn = SleekButton(
            self.canvas, self.width - 120, 10, 50, 25,
            "−", WindowStyle.FLUID_METAL, self.minimize_window
        )
    
    def setup_background(self):
        """Create animated background"""
        self.background_elements = []
        
        if self.style == WindowStyle.FLUID_METAL:
            self.create_fluid_metal_background()
        elif self.style == WindowStyle.AQUATIC:
            self.create_aquatic_background()
        elif self.style == WindowStyle.PLASMA:
            self.create_plasma_background()
        elif self.style == WindowStyle.CRYSTALLINE:
            self.create_crystalline_background()
        elif self.style == WindowStyle.ORGANIC:
            self.create_organic_background()
    
    def create_fluid_metal_background(self):
        """Create fluid metal background effect"""
        # Metallic gradient layers
        for i in range(10):
            y = 40 + i * (self.height - 40) / 10
            color_intensity = 20 + i * 3
            color = f"#{color_intensity:02x}{color_intensity:02x}{color_intensity+5:02x}"
            
            layer = self.canvas.create_rectangle(
                0, y, self.width, y + (self.height - 40) / 10,
                fill=color, outline="", tags="background"
            )
            self.background_elements.append(layer)
    
    def create_aquatic_background(self):
        """Create aquatic wave background"""
        import random
        
        # Wave layers
        for wave in range(5):
            points = []
            for x in range(0, self.width + 20, 20):
                y = 100 + wave * 100 + math.sin(x / 50 + wave) * 30
                points.extend([x, y])
            
            points.extend([self.width, self.height, 0, self.height])
            
            color_intensity = 0 + wave * 20
            color = f"#{0:02x}{color_intensity:02x}{color_intensity+40:02x}"
            
            wave_layer = self.canvas.create_polygon(
                points, fill=color, outline="", smooth=True, tags="background"
            )
            self.background_elements.append(wave_layer)
    
    def create_plasma_background(self):
        """Create plasma energy background"""
        import random
        
        # Energy particles
        for particle in range(50):
            x = random.randint(0, self.width)
            y = random.randint(40, self.height)
            size = random.randint(2, 8)
            
            particle = self.canvas.create_oval(
                x - size, y - size, x + size, y + size,
                fill=self.get_plasma_color(), outline="", tags="background"
            )
            self.background_elements.append(particle)
    
    def create_crystalline_background(self):
        """Create crystalline geometric background"""
        # Geometric patterns
        for i in range(0, self.width, 100):
            for j in range(40, self.height, 100):
                # Create crystal shapes
                points = [
                    i + 50, j,
                    i + 70, j + 30,
                    i + 50, j + 60,
                    i + 30, j + 30
                ]
                
                crystal = self.canvas.create_polygon(
                    points, fill="#2E5D8B", outline="#4A90E2",
                    width=1, tags="background"
                )
                self.background_elements.append(crystal)
    
    def create_organic_background(self):
        """Create organic flowing background"""
        # Organic curves
        for curve in range(8):
            points = []
            start_y = 100 + curve * 80
            
            for x in range(0, self.width + 50, 50):
                y = start_y + math.sin(x / 100 + curve) * 40
                points.extend([x, y])
            
            color_intensity = 20 + curve * 10
            color = f"#{20:02x}{color_intensity:02x}{30:02x}"
            
            curve_line = self.canvas.create_line(
                points, fill=color, width=3, smooth=True, tags="background"
            )
            self.background_elements.append(curve_line)
    
    def get_plasma_color(self):
        """Get random plasma color"""
        import random
        colors = ["#FF1493", "#FF69B4", "#FFB6C1", "#FFC0CB", "#8B008B"]
        return random.choice(colors)
    
    def setup_dragging(self):
        """Setup window dragging"""
        self.drag_data = {"x": 0, "y": 0}
        
        def start_drag(event):
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
        
        def drag(event):
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            x = self.root.winfo_x() + dx
            y = self.root.winfo_y() + dy
            self.root.geometry(f"+{x}+{y}")
        
        self.canvas.bind("<Button-1>", start_drag)
        self.canvas.bind("<B1-Motion>", drag)
    
    def minimize_window(self):
        """Minimize window"""
        self.root.iconify()
    
    def add_button(self, x: int, y: int, width: int, height: int, 
                   text: str, style: WindowStyle = None, 
                   command: Optional[Callable] = None) -> SleekButton:
        """Add a sleek button to the window"""
        if style is None:
            style = self.style
        
        button = SleekButton(self.canvas, x, y, width, height, text, style, command)
        self.buttons.append(button)
        return button
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()

class GasTechSleekGUI:
    """Main Gas Tech GUI with sleek styling"""
    
    def __init__(self):
        self.window = SleekWindow(
            "Gas Tech Suite - Industrial Gas Fitting Software",
            1400, 900, WindowStyle.FLUID_METAL
        )
        self.setup_main_interface()
    
    def setup_main_interface(self):
        """Setup main interface elements"""
        # Version selection buttons
        versions = [
            ("Consumer", 100, 100, 200, 60, self.open_consumer),
            ("Gas Tech", 350, 100, 200, 60, self.open_gas_tech),
            ("Office", 600, 100, 200, 60, self.open_office),
            ("Industrial", 850, 100, 200, 60, self.open_industrial),
            ("Scientist", 1100, 100, 200, 60, self.open_scientist),
            ("Mechanical", 100, 200, 200, 60, self.open_mechanical)
        ]
        
        for text, x, y, w, h, command in versions:
            style = WindowStyle.FLUID_METAL
            if text == "Gas Tech":
                style = WindowStyle.AQUATIC
            elif text == "Industrial":
                style = WindowStyle.PLASMA
            elif text == "Scientist":
                style = WindowStyle.CRYSTALLINE
            elif text == "Mechanical":
                style = WindowStyle.ORGANIC
            
            self.window.add_button(x, y, w, h, text, style, command)
        
        # Main workspace area
        self.workspace_text = self.window.canvas.create_text(
            700, 400, text="Select a version to begin",
            fill="white", font=("Arial", 24), anchor="center"
        )
    
    def open_consumer(self):
        """Open consumer version"""
        self.window.canvas.itemconfig(
            self.workspace_text,
            text="Consumer Version: Basic gas calculations and safety tools"
        )
    
    def open_gas_tech(self):
        """Open gas technician version"""
        self.window.canvas.itemconfig(
            self.workspace_text,
            text="Gas Tech Version: Professional technician tools and calculations"
        )
    
    def open_office(self):
        """Open office version"""
        self.window.canvas.itemconfig(
            self.workspace_text,
            text="Office Version: Administrative and management tools"
        )
    
    def open_industrial(self):
        """Open industrial version"""
        self.window.canvas.itemconfig(
            self.workspace_text,
            text="Industrial Version: Large-scale industrial gas systems"
        )
    
    def open_scientist(self):
        """Open scientist version"""
        self.window.canvas.itemconfig(
            self.workspace_text,
            text="Scientist Version: Research and analysis tools"
        )
    
    def open_mechanical(self):
        """Open mechanical version"""
        self.window.canvas.itemconfig(
            self.workspace_text,
            text="Mechanical Version: Advanced mechanical engineering tools"
        )
    
    def run(self):
        """Run the application"""
        self.window.run()

if __name__ == "__main__":
    app = GasTechSleekGUI()
    app.run()
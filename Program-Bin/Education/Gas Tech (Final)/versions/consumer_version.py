"""
Consumer Version - Entry-level gas fitting software for homeowners
Focus on safety, basic calculations, and user-friendly interface
"""

from typing import Dict, List, Optional
import math
from datetime import datetime

# Import our core engine (will be adjusted for relative imports)
from ..core.gas_physics_engine import GasPhysicsEngine, FuelType
from ..gui.sleek_gui_framework import SleekWindow, SleekButton, WindowStyle

class ConsumerVersion:
    """Consumer-focused gas fitting calculator and safety tool"""
    
    def __init__(self):
        self.engine = GasPhysicsEngine()
        self.window = None
        self.setup_gui()
    
    def setup_gui(self):
        """Setup consumer-friendly GUI"""
        self.window = SleekWindow(
            "Gas Tech - Consumer Version: Home Gas Safety Calculator",
            1200, 800, WindowStyle.FLUID_METAL
        )
        
        self.setup_consumer_interface()
    
    def setup_consumer_interface(self):
        """Setup consumer-specific interface elements"""
        # Main tool buttons
        tools = [
            ("Gas Cost Calculator", 50, 100, 250, 60, self.open_cost_calculator),
            ("Safety Checklist", 350, 100, 250, 60, self.open_safety_checklist),
            ("Appliance Guide", 650, 100, 250, 60, self.open_appliance_guide),
            ("Emergency Info", 950, 100, 250, 60, self.open_emergency_info),
            ("Fuel Comparison", 200, 200, 250, 60, self.open_fuel_comparison),
            ("Contractor Finder", 750, 200, 250, 60, self.open_contractor_finder)
        ]
        
        for text, x, y, w, h, command in tools:
            self.window.add_button(x, y, w, h, text, WindowStyle.FLUID_METAL, command)
        
        # Welcome message
        self.window.canvas.create_text(
            600, 350, 
            text="Welcome to Gas Tech Consumer Version\nYour home gas safety companion",
            fill="white", font=("Arial", 18), anchor="center"
        )
    
    def open_cost_calculator(self):
        """Open gas cost calculator"""
        # Clear workspace
        self.clear_workspace()
        
        # Cost calculator interface
        self.window.canvas.create_text(
            600, 120, text="Gas Cost Calculator",
            fill="white", font=("Arial", 16, "bold"), anchor="center"
        )
        
        # Fuel selection
        self.window.canvas.create_text(
            100, 200, text="Fuel Type:",
            fill="white", font=("Arial", 12), anchor="w"
        )
        
        fuel_buttons = [
            ("Natural Gas", 200, 180, 120, 40, FuelType.NATURAL_GAS),
            ("Propane", 330, 180, 120, 40, FuelType.PROPANE),
            ("Butane", 460, 180, 120, 40, FuelType.BUTANE),
            ("Heating Oil", 590, 180, 120, 40, FuelType.HEATING_OIL)
        ]
        
        for text, x, y, w, h, fuel_type in fuel_buttons:
            btn = SleekButton(
                self.window.canvas, x, y, w, h, text,
                WindowStyle.AQUATIC, lambda ft=fuel_type: self.calculate_cost(ft)
            )
        
        # Results area
        self.results_text = self.window.canvas.create_text(
            600, 400, text="Select a fuel type to see cost analysis",
            fill="white", font=("Arial", 12), anchor="center"
        )
    
    def calculate_cost(self, fuel_type: FuelType):
        """Calculate and display cost analysis"""
        props = self.engine.fuel_database[fuel_type]
        
        # Sample usage calculations
        monthly_usage_mj = 12000  # Average home monthly usage in MJ
        cost_per_mmbtu = self.engine.estimate_fuel_cost(fuel_type)
        
        monthly_cost = (monthly_usage_mj / 1000) * cost_per_mmbtu
        annual_cost = monthly_cost * 12
        
        results = f"""
{fuel_type.value.replace('_', ' ').title()} Cost Analysis:

Monthly Usage: {monthly_usage_mj:,} MJ
Cost per MMBTU: ${cost_per_mmbtu:.2f}
Estimated Monthly Cost: ${monthly_cost:.2f}
Estimated Annual Cost: ${annual_cost:.2f}

Efficiency Rating: {self.engine.calculate_safety_rating(fuel_type)}/10
Heating Value: {props.heating_value} MJ/kg

Environmental Impact:
- CO₂ per MMBTU: Natural Gas: 53 kg, Propane: 64 kg, Oil: 73 kg
- Renewable: Natural Gas is the cleanest fossil fuel
        """
        
        self.window.canvas.itemconfig(self.results_text, text=results)
    
    def open_safety_checklist(self):
        """Open safety checklist"""
        self.clear_workspace()
        
        self.window.canvas.create_text(
            600, 120, text="Home Gas Safety Checklist",
            fill="white", font=("Arial", 16, "bold"), anchor="center"
        )
        
        safety_items = [
            "✓ Carbon monoxide detectors installed and working",
            "✓ Gas appliances inspected annually",
            "✓ Proper ventilation in gas appliance areas",
            "✓ No flammable materials near gas equipment",
            "✓ Emergency shut-off valves accessible",
            "✓ Gas lines protected from damage",
            "✓ Pilot lights burning blue (not yellow/orange)",
            "✓ No gas odor detected in home",
            "✓ Professional technician contact available",
            "✓ Family emergency plan in place"
        ]
        
        y_pos = 200
        for item in safety_items:
            self.window.canvas.create_text(
                150, y_pos, text=item,
                fill="white", font=("Arial", 11), anchor="w"
            )
            y_pos += 30
        
        self.window.canvas.create_text(
            600, 600, text="Complete this checklist monthly for maximum safety",
            fill="#FFD700", font=("Arial", 12, "italic"), anchor="center"
        )
    
    def open_appliance_guide(self):
        """Open appliance compatibility guide"""
        self.clear_workspace()
        
        self.window.canvas.create_text(
            600, 120, text="Gas Appliance Compatibility Guide",
            fill="white", font=("Arial", 16, "bold"), anchor="center"
        )
        
        appliances = {
            "Furnace": {"btu": "80,000-120,000", "fuel": "Natural Gas/Propane"},
            "Water Heater": {"btu": "30,000-75,000", "fuel": "Natural Gas/Propane"},
            "Stove/Cooktop": {"btu": "40,000-60,000", "fuel": "Natural Gas/Propane"},
            "Dryer": {"btu": "22,000-30,000", "fuel": "Natural Gas/Propane"},
            "Fireplace": {"btu": "20,000-60,000", "fuel": "Natural Gas/Propane"},
            "Generator": {"btu": "10,000-150,000", "fuel": "Propane/Natural Gas"},
            "Pool Heater": {"btu": "100,000-400,000", "fuel": "Natural Gas/Propane"},
            "Outdoor Grill": {"btu": "30,000-60,000", "fuel": "Propane/Butane"}
        }
        
        y_pos = 180
        self.window.canvas.create_text(
            200, y_pos, text="Appliance", fill="#87CEEB", font=("Arial", 12, "bold"), anchor="w"
        )
        self.window.canvas.create_text(
            400, y_pos, text="BTU Range", fill="#87CEEB", font=("Arial", 12, "bold"), anchor="w"
        )
        self.window.canvas.create_text(
            600, y_pos, text="Compatible Fuels", fill="#87CEEB", font=("Arial", 12, "bold"), anchor="w"
        )
        y_pos += 30
        
        for appliance, specs in appliances.items():
            self.window.canvas.create_text(
                200, y_pos, text=appliance, fill="white", font=("Arial", 11), anchor="w"
            )
            self.window.canvas.create_text(
                400, y_pos, text=specs["btu"], fill="white", font=("Arial", 11), anchor="w"
            )
            self.window.canvas.create_text(
                600, y_pos, text=specs["fuel"], fill="white", font=("Arial", 11), anchor="w"
            )
            y_pos += 25
    
    def open_emergency_info(self):
        """Open emergency procedures"""
        self.clear_workspace()
        
        self.window.canvas.create_text(
            600, 120, text="Gas Emergency Procedures",
            fill="#FF6B6B", font=("Arial", 16, "bold"), anchor="center"
        )
        
        emergency_steps = [
            "🚨 IF YOU SMELL GAS:",
            "",
            "1. DO NOT use electrical switches or phones",
            "2. DO NOT smoke or use open flames", 
            "3. EVACUATE everyone immediately",
            "4. Call 911 and gas company from outside",
            "5. DO NOT re-enter until declared safe",
            "",
            "⚡ CARBON MONOXIDE SYMPTOMS:",
            "• Headache, dizziness, nausea",
            "• Confusion, shortness of breath",
            "• If suspected: GET FRESH AIR IMMEDIATELY",
            "",
            "📞 EMERGENCY NUMBERS:",
            "• 911 - All emergencies",
            "• Gas Company Emergency: [Local Number]",
            "• Poison Control: 1-800-222-1222"
        ]
        
        y_pos = 180
        for step in emergency_steps:
            if step.startswith("🚨") or step.startswith("⚡") or step.startswith("📞"):
                color = "#FFD700"
                font_style = ("Arial", 12, "bold")
            elif step.startswith("•"):
                color = "#FFB6C1"
                font_style = ("Arial", 10)
            else:
                color = "white"
                font_style = ("Arial", 11)
            
            self.window.canvas.create_text(
                150, y_pos, text=step, fill=color, font=font_style, anchor="w"
            )
            y_pos += 25 if step else 15
    
    def open_fuel_comparison(self):
        """Open fuel comparison tool"""
        self.clear_workspace()
        
        self.window.canvas.create_text(
            600, 120, text="Fuel Type Comparison",
            fill="white", font=("Arial", 16, "bold"), anchor="center"
        )
        
        comparison_data = self.engine.get_fuel_comparison_data()
        
        y_pos = 180
        headers = ["Fuel", "Cost/MMBTU", "Heating Value", "Safety", "Environment"]
        x_positions = [150, 300, 450, 600, 750]
        
        # Headers
        for i, header in enumerate(headers):
            self.window.canvas.create_text(
                x_positions[i], y_pos, text=header,
                fill="#87CEEB", font=("Arial", 12, "bold"), anchor="center"
            )
        
        y_pos += 35
        
        # Data rows
        for fuel_type, data in comparison_data.items():
            fuel_name = data["name"]
            cost = f"${data['cost_per_mmbtu_estimate']:.2f}"
            heating = f"{data['heating_value']} MJ/kg"
            safety = f"{data['safety_rating']}/10"
            env = self.get_environmental_rating(fuel_type)
            
            row_data = [fuel_name, cost, heating, safety, env]
            
            for i, item in enumerate(row_data):
                self.window.canvas.create_text(
                    x_positions[i], y_pos, text=item,
                    fill="white", font=("Arial", 10), anchor="center"
                )
            
            y_pos += 30
    
    def get_environmental_rating(self, fuel_type: str) -> str:
        """Get environmental rating for fuel"""
        ratings = {
            "natural_gas": "A+ (Cleanest)",
            "propane": "B+ (Good)",
            "butane": "B (Fair)",
            "heating_oil": "C+ (Moderate)"
        }
        return ratings.get(fuel_type, "N/A")
    
    def open_contractor_finder(self):
        """Open contractor finder interface"""
        self.clear_workspace()
        
        self.window.canvas.create_text(
            600, 120, text="Certified Gas Contractor Finder",
            fill="white", font=("Arial", 16, "bold"), anchor="center"
        )
        
        tips = [
            "How to Find a Qualified Gas Contractor:",
            "",
            "✓ Verify current license and insurance",
            "✓ Check for gas fitting certification",
            "✓ Read recent customer reviews",
            "✓ Get multiple written estimates",
            "✓ Ask about warranty on work",
            "✓ Verify they pull required permits",
            "✓ Check for manufacturer certifications",
            "",
            "Red Flags to Avoid:",
            "• No written estimates",
            "• Cash-only payments",
            "• No license verification",
            "• Pressure to sign immediately",
            "• Vague scope of work"
        ]
        
        y_pos = 180
        for tip in tips:
            if tip.startswith("How to") or tip.startswith("Red Flags"):
                color = "#87CEEB"
                font_style = ("Arial", 12, "bold")
            elif tip.startswith("✓"):
                color = "#90EE90"
                font_style = ("Arial", 10)
            elif tip.startswith("•"):
                color = "#FFB6C1"
                font_style = ("Arial", 10)
            else:
                color = "white"
                font_style = ("Arial", 11)
            
            self.window.canvas.create_text(
                150, y_pos, text=tip, fill=color, font=font_style, anchor="w"
            )
            y_pos += 25 if tip else 15
    
    def clear_workspace(self):
        """Clear the workspace area"""
        # Delete all text elements (keeping buttons)
        items = self.window.canvas.find_all()
        for item in items:
            if self.window.canvas.type(item) == "text":
                tags = self.window.canvas.gettags(item)
                if "button_text" not in tags and "title_text" not in tags:
                    self.window.canvas.delete(item)
    
    def run(self):
        """Run the consumer version"""
        self.window.run()

def main():
    """Main entry point for consumer version"""
    app = ConsumerVersion()
    app.run()

if __name__ == "__main__":
    main()
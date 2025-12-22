"""
Heat Load Distribution & Air Change Handling Workshop
Advanced thermal calculations for residential and commercial systems
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class BuildingType(Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"

class ClimateZone(Enum):
    HOT_HUMID = "hot_humid"
    HOT_DRY = "hot_dry"
    MIXED_HUMID = "mixed_humid"
    MIXED_DRY = "mixed_dry"
    COLD = "cold"
    VERY_COLD = "very_cold"

@dataclass
class RoomData:
    """Room specifications for heat load calculation"""
    name: str
    length_ft: float
    width_ft: float
    height_ft: float
    window_area_sqft: float
    door_area_sqft: float
    exterior_walls_ft: float
    insulation_r_value: float
    exposure_factor: float  # North=1.1, South=0.9, East/West=1.0

@dataclass
class HVACRequirements:
    """HVAC system requirements"""
    heating_load_btuhr: float
    cooling_load_tons: float
    air_changes_per_hour: float
    required_cfm: float
    humidity_control_needed: bool

class HeatLoadCalculator:
    """Advanced heat load and air change calculations"""
    
    def __init__(self):
        self.climate_factors = {
            ClimateZone.HOT_HUMID: {"heating": 0.8, "cooling": 1.3, "humidity": 1.2},
            ClimateZone.HOT_DRY: {"heating": 0.9, "cooling": 1.2, "humidity": 0.7},
            ClimateZone.MIXED_HUMID: {"heating": 1.0, "cooling": 1.0, "humidity": 1.1},
            ClimateZone.MIXED_DRY: {"heating": 1.1, "cooling": 0.9, "humidity": 0.8},
            ClimateZone.COLD: {"heating": 1.3, "cooling": 0.7, "humidity": 0.9},
            ClimateZone.VERY_COLD: {"heating": 1.5, "cooling": 0.6, "humidity": 0.8}
        }
        
        self.air_change_rates = {
            BuildingType.RESIDENTIAL: {
                "living_room": 0.5, "bedroom": 0.5, "kitchen": 1.5, "bathroom": 2.0,
                "laundry": 1.0, "garage": 0.5, "basement": 0.3, "attic": 0.2
            },
            BuildingType.COMMERCIAL: {
                "office": 0.5, "retail": 1.0, "restaurant": 2.0, "warehouse": 0.3,
                "classroom": 1.0, "hospital": 2.0, "hotel": 1.5, "gym": 2.5
            },
            BuildingType.INDUSTRIAL: {
                "manufacturing": 1.0, "paint_shop": 3.0, "welding": 5.0, "chemical": 10.0,
                "storage": 0.5, "loading_dock": 2.0, "boiler_room": 5.0, "lab": 3.0
            }
        }
    
    def calculate_room_heat_load(self, room: RoomData, outdoor_temp_f: float, 
                                indoor_temp_f: float, climate_zone: ClimateZone) -> Dict[str, float]:
        """Calculate heat load for individual room"""
        
        # Room volume
        volume_cuft = room.length_ft * room.width_ft * room.height_ft
        
        # Temperature difference
        temp_diff = abs(indoor_temp_f - outdoor_temp_f)
        
        # Heat loss through walls
        wall_area = room.exterior_walls_ft * room.height_ft
        wall_heat_loss = (wall_area * temp_diff) / room.insulation_r_value
        
        # Heat loss through windows (U-factor typically 0.35-0.5)
        window_u_factor = 0.4
        window_heat_loss = room.window_area_sqft * window_u_factor * temp_diff
        
        # Heat loss through doors (U-factor typically 0.2-0.3)
        door_u_factor = 0.25
        door_heat_loss = room.door_area_sqft * door_u_factor * temp_diff
        
        # Infiltration heat loss (air changes)
        infiltration_ach = 0.35 if room.name.lower() != "garage" else 0.5
        infiltration_heat_loss = (volume_cuft * infiltration_ach * temp_diff * 0.018)  # 0.018 BTU/cu·ft·°F
        
        # Total heat loss
        total_heat_loss = (wall_heat_loss + window_heat_loss + door_heat_loss + 
                          infiltration_heat_loss) * room.exposure_factor
        
        # Apply climate factor
        climate_factor = self.climate_factors[climate_zone]["heating"]
        adjusted_heat_loss = total_heat_loss * climate_factor
        
        return {
            "room_heat_loss_btuhr": adjusted_heat_loss,
            "wall_heat_loss_btuhr": wall_heat_loss,
            "window_heat_loss_btuhr": window_heat_loss,
            "door_heat_loss_btuhr": door_heat_loss,
            "infiltration_heat_loss_btuhr": infiltration_heat_loss,
            "volume_cuft": volume_cuft,
            "heat_loss_per_sqft": adjusted_heat_loss / (room.length_ft * room.width_ft)
        }
    
    def calculate_cooling_load(self, room: RoomData, outdoor_temp_f: float, 
                             indoor_temp_f: float, climate_zone: ClimateZone, 
                             occupancy: int = 0, equipment_watts: float = 0) -> Dict[str, float]:
        """Calculate cooling load for room"""
        
        # Sensible heat gain
        temp_diff = outdoor_temp_f - indoor_temp_f
        area_sqft = room.length_ft * room.width_ft
        
        # Solar heat gain through windows
        solar_heat_gain = room.window_area_sqft * 60 * room.exposure_factor  # 60 BTU/hr/sqft typical
        
        # Transmission heat gain
        window_u_factor = 0.4
        transmission_gain = room.window_area_sqft * window_u_factor * temp_diff
        
        # Internal heat gains
        occupancy_gain = occupancy * 400  # 400 BTU/hr per person
        equipment_gain = equipment_watts * 3.41  # Watts to BTU/hr
        
        # Lighting heat gain
        lighting_gain = area_sqft * 1.0  # 1 BTU/hr/sqft typical residential
        
        # Total sensible cooling load
        total_sensible = (solar_heat_gain + transmission_gain + occupancy_gain + 
                         equipment_gain + lighting_gain)
        
        # Latent heat load (humidity)
        climate_factor = self.climate_factors[climate_zone]["humidity"]
        latent_load = total_sensible * 0.3 * climate_factor  # 30% latent typically
        
        # Total cooling load
        total_cooling_load = total_sensible + latent_load
        cooling_tons = total_cooling_load / 12000  # 12,000 BTU/hr per ton
        
        return {
            "total_cooling_load_btuhr": total_cooling_load,
            "sensible_cooling_btuhr": total_sensible,
            "latent_cooling_btuhr": latent_load,
            "cooling_load_tons": cooling_tons,
            "solar_heat_gain_btuhr": solar_heat_gain,
            "internal_gains_btuhr": occupancy_gain + equipment_gain + lighting_gain
        }
    
    def calculate_air_change_requirements(self, rooms: List[RoomData], 
                                        building_type: BuildingType) -> Dict[str, float]:
        """Calculate air change requirements for building"""
        
        total_volume = 0
        weighted_ach = 0
        room_requirements = {}
        
        for room in rooms:
            volume = room.length_ft * room.width_ft * room.height_ft
            total_volume += volume
            
            # Determine room type for ACH lookup
            room_type = self.classify_room_type(room.name)
            base_ach = self.air_change_rates[building_type].get(room_type, 0.5)
            
            # Adjust for special conditions
            if room.window_area_sqft > 50:  # Lots of windows
                base_ach *= 1.2
            
            if room.name.lower() in ["kitchen", "bathroom", "laundry"]:
                base_ach *= 1.5  # Higher ventilation needs
            
            room_cfm = (volume * base_ach) / 60  # Convert ACH to CFM
            weighted_ach += room_cfm
            
            room_requirements[room.name] = {
                "volume_cuft": volume,
                "required_ach": base_ach,
                "required_cfm": room_cfm,
                "room_type": room_type
            }
        
        total_cfm = weighted_ach
        overall_ach = (total_cfm * 60) / total_volume
        
        return {
            "total_volume_cuft": total_volume,
            "total_required_cfm": total_cfm,
            "overall_ach": overall_ach,
            "room_requirements": room_requirements
        }
    
    def classify_room_type(self, room_name: str) -> str:
        """Classify room name to standard type"""
        name_lower = room_name.lower()
        
        room_mappings = {
            "living": "living_room", "family": "living_room", "great": "living_room",
            "bedroom": "bedroom", "sleep": "bedroom", "master": "bedroom",
            "kitchen": "kitchen", "cook": "kitchen", "chef": "kitchen",
            "bathroom": "bathroom", "bath": "bathroom", "powder": "bathroom",
            "laundry": "laundry", "utility": "laundry", "wash": "laundry",
            "garage": "garage", "parking": "garage", "car": "garage",
            "basement": "basement", "cellar": "basement", "foundation": "basement",
            "attic": "attic", "crawl": "attic", "loft": "attic"
        }
        
        for keyword, room_type in room_mappings.items():
            if keyword in name_lower:
                return room_type
        
        return "living_room"  # Default
    
    def calculate_hvac_system_requirements(self, rooms: List[RoomData], 
                                         climate_zone: ClimateZone, 
                                         building_type: BuildingType,
                                         outdoor_temp_winter_f: float = 0,
                                         outdoor_temp_summer_f: float = 95,
                                         indoor_temp_winter_f: float = 70,
                                         indoor_temp_summer_f: float = 75) -> HVACRequirements:
        """Calculate complete HVAC system requirements"""
        
        total_heating_load = 0
        total_cooling_load = 0
        
        for room in rooms:
            # Calculate heating load
            heating_result = self.calculate_room_heat_load(
                room, outdoor_temp_winter_f, indoor_temp_winter_f, climate_zone
            )
            total_heating_load += heating_result["room_heat_loss_btuhr"]
            
            # Calculate cooling load
            cooling_result = self.calculate_cooling_load(
                room, outdoor_temp_summer_f, indoor_temp_summer_f, climate_zone
            )
            total_cooling_load += cooling_result["total_cooling_load_btuhr"]
        
        # Calculate air change requirements
        air_requirements = self.calculate_air_change_requirements(rooms, building_type)
        
        # Determine if humidity control is needed
        humidity_needed = climate_zone in [ClimateZone.HOT_HUMID, ClimateZone.MIXED_HUMID]
        
        return HVACRequirements(
            heating_load_btuhr=total_heating_load,
            cooling_load_tons=total_cooling_load / 12000,
            air_changes_per_hour=air_requirements["overall_ach"],
            required_cfm=air_requirements["total_required_cfm"],
            humidity_control_needed=humidity_needed
        )
    
    def generate_heat_load_report(self, rooms: List[RoomData], climate_zone: ClimateZone,
                                building_type: BuildingType) -> str:
        """Generate comprehensive heat load report"""
        
        requirements = self.calculate_hvac_system_requirements(
            rooms, climate_zone, building_type
        )
        
        air_requirements = self.calculate_air_change_requirements(rooms, building_type)
        
        report = f"""
HEAT LOAD & AIR CHANGE ANALYSIS REPORT
=====================================

Building Information:
- Type: {building_type.value.title()}
- Climate Zone: {climate_zone.value.replace('_', ' ').title()}
- Total Rooms: {len(rooms)}

System Requirements:
- Heating Load: {requirements.heating_load_btuhr:,.0f} BTU/hr
- Cooling Load: {requirements.cooling_load_tons:.1f} tons
- Air Changes: {requirements.air_changes_per_hour:.2f} ACH
- Required CFM: {requirements.required_cfm:,.0f}
- Humidity Control: {'Required' if requirements.humidity_control_needed else 'Not Required'}

Room-by-Room Analysis:
"""
        
        for room in rooms:
            heating = self.calculate_room_heat_load(room, 0, 70, climate_zone)
            cooling = self.calculate_cooling_load(room, 95, 75, climate_zone)
            
            report += f"""
{room.name.title()}:
  - Size: {room.length_ft}'x{room.width_ft}'x{room.height_ft}' ({room.length_ft * room.width_ft:.0f} sqft)
  - Heating: {heating['room_heat_loss_btuhr']:,.0f} BTU/hr
  - Cooling: {cooling['total_cooling_load_btuhr']:,.0f} BTU/hr ({cooling['cooling_load_tons']:.2f} tons)
  - Ventilation: {air_requirements['room_requirements'][room.name]['required_cfm']:.0f} CFM
"""
        
        return report

# Example usage and demo functions
def demo_heat_load_calculator():
    """Demonstrate heat load calculator functionality"""
    calculator = HeatLoadCalculator()
    
    # Sample rooms
    rooms = [
        RoomData("living_room", 20, 15, 8, 40, 20, 35, 13, 1.0),
        RoomData("master_bedroom", 16, 14, 8, 25, 20, 16, 13, 0.9),
        RoomData("kitchen", 12, 10, 8, 15, 20, 10, 13, 1.1),
        RoomData("bathroom", 8, 6, 8, 5, 20, 6, 11, 1.0),
        RoomData("garage", 24, 20, 9, 20, 80, 44, 5, 1.2)
    ]
    
    # Calculate requirements
    requirements = calculator.calculate_hvac_system_requirements(
        rooms, ClimateZone.COLD, BuildingType.RESIDENTIAL
    )
    
    print("Heat Load Calculator Demo:")
    print(f"Heating Load: {requirements.heating_load_btuhr:,.0f} BTU/hr")
    print(f"Cooling Load: {requirements.cooling_load_tons:.1f} tons")
    print(f"Air Changes: {requirements.air_changes_per_hour:.2f} ACH")
    print(f"Required CFM: {requirements.required_cfm:,.0f}")

if __name__ == "__main__":
    demo_heat_load_calculator()
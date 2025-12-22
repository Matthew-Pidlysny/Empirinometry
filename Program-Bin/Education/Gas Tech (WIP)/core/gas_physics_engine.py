"""
Gas Physics Engine - Core Calculations for Gas Tech Suite
Handles all mathematical calculations for natural gas, propane, butane, and oil systems
"""

import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class FuelType(Enum):
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    BUTANE = "butane"
    HEATING_OIL = "heating_oil"

@dataclass
class FuelProperties:
    """Physical and chemical properties of fuels"""
    name: str
    molecular_weight: float  # g/mol
    density_gas: float  # kg/m³ at STP
    density_liquid: float  # kg/m³
    heating_value: float  # MJ/kg
    lower_heating_value: float  # MJ/kg
    boiling_point: float  # °C
    flash_point: float  # °C
    wobble_index: float  # for gas interchangeability
    methane_number: int  # for knocking resistance

class GasPhysicsEngine:
    """Advanced gas physics calculations and analysis"""
    
    def __init__(self):
        self.fuel_database = {
            FuelType.NATURAL_GAS: FuelProperties(
                name="Natural Gas",
                molecular_weight=16.04,
                density_gas=0.72,
                density_liquid=422.0,
                heating_value=55.5,
                lower_heating_value=50.0,
                boiling_point=-161.5,
                flash_point=-188,
                wobble_index=40.6,
                methane_number=100
            ),
            FuelType.PROPANE: FuelProperties(
                name="Propane",
                molecular_weight=44.10,
                density_gas=1.88,
                density_liquid=507.0,
                heating_value=50.3,
                lower_heating_value=46.3,
                boiling_point=-42.1,
                flash_point=-104,
                wobble_index=68.4,
                methane_number=35
            ),
            FuelType.BUTANE: FuelProperties(
                name="Butane",
                molecular_weight=58.12,
                density_gas=2.54,
                density_liquid=580.0,
                heating_value=49.5,
                lower_heating_value=45.7,
                boiling_point=-0.5,
                flash_point=-60,
                wobble_index=89.0,
                methane_number=10
            ),
            FuelType.HEATING_OIL: FuelProperties(
                name="Heating Oil",
                molecular_weight=200.0,
                density_gas=0.0,  # Not applicable
                density_liquid=850.0,
                heating_value=42.5,
                lower_heating_value=40.0,
                boiling_point=250.0,
                flash_point=66,
                wobble_index=0.0,
                methane_number=0
            )
        }
    
    def calculate_gas_flow_rate(self, pressure_psi: float, pipe_diameter_inches: float, 
                              length_feet: float, fuel_type: FuelType) -> float:
        """
        Calculate gas flow rate using Weymouth equation
        Returns flow rate in cubic feet per hour
        """
        if pressure_psi <= 0 or pipe_diameter_inches <= 0 or length_feet <= 0:
            return 0.0
        
        fuel = self.fuel_database[fuel_type]
        gravity = fuel.density_gas / 0.0765  # Specific gravity relative to air
        
        # Weymouth equation simplified
        # Q = 18.062 * d^(8/3) * sqrt((P1² - P2²) / (G * L * T * Z))
        # Simplified for pressure drop calculation
        base_flow = 18.062 * (pipe_diameter_inches ** (8/3))
        pressure_factor = math.sqrt(pressure_psi)
        correction_factor = math.sqrt(1.0 / (gravity * length_feet * 1.0))
        
        flow_rate = base_flow * pressure_factor * correction_factor
        return max(0, flow_rate)
    
    def calculate_pipe_sizing(self, required_flow_cfph: float, length_feet: float, 
                           pressure_psi: float, fuel_type: FuelType) -> Dict[str, float]:
        """
        Calculate required pipe diameter for given flow requirements
        Returns pipe sizes in inches
        """
        standard_sizes = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
        suitable_sizes = []
        
        for diameter in standard_sizes:
            flow_capacity = self.calculate_gas_flow_rate(pressure_psi, diameter, length_feet, fuel_type)
            if flow_capacity >= required_flow_cfph:
                suitable_sizes.append(diameter)
        
        return {
            "minimum_size": min(suitable_sizes) if suitable_sizes else 0,
            "recommended_size": suitable_sizes[1] if len(suitable_sizes) > 1 else (suitable_sizes[0] if suitable_sizes else 0),
            "sizes_available": suitable_sizes
        }
    
    def calculate_pressure_drop(self, flow_rate_cfph: float, pipe_diameter_inches: float, 
                             length_feet: float, fuel_type: FuelType) -> float:
        """
        Calculate pressure drop in gas pipeline
        Returns pressure drop in PSI
        """
        if flow_rate_cfph <= 0 or pipe_diameter_inches <= 0:
            return 0.0
        
        fuel = self.fuel_database[fuel_type]
        gravity = fuel.density_gas / 0.0765
        
        # Simplified pressure drop calculation
        # ΔP = (L * Q² * G) / (C * d⁵)
        # C = flow coefficient (simplified), using higher value for realistic drops
        C = 500.0  # Corrected coefficient for realistic pressure drops
        pressure_drop = (length_feet * (flow_rate_cfph ** 2) * gravity) / (C * (pipe_diameter_inches ** 5))
        
        return pressure_drop
    
    def calculate_combustion_air(self, btu_per_hour: float, fuel_type: FuelType) -> Dict[str, float]:
        """
        Calculate combustion air requirements
        Returns air requirements in CFM
        """
        fuel = self.fuel_database[fuel_type]
        
        # Standard air requirements per BTU
        # Natural gas: 10 CFM per 1000 BTU
        # Propane: 12.5 CFM per 1000 BTU
        # Butane: 13 CFM per 1000 BTU
        # Oil: 15 CFM per 1000 BTU
        
        air_requirements = {
            FuelType.NATURAL_GAS: 10.0,
            FuelType.PROPANE: 12.5,
            FuelType.BUTANE: 13.0,
            FuelType.HEATING_OIL: 15.0
        }
        
        cfm_per_1000_btu = air_requirements[fuel_type]
        combustion_air_cfm = (btu_per_hour / 1000) * cfm_per_1000_btu
        
        return {
            "combustion_air_cfm": combustion_air_cfm,
            "dilution_air_cfm": combustion_air_cfm * 0.5,  # 50% additional for dilution
            "total_air_cfm": combustion_air_cfm * 1.5
        }
    
    def calculate_vent_sizing(self, btu_per_hour: float, vent_type: str = "chimney") -> Dict[str, float]:
        """
        Calculate vent sizing requirements
        Returns vent diameter in inches
        """
        if vent_type.lower() == "chimney":
            # Chimney sizing: 1 square inch per 1000 BTU
            area_per_btu = 1.0 / 1000.0
        else:  # direct vent
            # Direct vent: 0.8 square inch per 1000 BTU
            area_per_btu = 0.8 / 1000.0
        
        required_area = btu_per_hour * area_per_btu
        diameter = math.sqrt((4 * required_area) / math.pi)
        
        return {
            "required_diameter_inches": diameter,
            "recommended_size": math.ceil(diameter * 2) / 2,  # Round up to nearest 0.5"
            "required_area_sq_inches": required_area
        }
    
    def calculate_appliance_capacity(self, flow_rate_cfph: float, fuel_type: FuelType) -> Dict[str, float]:
        """
        Convert flow rate to appliance capacity
        Returns BTU ratings
        """
        fuel = self.fuel_database[fuel_type]
        
        # Convert CFH to BTU (Natural gas: 1000 BTU/CFH, Propane: 2500 BTU/CFH)
        btu_per_cfh = {
            FuelType.NATURAL_GAS: 1000,
            FuelType.PROPANE: 2500,
            FuelType.BUTANE: 3200,
            FuelType.HEATING_OIL: 140000  # Per gallon
        }
        
        if fuel_type == FuelType.HEATING_OIL:
            # For oil, convert CFH to GPH (approximate)
            gph = flow_rate_cfph * 0.1337  # Conversion factor
            btu_per_hour = gph * btu_per_cfh[fuel_type]
        else:
            btu_per_hour = flow_rate_cfph * btu_per_cfh[fuel_type]
        
        return {
            "btu_per_hour": btu_per_hour,
            "btu_per_hour_millions": btu_per_hour / 1000000,
            "kilowatts": btu_per_hour * 0.000293071  # Convert BTU to kW
        }
    
    def check_gas_interchangeability(self, fuel1: FuelType, fuel2: FuelType) -> Dict[str, bool]:
        """
        Check if two gases are interchangeable using Wobble Index
        """
        f1 = self.fuel_database[fuel1]
        f2 = self.fuel_database[fuel2]
        
        wobble_diff = abs(f1.wobble_index - f2.wobble_index)
        wobble_compatible = wobble_diff < 10  # Within 10% is generally acceptable
        
        return {
            "wobble_compatible": wobble_compatible,
            "wobble_difference": wobble_diff,
            "methane_compatible": abs(f1.methane_number - f2.methane_number) < 20
        }
    
    def calculate_thermal_efficiency(self, btu_input: float, btu_output: float) -> Dict[str, float]:
        """
        Calculate thermal efficiency and heat losses
        """
        if btu_input <= 0:
            return {"efficiency_percent": 0, "heat_loss_btu": 0}
        
        efficiency = (btu_output / btu_input) * 100
        heat_loss = btu_input - btu_output
        
        return {
            "efficiency_percent": efficiency,
            "heat_loss_btu": heat_loss,
            "heat_loss_percent": (heat_loss / btu_input) * 100
        }
    
    def get_fuel_comparison_data(self) -> Dict[str, Dict]:
        """
        Get comprehensive fuel comparison data
        """
        comparison = {}
        for fuel_type, properties in self.fuel_database.items():
            comparison[fuel_type.value] = {
                "name": properties.name,
                "molecular_weight": properties.molecular_weight,
                "heating_value": properties.heating_value,
                "cost_per_mmbtu_estimate": self.estimate_fuel_cost(fuel_type),
                "safety_rating": self.calculate_safety_rating(fuel_type)
            }
        
        return comparison
    
    def estimate_fuel_cost(self, fuel_type: FuelType) -> float:
        """
        Estimate fuel cost per MMBTU (simplified market estimates)
        """
        # Estimated costs per MMBTU (will vary by market)
        cost_estimates = {
            FuelType.NATURAL_GAS: 3.50,
            FuelType.PROPANE: 25.00,
            FuelType.BUTANE: 28.00,
            FuelType.HEATING_OIL: 22.00
        }
        
        return cost_estimates.get(fuel_type, 0.0)
    
    def calculate_safety_rating(self, fuel_type: FuelType) -> int:
        """
        Calculate safety rating (1-10, 10 being safest)
        Based on flash point, flammability range, etc.
        """
        fuel = self.fuel_database[fuel_type]
        
        # Safety scoring based on flash point and other factors
        if fuel.flash_point > 60:  # Above room temperature
            flash_score = 10
        elif fuel.flash_point > 0:
            flash_score = 7
        elif fuel.flash_point > -40:
            flash_score = 4
        else:
            flash_score = 2
        
        # Additional safety factors
        if fuel_type == FuelType.NATURAL_GAS:
            safety_bonus = 2  # Lighter than air, dissipates quickly
        elif fuel_type == FuelType.HEATING_OIL:
            safety_bonus = 1  # Liquid at room temperature
        else:
            safety_bonus = 0
        
        return min(10, flash_score + safety_bonus)
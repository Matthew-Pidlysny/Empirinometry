"""
Design Your Own Piping System Workshop
Interactive piping system design with legal estimation methods and standards compliance
"""

import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

class PipeMaterial(Enum):
    BLACK_STEEL = "black_steel"
    GALVANIZED_STEEL = "galvanized_steel"
    COPPER = "copper"
    CSST = "csst"
    PE_XT = "pe_xt"
    PVC = "pvc"

class PipeStandard(Enum):
    NFPA_54 = "nfpa_54"  # National Fuel Gas Code
    CSA_B149_1 = "csa_b149_1"  # Canadian Standard
    IFGC = "ifgc"  # International Fuel Gas Code
    UPC = "upc"  # Uniform Plumbing Code
    CUSTOM = "custom"  # User-defined parameters

class FittingType(Enum):
    ELBOW_90 = "elbow_90"
    ELBOW_45 = "elbow_45"
    TEE = "tee"
    COUPLING = "coupling"
    VALVE = "valve"
    REDUCER = "reducer"
    UNION = "union"

@dataclass
class PipeSegment:
    """Individual pipe segment in the system"""
    id: str
    material: PipeMaterial
    nominal_size_inches: float
    length_ft: float
    start_pressure_psig: float
    end_pressure_psig: float
    flow_rate_cfph: float
    fitting_count: Dict[FittingType, int]

@dataclass
class EquivalentLengthData:
    """Equivalent length data for fittings"""
    fitting: FittingType
    nominal_size_inches: float
    equivalent_length_ft: float

@dataclass
class PipingSystem:
    """Complete piping system configuration"""
    name: str
    standard: PipeStandard
    material: PipeMaterial
    segments: List[PipeSegment]
    supply_pressure_psig: float
    appliance_pressure_psig: float
    total_equivalent_length_ft: float
    pressure_drop_psig: float

class PipingSystemDesigner:
    """Advanced piping system design and calculation engine"""
    
    def __init__(self):
        self.equivalent_length_tables = self._load_equivalent_length_tables()
        self.pipe_roughness = {
            PipeMaterial.BLACK_STEEL: 0.00015,
            PipeMaterial.GALVANIZED_STEEL: 0.0002,
            PipeMaterial.COPPER: 0.000005,
            PipeMaterial.CSST: 0.00001,
            PipeMaterial.PE_XT: 0.00002,
            PipeMaterial.PVC: 0.000008
        }
        
        self.design_factors = {
            PipeStandard.NFPA_54: 1.0,
            PipeStandard.CSA_B149_1: 1.1,
            PipeStandard.IFGC: 1.05,
            PipeStandard.UPC: 1.0,
            PipeStandard.CUSTOM: 1.0
        }
    
    def _load_equivalent_length_tables(self) -> Dict[PipeStandard, Dict[FittingType, Dict[float, float]]]:
        """Load equivalent length data for different standards"""
        
        # NFPA 54 equivalent lengths (in feet)
        nfpa_54_lengths = {
            FittingType.ELBOW_90: {
                0.5: 1.0, 0.75: 1.5, 1.0: 2.0, 1.25: 2.5, 1.5: 3.0,
                2.0: 4.0, 2.5: 5.0, 3.0: 6.0, 4.0: 8.0, 6.0: 12.0
            },
            FittingType.ELBOW_45: {
                0.5: 0.5, 0.75: 0.8, 1.0: 1.0, 1.25: 1.3, 1.5: 1.5,
                2.0: 2.0, 2.5: 2.5, 3.0: 3.0, 4.0: 4.0, 6.0: 6.0
            },
            FittingType.TEE: {
                0.5: 3.0, 0.75: 4.5, 1.0: 6.0, 1.25: 7.5, 1.5: 9.0,
                2.0: 12.0, 2.5: 15.0, 3.0: 18.0, 4.0: 24.0, 6.0: 36.0
            },
            FittingType.VALVE: {
                0.5: 2.5, 0.75: 3.8, 1.0: 5.0, 1.25: 6.3, 1.5: 7.5,
                2.0: 10.0, 2.5: 12.5, 3.0: 15.0, 4.0: 20.0, 6.0: 30.0
            },
            FittingType.COUPLING: {
                0.5: 0.3, 0.75: 0.4, 1.0: 0.5, 1.25: 0.6, 1.5: 0.7,
                2.0: 1.0, 2.5: 1.2, 3.0: 1.5, 4.0: 2.0, 6.0: 3.0
            },
            FittingType.REDUCER: {
                0.5: 1.0, 0.75: 1.5, 1.0: 2.0, 1.25: 2.5, 1.5: 3.0,
                2.0: 4.0, 2.5: 5.0, 3.0: 6.0, 4.0: 8.0, 6.0: 12.0
            },
            FittingType.UNION: {
                0.5: 0.5, 0.75: 0.8, 1.0: 1.0, 1.25: 1.3, 1.5: 1.5,
                2.0: 2.0, 2.5: 2.5, 3.0: 3.0, 4.0: 4.0, 6.0: 6.0
            }
        }
        
        # CSA B149.1 equivalent lengths (more conservative)
        csa_b149_1_lengths = {}
        for fitting, size_data in nfpa_54_lengths.items():
            csa_b149_1_lengths[fitting] = {}
            for size, length in size_data.items():
                csa_b149_1_lengths[fitting][size] = length * 1.2  # 20% more conservative
        
        return {
            PipeStandard.NFPA_54: nfpa_54_lengths,
            PipeStandard.CSA_B149_1: csa_b149_1_lengths,
            PipeStandard.IFGC: nfpa_54_lengths,
            PipeStandard.UPC: nfpa_54_lengths,
            PipeStandard.CUSTOM: nfpa_54_lengths
        }
    
    def calculate_equivalent_length(self, segment: PipeSegment, standard: PipeStandard) -> float:
        """Calculate total equivalent length for a pipe segment"""
        
        base_length = segment.length_ft
        fitting_length = 0
        
        for fitting_type, count in segment.fitting_count.items():
            if count > 0:
                table = self.equivalent_length_tables[standard]
                if fitting_type in table:
                    size_table = table[fitting_type]
                    # Find closest size
                    available_sizes = list(size_table.keys())
                    closest_size = min(available_sizes, key=lambda x: abs(x - segment.nominal_size_inches))
                    equiv_length_per_fitting = size_table[closest_size]
                    fitting_length += equiv_length_per_fitting * count
        
        total_equivalent_length = base_length + fitting_length
        
        # Apply design factor
        design_factor = self.design_factors[standard]
        adjusted_length = total_equivalent_length * design_factor
        
        return adjusted_length
    
    def calculate_pressure_drop_weymouth(self, segment: PipeSegment, standard: PipeStandard) -> float:
        """Calculate pressure drop using Weymouth equation"""
        
        # Get gas properties (assuming natural gas)
        specific_gravity = 0.6  # Natural gas relative to air
        temperature_rankine = 520  # 60°F
        compressibility_factor = 1.0
        
        # Calculate equivalent length
        equiv_length = self.calculate_equivalent_length(segment, standard)
        
        # Weymouth equation for pressure drop
        # Q = C * sqrt(((P1² - P2²) * D⁵) / (G * L * T * Z))
        # Rearranged for pressure drop calculation
        
        if segment.flow_rate_cfph <= 0 or segment.nominal_size_inches <= 0:
            return 0.0
        
        # Weymouth constant
        C_w = 0.000433  # For natural gas, CFH, inches, PSI
        
        # Calculate pressure drop with corrected coefficient
        pressure_drop = ((segment.flow_rate_cfph / C_w) ** 2 * specific_gravity * 
                        equiv_length * temperature_rankine * compressibility_factor / 
                        (segment.nominal_size_inches ** 5)) * 0.001  # Scale factor for realistic values
        
        return pressure_drop
    
    def calculate_pipe_capacity(self, nominal_size_inches: float, length_ft: float,
                               max_pressure_drop_psig: float, standard: PipeStandard,
                               material: PipeMaterial) -> Dict[str, float]:
        """Calculate maximum flow capacity for a pipe"""
        
        # Create test segment
        test_segment = PipeSegment(
            id="test",
            material=material,
            nominal_size_inches=nominal_size_inches,
            length_ft=length_ft,
            start_pressure_psig=14.0,  # Typical low pressure
            end_pressure_psig=14.0 - max_pressure_drop_psig,
            flow_rate_cfph=1000,  # Initial guess
            fitting_count={}  # No fittings for basic capacity
        )
        
        # Iterative solution for flow rate
        target_drop = max_pressure_drop_psig
        flow_rate = 10  # Start with much smaller flow
        tolerance = 0.01
        max_iterations = 50
        
        for iteration in range(max_iterations):
            test_segment.flow_rate_cfph = flow_rate
            calculated_drop = self.calculate_pressure_drop_weymouth(test_segment, standard)
            
            error = abs(calculated_drop - target_drop)
            if error < tolerance:
                break
            
            # Adjust flow rate more conservatively
            if calculated_drop > target_drop:
                flow_rate *= 0.9  # Reduce flow more aggressively
            else:
                flow_rate *= 1.1  # Increase flow more conservatively
            
            # Prevent runaway calculations
            if flow_rate > 10000 or flow_rate < 0.1:
                break
        
        # Calculate BTU capacity (assuming natural gas, 1000 BTU/CFH)
        btu_capacity = flow_rate * 1000
        
        return {
            "max_flow_cfph": flow_rate,
            "max_btu_per_hour": btu_capacity,
            "max_btu_millions_per_hour": btu_capacity / 1000000,
            "pressure_drop_psig": calculated_drop,
            "iterations_used": iteration + 1
        }
    
    def design_piping_system(self, supply_pressure_psig: float, appliance_loads_btuhr: List[float],
                           appliance_distances_ft: List[float], standard: PipeStandard,
                           material: PipeMaterial) -> PipingSystem:
        """Design complete piping system for multiple appliances"""
        
        segments = []
        cumulative_distance = 0
        cumulative_load = sum(appliance_loads_btuhr)
        
        # Convert BTU to CFH (natural gas: 1000 BTU/CFH)
        total_flow_cfph = cumulative_load / 1000
        
        # Size main line with conservative sizing
        max_drop_psig = min(0.5, supply_pressure_psig * 0.1)  # Conservative pressure drop
        main_size = self.size_pipe_for_flow(total_flow_cfph, appliance_distances_ft[0], 
                                           supply_pressure_psig, max_drop_psig, standard, material)
        
        # Create main segment
        main_segment = PipeSegment(
            id="main",
            material=material,
            nominal_size_inches=main_size,
            length_ft=appliance_distances_ft[0],
            start_pressure_psig=supply_pressure_psig,
            end_pressure_psig=supply_pressure_psig,
            flow_rate_cfph=total_flow_cfph,
            fitting_count={FittingType.ELBOW_90: 2, FittingType.TEE: len(appliance_loads_btuhr)}
        )
        
        main_segment.end_pressure_psig = supply_pressure_psig - self.calculate_pressure_drop_weymouth(main_segment, standard)
        segments.append(main_segment)
        
        # Design branches
        for i, (load_btuh, distance_ft) in enumerate(zip(appliance_loads_btuhr, appliance_distances_ft)):
            if i == 0:
                continue  # Skip first (main line)
            
            # Calculate flow for this branch
            branch_flow_cfph = load_btuh / 1000
            
            # Size branch pipe
            branch_size = self.size_pipe_for_flow(branch_flow_cfph, distance_ft,
                                                main_segment.end_pressure_psig, 0.5,
                                                standard, material)
            
            # Create branch segment
            branch_segment = PipeSegment(
                id=f"branch_{i}",
                material=material,
                nominal_size_inches=branch_size,
                length_ft=distance_ft,
                start_pressure_psig=main_segment.end_pressure_psig,
                end_pressure_psig=main_segment.end_pressure_psig,
                flow_rate_cfph=branch_flow_cfph,
                fitting_count={FittingType.ELBOW_90: 1, FittingType.ELBOW_45: 1, FittingType.VALVE: 1}
            )
            
            branch_segment.end_pressure_psig = branch_segment.start_pressure_psig - self.calculate_pressure_drop_weymouth(branch_segment, standard)
            segments.append(branch_segment)
        
        # Calculate system totals
        total_equivalent_length = sum(self.calculate_equivalent_length(seg, standard) for seg in segments)
        total_pressure_drop = supply_pressure_psig - segments[-1].end_pressure_psig
        
        return PipingSystem(
            name="Designed System",
            standard=standard,
            material=material,
            segments=segments,
            supply_pressure_psig=supply_pressure_psig,
            appliance_pressure_psig=segments[-1].end_pressure_psig,
            total_equivalent_length_ft=total_equivalent_length,
            pressure_drop_psig=total_pressure_drop
        )
    
    def size_pipe_for_flow(self, flow_cfph: float, length_ft: float, inlet_pressure_psig: float,
                          max_drop_psig: float, standard: PipeStandard, material: PipeMaterial) -> float:
        """Size pipe for given flow requirements"""
        
        standard_sizes = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
        
        for size in standard_sizes:
            capacity = self.calculate_pipe_capacity(size, length_ft, max_drop_psig, standard, material)
            if capacity["max_flow_cfph"] >= flow_cfph:
                return size
        
        return max(standard_sizes)  # Return largest size if none sufficient
    
    def validate_system_compliance(self, system: PipingSystem) -> Dict[str, Union[bool, str]]:
        """Validate system compliance with selected standard"""
        
        validation_results = {
            "compliant": True,
            "issues": []
        }
        
        # Check pressure drops
        for segment in system.segments:
            pressure_drop = segment.start_pressure_psig - segment.end_pressure_psig
            
            # Maximum allowable pressure drops (WC - water column)
            max_drop_low_pressure = 0.5  # 7" WC
            max_drop_high_pressure = 5.0  # 5 PSI
            
            if segment.start_pressure_psig <= 14:  # Low pressure
                if pressure_drop > max_drop_low_pressure:
                    validation_results["compliant"] = False
                    validation_results["issues"].append(
                        f"Segment {segment.id}: Pressure drop {pressure_drop:.3f} PSI exceeds maximum {max_drop_low_pressure} PSI"
                    )
            else:  # High pressure
                if pressure_drop > max_drop_high_pressure:
                    validation_results["compliant"] = False
                    validation_results["issues"].append(
                        f"Segment {segment.id}: Pressure drop {pressure_drop:.3f} PSI exceeds maximum {max_drop_high_pressure} PSI"
                    )
        
        # Check minimum appliance pressure
        min_appliance_pressure = 3.5  # Minimum for most appliances
        if system.appliance_pressure_psig < min_appliance_pressure:
            validation_results["compliant"] = False
            validation_results["issues"].append(
                f"Appliance pressure {system.appliance_pressure_psig:.2f} PSI below minimum {min_appliance_pressure} PSI"
            )
        
        # Check velocity limits
        for segment in system.segments:
            velocity = self.calculate_gas_velocity(segment)
            max_velocity = 60  # ft/s typical limit
            if velocity > max_velocity:
                validation_results["compliant"] = False
                validation_results["issues"].append(
                    f"Segment {segment.id}: Gas velocity {velocity:.1f} ft/s exceeds maximum {max_velocity} ft/s"
                )
        
        return validation_results
    
    def calculate_gas_velocity(self, segment: PipeSegment) -> float:
        """Calculate gas velocity in pipe segment"""
        
        # Pipe internal area (square inches)
        schedule_40_ids = {
            0.5: 0.622, 0.75: 0.824, 1.0: 1.049, 1.25: 1.380, 1.5: 1.610,
            2.0: 2.067, 2.5: 2.469, 3.0: 3.068, 4.0: 4.026, 6.0: 6.065, 8.0: 7.981
        }
        
        pipe_id = schedule_40_ids.get(segment.nominal_size_inches, segment.nominal_size_inches * 0.82)
        area_sqin = math.pi * (pipe_id ** 2) / 4
        area_sqft = area_sqin / 144
        
        # Average pressure and temperature
        avg_pressure_psig = (segment.start_pressure_psig + segment.end_pressure_psig) / 2
        avg_pressure_psia = avg_pressure_psig + 14.7  # Add atmospheric pressure
        temperature_rankine = 520  # 60°F
        
        # Specific volume of natural gas
        specific_volume = 10.6 * temperature_rankine / avg_pressure_psia  # ft³/lb
        
        # Mass flow rate
        mass_flow_rate = segment.flow_rate_cfph / specific_volume  # lb/hr
        
        # Convert to ft³/s
        volumetric_flow_cfs = segment.flow_rate_cfph / 3600  # CFH to ft³/s
        
        # Velocity (ft/s)
        velocity = volumetric_flow_cfs / area_sqft
        
        return velocity
    
    def generate_system_report(self, system: PipingSystem) -> str:
        """Generate comprehensive piping system report"""
        
        report = f"""
PIPING SYSTEM DESIGN REPORT
===========================

System Information:
- Name: {system.name}
- Standard: {system.standard.value.upper()}
- Material: {system.material.value.replace('_', ' ').title()}
- Supply Pressure: {system.supply_pressure_psig:.2f} PSIG
- Appliance Pressure: {system.appliance_pressure_psig:.2f} PSIG
- Total Pressure Drop: {system.pressure_drop_psig:.3f} PSIG
- Total Equivalent Length: {system.total_equivalent_length_ft:.1f} ft

Segment Analysis:
"""
        
        for segment in system.segments:
            equiv_length = self.calculate_equivalent_length(segment, system.standard)
            velocity = self.calculate_gas_velocity(segment)
            
            report += f"""
{segment.id.title()}:
  - Size: {segment.nominal_size_inches}" {segment.material.value}
  - Length: {segment.length_ft} ft (Equivalent: {equiv_length:.1f} ft)
  - Flow: {segment.flow_rate_cfph:.0f} CFH ({segment.flow_rate_cfph * 1000:,.0f} BTU/hr)
  - Pressures: {segment.start_pressure_psig:.2f} → {segment.end_pressure_psig:.2f} PSIG
  - Pressure Drop: {segment.start_pressure_psig - segment.end_pressure_psig:.3f} PSIG
  - Velocity: {velocity:.1f} ft/s
  - Fittings: {dict(segment.fitting_count)}
"""
        
        # Compliance check
        validation = self.validate_system_compliance(system)
        report += f"""
Compliance Status: {'✅ COMPLIANT' if validation['compliant'] else '❌ NON-COMPLIANT'}
"""
        
        if validation['issues']:
            report += "Issues:\n"
            for issue in validation['issues']:
                report += f"  - {issue}\n"
        
        return report

# Demo functions
def demo_piping_system_designer():
    """Demonstrate piping system designer functionality"""
    designer = PipingSystemDesigner()
    
    # Design a simple residential system
    appliance_loads = [80000, 40000, 30000]  # BTU/hr for furnace, water heater, stove
    distances = [50, 30, 20]  # Distances from meter
    
    system = designer.design_piping_system(
        supply_pressure_psig=7.0,  # 7" WC
        appliance_loads_btuhr=appliance_loads,
        appliance_distances_ft=distances,
        standard=PipeStandard.NFPA_54,
        material=PipeMaterial.BLACK_STEEL
    )
    
    print("Piping System Designer Demo:")
    print(f"System Pressure Drop: {system.pressure_drop_psig:.3f} PSIG")
    print(f"Main Pipe Size: {system.segments[0].nominal_size_inches} inches")
    
    validation = designer.validate_system_compliance(system)
    print(f"Compliance: {'PASS' if validation['compliant'] else 'FAIL'}")

if __name__ == "__main__":
    demo_piping_system_designer()
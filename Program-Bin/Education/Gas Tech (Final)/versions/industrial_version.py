"""
Gas Tech Suite - Industrial Version
Large-scale systems and commercial applications
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid
import math

# Import shared components without modification
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.gas_physics_engine import GasPhysicsEngine
from gui.sleek_gui_framework import SleekGUIFramework

class IndustrialVersion:
    """Industrial version for large-scale gas systems and commercial applications"""
    
    def __init__(self):
        self.physics_engine = GasPhysicsEngine()
        self.gui_framework = SleekGUIFramework()
        self.industrial_systems = {}
        self.safety_management = {}
        self.maintenance_schedules = {}
        
    def initialize_industrial_tools(self):
        """Initialize industrial-grade toolkit"""
        return {
            'industrial_piping_design': self.industrial_piping_designer,
            'large_hvac_systems': self.large_hvac_analyzer,
            'safety_management': self.safety_management_system,
            'industrial_compliance': self.industrial_compliance_checker,
            'maintenance_optimizer': self.maintenance_optimizer,
            'energy_analyzer': self.energy_efficiency_analyzer
        }
    
    def industrial_piping_designer(self, design_request: Dict) -> Dict:
        """Industrial-scale piping system design"""
        try:
            project_type = design_request.get('project_type', 'commercial')
            
            if project_type == 'commercial_building':
                return self._design_commercial_building_piping(design_request)
            
            elif project_type == 'industrial_plant':
                return self._design_industrial_plant_piping(design_request)
            
            elif project_type == 'hospitality':
                return self._design_hospitality_piping(design_request)
            
            elif project_type == 'healthcare':
                return self._design_healthcare_piping(design_request)
            
            else:
                return self._design_general_industrial_piping(design_request)
                
        except Exception as e:
            return {'error': f'Industrial piping design failed: {str(e)}'}
    
    def _design_commercial_building_piping(self, request: Dict) -> Dict:
        """Design for commercial buildings (offices, retail, etc.)"""
        building_info = request.get('building_info', {})
        
        # Calculate building gas load
        total_gas_load = 0
        appliance_loads = {}
        
        # Typical commercial appliances
        if 'appliances' in building_info:
            for appliance, count in building_info['appliances'].items():
                if appliance == 'furnaces':
                    load_per_unit = 120000  # BTU/hr
                elif appliance == 'boilers':
                    load_per_unit = 400000  # BTU/hr
                elif appliance == 'water_heaters':
                    load_per_unit = 200000  # BTU/hr
                elif appliance == 'cooking_equipment':
                    load_per_unit = 100000  # BTU/hr
                elif appliance == 'dryers':
                    load_per_unit = 35000  # BTU/hr
                else:
                    load_per_unit = 50000  # BTU/hr default
                
                total_appliance_load = load_per_unit * count
                appliance_loads[appliance] = total_appliance_load
                total_gas_load += total_appliance_load
        
        # Diversity factor for commercial buildings
        diversity_factor = 0.8  # 80% of maximum load expected
        design_load = total_gas_load * diversity_factor
        
        # Main line sizing
        main_pipe_size = self._calculate_main_pipe_size(design_load, building_info)
        
        # Branch line calculations
        branch_lines = self._design_branch_lines(appliance_loads, building_info)
        
        # Pressure drop calculations
        pressure_analysis = self._analyze_pressure_drops(main_pipe_size, branch_lines, building_info)
        
        design_result = {
            'project_id': str(uuid.uuid4()),
            'building_type': 'commercial',
            'timestamp': datetime.now().isoformat(),
            'load_analysis': {
                'total_connected_load': total_gas_load,
                'diversity_factor': diversity_factor,
                'design_load': design_load,
                'appliance_breakdown': appliance_loads
            },
            'piping_design': {
                'main_line': main_pipe_size,
                'branch_lines': branch_lines,
                'material_recommendation': 'Schedule 40 Black Steel',
                'support_spacing': self._calculate_support_spacing(main_pipe_size['diameter'])
            },
            'pressure_analysis': pressure_analysis,
            'safety_factors': {
                'flow_capacity': 1.25,  # 25% extra capacity
                'pressure_drop_limit': 3.0,  # inches w.c. for commercial
                'safety_factor': 1.5
            },
            'compliance_standards': ['IFGC', 'NFPA 54', 'CSA B149.1'],
            'recommendations': [
                'Install pressure regulators at each appliance',
                'Include isolation valves for maintenance',
                'Consider future expansion in main line sizing',
                'Install gas detection system in mechanical rooms'
            ]
        }
        
        return {
            'success': True,
            'design_type': 'commercial_building',
            'design_result': design_result
        }
    
    def _design_industrial_plant_piping(self, request: Dict) -> Dict:
        """Design for industrial plants and manufacturing facilities"""
        plant_info = request.get('plant_info', {})
        
        # Industrial gas load calculation
        process_equipment = plant_info.get('process_equipment', {})
        total_process_load = 0
        
        for equipment, specs in process_equipment.items():
            if isinstance(specs, dict):
                load = specs.get('gas_load', 100000)  # BTU/hr
                quantity = specs.get('quantity', 1)
                total_process_load += load * quantity
        
        # Support systems load
        support_loads = {
            'space_heating': plant_info.get('heating_load', 0),
            'water_heating': plant_info.get('water_heating_load', 0),
            'backup_generators': plant_info.get('generator_load', 0),
            'process_heating': plant_info.get('process_heating_load', 0)
        }
        
        total_support_load = sum(support_loads.values())
        total_plant_load = total_process_load + total_support_load
        
        # Industrial main line sizing
        main_line = self._calculate_industrial_main_line(total_plant_load, plant_info)
        
        # Distribution network design
        distribution_network = self._design_distribution_network(total_process_load, total_support_load, plant_info)
        
        # Safety systems design
        safety_systems = self._design_industrial_safety_systems(plant_info)
        
        plant_design = {
            'project_id': str(uuid.uuid4()),
            'facility_type': 'industrial_plant',
            'timestamp': datetime.now().isoformat(),
            'load_analysis': {
                'total_process_load': total_process_load,
                'total_support_load': total_support_load,
                'total_plant_load': total_plant_load,
                'process_equipment_breakdown': process_equipment,
                'support_systems_breakdown': support_loads
            },
            'piping_system': {
                'main_line': main_line,
                'distribution_network': distribution_network,
                'material_specification': 'ASTM A53 Grade B Schedule 40',
                'coating': 'Epoxy coating for corrosion protection',
                'support_system': self._design_pipe_supports(main_line['diameter'])
            },
            'safety_systems': safety_systems,
            'control_systems': {
                'pressure_monitoring': 'Automated with SCADA integration',
                'flow_metering': 'Ultrasonic flow meters',
                'leak_detection': 'Fixed gas detectors with alarms',
                'emergency_shutdown': 'Automatic ESD valves'
            },
            'compliance_standards': ['ASME B31.8', 'NFPA 54', 'OSHA 1910.103', 'IFGC'],
            'industrial_recommendations': [
                'Install redundant gas meters for critical processes',
                'Consider underground piping with cathodic protection',
                'Implement predictive maintenance program',
                'Design for 24/7 operation with maintenance access'
            ]
        }
        
        return {
            'success': True,
            'design_type': 'industrial_plant',
            'design_result': plant_design
        }
    
    def large_hvac_analyzer(self, hvac_request: Dict) -> Dict:
        """Large-scale HVAC system analysis"""
        try:
            building_type = hvac_request.get('building_type', 'commercial')
            
            if building_type == 'hospital':
                return self._analyze_hospital_hvac(hvac_request)
            
            elif building_type == 'data_center':
                return self._analyze_data_center_hvac(hvac_request)
            
            elif building_type == 'manufacturing':
                return self._analyze_manufacturing_hvac(hvac_request)
            
            else:
                return self._analyze_commercial_hvac(hvac_request)
                
        except Exception as e:
            return {'error': f'Large HVAC analysis failed: {str(e)}'}
    
    def _analyze_hospital_hvac(self, request: Dict) -> Dict:
        """Hospital HVAC system analysis with special requirements"""
        hospital_info = request.get('hospital_info', {})
        
        # Hospital-specific requirements
        areas = hospital_info.get('areas', {})
        total_heating_load = 0
        total_cooling_load = 0
        
        area_requirements = {
            'patient_rooms': {
                'heating_per_sqft': 25,
                'cooling_per_sqft': 20,
                'air_changes': 6
            },
            'operating_rooms': {
                'heating_per_sqft': 30,
                'cooling_per_sqft': 40,
                'air_changes': 25
            },
            'icu': {
                'heating_per_sqft': 30,
                'cooling_per_sqft': 30,
                'air_changes': 12
            },
            'laboratories': {
                'heating_per_sqft': 35,
                'cooling_per_sqft': 50,
                'air_changes': 15
            }
        }
        
        area_analysis = {}
        
        for area_name, area_data in areas.items():
            area_type = area_data.get('type', 'general')
            square_footage = area_data.get('sqft', 0)
            
            if area_type in area_requirements:
                requirements = area_requirements[area_type]
                heating_load = requirements['heating_per_sqft'] * square_footage
                cooling_load = requirements['cooling_per_sqft'] * square_footage
                air_changes = requirements['air_changes']
                
                area_analysis[area_name] = {
                    'type': area_type,
                    'sqft': square_footage,
                    'heating_load': heating_load,
                    'cooling_load': cooling_load,
                    'air_changes_required': air_changes,
                    'gas_heating_load': heating_load * 0.8  # 80% gas heating
                }
                
                total_heating_load += heating_load
                total_cooling_load += cooling_load
        
        # Convert to BTU/hr for gas sizing
        total_gas_load = total_heating_load * 0.8  # Assuming 80% gas heating
        
        # Boiler system sizing
        boiler_capacity = total_gas_load * 1.3  # 30% redundancy for hospitals
        
        # Gas piping for HVAC systems
        hvac_piping = self._size_hvac_gas_piping(total_gas_load, hospital_info)
        
        hospital_hvac_analysis = {
            'analysis_id': str(uuid.uuid4()),
            'facility_type': 'hospital',
            'timestamp': datetime.now().isoformat(),
            'load_analysis': {
                'total_heating_load': total_heating_load,
                'total_cooling_load': total_cooling_load,
                'total_gas_load': total_gas_load,
                'area_breakdown': area_analysis
            },
            'boiler_system': {
                'required_capacity': boiler_capacity,
                'recommended_boilers': [
                    {'capacity': boiler_capacity / 2, 'type': 'Condensing', 'quantity': 2}
                ],
                'redundancy': 'N+1 configuration required',
                'efficiency_rating': '95%+ condensing boilers recommended'
            },
            'gas_piping': hvac_piping,
            'hospital_requirements': {
                'redundancy': 'Critical - backup systems required',
                'air_quality': 'HEPA filtration in critical areas',
                'pressurization': 'Positive pressure in operating rooms',
                'isolation': 'Separate systems for different zones'
            },
            'compliance_standards': ['ASHRAE 170', 'NFPA 99', 'FGI Guidelines', 'IFGC'],
            'recommendations': [
                'Install modulating boilers for efficiency',
                'Implement building automation system',
                'Design for easy maintenance access',
                'Include comprehensive monitoring systems'
            ]
        }
        
        return {
            'success': True,
            'hvac_type': 'hospital',
            'analysis_result': hospital_hvac_analysis
        }
    
    def safety_management_system(self, safety_request: Dict) -> Dict:
        """Industrial safety management system"""
        try:
            safety_program_type = safety_request.get('program_type', 'comprehensive')
            
            if safety_program_type == 'comprehensive':
                return self._create_comprehensive_safety_program(safety_request)
            
            elif safety_program_type == 'gas_detection':
                return self._design_gas_detection_system(safety_request)
            
            elif safety_program_type == 'emergency_response':
                return self._create_emergency_response_plan(safety_request)
            
            else:
                return self._create_basic_safety_program(safety_request)
                
        except Exception as e:
            return {'error': f'Safety management system failed: {str(e)}'}
    
    def _create_comprehensive_safety_program(self, request: Dict) -> Dict:
        """Create comprehensive industrial safety program"""
        facility_info = request.get('facility_info', {})
        
        safety_program = {
            'program_id': str(uuid.uuid4()),
            'facility_name': facility_info.get('name', 'Industrial Facility'),
            'timestamp': datetime.now().isoformat(),
            'safety_components': {
                'gas_detection_system': {
                    'detector_types': [
                        'Catalytic bead (combustible gases)',
                        'Infrared (hydrocarbon gases)',
                        'Electrochemical (toxic gases)',
                        'Photoionization (VOCs)'
                    ],
                    'placement_strategy': 'Perimeter and process area coverage',
                    'alarm_levels': ['Low (10% LEL)', 'High (25% LEL)', 'Critical (50% LEL)'],
                    'integration': 'Building automation and emergency shutdown'
                },
                'emergency_shutdown_system': {
                    'esd_valves': 'Automated with manual override',
                    'trigger_points': 'Gas detection, overpressure, fire alarm',
                    'response_time': '< 2 seconds',
                    'backup_power': 'UPS and generator'
                },
                'ventilation_system': {
                    'normal_ventilation': '10-15 air changes per hour',
                    'emergency_ventilation': '20-30 air changes per hour',
                    'makeup_air': '100% outside air with filtration',
                    'explosion_proof': 'Class 1 Division 2 equipment'
                },
                'fire_suppression': {
                    'suppression_type': 'Water mist and CO2 systems',
                    'coverage': 'All gas handling areas',
                    'activation': 'Automatic with manual stations',
                    'integration': 'Fire alarm and ESD systems'
                }
            },
            'procedures': {
                'daily_safety_checks': [
                    'Gas leak inspection',
                    'Pressure gauge verification',
                    'Ventilation system check',
                    'Detection system test'
                ],
                'monthly_inspections': [
                    'Complete piping system inspection',
                    'Valve operation testing',
                    'Calibration of gas detectors',
                    'Emergency drill'
                ],
                'annual_maintenance': [
                    'Comprehensive system overhaul',
                    'Piping integrity testing',
                    'Safety valve testing',
                    'Training program review'
                ]
            },
            'training_program': {
                'operator_training': '40 hours initial, 8 hours annual',
                'maintenance_training': '24 hours initial, 16 hours annual',
                'emergency_response': 'All personnel annual drills',
                'management_training': 'Supervisory responsibilities'
            },
            'documentation': {
                'required_documents': [
                    'Process Safety Information (PSI)',
                    'Process Hazard Analysis (PHA)',
                    'Operating Procedures',
                    'Mechanical Integrity Program',
                    'Management of Change (MOC)',
                    'Emergency Response Plan'
                ],
                'record_keeping': 'Digital and physical copies, 5-year retention'
            },
            'compliance_standards': ['OSHA 1910.119', 'NFPA 54', 'ASME B31.8', 'IFGC'],
            'implementation_timeline': {
                'phase_1': 'Basic safety systems (3 months)',
                'phase_2': 'Advanced detection and control (6 months)',
                'phase_3': 'Full integration and training (9 months)',
                'phase_4': 'Verification and certification (12 months)'
            }
        }
        
        return {
            'success': True,
            'program_type': 'comprehensive',
            'safety_program': safety_program
        }
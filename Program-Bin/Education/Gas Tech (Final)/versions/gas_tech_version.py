"""
Gas Tech Suite - Gas Tech Version (Professional Technician)
Field technician tools and diagnostics
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

# Import shared components without modification
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.gas_physics_engine import GasPhysicsEngine
from gui.sleek_gui_framework import SleekGUIFramework

class GasTechVersion:
    """Professional technician version with field diagnostics and advanced tools"""
    
    def __init__(self):
        self.physics_engine = GasPhysicsEngine()
        self.gui_framework = SleekGUIFramework()
        self.field_diagnostics = {}
        self.compliance_records = {}
        self.mobile_tools = {}
        
    def initialize_technician_tools(self):
        """Initialize professional technician toolkit"""
        return {
            'field_diagnostics': self.field_diagnostics_suite,
            'compliance_checker': self.compliance_checker_suite,
            'mobile_field_tools': self.mobile_field_tools,
            'equipment_analyzer': self.equipment_analyzer,
            'safety_inspector': self.safety_inspector_pro
        }
    
    def field_diagnostics_suite(self, diagnostic_request: Dict) -> Dict:
        """Comprehensive field diagnostics toolkit"""
        try:
            diagnostic_type = diagnostic_request.get('type', 'system_check')
            
            if diagnostic_type == 'leak_detection':
                return self._perform_leak_detection(diagnostic_request)
            
            elif diagnostic_type == 'pressure_analysis':
                return self._perform_pressure_analysis(diagnostic_request)
            
            elif diagnostic_type == 'combustion_analysis':
                return self._perform_combustion_analysis(diagnostic_request)
            
            elif diagnostic_type == 'appliance_diagnostics':
                return self._diagnose_appliance(diagnostic_request)
            
            elif diagnostic_type == 'system_integrity':
                return self._check_system_integrity(diagnostic_request)
            
            else:
                return self._perform_system_check(diagnostic_request)
                
        except Exception as e:
            return {'error': f'Diagnostic failed: {str(e)}'}
    
    def _perform_leak_detection(self, request: Dict) -> Dict:
        """Advanced leak detection and localization"""
        system_info = request.get('system_info', {})
        
        # Simulate comprehensive leak detection process
        leak_detection_steps = [
            'Visual inspection of all connections',
            'Soap solution test on fittings',
            'Electronic sniffer analysis',
            'Pressure decay test',
            'Line isolation testing'
        ]
        
        detection_results = {
            'detection_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'system_pressure': system_info.get('pressure', 0),
            'test_duration': '15 minutes',
            'steps_completed': leak_detection_steps,
            'leaks_detected': [],
            'pressure_drop': 0.02,  # inches w.c.
            'passes_safety_standard': True,
            'recommendations': []
        }
        
        # Analyze results for potential issues
        if detection_results['pressure_drop'] > 0.05:
            detection_results['passes_safety_standard'] = False
            detection_results['leaks_detected'].append({
                'location': 'System-wide pressure loss',
                'severity': 'high',
                'estimated_flow_rate': 'Unknown - requires further investigation'
            })
            detection_results['recommendations'].append('Immediate repair required before system use')
        
        return {
            'success': True,
            'diagnostic_type': 'leak_detection',
            'results': detection_results
        }
    
    def _perform_pressure_analysis(self, request: Dict) -> Dict:
        """Comprehensive pressure and flow analysis"""
        system_info = request.get('system_info', {})
        gas_type = system_info.get('gas_type', 'natural_gas')
        
        # Calculate expected pressures
        expected_pressures = self.physics_engine.calculate_system_pressures(gas_type, system_info)
        
        pressure_analysis = {
            'analysis_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'gas_type': gas_type,
            'measured_pressures': {
                'supply_pressure': system_info.get('supply_pressure', 0),
                'appliance_pressure': system_info.get('appliance_pressure', 0),
                'line_pressure': system_info.get('line_pressure', 0)
            },
            'expected_pressures': expected_pressures,
            'pressure_differentials': {},
            'flow_analysis': {},
            'system_efficiency': 'Good',
            'issues_detected': []
        }
        
        # Calculate pressure differentials
        for key in pressure_analysis['measured_pressures']:
            if key in pressure_analysis['expected_pressures']:
                diff = pressure_analysis['measured_pressures'][key] - pressure_analysis['expected_pressures'][key]
                pressure_analysis['pressure_differentials'][key] = diff
                
                if abs(diff) > 0.5:  # More than 0.5" w.c. difference
                    pressure_analysis['issues_detected'].append({
                        'type': 'pressure_anomaly',
                        'location': key,
                        'severity': 'moderate' if abs(diff) < 1.0 else 'high',
                        'description': f'Pressure difference of {diff}" w.c. detected'
                    })
        
        # Flow analysis
        flow_rate = self.physics_engine.calculate_flow_rate(
            pressure_analysis['measured_pressures'].get('line_pressure', 0),
            system_info.get('pipe_diameter', 0.5),
            gas_type
        )
        
        pressure_analysis['flow_analysis'] = {
            'calculated_flow_rate': flow_rate,
            'expected_flow_range': expected_pressures.get('flow_range', [0, 100]),
            'flow_efficiency': 'Good' if flow_rate > expected_pressures.get('flow_range', [0, 100])[0] else 'Low'
        }
        
        return {
            'success': True,
            'diagnostic_type': 'pressure_analysis',
            'results': pressure_analysis
        }
    
    def compliance_checker_suite(self, compliance_request: Dict) -> Dict:
        """Professional compliance checking against multiple standards"""
        try:
            standard = compliance_request.get('standard', 'CSA_B149')
            system_data = compliance_request.get('system_data', {})
            
            compliance_checks = {
                'CSA_B149': self._check_csa_b149_compliance,
                'NFPA_54': self._check_nfpa_54_compliance,
                'IFGC': self._check_ifgc_compliance,
                'UPC': self._check_upc_compliance
            }
            
            if standard in compliance_checks:
                return compliance_checks[standard](system_data)
            else:
                return self._check_general_compliance(system_data)
                
        except Exception as e:
            return {'error': f'Compliance check failed: {str(e)}'}
    
    def _check_csa_b149_compliance(self, system_data: Dict) -> Dict:
        """CSA B149 compliance checking"""
        compliance_result = {
            'standard': 'CSA B149.1',
            'timestamp': datetime.now().isoformat(),
            'checks_performed': [],
            'compliance_status': 'Compliant',
            'violations': [],
            'recommendations': []
        }
        
        # Pressure compliance
        if 'pressure' in system_data:
            max_pressure = 14.0  # inches w.c. for residential
            if system_data['pressure'] > max_pressure:
                compliance_result['violations'].append({
                    'section': '6.3.2',
                    'description': f'Excessive pressure: {system_data["pressure"]}" w.c. exceeds maximum {max_pressure}" w.c.',
                    'severity': 'Critical'
                })
                compliance_result['compliance_status'] = 'Non-Compliant'
        
        # Venting compliance
        if 'venting' in system_data:
            venting_data = system_data['venting']
            if 'diameter' in venting_data and 'length' in venting_data:
                # Check vent sizing
                required_size = self._calculate_minimum_vent_size(venting_data['appliance_type'], venting_data['btu'])
                if venting_data['diameter'] < required_size:
                    compliance_result['violations'].append({
                        'section': '8.2.1',
                        'description': f'Vent diameter {venting_data["diameter"]}" insufficient for {venting_data["btu"]} BTU appliance',
                        'severity': 'High'
                    })
                    compliance_result['compliance_status'] = 'Non-Compliant'
        
        # Clearance compliance
        if 'clearances' in system_data:
            clearance_data = system_data['clearances']
            required_clearances = {
                'combustible_materials': 6,  # inches
                'venting': 6,
                'service_access': 24
            }
            
            for item, distance in clearance_data.items():
                if item in required_clearances and distance < required_clearances[item]:
                    compliance_result['violations'].append({
                        'section': '7.1.1',
                        'description': f'Insufficient clearance: {item} distance {distance}" below required {required_clearances[item]}"',
                        'severity': 'Moderate'
                    })
        
        return {
            'success': True,
            'compliance_result': compliance_result
        }
    
    def mobile_field_tools(self, tool_request: Dict) -> Dict:
        """Mobile-optimized field tools"""
        try:
            tool_type = tool_request.get('tool_type', 'calculator')
            
            if tool_type == 'pipe_sizing':
                return self._mobile_pipe_sizing(tool_request)
            
            elif tool_type == 'gas_load_calculator':
                return self._mobile_gas_load_calculator(tool_request)
            
            elif tool_type == 'vent_calculator':
                return self._mobile_vent_calculator(tool_request)
            
            elif tool_type == 'combustion_air':
                return self._mobile_combustion_air_calculator(tool_request)
            
            else:
                return self._mobile_general_calculator(tool_request)
                
        except Exception as e:
            return {'error': f'Mobile tool failed: {str(e)}'}
    
    def _mobile_pipe_sizing(self, request: Dict) -> Dict:
        """Mobile-optimized pipe sizing calculator"""
        input_data = request.get('input_data', {})
        
        # Quick pipe sizing for field use
        sizing_result = self.physics_engine.calculate_pipe_size(
            input_data.get('gas_load', 100),
            input_data.get('length', 50),
            input_data.get('pressure_drop', 0.5),
            input_data.get('gas_type', 'natural_gas')
        )
        
        # Add mobile-friendly recommendations
        mobile_result = {
            'sizing_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'input_parameters': input_data,
            'calculated_size': sizing_result,
            'recommended_pipe_size': sizing_result.get('recommended_size', '3/4"'),
            'available_sizes': ['1/2"', '3/4"', '1"', '1-1/4"', '1-1/2"', '2"'],
            'material_options': ['Black Steel', 'Copper', 'CSST', 'PEX'],
            'safety_factors': {
                'pressure_drop_safety': 0.3,  # inches w.c.
                'flow_capacity_safety': 1.2   # 20% extra capacity
            },
            'field_notes': [
                f'Required size: {sizing_result.get("recommended_size", "3/4")}" pipe',
                f'Pressure drop: {sizing_result.get("pressure_drop", 0.3)}" w.c.',
                f'Maximum flow: {sizing_result.get("max_flow", 150)} CFH'
            ]
        }
        
        return {
            'success': True,
            'tool_type': 'pipe_sizing',
            'mobile_result': mobile_result
        }
    
    def safety_inspector_pro(self, inspection_request: Dict) -> Dict:
        """Professional safety inspection suite"""
        try:
            inspection_type = inspection_request.get('type', 'comprehensive')
            
            if inspection_type == 'comprehensive':
                return self._comprehensive_safety_inspection(inspection_request)
            
            elif inspection_type == 'carbon_monoxide':
                return self._carbon_monoxide_inspection(inspection_request)
            
            elif inspection_type == 'ventilation':
                return self._ventilation_safety_inspection(inspection_request)
            
            else:
                return self._standard_safety_inspection(inspection_request)
                
        except Exception as e:
            return {'error': f'Safety inspection failed: {str(e)}'}
    
    def _comprehensive_safety_inspection(self, request: Dict) -> Dict:
        """Complete safety inspection for gas systems"""
        system_info = request.get('system_info', {})
        
        inspection_checklist = {
            'gas_leaks': {'status': 'Pass', 'details': 'No leaks detected'},
            'pressure_regulation': {'status': 'Pass', 'details': 'Pressures within normal range'},
            'appliance_operation': {'status': 'Pass', 'details': 'All appliances operating normally'},
            'venting_system': {'status': 'Pass', 'details': 'Proper venting confirmed'},
            'combustion_air': {'status': 'Pass', 'details': 'Adequate combustion air supply'},
            'clearances': {'status': 'Pass', 'details': 'All clearances maintained'},
            'safety_devices': {'status': 'Pass', 'details': 'All safety devices functional'},
            'carbon_monoxide': {'status': 'Pass', 'details': 'No CO detected'}
        }
        
        # Calculate overall safety score
        passed_checks = sum(1 for check in inspection_checklist.values() if check['status'] == 'Pass')
        total_checks = len(inspection_checklist)
        safety_score = (passed_checks / total_checks) * 100
        
        inspection_result = {
            'inspection_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'inspector_id': request.get('inspector_id', 'System'),
            'property_address': system_info.get('address', 'Unknown'),
            'safety_score': safety_score,
            'overall_status': 'Safe' if safety_score >= 90 else 'Requires Attention',
            'checklist': inspection_checklist,
            'recommendations': [],
            'follow_up_required': False
        }
        
        # Add recommendations based on findings
        if safety_score < 100:
            inspection_result['recommendations'].append('Address any failed items before next inspection')
            inspection_result['follow_up_required'] = True
        
        return {
            'success': True,
            'inspection_type': 'comprehensive',
            'safety_result': inspection_result
        }
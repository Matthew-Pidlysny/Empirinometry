"""
Gas Tech Suite - Office Version
Administrative management and business operations tools
"""

import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid

# Import shared components without modification
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.gas_physics_engine import GasPhysicsEngine
from gui.sleek_gui_framework import SleekGUIFramework

class OfficeVersion:
    """Office version for administrative management and business operations"""
    
    def __init__(self):
        self.physics_engine = GasPhysicsEngine()
        self.gui_framework = SleekGUIFramework()
        self.customer_database = {}
        self.schedule_data = {}
        self.invoice_data = {}
        self.employee_records = {}
        
    def initialize_office_management(self):
        """Initialize office management toolkit"""
        return {
            'customer_manager': self.customer_manager,
            'appointment_scheduler': self.appointment_scheduler,
            'invoice_generator': self.invoice_generator,
            'inventory_manager': self.inventory_manager,
            'report_analyzer': self.report_analyzer
        }
    
    def customer_manager(self, customer_data: Dict) -> Dict:
        """Comprehensive customer relationship management"""
        try:
            customer_id = customer_data.get('customer_id', str(uuid.uuid4()))
            action = customer_data.get('action', 'create')
            
            if action == 'create':
                # Create new customer record
                customer_record = {
                    'customer_id': customer_id,
                    'name': customer_data.get('name', ''),
                    'contact': {
                        'phone': customer_data.get('phone', ''),
                        'email': customer_data.get('email', ''),
                        'address': customer_data.get('address', '')
                    },
                    'service_history': [],
                    'properties': [],
                    'preferred_contact': customer_data.get('preferred_contact', 'phone'),
                    'customer_since': datetime.now().strftime('%Y-%m-%d'),
                    'status': 'active',
                    'notes': []
                }
                self.customer_database[customer_id] = customer_record
                return {'success': True, 'customer_id': customer_id, 'message': 'Customer created successfully'}
            
            elif action == 'update':
                # Update existing customer
                customer_id = customer_data.get('customer_id')
                if customer_id in self.customer_database:
                    updates = customer_data.get('updates', {})
                    for key, value in updates.items():
                        if key == 'contact':
                            self.customer_database[customer_id]['contact'].update(value)
                        else:
                            self.customer_database[customer_id][key] = value
                    return {'success': True, 'message': 'Customer updated successfully'}
                else:
                    return {'success': False, 'message': 'Customer not found'}
            
            elif action == 'retrieve':
                # Retrieve customer information
                customer_id = customer_data.get('customer_id')
                if customer_id in self.customer_database:
                    return {'success': True, 'customer': self.customer_database[customer_id]}
                else:
                    return {'success': False, 'message': 'Customer not found'}
            
            else:
                return {'success': False, 'message': 'Invalid action'}
                
        except Exception as e:
            return {'error': f'Customer management failed: {str(e)}'}
    
    def appointment_scheduler(self, schedule_data: Dict) -> Dict:
        """Advanced appointment scheduling system"""
        try:
            action = schedule_data.get('action', 'create')
            
            if action == 'create':
                appointment_id = str(uuid.uuid4())
                appointment = {
                    'appointment_id': appointment_id,
                    'customer_id': schedule_data.get('customer_id'),
                    'technician_id': schedule_data.get('technician_id'),
                    'date': schedule_data.get('date'),
                    'time': schedule_data.get('time'),
                    'duration': schedule_data.get('duration', 2),  # hours
                    'service_type': schedule_data.get('service_type', 'inspection'),
                    'priority': schedule_data.get('priority', 'normal'),
                    'location': schedule_data.get('location', ''),
                    'notes': schedule_data.get('notes', ''),
                    'status': 'scheduled',
                    'created_date': datetime.now().strftime('%Y-%m-%d')
                }
                
                # Check for conflicts
                conflicts = self._check_schedule_conflicts(appointment)
                if conflicts:
                    return {'success': False, 'message': 'Schedule conflict detected', 'conflicts': conflicts}
                
                self.schedule_data[appointment_id] = appointment
                return {'success': True, 'appointment_id': appointment_id, 'message': 'Appointment scheduled successfully'}
            
            elif action == 'update':
                appointment_id = schedule_data.get('appointment_id')
                if appointment_id in self.schedule_data:
                    updates = schedule_data.get('updates', {})
                    self.schedule_data[appointment_id].update(updates)
                    return {'success': True, 'message': 'Appointment updated successfully'}
                else:
                    return {'success': False, 'message': 'Appointment not found'}
            
            elif action == 'list':
                # List appointments with filters
                filters = schedule_data.get('filters', {})
                filtered_appointments = []
                
                for appointment in self.schedule_data.values():
                    match = True
                    if 'date_range' in filters:
                        start_date = filters['date_range'].get('start')
                        end_date = filters['date_range'].get('end')
                        if start_date and appointment['date'] >= start_date:
                                pass
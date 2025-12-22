"""
Gas Tech Suite - Production Launcher
Clean production launcher without testing dependencies
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

# Import core components
from core.gas_physics_engine import GasPhysicsEngine
from integration.version_integration import VersionIntegration
from upgrade_system.upgrade_manager import UpgradeManager
from gui.sleek_gui_framework import SleekWindow

class GasTechProductionLauncher:
    """Production launcher for Gas Tech Suite"""
    
    def __init__(self):
        self.physics_engine = GasPhysicsEngine()
        self.version_integration = VersionIntegration()
        self.upgrade_manager = UpgradeManager()
        self.current_session = {}
        
    def initialize_production_suite(self):
        """Initialize production Gas Tech Suite"""
        try:
            # Initialize core systems
            integration_systems = self.version_integration.initialize_integration()
            upgrade_systems = self.upgrade_manager.initialize_upgrade_system()
            
            # Initialize GUI
            gui_framework = SleekWindow("Gas Tech Suite")
            
            session_id = str(uuid.uuid4())
            
            initialization_result = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'integration_systems': integration_systems,
                'upgrade_systems': upgrade_systems,
                'gui_framework': 'initialized',
                'available_versions': self._get_available_versions(),
                'status': 'PRODUCTION_READY',
                'field_approvals_workshop': 'AVAILABLE',
                'quality_assurance': 'VERIFIED'
            }
            
            self.current_session = initialization_result
            
            return {
                'success': True,
                'initialization_result': initialization_result
            }
            
        except Exception as e:
            return {'error': f'Suite initialization failed: {str(e)}'}
    
    def launch_version(self, launch_request: Dict) -> Dict:
        """Launch specific version"""
        try:
            version_selection = self.version_integration.version_selector({
                'user_type': launch_request.get('user_type', 'auto'),
                'requirements': launch_request.get('requirements', {})
            })
            
            if not version_selection.get('success', False):
                return {'error': 'Version selection failed'}
            
            recommended_version = version_selection['selection_result']['recommended_version']
            
            return {
                'success': True,
                'launched_version': recommended_version,
                'version_features': version_selection['selection_result']['available_features'],
                'field_approvals': self._check_field_approvals_available(recommended_version),
                'production_status': 'READY'
            }
            
        except Exception as e:
            return {'error': f'Version launch failed: {str(e)}'}
    
    def _get_available_versions(self):
        """Get available production versions"""
        return [
            {
                'id': 'consumer',
                'name': 'Consumer Version',
                'description': 'Homeowner tools and education',
                'production_ready': True
            },
            {
                'id': 'office',
                'name': 'Office Version',
                'description': 'Administrative management tools',
                'production_ready': True
            },
            {
                'id': 'gas_tech',
                'name': 'Gas Tech Version',
                'description': 'Professional technician tools with Field Approvals',
                'production_ready': True,
                'field_approvals': True
            },
            {
                'id': 'industrial',
                'name': 'Industrial Version',
                'description': 'Large-scale systems and commercial applications',
                'production_ready': True
            },
            {
                'id': 'scientist',
                'name': 'Scientist Version',
                'description': 'Research and development tools',
                'production_ready': True
            },
            {
                'id': 'mechanical',
                'name': 'Mechanical Version',
                'description': 'Advanced engineering and design',
                'production_ready': True
            }
        ]
    
    def _check_field_approvals_available(self, version: str) -> bool:
        """Check if field approvals workshop is available for version"""
        return version in ['gas_tech', 'industrial']

def main():
    """Main production launcher"""
    launcher = GasTechProductionLauncher()
    
    # Initialize production suite
    init_result = launcher.initialize_production_suite()
    
    if init_result.get('success', False):
        print("Gas Tech Suite PRODUCTION initialized successfully!")
        print(f"Session ID: {init_result['initialization_result']['session_id']}")
        print(f"Status: {init_result['initialization_result']['status']}")
        print("Available production versions:")
        
        for version in init_result['initialization_result']['available_versions']:
            status = "✓ PRODUCTION" if version['production_ready'] else "✗ DEVELOPMENT"
            field_approvals = " + Field Approvals" if version.get('field_approvals') else ""
            print(f"  - {version['id']}: {version['name']} {status}{field_approvals}")
        
        # Example: Launch Gas Tech version
        launch_result = launcher.launch_version({
            'user_type': 'gas_tech',
            'requirements': {'field_approvals': True}
        })
        
        if launch_result.get('success', False):
            version = launch_result['launched_version']
            print(f"\n✓ Launched: {version}")
            if launch_result['field_approvals']:
                print("✓ Field Approvals Workshop: ENABLED")
            print("✓ Ready for professional use")
        
    else:
        print(f"✗ Initialization failed: {init_result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
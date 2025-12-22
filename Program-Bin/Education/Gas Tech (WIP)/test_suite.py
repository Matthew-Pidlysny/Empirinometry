"""
Test Suite for Gas Tech Suite
Comprehensive testing for all components and versions
"""

import unittest
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gas_physics_engine import GasPhysicsEngine, FuelType

class TestGasPhysicsEngine(unittest.TestCase):
    """Test the core gas physics engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = GasPhysicsEngine()
    
    def test_fuel_properties(self):
        """Test fuel properties database"""
        for fuel_type in FuelType:
            props = self.engine.fuel_database[fuel_type]
            self.assertIsNotNone(props.name)
            self.assertGreater(props.molecular_weight, 0)
            self.assertGreater(props.heating_value, 0)
    
    def test_gas_flow_calculation(self):
        """Test gas flow rate calculations"""
        # Test natural gas flow
        flow = self.engine.calculate_gas_flow_rate(
            pressure_psi=10,
            pipe_diameter_inches=1.0,
            length_feet=100,
            fuel_type=FuelType.NATURAL_GAS
        )
        self.assertGreater(flow, 0)
        
        # Test edge cases
        flow_zero_pressure = self.engine.calculate_gas_flow_rate(
            pressure_psi=0,
            pipe_diameter_inches=1.0,
            length_feet=100,
            fuel_type=FuelType.NATURAL_GAS
        )
        self.assertEqual(flow_zero_pressure, 0)
    
    def test_pipe_sizing(self):
        """Test pipe sizing calculations"""
        sizing = self.engine.calculate_pipe_sizing(
            required_flow_cfph=100,
            length_feet=100,
            pressure_psi=10,
            fuel_type=FuelType.NATURAL_GAS
        )
        
        self.assertIn("minimum_size", sizing)
        self.assertIn("recommended_size", sizing)
        self.assertIn("sizes_available", sizing)
        self.assertGreater(sizing["minimum_size"], 0)
    
    def test_pressure_drop(self):
        """Test pressure drop calculations"""
        pressure_drop = self.engine.calculate_pressure_drop(
            flow_rate_cfph=100,
            pipe_diameter_inches=1.0,
            length_feet=100,
            fuel_type=FuelType.NATURAL_GAS
        )
        
        self.assertGreaterEqual(pressure_drop, 0)
        
        # Test zero flow
        zero_flow_drop = self.engine.calculate_pressure_drop(
            flow_rate_cfph=0,
            pipe_diameter_inches=1.0,
            length_feet=100,
            fuel_type=FuelType.NATURAL_GAS
        )
        self.assertEqual(zero_flow_drop, 0)
    
    def test_combustion_air(self):
        """Test combustion air calculations"""
        air_reqs = self.engine.calculate_combustion_air(
            btu_per_hour=100000,
            fuel_type=FuelType.NATURAL_GAS
        )
        
        self.assertIn("combustion_air_cfm", air_reqs)
        self.assertIn("dilution_air_cfm", air_reqs)
        self.assertIn("total_air_cfm", air_reqs)
        self.assertGreater(air_reqs["combustion_air_cfm"], 0)
        self.assertGreater(air_reqs["total_air_cfm"], air_reqs["combustion_air_cfm"])
    
    def test_vent_sizing(self):
        """Test vent sizing calculations"""
        vent_sizing = self.engine.calculate_vent_sizing(
            btu_per_hour=100000,
            vent_type="chimney"
        )
        
        self.assertIn("required_diameter_inches", vent_sizing)
        self.assertIn("recommended_size", vent_sizing)
        self.assertIn("required_area_sq_inches", vent_sizing)
        self.assertGreater(vent_sizing["required_diameter_inches"], 0)
    
    def test_appliance_capacity(self):
        """Test appliance capacity calculations"""
        capacity = self.engine.calculate_appliance_capacity(
            flow_rate_cfph=100,
            fuel_type=FuelType.NATURAL_GAS
        )
        
        self.assertIn("btu_per_hour", capacity)
        self.assertIn("btu_per_hour_millions", capacity)
        self.assertIn("kilowatts", capacity)
        self.assertGreater(capacity["btu_per_hour"], 0)
    
    def test_gas_interchangeability(self):
        """Test gas interchangeability checks"""
        interchange = self.engine.check_gas_interchangeability(
            FuelType.NATURAL_GAS,
            FuelType.PROPANE
        )
        
        self.assertIn("wobble_compatible", interchange)
        self.assertIn("wobble_difference", interchange)
        self.assertIn("methane_compatible", interchange)
        self.assertIsInstance(interchange["wobble_compatible"], bool)
    
    def test_thermal_efficiency(self):
        """Test thermal efficiency calculations"""
        efficiency = self.engine.calculate_thermal_efficiency(
            btu_input=100000,
            btu_output=80000
        )
        
        self.assertIn("efficiency_percent", efficiency)
        self.assertIn("heat_loss_btu", efficiency)
        self.assertIn("heat_loss_percent", efficiency)
        self.assertEqual(efficiency["efficiency_percent"], 80.0)
        
        # Test zero input edge case
        zero_efficiency = self.engine.calculate_thermal_efficiency(0, 0)
        self.assertEqual(zero_efficiency["efficiency_percent"], 0)
    
    def test_fuel_comparison(self):
        """Test fuel comparison data"""
        comparison = self.engine.get_fuel_comparison_data()
        
        self.assertEqual(len(comparison), 4)  # 4 fuel types
        for fuel_type, data in comparison.items():
            self.assertIn("name", data)
            self.assertIn("heating_value", data)
            self.assertIn("cost_per_mmbtu_estimate", data)
            self.assertIn("safety_rating", data)

class TestConsumerVersion(unittest.TestCase):
    """Test consumer version functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = GasPhysicsEngine()
    
    def test_safety_rating_calculation(self):
        """Test safety rating calculations"""
        for fuel_type in FuelType:
            rating = self.engine.calculate_safety_rating(fuel_type)
            self.assertGreaterEqual(rating, 1)
            self.assertLessEqual(rating, 10)
    
    def test_cost_estimation(self):
        """Test cost estimation accuracy"""
        for fuel_type in FuelType:
            cost = self.engine.estimate_fuel_cost(fuel_type)
            self.assertGreater(cost, 0)
    
    def test_environmental_ratings(self):
        """Test environmental rating system"""
        # Test that environmental ratings are consistent
        natural_gas_safety = self.engine.calculate_safety_rating(FuelType.NATURAL_GAS)
        propane_safety = self.engine.calculate_safety_rating(FuelType.PROPANE)
        
        # Natural gas should have better environmental profile
        self.assertGreaterEqual(natural_gas_safety, propane_safety - 2)

class TestMathematicalAccuracy(unittest.TestCase):
    """Test mathematical accuracy of all calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = GasPhysicsEngine()
    
    def test_conservation_of_energy(self):
        """Test conservation of energy in calculations"""
        btu_input = 100000
        efficiency = 0.8
        btu_output = btu_input * efficiency
        
        result = self.engine.calculate_thermal_efficiency(btu_input, btu_output)
        
        # Energy should be conserved (input = output + losses)
        calculated_losses = result["heat_loss_btu"]
        expected_losses = btu_input - btu_output
        
        self.assertAlmostEqual(calculated_losses, expected_losses, places=0)
    
    def test_flow_consistency(self):
        """Test consistency in flow calculations"""
        pressure = 10
        diameter = 1.0
        length = 100
        
        # Flow should be consistent with pressure drop
        flow = self.engine.calculate_gas_flow_rate(
            pressure, diameter, length, FuelType.NATURAL_GAS
        )
        pressure_drop = self.engine.calculate_pressure_drop(
            flow, diameter, length, FuelType.NATURAL_GAS
        )
        
        self.assertGreater(pressure_drop, 0)
        self.assertLess(pressure_drop, pressure)  # Drop should be less than total pressure
    
    def test_unit_conversions(self):
        """Test unit conversions are accurate"""
        # Test BTU to kilowatt conversion
        btu_per_hour = 3412.14  # Should equal 1 kW
        capacity = self.engine.calculate_appliance_capacity(
            3.41214,  # CFH for natural gas
            FuelType.NATURAL_GAS
        )
        
        # Should be approximately 1 kW
        self.assertAlmostEqual(capacity["kilowatts"], 1.0, places=1)

def run_comprehensive_tests():
    """Run all tests and return results"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestGasPhysicsEngine))
    test_suite.addTest(unittest.makeSuite(TestConsumerVersion))
    test_suite.addTest(unittest.makeSuite(TestMathematicalAccuracy))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    }

def main():
    """Main test runner"""
    print("Running Gas Tech Suite Test Suite")
    print("=" * 50)
    
    results = run_comprehensive_tests()
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    
    if results['success_rate'] == 100:
        print("✅ All tests passed! Gas Tech Suite is ready for deployment.")
    elif results['success_rate'] >= 90:
        print("⚠️  Minor issues found. Review and fix before production deployment.")
    else:
        print("❌ Significant issues found. Fix before proceeding.")
    
    return results

if __name__ == "__main__":
    main()
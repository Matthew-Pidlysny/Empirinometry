/*
===============================================================================
PRIME TORSION DYNAMICS - Structural Analysis of Prime Distributions
===============================================================================

Purpose: Apply advanced torsion mechanics to prime distribution analysis
         Building on advanced-torsion.cpp computational framework for prime research

Author: SuperNinja AI Agent
Date: December 2024
Framework: Enhanced Primer Workshop - Prime Torsion Dynamics
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <random>
#include <chrono>
#include <complex>
#include <valarray>
#include <memory>
#include <future>
#include <thread>
#include <mutex>
#include <atomic>
#include <numeric>

using namespace std;

// Torsion-inspired prime analysis structures
struct PrimeTorsionMoment {
    double magnitude;
    double angle;
    double shear_stress;
    double twist_rate;
    int64_t associated_prime;
};

struct PrimeShaftGeometry {
    vector<double> radii;
    vector<double> lengths;
    vector<double> torsional_rigidity;
    vector<int64_t> boundary_primes;
};

struct PrimeStressDistribution {
    vector<double> radial_stress;
    vector<double> hoop_stress;
    vector<double> axial_stress;
    double von_mises_stress;
    double safety_factor;
};

class PrimeTorsionDynamics {
private:
    vector<int64_t> primes;
    vector<PrimeTorsionMoment> torsion_moments;
    vector<PrimeShaftGeometry> shaft_geometries;
    vector<PrimeStressDistribution> stress_distributions;
    
    // Material properties for torsion analysis
    class PrimeMaterial {
    private:
        double shear_modulus;
        double elastic_modulus;
        double poisson_ratio;
        double yield_strength;
        
    public:
        PrimeMaterial() {
            // Material properties optimized for prime analysis
            shear_modulus = 80.0e9;  // GPa (steel-like)
            elastic_modulus = 200.0e9; // GPa
            poisson_ratio = 0.3;
            yield_strength = 250.0e6;  // MPa
        }
        
        double getShearModulus() const { return shear_modulus; }
        double getElasticModulus() const { return elastic_modulus; }
        double getPoissonRatio() const { return poisson_ratio; }
        double getYieldStrength() const { return yield_strength; }
    };
    
    PrimeMaterial material;
    
    // Generate primes up to limit
    vector<int64_t> generatePrimes(int64_t limit) {
        vector<bool> sieve(limit + 1, true);
        sieve[0] = sieve[1] = false;
        
        for (int64_t p = 2; p * p <= limit; p++) {
            if (sieve[p]) {
                for (int64_t i = p * p; i <= limit; i += p) {
                    sieve[i] = false;
                }
            }
        }
        
        vector<int64_t> result;
        for (int64_t i = 2; i <= limit; i++) {
            if (sieve[i]) result.push_back(i);
        }
        
        return result;
    }
    
    // Calculate torsion moments from prime distribution
    void calculateTorsionMoments() {
        cout << "🔧 Calculating torsion moments from prime distribution..." << endl;
        torsion_moments.clear();
        
        for (size_t i = 0; i < primes.size(); i++) {
            PrimeTorsionMoment moment;
            
            // Torsion magnitude based on prime gap
            if (i > 0) {
                int64_t gap = primes[i] - primes[i - 1];
                moment.magnitude = static_cast<double>(gap) * log(primes[i]);
            } else {
                moment.magnitude = primes[i]; // First prime special case
            }
            
            // Torsion angle based on prime index
            moment.angle = 2.0 * M_PI * i / primes.size();
            
            // Shear stress from local prime density
            int local_density = 0;
            for (int64_t p : primes) {
                if (abs(p - primes[i]) <= 100) local_density++;
            }
            moment.shear_stress = moment.magnitude / (local_density + 1.0);
            
            // Twist rate based on prime distribution curvature
            if (i >= 2) {
                double gap1 = primes[i] - primes[i - 1];
                double gap2 = primes[i - 1] - primes[i - 2];
                moment.twist_rate = (gap1 - gap2) / gap2;
            } else {
                moment.twist_rate = 0.0;
            }
            
            moment.associated_prime = primes[i];
            torsion_moments.push_back(moment);
        }
        
        cout << "✅ Calculated " << torsion_moments.size() << " torsion moments" << endl;
    }
    
    // Create shaft geometry from prime clusters
    void createShaftGeometries() {
        cout << "🏗️ Creating shaft geometries from prime clusters..." << endl;
        shaft_geometries.clear();
        
        // Segment primes into shaft sections
        const int segment_size = max(1, static_cast<int>(primes.size() / 10));
        
        for (size_t i = 0; i < primes.size(); i += segment_size) {
            PrimeShaftGeometry geometry;
            
            size_t end_index = min(i + segment_size, primes.size());
            
            // Calculate geometry parameters for this segment
            for (size_t j = i; j < end_index; j++) {
                // Radius based on prime value
                double radius = 10.0 + log(primes[j]) / 10.0;
                geometry.radii.push_back(radius);
                
                // Length based on prime gaps
                double length = j > 0 ? static_cast<double>(primes[j] - primes[j - 1]) : 1.0;
                geometry.lengths.push_back(length);
                
                // Torsional rigidity based on material and geometry
                double J = M_PI * pow(radius, 4) / 2.0; // Polar moment of inertia
                double GJ = material.getShearModulus() * J;
                geometry.torsional_rigidity.push_back(GJ);
                
                if (j == i || j == end_index - 1) {
                    geometry.boundary_primes.push_back(primes[j]);
                }
            }
            
            shaft_geometries.push_back(geometry);
        }
        
        cout << "✅ Created " << shaft_geometries.size() << " shaft geometries" << endl;
    }
    
    // Calculate stress distributions
    void calculateStressDistributions() {
        cout << "⚡ Calculating stress distributions..." << endl;
        stress_distributions.clear();
        
        for (const auto& geometry : shaft_geometries) {
            PrimeStressDistribution stress;
            
            // Calculate radial and hoop stresses
            for (size_t i = 0; i < geometry.radii.size(); i++) {
                // Simplified stress calculations based on torsion theory
                double r = geometry.radii[i];
                double T = torsion_moments.empty() ? 0.0 : torsion_moments[min(i, torsion_moments.size() - 1)].magnitude;
                double J = M_PI * pow(r, 4) / 2.0;
                
                // Shear stress (tau = Tr/J)
                double shear_stress = T * r / J;
                stress.radial_stress.push_back(shear_stress);
                stress.hoop_stress.push_back(shear_stress * 0.8); // Approximate hoop stress
                
                // Axial stress from prime compression
                double axial_stress = static_cast<double>(primes[i]) / 1000.0;
                stress.axial_stress.push_back(axial_stress);
            }
            
            // Calculate von Mises stress
            double max_von_mises = 0.0;
            for (size_t i = 0; i < stress.radial_stress.size(); i++) {
                double sigma_radial = stress.radial_stress[i];
                double sigma_hoop = stress.hoop_stress[i];
                double sigma_axial = stress.axial_stress[i];
                
                // Von Mises stress formula
                double von_mises = sqrt(0.5 * ((sigma_radial - sigma_hoop) * (sigma_radial - sigma_hoop) +
                                              (sigma_hoop - sigma_axial) * (sigma_hoop - sigma_axial) +
                                              (sigma_axial - sigma_radial) * (sigma_axial - sigma_radial)));
                max_von_mises = max(max_von_mises, von_mises);
            }
            stress.von_mises_stress = max_von_mises;
            stress.safety_factor = material.getYieldStrength() / max_von_mises;
            
            stress_distributions.push_back(stress);
        }
        
        cout << "✅ Calculated " << stress_distributions.size() << " stress distributions" << endl;
    }
    
    // Modal analysis of prime shafts
    void performModalAnalysis() {
        cout << "🎵 Performing modal analysis..." << endl;
        
        ofstream modal_report("prime_modal_analysis.txt");
        modal_report << "PRIME SHAFT MODAL ANALYSIS REPORT\n";
        modal_report << "================================\n\n";
        
        for (size_t i = 0; i < shaft_geometries.size(); i++) {
            const auto& geometry = shaft_geometries[i];
            
            modal_report << "Shaft Segment " << (i + 1) << ":\n";
            modal_report << "  Boundary primes: ";
            for (int64_t prime : geometry.boundary_primes) {
                modal_report << prime << " ";
            }
            modal_report << "\n";
            
            // Simplified natural frequency calculation
            double avg_rigidity = 0.0;
            for (double rigidity : geometry.torsional_rigidity) {
                avg_rigidity += rigidity;
            }
            avg_rigidity /= geometry.torsional_rigidity.size();
            
            double total_length = 0.0;
            for (double length : geometry.lengths) {
                total_length += length;
            }
            
            // Natural frequency (simplified)
            double natural_frequency = sqrt(avg_rigidity / total_length) / (2.0 * M_PI);
            
            modal_report << "  Natural frequency: " << fixed << setprecision(2) << natural_frequency << " Hz\n";
            modal_report << "  Average rigidity: " << scientific << avg_rigidity << " N·m²\n";
            modal_report << "  Total length: " << fixed << setprecision(2) << total_length << " m\n\n";
        }
        
        modal_report.close();
        cout << "✅ Modal analysis saved to prime_modal_analysis.txt" << endl;
    }
    
    // Generate torsion visualization
    void generateTorsionVisualization() {
        cout << "📊 Generating torsion visualization..." << endl;
        
        ofstream python_script("prime_torsion_visualization.py");
        python_script << "# Prime Torsion Dynamics Visualization\n";
        python_script << "import matplotlib.pyplot as plt\n";
        python_script << "import numpy as np\n";
        python_script << "from mpl_toolkits.mplot3d import Axes3D\n\n";
        
        // Torsion moment visualization
        python_script << "# Torsion moments data\n";
        python_script << "primes = [";
        for (size_t i = 0; i < min(torsion_moments.size(), size_t(100)); i++) {
            if (i > 0) python_script << ", ";
            python_script << torsion_moments[i].associated_prime;
        }
        python_script << "]\n";
        
        python_script << "magnitudes = [";
        for (size_t i = 0; i < min(torsion_moments.size(), size_t(100)); i++) {
            if (i > 0) python_script << ", ";
            python_script << torsion_moments[i].magnitude;
        }
        python_script << "]\n";
        
        python_script << "angles = [";
        for (size_t i = 0; i < min(torsion_moments.size(), size_t(100)); i++) {
            if (i > 0) python_script << ", ";
            python_script << torsion_moments[i].angle;
        }
        python_script << "]\n\n";
        
        python_script << "# Create 3D torsion visualization\n";
        python_script << "fig = plt.figure(figsize=(15, 10))\n";
        python_script << "ax = fig.add_subplot(111, projection='3d')\n";
        python_script << "x = np.log(primes)\n";
        python_script << "y = magnitudes / np.max(magnitudes)\n";
        python_script << "z = angles / np.max(angles)\n";
        python_script << "scatter = ax.scatter(x, y, z, c=magnitudes, cmap='viridis', s=50)\n";
        python_script << "ax.set_xlabel('Log(Prime)')\n";
        python_script << "ax.set_ylabel('Normalized Torsion Magnitude')\n";
        python_script << "ax.set_zlabel('Normalized Angle')\n";
        python_script << "ax.set_title('Prime Torsion Dynamics 3D Visualization')\n";
        python_script << "plt.colorbar(scatter, label='Torsion Magnitude')\n";
        python_script << "plt.savefig('prime_torsion_3d.png', dpi=300, bbox_inches='tight')\n";
        python_script << "plt.show()\n\n";
        
        // Stress distribution plot
        python_script << "# Stress distribution plot\n";
        python_script << "fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n";
        python_script << "# Radial stress\n";
        python_script << "for i, stress in enumerate([";
        for (size_t i = 0; i < min(stress_distributions.size(), size_t(5)); i++) {
            if (i > 0) python_script << ", ";
            python_script << "stress_distributions[" << i << "].radial_stress";
        }
        python_script << "]):\n";
        python_script << "    ax1.plot(stress, label=f'Shaft {i+1}')\n";
        python_script << "ax1.set_xlabel('Position along shaft')\n";
        python_script << "ax1.set_ylabel('Radial Stress (Pa)')\n";
        python_script << "ax1.set_title('Radial Stress Distribution')\n";
        python_script << "ax1.legend()\n";
        python_script << "ax1.grid(True)\n\n";
        
        python_script << "# Von Mises stress\n";
        python_script << "von_mises = [";
        for (size_t i = 0; i < stress_distributions.size(); i++) {
            if (i > 0) python_script << ", ";
            python_script << stress_distributions[i].von_mises_stress;
        }
        python_script << "]\n";
        python_script << "safety_factors = [";
        for (size_t i = 0; i < stress_distributions.size(); i++) {
            if (i > 0) python_script << ", ";
            python_script << stress_distributions[i].safety_factor;
        }
        python_script << "]\n";
        python_script << "shaft_indices = list(range(1, len(von_mises) + 1))\n";
        python_script << "ax2.bar(shaft_indices, von_mises, alpha=0.7, label='Von Mises Stress')\n";
        python_script << "ax2.set_xlabel('Shaft Segment')\n";
        python_script << "ax2.set_ylabel('Von Mises Stress (Pa)')\n";
        python_script << "ax2.set_title('Von Mises Stress by Shaft Segment')\n";
        python_script << "ax2.legend()\n";
        python_script << "ax2.grid(True)\n\n";
        
        python_script << "plt.tight_layout()\n";
        python_script << "plt.savefig('prime_stress_analysis.png', dpi=300, bbox_inches='tight')\n";
        python_script << "plt.show()\n";
        
        python_script.close();
        
        cout << "✅ Torsion visualization saved to prime_torsion_visualization.py" << endl;
    }
    
    // Generate comprehensive report
    void generateReport() {
        cout << "📋 Generating torsion dynamics report..." << endl;
        
        ofstream report("prime_torsion_dynamics_report.txt");
        
        report << "===============================================================================\n";
        report << "PRIME TORSION DYNAMICS ANALYSIS REPORT\n";
        report << "===============================================================================\n\n";
        
        report << "Analysis Date: " << __DATE__ << " " << __TIME__ << "\n";
        report << "Prime Range: 2 to " << (primes.empty() ? 0 : primes.back()) << "\n";
        report << "Total Primes Analyzed: " << primes.size() << "\n\n";
        
        // Torsion moment statistics
        report << "TORSION MOMENT STATISTICS\n";
        report << "========================\n";
        
        double avg_magnitude = 0.0, max_magnitude = 0.0, min_magnitude = 1e20;
        double avg_shear = 0.0, max_shear = 0.0;
        
        for (const auto& moment : torsion_moments) {
            avg_magnitude += moment.magnitude;
            max_magnitude = max(max_magnitude, moment.magnitude);
            min_magnitude = min(min_magnitude, moment.magnitude);
            avg_shear += moment.shear_stress;
            max_shear = max(max_shear, moment.shear_stress);
        }
        avg_magnitude /= torsion_moments.size();
        avg_shear /= torsion_moments.size();
        
        report << "Average torsion magnitude: " << scientific << avg_magnitude << "\n";
        report << "Maximum torsion magnitude: " << scientific << max_magnitude << "\n";
        report << "Minimum torsion magnitude: " << scientific << min_magnitude << "\n";
        report << "Average shear stress: " << scientific << avg_shear << " Pa\n";
        report << "Maximum shear stress: " << scientific << max_shear << " Pa\n\n";
        
        // Shaft geometry statistics
        report << "SHAFT GEOMETRY STATISTICS\n";
        report << "========================\n";
        report << "Number of shaft segments: " << shaft_geometries.size() << "\n";
        
        for (size_t i = 0; i < min(shaft_geometries.size(), size_t(3)); i++) {
            const auto& geometry = shaft_geometries[i];
            report << "\nShaft Segment " << (i + 1) << ":\n";
            report << "  Number of sections: " << geometry.radii.size() << "\n";
            report << "  Boundary primes: ";
            for (int64_t prime : geometry.boundary_primes) {
                report << prime << " ";
            }
            report << "\n";
            
            double avg_radius = 0.0;
            for (double radius : geometry.radii) {
                avg_radius += radius;
            }
            avg_radius /= geometry.radii.size();
            report << "  Average radius: " << fixed << setprecision(3) << avg_radius << " m\n";
        }
        
        // Stress analysis summary
        report << "\nSTRESS ANALYSIS SUMMARY\n";
        report << "======================\n";
        double max_von_mises = 0.0, min_safety = 1e20;
        for (const auto& stress : stress_distributions) {
            max_von_mises = max(max_von_mises, stress.von_mises_stress);
            min_safety = min(min_safety, stress.safety_factor);
        }
        
        report << "Maximum von Mises stress: " << scientific << max_von_mises << " Pa\n";
        report << "Minimum safety factor: " << fixed << setprecision(3) << min_safety << "\n";
        report << "Material yield strength: " << scientific << material.getYieldStrength() << " Pa\n";
        
        if (min_safety > 1.0) {
            report << "✅ All shaft segments satisfy safety requirements\n";
        } else {
            report << "⚠️  Some shaft segments may be overstressed\n";
        }
        
        report.close();
        
        cout << "✅ Report saved to prime_torsion_dynamics_report.txt" << endl;
    }
    
public:
    PrimeTorsionDynamics() {
        cout << "🔧 Prime Torsion Dynamics Initialized" << endl;
    }
    
    void initialize(int64_t prime_limit) {
        cout << "📊 Generating primes up to " << prime_limit << "..." << endl;
        primes = generatePrimes(prime_limit);
        cout << "✅ Generated " << primes.size() << " primes" << endl;
    }
    
    void execute() {
        cout << "\n🚀 PRIME TORSION DYNAMICS ANALYZER" << endl;
        cout << "==================================" << endl;
        
        initialize(50000);
        calculateTorsionMoments();
        createShaftGeometries();
        calculateStressDistributions();
        performModalAnalysis();
        generateTorsionVisualization();
        generateReport();
        
        cout << "\n✅ Prime Torsion Dynamics Analysis Complete!" << endl;
        cout << "📊 Reports generated:" << endl;
        cout << "   • prime_torsion_dynamics_report.txt - Comprehensive torsion analysis" << endl;
        cout << "   • prime_modal_analysis.txt - Modal frequency analysis" << endl;
        cout << "   • prime_torsion_visualization.py - 3D torsion visualization" << endl;
    }
};

int main() {
    cout << fixed << setprecision(6);
    
    PrimeTorsionDynamics analyzer;
    analyzer.execute();
    
    return 0;
}
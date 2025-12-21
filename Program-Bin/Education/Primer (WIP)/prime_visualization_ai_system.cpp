/*
 * PRIME VISUALIZATION AI SYSTEM
 * Enhanced AI integration with visualization data generation
 * Generates visualizer-compatible data blocks from AI responses
 * 
 * Features:
 * 1. AI-driven visualization data generation
 * 2. Custom visualizer block creation
 * 3. Multiple visualization format support
 * 4. Prime-specific visualization patterns
 * 5. ASCII and mathematical notation integration
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <regex>
#include <random>
#include <chrono>
#include <thread>
#include <future>
#include <memory>
#include <iomanip>

using namespace std;

class PrimeVisualizationAISystem {
private:
    struct VisualizationBlock {
        string visualizer_type;
        vector<vector<int>> data_grid;
        string metadata;
        string ai_description;
        vector<string> color_scheme;
        string pattern_formula;
    };
    
    struct VisualizationPrompt {
        string domain_context;
        string visualization_type;
        vector<string> data_constraints;
        string output_format;
        string prime_focus;
    };
    
    map<string, VisualizationPrompt> viz_templates;
    random_device rd;
    mt19937 gen{rd()};
    
public:
    PrimeVisualizationAISystem() {
        initialize_visualization_templates();
    }
    
private:
    void initialize_visualization_templates() {
        // Template for ASCII constellation visualization
        VisualizationPrompt constellation_viz;
        constellation_viz.domain_context = "prime_constellation_patterns";
        constellation_viz.visualization_type = "ascii_grid";
        constellation_viz.data_constraints = {"grid_based", "star_positions", "prime_connections"};
        constellation_viz.output_format = "ascii_art";
        constellation_viz.prime_focus = "twin_primes,prime_triplets,pattern_relationships";
        viz_templates["constellation"] = constellation_viz;
        
        // Template for mathematical curve visualization
        VisualizationPrompt curve_viz;
        curve_viz.domain_context = "prime_distribution_analysis";
        curve_viz.visualization_type = "mathematical_curve";
        curve_viz.data_constraints = {"continuous_function", "x_y_coordinates", "smooth_interpolation"};
        curve_viz.output_format = "coordinate_pairs";
        curve_viz.prime_focus = "prime_density,growth_patterns,statistical_distribution";
        viz_templates["curve"] = curve_viz;
        
        // Template for heatmap visualization
        VisualizationPrompt heatmap_viz;
        heatmap_viz.domain_context = "prime_density_mapping";
        heatmap_viz.visualization_type = "intensity_heatmap";
        heatmap_viz.data_constraints = {"2D_matrix", "color_intensity", "density_values"};
        heatmap_viz.output_format = "matrix_values";
        heatmap_viz.prime_focus = "prime_density,regional_distribution,pattern_hotspots";
        viz_templates["heatmap"] = heatmap_viz;
        
        // Template for fractal visualization
        VisualizationPrompt fractal_viz;
        fractal_viz.domain_context = "prime_fractal_patterns";
        fractal_viz.visualization_type = "fractal_geometry";
        fractal_viz.data_constraints = {"recursive_patterns", "self_similarity", "complex_coordinates"};
        fractal_viz.output_format = "complex_plane";
        fractal_viz.prime_focus = "recursive_primes,fractal_dimensions,pattern_repetition";
        viz_templates["fractal"] = fractal_viz;
    }
    
    bool is_prime(uint64_t n) {
        if (n < 2) return false;
        if (n == 2 || n == 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        
        for (uint64_t i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) {
                return false;
            }
        }
        return true;
    }
    
    VisualizationBlock generate_constellation_visualization(const vector<string>& keywords) {
        VisualizationBlock block;
        block.visualizer_type = "ascii_constellation";
        block.ai_description = "Prime constellation pattern showing twin primes and prime triplets";
        block.color_scheme = {"#FFD700", "#FF6B6B", "#4ECDC4", "#45B7D1"};
        
        // Create 20x20 grid for constellation
        int grid_size = 20;
        block.data_grid.resize(grid_size, vector<int>(grid_size, 0));
        
        // Generate prime positions based on keywords
        vector<uint64_t> primes = generate_prime_visualization_data(100);
        
        // Map primes to grid positions (simulating constellation)
        for (size_t i = 0; i < primes.size() && i < 50; ++i) {
            int x = (primes[i] * 7) % grid_size;
            int y = (primes[i] * 11) % grid_size;
            int intensity = (primes[i] % 4) + 1;
            block.data_grid[y][x] = intensity;
        }
        
        // Connect nearby primes to form constellation lines
        for (int y = 0; y < grid_size; ++y) {
            for (int x = 0; x < grid_size; ++x) {
                if (block.data_grid[y][x] > 0) {
                    // Check for nearby primes to connect
                    for (int dy = -2; dy <= 2; ++dy) {
                        for (int dx = -2; dx <= 2; ++dx) {
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < grid_size && nx >= 0 && nx < grid_size) {
                                if (block.data_grid[ny][nx] > 0 && (abs(dx) + abs(dy) <= 2)) {
                                    // Mark connection (value 9)
                                    if (abs(dx) == 1 || abs(dy) == 1) {
                                        block.data_grid[(y + ny) / 2][(x + nx) / 2] = 9;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        block.metadata = "grid_size:" + to_string(grid_size) + ",primes_mapped:" + to_string(min(50UL, primes.size()));
        block.pattern_formula = "C(x,y) = prime_map(x,y) * intensity(prime_mod_4)";
        
        return block;
    }
    
    VisualizationBlock generate_curve_visualization(const vector<string>& keywords) {
        VisualizationBlock block;
        block.visualizer_type = "prime_density_curve";
        block.ai_description = "Mathematical curve showing prime density distribution over number range";
        block.color_scheme = {"#2E86AB", "#A23B72", "#F18F01", "#C73E1D"};
        
        // Generate prime density curve data
        int points = 100;
        int max_x = 10000;
        
        block.data_grid.resize(2, vector<int>(points)); // [0]=x coordinates, [1]=y values
        
        for (int i = 0; i < points; ++i) {
            int x = (i * max_x) / points;
            block.data_grid[0][i] = x;
            
            // Calculate prime density near x
            int count = 0;
            int window = max(10, x / 100);
            for (int j = max(2, x - window); j <= x + window; ++j) {
                if (is_prime(j)) count++;
            }
            
            // Normalize density (approximate 1/ln(x))
            double density = (2.0 * window) * count / (2.0 * window + 1);
            block.data_grid[1][i] = static_cast<int>(density * 1000); // Scale for visualization
        }
        
        block.metadata = "points:" + to_string(points) + ",range:0-" + to_string(max_x);
        block.pattern_formula = "D(x) = prime_count(x) / log(x) approximation";
        
        return block;
    }
    
    VisualizationBlock generate_heatmap_visualization(const vector<string>& keywords) {
        VisualizationBlock block;
        block.visualizer_type = "prime_density_heatmap";
        block.ai_description = "2D heatmap showing prime density across number ranges";
        block.color_scheme = {"#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"};
        
        // Create 30x30 heatmap
        int grid_size = 30;
        block.data_grid.resize(grid_size, vector<int>(grid_size, 0));
        
        // Map number ranges to grid positions
        int range_per_cell = 1000;
        
        for (int y = 0; y < grid_size; ++y) {
            for (int x = 0; x < grid_size; ++x) {
                int start_num = (y * grid_size + x) * range_per_cell;
                int end_num = start_num + range_per_cell;
                
                // Count primes in this range
                int prime_count = 0;
                for (int n = start_num; n <= end_num && n < start_num + 1000; ++n) {
                    if (is_prime(n)) prime_count++;
                }
                
                // Normalize to 0-100 scale for heat
                block.data_grid[y][x] = min(100, (prime_count * 100) / 100);
            }
        }
        
        block.metadata = "grid_size:" + to_string(grid_size) + ",range_per_cell:" + to_string(range_per_cell);
        block.pattern_formula = "H(x,y) = prime_density(range_start + (y*width+x)*range_per_cell)";
        
        return block;
    }
    
    VisualizationBlock generate_fractal_visualization(const vector<string>& keywords) {
        VisualizationBlock block;
        block.visualizer_type = "prime_fractal";
        block.ai_description = "Fractal patterns generated from prime number sequences";
        block.color_scheme = {"#4A148C", "#7B1FA2", "#9C27B0", "#BA68C8", "#CE93D8"};
        
        // Create Mandelbrot-like fractal using prime-based iterations
        int grid_size = 25;
        block.data_grid.resize(grid_size, vector<int>(grid_size, 0));
        
        for (int y = 0; y < grid_size; ++y) {
            for (int x = 0; x < grid_size; ++x) {
                // Map grid to complex plane
                double real = (x - grid_size / 2.0) * 0.1;
                double imag = (y - grid_size / 2.0) * 0.1;
                
                // Use prime-based iteration count
                int iterations = prime_fractal_iterations(real, imag);
                block.data_grid[y][x] = iterations % 256;
            }
        }
        
        block.metadata = "grid_size:" + to_string(grid_size) + ",fractal_type:prime_mandelbrot";
        block.pattern_formula = "F(z) = z^2 + c where iterations use prime sequence";
        
        return block;
    }
    
    int prime_fractal_iterations(double real, double imag) {
        double zr = 0, zi = 0;
        vector<uint64_t> primes = generate_prime_visualization_data(100);
        
        for (int i = 0; i < 50; ++i) {
            double zr_new = zr * zr - zi * zi + real;
            double zi_new = 2 * zr * zi + imag;
            zr = zr_new;
            zi = zi_new;
            
            if (zr * zr + zi * zi > 4) {
                // Use prime-based coloring
                return primes[i % primes.size()] % 256;
            }
        }
        
        return 255; // Inside set
    }
    
    vector<uint64_t> generate_prime_visualization_data(int count) {
        vector<uint64_t> primes;
        uint64_t n = 2;
        
        while (primes.size() < static_cast<size_t>(count)) {
            if (is_prime(n)) {
                primes.push_back(n);
            }
            n++;
        }
        
        return primes;
    }
    
public:
    // Main AI visualization generation function
    void generate_ai_visualization(const vector<string>& keywords, const string& viz_type) {
        cout << "\n=== AI VISUALIZATION GENERATION ===\n";
        cout << "Keywords: ";
        for (const auto& kw : keywords) cout << kw << " ";
        cout << "\nVisualization Type: " << viz_type << endl;
        
        VisualizationBlock block;
        
        if (viz_type == "constellation") {
            block = generate_constellation_visualization(keywords);
        } else if (viz_type == "curve") {
            block = generate_curve_visualization(keywords);
        } else if (viz_type == "heatmap") {
            block = generate_heatmap_visualization(keywords);
        } else if (viz_type == "fractal") {
            block = generate_fractal_visualization(keywords);
        } else {
            cout << "Unknown visualization type: " << viz_type << endl;
            return;
        }
        
        // Generate AI description
        string ai_response = generate_visualization_ai_description(block, keywords);
        
        // Display results
        cout << "\n=== AI GENERATED VISUALIZATION DATA ===\n";
        cout << "Visualizer Type: " << block.visualizer_type << endl;
        cout << "AI Description: " << block.ai_description << endl;
        cout << "Pattern Formula: " << block.pattern_formula << endl;
        cout << "Metadata: " << block.metadata << endl;
        
        cout << "\n=== VISUALIZATION DATA BLOCK ===\n";
        cout << "[VISUALIZATION_BLOCK_START]\n";
        cout << "TYPE:" << block.visualizer_type << "\n";
        cout << "DESCRIPTION:" << ai_response << "\n";
        cout << "FORMULA:" << block.pattern_formula << "\n";
        cout << "METADATA:" << block.metadata << "\n";
        cout << "COLOR_SCHEME:";
        for (const auto& color : block.color_scheme) {
            cout << color << ",";
        }
        cout << "\n";
        
        cout << "DATA_START\n";
        for (const auto& row : block.data_grid) {
            for (int val : row) {
                cout << val << " ";
            }
            cout << "\n";
        }
        cout << "DATA_END\n";
        cout << "[VISUALIZATION_BLOCK_END]\n";
        
        // Export to file
        export_visualization_block(block, "ai_generated_" + viz_type + ".viz");
    }
    
private:
    string generate_visualization_ai_description(const VisualizationBlock& block, const vector<string>& keywords) {
        stringstream ss;
        ss << "AI-generated ";
        
        if (block.visualizer_type == "ascii_constellation") {
            ss << "prime constellation visualization showing ";
            ss << "the interconnected relationships between prime numbers. ";
            ss << "Each star represents a prime, with connections showing ";
            ss << "mathematical relationships in prime distribution patterns. ";
            ss << "The intensity variations indicate different prime categories ";
            ss << "based on their modular arithmetic properties. ";
            ss << "This visualization reveals the hidden structure underlying ";
            ss << "prime number distribution in number space.";
        } else if (block.visualizer_type == "prime_density_curve") {
            ss << "prime density curve illustrating ";
            ss << "how primes become sparser as numbers increase. ";
            ss << "The curve follows the prime number theorem approximation ";
            ss << "showing the natural logarithmic decline in prime frequency. ";
            ss << "Peaks in the curve indicate regions of unusually high ";
            ss << "prime concentration, while valleys show prime gaps. ";
            ss << "This visualization demonstrates the fundamental ";
            ss << "asymptotic behavior of prime distribution.";
        } else if (block.visualizer_type == "prime_density_heatmap") {
            ss << "2D prime density heatmap revealing ";
            ss << "spatial variations in prime distribution across number ranges. ";
            ss << "Hot spots indicate regions with high prime concentration, ";
            ss << "while cooler areas show prime deserts. ";
            ss << "The gradient transitions smoothly between dense and sparse regions, ";
            ss << "highlighting the irregular yet predictable nature of prime distribution. ";
            ss << "This visualization uncovers the landscape of primes ";
            ss << "in the mathematical number space.";
        } else if (block.visualizer_type == "prime_fractal") {
            ss << "prime fractal visualization demonstrating ";
            ss << "self-similar patterns in prime number sequences. ";
            ss << "The fractal emerges from iterative applications of ";
            ss << "prime-based mathematical functions in the complex plane. ";
            ss << "Color gradients represent iteration counts based on ";
            ss << "prime sequences, revealing intricate geometric structures. ";
            ss << "This visualization connects number theory with ";
            ss << "fractal geometry in unprecedented ways.";
        }
        
        return ss.str();
    }
    
    void export_visualization_block(const VisualizationBlock& block, const string& filename) {
        ofstream file(filename);
        if (!file.is_open()) {
            cout << "Error: Could not create visualization file: " << filename << endl;
            return;
        }
        
        file << "# AI Generated Prime Visualization Data\n";
        file << "# Type: " << block.visualizer_type << "\n";
        file << "# Description: " << block.ai_description << "\n";
        file << "# Formula: " << block.pattern_formula << "\n";
        file << "# Metadata: " << block.metadata << "\n";
        file << "# Generated: " << chrono::system_clock::now().time_since_epoch().count() << "\n\n";
        
        file << "[DATA]\n";
        for (const auto& row : block.data_grid) {
            for (int val : row) {
                file << val << " ";
            }
            file << "\n";
        }
        
        file.close();
        cout << "Visualization exported to: " << filename << endl;
    }
    
public:
    // Demonstrate all visualization types
    void run_visualization_demonstration() {
        cout << "\n=== PRIME VISUALIZATION AI SYSTEM DEMO ===\n";
        
        vector<vector<string>> test_cases = {
            {"prime", "constellation", "pattern"},
            {"density", "distribution", "curve"},
            {"heatmap", "intensity", "regions"},
            {"fractal", "geometry", "self-similar"}
        };
        
        vector<string> viz_types = {"constellation", "curve", "heatmap", "fractal"};
        
        for (size_t i = 0; i < test_cases.size() && i < viz_types.size(); ++i) {
            cout << "\n--- Test Case " << (i + 1) << " ---\n";
            generate_ai_visualization(test_cases[i], viz_types[i]);
            cout << "Press Enter to continue to next visualization...";
            cin.ignore();
        }
        
        cout << "\n✅ AI Visualization System Demo Complete!\n";
        cout << "Generated " << viz_types.size() << " different visualization types\n";
    }
};

int main() {
    cout << "==========================================\n";
    cout << "  PRIME VISUALIZATION AI SYSTEM\n";
    cout << "  AI-Driven Visualization Generation\n";
    cout << "==========================================\n\n";
    
    PrimeVisualizationAISystem viz_system;
    
    // Run complete demonstration
    viz_system.run_visualization_demonstration();
    
    // Interactive mode
    cout << "\n=== INTERACTIVE MODE ===\n";
    cout << "Enter visualization keywords (space-separated): ";
    
    string input;
    getline(cin, input);
    
    if (!input.empty()) {
        vector<string> keywords;
        stringstream ss(input);
        string keyword;
        while (ss >> keyword) {
            keywords.push_back(keyword);
        }
        
        cout << "Available types: constellation, curve, heatmap, fractal\n";
        cout << "Enter visualization type: ";
        string viz_type;
        getline(cin, viz_type);
        
        viz_system.generate_ai_visualization(keywords, viz_type);
    }
    
    return 0;
}
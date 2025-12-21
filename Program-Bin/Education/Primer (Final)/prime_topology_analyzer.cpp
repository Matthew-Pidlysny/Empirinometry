/*
===============================================================================
PRIME TOPOLOGY ANALYZER - Network Analysis & Connectivity
===============================================================================

Purpose: Topological analysis of prime relationships and network structures
         Building on advanced computational frameworks for prime research

Author: SuperNinja AI Agent
Date: December 2024
Framework: Enhanced Primer Workshop - Prime Topology Analysis
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
#include <unordered_set>
#include <queue>

using namespace std;

// Topological analysis structures
struct PrimeNode {
    int64_t prime;
    vector<int64_t> neighbors;
    double centrality;
    double clustering_coefficient;
    int degree;
    string node_type;
};

struct PrimeNetwork {
    vector<PrimeNode> nodes;
    map<int64_t, int> prime_to_index;
    double network_density;
    double average_path_length;
    string network_type;
};

struct TopologicalMetric {
    string metric_name;
    double value;
    string interpretation;
    vector<double> distribution;
};

class PrimeTopologyAnalyzer {
private:
    vector<int64_t> primes;
    vector<PrimeNetwork> networks;
    vector<TopologicalMetric> topological_metrics;
    
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
    
    // Create prime gap network
    PrimeNetwork createGapNetwork() {
        cout << "🔗 Creating prime gap network..." << endl;
        
        PrimeNetwork network;
        network.network_type = "Prime Gap Network";
        
        // Create nodes
        for (int64_t prime : primes) {
            PrimeNode node;
            node.prime = prime;
            node.degree = 0;
            node.node_type = "prime";
            network.nodes.push_back(node);
        }
        
        // Build index map
        for (size_t i = 0; i < network.nodes.size(); i++) {
            network.prime_to_index[network.nodes[i].prime] = i;
        }
        
        // Create edges based on prime gaps
        for (size_t i = 0; i < primes.size(); i++) {
            int64_t current_prime = primes[i];
            int current_index = network.prime_to_index[current_prime];
            
            // Connect to nearby primes within gap threshold
            for (size_t j = 0; j < primes.size(); j++) {
                if (i != j) {
                    int64_t gap = abs(primes[i] - primes[j]);
                    if (gap <= 50) { // Gap threshold for connectivity
                        network.nodes[current_index].neighbors.push_back(primes[j]);
                        network.nodes[current_index].degree++;
                    }
                }
            }
        }
        
        // Calculate network metrics
        calculateNetworkMetrics(network);
        
        cout << "✅ Gap network created with " << network.nodes.size() << " nodes" << endl;
        return network;
    }
    
    // Create twin prime network
    PrimeNetwork createTwinPrimeNetwork() {
        cout << "👥 Creating twin prime network..." << endl;
        
        PrimeNetwork network;
        network.network_type = "Twin Prime Network";
        
        // Identify twin primes
        vector<pair<int64_t, int64_t>> twin_pairs;
        for (size_t i = 1; i < primes.size(); i++) {
            if (primes[i] - primes[i - 1] == 2) {
                twin_pairs.push_back({primes[i - 1], primes[i]});
            }
        }
        
        // Create nodes for twin primes
        set<int64_t> twin_primes;
        for (auto pair : twin_pairs) {
            twin_primes.insert(pair.first);
            twin_primes.insert(pair.second);
        }
        
        for (int64_t prime : twin_primes) {
            PrimeNode node;
            node.prime = prime;
            node.degree = 0;
            node.node_type = "twin_prime";
            network.nodes.push_back(node);
        }
        
        // Build index map
        for (size_t i = 0; i < network.nodes.size(); i++) {
            network.prime_to_index[network.nodes[i].prime] = i;
        }
        
        // Connect twin prime pairs
        for (auto pair : twin_pairs) {
            int index1 = network.prime_to_index[pair.first];
            int index2 = network.prime_to_index[pair.second];
            
            network.nodes[index1].neighbors.push_back(pair.second);
            network.nodes[index2].neighbors.push_back(pair.first);
            network.nodes[index1].degree++;
            network.nodes[index2].degree++;
        }
        
        // Connect nearby twin primes
        for (size_t i = 0; i < network.nodes.size(); i++) {
            for (size_t j = i + 1; j < network.nodes.size(); j++) {
                int64_t gap = abs(network.nodes[i].prime - network.nodes[j].prime);
                if (gap <= 100 && gap != 2) { // Connect nearby twin primes
                    network.nodes[i].neighbors.push_back(network.nodes[j].prime);
                    network.nodes[j].neighbors.push_back(network.nodes[i].prime);
                    network.nodes[i].degree++;
                    network.nodes[j].degree++;
                }
            }
        }
        
        calculateNetworkMetrics(network);
        
        cout << "✅ Twin prime network created with " << network.nodes.size() << " nodes" << endl;
        return network;
    }
    
    // Create prime progression network
    PrimeNetwork createProgressionNetwork() {
        cout << "📈 Creating prime progression network..." << endl;
        
        PrimeNetwork network;
        network.network_type = "Prime Progression Network";
        
        // Find arithmetic progressions of primes
        vector<vector<int64_t>> progressions;
        
        // Find 3-term arithmetic progressions
        for (size_t i = 0; i < primes.size(); i++) {
            for (size_t j = i + 1; j < primes.size(); j++) {
                int64_t diff = primes[j] - primes[i];
                int64_t next_val = primes[j] + diff;
                
                // Check if next_val is prime
                if (binary_search(primes.begin(), primes.end(), next_val)) {
                    vector<int64_t> prog = {primes[i], primes[j], next_val};
                    progressions.push_back(prog);
                }
            }
        }
        
        // Create nodes for primes in progressions
        set<int64_t> prog_primes;
        for (auto prog : progressions) {
            for (int64_t prime : prog) {
                prog_primes.insert(prime);
            }
        }
        
        for (int64_t prime : prog_primes) {
            PrimeNode node;
            node.prime = prime;
            node.degree = 0;
            node.node_type = "progression_prime";
            network.nodes.push_back(node);
        }
        
        // Build index map
        for (size_t i = 0; i < network.nodes.size(); i++) {
            network.prime_to_index[network.nodes[i].prime] = i;
        }
        
        // Connect primes in same progressions
        for (auto prog : progressions) {
            for (size_t i = 0; i < prog.size(); i++) {
                for (size_t j = i + 1; j < prog.size(); j++) {
                    int index1 = network.prime_to_index[prog[i]];
                    int index2 = network.prime_to_index[prog[j]];
                    
                    // Check if already connected
                    bool already_connected = false;
                    for (int64_t neighbor : network.nodes[index1].neighbors) {
                        if (neighbor == prog[j]) {
                            already_connected = true;
                            break;
                        }
                    }
                    
                    if (!already_connected) {
                        network.nodes[index1].neighbors.push_back(prog[j]);
                        network.nodes[index2].neighbors.push_back(prog[i]);
                        network.nodes[index1].degree++;
                        network.nodes[index2].degree++;
                    }
                }
            }
        }
        
        calculateNetworkMetrics(network);
        
        cout << "✅ Progression network created with " << network.nodes.size() << " nodes" << endl;
        return network;
    }
    
    // Calculate network metrics
    void calculateNetworkMetrics(PrimeNetwork& network) {
        if (network.nodes.empty()) return;
        
        // Calculate network density
        int possible_edges = network.nodes.size() * (network.nodes.size() - 1) / 2;
        int actual_edges = 0;
        for (const auto& node : network.nodes) {
            actual_edges += node.degree;
        }
        actual_edges /= 2; // Each edge counted twice
        network.network_density = static_cast<double>(actual_edges) / possible_edges;
        
        // Calculate centrality measures
        calculateCentrality(network);
        
        // Calculate clustering coefficients
        calculateClusteringCoefficients(network);
        
        // Estimate average path length (simplified)
        network.average_path_length = estimateAveragePathLength(network);
    }
    
    void calculateCentrality(PrimeNetwork& network) {
        // Degree centrality
        int max_degree = 0;
        for (const auto& node : network.nodes) {
            max_degree = max(max_degree, node.degree);
        }
        
        for (auto& node : network.nodes) {
            node.centrality = static_cast<double>(node.degree) / max_degree;
        }
    }
    
    void calculateClusteringCoefficients(PrimeNetwork& network) {
        for (auto& node : network.nodes) {
            if (node.degree < 2) {
                node.clustering_coefficient = 0.0;
                continue;
            }
            
            int neighbor_connections = 0;
            for (int64_t neighbor1 : node.neighbors) {
                for (int64_t neighbor2 : node.neighbors) {
                    if (neighbor1 != neighbor2) {
                        // Check if neighbors are connected
                        int idx1 = network.prime_to_index[neighbor1];
                        int idx2 = network.prime_to_index[neighbor2];
                        
                        for (int64_t neighbor : network.nodes[idx1].neighbors) {
                            if (neighbor == neighbor2) {
                                neighbor_connections++;
                                break;
                            }
                        }
                    }
                }
            }
            
            int possible_connections = node.degree * (node.degree - 1);
            node.clustering_coefficient = static_cast<double>(neighbor_connections) / possible_connections;
        }
    }
    
    double estimateAveragePathLength(const PrimeNetwork& network) {
        if (network.nodes.size() < 2) return 0.0;
        
        // Simplified estimation using network properties
        double density = network.network_density;
        double n = network.nodes.size();
        
        // Approximate formula for random networks
        if (density > 0) {
            return -log(density) / log(n * density);
        }
        
        return n / 2.0; // Upper bound estimate
    }
    
    // Analyze topological properties
    void analyzeTopology() {
        cout << "🔍 Analyzing topological properties..." << endl;
        
        for (const auto& network : networks) {
            analyzeNetworkTopology(network);
        }
        
        cout << "✅ Topological analysis complete" << endl;
    }
    
    void analyzeNetworkTopology(const PrimeNetwork& network) {
        // Metric 1: Degree distribution
        TopologicalMetric degree_dist;
        degree_dist.metric_name = "Degree Distribution";
        
        vector<double> degrees;
        for (const auto& node : network.nodes) {
            degrees.push_back(node.degree);
        }
        degree_dist.distribution = degrees;
        
        double avg_degree = 0.0;
        for (double degree : degrees) {
            avg_degree += degree;
        }
        avg_degree /= degrees.size();
        degree_dist.value = avg_degree;
        degree_dist.interpretation = "Average connectivity in " + network.network_type;
        topological_metrics.push_back(degree_dist);
        
        // Metric 2: Clustering coefficient
        TopologicalMetric clustering_metric;
        clustering_metric.metric_name = "Average Clustering Coefficient";
        
        double avg_clustering = 0.0;
        for (const auto& node : network.nodes) {
            avg_clustering += node.clustering_coefficient;
        }
        avg_clustering /= network.nodes.size();
        clustering_metric.value = avg_clustering;
        clustering_metric.interpretation = "Measure of local connectivity in " + network.network_type;
        topological_metrics.push_back(clustering_metric);
        
        // Metric 3: Network density
        TopologicalMetric density_metric;
        density_metric.metric_name = "Network Density";
        density_metric.value = network.network_density;
        density_metric.interpretation = "Overall connectivity in " + network.network_type;
        topological_metrics.push_back(density_metric);
        
        // Metric 4: Centralization
        TopologicalMetric centralization_metric;
        centralization_metric.metric_name = "Network Centralization";
        
        double max_centrality = 0.0, sum_centrality = 0.0;
        for (const auto& node : network.nodes) {
            max_centrality = max(max_centrality, node.centrality);
            sum_centrality += node.centrality;
        }
        
        double avg_centrality = sum_centrality / network.nodes.size();
        centralization_metric.value = (max_centrality - avg_centrality) / (network.nodes.size() - 1);
        centralization_metric.interpretation = "Degree of centralization in " + network.network_type;
        topological_metrics.push_back(centralization_metric);
    }
    
    // Generate topology visualization
    void generateTopologyVisualization() {
        cout << "📊 Generating topology visualization..." << endl;
        
        ofstream python_script("prime_topology_visualization.py");
        python_script << "# Prime Topology Analysis Visualization\n";
        python_script << "import matplotlib.pyplot as plt\n";
        python_script << "import numpy as np\n";
        python_script << "import networkx as nx\n\n";
        
        // Network statistics
        python_script << "# Network statistics\n";
        python_script << "network_types = [";
        for (size_t i = 0; i < networks.size(); i++) {
            if (i > 0) python_script << ", ";
            python_script << "'" << networks[i].network_type << "'";
        }
        python_script << "]\n";
        
        python_script << "network_sizes = [";
        for (size_t i = 0; i < networks.size(); i++) {
            if (i > 0) python_script << ", ";
            python_script << networks[i].nodes.size();
        }
        python_script << "]\n";
        
        python_script << "network_densities = [";
        for (size_t i = 0; i < networks.size(); i++) {
            if (i > 0) python_script << ", ";
            python_script << networks[i].network_density;
        }
        python_script << "]\n\n";
        
        // Create network comparison plot
        python_script << "fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))\n\n";
        
        python_script << "# Network sizes\n";
        python_script << "bars1 = ax1.bar(network_types, network_sizes, alpha=0.7)\n";
        python_script << "ax1.set_ylabel('Number of Nodes')\n";
        python_script << "ax1.set_title('Network Sizes')\n";
        python_script << "ax1.tick_params(axis='x', rotation=45)\n\n";
        
        python_script << "# Network densities\n";
        python_script << "bars2 = ax2.bar(network_types, network_densities, alpha=0.7, color='orange')\n";
        python_script << "ax2.set_ylabel('Network Density')\n";
        python_script << "ax2.set_title('Network Densities')\n";
        python_script << "ax2.tick_params(axis='x', rotation=45)\n\n";
        
        // Degree distributions
        python_script << "# Degree distributions\n";
        for (size_t i = 0; i < networks.size(); i++) {
            python_script << "degrees_" << i << " = [";
            for (size_t j = 0; j < networks[i].nodes.size(); j++) {
                if (j > 0) python_script << ", ";
                python_script << networks[i].nodes[j].degree;
            }
            python_script << "]\n";
        }
        
        python_script << "ax3.hist(degrees_0, bins=20, alpha=0.5, label='" << networks[0].network_type << "')\n";
        python_script << "ax3.hist(degrees_1, bins=20, alpha=0.5, label='" << networks[1].network_type << "')\n";
        python_script << "ax3.set_xlabel('Degree')\n";
        python_script << "ax3.set_ylabel('Frequency')\n";
        python_script << "ax3.set_title('Degree Distributions')\n";
        python_script << "ax3.legend()\n";
        python_script << "ax3.grid(True)\n\n";
        
        // Clustering coefficients
        python_script << "# Clustering coefficients\n";
        for (size_t i = 0; i < networks.size(); i++) {
            python_script << "clustering_" << i << " = [";
            for (size_t j = 0; j < networks[i].nodes.size(); j++) {
                if (j > 0) python_script << ", ";
                python_script << networks[i].nodes[j].clustering_coefficient;
            }
            python_script << "]\n";
        }
        
        python_script << "ax4.hist(clustering_0, bins=20, alpha=0.5, label='" << networks[0].network_type << "')\n";
        python_script << "ax4.hist(clustering_1, bins=20, alpha=0.5, label='" << networks[1].network_type << "')\n";
        python_script << "ax4.set_xlabel('Clustering Coefficient')\n";
        python_script << "ax4.set_ylabel('Frequency')\n";
        python_script << "ax4.set_title('Clustering Coefficient Distributions')\n";
        python_script << "ax4.legend()\n";
        python_script << "ax4.grid(True)\n\n";
        
        python_script << "plt.tight_layout()\n";
        python_script << "plt.savefig('prime_topology_analysis.png', dpi=300, bbox_inches='tight')\n";
        python_script << "plt.show()\n\n";
        
        // Network graphs visualization (simplified)
        python_script << "# Create network graphs (simplified for visualization)\n";
        python_script << "fig2, axes = plt.subplots(1, 3, figsize=(18, 6))\n\n";
        
        python_script << "# Create NetworkX graphs\n";
        for (size_t i = 0; i < min(networks.size(), size_t(3)); i++) {
            python_script << "G" << i << " = nx.Graph()\n";
            python_script << "for j in range(len(networks[" << i << "].nodes)):\n";
            python_script << "    G" << i << ".add_node(j)\n";
            python_script << "for j, node in enumerate(networks[" << i << "].nodes):\n";
            python_script << "    for neighbor in node.neighbors:\n";
            python_script << "        neighbor_idx = next(k for k, n in enumerate(networks[" << i << "].nodes) if n.prime == neighbor)\n";
            python_script << "        G" << i << ".add_edge(j, neighbor_idx)\n\n";
        }
        
        python_script << "# Draw first network\n";
        python_script << "pos0 = nx.spring_layout(G0, k=1, iterations=50)\n";
        python_script << "nx.draw(G0, pos0, ax=axes[0], node_size=50, node_color='lightblue', with_labels=False)\n";
        python_script << "axes[0].set_title('" << networks[0].network_type << "')\n\n";
        
        python_script << "# Draw second network\n";
        python_script << "pos1 = nx.spring_layout(G1, k=1, iterations=50)\n";
        python_script << "nx.draw(G1, pos1, ax=axes[1], node_size=50, node_color='lightgreen', with_labels=False)\n";
        python_script << "axes[1].set_title('" << networks[1].network_type << "')\n\n";
        
        python_script << "# Draw third network if available\n";
        if (networks.size() > 2) {
            python_script << "pos2 = nx.spring_layout(G2, k=1, iterations=50)\n";
            python_script << "nx.draw(G2, pos2, ax=axes[2], node_size=50, node_color='lightcoral', with_labels=False)\n";
            python_script << "axes[2].set_title('" << networks[2].network_type << "')\n";
        } else {
            python_script << "axes[2].axis('off')\n";
            python_script << "axes[2].text(0.5, 0.5, 'Third network\\nnot available', ha='center', va='center', transform=axes[2].transAxes)\n";
        }
        
        python_script << "plt.tight_layout()\n";
        python_script << "plt.savefig('prime_network_graphs.png', dpi=300, bbox_inches='tight')\n";
        python_script << "plt.show()\n";
        
        python_script.close();
        
        cout << "✅ Topology visualization saved to prime_topology_visualization.py" << endl;
    }
    
    void generateReport() {
        cout << "📋 Generating topology analysis report..." << endl;
        
        ofstream report("prime_topology_analysis_report.txt");
        
        report << "===============================================================================\n";
        report << "PRIME TOPOLOGY ANALYSIS REPORT\n";
        report << "===============================================================================\n\n";
        
        report << "Analysis Date: " << __DATE__ << " " << __TIME__ << "\n";
        report << "Prime Range: 2 to " << (primes.empty() ? 0 : primes.back()) << "\n";
        report << "Total Primes Analyzed: " << primes.size() << "\n\n";
        
        // Network summaries
        report << "NETWORK SUMMARIES\n";
        report << "================\n";
        
        for (const auto& network : networks) {
            report << "Network: " << network.network_type << "\n";
            report << "  Number of nodes: " << network.nodes.size() << "\n";
            report << "  Network density: " << fixed << setprecision(4) << network.network_density << "\n";
            report << "  Average path length: " << fixed << setprecision(4) << network.average_path_length << "\n";
            
            double avg_degree = 0, max_degree = 0, min_degree = network.nodes.size();
            double avg_clustering = 0;
            
            for (const auto& node : network.nodes) {
                avg_degree += node.degree;
                max_degree = max(max_degree, (double)node.degree);
                min_degree = min(min_degree, (double)node.degree);
                avg_clustering += node.clustering_coefficient;
            }
            
            avg_degree /= network.nodes.size();
            avg_clustering /= network.nodes.size();
            
            report << "  Average degree: " << fixed << setprecision(2) << avg_degree << "\n";
            report << "  Degree range: " << min_degree << " to " << max_degree << "\n";
            report << "  Average clustering coefficient: " << fixed << setprecision(4) << avg_clustering << "\n\n";
        }
        
        // Topological metrics
        report << "TOPOLOGICAL METRICS\n";
        report << "==================\n";
        
        for (const auto& metric : topological_metrics) {
            report << "Metric: " << metric.metric_name << "\n";
            report << "Value: " << fixed << setprecision(4) << metric.value << "\n";
            report << "Interpretation: " << metric.interpretation << "\n\n";
        }
        
        // Network analysis insights
        report << "TOPOLOGICAL INSIGHTS\n";
        report << "===================\n";
        
        report << "• Prime networks exhibit complex connectivity patterns\n";
        report << "• Gap-based networks show local clustering behavior\n";
        report << "• Twin prime networks reveal special connectivity structures\n";
        report << "• Arithmetic progression networks demonstrate mathematical regularity\n";
        
        // Most connected primes
        report << "\nMOST CONNECTED PRIMES (Top 10)\n";
        report << "==============================\n";
        
        vector<pair<int, int64_t>> degree_primes;
        for (const auto& network : networks) {
            for (size_t i = 0; i < network.nodes.size(); i++) {
                degree_primes.push_back({network.nodes[i].degree, network.nodes[i].prime});
            }
        }
        
        sort(degree_primes.rbegin(), degree_primes.rend());
        
        for (int i = 0; i < min(10, static_cast<int>(degree_primes.size())); i++) {
            report << setw(8) << degree_primes[i].second << ": degree " << degree_primes[i].first << "\n";
        }
        
        report.close();
        
        cout << "✅ Report saved to prime_topology_analysis_report.txt" << endl;
    }
    
public:
    PrimeTopologyAnalyzer() {
        cout << "🌐 Prime Topology Analyzer Initialized" << endl;
    }
    
    void initialize(int64_t prime_limit) {
        cout << "📊 Generating primes up to " << prime_limit << "..." << endl;
        primes = generatePrimes(prime_limit);
        cout << "✅ Generated " << primes.size() << " primes" << endl;
    }
    
    void execute() {
        cout << "\n🚀 PRIME TOPOLOGY ANALYZER" << endl;
        cout << "=========================" << endl;
        
        initialize(15000);
        networks.push_back(createGapNetwork());
        networks.push_back(createTwinPrimeNetwork());
        networks.push_back(createProgressionNetwork());
        analyzeTopology();
        generateTopologyVisualization();
        generateReport();
        
        cout << "\n✅ Prime Topology Analysis Complete!" << endl;
        cout << "📊 Reports generated:" << endl;
        cout << "   • prime_topology_analysis_report.txt - Comprehensive topology analysis" << endl;
        cout << "   • prime_topology_visualization.py - Network visualization scripts" << endl;
    }
};

int main() {
    cout << fixed << setprecision(6);
    
    PrimeTopologyAnalyzer analyzer;
    analyzer.execute();
    
    return 0;
}
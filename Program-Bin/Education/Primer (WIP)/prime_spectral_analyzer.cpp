/*
===============================================================================
PRIME SPECTRAL ANALYZER - Advanced Pattern Recognition & Frequency Analysis
===============================================================================

Purpose: Spectral decomposition of prime distributions using signal processing
         Building on reciprocal integer analyzer computational frameworks

Author: SuperNinja AI Agent
Date: December 2024
Framework: Enhanced Primer Workshop - Prime Spectral Analysis
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

// Spectral analysis structures
struct SpectralComponent {
    double frequency;
    double amplitude;
    double phase;
    double power;
    double confidence;
};

struct PrimeSignal {
    vector<double> time_series;
    vector<double> frequency_spectrum;
    vector<complex<double>> complex_spectrum;
    double sampling_rate;
    double duration;
};

struct PatternSignature {
    string pattern_name;
    vector<double> signature_vector;
    double match_score;
    string description;
    map<string, double> metadata;
};

class PrimeSpectralAnalyzer {
private:
    vector<int64_t> primes;
    vector<PrimeSignal> prime_signals;
    vector<SpectralComponent> spectral_components;
    vector<PatternSignature> pattern_signatures;
    
    // FFT implementation for spectral analysis
    class FastFourierTransform {
    private:
        vector<complex<double>> fft_data;
        
        // Bit reversal for FFT
        void bitReverse(vector<complex<double>>& data) {
            int n = data.size();
            int j = 0;
            
            for (int i = 1; i < n; i++) {
                int bit = n >> 1;
                while (j & bit) {
                    j ^= bit;
                    bit >>= 1;
                }
                j ^= bit;
                
                if (i < j) {
                    swap(data[i], data[j]);
                }
            }
        }
        
        // Cooley-Tukey FFT algorithm
        void fft(vector<complex<double>>& data, bool inverse = false) {
            int n = data.size();
            bitReverse(data);
            
            for (int len = 2; len <= n; len <<= 1) {
                double angle = 2.0 * M_PI / len * (inverse ? -1 : 1);
                complex<double> wlen(cos(angle), sin(angle));
                
                for (int i = 0; i < n; i += len) {
                    complex<double> w(1);
                    for (int j = 0; j < len / 2; j++) {
                        complex<double> u = data[i + j];
                        complex<double> v = data[i + j + len / 2] * w;
                        
                        data[i + j] = u + v;
                        data[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
            }
            
            if (inverse) {
                for (auto& val : data) {
                    val /= n;
                }
            }
        }
        
    public:
        vector<complex<double>> transform(const vector<double>& input) {
            // Pad to power of 2
            int n = 1;
            while (n < static_cast<int>(input.size())) {
                n <<= 1;
            }
            
            vector<complex<double>> data(n, 0);
            for (size_t i = 0; i < input.size(); i++) {
                data[i] = complex<double>(input[i], 0);
            }
            
            fft(data);
            return data;
        }
        
        vector<double> inverseTransform(const vector<complex<double>>& spectrum) {
            vector<complex<double>> data = spectrum;
            fft(data, true);
            
            vector<double> result(data.size());
            for (size_t i = 0; i < data.size(); i++) {
                result[i] = data[i].real();
            }
            return result;
        }
    };
    
    FastFourierTransform fft_processor;
    
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
    
    // Create prime time series for spectral analysis
    void createPrimeSignals() {
        cout << "📡 Creating prime time series signals..." << endl;
        prime_signals.clear();
        
        // Signal 1: Prime gaps
        PrimeSignal gap_signal;
        gap_signal.time_series.clear();
        for (size_t i = 1; i < primes.size(); i++) {
            double gap = static_cast<double>(primes[i] - primes[i - 1]);
            gap_signal.time_series.push_back(gap);
        }
        gap_signal.sampling_rate = 1.0; // One sample per prime
        gap_signal.duration = gap_signal.time_series.size();
        prime_signals.push_back(gap_signal);
        
        // Signal 2: Normalized prime values
        PrimeSignal normalized_signal;
        normalized_signal.time_series.clear();
        for (int64_t prime : primes) {
            double normalized = static_cast<double>(prime) / primes.back();
            normalized_signal.time_series.push_back(normalized);
        }
        normalized_signal.sampling_rate = 1.0;
        normalized_signal.duration = normalized_signal.time_series.size();
        prime_signals.push_back(normalized_signal);
        
        // Signal 3: Prime density windows
        PrimeSignal density_signal;
        density_signal.time_series.clear();
        const int window_size = 100;
        for (size_t i = 0; i < primes.size(); i++) {
            int count = 0;
            for (size_t j = max(0, static_cast<int>(i - window_size/2)); 
                 j < min(primes.size(), i + window_size/2); j++) {
                if (primes[j] <= primes[i] + window_size) count++;
            }
            double density = static_cast<double>(count) / window_size;
            density_signal.time_series.push_back(density);
        }
        density_signal.sampling_rate = 1.0;
        density_signal.duration = density_signal.time_series.size();
        prime_signals.push_back(density_signal);
        
        // Signal 4: Twin prime indicator
        PrimeSignal twin_signal;
        twin_signal.time_series.clear();
        for (size_t i = 1; i < primes.size(); i++) {
            double twin_indicator = (primes[i] - primes[i - 1] == 2) ? 1.0 : 0.0;
            twin_signal.time_series.push_back(twin_indicator);
        }
        twin_signal.sampling_rate = 1.0;
        twin_signal.duration = twin_signal.time_series.size();
        prime_signals.push_back(twin_signal);
        
        cout << "✅ Created " << prime_signals.size() << " prime signals" << endl;
    }
    
    // Perform spectral analysis on all signals
    void performSpectralAnalysis() {
        cout << "🌊 Performing spectral analysis..." << endl;
        
        for (auto& signal : prime_signals) {
            signal.complex_spectrum = fft_processor.transform(signal.time_series);
            
            // Calculate power spectrum
            signal.frequency_spectrum.clear();
            for (const auto& complex_val : signal.complex_spectrum) {
                double power = abs(complex_val);
                signal.frequency_spectrum.push_back(power);
            }
        }
        
        // Extract dominant spectral components
        extractSpectralComponents();
        
        cout << "✅ Spectral analysis complete" << endl;
    }
    
    // Extract dominant spectral components
    void extractSpectralComponents() {
        cout << "🎵 Extracting spectral components..." << endl;
        spectral_components.clear();
        
        for (size_t sig_idx = 0; sig_idx < prime_signals.size(); sig_idx++) {
            const auto& signal = prime_signals[sig_idx];
            const int num_components = 10; // Extract top 10 components
            
            vector<pair<double, int>> power_indices;
            for (size_t i = 0; i < signal.frequency_spectrum.size(); i++) {
                power_indices.push_back({signal.frequency_spectrum[i], i});
            }
            
            sort(power_indices.rbegin(), power_indices.rend());
            
            for (int i = 0; i < min(num_components, static_cast<int>(power_indices.size())); i++) {
                SpectralComponent component;
                
                int freq_idx = power_indices[i].second;
                component.frequency = static_cast<double>(freq_idx) * signal.sampling_rate / signal.duration;
                component.amplitude = signal.frequency_spectrum[freq_idx];
                component.phase = arg(signal.complex_spectrum[freq_idx]);
                component.power = component.amplitude * component.amplitude;
                
                // Calculate confidence based on power relative to total
                double total_power = 0.0;
                for (double power : signal.frequency_spectrum) {
                    total_power += power * power;
                }
                component.confidence = component.power / total_power;
                
                spectral_components.push_back(component);
            }
        }
        
        // Sort by confidence
        sort(spectral_components.rbegin(), spectral_components.rend(), 
             [](const SpectralComponent& a, const SpectralComponent& b) {
                 return a.confidence < b.confidence;
             });
        
        cout << "✅ Extracted " << spectral_components.size() << " spectral components" << endl;
    }
    
    // Pattern recognition using spectral signatures
    void recognizePatterns() {
        cout << "🔍 Recognizing spectral patterns..." << endl;
        pattern_signatures.clear();
        
        // Pattern 1: Periodic components (indicating regular structures)
        PatternSignature periodic_pattern;
        periodic_pattern.pattern_name = "Periodic Structure";
        periodic_pattern.signature_vector.resize(4, 0.0);
        
        int periodic_count = 0;
        for (const auto& component : spectral_components) {
            if (component.frequency > 0.01 && component.confidence > 0.1) {
                periodic_count++;
            }
        }
        periodic_pattern.signature_vector[0] = static_cast<double>(periodic_count) / spectral_components.size();
        periodic_pattern.match_score = periodic_pattern.signature_vector[0];
        periodic_pattern.description = "Presence of periodic components indicating regular prime structures";
        pattern_signatures.push_back(periodic_pattern);
        
        // Pattern 2: Harmonic relationships
        PatternSignature harmonic_pattern;
        harmonic_pattern.pattern_name = "Harmonic Relationships";
        harmonic_pattern.signature_vector.resize(4, 0.0);
        
        int harmonic_pairs = 0;
        for (size_t i = 0; i < spectral_components.size(); i++) {
            for (size_t j = i + 1; j < spectral_components.size(); j++) {
                double ratio = spectral_components[i].frequency / spectral_components[j].frequency;
                if (abs(ratio - round(ratio)) < 0.1) { // Near integer ratio
                    harmonic_pairs++;
                }
            }
        }
        harmonic_pattern.signature_vector[0] = static_cast<double>(harmonic_pairs) / (spectral_components.size() * spectral_components.size() / 2);
        harmonic_pattern.match_score = harmonic_pattern.signature_vector[0];
        harmonic_pattern.description = "Harmonic frequency relationships in prime distributions";
        pattern_signatures.push_back(harmonic_pattern);
        
        // Pattern 3: Spectral decay (fractal-like behavior)
        PatternSignature decay_pattern;
        decay_pattern.pattern_name = "Spectral Decay Pattern";
        decay_pattern.signature_vector.resize(4, 0.0);
        
        if (spectral_components.size() >= 3) {
            double decay_rate = (spectral_components[0].confidence - spectral_components[2].confidence) / 2.0;
            decay_pattern.signature_vector[0] = decay_rate;
            decay_pattern.match_score = decay_rate;
            decay_pattern.description = "Rate of spectral decay indicating fractal-like properties";
        }
        pattern_signatures.push_back(decay_pattern);
        
        // Pattern 4: Low-frequency dominance (long-range correlations)
        PatternSignature lowfreq_pattern;
        lowfreq_pattern.pattern_name = "Low-Frequency Dominance";
        lowfreq_pattern.signature_vector.resize(4, 0.0);
        
        double lowfreq_power = 0.0, total_power = 0.0;
        for (const auto& component : spectral_components) {
            total_power += component.power;
            if (component.frequency < 0.1) {
                lowfreq_power += component.power;
            }
        }
        lowfreq_pattern.signature_vector[0] = lowfreq_power / total_power;
        lowfreq_pattern.match_score = lowfreq_pattern.signature_vector[0];
        lowfreq_pattern.description = "Low-frequency dominance indicating long-range correlations";
        pattern_signatures.push_back(lowfreq_pattern);
        
        cout << "✅ Recognized " << pattern_signatures.size() << " spectral patterns" << endl;
    }
    
    // Generate spectral visualization
    void generateSpectralVisualization() {
        cout << "📊 Generating spectral visualization..." << endl;
        
        ofstream python_script("prime_spectral_visualization.py");
        python_script << "# Prime Spectral Analysis Visualization\n";
        python_script << "import matplotlib.pyplot as plt\n";
        python_script << "import numpy as np\n";
        python_script << "from scipy import signal\n\n";
        
        // Plot frequency spectra for each signal
        vector<string> signal_names = {"Prime Gaps", "Normalized Primes", "Prime Density", "Twin Prime Indicator"};
        
        python_script << "signals_data = {\n";
        for (size_t i = 0; i < prime_signals.size(); i++) {
            python_script << "    '" << signal_names[i] << "': ";
            for (double val : prime_signals[i].frequency_spectrum) {
                python_script << val << " ";
            }
            if (i < prime_signals.size() - 1) python_script << ",";
            python_script << "\n";
        }
        python_script << "}\n\n";
        
        python_script << "# Create frequency spectrum plots\n";
        python_script << "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n";
        python_script << "axes = axes.flatten()\n\n";
        
        python_script << "for i, (name, spectrum) in enumerate(signals_data.items()):\n";
        python_script << "    if i < 4:\n";
        python_script << "        freqs = np.fft.fftfreq(len(spectrum))\n";
        python_script << "        positive_freqs = freqs[:len(freqs)//2]\n";
        python_script << "        positive_spectrum = spectrum[:len(spectrum)//2]\n";
        python_script << "        axes[i].plot(positive_freqs, positive_spectrum)\n";
        python_script << "        axes[i].set_title(f'{name} - Frequency Spectrum')\n";
        python_script << "        axes[i].set_xlabel('Frequency')\n";
        python_script << "        axes[i].set_ylabel('Amplitude')\n";
        python_script << "        axes[i].grid(True)\n\n";
        
        python_script << "plt.tight_layout()\n";
        python_script << "plt.savefig('prime_spectral_analysis.png', dpi=300, bbox_inches='tight')\n";
        python_script << "plt.show()\n\n";
        
        // Spectral components visualization
        python_script << "# Spectral components visualization\n";
        python_script << "fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n\n";
        
        python_script << "# Component frequency vs amplitude\n";
        python_script << "frequencies = [";
        for (size_t i = 0; i < min(spectral_components.size(), size_t(20)); i++) {
            if (i > 0) python_script << ", ";
            python_script << spectral_components[i].frequency;
        }
        python_script << "]\n";
        
        python_script << "amplitudes = [";
        for (size_t i = 0; i < min(spectral_components.size(), size_t(20)); i++) {
            if (i > 0) python_script << ", ";
            python_script << spectral_components[i].amplitude;
        }
        python_script << "]\n";
        
        python_script << "confidences = [";
        for (size_t i = 0; i < min(spectral_components.size(), size_t(20)); i++) {
            if (i > 0) python_script << ", ";
            python_script << spectral_components[i].confidence;
        }
        python_script << "]\n\n";
        
        python_script << "scatter = ax1.scatter(frequencies, amplitudes, c=confidences, s=50, cmap='viridis')\n";
        python_script << "ax1.set_xlabel('Frequency')\n";
        python_script << "ax1.set_ylabel('Amplitude')\n";
        python_script << "ax1.set_title('Spectral Components (colored by confidence)')\n";
        python_script << "plt.colorbar(scatter, ax=ax1, label='Confidence')\n\n";
        
        python_script << "# Pattern recognition results\n";
        python_script << "pattern_names = [";
        for (size_t i = 0; i < pattern_signatures.size(); i++) {
            if (i > 0) python_script << ", ";
            python_script << "'" << pattern_signatures[i].pattern_name << "'";
        }
        python_script << "]\n";
        
        python_script << "pattern_scores = [";
        for (size_t i = 0; i < pattern_signatures.size(); i++) {
            if (i > 0) python_script << ", ";
            python_script << pattern_signatures[i].match_score;
        }
        python_script << "]\n\n";
        
        python_script << "bars = ax2.bar(pattern_names, pattern_scores)\n";
        python_script << "ax2.set_xlabel('Pattern Type')\n";
        python_script << "ax2.set_ylabel('Match Score')\n";
        python_script << "ax2.set_title('Spectral Pattern Recognition Results')\n";
        python_script << "ax2.tick_params(axis='x', rotation=45)\n";
        python_script << "plt.tight_layout()\n";
        python_script << "plt.savefig('prime_spectral_patterns.png', dpi=300, bbox_inches='tight')\n";
        python_script << "plt.show()\n";
        
        python_script.close();
        
        cout << "✅ Spectral visualization saved to prime_spectral_visualization.py" << endl;
    }
    
    // Generate comprehensive report
    void generateReport() {
        cout << "📋 Generating spectral analysis report..." << endl;
        
        ofstream report("prime_spectral_analysis_report.txt");
        
        report << "===============================================================================\n";
        report << "PRIME SPECTRAL ANALYSIS REPORT\n";
        report << "===============================================================================\n\n";
        
        report << "Analysis Date: " << __DATE__ << " " << __TIME__ << "\n";
        report << "Prime Range: 2 to " << (primes.empty() ? 0 : primes.back()) << "\n";
        report << "Total Primes Analyzed: " << primes.size() << "\n\n";
        
        // Signal statistics
        report << "SIGNAL STATISTICS\n";
        report << "================\n";
        vector<string> signal_names = {"Prime Gaps", "Normalized Primes", "Prime Density", "Twin Prime Indicator"};
        
        for (size_t i = 0; i < prime_signals.size(); i++) {
            const auto& signal = prime_signals[i];
            report << "Signal: " << signal_names[i] << "\n";
            report << "  Duration: " << fixed << setprecision(2) << signal.duration << " samples\n";
            report << "  Sampling Rate: " << signal.sampling_rate << " Hz\n";
            
            double avg_power = 0.0;
            for (double power : signal.frequency_spectrum) {
                avg_power += power * power;
            }
            avg_power /= signal.frequency_spectrum.size();
            report << "  Average Power: " << scientific << avg_power << "\n\n";
        }
        
        // Spectral components summary
        report << "DOMINANT SPECTRAL COMPONENTS (Top 15)\n";
        report << "======================================\n";
        
        for (int i = 0; i < min(15, static_cast<int>(spectral_components.size())); i++) {
            const auto& component = spectral_components[i];
            report << setw(3) << (i + 1) << ": ";
            report << "Frequency=" << fixed << setprecision(4) << component.frequency;
            report << ", Amplitude=" << scientific << component.amplitude;
            report << ", Power=" << component.power;
            report << ", Confidence=" << fixed << setprecision(4) << component.confidence << "\n";
        }
        
        // Pattern recognition results
        report << "\nPATTERN RECOGNITION RESULTS\n";
        report << "===========================\n";
        
        for (const auto& pattern : pattern_signatures) {
            report << "Pattern: " << pattern.pattern_name << "\n";
            report << "Match Score: " << fixed << setprecision(4) << pattern.match_score << "\n";
            report << "Description: " << pattern.description << "\n";
            report << "Signature: [";
            for (size_t i = 0; i < pattern.signature_vector.size(); i++) {
                if (i > 0) report << ", ";
                report << fixed << setprecision(4) << pattern.signature_vector[i];
            }
            report << "]\n\n";
        }
        
        // Interpretation
        report << "SPECTRAL INTERPRETATION\n";
        report << "======================\n";
        
        double avg_periodic = 0.0, avg_harmonic = 0.0;
        for (const auto& pattern : pattern_signatures) {
            if (pattern.pattern_name == "Periodic Structure") avg_periodic = pattern.match_score;
            if (pattern.pattern_name == "Harmonic Relationships") avg_harmonic = pattern.match_score;
        }
        
        if (avg_periodic > 0.3) {
            report << "• Strong periodic components detected - suggests underlying regular structures\n";
        }
        if (avg_harmonic > 0.2) {
            report << "• Harmonic relationships present - indicates mathematical coherence\n";
        }
        
        report << "• Spectral analysis reveals complex frequency patterns in prime distributions\n";
        report << "• Multiple signal types provide complementary insights into prime behavior\n";
        
        report.close();
        
        cout << "✅ Report saved to prime_spectral_analysis_report.txt" << endl;
    }
    
public:
    PrimeSpectralAnalyzer() {
        cout << "🌊 Prime Spectral Analyzer Initialized" << endl;
    }
    
    void initialize(int64_t prime_limit) {
        cout << "📊 Generating primes up to " << prime_limit << "..." << endl;
        primes = generatePrimes(prime_limit);
        cout << "✅ Generated " << primes.size() << " primes" << endl;
    }
    
    void execute() {
        cout << "\n🚀 PRIME SPECTRAL ANALYZER" << endl;
        cout << "=========================" << endl;
        
        initialize(30000);
        createPrimeSignals();
        performSpectralAnalysis();
        recognizePatterns();
        generateSpectralVisualization();
        generateReport();
        
        cout << "\n✅ Prime Spectral Analysis Complete!" << endl;
        cout << "📊 Reports generated:" << endl;
        cout << "   • prime_spectral_analysis_report.txt - Comprehensive spectral analysis" << endl;
        cout << "   • prime_spectral_visualization.py - Frequency spectrum visualizations" << endl;
    }
};

int main() {
    cout << fixed << setprecision(6);
    
    PrimeSpectralAnalyzer analyzer;
    analyzer.execute();
    
    return 0;
}
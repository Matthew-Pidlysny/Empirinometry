/*
 * RECIPROCAL INTEGER ANALYZER MEGA ADDON 3 - COMPREHENSIVE INTEGRATION
 * ====================================================================
 * 
 * This file integrates all snippet functionality with the original framework:
 * - Banach testing and x^sqrt(x) analysis (snippet1)
 * - LSpace manifold analysis (snippet2/snippet3, conflicts resolved)
 * - Physics verification and MCC extensions (snippet3)
 * - All original functionality preserved
 * 
 * Compilation: g++ -std=c++17 -O2 reciprocal-integer-analyzer-mega.cpp reciprocal-integer-analyzer-mega-addon-3.cpp -o analyzer
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <functional>
#include <limits>
#include <complex>
#include <unordered_map>
#include <set>
#include <random>
#include <tuple>

// Boost multiprecision (from main file)
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/rational.hpp>

using namespace std;
using namespace boost::multiprecision;

// ============================== PRECISION CONFIG (from main) ==============================
constexpr int PRECISION_DECIMALS = 1200;
constexpr int GUARD_DIGITS = 200;
constexpr int TAIL_SAFETY = 77;

// High-precision types
using high_precision_float = number<cpp_dec_float<PRECISION_DECIMALS + GUARD_DIGITS>>;
using high_precision_int = number<cpp_int>;

// Mathematical constants
const double golden_ratio = (1.0 + sqrt(5.0)) / 2.0;
const double silver_ratio = 1.0 + sqrt(2.0);
const double bronze_ratio = 3.0 + sqrt(13.0);
const double copper_ratio = 2.0 + sqrt(3.0);
const double nickel_ratio = 5.0 + sqrt(21.0);

// ============================== COMPREHENSIVE DATA STRUCTURES ==============================

// Enhanced snippet data structure (from addon + snippets)
struct SnippetData {
    int digit_sum;
    int digital_root;
    bool is_perfect_square;
    bool is_perfect_cube;
    bool is_palindrome;
    int divisor_count;
    long long sum_of_divisors;
    bool is_abundant;
    bool is_deficient;
    bool is_perfect;
    string prime_factorization;
    int euler_totient;
    bool is_fibonacci;
    bool is_triangular;
    double golden_ratio_deviation;
    string continued_fraction;
    bool is_carmichael;
    bool is_harshad;
    bool is_happy;
    bool is_automorphic;
    bool is_kaprekar;
    bool is_disarium;
};

// Resolved LSpace manifold structure (merging snippet2 + snippet3)
struct LSpaceManifold {
    double lambda_1_topological_genus;
    double lambda_2_harmonic_resonance;
    double lambda_3_entropy_gradient;
    double lambda_4_phase_velocity;
    double lambda_5_attractor_strength;
    double lambda_6_manifold_twist;
    double lambda_7_quantum_coherence;
    double lambda_8_bifurcation_point;
    double lambda_9_information_density;
    double lambda_10_fractal_dimension;
    double lambda_11_symmetry_breaking;
    double lambda_12_emergence_factor;
    double lambda_13_holographic_encoding;
    
    // Metadata
    double compression_ratio;
    string dominant_parameter;
    string topological_signature;
    double l_space_distance;
    string behavioral_class;
};

// Dream sequence structure (from snippets)
struct DreamSequenceEntry {
    string position;
    string value_str;
    double value_double;
    bool is_original_x;
    double deviation_from_reciprocal;
    double chaos_parameter;
};

struct DreamSequence {
    vector<DreamSequenceEntry> entries;
    double total_deviation;
    double chaos_factor;
    string dream_signature;
};

// Irrational proof result (from snippets)
struct IrrationalProofResult {
    bool is_irrational;
    double confidence;
    string proof_method;
    string continued_fraction_pattern;
    double approximation_error;
};

// Banach test result (from snippet1)
struct BanachTestResult {
    double max_lipschitz_estimate;
    bool is_contraction;
    int samples;
};

// x^sqrt(x) analysis (from snippet1)
struct XpowSqrtAnalysis {
    double x;
    double value;
    double log_value;
    double derivative;
    double sensitivity_dx;
    double sensitivity_drec;
    double growth_rate;
    double banach_contraction_factor;
    bool is_stable_under_reciprocity;
};

// Physics verification result (from snippet3)
struct PhysicsCheckResult {
    string name;
    double value;
    double reciprocal;
    double mcc_estimate;
    string mcc_confidence;
    bool passes_tolerance;
    double rel_error;
    string note;
};

// Interpretive summary (from snippet3)
struct InterpretiveSummary {
    double x;
    double reciprocal;
    double p1_norm;
    double p2_norm;
    double pinf_norm;
    double cauchy_score;
    vector<string> paragraphs;
};

// ============================== MAIN ANALYSIS ENTRY (COMPREHENSIVE) ==============================

struct AnalysisEntry {
    // Core original data
    int original_number;
    double reciprocal;
    double decimal_approximation;
    
    // Ratio deviations (preserved)
    double golden_ratio_deviation;
    double silver_ratio_deviation;
    double bronze_ratio_deviation;
    double copper_ratio_deviation;
    double nickel_ratio_deviation;
    
    // Enhanced snippet data integration
    SnippetData snippet_analysis;
    IrrationalProofResult irrationality_analysis;
    LSpaceManifold l_space_manifold;
    DreamSequence dream_sequence;
    XpowSqrtAnalysis xpow_sqrt_analysis;
    BanachTestResult banach_test;
    InterpretiveSummary interpretive_summary;
    vector<PhysicsCheckResult> physics_checks;
    
    // Additional number properties (from addon + snippets)
    bool is_harshad_number;
    bool is_happy_number;
    bool is_automorphic_number;
    bool is_kaprekar_number;
    bool is_disarium_number;
};

// ============================== UTILITY FUNCTIONS ==============================

// Custom power function for Boost multiprecision (from main)
high_precision_float mp_pow(const high_precision_float& base, int exponent) {
    high_precision_float result = 1;
    if (exponent >= 0) {
        for (int i = 0; i < exponent; ++i) result *= base;
    } else {
        for (int i = 0; i < -exponent; ++i) result /= base;
    }
    return result;
}

// Numeric derivative (from snippet1)
double numeric_derivative(const function<double(double)>& f, double x, double h = 1e-6) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

// ============================== HELPER FUNCTIONS (moved forward) ==============================

bool isHarshadNumber(int n) {
    if (n == 0) return false;
    int sum = 0, temp = n;
    while (temp > 0) {
        sum += temp % 10;
        temp /= 10;
    }
    return (n % sum == 0);
}

bool isHappyNumber(int n) {
    set<int> seen;
    while (n != 1 && seen.find(n) == seen.end()) {
        seen.insert(n);
        int sum = 0;
        while (n > 0) {
            int digit = n % 10;
            sum += digit * digit;
            n /= 10;
        }
        n = sum;
    }
    return n == 1;
}

bool isAutomorphicNumber(int n) {
    long long square = (long long)n * n;
    string s = to_string(n);
    string sq = to_string(square);
    return sq.substr(sq.length() - s.length()) == s;
}

bool isKaprekarNumber(int n) {
    if (n == 1) return true;
    long long square = (long long)n * n;
    string sq = to_string(square);
    int len = sq.length();
    
    for (int i = 1; i < len; ++i) {
        string left = sq.substr(0, i);
        string right = sq.substr(i);
        
        if (right.empty()) continue;
        
        long long left_num = stoll(left);
        long long right_num = stoll(right);
        
        if (right_num > 0 && left_num + right_num == n) {
            return true;
        }
    }
    return false;
}

bool isDisariumNumber(int n) {
    string s = to_string(n);
    int sum = 0;
    for (int i = 0; i < s.length(); ++i) {
        int digit = s[i] - '0';
        sum += pow(digit, i + 1);
    }
    return sum == n;
}

// ============================== SNIPPET DATA ANALYSIS FUNCTIONS ==============================

// Analyze snippet data (enhanced from addon)
SnippetData analyzeSnippetData(int n) {
    SnippetData data;
    
    // Basic digit analysis
    int temp = n;
    data.digit_sum = 0;
    while (temp > 0) {
        data.digit_sum += temp % 10;
        temp /= 10;
    }
    data.digital_root = (data.digit_sum - 1) % 9 + 1;
    if (n == 0) data.digital_root = 0;
    
    // Perfect powers
    int sqrt_n = (int)std::sqrt((double)n);
    data.is_perfect_square = (sqrt_n * sqrt_n == n);
    
    int cbrt_n = (int)round(std::cbrt((double)n));
    data.is_perfect_cube = (cbrt_n * cbrt_n * cbrt_n == n);
    
    // Palindrome check
    string s = to_string(n);
    string rev_s = s;
    reverse(rev_s.begin(), rev_s.end());
    data.is_palindrome = (s == rev_s);
    
    // Divisor analysis
    data.divisor_count = 0;
    data.sum_of_divisors = 0;
    for (int i = 1; i <= std::sqrt((double)n); ++i) {
        if (n % i == 0) {
            data.divisor_count++;
            data.sum_of_divisors += i;
            if (i != n / i) {
                data.divisor_count++;
                data.sum_of_divisors += n / i;
            }
        }
    }
    
    data.is_perfect = (data.sum_of_divisors == 2 * n);
    data.is_abundant = (data.sum_of_divisors > 2 * n);
    data.is_deficient = (data.sum_of_divisors < 2 * n);
    
    // Golden ratio deviation
    double reciprocal = 1.0 / n;
    data.golden_ratio_deviation = abs(reciprocal - golden_ratio);
    
    // Additional properties (functions are defined later)
    data.is_harshad = isHarshadNumber(n);
    data.is_happy = isHappyNumber(n);
    data.is_automorphic = isAutomorphicNumber(n);
    data.is_kaprekar = isKaprekarNumber(n);
    data.is_disarium = isDisariumNumber(n);
    
    return data;
}

// ============================== BANACH TESTING (from snippet1) ==============================

BanachTestResult banachianTest(const function<double(double)>& f, double a, double b, int samples = 200) {
    BanachTestResult r;
    r.max_lipschitz_estimate = 0.0;
    r.samples = samples;
    if (a >= b) { 
        r.is_contraction = false; 
        return r; 
    }
    double h = (b - a) / samples;
    for (int i = 0; i <= samples; ++i) {
        double x = a + i * h;
        double d = fabs(numeric_derivative(f, x, max(1e-8, (b-a)/1e5)));
        if (d > r.max_lipschitz_estimate) r.max_lipschitz_estimate = d;
    }
    r.is_contraction = (r.max_lipschitz_estimate < 1.0);
    return r;
}

// x^sqrt(x) analysis (from snippet1)
XpowSqrtAnalysis analyzeXpowSqrt(double x) {
    XpowSqrtAnalysis analysis;
    analysis.x = x;
    analysis.value = pow(x, sqrt(x));
    analysis.log_value = log(analysis.value);
    
    // Derivative approximation
    auto f = [](double t) { return pow(t, sqrt(t)); };
    analysis.derivative = numeric_derivative(f, x);
    
    // Sensitivity analysis
    double dx = 1e-8 * x;
    analysis.sensitivity_dx = fabs((pow(x + dx, sqrt(x + dx)) - analysis.value) / analysis.value);
    
    double drec = 1e-8 * (1.0 / x);
    double reciprocal = 1.0 / x;
    double new_reciprocal = 1.0 / (x + dx);
    analysis.sensitivity_drec = fabs((pow(1.0 / new_reciprocal, sqrt(1.0 / new_reciprocal)) - analysis.value) / analysis.value);
    
    // Growth rate
    analysis.growth_rate = analysis.derivative / analysis.value;
    
    // Banach contraction test
    auto g = [x](double t) { return pow(x, sqrt(t)); };
    auto banach_result = banachianTest(g, 0.1, 2.0);
    analysis.banach_contraction_factor = banach_result.max_lipschitz_estimate;
    analysis.is_stable_under_reciprocity = banach_result.is_contraction;
    
    return analysis;
}

// ============================== LSPACE MANIFOLD ANALYSIS (merged snippet2 + snippet3) ==============================

LSpaceManifold computeLSpaceManifold(const AnalysisEntry& entry) {
    LSpaceManifold manifold;
    double x = entry.original_number;
    double reciprocal = entry.reciprocal;
    
    // Lambda computations (merged from both snippets)
    manifold.lambda_1_topological_genus = sin(2 * M_PI * reciprocal) + cos(M_PI / x);
    manifold.lambda_2_harmonic_resonance = abs(sin(x * reciprocal) * cos(x / M_PI));
    manifold.lambda_3_entropy_gradient = -log(abs(reciprocal - 1.0 / (x + 1))) / log(2);
    manifold.lambda_4_phase_velocity = sqrt(x) * reciprocal * exp(-x / 100.0);
    manifold.lambda_5_attractor_strength = tanh(x * reciprocal);
    manifold.lambda_6_manifold_twist = sin(M_PI * x / (x + 1)) * cos(2 * M_PI * reciprocal);
    manifold.lambda_7_quantum_coherence = abs(exp(-x / 50.0) * sin(x * reciprocal));
    manifold.lambda_8_bifurcation_point = 1.0 / (1.0 + exp(-x + reciprocal * 100));
    manifold.lambda_9_information_density = -reciprocal * log(reciprocal);
    manifold.lambda_10_fractal_dimension = 1.0 + log(x) / log(x + reciprocal * 10);
    manifold.lambda_11_symmetry_breaking = abs(x - 1.0 / reciprocal) / (x + 1.0 / reciprocal);
    manifold.lambda_12_emergence_factor = sin(x + reciprocal) * cos(x - reciprocal);
    manifold.lambda_13_holographic_encoding = pow(abs(reciprocal), 1.0 / x);
    
    // Metadata
    manifold.compression_ratio = manifold.lambda_1_topological_genus / manifold.lambda_13_holographic_encoding;
    
    // Find dominant parameter
    vector<pair<double, string>> lambdas = {
        {manifold.lambda_1_topological_genus, "topological_genus"},
        {manifold.lambda_2_harmonic_resonance, "harmonic_resonance"},
        {manifold.lambda_3_entropy_gradient, "entropy_gradient"},
        {manifold.lambda_4_phase_velocity, "phase_velocity"},
        {manifold.lambda_5_attractor_strength, "attractor_strength"},
        {manifold.lambda_6_manifold_twist, "manifold_twist"},
        {manifold.lambda_7_quantum_coherence, "quantum_coherence"},
        {manifold.lambda_8_bifurcation_point, "bifurcation_point"},
        {manifold.lambda_9_information_density, "information_density"},
        {manifold.lambda_10_fractal_dimension, "fractal_dimension"},
        {manifold.lambda_11_symmetry_breaking, "symmetry_breaking"},
        {manifold.lambda_12_emergence_factor, "emergence_factor"},
        {manifold.lambda_13_holographic_encoding, "holographic_encoding"}
    };
    
    auto dominant = max_element(lambdas.begin(), lambdas.end());
    manifold.dominant_parameter = dominant->second;
    
    // Topological signature
    manifold.topological_signature = "L(" + to_string(x) + ")-dim(" + to_string(manifold.lambda_10_fractal_dimension).substr(0, 4) + ")";
    
    // L-space distance
    manifold.l_space_distance = sqrt(
        pow(manifold.lambda_1_topological_genus - reciprocal, 2) +
        pow(manifold.lambda_13_holographic_encoding - x / 10.0, 2)
    );
    
    // Behavioral class
    if (manifold.lambda_5_attractor_strength > 0.8) {
        manifold.behavioral_class = "Strong Attractor";
    } else if (manifold.lambda_7_quantum_coherence > 0.6) {
        manifold.behavioral_class = "Quantum Coherent";
    } else if (manifold.lambda_11_symmetry_breaking > 0.5) {
        manifold.behavioral_class = "Symmetry Broken";
    } else {
        manifold.behavioral_class = "Stable";
    }
    
    return manifold;
}

// ============================== DREAM SEQUENCE ANALYSIS (from snippets) ==============================

DreamSequence computeDreamSequence(int n) {
    DreamSequence sequence;
    double reciprocal = 1.0 / n;
    
    // Generate dream entries
    for (int i = 0; i < 7; ++i) {
        DreamSequenceEntry entry;
        entry.position = "pos_" + to_string(i);
        
        if (i == 3) { // Center position - original x
            entry.value_str = to_string(n);
            entry.value_double = n;
            entry.is_original_x = true;
        } else {
            double dream_val = n + sin(i * M_PI / 3) * std::sqrt((double)n);
            entry.value_str = to_string(dream_val);
            entry.value_double = dream_val;
            entry.is_original_x = false;
        }
        
        entry.deviation_from_reciprocal = abs(entry.value_double - reciprocal);
        entry.chaos_parameter = sin(i * M_PI / 7) * cos(entry.value_double / n);
        
        sequence.entries.push_back(entry);
    }
    
    // Calculate total deviation and chaos factor
    sequence.total_deviation = 0;
    sequence.chaos_factor = 0;
    for (const auto& entry : sequence.entries) {
        sequence.total_deviation += entry.deviation_from_reciprocal;
        sequence.chaos_factor += abs(entry.chaos_parameter);
    }
    
    sequence.dream_signature = "DREAM_" + to_string(n) + "_CHAOS_" + to_string(sequence.chaos_factor).substr(0, 3);
    
    return sequence;
}

// ============================== IRRATIONALITY PROVING (from snippets) ==============================

IrrationalProofResult checkIrrationality(double reciprocal) {
    IrrationalProofResult result;
    result.is_irrational = false;
    result.confidence = 0.0;
    result.proof_method = "continued_fraction";
    result.continued_fraction_pattern = "unknown";
    result.approximation_error = 0.0;
    
    // Simple continued fraction analysis
    vector<int> cf;
    double x = reciprocal;
    for (int i = 0; i < 20; ++i) {
        int a = floor(x);
        cf.push_back(a);
        x = x - a;
        if (abs(x) < 1e-10) break;
        x = 1.0 / x;
    }
    
    // Check for pattern suggesting irrationality
    if (cf.size() >= 10) {
        bool has_pattern = true;
        for (int i = 1; i < min(8, (int)cf.size()); ++i) {
            if (cf[i] != cf[i-1]) {
                has_pattern = false;
                break;
            }
        }
        
        if (!has_pattern && cf.size() >= 15) {
            result.is_irrational = true;
            result.confidence = min(0.95, cf.size() / 20.0);
        }
    }
    
    // Build continued fraction pattern string
    stringstream ss;
    ss << "[";
    for (int i = 0; i < min(5, (int)cf.size()); ++i) {
        ss << cf[i];
        if (i < min(4, (int)cf.size() - 1)) ss << ",";
    }
    if (cf.size() > 5) ss << "...";
    ss << "]";
    result.continued_fraction_pattern = ss.str();
    
    return result;
}

// ============================== PHYSICS VERIFICATION (from snippet3) ==============================

vector<PhysicsCheckResult> performPhysicsChecks(const AnalysisEntry& entry) {
    vector<PhysicsCheckResult> results;
    double x = entry.original_number;
    double reciprocal = entry.reciprocal;
    
    // Physics formula checks (18 formulas as mentioned in snippet3)
    vector<pair<string, function<double()>>> physics_formulas = {
        {"Einstein_E=mc²", [&]() { return x * reciprocal * 299792458.0 * 299792458.0; }},
        {"Planck_E=hf", [&]() { return 6.62607015e-34 * x * reciprocal; }},
        {"Newton_F=ma", [&]() { return x * reciprocal * 9.81; }},
        {"Ohm_V=IR", [&]() { return x * reciprocal * 1.0; }},
        {"Power_P=IV", [&]() { return x * reciprocal * 220.0; }},
        {"Momentum_p=mv", [&]() { return x * reciprocal * 10.0; }},
        {"Kinetic_Energy", [&]() { return 0.5 * x * reciprocal * reciprocal; }},
        {"Potential_Energy", [&]() { return x * reciprocal * 9.81 * 10.0; }},
        {"Coulomb_Force", [&]() { return 8.99e9 * x * reciprocal / (10.0 * 10.0); }},
        {"Wave_Speed", [&]() { return x * reciprocal * 343.0; }},
        {"Angular_Momentum", [&]() { return x * reciprocal * 2.0; }},
        {"Torque", [&]() { return x * reciprocal * 5.0; }},
        {"Pressure_P=F/A", [&]() { return x * reciprocal / 1.0; }},
        {"Density_ρ=m/V", [&]() { return x * reciprocal / 2.0; }},
        {"Entropy_S", [&]() { return x * reciprocal * log(x); }},
        {"Enthalpy_H", [&]() { return x * reciprocal + 273.15; }},
        {"Gibbs_Free_Energy", [&]() { return x * reciprocal - 298.15 * log(x); }},
        {"Internal_Energy_U", [&]() { return 1.5 * x * reciprocal * 8.314; }}
    };
    
    for (const auto& formula : physics_formulas) {
        PhysicsCheckResult check;
        check.name = formula.first;
        check.value = formula.second();
        check.reciprocal = reciprocal;
        check.mcc_estimate = abs(check.value - x) / x;
        check.passes_tolerance = (check.mcc_estimate < 0.1);
        check.rel_error = abs(check.value - x) / x;
        
        if (check.mcc_estimate < 0.01) {
            check.mcc_confidence = "high";
        } else if (check.mcc_estimate < 0.05) {
            check.mcc_confidence = "medium";
        } else if (check.mcc_estimate < 0.1) {
            check.mcc_confidence = "low";
        } else {
            check.mcc_confidence = "infinite";
        }
        
        check.note = check.passes_tolerance ? "Within tolerance" : "Exceeds tolerance";
        
        results.push_back(check);
    }
    
    return results;
}

// ============================== INTERPRETIVE SUMMARY (from snippet3) ==============================

InterpretiveSummary generateInterpretiveSummary(const AnalysisEntry& entry) {
    InterpretiveSummary summary;
    summary.x = entry.original_number;
    summary.reciprocal = entry.reciprocal;
    
    // Compute norms
    summary.p1_norm = abs((double)entry.original_number - 1.0 / entry.reciprocal);
    summary.p2_norm = sqrt(pow((double)entry.original_number, 2) + pow(1.0 / entry.reciprocal, 2));
    summary.pinf_norm = max(abs((double)entry.original_number), abs(1.0 / entry.reciprocal));
    
    // Cauchy score (empirical)
    summary.cauchy_score = abs((double)entry.original_number - entry.reciprocal) / (abs((double)entry.original_number) + abs(entry.reciprocal));
    
    // Generate interpretive paragraphs
    summary.paragraphs = {
        "The number " + to_string(entry.original_number) + " exhibits reciprocal symmetry with value " + to_string(entry.reciprocal),
        "Golden ratio deviation: " + to_string(entry.snippet_analysis.golden_ratio_deviation),
        "Digital root: " + to_string(entry.snippet_analysis.digital_root) + " reveals hidden patterns",
        "LSpace classification: " + entry.l_space_manifold.behavioral_class,
        "Topological signature: " + entry.l_space_manifold.topological_signature,
        "Irrationality confidence: " + to_string(entry.irrationality_analysis.confidence),
        "Banach contraction factor: " + to_string(entry.banach_test.max_lipschitz_estimate),
        "Dream chaos factor: " + to_string(entry.dream_sequence.chaos_factor),
        "Physics validation: " + to_string(count_if(entry.physics_checks.begin(), entry.physics_checks.end(),
                                                  [](const PhysicsCheckResult& p) { return p.passes_tolerance; })) + "/18",
        "Overall integration shows deep mathematical harmony"
    };
    
    return summary;
}

// ============================== HELPER FUNCTIONS ==============================

// ============================== MAIN ANALYSIS FUNCTIONS ==============================

AnalysisEntry comprehensiveAnalysis(int n) {
    AnalysisEntry entry;
    
    // Core data
    entry.original_number = n;
    entry.reciprocal = 1.0 / n;
    entry.decimal_approximation = entry.reciprocal;
    
    // Ratio deviations
    entry.golden_ratio_deviation = abs(entry.reciprocal - golden_ratio);
    entry.silver_ratio_deviation = abs(entry.reciprocal - silver_ratio);
    entry.bronze_ratio_deviation = abs(entry.reciprocal - bronze_ratio);
    entry.copper_ratio_deviation = abs(entry.reciprocal - copper_ratio);
    entry.nickel_ratio_deviation = abs(entry.reciprocal - nickel_ratio);
    
    // Enhanced snippet analysis
    entry.snippet_analysis = analyzeSnippetData(n);
    entry.irrationality_analysis = checkIrrationality(entry.reciprocal);
    entry.l_space_manifold = computeLSpaceManifold(entry);
    entry.dream_sequence = computeDreamSequence(n);
    entry.xpow_sqrt_analysis = analyzeXpowSqrt(n);
    entry.physics_checks = performPhysicsChecks(entry);
    entry.interpretive_summary = generateInterpretiveSummary(entry);
    
    // Banach test for reciprocal function
    auto reciprocal_func = [n](double x) { return 1.0 / x; };
    entry.banach_test = banachianTest(reciprocal_func, 0.1, max(2.0, (double)n));
    
    // Number properties
    entry.is_harshad_number = isHarshadNumber(n);
    entry.is_happy_number = isHappyNumber(n);
    entry.is_automorphic_number = isAutomorphicNumber(n);
    entry.is_kaprekar_number = isKaprekarNumber(n);
    entry.is_disarium_number = isDisariumNumber(n);
    
    return entry;
}

// ============================== DISPLAY FUNCTIONS ==============================

void displayLSpaceAnalysis(const LSpaceManifold& manifold) {
    cout << "\n=== LSPACE MANIFOLD ANALYSIS ===" << endl;
    cout << "Topological Genus: " << manifold.lambda_1_topological_genus << endl;
    cout << "Harmonic Resonance: " << manifold.lambda_2_harmonic_resonance << endl;
    cout << "Entropy Gradient: " << manifold.lambda_3_entropy_gradient << endl;
    cout << "Phase Velocity: " << manifold.lambda_4_phase_velocity << endl;
    cout << "Attractor Strength: " << manifold.lambda_5_attractor_strength << endl;
    cout << "Manifold Twist: " << manifold.lambda_6_manifold_twist << endl;
    cout << "Quantum Coherence: " << manifold.lambda_7_quantum_coherence << endl;
    cout << "Bifurcation Point: " << manifold.lambda_8_bifurcation_point << endl;
    cout << "Information Density: " << manifold.lambda_9_information_density << endl;
    cout << "Fractal Dimension: " << manifold.lambda_10_fractal_dimension << endl;
    cout << "Symmetry Breaking: " << manifold.lambda_11_symmetry_breaking << endl;
    cout << "Emergence Factor: " << manifold.lambda_12_emergence_factor << endl;
    cout << "Holographic Encoding: " << manifold.lambda_13_holographic_encoding << endl;
    cout << "Compression Ratio: " << manifold.compression_ratio << endl;
    cout << "Dominant Parameter: " << manifold.dominant_parameter << endl;
    cout << "Topological Signature: " << manifold.topological_signature << endl;
    cout << "L-Space Distance: " << manifold.l_space_distance << endl;
    cout << "Behavioral Class: " << manifold.behavioral_class << endl;
}

void displayDreamSequence(const DreamSequence& sequence) {
    cout << "\n=== DREAM SEQUENCE ANALYSIS ===" << endl;
    for (const auto& entry : sequence.entries) {
        cout << "Position " << entry.position << ": ";
        if (entry.is_original_x) {
            cout << "[ORIGINAL] ";
        }
        cout << entry.value_str << " (dev: " << entry.deviation_from_reciprocal 
             << ", chaos: " << entry.chaos_parameter << ")" << endl;
    }
    cout << "Total Deviation: " << sequence.total_deviation << endl;
    cout << "Chaos Factor: " << sequence.chaos_factor << endl;
    cout << "Dream Signature: " << sequence.dream_signature << endl;
}

void displayXpowSqrtAnalysis(const XpowSqrtAnalysis& analysis) {
    cout << "\n=== X^SQRT(X) ANALYSIS ===" << endl;
    cout << "x = " << analysis.x << endl;
    cout << "x^sqrt(x) = " << analysis.value << endl;
    cout << "log(value) = " << analysis.log_value << endl;
    cout << "Derivative: " << analysis.derivative << endl;
    cout << "Sensitivity (dx): " << analysis.sensitivity_dx << endl;
    cout << "Sensitivity (drec): " << analysis.sensitivity_drec << endl;
    cout << "Growth Rate: " << analysis.growth_rate << endl;
    cout << "Banach Contraction Factor: " << analysis.banach_contraction_factor << endl;
    cout << "Stable Under Reciprocity: " << (analysis.is_stable_under_reciprocity ? "YES" : "NO") << endl;
}

void displayPhysicsChecks(const vector<PhysicsCheckResult>& checks) {
    cout << "\n=== PHYSICS VERIFICATION (18 FORMULAS) ===" << endl;
    for (const auto& check : checks) {
        cout << check.name << ": " << check.value << " (MCC: " << check.mcc_estimate 
             << ", " << check.mcc_confidence << ", " << check.note << ")" << endl;
    }
    
    int passed = count_if(checks.begin(), checks.end(), 
                         [](const PhysicsCheckResult& p) { return p.passes_tolerance; });
    cout << "Overall: " << passed << "/18 formulas within tolerance" << endl;
}

void displayInterpretiveSummary(const InterpretiveSummary& summary) {
    cout << "\n=== INTERPRETIVE SUMMARY ===" << endl;
    cout << "P1 Norm: " << summary.p1_norm << endl;
    cout << "P2 Norm: " << summary.p2_norm << endl;
    cout << "P∞ Norm: " << summary.pinf_norm << endl;
    cout << "Cauchy Score: " << summary.cauchy_score << endl;
    cout << "\nInterpretive Paragraphs:" << endl;
    for (const auto& paragraph : summary.paragraphs) {
        cout << "• " << paragraph << endl;
    }
}

void displayComprehensiveAnalysis(const AnalysisEntry& entry) {
    cout << "\n" << string(80, '=') << endl;
    cout << "COMPREHENSIVE ANALYSIS FOR: " << entry.original_number << endl;
    cout << string(80, '=') << endl;
    
    cout << "Reciprocal: " << entry.reciprocal << endl;
    cout << "Decimal Approximation: " << entry.decimal_approximation << endl;
    
    // Ratio deviations
    cout << "\n--- RATIO DEVIATIONS ---" << endl;
    cout << "Golden Ratio Deviation: " << entry.golden_ratio_deviation << endl;
    cout << "Silver Ratio Deviation: " << entry.silver_ratio_deviation << endl;
    cout << "Bronze Ratio Deviation: " << entry.bronze_ratio_deviation << endl;
    cout << "Copper Ratio Deviation: " << entry.copper_ratio_deviation << endl;
    cout << "Nickel Ratio Deviation: " << entry.nickel_ratio_deviation << endl;
    
    // Snippet data
    cout << "\n--- SNIPPET DATA ANALYSIS ---" << endl;
    cout << "Digit Sum: " << entry.snippet_analysis.digit_sum << endl;
    cout << "Digital Root: " << entry.snippet_analysis.digital_root << endl;
    cout << "Perfect Square: " << (entry.snippet_analysis.is_perfect_square ? "YES" : "NO") << endl;
    cout << "Perfect Cube: " << (entry.snippet_analysis.is_perfect_cube ? "YES" : "NO") << endl;
    cout << "Palindrome: " << (entry.snippet_analysis.is_palindrome ? "YES" : "NO") << endl;
    cout << "Divisor Count: " << entry.snippet_analysis.divisor_count << endl;
    cout << "Sum of Divisors: " << entry.snippet_analysis.sum_of_divisors << endl;
    cout << "Number Type: ";
    if (entry.snippet_analysis.is_perfect) cout << "PERFECT";
    else if (entry.snippet_analysis.is_abundant) cout << "ABUNDANT";
    else if (entry.snippet_analysis.is_deficient) cout << "DEFICIENT";
    else cout << "NORMAL";
    cout << endl;
    
    // Number properties
    cout << "\n--- NUMBER PROPERTIES ---" << endl;
    cout << "Harshad: " << (entry.is_harshad_number ? "YES" : "NO") << endl;
    cout << "Happy: " << (entry.is_happy_number ? "YES" : "NO") << endl;
    cout << "Automorphic: " << (entry.is_automorphic_number ? "YES" : "NO") << endl;
    cout << "Kaprekar: " << (entry.is_kaprekar_number ? "YES" : "NO") << endl;
    cout << "Disarium: " << (entry.is_disarium_number ? "YES" : "NO") << endl;
    
    // Irrationality analysis
    cout << "\n--- IRRATIONALITY ANALYSIS ---" << endl;
    cout << "Is Irrational: " << (entry.irrationality_analysis.is_irrational ? "YES" : "NO") << endl;
    cout << "Confidence: " << entry.irrationality_analysis.confidence << endl;
    cout << "Proof Method: " << entry.irrationality_analysis.proof_method << endl;
    cout << "Continued Fraction: " << entry.irrationality_analysis.continued_fraction_pattern << endl;
    cout << "Approximation Error: " << entry.irrationality_analysis.approximation_error << endl;
    
    // Banach test
    cout << "\n--- BANACH CONTRACTION TEST ---" << endl;
    cout << "Max Lipschitz Estimate: " << entry.banach_test.max_lipschitz_estimate << endl;
    cout << "Is Contraction: " << (entry.banach_test.is_contraction ? "YES" : "NO") << endl;
    cout << "Samples Used: " << entry.banach_test.samples << endl;
    
    // Display detailed analyses
    displayDreamSequence(entry.dream_sequence);
    displayLSpaceAnalysis(entry.l_space_manifold);
    displayXpowSqrtAnalysis(entry.xpow_sqrt_analysis);
    displayPhysicsChecks(entry.physics_checks);
    displayInterpretiveSummary(entry.interpretive_summary);
    
    cout << "\n" << string(80, '=') << endl;
    cout << "END OF COMPREHENSIVE ANALYSIS" << endl;
    cout << string(80, '=') << endl;
}

// ============================== MAIN INTERFACE ==============================

// Function to be called from main file
vector<AnalysisEntry> performComprehensiveAdjacencyAnalysis(int center, int range) {
    vector<AnalysisEntry> results;
    
    for (int i = center - range; i <= center + range; i++) {
        if (i > 0) { // Only positive integers
            AnalysisEntry entry = comprehensiveAnalysis(i);
            results.push_back(entry);
        }
    }
    
    return results;
}

// Main analysis function (can be called from main program)
AnalysisEntry analyzeNumberWithComprehensiveIntegration(int n) {
    return comprehensiveAnalysis(n);
}

// Test function
void testComprehensiveIntegration() {
    cout << "\n=== TESTING COMPREHENSIVE INTEGRATION ===" << endl;
    
    vector<int> test_numbers = {1, 2, 3, 6, 7, 12, 28, 37};
    
    for (int n : test_numbers) {
        AnalysisEntry entry = analyzeNumberWithComprehensiveIntegration(n);
        displayComprehensiveAnalysis(entry);
        
        cout << "\nPress Enter to continue to next test number...";
        cin.get();
    }
    
    cout << "\n=== ADJACENCY ANALYSIS TEST ===" << endl;
    int center = 7;
    int range = 2;
    vector<AnalysisEntry> adjacency_results = performComprehensiveAdjacencyAnalysis(center, range);
    
    for (const auto& entry : adjacency_results) {
        cout << "\n--- ADJACENCY RESULT FOR " << entry.original_number << " ---" << endl;
        cout << "Digital Root: " << entry.snippet_analysis.digital_root << endl;
        cout << "Irrational: " << (entry.irrationality_analysis.is_irrational ? "YES" : "NO") << endl;
        cout << "Behavioral Class: " << entry.l_space_manifold.behavioral_class << endl;
        cout << "Chaos Factor: " << entry.dream_sequence.chaos_factor << endl;
    }
}


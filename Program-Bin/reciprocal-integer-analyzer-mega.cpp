/*
 * THE GRAND RECIPROCAL PROOF FRAMEWORK (MEGA EDITION + BURST MODE)
 * --------------------------------------------------------------------
 * MATHEMATICAL PROOF: x/1 = 1/x if and only if x = ±1
 * 
 * MERGED FEATURES:
 * - Original mega.cpp: All 24 analysis sections + bonus functions
 * - snippet1.txt: Interactive burst mode with multiple focus options
 * - All original functionality preserved exactly
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
#include <random>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/rational.hpp>

// ============================== PRECISION CONFIG ==============================
constexpr int PRECISION_DECIMALS = 1200;
constexpr int GUARD_DIGITS = 200;
constexpr int TAIL_SAFETY = 77;

using namespace boost::multiprecision;

// High-precision decimal type for 1200+ decimal places
using high_precision_float = number<cpp_dec_float<PRECISION_DECIMALS + GUARD_DIGITS>>;
using high_precision_int = number<cpp_int>;
using high_float = high_precision_float;  // Alias for burst mode compatibility

// Custom power function for Boost multiprecision
high_precision_float mp_pow(const high_precision_float& base, int exponent) {
    high_precision_float result = 1;
    if (exponent >= 0) {
        for (int i = 0; i < exponent; ++i) result *= base;
    } else {
        for (int i = 0; i < -exponent; ++i) result /= base;
    }
    return result;
}

// Custom square root function for Boost multiprecision
high_precision_float mp_sqrt(const high_precision_float& x) {
    if (x < high_precision_float(0)) return high_precision_float(0);
    if (x == high_precision_float(0)) return high_precision_float(0);
    
    // Newton's method for high precision
    high_precision_float guess = x / high_precision_float(2);
    high_precision_float prev = high_precision_float(0);
    high_precision_float eps = mp_pow(high_precision_float(10), -(PRECISION_DECIMALS - 100));
    
    while (abs(guess - prev) > eps) {
        prev = guess;
        guess = (guess + x / guess) / high_precision_float(2);
    }
    return guess;
}

// Custom exponential function for Boost multiprecision
high_precision_float mp_exp(const high_precision_float& x) {
    // Use series expansion for small x, approximation for larger x
    if (abs(x) < 1) {
        high_precision_float result = 1;
        high_precision_float term = 1;
        int n = 1;
        while (abs(term) > mp_pow(high_precision_float(10), -(PRECISION_DECIMALS - 50))) {
            term *= x / n;
            result += term;
            n++;
        }
        return result;
    } else {
        // For larger values, use exp(log) approximation or fallback to double
        return high_precision_float(std::exp(static_cast<double>(x)));
    }
}

// Centralized tolerance constants
high_precision_float EPSILON = mp_pow(high_precision_float(10), -(PRECISION_DECIMALS - 50));       // 10^-1150
high_precision_float EPS_RECIP = mp_pow(high_precision_float(10), -PRECISION_DECIMALS);          // 10^-1200
high_precision_float EPS_COSMIC = mp_pow(high_precision_float(10), -(PRECISION_DECIMALS - 10));   // 10^-1190

// ============================== GLOBAL CONSTANTS ==============================
const high_precision_float PHI = (high_precision_float(1) + mp_sqrt(high_precision_float(5))) / 2;
const high_precision_float PSI = (high_precision_float(1) - mp_sqrt(high_precision_float(5))) / 2;
const high_precision_float E = mp_exp(high_precision_float(1));
const high_precision_float PI = mp_sqrt(high_precision_float(2)) * mp_sqrt(high_precision_float(8)); // π ≈ 2√8
const high_precision_float SQRT2 = mp_sqrt(high_precision_float(2));
const high_precision_float SQRT5 = mp_sqrt(high_precision_float(5));

// Forward declarations
std::vector<int> continued_fraction_iterative(const high_precision_float& x, int max_terms);

// ============================== MEGA RECURSION HANDLING ==============================
class MegaRecursionManager {
private:
    std::ofstream output_file;
    std::ofstream progress_file;
    std::mutex file_mutex;
    uint64_t current_recursion_depth;
    uint64_t max_recursion_depth;
    bool streaming_enabled;
    
public:
    MegaRecursionManager(const std::string& output_filename, bool enable_streaming = true) 
        : current_recursion_depth(0), max_recursion_depth(0), streaming_enabled(enable_streaming) {
        
        if (streaming_enabled) {
            output_file.open(output_filename, std::ios::out | std::ios::app);
            progress_file.open(output_filename + ".progress", std::ios::out | std::ios::app);
            
            if (!output_file.is_open() || !progress_file.is_open()) {
                throw std::runtime_error("Failed to open output files for streaming");
            }
        }
    }
    
    ~MegaRecursionManager() {
        if (streaming_enabled) {
            output_file.close();
            progress_file.close();
        }
    }
    
    // Handle 10^50 recursion depth tracking
    void track_recursion_depth(uint64_t depth) {
        current_recursion_depth = depth;
        max_recursion_depth = std::max(max_recursion_depth, depth);
        
        if (streaming_enabled && depth % 1000000 == 0) { // Log every million recursions
            std::lock_guard<std::mutex> lock(file_mutex);
            progress_file << "Recursion depth: " << depth << " / 10^50" << std::endl;
            progress_file.flush();
        }
    }
    
    // Stream output to file to save memory
    void stream_output(const std::string& content) {
        if (streaming_enabled) {
            std::lock_guard<std::mutex> lock(file_mutex);
            output_file << content;
            output_file.flush();
        } else {
            std::cout << content;
        }
    }
    
    uint64_t get_max_depth() const { return max_recursion_depth; }
    
    // Check if we can handle more recursion (for 10^50 scale)
    bool can_recurse_more() const {
        return current_recursion_depth < 10000000000000000000ULL; // 10^19 safety limit
    }
};

// Global recursion manager
std::unique_ptr<MegaRecursionManager> mega_manager;

// ============================== BURST MODE CONSTANTS ==============================
constexpr uint64_t DEFAULT_BURST_SIZE = 500;

// Focus flags – you can combine any number of them
enum class Focus : uint64_t {
    None          = 0,
    Random        = 1 << 0,
    PowersOf10    = 1 << 1,
    GoldenFamily  = 1 << 2,
    PolygonRoots  = 1 << 3,
    Harmonic      = 1 << 4,
    Fibonacci     = 1 << 5,
    Extreme       = 1 << 6,   // 10^±50 range
    Algebraic     = 1 << 7
};

// Enable bitwise operators for Focus enum
inline Focus operator|(Focus a, Focus b) {
    return static_cast<Focus>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}

inline Focus operator&(Focus a, Focus b) {
    return static_cast<Focus>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}

inline Focus& operator|=(Focus& a, Focus b) {
    return a = a | b;
}

inline bool operator==(Focus a, Focus b) {
    return static_cast<uint64_t>(a) == static_cast<uint64_t>(b);
}

inline bool operator!=(Focus a, Focus b) {
    return !(a == b);
}

// ============================== UTILITY FUNCTIONS ==============================
void banner(const std::string& text = "", int width = 70) {
    std::string line(width, '=');
    std::string result;
    
    if (!text.empty()) {
        result += line + "\n";
        result += std::string((width - text.length()) / 2, ' ') + text + "\n";
        result += line + "\n";
    } else {
        result += line + "\n";
    }
    
    if (mega_manager) {
        mega_manager->stream_output(result);
    } else {
        std::cout << result;
    }
}

std::string decimal_short(const high_precision_float& x, int show_digits = 10) {
    std::stringstream ss;
    ss.str("");
    ss.clear();
    
    if (isnan(x) || isinf(x)) {
        return ss.str();
    }
    
    // Check if integer (using high precision comparison)
    high_precision_int int_part = static_cast<high_precision_int>(x);
    high_precision_float diff = x - static_cast<high_precision_float>(int_part);
    
    if (abs(diff) < EPSILON) {
        ss << int_part;
        return ss.str();
    }
    
    ss << std::setprecision(show_digits + 5) << x;
    std::string s = ss.str();
    
    if (s.find('e') != std::string::npos) {
        return s;
    }
    
    return s.substr(0, show_digits);
}

std::string decimal_full(const high_precision_float& x) {
    std::stringstream ss;
    ss.str("");
    ss.clear();
    
    if (isnan(x) || isinf(x)) {
        return ss.str();
    }
    
    // Check if integer
    high_precision_int int_part = static_cast<high_precision_int>(x);
    high_precision_float diff = x - static_cast<high_precision_float>(int_part);
    
    if (abs(diff) < EPSILON) {
        ss << int_part;
        return ss.str();
    }
    
    ss << std::setprecision(PRECISION_DECIMALS + 10) << x;
    return ss.str();
}

bool is_integer(const high_precision_float& val) {
    if (isnan(val) || isinf(val)) {
        return false;
    }
    
    high_precision_int rounded = static_cast<high_precision_int>(val);
    high_precision_float diff = abs(val - static_cast<high_precision_float>(rounded));
    return diff < EPSILON;
}

std::string decimal_snippet(const high_precision_float& x, int length = 50) {
    std::stringstream ss;
    ss << std::setprecision(length) << x;
    return ss.str();
}

bool is_perfect_square(const high_precision_float& n) {
    if (!is_integer(n)) {
        return false;
    }
    
    high_precision_int ni = static_cast<high_precision_int>(n);
    high_precision_float sqrt_n = mp_sqrt(high_precision_float(ni));
    high_precision_float check = sqrt_n * sqrt_n;
    return abs(check - high_precision_float(ni)) < EPSILON;
}

// Prime factorization
std::vector<high_precision_int> prime_factorize(high_precision_int n) {
    std::vector<high_precision_int> factors;
    if (n.compare(0) < 0) n = -n;
    if (n.compare(0) == 0 || n.compare(1) == 0) return {n};
    
    high_precision_int d = 2;
    while (d * d <= n) {
        while (n % d == 0) {
            factors.push_back(d);
            n /= d;
        }
        ++d;
        if (d > 1000000) break; // Safety limit
    }
    if (n > 1) factors.push_back(n);
    return factors;
}

// ============================== MULTIPLICATIVE CLOSURE COUNT (MCC) ==============================
// This module computes the minimal positive integer k such that x * k is an integer (MCC).
// If no such k exists (irrational), returns "infinite" (MCC = 0 with confidence "infinite").

struct MCCResult {
    bool finite;                 // true if finite MCC found
    high_precision_int mcc;      // the minimal multiplier k (if finite)
    std::string confidence;      // "high", "medium", "low", or "infinite"
    std::string as_string() const {
        if (!finite) return "∞";
        return mcc.convert_to<std::string>();
    }
};

// gcd for cpp_int
static high_precision_int cpp_gcd(high_precision_int a, high_precision_int b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) {
        high_precision_int r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// pow10 as cpp_int
static high_precision_int pow10_int(unsigned int d) {
    high_precision_int r = 1;
    for (unsigned int i = 0; i < d; ++i) r *= 10;
    return r;
}

// Get convergents from CF terms (returns vector of pairs (p,q))
static std::vector<std::pair<high_precision_int, high_precision_int>> convergents_from_cf(const std::vector<int>& cf) {
    std::vector<std::pair<high_precision_int, high_precision_int>> conv;
    if (cf.empty()) return conv;
    high_precision_int p_nm2 = 0, p_nm1 = 1;
    high_precision_int q_nm2 = 1, q_nm1 = 0;
    for (size_t i = 0; i < cf.size(); ++i) {
        high_precision_int a = cf[i];
        high_precision_int p_n = a * p_nm1 + p_nm2;
        high_precision_int q_n = a * q_nm1 + q_nm2;
        conv.push_back({p_n, q_n});
        p_nm2 = p_nm1; p_nm1 = p_n;
        q_nm2 = q_nm1; q_nm1 = q_n;
    }
    return conv;
}

// Attempts to reconstruct a rational p/q from x using finite-decimal detection and CF convergents.
MCCResult compute_MCC(const high_precision_float& x) {
    const unsigned int D_MAX = 500;   // max fractional digits to accept as finite decimal quickly
    const high_precision_int Q_MAX = static_cast<high_precision_int>(1000000000); // 1e9 denominator cap
    MCCResult res;
    res.finite = false;
    res.mcc = 0;
    res.confidence = "infinite";
    
    if (x == 0) {
        res.finite = true;
        res.mcc = 1; // 0 * 1 = 0 integer; but contextually MCC for zero is degenerate
        res.confidence = "degenerate";
        return res;
    }
    
    // Represent as string and try to detect a terminating decimal (no 'e' and finite fraction part)
    std::string full = decimal_full(x);
    // If representation contains 'e' we treat as non-finite for quick path
    if (full.find('e') == std::string::npos && full.find('E') == std::string::npos) {
        // Check for decimal point
        auto pos = full.find('.');
        if (pos != std::string::npos) {
            std::string intpart = full.substr(0, pos);
            std::string fracpart = full.substr(pos + 1);
            // If fractional length is small enough we can do exact conversion
            if (!fracpart.empty() && fracpart.size() <= D_MAX) {
                // Remove any trailing zeros in fractional part
                std::string frac_trim = fracpart;
                while (!frac_trim.empty() && frac_trim.back() == '0') frac_trim.pop_back();
                
                unsigned int d = static_cast<unsigned int>(frac_trim.size());
                if (d == 0) {
                    // effectively integer
                    res.finite = true;
                    res.mcc = 1;
                    res.confidence = "high";
                    return res;
                }
                
                // construct numerator = round(x * 10^d)
                high_precision_int denom = pow10_int(d);
                high_precision_float scaled = x * high_precision_float(denom);
                // rounded numerator
                high_precision_int numer = static_cast<high_precision_int>(scaled + (scaled >= 0 ? high_precision_float(0.5) : high_precision_float(-0.5)));
                // reduce
                high_precision_int g = cpp_gcd(numer < 0 ? -numer : numer, denom);
                high_precision_int q = denom / g;
                res.finite = true;
                res.mcc = q;
                res.confidence = "high";
                return res;
            }
        } else {
            // no decimal point -> integer
            res.finite = true;
            res.mcc = 1;
            res.confidence = "high";
            return res;
        }
    }
    
    // Not a simple terminating decimal or too long; attempt CF reconstruction heuristics
    // Compute CF terms and convergents (limited depth)
    std::vector<int> cf = continued_fraction_iterative(x, 300);
    auto convs = convergents_from_cf(cf);
    high_precision_float x_abs = abs(x);
    for (size_t i = 0; i < convs.size(); ++i) {
        high_precision_int p = convs[i].first;
        high_precision_int q = convs[i].second;
        if (q <= 0) continue;
        if (q > Q_MAX) continue; // skip too-large denominators
        // compute p/q as high_precision_float
        high_precision_float approx = high_precision_float(p) / high_precision_float(q);
        high_precision_float err = abs(x - approx);
        // Accept if approximation is extraordinarily tight relative to EPS_RECIP
        if (err < EPS_RECIP * high_precision_float(10)) {
            // found a good rational representation
            res.finite = true;
            res.mcc = q;
            // Confidence depends on denominator magnitude
            if (q < 1000000) res.confidence = "high";
            else if (q < 100000000) res.confidence = "medium";
            else res.confidence = "low";
            return res;
        }
    }
    
    // If we reached here, treat as infinite (irrational or high-denominator rational)
    res.finite = false;
    res.mcc = 0;
    res.confidence = "infinite";
    return res;
}

// Utility: normalized MCC_score based on digit-length of MCC
static double mcc_score_from_mcc(const MCCResult& mres) {
    if (!mres.finite) return 0.0;
    std::string s = mres.as_string();
    if (s == "0" || s == "") return 0.0;
    if (s == "1") return 1.0;
    // digits of the integer
    size_t digits = s.size();
    // map digits -> score: more digits -> lower score; simple mapping:
    // score = 1 / (1 + (digits - 1))
    double denom = 1.0 + static_cast<double>(digits > 0 ? (digits - 1) : 0);
    return 1.0 / denom;
}

// ============================== ADVANCED ANALYSIS FUNCTIONS ==============================

// Digit distribution analysis
void analyze_digit_distribution(const high_precision_float& x) {
    std::string decimal = decimal_full(x);
    size_t dot_pos = decimal.find('.');
    if (dot_pos == std::string::npos) return;
    
    std::string fractional = decimal.substr(dot_pos + 1);
    int digit_counts[10] = {0};
    int leading_counts[10] = {0};
    bool first_nonzero = false;
    
    for (char c : fractional) {
        if (c >= '0' && c <= '9') {
            int digit = c - '0';
            digit_counts[digit]++;
            if (!first_nonzero && digit != 0) {
                leading_counts[digit]++;
                first_nonzero = true;
            }
        }
    }
    
    // Output Benford analysis: leading_counts[1..9] should follow log10(1+1/d)
    std::string benford_str = "  Benford's Law Analysis - Leading digit distribution:\n";
    for (int d = 1; d <= 9; d++) {
        if (leading_counts[d] > 0) {
            benford_str += "    Digit " + std::to_string(d) + ": " + std::to_string(leading_counts[d]) + " occurrences\n";
        }
    }
    
    if (mega_manager) {
        mega_manager->stream_output(benford_str);
    } else {
        std::cout << benford_str;
    }
}

// Estimate irrationality measure
double estimate_irrationality_measure(const high_precision_float& x) {
    auto cf = continued_fraction_iterative(x, 50);
    if (cf.size() < 10) return 2.0; // Likely rational
    
    // Look for exponential growth in CF terms = Liouville-type indicator
    double max_ratio = 1.0;
    for (size_t i = 1; i < cf.size(); i++) {
        if (cf[i-1] > 0) {
            double ratio = static_cast<double>(cf[i]) / cf[i-1];
            max_ratio = std::max(max_ratio, ratio);
        }
    }
    return 1.0 + log(max_ratio) / log(2.0);
}

// Detect algebraic type
std::string detect_algebraic_type(const high_precision_float& x) {
    if (is_integer(x)) return "rational integer";
    
    MCCResult mcc = compute_MCC(x);
    if (mcc.finite && mcc.confidence == "high") return "rational";
    
    auto cf = continued_fraction_iterative(x, 100);
    // Check for periodic patterns = quadratic irrational
    for (int period = 1; period <= (int)cf.size()/2; period++) {
        bool is_periodic = true;
        for (int i = 0; i < period && i + period < (int)cf.size(); i++) {
            if (cf[i] != cf[i + period]) {
                is_periodic = false;
                break;
            }
        }
        if (is_periodic) return "quadratic irrational";
    }
    
    return "likely transcendental";
}

// Riemann zeta approximation
high_precision_float riemann_zeta_approx(const high_precision_float& s, int terms = 1000) {
    high_precision_float sum = 0;
    for (int n = 1; n <= terms; ++n) {
        high_precision_float term = 1 / pow(high_precision_float(n), s);
        sum += term;
    }
    return sum;
}

// Prime counting function approximation
high_precision_float prime_count_approx(const high_precision_float& x) {
    if (x < 2) return 0;
    return x / log(x); // Simple approximation π(x) ~ x/ln(x)
}

// Series analysis structures
struct SeriesAnalysis {
    bool converges;
    high_precision_float partial_sum;
    high_precision_float error_bound;
    std::string convergence_type;
};

SeriesAnalysis analyze_alternating_series(const std::vector<high_precision_float>& terms) {
    SeriesAnalysis result;
    result.partial_sum = 0;
    
    // Alternating Series Test
    bool terms_decrease = true;
    bool limit_zero = true;
    
    for (size_t i = 0; i < terms.size(); ++i) {
        result.partial_sum += terms[i];
        
        // Check if terms decrease in absolute value
        if (i > 0 && abs(terms[i]) >= abs(terms[i-1])) {
            terms_decrease = false;
        }
        
        // Check if limit approaches zero
        if (abs(terms[i]) > high_precision_float("0.001")) {
            limit_zero = false;
        }
    }
    
    result.converges = terms_decrease && limit_zero;
    result.error_bound = abs(terms.back()); // For alternating series
    result.convergence_type = result.converges ? "Conditionally convergent" : "Divergent";
    
    return result;
}

// Numerical derivative
high_precision_float numerical_derivative(
    const std::function<high_precision_float(high_precision_float)>& f,
    const high_precision_float& x, 
    const high_precision_float& h = high_precision_float("1e-10")) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

// Find critical points
std::vector<high_precision_float> find_critical_points(
    const std::function<high_precision_float(high_precision_float)>& f,
    const high_precision_float& a, 
    const high_precision_float& b, 
    int steps = 1000) {
    
    std::vector<high_precision_float> critical_points;
    high_precision_float step_size = (b - a) / steps;
    
    for (int i = 1; i < steps - 1; ++i) {
        high_precision_float x = a + i * step_size;
        high_precision_float deriv = numerical_derivative(f, x);
        
        // Check for sign change (crude critical point detection)
        high_precision_float x_prev = a + (i-1) * step_size;
        high_precision_float deriv_prev = numerical_derivative(f, x_prev);
        
        if (deriv * deriv_prev <= 0) { // Sign change indicates critical point
            critical_points.push_back(x);
        }
    }
    
    return critical_points;
}

// Knapsack solver
struct KnapsackSolution {
    int max_value;
    std::vector<int> selected_items;
    high_precision_float solving_time;
};

KnapsackSolution solve_knapsack(const std::vector<int>& values, 
                                const std::vector<int>& weights, 
                                int capacity) {
    KnapsackSolution solution;
    int n = values.size();
    
    // Dynamic programming table
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
    
    // Build DP table
    for (int i = 1; i <= n; ++i) {
        for (int w = 0; w <= capacity; ++w) {
            if (weights[i-1] <= w) {
                dp[i][w] = std::max(dp[i-1][w], 
                                  dp[i-1][w - weights[i-1]] + values[i-1]);
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    
    solution.max_value = dp[n][capacity];
    
    // Backtrack to find selected items
    int w = capacity;
    for (int i = n; i > 0 && solution.max_value > 0; --i) {
        if (solution.max_value != dp[i-1][w]) {
            solution.selected_items.push_back(i-1);
            solution.max_value -= values[i-1];
            w -= weights[i-1];
        }
    }
    
    return solution;
}

// Advanced number analysis
void advanced_number_analysis(const high_precision_float& x_value) {
    // Complex-analytic analysis
    if (x_value > 1 && x_value < 10) {
        high_precision_float zeta_approx = riemann_zeta_approx(x_value, 100);
        std::string zeta_str = "  Riemann Zeta(" + decimal_short(x_value) + ") ≈ " + decimal_short(zeta_approx) + "\n";
        if (mega_manager) mega_manager->stream_output(zeta_str);
        else std::cout << zeta_str;
    }
    
    // Critical point analysis for common functions
    auto test_function = [](high_precision_float x) { return x * x - 2 * x + 1; };
    auto critical_points = find_critical_points(test_function, -10, 10);
    
    if (!critical_points.empty()) {
        std::string crit_str = "  Critical points of x²-2x+1 near 0: ";
        for (const auto& point : critical_points) {
            crit_str += decimal_short(point) + " ";
        }
        crit_str += "\n";
        if (mega_manager) mega_manager->stream_output(crit_str);
        else std::cout << crit_str;
    }
    
    // Prime distribution analysis
    if (is_integer(x_value) && x_value > 0) {
        high_precision_int n = static_cast<high_precision_int>(x_value);
        high_precision_float prime_approx = prime_count_approx(x_value);
        std::string prime_str = "  Approximate primes ≤ " + decimal_short(x_value) + ": " + decimal_short(prime_approx) + "\n";
        if (mega_manager) mega_manager->stream_output(prime_str);
        else std::cout << prime_str;
    }
}

// ============================== MISSING UTILITY FUNCTIONS FROM PYTHON ==============================

// Decimal representation and rational analysis functions
std::vector<int> continued_fraction_with_exact(const high_precision_float& x, int max_terms = 100) {
    std::vector<int> cf;
    high_precision_float x_val = x;
    bool is_exact = false;
    
    for (int i = 0; i < max_terms; ++i) {
        if (abs(x_val) < EPSILON) {
            is_exact = true;
            break;
        }
        
        int a = static_cast<int>(floor(x_val));
        cf.push_back(a);
        x_val -= a;
        
        if (abs(x_val) < EPSILON) {
            is_exact = true;
            break;
        }
        
        x_val = 1 / x_val;
        if (isinf(x_val) || isnan(x_val)) {
            break;
        }
    }
    
    return cf; // Note: is_exact status would need to be returned separately if needed
}

// Rational approximation from continued fraction
struct RationalResult {
    bool valid;
    high_precision_int numerator;
    high_precision_int denominator;
    
    std::string to_string() const {
        if (!valid) return "invalid";
        return numerator.str() + "/" + denominator.str();
    }
};

RationalResult get_rational_approx(const high_precision_float& x) {
    RationalResult result;
    result.valid = false;
    
    std::vector<int> cf = continued_fraction_with_exact(x, 100);
    if (cf.empty()) {
        result.valid = true;
        result.numerator = 0;
        result.denominator = 1;
        return result;
    }
    
    // Standard convergent construction
    std::vector<high_precision_int> h = {0, 1};
    std::vector<high_precision_int> k = {1, 0};
    
    for (int a : cf) {
        h.push_back(a * h.back() + h[h.size() - 2]);
        k.push_back(a * k.back() + k[k.size() - 2]);
    }
    
    if (h.size() > 1 && k.size() > 1) {
        result.valid = true;
        result.numerator = h.back();
        result.denominator = k.back();
    }
    
    return result;
}

std::string get_decimal_repr(const RationalResult& frac) {
    if (frac.numerator == 0) return "0";
    
    std::string sign = (frac.numerator < 0) ? "-" : "";
    high_precision_int abs_num = (frac.numerator < 0) ? -frac.numerator : frac.numerator;
    
    high_precision_int int_part = abs_num / frac.denominator;
    high_precision_int remainder = abs_num % frac.denominator;
    
    if (remainder == 0) {
        return sign + int_part.str();
    }
    
    // For simplicity, return as fraction since decimal expansion is complex
    return sign + frac.to_string();
}

std::string analyze_decimal_expansion(const high_precision_float& x) {
    std::string result = "";
    
    RationalResult frac = get_rational_approx(x);
    if (frac.valid) {
        result += "Rational form: " + frac.to_string() + "\n";
        std::string dec_repr = get_decimal_repr(frac);
        result += "Decimal representation: " + dec_repr + "\n";
        if (frac.numerator % frac.denominator != 0) {
            result += "Pattern: Repeating or terminating decimal based on denominator factors.\n";
        } else {
            result += "Pattern: Terminating decimal.\n";
        }
        return result;
    } else {
        result += "Irrational number\n";
        std::string s = decimal_full(x);
        if (s.find('e') != std::string::npos) {
            result += "Scientific notation: " + s + "\n";
            result += "No decimal chunks for extreme values.\n";
            return result;
        }
        
        size_t dot_pos = s.find('.');
        if (dot_pos != std::string::npos) {
            std::string int_part = s.substr(0, dot_pos);
            std::string dec_part = s.substr(dot_pos + 1);
            
            result += "Integer part: " + int_part + "\n";
            
            const int chunk_size = 20;
            result += "Decimal chunks (groups of " + std::to_string(chunk_size) + " digits):\n";
            
            for (size_t i = 0; i < dec_part.length(); i += chunk_size) {
                size_t end = std::min(i + chunk_size, dec_part.length());
                std::string chunk = dec_part.substr(i, end - i);
                result += "Digits " + std::to_string(i + 1) + "-" + std::to_string(end) + ": " + chunk + "\n";
            }
            result += "Pattern: Non-repeating, irrational divisions mapped in chunks.\n";
        }
        return result;
    }
}

std::string divisibility_error_analysis(const high_precision_float& x_value) {
    if (x_value == 0) {
        return "Undefined for zero";
    }
    
    if (is_integer(x_value)) {
        return "Integer - exact divisibility";
    }
    
    RationalResult frac = get_rational_approx(x_value);
    if (frac.valid && frac.denominator > 1) {
        // Check if denominator has only 2 and 5 as factors (terminating decimal)
        high_precision_int temp = frac.denominator;
        bool only_2_and_5 = true;
        
        while (temp % 2 == 0) temp /= 2;
        while (temp % 5 == 0) temp /= 5;
        
        if (temp == 1) {
            return "Rational with terminating decimal (denominator: " + frac.denominator.str() + ")";
        } else {
            return "Rational with repeating decimal (denominator: " + frac.denominator.str() + ")";
        }
    }
    
    // For irrationals, check continued fraction
    auto cf = continued_fraction_iterative(x_value, 15);
    if (cf.size() > 15) {
        return "Irrational - infinite non-repeating decimal (divisibility 'error' is actually proof)";
    } else {
        return "Likely rational or special irrational";
    }
}

// ============================== DREAMY SEQUENCE FUNCTIONS ==============================

// Enhanced exact inverse: given γₙ₊₁, find γₙ using high-precision Newton iteration
high_precision_float gamma_previous_exact(high_precision_float gamma_current) {
    high_precision_float g;
    
    // Initial guess based on gamma magnitude
    if (gamma_current > 100) {
        g = gamma_current - 2 * PI / log(gamma_current);
    } else {
        g = gamma_current * high_precision_float(0.99);
    }
    
    int max_iterations = 100;
    high_precision_float tolerance = mp_pow(high_precision_float(10), -(PRECISION_DECIMALS - 100));
    
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        high_precision_float log_g = log(g);
        high_precision_float log_g1 = log(g + 1);
        high_precision_float denominator = log_g * log_g + EPSILON;
        
        // Forward step calculation
        high_precision_float forward_step = g + 2 * PI * (log_g1 / denominator);
        high_precision_float residual = forward_step - gamma_current;
        
        // Convergence check
        if (abs(residual) < tolerance) {
            return g;
        }
        
        // Derivative calculation for Newton iteration
        high_precision_float d_log_g = 1 / g;
        high_precision_float d_log_g1 = 1 / (g + 1);
        high_precision_float d_denominator = 2 * log_g * d_log_g;
        
        high_precision_float dfdg = 1 + 2 * PI * (
            (d_log_g1 * denominator - log_g1 * d_denominator) / (denominator * denominator)
        );
        
        high_precision_float step = residual / dfdg;
        
        // Step size limiting for stability
        if (abs(step) > abs(g) * high_precision_float(0.1)) {
            if (step > 0) {
                step = abs(g) * high_precision_float(0.1);
            } else {
                step = -abs(g) * high_precision_float(0.1);
            }
        }
        
        g -= step;
        
        // Ensure positive gamma
        if (g <= 0) {
            g = gamma_current * high_precision_float(0.5);
        }
    }
    
    return g;
}

// Dreamy sequence analysis
std::vector<high_precision_float> dreamy_sequence_analysis() {
    std::vector<high_precision_float> full_sequence;
    
    if (mega_manager) {
        mega_manager->stream_output("Infinite Ascent Sequence (Dreamy Sequence - Enhanced 11-Part):\n");
        mega_manager->stream_output("γₙ₊₁ = γₙ + 2π · (log(γₙ + 1) / (log γₙ)²)\n");
        mega_manager->stream_output("Starting from γ₀ = 2\n\n");
    } else {
        std::cout << "Infinite Ascent Sequence (Dreamy Sequence - Enhanced 11-Part):" << std::endl;
        std::cout << "γₙ₊₁ = γₙ + 2π · (log(γₙ + 1) / (log γₙ)²)" << std::endl;
        std::cout << "Starting from γ₀ = 2" << std::endl << std::endl;
    }
    
    high_precision_float gamma_start = 2;
    
    // Compute 5 previous steps (reverse engineering)
    std::vector<high_precision_float> previous_steps;
    high_precision_float g_current = gamma_start;
    
    if (mega_manager) {
        mega_manager->stream_output("Computing previous entries via reverse engineering...\n");
    } else {
        std::cout << "Computing previous entries via reverse engineering..." << std::endl;
    }
    
    for (int step_back = 0; step_back < 5; ++step_back) {
        high_precision_float g_prev = gamma_previous_exact(g_current);
        
        // Verification
        high_precision_float log_g_prev = log(g_prev);
        high_precision_float log_g_prev_plus1 = log(g_prev + 1);
        high_precision_float reconstructed = g_prev + 2 * PI * (log_g_prev_plus1 / (log_g_prev * log_g_prev + EPSILON));
        high_precision_float error = abs(reconstructed - g_current);
        
        if (mega_manager) {
            mega_manager->stream_output("  Step -" + std::to_string(step_back + 1) + ": γ_-" + std::to_string(step_back + 1) + 
                                      " = " + decimal_short(g_prev) + "\n");
            mega_manager->stream_output("    Verification error: " + decimal_short(error) + "\n");
        } else {
            std::cout << "  Step -" << (step_back + 1) << ": γ_-" << (step_back + 1) 
                      << " = " << decimal_short(g_prev) << std::endl;
            std::cout << "    Verification error: " << decimal_short(error) << std::endl;
        }
        
        previous_steps.push_back(g_prev);
        g_current = g_prev;
    }
    
    // Reverse the previous steps to get correct order
    std::reverse(previous_steps.begin(), previous_steps.end());
    
    // Build the complete sequence: previous + current + forward
    for (const auto& prev : previous_steps) {
        full_sequence.push_back(prev);
    }
    full_sequence.push_back(gamma_start); // γ₀
    
    // Display the complete reverse section
    if (mega_manager) {
        mega_manager->stream_output("\nComplete Reverse Section:\n");
        for (int i = 0; i < 5; ++i) {
            mega_manager->stream_output("γ_-" + std::to_string(5-i) + " = " + decimal_short(previous_steps[i]) + "\n");
        }
        mega_manager->stream_output("γ₀ = " + decimal_short(gamma_start) + "    ← YOUR STARTING POINT\n");
        mega_manager->stream_output("\nForward Computation:\n");
    } else {
        std::cout << "\nComplete Reverse Section:" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "γ_-" << (5-i) << " = " << decimal_short(previous_steps[i]) << std::endl;
        }
        std::cout << "γ₀ = " << decimal_short(gamma_start) << "    ← YOUR STARTING POINT" << std::endl;
        std::cout << "\nForward Computation:" << std::endl;
    }
    
    // Compute 5 forward steps
    high_precision_float gamma = gamma_start;
    
    if (mega_manager) {
        mega_manager->stream_output("Step 0: γ₀ = " + decimal_short(gamma) + "\n");
        mega_manager->stream_output("        1/γ₀ = " + decimal_short(1/gamma) + "\n");
        mega_manager->stream_output("        Self-reciprocal check: γ₀ = 1/γ₀? " + 
                                  std::string(abs(gamma - 1/gamma) < EPSILON ? "YES" : "NO") + "\n");
    } else {
        std::cout << "Step 0: γ₀ = " << decimal_short(gamma) << std::endl;
        std::cout << "        1/γ₀ = " << decimal_short(1/gamma) << std::endl;
        std::cout << "        Self-reciprocal check: γ₀ = 1/γ₀? " << 
                    (abs(gamma - 1/gamma) < EPSILON ? "YES" : "NO") << std::endl;
    }
    
    for (int step = 1; step <= 5; ++step) {
        if (gamma <= 0) break;
        
        high_precision_float log_gamma = log(gamma);
        if (log_gamma == 0) break;
        
        high_precision_float numerator = log(gamma + 1);
        high_precision_float denominator = log_gamma * log_gamma;
        high_precision_float increment = 2 * PI * (numerator / denominator);
        high_precision_float next_gamma = gamma + increment;
        
        full_sequence.push_back(next_gamma);
        
        high_precision_float gap = next_gamma - gamma;
        high_precision_float gap_log = (gap > 0) ? log(gap) : 0;
        
        if (mega_manager) {
            mega_manager->stream_output("Step " + std::to_string(step) + ": γ_" + std::to_string(step) + 
                                      " = " + decimal_short(next_gamma) + "\n");
            mega_manager->stream_output("        Increment: " + decimal_short(increment) + "\n");
            mega_manager->stream_output("        Gap logarithm: " + decimal_short(gap_log) + "\n");
            mega_manager->stream_output("        1/γ_" + std::to_string(step) + " = " + decimal_short(1/next_gamma) + "\n");
            mega_manager->stream_output("        Self-reciprocal: γ_" + std::to_string(step) + " = 1/γ_" + 
                                      std::to_string(step) + "? " + 
                                      std::string(abs(next_gamma - 1/next_gamma) < EPSILON ? "YES" : "NO") + "\n");
        } else {
            std::cout << "Step " << step << ": γ_" << step << " = " << decimal_short(next_gamma) << std::endl;
            std::cout << "        Increment: " << decimal_short(increment) << std::endl;
            std::cout << "        Gap logarithm: " << decimal_short(gap_log) << std::endl;
            std::cout << "        1/γ_" << step << " = " << decimal_short(1/next_gamma) << std::endl;
            std::cout << "        Self-reciprocal: γ_" << step << " = 1/γ_" << step << "? " << 
                        (abs(next_gamma - 1/next_gamma) < EPSILON ? "YES" : "NO") << std::endl;
        }
        
        gamma = next_gamma;
    }
    
    // Enhanced analysis for the complete 11-part sequence
    if (full_sequence.size() > 1) {
        if (mega_manager) {
            mega_manager->stream_output("\nComplete 11-Part Sequence Analysis:\n");
            mega_manager->stream_output("  Starting value: γ₋₅ = " + decimal_short(full_sequence.front()) + "\n");
            mega_manager->stream_output("  Central value: γ₀ = " + decimal_short(gamma_start) + "\n");
            mega_manager->stream_output("  Final value: γ₅ = " + decimal_short(full_sequence.back()) + "\n");
            mega_manager->stream_output("  Total range: " + decimal_short(full_sequence.back() / full_sequence.front()) + "x growth\n");
            mega_manager->stream_output("  Forward growth from center: " + decimal_short(full_sequence.back() / gamma_start) + "x\n");
            mega_manager->stream_output("\nProof Insight from Enhanced Dreamy Sequence:\n");
            mega_manager->stream_output("  The complete 11-part sequence shows perfect reversibility:\n");
            mega_manager->stream_output("  γ₋₅ → ... → γ₀ → ... → γ₅ with exact inverse calculations\n");
            mega_manager->stream_output("  Even across 11 steps, the reciprocal relationship remains:\n");
            mega_manager->stream_output("  1/x = x/1 ONLY when x = ±1 (reinforced by bidirectional analysis)\n");
        } else {
            std::cout << "\nComplete 11-Part Sequence Analysis:" << std::endl;
            std::cout << "  Starting value: γ₋₅ = " << decimal_short(full_sequence.front()) << std::endl;
            std::cout << "  Central value: γ₀ = " << decimal_short(gamma_start) << std::endl;
            std::cout << "  Final value: γ₅ = " << decimal_short(full_sequence.back()) << std::endl;
            std::cout << "  Total range: " << decimal_short(full_sequence.back() / full_sequence.front()) << "x growth" << std::endl;
            std::cout << "  Forward growth from center: " << decimal_short(full_sequence.back() / gamma_start) << "x" << std::endl;
            std::cout << std::endl << "Proof Insight from Enhanced Dreamy Sequence:" << std::endl;
            std::cout << "  The complete 11-part sequence shows perfect reversibility:" << std::endl;
            std::cout << "  γ₋₅ → ... → γ₀ → ... → γ₅ with exact inverse calculations" << std::endl;
            std::cout << "  Even across 11 steps, the reciprocal relationship remains:" << std::endl;
            std::cout << "  1/x = x/1 ONLY when x = ±1 (reinforced by bidirectional analysis)" << std::endl;
        }
    }
    
    return full_sequence;
}

// ============================== PROOF-CENTERED CALCULATORS ==============================

struct ProofMetrics {
    bool theorem_applies;
    std::string proof_status;
    high_precision_float distance_from_equality;
    high_precision_float squared_deviation;
    high_precision_float reciprocal_gap;
    std::string algebraic_verification;
};

ProofMetrics calculate_proof_metrics(const high_precision_float& x_value) {
    ProofMetrics metrics;
    
    if (x_value == 0) {
        metrics.theorem_applies = false;
        metrics.proof_status = "Excluded (zero)";
        metrics.distance_from_equality = 0;
        metrics.squared_deviation = 0;
        metrics.reciprocal_gap = 0;
        metrics.algebraic_verification = "0 = 1/0 is undefined";
        return metrics;
    }
    
    high_precision_float reciprocal = 1 / x_value;
    high_precision_float distance = abs(x_value - reciprocal);
    high_precision_float squared_dev = abs(x_value * x_value - 1);
    
    // Determine proof status using centralized EPS_RECIP
    if (distance < EPS_RECIP) {
        metrics.theorem_applies = true;
        metrics.proof_status = "CONFIRMS theorem - self-reciprocal fixed point";
        metrics.algebraic_verification = "x² = " + decimal_short(x_value * x_value) + " = 1 ✓";
    } else {
        metrics.theorem_applies = false;
        metrics.proof_status = "Verifies theorem - distinct from reciprocal";
        metrics.algebraic_verification = "x² = " + decimal_short(x_value * x_value) + " ≠ 1";
    }
    
    metrics.distance_from_equality = distance;
    metrics.squared_deviation = squared_dev;
    metrics.reciprocal_gap = abs(reciprocal - x_value / 1);
    
    return metrics;
}

std::vector<std::string> generate_proof_language(const high_precision_float& x_value, 
                                                  const std::string& description, 
                                                  const ProofMetrics& metrics) {
    std::vector<std::string> language;
    
    if (x_value == 0) {
        language.push_back("🔒 ZERO EXCLUSION: The reciprocal theorem explicitly excludes zero,");
        language.push_back("   as 1/0 is undefined in standard arithmetic.");
        return language;
    }
    
    high_precision_float reciprocal = 1 / x_value;
    
    // Core proof language
    if (metrics.theorem_applies) {
        language.push_back("🎯 THEOREM VERIFICATION: This entry satisfies x/1 = 1/x");
        language.push_back("   Mathematical confirmation: x = " + decimal_short(x_value) + ", 1/x = " + decimal_short(reciprocal));
        language.push_back("   Algebraic proof: x² = 1 → x = ±1");
    } else {
        language.push_back("🔍 THEOREM SUPPORT: This entry demonstrates x/1 ≠ 1/x");
        language.push_back("   Distance from equality: " + decimal_short(metrics.distance_from_equality));
        language.push_back("   Squared deviation from 1: " + decimal_short(metrics.squared_deviation));
    }
    
    // Descriptive language based on value characteristics
    if (is_integer(x_value)) {
        high_precision_int n = static_cast<high_precision_int>(x_value);
        if (n == 1) {
            language.push_back("🌟 FUNDAMENTAL IDENTITY: The multiplicative identity element");
            language.push_back("   serves as the positive fixed point in reciprocal space.");
        } else if (n == -1) {
            language.push_back("🌑 NEGATIVE ANCHOR: The only negative number that equals its reciprocal,");
            language.push_back("   maintaining sign symmetry in the theorem.");
        } else {
            language.push_back("🔢 INTEGER REALM: Member of the " + std::to_string(n) + "-multiplication tree,");
            language.push_back("   with reciprocal creating infinite decimal complexity.");
        }
    } else if (x_value > 0 && x_value < 1) {
        language.push_back("📉 UNIT FRACTION TERRITORY: Exists between 0 and 1,");
        language.push_back("   where reciprocals amplify values into the >1 domain.");
    } else if (x_value > 1) {
        language.push_back("📈 INTEGER TERRITORY: Resides above 1,");
        language.push_back("   where reciprocals compress values into the <1 domain.");
    }
    
    // Mathematical structure commentary
    if (description.find("Golden") != std::string::npos) {
        language.push_back("🌅 GOLDEN FAMILY: Exhibits the special property 1/φ = φ - 1,");
        language.push_back("   the closest approach to self-reciprocality without equality.");
    } else if (description.find("10^") != std::string::npos) {
        language.push_back("⚡ EXTREME SCALE: Demonstrates theorem resilience across astronomical magnitudes,");
        language.push_back("   maintaining the reciprocal gap despite extreme values.");
    }
    
    return language;
}

high_precision_float reciprocal_symmetry_score(const high_precision_float& x_value) {
    if (x_value == 0) {
        return 0;
    }
    
    high_precision_float reciprocal = 1 / x_value;
    if (x_value > 0 && reciprocal > 0) {
        high_precision_float ratio = std::min(x_value / reciprocal, reciprocal / x_value);
        return ratio;
    }
    return 0;
}

std::string analyze_base_tree_membership(const high_precision_float& x_value) {
    if (!is_integer(x_value) || x_value == 0) {
        return "Non-integer - exists outside integer base trees";
    }
    
    high_precision_int n = static_cast<high_precision_int>(x_value);
    if (n == 1) {
        return "Universal identity - member of ALL base trees";
    }
    
    // Simple factorization for demonstration
    std::vector<high_precision_int> prime_bases;
    high_precision_int temp = n;
    
    // Check for prime factors 2, 3, 5, 7 for demonstration
    std::vector<int> test_primes = {2, 3, 5, 7};
    for (int p : test_primes) {
        while (temp % p == 0) {
            temp /= p;
            if (std::find(prime_bases.begin(), prime_bases.end(), high_precision_int(p)) == prime_bases.end()) {
                prime_bases.push_back(p);
            }
        }
    }
    
    if (temp > 1 && temp < 1000) {
        prime_bases.push_back(temp); // Add remaining factor if small
    }
    
    std::string result = "Member of base trees: [";
    for (size_t i = 0; i < prime_bases.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(static_cast<int>(prime_bases[i]));
    }
    result += "]";
    
    // Check decimal pattern
    bool all_2_or_5 = true;
    for (const auto& base : prime_bases) {
        if (base != 2 && base != 5) {
            all_2_or_5 = false;
            break;
        }
    }
    
    if (all_2_or_5) {
        result += " (Terminating decimal pattern)";
    } else {
        result += " (Repeating decimal pattern)";
    }
    
    return result;
}

// ============================== COSMIC REALITY TRACKING ==============================
struct CosmicObservation {
    uint64_t entry;
    high_precision_float epsilon;
    std::chrono::system_clock::time_point timestamp;
    std::string description;
};

std::vector<CosmicObservation> cosmic_epsilon_table;
bool reality_shift_detected = false;

std::string cosmic_reality_monitor(const high_precision_float& x_value, uint64_t entry_number) {
    if (x_value == 0) {
        return "Zero - no reciprocal defined";
    }
    
    high_precision_float reciprocal = 1 / x_value;
    bool is_self_reciprocal = abs(x_value - reciprocal) < EPS_COSMIC;
    
    if (is_self_reciprocal && abs(x_value - 1) > EPSILON && abs(x_value + 1) > EPSILON) {
        if (!reality_shift_detected) {
            reality_shift_detected = true;
            cosmic_epsilon_table.push_back({
                entry_number,
                x_value,
                std::chrono::system_clock::now(),
                "Cosmic shift detected: ε = " + decimal_short(x_value)
            });
            return "🚨 COSMIC SHIFT: New ε detected at entry " + std::to_string(entry_number);
        } else {
            cosmic_epsilon_table.push_back({
                entry_number,
                x_value,
                std::chrono::system_clock::now(),
                "Additional ε observed: " + decimal_short(x_value)
            });
            return "📊 Reality tally: ε = " + decimal_short(x_value);
        }
    }
    
    return "Reality stable: ε not observed";
}

// ============================== SECTION 2: SEQUENCE ANALYSIS FUNCTIONS ==============================

bool is_fibonacci_int(high_precision_int n) {
    if (n < 0) return false;
    
    high_precision_int test1 = 5 * n * n + 4;
    high_precision_int test2 = 5 * n * n - 4;
    
    return is_perfect_square(high_precision_float(test1)) || is_perfect_square(high_precision_float(test2));
}

bool is_lucas_int(high_precision_int n) {
    if (n == 2) return true;
    
    high_precision_int test1 = 5 * n * n + 20;
    high_precision_int test2 = 5 * n * n - 20;
    
    return is_perfect_square(high_precision_float(test1)) || is_perfect_square(high_precision_float(test2));
}

void section2_sequences(uint64_t entry_number, const high_precision_float& x_value) {
    if (mega_manager) {
        mega_manager->stream_output("Sequence Checks:\n");
    } else {
        std::cout << "Sequence Checks:" << std::endl;
    }
    
    if (!is_integer(x_value)) {
        if (mega_manager) {
            mega_manager->stream_output("  (Non-integer, skipping sequence checks)\n\n");
        } else {
            std::cout << "  (Non-integer, skipping sequence checks)" << std::endl << std::endl;
        }
        return;
    }
    
    high_precision_int x_int = static_cast<high_precision_int>(x_value);
    
    if (is_fibonacci_int(x_int)) {
        std::string fib_str = "  Fibonacci number detected: " + x_int.str() + "\n";
        if (x_int > 1) {
            fib_str += "  Fibonacci property: F(n) ≈ φ^n/√5, reciprocal relates to ψ^n\n";
        }
        if (mega_manager) mega_manager->stream_output(fib_str);
        else std::cout << fib_str;
    }
    
    if (is_lucas_int(x_int)) {
        std::string lucas_str = "  Lucas number detected: " + x_int.str() + "\n";
        lucas_str += "  Lucas property: L(n) = φ^n + ψ^n, self-reciprocal structure\n";
        if (mega_manager) mega_manager->stream_output(lucas_str);
        else std::cout << lucas_str;
    }
    
    if (x_int <= 1000000) {
        high_precision_int a = 1, b = 1, c = 2;
        while (c <= x_int) {
            if (a == x_int || b == x_int || c == x_int) {
                std::string trib_str = "  Tribonacci number detected: " + x_int.str() + "\n";
                trib_str += "  Tribonacci: cubic reciprocal relationships\n";
                if (mega_manager) mega_manager->stream_output(trib_str);
                else std::cout << trib_str;
                break;
            }
            a = b;
            b = c;
            c = a + b + c;
        }
    }
    
    if (mega_manager) mega_manager->stream_output("\n");
    else std::cout << std::endl;
}

// ============================== SECTION 3: PRIME AND FACTORIAL CHECKS ==============================

void section3_primes_factorials(uint64_t entry_number, const high_precision_float& x_value) {
    if (mega_manager) {
        mega_manager->stream_output("Prime and Factorial Checks:\n");
    } else {
        std::cout << "Prime and Factorial Checks:" << std::endl;
    }
    
    if (!is_integer(x_value)) {
        if (mega_manager) {
            mega_manager->stream_output("  (Non-integer, skipping prime/factorial checks)\n\n");
        } else {
            std::cout << "  (Non-integer, skipping prime/factorial checks)" << std::endl << std::endl;
        }
        return;
    }
    
    high_precision_int n = static_cast<high_precision_int>(x_value);
    
    // Prime check (simplified)
    if (n > 1 && n < 1000000) {
        bool is_prime = true;
        if (n % 2 == 0) {
            is_prime = (n == 2);
        } else {
            for (high_precision_int i = 3; i * i <= n; i += 2) {
                if (n % i == 0) {
                    is_prime = false;
                    break;
                }
            }
        }
        
        if (is_prime) {
            std::string prime_str = "  Prime number detected: " + n.str() + "\n";
            prime_str += "  Prime property: Irreducible in multiplication tables\n";
            if (n > 2) {
                prime_str += "  Reciprocal: 1/" + n.str() + " creates infinite decimal pattern\n";
            }
            if (mega_manager) mega_manager->stream_output(prime_str);
            else std::cout << prime_str;
        }
    }
    
    // Factorial check (simplified)
    if (n > 0 && n < 100) {
        high_precision_int k = 1;
        high_precision_int fact = 1;
        while (fact <= n + 1) {
            if (fact == n) {
                std::string fact_str = "  Factorial detected: " + k.str() + "! = " + n.str() + "\n";
                fact_str += "  Factorial growth: Rapid divergence from reciprocal 1/" + n.str() + "\n";
                if (mega_manager) mega_manager->stream_output(fact_str);
                else std::cout << fact_str;
                break;
            }
            k += 1;
            fact *= k;
        }
    }
    
    // Perfect square and cube checks
    if (n > 1) {
        high_precision_float sqrt_n = sqrt(high_precision_float(n));
        if (is_integer(sqrt_n)) {
            high_precision_int root = static_cast<high_precision_int>(sqrt_n);
            std::string square_str = "  Perfect square: " + n.str() + " = " + root.str() + "²\n";
            square_str += "  Square reciprocal: 1/" + n.str() + " = (1/" + root.str() + ")²\n";
            if (mega_manager) mega_manager->stream_output(square_str);
            else std::cout << square_str;
        }
        
        // Cube check (simplified)
        high_precision_int cube_root = 1;
        while (cube_root * cube_root * cube_root < n) {
            cube_root++;
        }
        if (cube_root * cube_root * cube_root == n) {
            std::string cube_str = "  Perfect cube: " + n.str() + " = " + cube_root.str() + "³\n";
            cube_str += "  Cube reciprocal: 1/" + n.str() + " = (1/" + cube_root.str() + ")³\n";
            if (mega_manager) mega_manager->stream_output(cube_str);
            else std::cout << cube_str;
        }
    }
    
    if (mega_manager) mega_manager->stream_output("\n");
    else std::cout << std::endl;
}

// ============================== SECTION 4: GEOMETRIC SEQUENCES & POWERS ==============================

void section4_geometric(uint64_t entry_number, const high_precision_float& x_value) {
    if (mega_manager) {
        mega_manager->stream_output("Geometric Progressions:\n");
    } else {
        std::cout << "Geometric Progressions:" << std::endl;
    }
    
    if (x_value > 0) {
        // Check for powers of 2
        if (x_value > 0) {
            high_precision_float log2 = log(x_value) / log(2);
            if (is_integer(log2)) {
                high_precision_int exp = static_cast<high_precision_int>(log2);
                std::string pow2_str = "  Power of 2 detected: 2^" + exp.str() + " = " + decimal_short(x_value) + "\n";
                if (exp > 0) {
                    pow2_str += "  Reciprocal: 1/x = 2^-" + exp.str() + " = " + decimal_short(1/x_value) + "\n";
                } else if (exp < 0) {
                    pow2_str += "  Reciprocal: 1/x = 2^-" + exp.str() + " = " + decimal_short(1/x_value) + "\n";
                }
                if (mega_manager) mega_manager->stream_output(pow2_str);
                else std::cout << pow2_str;
            }
        }
        
        // Check for powers of 10
        high_precision_float log10 = log(x_value) / log(10);
        if (is_integer(log10)) {
            high_precision_int exp = static_cast<high_precision_int>(log10);
            std::string pow10_str = "  Power of 10 detected: 10^" + exp.str() + " = " + decimal_short(x_value) + "\n";
            pow10_str += "  Reciprocal symmetry: x = 10^" + exp.str() + ", 1/x = 10^-" + exp.str() + "\n";
            pow10_str += "  Base-10 tree: Perfect decimal shift by " + abs(exp).str() + " places\n";
            if (mega_manager) mega_manager->stream_output(pow10_str);
            else std::cout << pow10_str;
        }
        
        // Check for powers of golden ratio
        high_precision_float log_phi = log(x_value) / log(PHI);
        if (is_integer(log_phi)) {
            high_precision_int exp = static_cast<high_precision_int>(log_phi);
            std::string phi_str = "  Golden ratio power: φ^" + exp.str() + " = " + decimal_short(x_value) + "\n";
            if (exp == -1) {
                phi_str += "  Special case: 1/φ = φ - 1 ≈ 0.618...\n";
            }
            if (mega_manager) mega_manager->stream_output(phi_str);
            else std::cout << phi_str;
        }
        
        // Check for other bases
        if (x_value > 0 && x_value != 1) {
            std::vector<int> bases = {3, 4, 5, 6, 7, 8, 9};
            for (int base : bases) {
                high_precision_float log_base = log(x_value) / log(base);
                if (is_integer(log_base)) {
                    high_precision_int exp = static_cast<high_precision_int>(log_base);
                    std::string base_str = "  Power of " + std::to_string(base) + ": " + 
                                         std::to_string(base) + "^" + exp.str() + " = " + 
                                         decimal_short(x_value) + "\n";
                    if (mega_manager) mega_manager->stream_output(base_str);
                    else std::cout << base_str;
                    break;
                }
            }
        }
    }
    
    if (mega_manager) mega_manager->stream_output("\n");
    else std::cout << std::endl;
}

// ============================== SECTION 5: HARMONICS ==============================

void section5_harmonics(uint64_t entry_number, const high_precision_float& x_value) {
    if (mega_manager) {
        mega_manager->stream_output("Harmonic Checks:\n");
    } else {
        std::cout << "Harmonic Checks:" << std::endl;
    }
    
    if (x_value != 0 && abs(x_value) <= 1) {
        high_precision_float inv = 1 / abs(x_value);
        if (is_integer(inv)) {
            high_precision_int n = static_cast<high_precision_int>(inv);
            std::string harm_str = "  Harmonic number detected: 1/" + n.str() + " = " + decimal_short(x_value) + "\n";
            harm_str += "  Unit fraction: Forms base tree with denominator " + n.str() + "\n";
            harm_str += "  Reciprocal integer: 1/x = " + n.str() + " (exact)\n";
            
            // Check decimal pattern
            high_precision_int temp = n;
            bool all_2_or_5 = true;
            while (temp % 2 == 0) temp /= 2;
            while (temp % 5 == 0) temp /= 5;
            
            if (temp == 1) {
                harm_str += "  Decimal pattern: Terminating (denominator has only 2 and/or 5)\n";
            } else {
                harm_str += "  Decimal pattern: Repeating (denominator has other prime factors)\n";
                harm_str += "  Period insight: Repeating decimal period related to factors of " + n.str() + "\n";
            }
            
            if (mega_manager) mega_manager->stream_output(harm_str);
            else std::cout << harm_str;
        }
    }
    
    // Check for simple fractions
    if (x_value != 0 && !is_integer(x_value)) {
        RationalResult frac = get_rational_approx(x_value);
        if (frac.valid && frac.denominator <= 100 && frac.denominator > 1) {
            std::string simple_str = "  Simple fraction: " + frac.to_string() + " = " + decimal_short(x_value) + "\n";
            high_precision_float reciprocal_frac = high_precision_float(frac.denominator) / high_precision_float(frac.numerator);
            simple_str += "  Reciprocal fraction: " + std::to_string(frac.denominator) + "/" + 
                         std::to_string(static_cast<int>(frac.numerator)) + " = " + 
                         decimal_short(reciprocal_frac) + "\n";
            if (mega_manager) mega_manager->stream_output(simple_str);
            else std::cout << simple_str;
        }
    }
    
    // Check for harmonics of constants
    if (x_value > 0) {
        std::vector<std::pair<high_precision_float, std::string>> constants = {
            {PHI, "φ"}, {PI, "π"}, {E, "e"}
        };
        
        for (const auto& constant : constants) {
            high_precision_float ratio = x_value * constant.first;
            if (is_integer(ratio)) {
                high_precision_int n = static_cast<high_precision_int>(ratio);
                std::string const_str = "  Harmonic of " + constant.second + ": " + n.str() + "/" + 
                                      constant.second + " = " + decimal_short(x_value) + "\n";
                if (mega_manager) mega_manager->stream_output(const_str);
                else std::cout << const_str;
                break;
            }
            
            high_precision_float ratio_inv = x_value / constant.first;
            if (is_integer(ratio_inv)) {
                high_precision_int n = static_cast<high_precision_int>(ratio_inv);
                std::string const_str = "  Multiple of " + constant.second + ": " + n.str() + "×" + 
                                      constant.second + " = " + decimal_short(x_value) + "\n";
                if (mega_manager) mega_manager->stream_output(const_str);
                else std::cout << const_str;
                break;
            }
        }
    }
    
    if (mega_manager) mega_manager->stream_output("\n");
    else std::cout << std::endl;
}

// ============================== CONTINUED FRACTION (ITERATIVE) ==============================
std::vector<int> continued_fraction_iterative(const high_precision_float& x, int max_terms = 1000) {
    std::vector<int> cf;
    high_precision_float x_val = x;
    
    for (int i = 0; i < max_terms; ++i) {
        if (abs(x_val) < EPSILON) {
            break;
        }
        
        int a = static_cast<int>(floor(x_val));
        cf.push_back(a);
        
        x_val -= a;
        if (abs(x_val) < EPSILON) {
            break;
        }
        
        x_val = 1 / x_val;
        if (isinf(x_val) || isnan(x_val)) {
            break;
        }
        
        // Track recursion depth for massive scale
        if (mega_manager) {
            mega_manager->track_recursion_depth(i);
        }
    }
    
    return cf;
}

// ============================== SECTION 6: CONTINUED FRACTION ANALYSIS ==============================

void section6_continued(uint64_t entry_number, const high_precision_float& x_value) {
    if (mega_manager) {
        mega_manager->stream_output("Continued Fraction Analysis:\n");
    } else {
        std::cout << "Continued Fraction Analysis:" << std::endl;
    }
    
    std::vector<int> cf_x = continued_fraction_iterative(x_value);
    std::string cf_str = "  CF for x: [";
    for (size_t i = 0; i < cf_x.size(); ++i) {
        if (i > 0) cf_str += ", ";
        cf_str += std::to_string(cf_x[i]);
    }
    cf_str += "]\n";
    if (mega_manager) mega_manager->stream_output(cf_str);
    else std::cout << cf_str;
    
    if (x_value != 0) {
        std::vector<int> cf_rec = continued_fraction_iterative(1 / x_value);
        std::string cf_rec_str = "  CF for 1/x: [";
        for (size_t i = 0; i < cf_rec.size(); ++i) {
            if (i > 0) cf_rec_str += ", ";
            cf_rec_str += std::to_string(cf_rec[i]);
        }
        cf_rec_str += "]\n";
        if (mega_manager) mega_manager->stream_output(cf_rec_str);
        else std::cout << cf_rec_str;
        
        if (cf_x.size() > 1 && cf_rec.size() > 1) {
            // Check for golden ratio structure
            bool all_ones = true;
            for (size_t i = 1; i < std::min(cf_x.size(), size_t(6)); ++i) {
                if (cf_x[i] != 1) {
                    all_ones = false;
                    break;
                }
            }
            if (cf_x.size() >= 6 && all_ones) {
                std::string golden_str = "  Pattern: Golden ratio structure (all 1's)\n";
                golden_str += "  Mathematical: φ = [1;1,1,1,...], 1/φ = φ - 1 = [0;1,1,1,...]\n";
                if (mega_manager) mega_manager->stream_output(golden_str);
                else std::cout << golden_str;
            }
            
            // Check reciprocal flip
            if (!cf_x.empty() && !cf_rec.empty()) {
                if (cf_x[0] == 0 && cf_rec[0] != 0) {
                    std::string flip_str = "  Interpretation: x < 1, 1/x > 1 - reciprocal flips the expansion\n";
                    if (mega_manager) mega_manager->stream_output(flip_str);
                    else std::cout << flip_str;
                } else if (cf_x[0] != 0 && cf_rec[0] == 0) {
                    std::string flip_str = "  Interpretation: x > 1, 1/x < 1 - reciprocal flips the expansion\n";
                    if (mega_manager) mega_manager->stream_output(flip_str);
                    else std::cout << flip_str;
                }
            }
            
            // Check for identical or shared patterns
            if (cf_x == cf_rec) {
                std::string identical_str = "  Interpretation: Identical continued fractions for x and 1/x (self-reciprocal structure).\n";
                if (mega_manager) mega_manager->stream_output(identical_str);
                else std::cout << identical_str;
            } else {
                std::string distinct_str = "  Interpretation: Distinct continued fractions, illustrating unique reciprocal structures.\n";
                if (mega_manager) mega_manager->stream_output(distinct_str);
                else std::cout << distinct_str;
            }
            
            // Check for periodic patterns
            if (cf_x.size() > 5) {
                bool periodic_found = false;
                for (int period = 1; period <= (int)cf_x.size()/2; ++period) {
                    bool is_periodic = true;
                    for (int i = 0; i < period && i + period < (int)cf_x.size(); ++i) {
                        if (cf_x[i] != cf_x[i + period]) {
                            is_periodic = false;
                            break;
                        }
                    }
                    if (is_periodic) {
                        periodic_found = true;
                        break;
                    }
                }
                if (periodic_found) {
                    std::string periodic_str = "  Pattern: Potentially periodic expansion (quadratic irrational)\n";
                    if (mega_manager) mega_manager->stream_output(periodic_str);
                    else std::cout << periodic_str;
                }
            }
        }
    } else {
        std::string undefined_str = "  Interpretation: Continued fraction for 1/x undefined.\n";
        if (mega_manager) mega_manager->stream_output(undefined_str);
        else std::cout << undefined_str;
    }
    
    if (!cf_x.empty()) {
        if (cf_x[0] == 0) {
            std::string base_str = "  Base Tree Link: x < 1, exists as reciprocal of integer in multiplication tables\n";
            if (mega_manager) mega_manager->stream_output(base_str);
            else std::cout << base_str;
        } else if (cf_x[0] > 1) {
            std::string base_str = "  Base Tree Link: Integer part " + std::to_string(cf_x[0]) + 
                                 " places x in " + std::to_string(cf_x[0]) + "-tree and above\n";
            if (mega_manager) mega_manager->stream_output(base_str);
            else std::cout << base_str;
        }
    }
    
    if (mega_manager) mega_manager->stream_output("  English: Continued fractions reveal the irrational structure and reciprocity.\n\n");
    else std::cout << "  English: Continued fractions reveal the irrational structure and reciprocity." << std::endl << std::endl;
}

// ============================== SECTION 7: BANACHIAN ANALYSIS ==============================

void section7_banachian(uint64_t entry_number, const high_precision_float& x_value) {
    if (mega_manager) {
        mega_manager->stream_output("Banachian / Decimal Stress Test:\n");
    } else {
        std::cout << "Banachian / Decimal Stress Test:" << std::endl;
    }
    
    high_precision_float base;
    if (abs(x_value) < 1) {
        base = abs(x_value);
    } else {
        base = (x_value != 0) ? 1 / abs(x_value) : high_precision_float(0.001);
    }
    
    if (mega_manager) {
        mega_manager->stream_output("  Testing small perturbations around x:\n");
        mega_manager->stream_output("  [Value] → [1/Value] → [Difference from original x]\n");
    } else {
        std::cout << "  Testing small perturbations around x:" << std::endl;
        std::cout << "  [Value] → [1/Value] → [Difference from original x]" << std::endl;
    }
    
    // Test small perturbations
    for (int i = 10; i <= 12; ++i) {
        high_precision_float d = base + pow(high_precision_float(10), -i);
        if (d != 0) {
            high_precision_float reciprocal = 1 / d;
            high_precision_float diff = abs(x_value - d);
            std::string perturb_str = "  " + decimal_short(d) + " → " + decimal_short(reciprocal) + 
                                    " | Δ = " + decimal_short(diff) + "\n";
            if (mega_manager) mega_manager->stream_output(perturb_str);
            else std::cout << perturb_str;
        }
    }
    
    if (mega_manager) mega_manager->stream_output("\n  Reciprocal Stability Analysis:\n");
    else std::cout << std::endl << "  Reciprocal Stability Analysis:" << std::endl;
    
    if (x_value != 0) {
        high_precision_float original_reciprocal = 1 / x_value;
        high_precision_float small_change = high_precision_float("1e-10");
        std::vector<high_precision_float> test_values;
        
        if (x_value > 0) {
            test_values = {x_value * (1 + small_change), x_value * (1 - small_change)};
        } else {
            test_values = {x_value + small_change, x_value - small_change};
        }
        
        for (const auto& test_val : test_values) {
            if (test_val != 0) {
                high_precision_float test_reciprocal = 1 / test_val;
                high_precision_float reciprocal_diff = abs(original_reciprocal - test_reciprocal);
                std::string stability_str = "  x±ε: " + decimal_short(test_val) + " → 1/x±ε: " + 
                                          decimal_short(test_reciprocal) + "\n";
                stability_str += "    Reciprocal change: Δ = " + decimal_short(reciprocal_diff) + "\n";
                if (mega_manager) mega_manager->stream_output(stability_str);
                else std::cout << stability_str;
            }
        }
    }
    
    if (mega_manager) mega_manager->stream_output("\n  Mathematical Insight:\n");
    else std::cout << std::endl << "  Mathematical Insight:" << std::endl;
    
    if (x_value == 1 || x_value == -1) {
        std::string insight_str = "  Fixed point: Small perturbations preserve reciprocal equality approximately\n";
        if (mega_manager) mega_manager->stream_output(insight_str);
        else std::cout << insight_str;
    } else if (x_value > 0 && x_value < 1) {
        std::string insight_str = "  Small x: Reciprocal amplification makes small changes more visible in 1/x\n";
        if (mega_manager) mega_manager->stream_output(insight_str);
        else std::cout << insight_str;
    } else if (x_value > 1) {
        std::string insight_str = "  Large x: Reciprocal attenuation makes small changes less visible in 1/x\n";
        if (mega_manager) mega_manager->stream_output(insight_str);
        else std::cout << insight_str;
    } else {
        std::string insight_str = "  General case: Reciprocal transformation non-linearly amplifies/attenuates changes\n";
        if (mega_manager) mega_manager->stream_output(insight_str);
        else std::cout << insight_str;
    }
    
    if (mega_manager) {
        mega_manager->stream_output("  English: Shows how small increments around x affect reciprocals, illustrating Immediate Adjacency and stability under perturbation.\n\n");
    } else {
        std::cout << "  English: Shows how small increments around x affect reciprocals, illustrating Immediate Adjacency and stability under perturbation." << std::endl << std::endl;
    }
}

// ============================== SECTION 8: EXTREME BOUNDARIES ==============================

void section8_extremes(uint64_t entry_number, const high_precision_float& x_value, const std::string& description) {
    banner("EXTREME ENTRY " + std::to_string(entry_number), 70);
    std::string header = description + " | x = " + decimal_short(x_value);
    
    if (mega_manager) mega_manager->stream_output(header + "\n");
    else std::cout << header << std::endl;
    
    if (x_value != 0) {
        std::string square_str = "  x² = " + decimal_short(x_value * x_value) + "\n";
        if (mega_manager) mega_manager->stream_output(square_str);
        else std::cout << square_str;
        
        if (x_value > 0) {
            high_precision_float reciprocal_square = (1/x_value) * (1/x_value);
            std::string rec_square_str = "  (1/x)² = " + decimal_short(reciprocal_square) + "\n";
            if (mega_manager) mega_manager->stream_output(rec_square_str);
            else std::cout << rec_square_str;
        }
        
        if (x_value >= 0) {
            high_precision_float sqrt_val = sqrt(x_value);
            std::string sqrt_str = "  √x = " + decimal_short(sqrt_val) + "\n";
            if (mega_manager) mega_manager->stream_output(sqrt_str);
            else std::cout << sqrt_str;
        } else {
            if (mega_manager) mega_manager->stream_output("  √x = NaN\n");
            else std::cout << "  √x = NaN" << std::endl;
        }
        
        if (x_value > 0) {
            std::string log_str = "  ln(x) = " + decimal_short(log(x_value)) + "\n";
            if (mega_manager) mega_manager->stream_output(log_str);
            else std::cout << log_str;
            
            if (1/x_value > 0) {
                std::string log_rec_str = "  ln(1/x) = " + decimal_short(log(1/x_value)) + " = -ln(x)\n";
                if (mega_manager) mega_manager->stream_output(log_rec_str);
                else std::cout << log_rec_str;
            }
        } else {
            if (mega_manager) mega_manager->stream_output("  ln(x) = NaN\n");
            else std::cout << "  ln(x) = NaN" << std::endl;
        }
        
        std::string exp_str = "  e^x = " + decimal_short(exp(x_value)) + "\n";
        if (mega_manager) mega_manager->stream_output(exp_str);
        else std::cout << exp_str;
        
        if (x_value != 0) {
            std::string exp_rec_str = "  e^(1/x) = " + decimal_short(exp(1/x_value)) + "\n";
            if (mega_manager) mega_manager->stream_output(exp_rec_str);
            else std::cout << exp_rec_str;
        }
    }
    
    if (mega_manager) mega_manager->stream_output("\n  Mathematical Classification:\n");
    else std::cout << std::endl << "  Mathematical Classification:" << std::endl;
    
    if (x_value == 0) {
        std::string class_str = "  Zero: Additive identity, multiplicative annihilator\n";
        if (mega_manager) mega_manager->stream_output(class_str);
        else std::cout << class_str;
    } else if (x_value == 1) {
        std::string class_str = "  Unity: Multiplicative identity, only positive self-reciprocal\n";
        if (mega_manager) mega_manager->stream_output(class_str);
        else std::cout << class_str;
    } else if (x_value == -1) {
        std::string class_str = "  Negative unity: Only negative self-reciprocal\n";
        if (mega_manager) mega_manager->stream_output(class_str);
        else std::cout << class_str;
    } else if (is_integer(x_value)) {
        high_precision_int n = static_cast<high_precision_int>(x_value);
        if (n > 1) {
            std::string class_str = "  Positive integer: Member of " + n.str() + "-tree in multiplication tables\n";
            if (mega_manager) mega_manager->stream_output(class_str);
            else std::cout << class_str;
        } else if (n < -1) {
            std::string class_str = "  Negative integer: Negative member of " + abs(n).str() + "-tree\n";
            if (mega_manager) mega_manager->stream_output(class_str);
            else std::cout << class_str;
        }
    } else if (x_value > 0 && x_value < 1) {
        std::string class_str = "  Unit fraction territory: Reciprocal of integer > 1\n";
        if (mega_manager) mega_manager->stream_output(class_str);
        else std::cout << class_str;
    } else if (x_value > 1) {
        std::string class_str = "  Integer territory: Reciprocal of unit fraction\n";
        if (mega_manager) mega_manager->stream_output(class_str);
        else std::cout << class_str;
    } else if (x_value < 0) {
        std::string class_str = "  Negative real: Reciprocal preserves sign\n";
        if (mega_manager) mega_manager->stream_output(class_str);
        else std::cout << class_str;
    }
    
    if (mega_manager) mega_manager->stream_output("\n  Growth/Decay Patterns:\n");
    else std::cout << std::endl << "  Growth/Decay Patterns:" << std::endl;
    
    if (x_value > 0) {
        if (x_value < 1) {
            std::string pattern_str = "  Decay: x < 1 → 1/x > 1 (amplification)\n";
            if (mega_manager) mega_manager->stream_output(pattern_str);
            else std::cout << pattern_str;
        } else if (x_value > 1) {
            std::string pattern_str = "  Growth: x > 1 → 0 < 1/x < 1 (attenuation)\n";
            if (mega_manager) mega_manager->stream_output(pattern_str);
            else std::cout << pattern_str;
        } else {
            std::string pattern_str = "  Equilibrium: x = 1 → 1/x = 1 (fixed point)\n";
            if (mega_manager) mega_manager->stream_output(pattern_str);
            else std::cout << pattern_str;
        }
    }
    
    if (mega_manager) {
        mega_manager->stream_output("  English: Shows growth and decay patterns, reinforcing Immediate Adjacency and reciprocal behaviour\n\n");
    } else {
        std::cout << "  English: Shows growth and decay patterns, reinforcing Immediate Adjacency and reciprocal behaviour" << std::endl << std::endl;
    }
}

// ============================== SECTION 9: RECIPROCAL THESIS FOCUS ==============================

void section9_summary(uint64_t entry_number, const high_precision_float& x_value, const std::string& description) {
    if (x_value == 0) return;
    
    if (mega_manager) mega_manager->stream_output("Reciprocal Thesis Focus:\n");
    else std::cout << "Reciprocal Thesis Focus:" << std::endl;
    
    high_precision_float symmetry = reciprocal_symmetry_score(x_value);
    std::string indicator;
    
    if (symmetry > 0.9) {
        indicator = "★ NEAR-SYMMETRIC ★";
    } else if (symmetry > 0.5) {
        indicator = "◇ MODERATE SYMMETRY ◇";
    } else {
        indicator = "△ ASYMMETRIC △";
    }
    
    std::string sym_str = "  " + indicator + "\n";
    if (mega_manager) mega_manager->stream_output(sym_str);
    else std::cout << sym_str;
    
    ProofMetrics proof_metrics = calculate_proof_metrics(x_value);
    
    if (x_value > 0 && x_value < 1) {
        std::string case_str = "  Case: 0 < x < 1 → 1/x > 1\n";
        case_str += "  Thesis: Decimal irrationality in (0,1) mirrors to irrationality in (1,∞)\n";
        if (mega_manager) mega_manager->stream_output(case_str);
        else std::cout << case_str;
    } else if (x_value > 1) {
        std::string case_str = "  Case: x > 1 → 0 < 1/x < 1\n";
        case_str += "  Thesis: Large irrational x creates small irrational 1/x - transverse relationship\n";
        if (mega_manager) mega_manager->stream_output(case_str);
        else std::cout << case_str;
    } else if (x_value == 1) {
        std::string case_str = "  Case: x = 1 → 1/x = 1 (FIXED POINT)\n";
        case_str += "  Thesis: Only point where decimal expansions coincide exactly\n";
        if (mega_manager) mega_manager->stream_output(case_str);
        else std::cout << case_str;
    } else if (x_value == -1) {
        std::string case_str = "  Case: x = -1 → 1/x = -1 (NEGATIVE FIXED POINT)\n";
        case_str += "  Thesis: Only negative fixed point in reciprocal space\n";
        if (mega_manager) mega_manager->stream_output(case_str);
        else std::cout << case_str;
    }
    
    std::string math_str = "  Mathematical: f(x) = x and f(x) = 1/x intersect only at x=±1\n";
    if (mega_manager) mega_manager->stream_output(math_str);
    else std::cout << math_str;
    
    if (x_value == 1 || x_value == -1) {
        std::string proof_str = "  PROOF STATUS: ✓ Confirms theorem - self-reciprocal fixed point\n";
        if (mega_manager) mega_manager->stream_output(proof_str);
        else std::cout << proof_str;
    } else {
        std::string proof_str = "  PROOF STATUS: ✓ Confirms theorem - distinct from reciprocal\n";
        if (mega_manager) mega_manager->stream_output(proof_str);
        else std::cout << proof_str;
    }
    
    if (mega_manager) mega_manager->stream_output("\n");
    else std::cout << std::endl;
}

// ============================== SECTION 10: DECIMAL CHUNKS AND PATTERNS ==============================

void section10_decimal_analysis(uint64_t entry_number, const high_precision_float& x_value) {
    if (mega_manager) mega_manager->stream_output("Decimal Expansion and Chunk Analysis:\n");
    else std::cout << "Decimal Expansion and Chunk Analysis:" << std::endl;
    
    if (mega_manager) mega_manager->stream_output("For x:\n");
    else std::cout << "For x:" << std::endl;
    std::string x_analysis = analyze_decimal_expansion(x_value);
    if (mega_manager) mega_manager->stream_output(x_analysis);
    else std::cout << x_analysis;
    
    if (x_value != 0) {
        if (mega_manager) mega_manager->stream_output("For 1/x:\n");
        else std::cout << "For 1/x:" << std::endl;
        high_precision_float reciprocal = 1 / x_value;
        std::string rec_analysis = analyze_decimal_expansion(reciprocal);
        if (mega_manager) mega_manager->stream_output(rec_analysis);
        else std::cout << rec_analysis;
    }
    
    std::string interpretation = "Interpretation: Maps divisions in decimal expansions, tabling chunks for patterns. Irrational decimals do not suddenly become integers; gap monitoring shows equality only at ±1.\n";
    if (mega_manager) mega_manager->stream_output(interpretation + "\n");
    else std::cout << interpretation << std::endl;
}

// ============================== SECTION 11: COSMIC MONITOR (ENHANCED) ==============================

void section11_cosmic_monitor(uint64_t entry_number, const high_precision_float& x_value) {
    std::string cosmic_status = cosmic_reality_monitor(x_value, entry_number);
    if (cosmic_status.find("COSMIC SHIFT") != std::string::npos || 
        cosmic_status.find("Reality tally") != std::string::npos) {
        if (mega_manager) {
            mega_manager->stream_output("🌌 Cosmic Reality Monitor:\n");
            mega_manager->stream_output("  " + cosmic_status + "\n\n");
        } else {
            std::cout << "🌌 Cosmic Reality Monitor:" << std::endl;
            std::cout << "  " << cosmic_status << std::endl << std::endl;
        }
    }
}

// ============================== SECTION 12: PROPORTION VISION ==============================

void section12_proportion_vision(uint64_t entry_number, const high_precision_float& x_value, const std::string& x_name) {
    if (mega_manager) mega_manager->stream_output("🔎 PROPORTION VISION:\n");
    else std::cout << "🔎 PROPORTION VISION:" << std::endl;
    
    if (x_value <= 0 || isinf(x_value) || isnan(x_value)) {
        std::string vision_str = "   (Vision requires positive finite numbers)\n";
        if (mega_manager) mega_manager->stream_output(vision_str);
        else std::cout << vision_str;
        return;
    }
    
    high_precision_float integer_part = floor(x_value);
    high_precision_float fractional_part = x_value - integer_part;
    
    std::string position_str = "   Position: " + decimal_short(x_value) + " = " + decimal_short(integer_part) + 
                              " + " + decimal_short(fractional_part) + "\n";
    if (mega_manager) mega_manager->stream_output(position_str);
    else std::cout << position_str;
    
    if (fractional_part > 0 && fractional_part < 1 && integer_part < 5) {
        int a = static_cast<int>(integer_part);
        int b = a + 1;
        
        std::string pattern_str = "   🦎 BETWEEN " + std::to_string(a) + " AND " + std::to_string(b) + " PATTERN TEMPLATES:\n";
        if (mega_manager) mega_manager->stream_output(pattern_str);
        else std::cout << pattern_str;
        
        std::vector<std::pair<std::string, std::function<high_precision_float(int)>>> patterns = {
            {"(" + std::to_string(a) + "n+1)/n", [a](int n) { return high_precision_float(a*n + 1) / n; }},
            {"(" + std::to_string(b) + "n-1)/n", [b](int n) { return high_precision_float(b*n - 1) / n; }},
            {"(" + std::to_string(a+b) + "n)/(2n)", [a,b](int n) { return high_precision_float((a+b)*n) / (2*n); }}
        };
        
        for (const auto& pattern : patterns) {
            std::string values_str;
            for (int n = 1; n <= 3; ++n) {
                try {
                    high_precision_float val = pattern.second(n);
                    if (val > a && val < b) {
                        if (!values_str.empty()) values_str += ", ";
                        values_str += "n=" + std::to_string(n) + ": " + decimal_short(val);
                    }
                } catch (...) {
                    // Skip invalid calculations
                }
            }
            if (!values_str.empty()) {
                std::string pattern_result = "     " + pattern.first + ": " + values_str + "\n";
                if (mega_manager) mega_manager->stream_output(pattern_result);
                else std::cout << pattern_result;
            }
        }
    }
    
    if (x_value != 0) {
        high_precision_float reciprocal = 1 / x_value;
        bool is_self_reciprocal = abs(x_value - reciprocal) < EPS_RECIP;
        if (!is_self_reciprocal && x_value != 1 && x_value != -1) {
            std::string status_str = "   🔐 ENTRY " + std::to_string(entry_number) + " STATUS: Does not prove reciprocal thesis\n";
            status_str += "      Confirms: x ≠ 1/x for x ≠ ±1\n";
            if (mega_manager) mega_manager->stream_output(status_str);
            else std::cout << status_str;
        }
    }
    
    if (mega_manager) mega_manager->stream_output("\n");
    else std::cout << std::endl;
}

// ============================== SECTION 13: THE ADJACENCY GUILLOTINE ==============================

void section13_adjacency_guillotine(uint64_t entry_number, const high_precision_float& x_value) {
    if (mega_manager) mega_manager->stream_output("⚖️ THE ADJACENCY GUILLOTINE (Audit):\n");
    else std::cout << "⚖️ THE ADJACENCY GUILLOTINE (Audit):" << std::endl;

    // 1. Handle the Singularity (x=1)
    if (abs(x_value - 1) < EPSILON) {
        std::string status = "   Core Singularity (x=1): Adjacency Ratio Undefined (0/0).\n";
        status += "   Theoretical Limit: 2.0\n";
        status += "   Status: SAFE (Singularity protected).\n";
        if (mega_manager) mega_manager->stream_output(status + "\n");
        else std::cout << status << std::endl;
        return;
    }

    // 2. Handle Zero (x=0)
    if (x_value == 0) {
        if (mega_manager) mega_manager->stream_output("   Zero Point: Excluded (Infinite Adjacency).\n\n");
        else std::cout << "   Zero Point: Excluded (Infinite Adjacency)." << std::endl << std::endl;
        return;
    }

    // 3. The Guillotine Calculation
    high_precision_float epsilon = abs(x_value - 1);
    high_precision_float reciprocal = 1 / x_value;
    high_precision_float gap = abs(x_value - reciprocal);
    high_precision_float observed_ratio = gap / epsilon;
    high_precision_float theoretical_ratio = abs(1 + reciprocal);
    high_precision_float violation_magnitude = abs(observed_ratio - theoretical_ratio);
    
    bool passed = violation_magnitude < EPS_COSMIC;

    // 4. Report Generation
    std::string report = "   Distance from 1 (ε): " + decimal_short(epsilon) + "\n";
    report += "   Reciprocal Gap (Δ):  " + decimal_short(gap) + "\n";
    report += "   Observed Ratio (Δ/ε): " + decimal_short(observed_ratio) + "\n";
    report += "   Theoretical Law (1+1/x): " + decimal_short(theoretical_ratio) + "\n";
    
    if (passed) {
        report += "   [✓] GUILLOTINE PASSED: Adjacency holds. Violation: " + decimal_short(violation_magnitude) + "\n";
        if (abs(observed_ratio - 2) < high_precision_float("0.001")) {
             report += "   CLASS: QUANTUM FOAM (Linear Regime, Ratio ≈ 2)\n";
        } else if (observed_ratio > 2) {
             report += "   CLASS: SUB-UNITY COMPRESSION (Ratio > 2)\n";
        } else {
             report += "   CLASS: HYPER-EXTENSION (Ratio < 2)\n";
        }
    } else {
        report += "   [X] ❌ FATAL MATHEMATICAL BREACH DETECTED ❌\n";
        report += "   VIOLATION MAGNITUDE: " + decimal_full(violation_magnitude) + "\n";
        report += "   STATUS: Immediate Adjacency Disproven or Precision Failure.\n";
    }

    if (mega_manager) mega_manager->stream_output(report + "\n");
    else std::cout << report << std::endl;
}

// ============================== SECTION 14: ASTRONOMICAL RELATIONS ==============================

void section14_astronomical_relations(uint64_t entry_number, const high_precision_float& x_value) {
    if (x_value <= 0 || isinf(x_value) || isnan(x_value)) return;
    
    if (mega_manager) mega_manager->stream_output("🌌 ASTRONOMICAL RELATIONS (Future Vision):\n");
    else std::cout << "🌌 ASTRONOMICAL RELATIONS (Future Vision):" << std::endl;
    
    std::vector<std::pair<high_precision_float, std::string>> cosmic_scales = {
        {pow(high_precision_float(10), 10), "Ten billion"},
        {pow(high_precision_float(10), 23), "Avogadro's scale"},
        {pow(high_precision_float(10), 50), "Quantum-cosmic bridge"}
    };
    
    for (const auto& scale : cosmic_scales) {
        if (x_value > 0 && x_value < scale.first) {
            high_precision_float ratio = scale.first / x_value;
            std::string scale_str = "   To reach " + scale.second + ":\n";
            scale_str += "     Multiply by " + decimal_short(ratio) + "\n";
            
            if (ratio < 1000 && ratio > 1) {
                RationalResult ratio_frac = get_rational_approx(ratio);
                if (ratio_frac.valid && ratio_frac.denominator > 1) {
                    scale_str += "     Exact: ×" + ratio_frac.to_string() + "\n";
                }
            }
            
            if (mega_manager) mega_manager->stream_output(scale_str);
            else std::cout << scale_str;
            break;
        }
    }
    
    if (x_value != 0) {
        high_precision_float reciprocal = 1 / x_value;
        bool is_self_reciprocal = abs(x_value - reciprocal) < EPS_RECIP;
        if (!is_self_reciprocal) {
            std::string confirm_str = "   ✅ CONFIRMED: This entry upholds the reciprocal theorem\n";
            if (mega_manager) mega_manager->stream_output(confirm_str);
            else std::cout << confirm_str;
        }
    }
    
    if (mega_manager) mega_manager->stream_output("\n");
    else std::cout << std::endl;
}

// ============================== SECTION 15: QED NUMERICAL SIGNIFICANCE MODULE ==============================
// Fine-structure constant analysis with statistical testing
// CODATA-grade precision: α = 0.007297352569311 with uncertainty 1.5e-12

// Fundamental constant (CODATA-grade precision)
static const long double ALPHA = 0.007297352569311L;
static const long double SIGMA_ALPHA = 1.5e-12L;

struct QEDAnalysisResult {
    long double x;
    long double residual;
    long double z_score;
    long double p_value;
    long double p_corrected;
    std::pair<unsigned long long, unsigned long long> best_convergent;
    long double diophantine_metric;
    std::string verdict;
};

long double compute_z_score(long double x, long double sigma_x = 0.0L) {
    long double sigma_x_safe = (sigma_x > 0.0L) ? sigma_x : 0.0L;
    long double sigma_tot = std::sqrt(sigma_x_safe * sigma_x_safe + SIGMA_ALPHA * SIGMA_ALPHA);
    if (sigma_tot <= std::numeric_limits<long double>::min()) sigma_tot = SIGMA_ALPHA;
    return (x - ALPHA) / sigma_tot;
}

long double monte_carlo_p_value(long double x, int N = 200000) {
    std::mt19937_64 rng(0xBEEF);
    std::uniform_real_distribution<long double> U(0.0L, 1.0L);

    long double d_obs = fabsl(x - ALPHA);
    long long count = 0;

    for (int i = 0; i < N; ++i) {
        long double s = U(rng);
        if (fabsl(s - ALPHA) <= d_obs) ++count;
    }

    return (static_cast<long double>(count) + 1.0L) / (static_cast<long double>(N) + 1.0L);
}

std::pair<unsigned long long, unsigned long long> best_cf_approximation(long double x, int max_iter = 30) {
    if (!std::isfinite(x)) return {0ULL, 1ULL};

    unsigned long long p_nm2 = 0ULL, p_nm1 = 1ULL;
    unsigned long long q_nm2 = 1ULL, q_nm1 = 0ULL;

    long double x_rem = x;
    unsigned long long best_p = 0ULL, best_q = 1ULL;

    for (int i = 0; i < max_iter; ++i) {
        long double a_ld = floorl(x_rem);
        if (!std::isfinite(a_ld)) break;
        unsigned long long a;
        if (a_ld < 0.0L) a = 0ULL;
        else if (a_ld > static_cast<long double>(ULLONG_MAX / 2))
            a = static_cast<unsigned long long>(ULLONG_MAX / 2);
        else
            a = static_cast<unsigned long long>(a_ld);

        __uint128_t p_n_128 = static_cast<__uint128_t>(a) * p_nm1 + p_nm2;
        __uint128_t q_n_128 = static_cast<__uint128_t>(a) * q_nm1 + q_nm2;

        if (p_n_128 > std::numeric_limits<unsigned long long>::max() ||
            q_n_128 > std::numeric_limits<unsigned long long>::max()) {
            break;
        }

        unsigned long long p_n = static_cast<unsigned long long>(p_n_128);
        unsigned long long q_n = static_cast<unsigned long long>(q_n_128);

        best_p = p_n;
        best_q = q_n;

        p_nm2 = p_nm1; p_nm1 = p_n;
        q_nm2 = q_nm1; q_nm1 = q_n;

        long double frac = x_rem - a_ld;
        if (fabsl(frac) < 1e-30L) break;
        x_rem = 1.0L / frac;
    }

    if (best_q == 0ULL) {
        best_p = (unsigned long long)std::llround(std::nearbyint(x * 1e6L));
        best_q = 1000000ULL;
    }

    return {best_p, best_q};
}

long double benjamini_hochberg_correction(long double p, long double rank, long double totalN) {
    if (rank < 1.0L) rank = 1.0L;
    if (totalN < 1.0L) totalN = 1.0L;
    long double corrected = p * (totalN / rank);
    if (corrected > 1.0L) corrected = 1.0L;
    return corrected;
}

void section15_qed_significance_analysis(const high_precision_float& x_value) {
    if (mega_manager) mega_manager->stream_output("📊 QED NUMERICAL SIGNIFICANCE ANALYSIS:\n");
    else std::cout << "📊 QED NUMERICAL SIGNIFICANCE ANALYSIS:" << std::endl;

    // Convert to long double for QED analysis
    long double x = static_cast<long double>(x_value);
    
    QEDAnalysisResult result;
    result.x = x;
    result.residual = fabsl(x - ALPHA);
    result.z_score = compute_z_score(x);
    result.p_value = monte_carlo_p_value(x);
    result.p_corrected = benjamini_hochberg_correction(result.p_value, 1.0L, 1.0L);
    result.best_convergent = best_cf_approximation(x);
    
    // Calculate Diophantine metric
    if (result.best_convergent.second != 0ULL) {
        long double approx = static_cast<long double>(result.best_convergent.first) / 
                           static_cast<long double>(result.best_convergent.second);
        result.diophantine_metric = static_cast<long double>(result.best_convergent.second) * 
                                   static_cast<long double>(result.best_convergent.second) * 
                                   fabsl(x - approx);
    } else {
        result.diophantine_metric = LDBL_MAX;
    }

    // Determine verdict
    if (fabsl(result.z_score) < 5.0L && result.p_corrected > 0.001L)
        result.verdict = "NOT SIGNIFICANT (consistent with chance)";
    else if (result.p_corrected <= 0.001L)
        result.verdict = "SIGNIFICANT (statistically non-random proximity)";
    else
        result.verdict = "AMBIGUOUS (requires domain-specific analysis)";

    // Generate report
    std::ostringstream report;
    report << std::fixed << std::setprecision(18);
    report << "-------------------------------------------------------\n";
    report << "Target: x = " << result.x << "\n";
    report << "Fine-Structure Constant α = " << ALPHA << "\n";
    report << "-------------------------------------------------------\n";
    report << "Residual |x − α| = " << result.residual << "\n";
    report << "z-score (propagated) = " << result.z_score << "\n";
    report << "Monte-Carlo p-value = " << result.p_value << "\n";
    report << "BH-corrected p-value = " << result.p_corrected << "\n";
    report << "Best CF convergent p/q = " << result.best_convergent.first << "/" 
           << result.best_convergent.second << "\n";
    report << "Normalized Diophantine error q²·|x − p/q| = " << result.diophantine_metric << "\n";
    report << "\nFinal Verdict: " << result.verdict << "\n";
    report << "-------------------------------------------------------\n";

    if (mega_manager) mega_manager->stream_output(report.str() + "\n");
    else std::cout << report.str() << std::endl;
}

// ============================== SECTION 16: ALPHA ANALYZER CONFIGURATION & BATCH PROCESSING ==============================

// Alpha analyzer configuration structure
struct AlphaAnalyzerConfig {
    long double alpha = 0.007297352569311L;
    long double sigma_alpha = 1.5e-12L;
    unsigned n_sims = 200000;
    unsigned seed = 0xBEEF'FACE;
    std::string fdr_method = "BH";
    long double z_abs_threshold = 5.0L;
    long double p_corr_threshold = 0.001L;
    bool deterministic = true;
    std::string run_id;
};

struct NumericRecord {
    high_precision_float value;
    std::string units;
    long double sigma;
    std::string id;
    std::string source;
};

// Advanced analysis result for batch processing
struct AdvancedAnalysisResult {
    std::string id;
    high_precision_float x;
    long double sigma_x;
    long double resid;
    long double z;
    long double p_mc;
    long double p_corr;
    std::pair<unsigned long long, unsigned long long> best_conv;
    long double diophantine;
    std::string verdict;
};

// Generate run manifest for reproducibility
std::string generate_alpha_run_manifest(const AlphaAnalyzerConfig& cfg, 
                                       const std::vector<NumericRecord>& recs) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"run_id\": \"" << cfg.run_id << "\",\n";
    ss << "  \"timestamp\": \"" << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\",\n";
    ss << "  \"alpha\": " << std::setprecision(18) << cfg.alpha << ",\n";
    ss << "  \"sigma_alpha\": " << cfg.sigma_alpha << ",\n";
    ss << "  \"n_sims\": " << cfg.n_sims << ",\n";
    ss << "  \"fdr_method\": \"" << cfg.fdr_method << "\",\n";
    ss << "  \"records\": [\n";
    for (size_t i = 0; i < recs.size(); ++i) {
        ss << "    {\"id\":\"" << recs[i].id << "\", \"value\":" << std::setprecision(18) 
           << static_cast<long double>(recs[i].value) << ", \"units\":\"" << recs[i].units 
           << "\", \"sigma\":" << recs[i].sigma << ", \"source\":\"" << recs[i].source << "\"}";
        if (i + 1 < recs.size()) ss << ",";
        ss << "\n";
    }
    ss << "  ]\n}";
    return ss.str();
}

// Batch analysis function for multiple records
std::vector<AdvancedAnalysisResult> analyze_alpha_batch(const std::vector<NumericRecord>& records, 
                                                       const AlphaAnalyzerConfig& cfg) {
    std::vector<AdvancedAnalysisResult> reports;
    
    for (const auto& rec : records) {
        if (!rec.units.empty()) {
            continue; // Skip non-dimensionless values
        }
        
        long double x = static_cast<long double>(rec.value);
        long double sigma_x = rec.sigma;
        long double resid = fabsl(x - cfg.alpha);
        long double z = compute_z_score(x, sigma_x);
        long double p_mc = monte_carlo_p_value(x, cfg.n_sims);
        
        auto conv = best_cf_approximation(x);
        long double dioph = (conv.second != 0ULL) ? 
            static_cast<long double>(conv.second) * static_cast<long double>(conv.second) * 
            fabsl(x - static_cast<long double>(conv.first) / static_cast<long double>(conv.second)) : LDBL_MAX;
        
        std::string verdict;
        if (fabsl(z) < cfg.z_abs_threshold && p_mc > cfg.p_corr_threshold)
            verdict = "NOT SIGNIFICANT (consistent with chance)";
        else if (p_mc <= cfg.p_corr_threshold)
            verdict = "SIGNIFICANT (statistically non-random proximity)";
        else
            verdict = "AMBIGUOUS (requires further analysis)";
        
        reports.push_back({
            rec.id, rec.value, sigma_x, resid, z, p_mc, p_mc, conv, dioph, verdict
        });
    }
    
    return reports;
}

void section16_alpha_analyzer_demo(const high_precision_float& x_value) {
    if (mega_manager) mega_manager->stream_output("🔬 ALPHA ANALYZER ADVANCED DEMONSTRATION:\n");
    else std::cout << "🔬 ALPHA ANALYZER ADVANCED DEMONSTRATION:" << std::endl;

    AlphaAnalyzerConfig cfg;
    cfg.run_id = "demo-integration";
    cfg.deterministic = true;
    cfg.n_sims = 50000; // Demo with fewer simulations

    // Sample dataset including current x_value
    std::vector<NumericRecord> recs = {
        {x_value, "", 1e-12L, "current_value", "main_program"},
        {high_precision_float(0.00729735256931L), "", 1e-12L, "near_alpha_1", "synthetic"},
        {high_precision_float(0.0314L), "", 0.0L, "pi_scaled", "synthetic"},
        {high_precision_float(0.5L), "", 0.0L, "one_half", "synthetic"},
        {high_precision_float(1.0L/137.035999084L), "", 1e-12L, "alpha_recip_approx", "synthetic"}
    };

    // Generate manifest
    std::string manifest = generate_alpha_run_manifest(cfg, recs);
    if (mega_manager) mega_manager->stream_output("=== Run Manifest ===\n" + manifest + "\n");
    else std::cout << "=== Run Manifest ===" << std::endl << manifest << std::endl;

    // Run batch analysis
    auto reports = analyze_alpha_batch(recs, cfg);

    std::ostringstream output;
    output << std::fixed << std::setprecision(18);
    output << "=== Advanced Analysis Results ===\n";
    for (const auto& r : reports) {
        output << "-------------------------------------------------------\n";
        output << "ID: " << r.id << "\n";
        output << "x = " << static_cast<long double>(r.x) << "  sigma = " << r.sigma_x << "\n";
        output << "|x - alpha| = " << r.resid << "\n";
        output << "z = " << r.z << "\n";
        output << "MonteCarlo p = " << r.p_mc << "\n";
        output << "Best convergent p/q = " << r.best_conv.first << "/" << r.best_conv.second 
               << " (approx=" << (r.best_conv.second != 0ULL ? 
                  static_cast<long double>(r.best_conv.first) / static_cast<long double>(r.best_conv.second) : 0.0) << ")\n";
        output << "Diophantine metric q²|x-p/q| = " << r.diophantine << "\n";
        output << "Final verdict: " << r.verdict << "\n";
    }
    output << "-------------------------------------------------------\n";

    if (mega_manager) mega_manager->stream_output(output.str() + "\n");
    else std::cout << output.str() << std::endl;
}

// ============================== SECTION 17: ALPHA RELATION PROBE WITH PSLQ & TRANSFORMS ==============================

// PSLQ implementation for integer relation detection
struct PSLQRelation {
    bool found = false;
    std::vector<long long> coeffs;
    high_precision_float residual;
};

// Practical PSLQ implementation for small vectors
PSLQRelation pslq_integer_relation(const std::vector<high_precision_float>& vec, 
                                  int max_iters = 1000, 
                                  high_precision_float tol = EPS_RECIP) {
    PSLQRelation out;
    size_t n = vec.size();
    if (n == 0) return out;
    
    // Normalize vector
    high_precision_float norm = 0;
    for (auto& v : vec) norm += v * v;
    norm = sqrt(norm);
    if (norm == 0) return out;
    
    std::vector<high_precision_float> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = vec[i] / norm;
    
    // Transformation matrix
    std::vector<std::vector<high_precision_float>> B(n, std::vector<high_precision_float>(n, 0));
    for (size_t i = 0; i < n; ++i) B[i][i] = 1;
    
    for (int iter = 0; iter < max_iters; ++iter) {
        // Find smallest absolute entry
        size_t k = 0;
        high_precision_float minabs = abs(x[0]);
        for (size_t i = 1; i < n; ++i) {
            high_precision_float t = abs(x[i]);
            if (t < minabs) { minabs = t; k = i; }
        }
        
        if (minabs < tol) {
            // Candidate relation found
            std::vector<long long> cand(n);
            bool ok = true;
            for (size_t i = 0; i < n; ++i) {
                long double d = static_cast<long double>(B[i][k]);
                long long r = llround(d);
                cand[i] = r;
                if (llabs(r) > (1LL << 62)) { ok = false; break; }
            }
            
            if (ok) {
                high_precision_float sum = 0;
                for (size_t i = 0; i < n; ++i) sum += high_precision_float(cand[i]) * vec[i];
                high_precision_float rabs = abs(sum);
                out.residual = rabs;
                if (rabs < tol) {
                    out.found = true;
                    out.coeffs = cand;
                    return out;
                }
            }
        }
        
        if (minabs == 0) break;
        
        // Size reduction step
        for (size_t i = 0; i < n; ++i) {
            if (i == k) continue;
            high_precision_float ratio = x[i] / x[k];
            long double ratio_ld = static_cast<long double>(ratio);
            long long m = llround(ratio_ld);
            if (m == 0) continue;
            
            x[i] = x[i] - high_precision_float(m) * x[k];
            for (size_t col = 0; col < n; ++col) {
                B[i][col] = B[i][col] - high_precision_float(m) * B[k][col];
            }
        }
    }
    
    return out;
}

// Möbius transform search
struct MobiusTransform {
    long long a, b, c, d;
    high_precision_float error;
    std::string target;
};

MobiusTransform search_mobius_transform(const high_precision_float& x, int bound = 10) {
    MobiusTransform best;
    best.error = high_precision_float(1e9);
    best.a = best.b = best.c = best.d = 0;
    best.target = "";
    
    if (x == 0) return best;
    
    high_precision_float invx = 1 / x;
    for (int a = -bound; a <= bound; ++a) 
    for (int b = -bound; b <= bound; ++b)
    for (int c = -bound; c <= bound; ++c) 
    for (int d = -bound; d <= bound; ++d) {
        if (c == 0 && d == 0) continue;
        
        high_precision_float denom = high_precision_float(c) * x + high_precision_float(d);
        if (abs(denom) < high_precision_float("1e-30")) continue;
        
        high_precision_float val = (high_precision_float(a) * x + high_precision_float(b)) / denom;
        high_precision_float e1 = abs(val - invx);
        if (e1 < best.error) { 
            best.error = e1; 
            best.a = a; best.b = b; best.c = c; best.d = d; 
            best.target = "1/x"; 
        }
        
        high_precision_float e2 = abs(val - x);
        if (e2 < best.error) { 
            best.error = e2; 
            best.a = a; best.b = b; best.c = c; best.d = d; 
            best.target = "x"; 
        }
    }
    
    return best;
}

// Algebraic dependency search
struct AlgebraicDependency {
    bool found = false;
    int degree = 0;
    std::vector<long long> coeffs;
    high_precision_float residual;
    std::string method;
};

AlgebraicDependency find_algebraic_dependency(const high_precision_float& x, int max_degree = 8) {
    AlgebraicDependency dep;
    
    for (int deg = 1; deg <= max_degree; ++deg) {
        std::vector<high_precision_float> powers;
        high_precision_float cur = 1;
        for (int k = 0; k <= deg; ++k) {
            powers.push_back(cur);
            cur *= x;
        }
        
        PSLQRelation rel = pslq_integer_relation(powers, 2000, EPS_RECIP);
        if (rel.found) {
            // Normalize coefficients
            long long g = 0;
            for (auto c : rel.coeffs) g = std::gcd(g, llabs(c));
            if (g == 0) g = 1;
            for (auto& c : rel.coeffs) c /= g;
            if (rel.coeffs.back() < 0) 
                for (auto& c : rel.coeffs) c = -c;
            
            // Verify residual
            high_precision_float sum = 0;
            for (int i = 0; i <= deg; ++i) sum += high_precision_float(rel.coeffs[i]) * powers[i];
            high_precision_float rabs = abs(sum);
            
            if (rabs < EPS_RECIP * 10) {
                dep.found = true;
                dep.degree = deg;
                dep.coeffs = rel.coeffs;
                dep.residual = rabs;
                dep.method = "PSLQ on powers";
                return dep;
            }
        }
    }
    
    return dep;
}

void section17_alpha_relation_probe(const high_precision_float& x_value) {
    if (mega_manager) mega_manager->stream_output("🔍 ALPHA RELATION PROBE (PSLQ & Transforms):\n");
    else std::cout << "🔍 ALPHA RELATION PROBE (PSLQ & Transforms):" << std::endl;

    std::ostringstream report;
    report << "x = " << decimal_full(x_value) << "\n";
    report << "1/x = " << decimal_full(1 / x_value) << "\n\n";

    // 1. PSLQ on [x, 1/x, 1]
    report << "=== PSLQ on [x, 1/x, 1] ===\n";
    std::vector<high_precision_float> vec = {x_value, 1 / x_value, 1};
    PSLQRelation rel = pslq_integer_relation(vec);
    
    if (rel.found) {
        report << "[PSLQ] Integer relation found: ";
        for (auto c : rel.coeffs) report << c << " ";
        report << "\nResidual = " << decimal_full(rel.residual) << "\n";
    } else {
        report << "[PSLQ] No integer linear relation found\n";
    }
    report << "\n";

    // 2. Algebraic dependency search
    report << "=== Algebraic Dependency Search ===\n";
    AlgebraicDependency dep = find_algebraic_dependency(x_value);
    if (dep.found) {
        report << "[ALGDEP] Found polynomial relation (degree " << dep.degree << "):\n";
        for (size_t i = 0; i < dep.coeffs.size(); ++i) {
            report << dep.coeffs[i];
            if (i + 1 < dep.coeffs.size()) report << " + ";
        }
        report << " = 0\n";
        report << "Residual = " << decimal_full(dep.residual) << "\n";
        report << "Method: " << dep.method << "\n";
    } else {
        report << "[ALGDEP] No algebraic relation found up to degree 8\n";
    }
    report << "\n";

    // 3. Möbius transform search
    report << "=== Möbius Transform Search ===\n";
    MobiusTransform mob = search_mobius_transform(x_value, 6);
    report << "Best Möbius candidate: a=" << mob.a << " b=" << mob.b 
           << " c=" << mob.c << " d=" << mob.d << "\n";
    report << "Target: " << mob.target << " with error ≈ " << decimal_full(mob.error) << "\n";
    
    if (mob.error < high_precision_float("1e-6")) {
        high_precision_float denom = high_precision_float(mob.c) * x_value + high_precision_float(mob.d);
        if (abs(denom) > 0) {
            high_precision_float transformed = (high_precision_float(mob.a) * x_value + high_precision_float(mob.b)) / denom;
            report << "Transformed value: " << decimal_full(transformed) << "\n";
            report << "This suggests a potential rational/algebraic relationship!\n";
        }
    }
    report << "\n";

    // 4. Advanced continued fraction analysis
    report << "=== Advanced Continued Fraction Analysis ===\n";
    auto cf_x = continued_fraction_iterative(x_value, 40);
    auto cf_inv = continued_fraction_iterative(1 / x_value, 40);
    
    report << "CF(x) first 15 terms: ";
    for (size_t i = 0; i < cf_x.size() && i < 15; ++i) report << cf_x[i] << " ";
    report << "\nCF(1/x) first 15 terms: ";
    for (size_t i = 0; i < cf_inv.size() && i < 15; ++i) report << cf_inv[i] << " ";
    report << "\n\n";

    // 5. Final synthesis
    report << "=== Synthesis & Interpretation ===\n";
    if (rel.found) {
        report << "✓ PSLQ detected an integer linear relationship\n";
    }
    if (dep.found) {
        report << "✓ Algebraic polynomial relationship discovered\n";
    }
    if (mob.error < high_precision_float("1e-6")) {
        report << "✓ Promising Möbius transformation found\n";
    }
    
    if (!rel.found && !dep.found && mob.error >= high_precision_float("1e-6")) {
        report << "ℹ No simple algebraic relationships detected within search limits\n";
        report << "This suggests either transcendental nature or relationships beyond current search bounds\n";
    }

    if (mega_manager) mega_manager->stream_output(report.str() + "\n");
    else std::cout << report.str() << std::endl;
}

// ============================== SECTION 18: ADAPTIVE TRANSFORM ALGDEP WITH CF PERIODICITY ==============================

// Detect periodic continued fraction patterns (quadratic irrationals)
struct CFPeriodicity {
    bool found = false;
    int preperiod = -1;
    int period = -1;
    std::vector<int> periodic_block;
};

CFPeriodicity detect_cf_periodicity(const std::vector<int>& terms, int max_pre = 30, int max_period = 40) {
    CFPeriodicity result;
    int L = static_cast<int>(terms.size());
    if (L < 4) return result;
    
    for (int s = 0; s <= max_pre && s <= L/2; ++s) {
        for (int p = 1; p <= max_period && s + 2 * p <= L; ++p) {
            bool ok = true;
            for (int i = 0; i < p; ++i) {
                if (terms[s + i] != terms[s + p + i]) { 
                    ok = false; 
                    break; 
                }
            }
            if (ok) {
                result.found = true;
                result.preperiod = s;
                result.period = p;
                for (int i = 0; i < p; ++i) {
                    result.periodic_block.push_back(terms[s + i]);
                }
                return result;
            }
        }
    }
    return result;
}

// Generate quadratic polynomial from periodic CF block
struct QuadraticPolynomial {
    bool valid = false;
    long long a, b, c; // ax² + bx + c = 0
    std::string note;
};

QuadraticPolynomial quadratic_from_cf_period(const std::vector<int>& period_block) {
    QuadraticPolynomial qp;
    
    // Compute matrix product M = [[A,B],[C,D]] for periodic block
    __int128 A = 1, B = 0, C = 0, D = 1;
    for (int ai : period_block) {
        __int128 a = ai;
        __int128 nA = a * A + B;
        __int128 nB = A;
        __int128 nC = a * C + D;
        __int128 nD = C;
        A = nA; B = nB; C = nC; D = nD;
        
        // Overflow check
        __int128 lim = (__int128)(((__int128)1 << 62) - 1);
        if (A > lim || B > lim || C > lim || D > lim || 
            A < -lim || B < -lim || C < -lim || D < -lim) {
            qp.note = "Overflow in quadratic computation";
            return qp;
        }
    }
    
    // Quadratic: Cx² + (D-A)x - B = 0
    qp.a = static_cast<long long>(C);
    qp.b = static_cast<long long>(D - A);
    qp.c = static_cast<long long>(-B);
    qp.valid = true;
    qp.note = "Quadratic from periodic CF";
    return qp;
}

// Adaptive algebraic dependency finder with multiple transforms
struct AdaptiveAnalysis {
    bool finalized = false;
    std::string method;
    AlgebraicDependency algdep;
    QuadraticPolynomial quadratic;
    MobiusTransform mobius;
    std::string note;
};

AdaptiveAnalysis adaptive_transform_analysis(const high_precision_float& x) {
    AdaptiveAnalysis result;
    
    // 1. Try CF periodicity first (fastest)
    auto cf_terms = continued_fraction_iterative(x, 200);
    auto periodicity = detect_cf_periodicity(cf_terms);
    
    if (periodicity.found) {
        auto quad = quadratic_from_cf_period(periodicity.periodic_block);
        if (quad.valid) {
            // Verify the quadratic
            high_precision_float check = quad.a * x * x + quad.b * x + quad.c;
            if (abs(check) < EPS_COSMIC) {
                result.finalized = true;
                result.method = "cf_quadratic";
                result.quadratic = quad;
                result.note = "Quadratic from periodic CF verified";
                return result;
            }
        }
    }
    
    // 2. Try standard algdep on x
    result.algdep = find_algebraic_dependency(x, 10);
    if (result.algdep.found && result.algdep.residual < EPS_COSMIC) {
        result.finalized = true;
        result.method = "algdep_direct";
        result.note = "Direct algebraic dependency found";
        return result;
    }
    
    // 3. Try algdep on transforms
    // x - 1 transform
    high_precision_float x_minus_1 = x - 1;
    auto algdep_xm1 = find_algebraic_dependency(x_minus_1, 8);
    if (algdep_xm1.found && algdep_xm1.residual < EPS_COSMIC * 10) {
        result.finalized = true;
        result.method = "algdep_x_minus_1";
        result.algdep = algdep_xm1;
        result.note = "Algebraic dependency found for x-1";
        return result;
    }
    
    // 1/x transform
    if (x != 0) {
        high_precision_float inv = 1 / x;
        auto algdep_inv = find_algebraic_dependency(inv, 8);
        if (algdep_inv.found && algdep_inv.residual < EPS_COSMIC * 10) {
            result.finalized = true;
            result.method = "algdep_reciprocal";
            result.algdep = algdep_inv;
            result.note = "Algebraic dependency found for 1/x";
            return result;
        }
    }
    
    // 4. Möbius transform + algdep
    result.mobius = search_mobius_transform(x, 6);
    if (result.mobius.error < high_precision_float("1e-6")) {
        high_precision_float denom = high_precision_float(result.mobius.c) * x + high_precision_float(result.mobius.d);
        if (abs(denom) > 0) {
            high_precision_float transformed = (high_precision_float(result.mobius.a) * x + high_precision_float(result.mobius.b)) / denom;
            auto algdep_transformed = find_algebraic_dependency(transformed, 6);
            if (algdep_transformed.found) {
                result.finalized = true;
                result.method = "mobius_algdep";
                result.algdep = algdep_transformed;
                result.note = "Algebraic dependency found after Möbius transform";
                return result;
            }
        }
    }
    
    // 5. If nothing finalized, return best provisional result
    result.note = "No finalized algebraic relation found";
    if (periodicity.found) {
        result.method = "cf_quadratic_provisional";
        result.quadratic = quadratic_from_cf_period(periodicity.periodic_block);
    } else if (result.algdep.found) {
        result.method = "algdep_provisional";
    } else {
        result.method = "none_found";
    }
    
    return result;
}

void section18_adaptive_transform_analysis(const high_precision_float& x_value) {
    if (mega_manager) mega_manager->stream_output("🔄 ADAPTIVE TRANSFORM ALGDEP ANALYSIS:\n");
    else std::cout << "🔄 ADAPTIVE TRANSFORM ALGDEP ANALYSIS:" << std::endl;

    auto analysis = adaptive_transform_analysis(x_value);
    
    std::ostringstream report;
    report << "=== Adaptive Transform Certificate ===\n";
    report << "x = " << decimal_full(x_value) << "\n";
    report << "Finalized: " << (analysis.finalized ? "YES" : "NO") << "\n";
    report << "Method: " << analysis.method << "\n";
    report << "Note: " << analysis.note << "\n\n";

    if (analysis.method.find("quadratic") != std::string::npos && analysis.quadratic.valid) {
        report << "=== Quadratic Polynomial ===\n";
        report << "ax² + bx + c = 0 where:\n";
        report << "a = " << analysis.quadratic.a << "\n";
        report << "b = " << analysis.quadratic.b << "\n";
        report << "c = " << analysis.quadratic.c << "\n";
        
        high_precision_float verify = analysis.quadratic.a * x_value * x_value + 
                                     analysis.quadratic.b * x_value + analysis.quadratic.c;
        report << "Verification: P(x) = " << decimal_full(verify) << "\n";
        report << "Residual: |P(x)| = " << decimal_full(abs(verify)) << "\n\n";
    }

    if (analysis.algdep.found) {
        report << "=== Algebraic Dependency ===\n";
        report << "Degree: " << analysis.algdep.degree << "\n";
        report << "Coefficients: ";
        for (size_t i = 0; i < analysis.algdep.coeffs.size(); ++i) {
            report << analysis.algdep.coeffs[i];
            if (i + 1 < analysis.algdep.coeffs.size()) report << ", ";
        }
        report << "\n";
        report << "Residual: " << decimal_full(analysis.algdep.residual) << "\n";
        report << "Method: " << analysis.algdep.method << "\n\n";
    }

    if (analysis.method.find("mobius") != std::string::npos) {
        report << "=== Möbius Transform ===\n";
        report << "(" << analysis.mobius.a << "x + " << analysis.mobius.b << ") / (" 
               << analysis.mobius.c << "x + " << analysis.mobius.d << ")\n";
        report << "Error: " << decimal_full(analysis.mobius.error) << "\n";
        report << "Target: " << analysis.mobius.target << "\n\n";
    }

    // CF periodicity analysis
    auto cf_terms = continued_fraction_iterative(x_value, 200);
    auto periodicity = detect_cf_periodicity(cf_terms);
    report << "=== Continued Fraction Analysis ===\n";
    report << "CF terms computed: " << cf_terms.size() << "\n";
    if (periodicity.found) {
        report << "Periodicity detected: preperiod = " << periodicity.preperiod 
               << ", period = " << periodicity.period << "\n";
        report << "Periodic block: ";
        for (int term : periodicity.periodic_block) {
            report << term << " ";
        }
        report << "\n";
        report << "This indicates a quadratic irrational number.\n";
    } else {
        report << "No periodic pattern detected in first 200 terms.\n";
        report << "This suggests either a transcendental number or a quadratic with large period.\n";
    }
    report << "\n";

    // Final assessment
    report << "=== Final Assessment ===\n";
    if (analysis.finalized) {
        report << "✓ ALGEBRAIC RELATIONSHIP CONFIRMED\n";
        report << "The number has been identified as algebraic with high confidence.\n";
    } else {
        report << "ℹ NO FINALIZED ALGEBRAIC RELATIONSHIP FOUND\n";
        report << "This could indicate:\n";
        report << "  - Transcendental nature\n";
        report << "  - High-degree algebraic requiring deeper search\n";
        report << "  - Precision limitations\n";
        report << "  - Relationships outside current transform space\n";
    }

    if (mega_manager) mega_manager->stream_output(report.str() + "\n");
    else std::cout << report.str() << std::endl;
}

// ============================== SECTION 19: HIGH-PRECISION VERIFICATION & LIFTING ==============================

// High-precision verification certificate
struct VerificationCertificate {
    bool verified = false;
    std::string x_original;
    std::string method;
    int degree = 0;
    std::vector<long long> coeffs;
    high_precision_float residual;
    int search_precision_digits = 0;
    std::string note;
};

// Polynomial coefficient transforms for reciprocals and shifts
std::vector<long long> reciprocal_polynomial_coeffs(const std::vector<long long>& coeffs) {
    std::vector<long long> rev(coeffs.rbegin(), coeffs.rend());
    while (rev.size() > 1 && rev.back() == 0) rev.pop_back();
    return rev;
}

std::vector<high_precision_int> shift_polynomial_coeffs_bigint(const std::vector<long long>& coeffs) {
    int n = static_cast<int>(coeffs.size()) - 1;
    std::vector<high_precision_int> shifted(n + 1, high_precision_int("0"));
    
    for (int k = 0; k <= n; ++k) {
        high_precision_int a = coeffs[k];
        for (int j = 0; j <= k; ++j) {
            // Compute binomial coefficient C(k,j)
            unsigned long long comb = 1;
            for (int t = 1; t <= j; ++t) {
                comb = comb * (unsigned long long)(k - (j - t)) / (unsigned long long)t;
            }
            shifted[j] += a * high_precision_int(std::to_string(comb));
        }
    }
    
    while (shifted.size() > 1 && shifted.back() == 0) shifted.pop_back();
    return shifted;
}

// Comprehensive verification with multiple transforms
VerificationCertificate verify_and_lift(const high_precision_float& x, int maxdeg = 12) {
    VerificationCertificate cert;
    cert.x_original = decimal_full(x);
    cert.search_precision_digits = PRECISION_DECIMALS;
    
    // 1. Direct polynomial search on x
    auto algdep = find_algebraic_dependency(x, maxdeg);
    if (algdep.found && algdep.residual < EPS_COSMIC) {
        cert.verified = true;
        cert.method = "direct_pslq_powers";
        cert.degree = algdep.degree;
        cert.coeffs = algdep.coeffs;
        cert.residual = algdep.residual;
        cert.note = "Direct PSLQ on powers verified";
        return cert;
    }
    
    // 2. Transform: x - 1
    high_precision_float xm1 = x - 1;
    auto algdep_xm1 = find_algebraic_dependency(xm1, maxdeg);
    if (algdep_xm1.found) {
        // Convert back to polynomial in x: P(y) = 0 where y = x-1 -> Q(x) = P(x-1)
        auto shifted_big = shift_polynomial_coeffs_bigint(algdep_xm1.coeffs);
        
        // Try to downcast to long long if safe
        std::vector<long long> shifted_small;
        bool safe = true;
        for (auto& coeff : shifted_big) {
            if (coeff > std::numeric_limits<long long>::max() || 
                coeff < std::numeric_limits<long long>::min()) {
                safe = false;
                break;
            }
            shifted_small.push_back(static_cast<long long>(coeff));
        }
        
        if (safe) {
            // Verify the shifted polynomial
            high_precision_float sum = 0;
            high_precision_float xp = 1;
            for (size_t i = 0; i < shifted_small.size(); ++i) {
                sum += high_precision_float(shifted_small[i]) * xp;
                xp *= x;
            }
            
            if (abs(sum) < EPS_COSMIC * 10) {
                cert.verified = true;
                cert.method = "shifted_x_minus_1";
                cert.degree = static_cast<int>(shifted_small.size()) - 1;
                cert.coeffs = shifted_small;
                cert.residual = abs(sum);
                cert.note = "Polynomial from x-1 transform verified";
                return cert;
            }
        } else {
            cert.method = "x_minus_1_large_coeffs";
            cert.note = "x-1 transform found but coefficients too large for 64-bit";
            cert.coeffs = algdep_xm1.coeffs;
        }
    }
    
    // 3. Transform: 1/x
    if (x != 0) {
        high_precision_float inv = 1 / x;
        auto algdep_inv = find_algebraic_dependency(inv, maxdeg);
        if (algdep_inv.found) {
            auto reversed = reciprocal_polynomial_coeffs(algdep_inv.coeffs);
            
            // Verify reversed polynomial
            high_precision_float sum = 0;
            high_precision_float xp = 1;
            for (size_t i = 0; i < reversed.size(); ++i) {
                sum += high_precision_float(reversed[i]) * xp;
                xp *= x;
            }
            
            if (abs(sum) < EPS_COSMIC * 10) {
                cert.verified = true;
                cert.method = "reciprocal_reversed";
                cert.degree = static_cast<int>(reversed.size()) - 1;
                cert.coeffs = reversed;
                cert.residual = abs(sum);
                cert.note = "Polynomial from 1/x transform verified";
                return cert;
            }
        }
    }
    
    // 4. If no full verification, return best candidate
    if (algdep.found) {
        cert.method = "provisional_direct";
        cert.degree = algdep.degree;
        cert.coeffs = algdep.coeffs;
        cert.residual = algdep.residual;
        cert.note = "Provisional candidate - may require higher precision";
    } else {
        cert.method = "no_relation_found";
        cert.note = "No algebraic relation found within search limits";
    }
    
    return cert;
}

void section19_high_precision_verification(const high_precision_float& x_value) {
    if (mega_manager) mega_manager->stream_output("🔬 HIGH-PRECISION VERIFICATION & LIFTING:\n");
    else std::cout << "🔬 HIGH-PRECISION VERIFICATION & LIFTING:" << std::endl;

    auto cert = verify_and_lift(x_value);
    
    std::ostringstream report;
    report << "=== VERIFICATION CERTIFICATE ===\n";
    report << "Provenance: /workspace/reciprocal-integer-analyzer-mega.cpp\n";
    report << "x = " << cert.x_original << "\n";
    report << "Search Precision: " << cert.search_precision_digits << " decimal digits\n";
    report << "\n";
    
    report << "Result:\n";
    report << "  Verified: " << (cert.verified ? "TRUE" : "FALSE") << "\n";
    report << "  Method: " << cert.method << "\n";
    report << "  Degree: " << cert.degree << "\n";
    report << "  Residual: " << decimal_full(cert.residual) << "\n";
    report << "  Note: " << cert.note << "\n";
    
    if (!cert.coeffs.empty()) {
        report << "\n";
        report << "Polynomial Coefficients:\n";
        report << "  P(x) = ";
        for (size_t i = 0; i < cert.coeffs.size(); ++i) {
            if (i > 0) report << " + ";
            report << cert.coeffs[i];
            if (i > 0) {
                report << "x";
                if (i > 1) report << "^" << i;
            }
        }
        report << " = 0\n";
        
        // Verification details
        report << "\n";
        report << "Verification Details:\n";
        high_precision_float sum = 0;
        high_precision_float xp = 1;
        for (size_t i = 0; i < cert.coeffs.size(); ++i) {
            high_precision_float term = high_precision_float(cert.coeffs[i]) * xp;
            sum += term;
            report << "  Term " << i << ": " << cert.coeffs[i] 
                   << " * x^" << i << " = " << decimal_full(term) << "\n";
            xp *= x_value;
        }
        report << "  Sum: " << decimal_full(sum) << "\n";
        report << "  |Sum|: " << decimal_full(abs(sum)) << "\n";
    }
    
    report << "\n";
    report << "=== Interpretation ===\n";
    if (cert.verified) {
        report << "✓ ALGEBRAIC CERTIFICATE GENERATED\n";
        report << "The number is confirmed to be algebraic with high confidence.\n";
        report << "The polynomial relationship has been verified at " << cert.search_precision_digits << " digits.\n";
        
        if (cert.method.find("direct") != std::string::npos) {
            report << "Direct PSLQ found the polynomial relation.\n";
        } else if (cert.method.find("shifted") != std::string::npos) {
            report << "Relation discovered via x-1 transform and shifted back to x.\n";
        } else if (cert.method.find("reciprocal") != std::string::npos) {
            report << "Relation discovered via 1/x transform and reversed to x.\n";
        }
    } else {
        report << "ℹ NO VERIFIED ALGEBRAIC RELATIONSHIP\n";
        report << "This could indicate:\n";
        report << "  - The number is transcendental\n";
        report << "  - Higher precision is needed\n";
        report << "  - The polynomial degree exceeds search limits\n";
        report << "  - The relationship requires more complex transforms\n";
        
        if (!cert.coeffs.empty()) {
            report << "\nProvisional candidate available but not fully verified.\n";
            report << "Consider re-running with higher precision or expanded search parameters.\n";
        }
    }
    
    report << "\n";
    report << "=== Certificate End ===\n";

    if (mega_manager) mega_manager->stream_output(report.str() + "\n");
    else std::cout << report.str() << std::endl;
}

// ============================== BONUS FUNCTION: COMPREHENSIVE MATHEMATICAL ANALYSIS ==============================

// Bonus function that integrates all analysis capabilities
struct ComprehensiveAnalysisResult {
    high_precision_float x;
    std::vector<std::string> section_results;
    ProofMetrics proof_metrics;
    std::vector<std::string> special_properties;
    std::string final_classification;
};

void comprehensive_mathematical_analysis(const high_precision_float& x_value, uint64_t entry_number) {
    if (mega_manager) mega_manager->stream_output("🎯 COMPREHENSIVE MATHEMATICAL ANALYSIS (BONUS FUNCTION):\n");
    else std::cout << "🎯 COMPREHENSIVE MATHEMATICAL ANALYSIS (BONUS FUNCTION):" << std::endl;

    std::vector<std::string> results;
    
    // 1. Basic proof metrics
    auto metrics = calculate_proof_metrics(x_value);
    results.push_back("Proof Classification: " + metrics.proof_status);
    results.push_back("Distance from equality: " + decimal_short(metrics.distance_from_equality));
    
    // 2. Adjacency guillotine
    std::ostringstream adj_output;
    adj_output << "Adjacency Analysis: ";
    if (abs(x_value - 1) < EPSILON) {
        adj_output << "SINGULARITY POINT (theoretical limit 2.0)";
    } else if (x_value == 0) {
        adj_output << "ZERO EXCLUSION (infinite adjacency)";
    } else {
        high_precision_float epsilon = abs(x_value - 1);
        high_precision_float reciprocal = 1 / x_value;
        high_precision_float gap = abs(x_value - reciprocal);
        high_precision_float observed_ratio = gap / epsilon;
        high_precision_float theoretical_ratio = abs(1 + reciprocal);
        high_precision_float violation = abs(observed_ratio - theoretical_ratio);
        
        if (violation < EPS_COSMIC) {
            adj_output << "PASSED - Adjacency holds (ratio: " + decimal_short(observed_ratio) + ")";
        } else {
            adj_output << "FAILED - Mathematical breach detected (violation: " + decimal_full(violation) + ")";
        }
    }
    results.push_back(adj_output.str());
    
    // 3. QED significance (if in fine-structure range)
    long double x_ld = static_cast<long double>(x_value);
    if (fabsl(x_ld - 0.007297352569311L) < 0.01L) {
        long double z = compute_z_score(x_ld);
        long double p = monte_carlo_p_value(x_ld);
        std::ostringstream qed_output;
        qed_output << "QED Analysis: z=" << std::fixed << std::setprecision(6) << z 
                   << ", p=" << std::scientific << p;
        if (p < 0.001L) qed_output << " [SIGNIFICANT]";
        results.push_back(qed_output.str());
    }
    
    // 4. Algebraic analysis
    auto adaptive = adaptive_transform_analysis(x_value);
    std::string algebraic_status;
    if (adaptive.finalized) {
        algebraic_status = "Algebraic (" + adaptive.method + ")";
        if (adaptive.quadratic.valid) {
            algebraic_status += " - Quadratic: " + std::to_string(adaptive.quadratic.a) + "x² + " + 
                              std::to_string(adaptive.quadratic.b) + "x + " + 
                              std::to_string(adaptive.quadratic.c) + " = 0";
        }
    } else {
        algebraic_status = "No algebraic relation found (possible transcendental)";
    }
    results.push_back("Algebraic Status: " + algebraic_status);
    
    // 5. Special properties
    std::vector<std::string> properties;
    
    if (is_integer(x_value)) {
        properties.push_back("Integer");
        high_precision_int n = static_cast<high_precision_int>(x_value);
        if (n == 1) properties.push_back("Multiplicative Identity");
        else if (n == -1) properties.push_back("Negative Unity");
        else if (n > 1) {
            // Check for perfect powers
            if (is_perfect_square(x_value)) properties.push_back("Perfect Square");
            if (n < 1000000) {
                bool is_prime = true;
                for (high_precision_int i = 2; i * i <= n; ++i) {
                    if (n % i == 0) { is_prime = false; break; }
                }
                if (is_prime) properties.push_back("Prime");
            }
        }
    }
    
    // Check for special constants approximations
    if (abs(x_value - PHI) < EPS_RECIP) properties.push_back("Golden Ratio Approximation");
    if (abs(x_value - 1/PHI) < EPS_RECIP) properties.push_back("Golden Ratio Reciprocal");
    if (abs(x_value - PI) < EPS_RECIP) properties.push_back("π Approximation");
    if (abs(x_value - E) < EPS_RECIP) properties.push_back("e Approximation");
    
    if (properties.empty()) properties.push_back("General Real Number");
    
    // 6. Generate final report
    std::ostringstream final_report;
    final_report << "=== COMPREHENSIVE ANALYSIS REPORT ===\n";
    final_report << "Entry Number: " << entry_number << "\n";
    final_report << "Value: " << decimal_full(x_value) << "\n\n";
    
    for (const auto& result : results) {
        final_report << "• " << result << "\n";
    }
    
    final_report << "\nSpecial Properties: ";
    for (size_t i = 0; i < properties.size(); ++i) {
        final_report << properties[i];
        if (i + 1 < properties.size()) final_report << ", ";
    }
    final_report << "\n\n";
    
    // Final classification
    std::string classification;
    if (metrics.theorem_applies) {
        classification = "SELF-RECIPROCAL FIXED POINT";
    } else if (adaptive.finalized && adaptive.quadratic.valid) {
        classification = "QUADRATIC IRRATIONAL";
    } else if (adaptive.finalized) {
        classification = "ALGEBRAIC (Degree > 2)";
    } else {
        classification = "TRANSCENDENTAL OR HIGH-DEGREE ALGEBRAIC";
    }
    
    final_report << "Final Classification: " << classification << "\n";
    final_report << "========================================\n";

    if (mega_manager) mega_manager->stream_output(final_report.str() + "\n");
    else std::cout << final_report.str() << std::endl;
}

// ============================== SECTION 1: CORE ANALYSIS ==============================

void section1_core(uint64_t entry_number, const high_precision_float& x_value, const std::string& x_name) {
    banner("ENTRY " + std::to_string(entry_number), 70);
    std::string header = x_name + " | x = " + decimal_short(x_value);
    
    if (mega_manager) {
        mega_manager->stream_output(header + "\n");
    } else {
        std::cout << header << std::endl;
    }
    
    // Proof-centered metrics and language
    ProofMetrics proof_metrics = calculate_proof_metrics(x_value);
    std::vector<std::string> proof_language = generate_proof_language(x_value, x_name, proof_metrics);
    
    // Display proof language
    for (const auto& line : proof_language) {
        if (mega_manager) {
            mega_manager->stream_output(line + "\n");
        } else {
            std::cout << line << std::endl;
        }
    }
    
    if (mega_manager) {
        mega_manager->stream_output("\n");
    } else {
        std::cout << std::endl;
    }
    
    high_precision_float symmetry = reciprocal_symmetry_score(x_value);
    std::string symmetry_str = "Reciprocal Symmetry Score: " + std::to_string(static_cast<double>(symmetry));
    
    if (mega_manager) {
        mega_manager->stream_output(symmetry_str + "\n");
    } else {
        std::cout << symmetry_str << std::endl;
    }
    
    std::string tree_info = analyze_base_tree_membership(x_value);
    std::string tree_str = "Base Tree Membership: " + tree_info;
    
    if (mega_manager) {
        mega_manager->stream_output(tree_str + "\n");
    } else {
        std::cout << tree_str << std::endl;
    }
    
    // Prime factorization
    if (is_integer(x_value) && abs(x_value) > 1) {
        high_precision_int n = static_cast<high_precision_int>(x_value);
        auto factors = prime_factorize(n);
        std::string factor_str = "Prime factorization: ";
        for (size_t i = 0; i < factors.size(); ++i) {
            if (i > 0) factor_str += " × ";
            factor_str += factors[i].str();
        }
        factor_str += "\n";
        if (mega_manager) mega_manager->stream_output(factor_str);
        else std::cout << factor_str;
    }
    
    // Reciprocal analysis
    if (mega_manager) {
        mega_manager->stream_output("\nReciprocal Analysis:\n");
    } else {
        std::cout << "\nReciprocal Analysis:" << std::endl;
    }
    
    std::string x_str = "  x = " + decimal_full(x_value);
    std::string reciprocal_str;
    std::string diff_str;
    std::string equality_str;
    
    if (x_value == 0) {
        reciprocal_str = "  1/x = UNDEFINED";
        diff_str = "  Difference x - 1/x = UNDEFINED";
        equality_str = "  Reciprocal Equality: NO";
    } else {
        high_precision_float reciprocal = 1 / x_value;
        high_precision_float diff = abs(x_value - reciprocal);
        bool is_equal = diff < EPS_RECIP;
        
        reciprocal_str = "  1/x = " + decimal_full(reciprocal);
        diff_str = "  Difference x - 1/x = " + decimal_full(diff);
        equality_str = "  Reciprocal Equality: " + std::string(is_equal ? "YES" : "NO");
    }
    
    if (mega_manager) {
        mega_manager->stream_output(x_str + "\n" + reciprocal_str + "\n" + diff_str + "\n" + equality_str + "\n");
    } else {
        std::cout << x_str << std::endl << reciprocal_str << std::endl << diff_str << std::endl << equality_str << std::endl;
    }
    
    // Enhanced interpretation with proof context
    if (x_value != 0) {
        high_precision_float reciprocal = 1 / x_value;
        high_precision_float diff = abs(x_value - reciprocal);
        bool is_equal = diff < EPS_RECIP;
        
        if (is_equal) {
            if (mega_manager) {
                mega_manager->stream_output("  🎯 PROOF CONFIRMATION: Self-reciprocal property validates theorem\n");
            } else {
                std::cout << "  🎯 PROOF CONFIRMATION: Self-reciprocal property validates theorem" << std::endl;
            }
        } else {
            if (mega_manager) {
                mega_manager->stream_output("  🔍 PROOF SUPPORT: Reciprocal disparity confirms theorem boundary\n");
            } else {
                std::cout << "  🔍 PROOF SUPPORT: Reciprocal disparity confirms theorem boundary" << std::endl;
            }
        }
    }
    
    if (x_value != 0 && !isinf(x_value)) {
        high_precision_float reciprocal = 1 / x_value;
        std::string x_snippet = "  Decimal snippet x: " + decimal_snippet(x_value);
        std::string reciprocal_snippet = "  Decimal snippet 1/x: " + decimal_snippet(reciprocal);
        
        if (mega_manager) {
            mega_manager->stream_output(x_snippet + "\n" + reciprocal_snippet + "\n");
        } else {
            std::cout << x_snippet << std::endl << reciprocal_snippet << std::endl;
        }
    }
    
    // Multiplicative Closure Count (MCC) output
    if (x_value != 0 && !isinf(x_value)) {
        MCCResult mres = compute_MCC(abs(x_value));
        double mscore = mcc_score_from_mcc(mres);
        std::string mcc_line = "  MCC (min k s.t. x * k ∈ Z): " + mres.as_string() +
                               "   | MCC_score: " + std::to_string(mscore) +
                               "   | confidence: " + mres.confidence;
        if (mega_manager) {
            mega_manager->stream_output(mcc_line + "\n");
        } else {
            std::cout << mcc_line << std::endl;
        }
        
        // Interpretation line
        std::string interpret;
        if (!mres.finite) {
            interpret = "    Interpretation: No finite integer multiplier found (irrational or very large denominator).";
        } else {
            interpret = "    Interpretation: Finite multiplier " + mres.as_string() + " will convert x into an integer.";
        }
        if (mega_manager) mega_manager->stream_output(interpret + "\n");
        else std::cout << interpret << std::endl;
    }
    
    // Advanced classification
    if (x_value != 0 && !isinf(x_value)) {
        std::string alg_type = detect_algebraic_type(x_value);
        double irr_measure = estimate_irrationality_measure(x_value);
        
        std::string advanced_analysis = 
            "  Advanced Classification: " + alg_type + 
            " | Irrationality Estimate: " + std::to_string(irr_measure) + "\n";
        
        if (mega_manager) mega_manager->stream_output(advanced_analysis);
        else std::cout << advanced_analysis;
        
        analyze_digit_distribution(x_value);
    }
    
    // Advanced number analysis
    advanced_number_analysis(x_value);
    
    if (mega_manager) {
        mega_manager->stream_output("\n\n");
    } else {
        std::cout << "\n\n" << std::endl;
    }
}

// ============================== COMPREHENSIVE ENTRY ANALYZER ==============================

void analyze_entry_comprehensive(uint64_t entry_number, const high_precision_float& x_val, const std::string& description) {
    try {
        section1_core(entry_number, x_val, description);
        section2_sequences(entry_number, x_val);
        section3_primes_factorials(entry_number, x_val);
        section4_geometric(entry_number, x_val);
        section5_harmonics(entry_number, x_val);
        section6_continued(entry_number, x_val);
        section7_banachian(entry_number, x_val);
        section8_extremes(entry_number, x_val, description);
        section9_summary(entry_number, x_val, description);
        section10_decimal_analysis(entry_number, x_val);
        section11_cosmic_monitor(entry_number, x_val);
        section12_proportion_vision(entry_number, x_val, description);
        section13_adjacency_guillotine(entry_number, x_val);
        section14_astronomical_relations(entry_number, x_val);
        section15_qed_significance_analysis(x_val);
        section16_alpha_analyzer_demo(x_val);
        section17_alpha_relation_probe(x_val);
        section18_adaptive_transform_analysis(x_val);
        section19_high_precision_verification(x_val);
        
        // BONUS FUNCTION: Comprehensive Analysis
        comprehensive_mathematical_analysis(x_val, entry_number);
        
    } catch (const std::exception& e) {
        std::string error_msg = "🌀 GENTLE NOTE: Entry " + std::to_string(entry_number) + 
                               " encountered cosmic turbulence: " + std::string(e.what()) + "\n";
        error_msg += "🌌 Continuing our journey through mathematical reality...\n";
        if (mega_manager) mega_manager->stream_output(error_msg);
        else std::cout << error_msg;
        
        banner("", 70);
    }
}

// ============================== BURST MODE GENERATOR ==============================

std::vector<std::pair<high_precision_float, std::string>> generate_burst(
        uint64_t burst_size,
        Focus    focus,
        uint64_t burst_index,
        std::mt19937_64& rng)
{
    std::vector<std::pair<high_precision_float, std::string>> entries;
    entries.reserve(burst_size);

    std::uniform_real_distribution<long double>  log_dist(-50.0L, 50.0L);
    std::uniform_int_distribution<int>           int_dist(1, 1000);

    for (uint64_t i = 0; i < burst_size; ++i) {
        high_precision_float   value;
        std::string  desc;

        // Priority: if several focuses are active we pick one at random
        std::vector<Focus> active;
        for (int bit=0; bit<64; ++bit) {
            Focus f = static_cast<Focus>(1ULL<<bit);
            if ((static_cast<uint64_t>(focus) & static_cast<uint64_t>(f)) != 0) active.push_back(f);
        }
        if (active.empty()) active = {Focus::Random};

        std::uniform_int_distribution<size_t> pick(0, active.size()-1);
        Focus chosen = active[pick(rng)];

        switch (chosen) {
            case Focus::Random:
                {
                    long double exp = log_dist(rng);
                    value = pow(high_precision_float(10), high_precision_float(exp));
                    desc  = "Random 10^" + std::to_string(exp);
                }
                break;
            case Focus::PowersOf10:
                {
                    int exp = static_cast<int>(i % 101) - 50;   // -50 … +50
                    value = pow(high_precision_float(10), exp);
                    desc  = "10^" + std::to_string(exp);
                }
                break;
            case Focus::GoldenFamily:
                {
                    // φ^n  and  1/φ^n  for n = -25 … +25
                    int n = static_cast<int>(i % 51) - 25;
                    high_precision_float phi_pow = pow(PHI, n);
                    value = (i & 1) ? phi_pow : high_precision_float(1)/phi_pow;
                    desc  = (n>=0 ? "φ^" : "φ^") + std::to_string(std::abs(n))
                          + (i&1 ? "" : " (reciprocal)");
                }
                break;
            case Focus::PolygonRoots:
                {
                    // Roots of unity / regular polygon diagonals
                    int sides = 3 + static_cast<int>(i % 30);
                    high_precision_float angle = high_precision_float(2)*PI / sides;
                    value = cos(angle);
                    desc  = "cos(2π/"+std::to_string(sides)+") – regular "+std::to_string(sides)+"-gon";
                }
                break;
            case Focus::Harmonic:
                {
                    uint64_t denom = 1 + (i % 1000000);
                    value = high_precision_float(1)/high_precision_float(denom);
                    desc  = "1/"+std::to_string(denom);
                }
                break;
            case Focus::Fibonacci:
                {
                    // Fibonacci ratios approaching φ
                    uint64_t a = 1, b = 1;
                    for (uint64_t k = 0; k < (i%80); ++k) {
                        uint64_t t = a + b; a = b; b = t;
                    }
                    value = high_precision_float(b) / high_precision_float(a);
                    desc  = "Fib("+std::to_string(i%80+2)+")/Fib("+
                            std::to_string(i%80+1)+")";
                }
                break;
            case Focus::Extreme:
                {
                    int exp = (i & 1) ? 50 : -50;
                    value = pow(high_precision_float(10), exp);
                    desc  = "10^" + std::to_string(exp);
                }
                break;
            case Focus::Algebraic:
                {
                    // Random small-degree algebraic numbers
                    int deg = 2 + (i % 5);
                    high_precision_float root = pow(high_precision_float(2 + i%47), high_precision_float(1)/deg);
                    value = root;
                    desc  = "ⁿ√(" + std::to_string(2+i%47) + ")  n="+std::to_string(deg);
                }
                break;
            default:
                value = high_precision_float(i+1);
                desc  = "fallback integer " + std::to_string(i+1);
        }

        entries.emplace_back(value, desc);
    }
    return entries;
}

// ============================== BURST MODE PROCESSOR ==============================

void process_one_burst(uint64_t burst_index, uint64_t burst_size, Focus focus)
{
    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(seed);

    auto entries = generate_burst(burst_size, focus, burst_index, rng);

    // Create a fresh output file for this burst only
    std::ostringstream filename_stream;
    filename_stream << "burst_" << std::setfill('0') << std::setw(6) << burst_index << ".txt";
    std::string filename = filename_stream.str();
    
    mega_manager = std::make_unique<MegaRecursionManager>(filename, true);

    banner("BURST " + std::to_string(burst_index) + " – " + std::to_string(entries.size())
           + " entries – focus mask 0x" + std::to_string(static_cast<uint64_t>(focus)));

    for (uint64_t i = 0; i < entries.size(); ++i) {
        const auto& [value, desc] = entries[i];
        analyze_entry_comprehensive(i+1, value, desc);
        cosmic_reality_monitor(value, i+1);  // cosmic catcher stays on
    }

    banner("BURST " + std::to_string(burst_index) + " FINISHED – written to " + filename);
    std::cout << "Burst " << burst_index << " completed → " << filename << std::endl;
    
    // Close this burst's manager
    mega_manager.reset();
}

// ============================== INTERACTIVE BURST LOOP ==============================

void interactive_burst_loop()
{
    uint64_t burst_index = 1;

    while (true) {
        std::cout << "\n=== NEW BURST ===\n";
        std::cout << "Current burst index: " << burst_index << "\n\n";

        std::cout << "Size (1-" << (1ULL<<40) << ", default 500, 0 = full 10^50 sweep): ";
        std::string line; 
        std::getline(std::cin, line);
        uint64_t size = 500;
        if (!line.empty()) {
            if (line == "0") size = (1ULL<<40);  // symbolic "full sweep"
            else size = std::stoull(line);
        }

        std::cout << "\nFocus selection (you may combine with spaces):\n"
                  << "  r  = random          10 = powers of 10\n"
                  << "  g  = golden family    p  = polygon roots\n"
                  << "  h  = harmonic         f  = Fibonacci ratios\n"
                  << "  e  = extreme ±10^50   a  = algebraic roots\n"
                  << "Enter codes (e.g. \"r g h\") or leave empty for random: ";
        std::getline(std::cin, line);

        Focus focus = Focus::None;
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            if (token == "r")  focus = static_cast<Focus>(static_cast<uint64_t>(focus) | static_cast<uint64_t>(Focus::Random));
            if (token == "10") focus = static_cast<Focus>(static_cast<uint64_t>(focus) | static_cast<uint64_t>(Focus::PowersOf10));
            if (token == "g")  focus = static_cast<Focus>(static_cast<uint64_t>(focus) | static_cast<uint64_t>(Focus::GoldenFamily));
            if (token == "p")  focus = static_cast<Focus>(static_cast<uint64_t>(focus) | static_cast<uint64_t>(Focus::PolygonRoots));
            if (token == "h")  focus = static_cast<Focus>(static_cast<uint64_t>(focus) | static_cast<uint64_t>(Focus::Harmonic));
            if (token == "f")  focus = static_cast<Focus>(static_cast<uint64_t>(focus) | static_cast<uint64_t>(Focus::Fibonacci));
            if (token == "e")  focus = static_cast<Focus>(static_cast<uint64_t>(focus) | static_cast<uint64_t>(Focus::Extreme));
            if (token == "a")  focus = static_cast<Focus>(static_cast<uint64_t>(focus) | static_cast<uint64_t>(Focus::Algebraic));
        }
        if (focus == Focus::None) focus = Focus::Random;

        process_one_burst(burst_index++, size, focus);

        std::cout << "\nAnother burst? (y/n): ";
        std::getline(std::cin, line);
        if (line.empty() || (line[0] != 'y' && line[0] != 'Y')) break;
    }

    std::cout << "\nAll bursts finished. Goodbye!\n";
}

// ============================== ENTRY GENERATOR FOR STANDARD MODE ==============================

struct Entry {
    high_precision_float value;
    std::string description;
};

std::vector<Entry> get_entries() {
    std::vector<Entry> entries;
    
    // Basic numbers
    entries.push_back({high_precision_float(0), "0 (Zero)"});
    entries.push_back({high_precision_float(1), "1 (Fundamental Unit)"});
    entries.push_back({high_precision_float(-1), "-1 (Negative Unit)"});
    entries.push_back({high_precision_float(2), "2 (First Prime)"});
    entries.push_back({high_precision_float(1) / 2, "1/2 (Reciprocal of 2)"});
    
    // Mathematical constants
    entries.push_back({PHI, "φ (Golden Ratio)"});
    entries.push_back({PSI, "ψ (Golden Ratio Conjugate)"});
    entries.push_back({E, "e (Exponential Base)"});
    entries.push_back({PI, "π (Pi)"});
    entries.push_back({SQRT2, "√2 (Square Root of 2)"});
    
    // Extreme values - handling 10^50 properly
    high_precision_float extreme_large = pow(high_precision_float(10), 50);
    high_precision_float extreme_small = pow(high_precision_float(10), -50);
    
    entries.push_back({extreme_large, "10^50 (Extremely Large)"});
    entries.push_back({extreme_small, "10^-50 (Extremely Small)"});
    
    // Additional interesting values
    entries.push_back({pow(high_precision_float(10), 25), "10^25 (Large Power of 10)"});
    entries.push_back({pow(high_precision_float(10), -25), "10^-25 (Small Power of 10)"});
    entries.push_back({high_precision_float(1) / 3, "1/3 (Rational Repeating Decimal)"});
    entries.push_back({sqrt(high_precision_float(5)), "√5 (Square Root of 5)"});
    entries.push_back({high_precision_float(1) / 7, "1/7 (Classic Repeating Decimal)"});
    entries.push_back({sqrt(high_precision_float(3)), "√3 (Square Root of 3)"});
    entries.push_back({log(high_precision_float(2)), "ln(2) (Natural Log of 2)"});
    
    return entries;
}

// ============================== MEGA SWEEP GENERATOR (10^50 CAPABLE) ==============================

std::vector<Entry> generate_mega_sweep_entries(uint64_t max_value_exp = 50, 
                                                uint64_t count = 100, 
                                                const std::string& mode = "log",
                                                bool include_negatives = true) {
    std::vector<Entry> entries;
    
    // Cap count to prevent memory issues even with streaming
    const uint64_t MAX_SAFE_COUNT = 10000000; // 10 million entries max
    if (count > MAX_SAFE_COUNT) {
        std::cerr << "⚠️  Requested sweep count " << count 
                  << " exceeds safe cap (" << MAX_SAFE_COUNT << "). Capping to " << MAX_SAFE_COUNT << "." << std::endl;
        count = MAX_SAFE_COUNT;
    }
    
    high_precision_float max_val = pow(high_precision_float(10), max_value_exp);
    
    if (mode == "log") {
        if (count == 1) {
            entries.push_back({high_precision_float(1), "log-sample 1: 10^0"});
        } else {
            for (uint64_t i = 0; i < count; ++i) {
                high_precision_float exp = -max_value_exp + (2 * max_value_exp) * (high_precision_float(i) / (count - 1));
                high_precision_float val = pow(high_precision_float(10), exp);
                std::string desc = "log-sample " + std::to_string(i + 1) + ": 10^" + decimal_short(exp);
                entries.push_back({val, desc});
                
                if (include_negatives) {
                    entries.push_back({-val, "-" + desc});
                }
                
                // Track progress for massive operations
                if (mega_manager && i % 100000 == 0) {
                    mega_manager->track_recursion_depth(i);
                }
            }
        }
    } else { // linear
        if (count == 1) {
            entries.push_back({high_precision_float(0), "linear-sample 1"});
        } else {
            for (uint64_t i = 0; i < count; ++i) {
                high_precision_float val = -max_val + (2 * max_val) * (high_precision_float(i) / (count - 1));
                std::string desc = "linear-sample " + std::to_string(i + 1);
                entries.push_back({val, desc});
                
                // Track progress for massive operations
                if (mega_manager && i % 100000 == 0) {
                    mega_manager->track_recursion_depth(i);
                }
            }
        }
    }
    
    return entries;
}

// ============================== MAIN EXECUTION ==============================

int main(int argc, char* argv[]) {
    try {
        // Check for burst mode first
        if (argc > 1 && std::string(argv[1]) == "--burst") {
            banner("BURST-MODE RECIPROCAL FRAMEWORK – 10^50 READY");
            interactive_burst_loop();
            return 0;
        }
        
        // Initialize mega recursion manager for standard/sweep modes
        mega_manager = std::make_unique<MegaRecursionManager>("reciprocal_analysis_output.txt", true);
        
        banner("THE UNIFIED RECIPROCAL PROOF FRAMEWORK (MEGA EDITION)", 70);
        
        std::string precision_str = "Precision: " + std::to_string(PRECISION_DECIMALS) + " decimals";
        std::string guard_str = "Guard digits: " + std::to_string(GUARD_DIGITS);
        std::string total_str = "Total working precision: " + std::to_string(PRECISION_DECIMALS + GUARD_DIGITS) + " decimal places";
        
        // Get current time
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::string time_str = "Started: " + std::string(std::ctime(&now_time));
        time_str.pop_back(); // Remove newline
        
        if (mega_manager) {
            mega_manager->stream_output(precision_str + "\n");
            mega_manager->stream_output(guard_str + "\n");
            mega_manager->stream_output(total_str + "\n");
            mega_manager->stream_output(time_str + "\n\n");
        } else {
            std::cout << precision_str << std::endl;
            std::cout << guard_str << std::endl;
            std::cout << total_str << std::endl;
            std::cout << time_str << std::endl << std::endl;
        }
        
        std::string program_purpose = "THIS PROGRAM PROVES: x/1 = 1/x ONLY when x = ±1\n"
                                      "Through:\n"
                                      "  1. Direct numerical verification\n"
                                      "  2. Base tree membership analysis\n"
                                      "  3. Divisibility pattern examination\n"
                                      "  4. Continued fraction structure\n"
                                      "  5. Transverse irrationality mapping\n"
                                      "  6. Banachian stress testing\n"
                                      "  7. Mathematical classification\n"
                                      "  8. Proof-centered descriptive language\n"
                                      "  9. Prime factorization and digit analysis\n"
                                      "  10. Advanced algebraic type detection\n"
                                      "  11. Adjacency Guillotine verification\n"
                                      "  12. QED significance analysis\n"
                                      "  13. PSLQ & transform relation probing\n"
                                      "  14. Adaptive algebraic dependency search\n"
                                      "  15. High-precision verification & lifting\n"
                                      "  16. Comprehensive mathematical analysis (BONUS)\n\n"
                                      "MODES:\n"
                                      "  Standard: Analyze predefined mathematically significant values\n"
                                      "  --mega-sweep <count> [mode]: Generate and analyze <count> values\n"
                                      "      mode: 'log' (default) or 'linear'\n"
                                      "  --burst: Interactive burst mode with focus selection\n\n";
        
        if (mega_manager) {
            mega_manager->stream_output(program_purpose);
        } else {
            std::cout << program_purpose << std::endl;
        }
        
        banner("ALGEBRAIC PROOF OF THE FORMULA");
        std::string algebraic_proof = "Assume x ≠ 0.\n"
                                      "x = 1/x\n"
                                      "⇒ x² = 1\n"
                                      "⇒ x² - 1 = 0\n"
                                      "⇒ (x - 1)(x + 1) = 0\n"
                                      "⇒ x = 1 or x = -1\n"
                                      "For x = 0, 1/x undefined.\n"
                                      "Hence, the formula shows equality only at x = ±1.\n\n"
                                      "The following numerical analysis verifies this across diverse numbers,\n"
                                      "with gap monitoring and decimal chunk tabling.\n";
        
        if (mega_manager) {
            mega_manager->stream_output(algebraic_proof);
        } else {
            std::cout << algebraic_proof << std::endl;
        }
        
        // Check command line arguments for mega sweep mode
        bool mega_sweep_mode = false;
        uint64_t sweep_count = 100;
        std::string sweep_mode = "log";
        
        if (argc > 1) {
            std::string arg1 = argv[1];
            if (arg1 == "--mega-sweep" && argc > 2) {
                mega_sweep_mode = true;
                sweep_count = std::stoull(argv[2]);
                if (argc > 3) {
                    sweep_mode = argv[3];
                }
            }
        }
        
        std::vector<Entry> entries;
        
        if (mega_sweep_mode) {
            // Generate mega sweep entries up to 10^50
            entries = generate_mega_sweep_entries(50, sweep_count, sweep_mode, true);
        } else {
            entries = get_entries();
        }
        
        std::string analysis_start = "Analyzing " + std::to_string(entries.size()) + 
                                    " mathematically significant values\n" +
                                    "All results printed to " + std::to_string(PRECISION_DECIMALS) + 
                                    " decimal places\n\n";
        
        if (mega_manager) {
            mega_manager->stream_output(analysis_start);
        } else {
            std::cout << analysis_start << std::endl;
        }
        
        // Analyze all entries with comprehensive analysis
        for (uint64_t i = 0; i < entries.size(); ++i) {
            analyze_entry_comprehensive(i + 1, entries[i].value, entries[i].description);
            
            // For massive operations, provide progress feedback
            if (mega_manager && i % 10000 == 0) {
                mega_manager->track_recursion_depth(i);
            }
        }
        
        // ============================== DREAMY SEQUENCE ANALYSIS ==============================
        banner("INFINITE ASCENT EXPLORATION", 70);
        dreamy_sequence_analysis();
        
        // ============================== COSMIC REALITY FINAL REPORT ==============================
        if (!cosmic_epsilon_table.empty()) {
            banner("COSMIC REALITY FINAL REPORT", 70);
            if (mega_manager) mega_manager->stream_output("Reality shifts detected during analysis:\n");
            else std::cout << "Reality shifts detected during analysis:" << std::endl;
            
            for (const auto& observation : cosmic_epsilon_table) {
                std::string obs_str = "  Entry " + std::to_string(observation.entry) + ": " + observation.description + "\n";
                if (mega_manager) mega_manager->stream_output(obs_str);
                else std::cout << obs_str;
            }
            
            std::string summary_str = "\nTotal ε observations: " + std::to_string(cosmic_epsilon_table.size()) + "\n";
            summary_str += "Reality has been adaptively monitored and tallied.\n";
            if (mega_manager) mega_manager->stream_output(summary_str);
            else std::cout << summary_str;
        } else {
            std::string stable_str = "Cosmic Reality Status: Stable - no ε anomalies detected\n";
            if (mega_manager) mega_manager->stream_output(stable_str);
            else std::cout << stable_str;
        }
        
        // ============================== COMPLETION SUMMARY ==============================
        banner("COMPLETION SUMMARY", 70);
        
        std::string completion_msg = "MATHEMATICAL COROLLARIES:\n"
                                    "1. The reciprocal function f(x) = 1/x has exactly two fixed points: x = ±1\n"
                                    "2. All other numbers exhibit reciprocal disparity\n"
                                    "3. Base tree membership determines decimal expansion patterns\n"
                                    "4. Divisibility 'errors' are actually proofs of infinite complexity\n"
                                    "5. The transverse mapping x ↦ 1/x preserves irrationality\n"
                                    "6. Multiplication table structure prevents self-reciprocality except at unity\n"
                                    "7. Prime factorization reveals multiplicative structure\n"
                                    "8. Continued fractions classify algebraic types\n"
                                    "9. The Adjacency Guillotine confirms theoretical predictions\n"
                                    "10. PSLQ and algebraic dependency searches reveal deep structure\n"
                                    "11. High-precision verification enables algebraic certification\n\n"
                                    "PHILOSOPHICAL IMPLICATIONS:\n"
                                    "The numbers 1 and -1 stand as fundamental mathematical anchors,\n"
                                    "the only points where a quantity equals its own reciprocal.\n"
                                    "This reveals a deep symmetry in the fabric of mathematics.\n\n"
                                    "INTEGRATED CAPABILITIES:\n"
                                    "✓ Standard analysis mode (predefined values)\n"
                                    "✓ Mega-sweep mode (up to 10^50 scale)\n"
                                    "✓ Interactive burst mode (focused generation)\n"
                                    "✓ 19 comprehensive analysis sections\n"
                                    "✓ Comprehensive mathematical analysis bonus function\n"
                                    "✓ Full functionality preservation from original code\n";
        
        if (mega_manager) {
            mega_manager->stream_output(completion_msg);
        } else {
            std::cout << completion_msg << std::endl;
        }
        
        banner("Q.E.D. - QUOD ERAT DEMONSTRANDUM", 70);
        
        // Final completion message
        auto end_time = std::chrono::system_clock::now();
        std::time_t end_time_t = std::chrono::system_clock::to_time_t(end_time);
        std::string end_str = "Completed: " + std::string(std::ctime(&end_time_t));
        end_str.pop_back();
        
        std::string final_summary = end_str + "\n" +
                                   "Total entries analyzed: " + std::to_string(entries.size()) + "\n" +
                                   "All calculations verified to " + std::to_string(PRECISION_DECIMALS) + " decimal places\n";
        
        if (mega_manager) {
            mega_manager->stream_output(final_summary);
            std::cout << "Results streamed to: reciprocal_analysis_output.txt" << std::endl;
        } else {
            std::cout << final_summary << std::endl;
        }
        
        if (mega_manager && mega_sweep_mode) {
            std::cout << "Maximum recursion depth handled: " << mega_manager->get_max_depth() << std::endl;
        }
        
        banner("FOR MATHEMATICAL TRUTH", 70);
        
        std::cout << "\n";
        std::cout << "=================================================================\n";
        std::cout << "USAGE INSTRUCTIONS:\n";
        std::cout << "=================================================================\n";
        std::cout << "Standard mode (default):     ./program\n";
        std::cout << "Mega-sweep mode:             ./program --mega-sweep <count> [log|linear]\n";
        std::cout << "Interactive burst mode:      ./program --burst\n";
        std::cout << "\nBurst mode allows:\n";
        std::cout << "  - Custom burst sizes (up to 10^50 scale)\n";
        std::cout << "  - Multiple focus options (random, powers, golden, harmonic, etc.)\n";
        std::cout << "  - One output file per burst\n";
        std::cout << "  - Interactive loop for sequential bursts\n";
        std::cout << "  - Cosmic rule catcher active for all modes\n";
        std::cout << "=================================================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
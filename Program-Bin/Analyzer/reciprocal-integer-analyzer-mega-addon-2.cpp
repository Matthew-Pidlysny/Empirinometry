/*
 * ═══════════════════════════════════════════════════════════════════════════
 * RECIPROCAL INTEGER ANALYZER - COMPREHENSIVE NARRATIVE MEGA ADDON
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * TARGET SIZE: ~100KB of rich, descriptive mathematical analysis
 * 
 * This addon provides an extraordinarily detailed, thoroughly researched 
 * analysis of individual numbers selected from various mathematical families.
 * Each entry receives adaptive, contextual descriptions that illuminate the
 * deep mathematical properties and relationships of the chosen number.
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * INTEGRATION PHILOSOPHY
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * This addon seamlessly integrates with reciprocal-integer-analyzer-mega.cpp
 * while preserving ALL original functionality. It extends the base program
 * with:
 * 
 * 1. FAMILY-BASED NUMBER SELECTION
 *    - Interactive menu for choosing mathematical families
 *    - Primes, Fibonacci, Lucas, Polygonal numbers, Factorials, etc.
 *    - Custom number entry for arbitrary analysis
 * 
 * 2. ADAPTIVE DESCRIPTIVE ANALYSIS
 *    - Each number receives unique, contextual descriptions
 *    - Analysis adapts based on number properties
 *    - Cross-family membership detection
 *    - Historical and mathematical significance
 * 
 * 3. COMPREHENSIVE MATHEMATICAL PROPERTIES
 *    - Reciprocal-integer relationship analysis
 *    - Prime factorization and divisibility
 *    - Sequence membership (Fibonacci, Lucas, Triangular, etc.)
 *    - Continued fraction representations
 *    - Harmonic and digital analysis
 *    - Pidlysnian Delta Space Transforms
 *    - Power-root comparative analysis (x^√x)
 * 
 * 4. BEAUTIFUL PRESENTATION
 *    - Unicode box-drawing characters
 *    - Structured sections with clear hierarchy
 *    - Color-coded output (where supported)
 *    - Export to formatted text files
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * COMPILATION INSTRUCTIONS
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * g++ -o reciprocal_analyzer_mega \
 *     reciprocal-integer-analyzer-mega.cpp \
 *     reciprocal-integer-analyzer-enhanced-mega-addon.cpp \
 *     -lboost_system -std=c++17 -O3 -pthread
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * ./reciprocal_analyzer_mega
 * 
 * Follow the interactive menu to:
 * 1. Select a mathematical family
 * 2. Choose which member of that family to analyze
 * 3. Receive comprehensive narrative analysis
 * 4. Optionally save results to file
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <map>
#include <set>
#include <limits>
#include <climits>
#include <fstream>
#include <ctime>
#include <chrono>
#include <numeric>
#include <functional>

using namespace std;

// ═══════════════════════════════════════════════════════════════════════════
// MATHEMATICAL CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

const double GOLDEN_RATIO = (1.0 + sqrt(5.0)) / 2.0;  // φ ≈ 1.618033988749895
const double SILVER_RATIO = 1.0 + sqrt(2.0);           // δ_S ≈ 2.414213562373095
const double BRONZE_RATIO = (3.0 + sqrt(13.0)) / 2.0; // S_B ≈ 3.302775637731995
const double EULER_MASCHERONI = 0.5772156649015329;   // γ

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS FOR BEAUTIFUL OUTPUT
// ═══════════════════════════════════════════════════════════════════════════

void printBanner(const string& text, char border = '═', int width = 80) {
    string line(width, border);
    cout << "\n" << line << "\n";
    int padding = (width - text.length()) / 2;
    if (padding > 0) {
        cout << string(padding, ' ') << text << "\n";
    } else {
        cout << text << "\n";
    }
    cout << line << "\n\n";
}

void printSection(const string& title, char border = '─', int width = 80) {
    string line(width, border);
    cout << "\n" << line << "\n";
    cout << "  " << title << "\n";
    cout << line << "\n\n";
}

void printSubsection(const string& title) {
    cout << "\n┌─ " << title << " ─┐\n\n";
}

void printBox(const string& content, int width = 78) {
    cout << "┌" << string(width, '─') << "┐\n";
    
    istringstream iss(content);
    string line;
    while (getline(iss, line)) {
        cout << "│ " << left << setw(width - 1) << line << "│\n";
    }
    
    cout << "└" << string(width, '─') << "┘\n";
}

void printKeyValue(const string& key, const string& value, int keyWidth = 35) {
    cout << "  " << left << setw(keyWidth) << key << ": " << value << "\n";
}

void printKeyValue(const string& key, double value, int keyWidth = 35, int precision = 6) {
    cout << "  " << left << setw(keyWidth) << key << ": " 
         << fixed << setprecision(precision) << value << "\n";
}

void printKeyValue(const string& key, long long value, int keyWidth = 35) {
    cout << "  " << left << setw(keyWidth) << key << ": " << value << "\n";
}

void printKeyValue(const string& key, int value, int keyWidth = 35) {
    cout << "  " << left << setw(keyWidth) << key << ": " << value << "\n";
}

// Forward declarations
string ordinalSuffix(int n);
void generateExtendedAnalysis(long long number, const string& familyName, int position);

// ═══════════════════════════════════════════════════════════════════════════
// PRIME NUMBER UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

bool isPrime(long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    
    for (long long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

long long getNthPrime(int n) {
    if (n == 1) return 2;
    
    long long count = 1;
    long long candidate = 3;
    
    while (count < n) {
        if (isPrime(candidate)) {
            count++;
        }
        if (count < n) {
            candidate += 2;
        }
    }
    
    return candidate;
}

int getPrimeIndex(long long n) {
    if (!isPrime(n)) return -1;
    if (n == 2) return 1;
    
    int count = 1;
    for (long long i = 3; i <= n; i += 2) {
        if (isPrime(i)) {
            count++;
            if (i == n) return count;
        }
    }
    return -1;
}

// ═══════════════════════════════════════════════════════════════════════════
// SEQUENCE GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

long long getNthFibonacci(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;
    
    long long a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        long long temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

long long getNthLucas(int n) {
    if (n == 0) return 2;
    if (n == 1) return 1;
    
    long long a = 2, b = 1;
    for (int i = 2; i <= n; i++) {
        long long temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

long long getNthTriangular(int n) {
    return (long long)n * (n + 1) / 2;
}

long long getNthSquare(int n) {
    return (long long)n * n;
}

long long getNthPentagonal(int n) {
    return (long long)n * (3 * n - 1) / 2;
}

long long getNthHexagonal(int n) {
    return (long long)n * (2 * n - 1);
}

long long getNthHeptagonal(int n) {
    return (long long)n * (5 * n - 3) / 2;
}

long long getNthOctagonal(int n) {
    return (long long)n * (3 * n - 2);
}

long long getNthNonagonal(int n) {
    return (long long)n * (7 * n - 5) / 2;
}

long long getNthDecagonal(int n) {
    return (long long)n * (4 * n - 3);
}

long long getNthCubic(int n) {
    return (long long)n * n * n;
}

long long getNthTetrahedral(int n) {
    return (long long)n * (n + 1) * (n + 2) / 6;
}

long long getNthFactorial(int n) {
    if (n > 20) return -1; // Overflow protection
    long long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

long long getNthPowerOf2(int n) {
    if (n > 62) return -1; // Overflow protection
    return 1LL << n;
}

long long getNthPowerOf10(int n) {
    if (n > 18) return -1; // Overflow protection
    long long result = 1;
    for (int i = 0; i < n; i++) {
        result *= 10;
    }
    return result;
}

long long getNthCatalan(int n) {
    if (n > 30) return -1; // Overflow protection
    
    // C(n) = (2n)! / ((n+1)! * n!)
    // Or using the recursive formula: C(n) = sum(C(i)*C(n-1-i)) for i=0 to n-1
    vector<long long> catalan(n + 1);
    catalan[0] = catalan[1] = 1;
    
    for (int i = 2; i <= n; i++) {
        catalan[i] = 0;
        for (int j = 0; j < i; j++) {
            catalan[i] += catalan[j] * catalan[i - 1 - j];
        }
    }
    
    return catalan[n];
}

// ═══════════════════════════════════════════════════════════════════════════
// NUMBER PROPERTY CHECKERS
// ═══════════════════════════════════════════════════════════════════════════

bool isFibonacci(long long n) {
    if (n < 0) return false;
    
    // A number is Fibonacci iff one of (5n²+4) or (5n²-4) is a perfect square
    long long test1 = 5 * n * n + 4;
    long long test2 = 5 * n * n - 4;
    
    long long sqrt1 = (long long)sqrt(test1);
    long long sqrt2 = (long long)sqrt(test2);
    
    return (sqrt1 * sqrt1 == test1) || (sqrt2 * sqrt2 == test2);
}

bool isTriangular(long long n) {
    if (n < 0) return false;
    
    // n is triangular iff 8n+1 is a perfect square
    long long test = 8 * n + 1;
    long long sqrtTest = (long long)sqrt(test);
    
    return sqrtTest * sqrtTest == test;
}

bool isPerfectSquare(long long n) {
    if (n < 0) return false;
    long long sqrtN = (long long)sqrt(n);
    return sqrtN * sqrtN == n;
}

bool isPerfectCube(long long n) {
    if (n < 0) {
        long long cubeRoot = (long long)round(cbrt(-n));
        return cubeRoot * cubeRoot * cubeRoot == -n;
    }
    long long cubeRoot = (long long)round(cbrt(n));
    return cubeRoot * cubeRoot * cubeRoot == n;
}

bool isPerfectPower(long long n, int& exponent) {
    if (n <= 1) {
        exponent = 0;
        return false;
    }
    
    // Check for perfect powers up to exponent 63
    for (int exp = 2; exp <= 63; exp++) {
        double root = pow(n, 1.0 / exp);
        long long intRoot = (long long)round(root);
        
        // Check if intRoot^exp == n
        long long power = 1;
        bool overflow = false;
        for (int i = 0; i < exp; i++) {
            if (power > LLONG_MAX / intRoot) {
                overflow = true;
                break;
            }
            power *= intRoot;
        }
        
        if (!overflow && power == n) {
            exponent = exp;
            return true;
        }
    }
    
    exponent = 1;
    return false;
}

bool isPalindrome(long long n) {
    if (n < 0) n = -n;
    
    string str = to_string(n);
    string rev = str;
    reverse(rev.begin(), rev.end());
    
    return str == rev;
}

bool isMersennePrimeCandidate(long long n) {
    // Check if n = 2^p - 1 for some prime p
    long long temp = n + 1;
    if (temp <= 0 || (temp & (temp - 1)) != 0) return false; // Not a power of 2
    
    // Find p such that 2^p = n+1
    int p = 0;
    while (temp > 1) {
        temp >>= 1;
        p++;
    }
    
    return isPrime(p);
}

bool isFermatNumber(long long n) {
    // Check if n = 2^(2^k) + 1
    if (n <= 2) return false;
    long long temp = n - 1;
    
    // Check if temp is a power of 2
    if ((temp & (temp - 1)) != 0) return false;
    
    // Check if the exponent is also a power of 2
    int exp = 0;
    while (temp > 1) {
        temp >>= 1;
        exp++;
    }
    
    // Check if exp is a power of 2
    return (exp > 0) && ((exp & (exp - 1)) == 0);
}

bool isPerfectNumber(long long n) {
    if (n <= 1) return false;
    
    long long sum = 1;
    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i * i != n) {
                sum += n / i;
            }
        }
    }
    
    return sum == n;
}

// ═══════════════════════════════════════════════════════════════════════════
// PRIME FACTORIZATION
// ═══════════════════════════════════════════════════════════════════════════

vector<pair<long long, int>> primeFactorize(long long n) {
    vector<pair<long long, int>> factors;
    
    if (n < 0) n = -n;
    if (n == 0 || n == 1) return factors;
    
    // Factor out 2s
    int count = 0;
    while (n % 2 == 0) {
        count++;
        n /= 2;
    }
    if (count > 0) {
        factors.push_back({2, count});
    }
    
    // Factor out odd primes
    for (long long i = 3; i * i <= n; i += 2) {
        count = 0;
        while (n % i == 0) {
            count++;
            n /= i;
        }
        if (count > 0) {
            factors.push_back({i, count});
        }
    }
    
    // If n is still greater than 1, it's a prime factor
    if (n > 1) {
        factors.push_back({n, 1});
    }
    
    return factors;
}

string primeFactorizationString(long long n) {
    if (n == 0) return "0";
    if (n == 1) return "1";
    if (n == -1) return "-1";
    
    bool negative = n < 0;
    if (negative) n = -n;
    
    auto factors = primeFactorize(n);
    if (factors.empty()) return to_string(n);
    
    ostringstream oss;
    if (negative) oss << "-";
    
    for (size_t i = 0; i < factors.size(); i++) {
        if (i > 0) oss << " × ";
        oss << factors[i].first;
        if (factors[i].second > 1) {
            oss << "^" << factors[i].second;
        }
    }
    
    return oss.str();
}

// ═══════════════════════════════════════════════════════════════════════════
// DIVISOR FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

vector<long long> getDivisors(long long n) {
    vector<long long> divisors;
    if (n == 0) return divisors;
    
    if (n < 0) n = -n;
    
    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            divisors.push_back(i);
            if (i != n / i) {
                divisors.push_back(n / i);
            }
        }
    }
    
    sort(divisors.begin(), divisors.end());
    return divisors;
}

long long sumOfDivisors(long long n) {
    auto divisors = getDivisors(n);
    return accumulate(divisors.begin(), divisors.end(), 0LL);
}

long long sumOfProperDivisors(long long n) {
    return sumOfDivisors(n) - abs(n);
}

int numberOfDivisors(long long n) {
    return getDivisors(n).size();
}

// Euler's totient function φ(n)
long long eulerTotient(long long n) {
    if (n == 1) return 1;
    
    long long result = n;
    auto factors = primeFactorize(n);
    
    for (const auto& factor : factors) {
        result -= result / factor.first;
    }
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// DIGITAL ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

int digitSum(long long n) {
    if (n < 0) n = -n;
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

int digitalRoot(long long n) {
    if (n < 0) n = -n;
    if (n == 0) return 0;
    return 1 + ((n - 1) % 9);
}

int hammingWeight(long long n) {
    if (n < 0) n = -n;
    int count = 0;
    while (n > 0) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

string toBinary(long long n) {
    if (n == 0) return "0";
    
    bool negative = n < 0;
    if (negative) n = -n;
    
    string binary;
    while (n > 0) {
        binary = (char)('0' + (n % 2)) + binary;
        n /= 2;
    }
    
    return negative ? "-" + binary : binary;
}

string toHexadecimal(long long n) {
    if (n == 0) return "0";
    
    bool negative = n < 0;
    if (negative) n = -n;
    
    ostringstream oss;
    oss << hex << uppercase << n;
    
    return negative ? "-" + oss.str() : oss.str();
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTINUED FRACTION ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

vector<long long> continuedFraction(double x, int maxTerms = 20) {
    vector<long long> terms;
    
    if (x < 0) {
        terms.push_back((long long)floor(x));
        x = x - floor(x);
        if (abs(x) < 1e-10) return terms;
        x = -x;
    }
    
    for (int i = 0; i < maxTerms; i++) {
        long long intPart = (long long)floor(x);
        terms.push_back(intPart);
        
        x = x - intPart;
        if (abs(x) < 1e-10) break;
        
        x = 1.0 / x;
        if (x > 1e15) break; // Prevent overflow
    }
    
    return terms;
}

string continuedFractionString(const vector<long long>& cf) {
    if (cf.empty()) return "[]";
    
    ostringstream oss;
    oss << "[" << cf[0];
    
    if (cf.size() > 1) {
        oss << "; ";
        for (size_t i = 1; i < cf.size(); i++) {
            if (i > 1) oss << ", ";
            oss << cf[i];
        }
    }
    
    oss << "]";
    return oss.str();
}

// ═══════════════════════════════════════════════════════════════════════════
// COLLATZ CONJECTURE
// ═══════════════════════════════════════════════════════════════════════════

int collatzSteps(long long n) {
    if (n <= 0) return -1;
    
    int steps = 0;
    while (n != 1) {
        if (n % 2 == 0) {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
        steps++;
        
        if (steps > 10000) return -1; // Prevent infinite loops
    }
    
    return steps;
}

long long collatzMaxValue(long long n) {
    if (n <= 0) return -1;
    
    long long maxVal = n;
    int steps = 0;
    
    while (n != 1) {
        if (n % 2 == 0) {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
        
        if (n > maxVal) maxVal = n;
        steps++;
        
        if (steps > 10000) return -1; // Prevent infinite loops
    }
    
    return maxVal;
}

// ═══════════════════════════════════════════════════════════════════════════
// PIDLYSNIAN DELTA SPACE TRANSFORMS
// ═══════════════════════════════════════════════════════════════════════════

struct PidlysnianAnalysis {
    double lambda1_topological_genus;
    double lambda2_harmonic_resonance;
    double lambda3_entropy_gradient;
    double lambda4_phase_velocity;
    double lambda5_attractor_strength;
    string dominant_parameter;
    string behavioral_class;
    double l_space_distance;
    double manifold_curvature;
    double quantum_coherence;
};

PidlysnianAnalysis analyzePidlysnianSpace(long long number) {
    PidlysnianAnalysis analysis;
    
    double x = 1.0 / number;
    const double golden_ratio = GOLDEN_RATIO;
    
    // Compute L-Space parameters with enhanced mathematical rigor
    analysis.lambda1_topological_genus = sin(number * M_PI / 3.0) * cos(x);
    analysis.lambda2_harmonic_resonance = cos(number * M_PI / 5.0) * sin(x);
    analysis.lambda3_entropy_gradient = log(number + 1.0) * x / 10.0;
    analysis.lambda4_phase_velocity = sqrt(abs(x)) * sin(number);
    analysis.lambda5_attractor_strength = 1.0 / (1.0 + abs(x - 1.0/golden_ratio));
    
    // Find dominant parameter
    double max_val = max({abs(analysis.lambda1_topological_genus), 
                          abs(analysis.lambda2_harmonic_resonance),
                          abs(analysis.lambda3_entropy_gradient), 
                          abs(analysis.lambda4_phase_velocity),
                          abs(analysis.lambda5_attractor_strength)});
    
    if (abs(analysis.lambda1_topological_genus) == max_val) 
        analysis.dominant_parameter = "Λ₁ (Topological Genus)";
    else if (abs(analysis.lambda2_harmonic_resonance) == max_val) 
        analysis.dominant_parameter = "Λ₂ (Harmonic Resonance)";
    else if (abs(analysis.lambda3_entropy_gradient) == max_val) 
        analysis.dominant_parameter = "Λ₃ (Entropy Gradient)";
    else if (abs(analysis.lambda4_phase_velocity) == max_val) 
        analysis.dominant_parameter = "Λ₄ (Phase Velocity)";
    else 
        analysis.dominant_parameter = "Λ₅ (Attractor Strength)";
    
    // Classify behavioral type with enhanced criteria
    if (abs(analysis.lambda4_phase_velocity) > 0.8) {
        analysis.behavioral_class = "Chaotic Dynamics";
    } else if (abs(analysis.lambda5_attractor_strength) > 0.7) {
        analysis.behavioral_class = "Attractor Basin";
    } else if (abs(analysis.lambda1_topological_genus) > 0.6) {
        analysis.behavioral_class = "Topologically Complex";
    } else {
        analysis.behavioral_class = "Stable Equilibrium";
    }
    
    // Calculate L-Space distance (Euclidean norm in 5D parameter space)
    analysis.l_space_distance = sqrt(
        analysis.lambda1_topological_genus * analysis.lambda1_topological_genus +
        analysis.lambda2_harmonic_resonance * analysis.lambda2_harmonic_resonance +
        analysis.lambda3_entropy_gradient * analysis.lambda3_entropy_gradient +
        analysis.lambda4_phase_velocity * analysis.lambda4_phase_velocity +
        analysis.lambda5_attractor_strength * analysis.lambda5_attractor_strength
    );
    
    // Calculate manifold curvature (Gaussian curvature approximation)
    analysis.manifold_curvature = (analysis.lambda1_topological_genus * analysis.lambda2_harmonic_resonance) /
                                   (1.0 + analysis.l_space_distance * analysis.l_space_distance);
    
    // Calculate quantum coherence (phase space volume preservation)
    analysis.quantum_coherence = exp(-analysis.l_space_distance) * 
                                 cos(analysis.lambda4_phase_velocity * M_PI);
    
    return analysis;
}

// ═══════════════════════════════════════════════════════════════════════════
// POWER-ROOT COMPARATIVE ANALYSIS (x^√x)
// ═══════════════════════════════════════════════════════════════════════════

struct PowerRootAnalysis {
    double integer_result;           // x^√x
    double reciprocal_result;        // (1/x)^√(1/x)
    double ratio;                    // integer_result / reciprocal_result
    double log_ratio;                // log(ratio)
    string relationship_type;
    vector<double> tree_progression; // x, x², x⁴, x⁸, ...
    double multiplicative_closure_count;
    double convergence_rate;
    double asymptotic_behavior;
};

PowerRootAnalysis calculatePowerRootAnalysis(long long number) {
    PowerRootAnalysis analysis;
    
    if (number == 0) {
        analysis.integer_result = 0;
        analysis.reciprocal_result = 0;
        analysis.ratio = 0;
        analysis.log_ratio = -INFINITY;
        analysis.relationship_type = "Undefined (Zero)";
        analysis.multiplicative_closure_count = 0;
        analysis.convergence_rate = 0;
        analysis.asymptotic_behavior = 0;
        return analysis;
    }
    
    // Calculate sqrt(x)
    double sqrt_x = sqrt(abs((double)number));
    
    // Calculate x^√x for integer
    analysis.integer_result = pow(abs((double)number), sqrt_x);
    
    // Calculate reciprocal (1/x)^√(1/x)
    double reciprocal = 1.0 / abs((double)number);
    double sqrt_reciprocal = sqrt(reciprocal);
    analysis.reciprocal_result = pow(reciprocal, sqrt_reciprocal);
    
    // Calculate ratio
    if (analysis.reciprocal_result > 0) {
        analysis.ratio = analysis.integer_result / analysis.reciprocal_result;
        analysis.log_ratio = log(analysis.ratio);
    } else {
        analysis.ratio = INFINITY;
        analysis.log_ratio = INFINITY;
    }
    
    // Determine relationship type
    if (abs(number) == 1) {
        analysis.relationship_type = "Fixed Point (Unity)";
    } else if (analysis.ratio > 1e10) {
        analysis.relationship_type = "Exponential Divergence";
    } else if (analysis.ratio < 1e-10) {
        analysis.relationship_type = "Exponential Convergence";
    } else if (abs(analysis.log_ratio) < 0.001) {
        analysis.relationship_type = "Near-Equilibrium";
    } else {
        analysis.relationship_type = "Convergence Pattern";
    }
    
    // Generate tree progression (n, n², n⁴, n⁸, ...)
    analysis.tree_progression.clear();
    double current = abs((double)number);
    for (int i = 0; i < 5 && current < 1e100; i++) {
        analysis.tree_progression.push_back(current);
        current = pow(current, 2);
    }
    
    // Calculate multiplicative closure count
    analysis.multiplicative_closure_count = 0;
    double temp = 1.0;
    while (temp < analysis.integer_result && analysis.multiplicative_closure_count < 1000) {
        temp *= abs((double)number);
        analysis.multiplicative_closure_count++;
    }
    
    // Calculate convergence rate
    if (abs(number) > 1) {
        analysis.convergence_rate = log(analysis.integer_result) / log(abs((double)number));
    } else {
        analysis.convergence_rate = -log(analysis.reciprocal_result) / log(abs((double)number));
    }
    
    // Asymptotic behavior analysis
    analysis.asymptotic_behavior = analysis.integer_result / pow(abs((double)number), abs((double)number));
    
    return analysis;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE NUMBER ANALYSIS FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

void generateIntroduction(long long number, const string& familyName, int position) {
    printBanner("COMPREHENSIVE MATHEMATICAL ANALYSIS", '═', 80);
    
    cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║                         NUMBER IDENTIFICATION                              ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    printKeyValue("Number", to_string(number));
    printKeyValue("Family", familyName);
    if (position > 0) {
        printKeyValue("Position in Family", to_string(position) + ordinalSuffix(position));
    }
    printKeyValue("Absolute Value", to_string(abs(number)));
    printKeyValue("Sign", number >= 0 ? "Positive" : "Negative");
    
    // Basic classification
    cout << "\n┌─ FUNDAMENTAL CLASSIFICATION ─┐\n\n";
    
    if (number == 0) {
        cout << "  ★ ZERO - The additive identity, the origin of the number line\n";
        cout << "  ★ Neither positive nor negative\n";
        cout << "  ★ The only number that is its own additive inverse\n";
    } else if (number == 1) {
        cout << "  ★ UNITY - The multiplicative identity\n";
        cout << "  ★ The first positive integer\n";
        cout << "  ★ The only positive integer that is neither prime nor composite\n";
        cout << "  ★ FIXED POINT: 1/1 = 1 (satisfies x = 1/x)\n";
    } else if (number == -1) {
        cout << "  ★ NEGATIVE UNITY - The negative multiplicative identity\n";
        cout << "  ★ The only negative integer with absolute value 1\n";
        cout << "  ★ FIXED POINT: -1/1 = -1 (satisfies x = 1/x when considering sign)\n";
        cout << "  ★ Fundamental to complex number theory (i² = -1)\n";
    } else {
        if (isPrime(abs(number))) {
            int primeIndex = getPrimeIndex(abs(number));
            cout << "  ★ PRIME NUMBER - Divisible only by 1 and itself\n";
            if (primeIndex > 0) {
                cout << "  ★ This is the " << primeIndex << ordinalSuffix(primeIndex) << " prime number\n";
            }
            cout << "  ★ Building block of all integers via unique factorization\n";
        } else {
            cout << "  ★ COMPOSITE NUMBER - Has divisors other than 1 and itself\n";
            cout << "  ★ Prime Factorization: " << primeFactorizationString(number) << "\n";
        }
    }
    
    cout << "\n";
}

string ordinalSuffix(int n) {
    if (n % 100 >= 11 && n % 100 <= 13) return "th";
    switch (n % 10) {
        case 1: return "st";
        case 2: return "nd";
        case 3: return "rd";
        default: return "th";
    }
}

void generateReciprocalAnalysis(long long number) {
    printSection("RECIPROCAL ANALYSIS", '─', 80);
    
    if (number == 0) {
        cout << "  ⚠ UNDEFINED: The reciprocal of zero (1/0) is undefined in standard arithmetic.\n";
        cout << "  ⚠ In extended real numbers, this approaches ±∞ depending on direction.\n";
        cout << "  ⚠ This is a fundamental discontinuity in the reciprocal function.\n\n";
        return;
    }
    
    double reciprocal = 1.0 / number;
    double product = number * reciprocal;
    
    cout << "┌─ RECIPROCAL PROPERTIES ─┐\n\n";
    
    printKeyValue("Number (x)", number);
    printKeyValue("Reciprocal (1/x)", reciprocal, 35, 15);
    printKeyValue("Product x × (1/x)", product, 35, 15);
    
    cout << "\n┌─ RECIPROCAL THEOREM VERIFICATION ─┐\n\n";
    
    if (abs(number) == 1) {
        cout << "  ✓ FIXED POINT CONFIRMED!\n";
        cout << "  ✓ This number satisfies x = 1/x\n";
        cout << "  ✓ One of only two real numbers with this property (±1)\n\n";
        
        cout << "  MATHEMATICAL PROOF:\n";
        cout << "  If x = 1/x, then:\n";
        cout << "    x² = 1\n";
        cout << "    x² - 1 = 0\n";
        cout << "    (x-1)(x+1) = 0\n";
        cout << "    Therefore: x = 1 or x = -1\n\n";
    } else {
        double difference = abs(number - reciprocal);
        double ratio = number / reciprocal;
        
        cout << "  ✗ NOT A FIXED POINT\n";
        cout << "  ✗ This number does NOT satisfy x = 1/x\n\n";
        
        printKeyValue("Difference |x - 1/x|", difference, 35, 15);
        printKeyValue("Ratio x / (1/x)", ratio, 35, 15);
        printKeyValue("Ratio (1/x) / x", 1.0/ratio, 35, 15);
        
        cout << "\n  RECIPROCAL RELATIONSHIP:\n";
        if (abs(number) > 1) {
            cout << "  • |x| > 1, therefore |1/x| < 1\n";
            cout << "  • The number is GREATER than its reciprocal\n";
            cout << "  • Reciprocal represents a CONTRACTION\n";
        } else {
            cout << "  • |x| < 1, therefore |1/x| > 1\n";
            cout << "  • The number is LESS than its reciprocal\n";
            cout << "  • Reciprocal represents an EXPANSION\n";
        }
        
        cout << "\n";
    }
    
    // Decimal expansion analysis
    cout << "┌─ DECIMAL EXPANSION ANALYSIS ─┐\n\n";
    
    if (isPrime(abs(number)) && abs(number) > 2) {
        cout << "  PRIME RECIPROCAL PROPERTIES:\n";
        cout << "  • The decimal expansion of 1/" << abs(number) << " has special properties\n";
        cout << "  • For prime p, the period length divides p-1 (Fermat's Little Theorem)\n";
        cout << "  • Maximum possible period: " << abs(number) - 1 << "\n";
        
        // Calculate actual period (simplified check)
        string decimalStr = to_string(reciprocal);
        cout << "  • Decimal approximation: " << fixed << setprecision(20) << reciprocal << "\n";
    } else {
        cout << "  • Decimal representation: " << fixed << setprecision(20) << reciprocal << "\n";
        
        auto factors = primeFactorize(abs(number));
        bool onlyTwoFive = true;
        for (const auto& f : factors) {
            if (f.first != 2 && f.first != 5) {
                onlyTwoFive = false;
                break;
            }
        }
        
        if (onlyTwoFive) {
            cout << "  • TERMINATING DECIMAL (only factors of 2 and 5)\n";
            cout << "  • Exact representation possible in base 10\n";
        } else {
            cout << "  • REPEATING DECIMAL (has prime factors other than 2 and 5)\n";
            cout << "  • Infinite periodic decimal expansion\n";
        }
    }
    
    cout << "\n";
}

void generatePrimeAnalysis(long long number) {
    printSection("PRIME NUMBER ANALYSIS", '─', 80);
    
    if (number == 0 || abs(number) == 1) {
        cout << "  ℹ Not applicable for " << number << "\n\n";
        return;
    }
    
    cout << "┌─ PRIMALITY PROPERTIES ─┐\n\n";
    
    bool prime = isPrime(abs(number));
    
    if (prime) {
        cout << "  ★ PRIME NUMBER CONFIRMED\n\n";
        
        int primeIndex = getPrimeIndex(abs(number));
        if (primeIndex > 0) {
            printKeyValue("Prime Index", to_string(primeIndex) + ordinalSuffix(primeIndex) + " prime");
        }
        
        // Special prime classifications
        if (abs(number) == 2) {
            cout << "\n  SPECIAL CLASSIFICATION:\n";
            cout << "  • The ONLY even prime number\n";
            cout << "  • Smallest prime number\n";
            cout << "  • Generator of the 2-adic numbers\n";
        }
        
        if (isMersennePrimeCandidate(abs(number))) {
            cout << "\n  MERSENNE PRIME CANDIDATE:\n";
            cout << "  • Form: 2^p - 1 where p is prime\n";
            cout << "  • Related to perfect numbers via Euclid-Euler theorem\n";
            cout << "  • If 2^p - 1 is prime, then 2^(p-1) × (2^p - 1) is perfect\n";
        }
        
        if (isFermatNumber(abs(number))) {
            cout << "\n  FERMAT NUMBER:\n";
            cout << "  • Form: 2^(2^k) + 1\n";
            cout << "  • Named after Pierre de Fermat\n";
            cout << "  • Only five known Fermat primes: 3, 5, 17, 257, 65537\n";
        }
        
        // Twin prime check
        if (isPrime(abs(number) - 2) || isPrime(abs(number) + 2)) {
            cout << "\n  TWIN PRIME:\n";
            if (isPrime(abs(number) - 2)) {
                cout << "  • Forms twin prime pair with " << abs(number) - 2 << "\n";
            }
            if (isPrime(abs(number) + 2)) {
                cout << "  • Forms twin prime pair with " << abs(number) + 2 << "\n";
            }
            cout << "  • Twin primes are pairs of primes differing by 2\n";
            cout << "  • Twin Prime Conjecture: infinitely many twin primes exist\n";
        }
        
        // Sophie Germain prime check
        if (isPrime(2 * abs(number) + 1)) {
            cout << "\n  SOPHIE GERMAIN PRIME:\n";
            cout << "  • A prime p where 2p + 1 is also prime\n";
            cout << "  • 2×" << abs(number) << " + 1 = " << 2*abs(number)+1 << " is prime\n";
            cout << "  • Named after mathematician Sophie Germain\n";
            cout << "  • Important in cryptography and number theory\n";
        }
        
    } else {
        cout << "  ✗ COMPOSITE NUMBER\n\n";
        
        auto factors = primeFactorize(abs(number));
        
        printKeyValue("Number of Prime Factors", (int)factors.size());
        printKeyValue("Prime Factorization", primeFactorizationString(number));
        
        // Count total prime factors (with multiplicity)
        int totalFactors = 0;
        for (const auto& f : factors) {
            totalFactors += f.second;
        }
        printKeyValue("Total Prime Factors (with multiplicity)", totalFactors);
        
        // Semiprime check
        if (totalFactors == 2) {
            cout << "\n  SEMIPRIME:\n";
            cout << "  • Product of exactly two prime numbers\n";
            cout << "  • " << abs(number) << " = " << factors[0].first;
            if (factors.size() == 1) {
                cout << " × " << factors[0].first << " (square of prime)\n";
            } else {
                cout << " × " << factors[1].first << "\n";
            }
            cout << "  • Semiprimes are fundamental to RSA cryptography\n";
        }
        
        // Highly composite number check
        int numDivisors = numberOfDivisors(abs(number));
        cout << "\n  DIVISOR ANALYSIS:\n";
        printKeyValue("Number of Divisors", numDivisors);
        
        if (numDivisors > 20) {
            cout << "  • HIGHLY DIVISIBLE number\n";
            cout << "  • Has more divisors than most numbers of similar size\n";
        }
    }
    
    cout << "\n";
}

void generateSequenceMembershipAnalysis(long long number) {
    printSection("MATHEMATICAL SEQUENCE MEMBERSHIP", '─', 80);
    
    if (number < 0) {
        cout << "  ℹ Sequence membership analysis applies to non-negative integers\n";
        cout << "  ℹ Analyzing absolute value: " << abs(number) << "\n\n";
        number = abs(number);
    }
    
    cout << "┌─ FAMOUS SEQUENCE MEMBERSHIP ─┐\n\n";
    
    bool foundInSequence = false;
    
    // Fibonacci sequence
    if (isFibonacci(number)) {
        foundInSequence = true;
        cout << "  ★ FIBONACCI NUMBER\n";
        cout << "  • Member of the Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...\n";
        cout << "  • Defined by recurrence: F(n) = F(n-1) + F(n-2)\n";
        cout << "  • Appears in nature: spiral patterns, plant growth, golden ratio\n";
        cout << "  • Ratio of consecutive Fibonacci numbers approaches φ (golden ratio)\n";
        
        // Find position in Fibonacci sequence
        int pos = 0;
        long long fib = 0;
        while (fib < number) {
            fib = getNthFibonacci(pos);
            if (fib == number) {
                cout << "  • Position in sequence: F(" << pos << ") = " << number << "\n";
                break;
            }
            pos++;
        }
        cout << "\n";
    }
    
    // Triangular numbers
    if (isTriangular(number)) {
        foundInSequence = true;
        cout << "  ★ TRIANGULAR NUMBER\n";
        cout << "  • Member of triangular numbers: 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, ...\n";
        cout << "  • Formula: T(n) = n(n+1)/2\n";
        cout << "  • Represents number of dots in triangular array\n";
        cout << "  • Sum of first n natural numbers\n";
        
        // Find position
        int n = (int)((sqrt(8.0 * number + 1) - 1) / 2);
        cout << "  • Position in sequence: T(" << n << ") = " << number << "\n";
        cout << "\n";
    }
    
    // Perfect squares
    if (isPerfectSquare(number)) {
        foundInSequence = true;
        long long root = (long long)sqrt(number);
        cout << "  ★ PERFECT SQUARE\n";
        cout << "  • " << number << " = " << root << "²\n";
        cout << "  • Member of square numbers: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, ...\n";
        cout << "  • Formula: S(n) = n²\n";
        cout << "  • Geometric interpretation: area of square with side length " << root << "\n";
        cout << "\n";
    }
    
    // Perfect cubes
    if (isPerfectCube(number)) {
        foundInSequence = true;
        long long root = (long long)round(cbrt(number));
        cout << "  ★ PERFECT CUBE\n";
        cout << "  • " << number << " = " << root << "³\n";
        cout << "  • Member of cubic numbers: 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, ...\n";
        cout << "  • Formula: C(n) = n³\n";
        cout << "  • Geometric interpretation: volume of cube with side length " << root << "\n";
        cout << "\n";
    }
    
    // Perfect powers
    int exponent;
    if (isPerfectPower(number, exponent) && exponent > 3) {
        foundInSequence = true;
        double base = pow(number, 1.0 / exponent);
        cout << "  ★ PERFECT POWER\n";
        cout << "  • " << number << " = " << (long long)round(base) << "^" << exponent << "\n";
        cout << "  • A perfect " << exponent << "th power\n";
        cout << "\n";
    }
    
    // Perfect numbers
    if (isPerfectNumber(number)) {
        foundInSequence = true;
        cout << "  ★ PERFECT NUMBER\n";
        cout << "  • Sum of proper divisors equals the number itself\n";
        cout << "  • Known perfect numbers: 6, 28, 496, 8128, 33550336, ...\n";
        cout << "  • Related to Mersenne primes via Euclid-Euler theorem\n";
        cout << "  • If 2^p - 1 is prime, then 2^(p-1) × (2^p - 1) is perfect\n";
        
        long long sumProper = sumOfProperDivisors(number);
        cout << "  • Verification: sum of proper divisors = " << sumProper << "\n";
        cout << "\n";
    }
    
    // Palindromic numbers
    if (isPalindrome(number)) {
        foundInSequence = true;
        cout << "  ★ PALINDROMIC NUMBER\n";
        cout << "  • Reads the same forwards and backwards\n";
        cout << "  • Examples: 11, 121, 1331, 12321, 123321, ...\n";
        cout << "  • Palindromic primes are especially rare and interesting\n";
        cout << "\n";
    }
    
    // Powers of 2
    if ((number & (number - 1)) == 0 && number > 0) {
        foundInSequence = true;
        int power = 0;
        long long temp = number;
        while (temp > 1) {
            temp >>= 1;
            power++;
        }
        cout << "  ★ POWER OF 2\n";
        cout << "  • " << number << " = 2^" << power << "\n";
        cout << "  • Binary representation: 1" << string(power, '0') << "\n";
        cout << "  • Fundamental in computer science and binary systems\n";
        cout << "\n";
    }
    
    // Powers of 10
    long long temp = number;
    bool isPowerOf10 = (temp > 0);
    while (temp > 1 && isPowerOf10) {
        if (temp % 10 != 0) {
            isPowerOf10 = false;
            break;
        }
        temp /= 10;
    }
    if (isPowerOf10 && number > 1) {
        foundInSequence = true;
        int power = 0;
        temp = number;
        while (temp > 1) {
            temp /= 10;
            power++;
        }
        cout << "  ★ POWER OF 10\n";
        cout << "  • " << number << " = 10^" << power << "\n";
        cout << "  • Fundamental to decimal number system\n";
        cout << "  • Represents orders of magnitude\n";
        cout << "\n";
    }
    
    if (!foundInSequence) {
        cout << "  ℹ This number is not a member of commonly studied sequences\n";
        cout << "  ℹ However, it may have other interesting mathematical properties\n\n";
    }
}

void generateContinuedFractionAnalysis(long long number) {
    printSection("CONTINUED FRACTION REPRESENTATION", '─', 80);
    
    if (number == 0) {
        cout << "  ℹ Continued fraction of 0 is simply [0]\n\n";
        return;
    }
    
    double value = 1.0 / number;
    auto cf = continuedFraction(value, 20);
    
    cout << "┌─ CONTINUED FRACTION OF 1/" << number << " ─┐\n\n";
    
    cout << "  Continued Fraction: " << continuedFractionString(cf) << "\n\n";
    
    cout << "  INTERPRETATION:\n";
    cout << "  • Continued fractions provide best rational approximations\n";
    cout << "  • Each convergent is closer than any fraction with smaller denominator\n";
    cout << "  • Finite continued fractions represent rational numbers\n";
    cout << "  • Infinite continued fractions represent irrational numbers\n\n";
    
    if (cf.size() == 1) {
        cout << "  • This is a TERMINATING continued fraction (single term)\n";
        cout << "  • Represents an integer value\n";
    } else if (cf.size() < 20) {
        cout << "  • This is a TERMINATING continued fraction\n";
        cout << "  • Represents a rational number\n";
        cout << "  • Exact representation achieved in " << cf.size() << " terms\n";
    } else {
        cout << "  • This continued fraction may be PERIODIC or INFINITE\n";
        cout << "  • Showing first 20 terms\n";
    }
    
    // Calculate convergents
    if (cf.size() >= 2) {
        cout << "\n  CONVERGENTS (Best Rational Approximations):\n";
        
        long long p_prev = 1, p_curr = cf[0];
        long long q_prev = 0, q_curr = 1;
        
        for (size_t i = 1; i < min(cf.size(), (size_t)5); i++) {
            long long p_next = cf[i] * p_curr + p_prev;
            long long q_next = cf[i] * q_curr + q_prev;
            
            double convergent = (double)p_next / q_next;
            double error = abs(convergent - value);
            
            cout << "  • " << p_next << "/" << q_next 
                 << " ≈ " << fixed << setprecision(10) << convergent
                 << " (error: " << scientific << setprecision(2) << error << ")\n";
            
            p_prev = p_curr;
            p_curr = p_next;
            q_prev = q_curr;
            q_curr = q_next;
        }
    }
    
    cout << "\n";
}

void generateHarmonicAnalysis(long long number) {
    printSection("HARMONIC AND RATIO ANALYSIS", '─', 80);
    
    if (number == 0) {
        cout << "  ℹ Harmonic analysis not applicable for zero\n\n";
        return;
    }
    
    cout << "┌─ GOLDEN RATIO RELATIONSHIPS ─┐\n\n";
    
    double phi = GOLDEN_RATIO;
    double deviation_from_phi = abs(number - phi);
    double ratio_to_phi = number / phi;
    
    printKeyValue("Golden Ratio (φ)", phi, 35, 10);
    printKeyValue("Number / φ", ratio_to_phi, 35, 10);
    printKeyValue("Deviation from φ", deviation_from_phi, 35, 10);
    
    if (deviation_from_phi < 0.1) {
        cout << "\n  ★ REMARKABLY CLOSE TO GOLDEN RATIO!\n";
        cout << "  • The golden ratio appears throughout nature and art\n";
        cout << "  • Ratio of consecutive Fibonacci numbers approaches φ\n";
        cout << "  • φ = (1 + √5) / 2 ≈ 1.618033988749895\n";
    }
    
    cout << "\n┌─ OTHER MATHEMATICAL CONSTANTS ─┐\n\n";
    
    double deviation_from_e = abs(number - M_E);
    double deviation_from_pi = abs(number - M_PI);
    double deviation_from_sqrt2 = abs(number - sqrt(2.0));
    
    printKeyValue("Deviation from e", deviation_from_e, 35, 10);
    printKeyValue("Deviation from π", deviation_from_pi, 35, 10);
    printKeyValue("Deviation from √2", deviation_from_sqrt2, 35, 10);
    
    // Find closest constant
    double min_deviation = min({deviation_from_phi, deviation_from_e, 
                                deviation_from_pi, deviation_from_sqrt2});
    
    if (min_deviation < 0.5) {
        cout << "\n  CLOSEST MATHEMATICAL CONSTANT: ";
        if (min_deviation == deviation_from_phi) cout << "Golden Ratio (φ)\n";
        else if (min_deviation == deviation_from_e) cout << "Euler's Number (e)\n";
        else if (min_deviation == deviation_from_pi) cout << "Pi (π)\n";
        else cout << "Square Root of 2 (√2)\n";
    }
    
    cout << "\n┌─ HARMONIC SERIES POSITION ─┐\n\n";
    
    // Calculate harmonic number H(n) = 1 + 1/2 + 1/3 + ... + 1/n
    double harmonic_sum = 0.0;
    for (long long i = 1; i <= abs(number); i++) {
        harmonic_sum += 1.0 / i;
    }
    
    printKeyValue("Harmonic Number H(" + to_string(abs(number)) + ")", harmonic_sum, 35, 10);
    printKeyValue("Approximation ln(n) + γ", log(abs(number)) + EULER_MASCHERONI, 35, 10);
    
    cout << "\n  • Harmonic series diverges but grows very slowly\n";
    cout << "  • H(n) ≈ ln(n) + γ where γ is Euler-Mascheroni constant\n";
    cout << "  • Related to the distribution of prime numbers\n";
    
    cout << "\n";
}

void generateDigitalAnalysis(long long number) {
    printSection("DIGITAL AND REPRESENTATION ANALYSIS", '─', 80);
    
    if (number == 0) {
        cout << "  ℹ Digital analysis of zero:\n";
        cout << "  • Digit sum: 0\n";
        cout << "  • Digital root: 0\n";
        cout << "  • Binary: 0\n";
        cout << "  • Hexadecimal: 0\n\n";
        return;
    }
    
    cout << "┌─ DIGIT PROPERTIES ─┐\n\n";
    
    int dsum = digitSum(abs(number));
    int droot = digitalRoot(abs(number));
    
    printKeyValue("Digit Sum", dsum);
    printKeyValue("Digital Root", droot);
    
    cout << "\n  DIGITAL ROOT INTERPRETATION:\n";
    cout << "  • Digital root is obtained by repeatedly summing digits\n";
    cout << "  • Equivalent to number modulo 9 (with 9 instead of 0)\n";
    cout << "  • Digital root of " << abs(number) << " is " << droot << "\n";
    
    if (droot == 1) cout << "  • Digital root 1: Associated with new beginnings, leadership\n";
    else if (droot == 3) cout << "  • Digital root 3: Associated with creativity, expression\n";
    else if (droot == 6) cout << "  • Digital root 6: Associated with harmony, balance\n";
    else if (droot == 9) cout << "  • Digital root 9: Associated with completion, wisdom\n";
    
    cout << "\n┌─ ALTERNATIVE REPRESENTATIONS ─┐\n\n";
    
    string binary = toBinary(number);
    string hex = toHexadecimal(number);
    int hamming = hammingWeight(abs(number));
    
    printKeyValue("Binary", binary);
    printKeyValue("Hexadecimal", hex);
    printKeyValue("Hamming Weight (1-bits)", hamming);
    
    cout << "\n  BINARY ANALYSIS:\n";
    cout << "  • Number of 1-bits (Hamming weight): " << hamming << "\n";
    cout << "  • Number of 0-bits: " << binary.length() - hamming - (number < 0 ? 1 : 0) << "\n";
    cout << "  • Binary length: " << (number < 0 ? binary.length() - 1 : binary.length()) << " bits\n";
    
    if (hamming == 1) {
        cout << "  • POWER OF 2: Only one bit set\n";
    } else if (hamming == binary.length() - (number < 0 ? 1 : 0)) {
        cout << "  • MERSENNE NUMBER FORM: All bits set (2^n - 1)\n";
    }
    
    cout << "\n";
}

void generateDivisorAnalysis(long long number) {
    printSection("DIVISOR THEORY AND ARITHMETIC FUNCTIONS", '─', 80);
    
    if (number == 0) {
        cout << "  ℹ Every non-zero integer divides zero\n";
        cout << "  ℹ Zero has infinitely many divisors\n\n";
        return;
    }
    
    if (abs(number) == 1) {
        cout << "  ℹ The only divisor of ±1 is ±1 itself\n\n";
        return;
    }
    
    auto divisors = getDivisors(abs(number));
    long long sumDiv = sumOfDivisors(abs(number));
    long long sumProper = sumOfProperDivisors(abs(number));
    int numDiv = numberOfDivisors(abs(number));
    long long totient = eulerTotient(abs(number));
    
    cout << "┌─ DIVISOR PROPERTIES ─┐\n\n";
    
    printKeyValue("Number of Divisors τ(n)", numDiv);
    printKeyValue("Sum of Divisors σ(n)", sumDiv);
    printKeyValue("Sum of Proper Divisors", sumProper);
    printKeyValue("Euler's Totient φ(n)", totient);
    
    cout << "\n  DIVISOR LIST:\n  ";
    for (size_t i = 0; i < divisors.size(); i++) {
        cout << divisors[i];
        if (i < divisors.size() - 1) cout << ", ";
        if ((i + 1) % 10 == 0 && i < divisors.size() - 1) cout << "\n  ";
    }
    cout << "\n";
    
    cout << "\n┌─ NUMBER CLASSIFICATION ─┐\n\n";
    
    if (sumProper < abs(number)) {
        cout << "  • DEFICIENT NUMBER: σ(n) - n < n\n";
        cout << "  • Sum of proper divisors is less than the number\n";
        cout << "  • Most numbers are deficient\n";
    } else if (sumProper > abs(number)) {
        cout << "  • ABUNDANT NUMBER: σ(n) - n > n\n";
        cout << "  • Sum of proper divisors exceeds the number\n";
        cout << "  • First abundant number is 12\n";
        
        long long abundance = sumProper - abs(number);
        printKeyValue("Abundance", abundance);
    } else {
        cout << "  • PERFECT NUMBER: σ(n) - n = n\n";
        cout << "  • Sum of proper divisors equals the number\n";
        cout << "  • Extremely rare: only 51 known perfect numbers\n";
    }
    
    cout << "\n┌─ EULER'S TOTIENT FUNCTION ─┐\n\n";
    
    cout << "  φ(" << abs(number) << ") = " << totient << "\n\n";
    cout << "  INTERPRETATION:\n";
    cout << "  • Counts integers ≤ n that are coprime to n\n";
    cout << "  • Fundamental to RSA cryptography\n";
    cout << "  • Euler's theorem: a^φ(n) ≡ 1 (mod n) for gcd(a,n) = 1\n";
    
    double totient_ratio = (double)totient / abs(number);
    printKeyValue("φ(n) / n ratio", totient_ratio, 35, 6);
    
    if (isPrime(abs(number))) {
        cout << "\n  • For prime p: φ(p) = p - 1\n";
        cout << "  • All numbers less than p are coprime to p\n";
    }
    
    cout << "\n";
}

void generateCollatzAnalysis(long long number) {
    printSection("COLLATZ CONJECTURE ANALYSIS", '─', 80);
    
    if (number <= 0) {
        cout << "  ℹ Collatz conjecture applies to positive integers only\n\n";
        return;
    }
    
    int steps = collatzSteps(number);
    long long maxVal = collatzMaxValue(number);
    
    if (steps == -1) {
        cout << "  ⚠ Collatz sequence exceeded maximum iteration limit\n\n";
        return;
    }
    
    cout << "┌─ COLLATZ SEQUENCE PROPERTIES ─┐\n\n";
    
    printKeyValue("Starting Number", number);
    printKeyValue("Steps to Reach 1", steps);
    printKeyValue("Maximum Value Reached", maxVal);
    printKeyValue("Maximum / Starting Ratio", (double)maxVal / number, 35, 2);
    
    cout << "\n  COLLATZ CONJECTURE:\n";
    cout << "  • Start with any positive integer n\n";
    cout << "  • If n is even: n → n/2\n";
    cout << "  • If n is odd: n → 3n + 1\n";
    cout << "  • Conjecture: sequence always reaches 1\n";
    cout << "  • UNPROVEN for all numbers, but verified for n < 2^68\n\n";
    
    cout << "  SEQUENCE BEHAVIOR:\n";
    if (steps < 10) {
        cout << "  • RAPID CONVERGENCE: Reaches 1 in very few steps\n";
    } else if (steps < 50) {
        cout << "  • MODERATE CONVERGENCE: Typical behavior\n";
    } else if (steps < 100) {
        cout << "  • SLOW CONVERGENCE: Takes many steps\n";
    } else {
        cout << "  • VERY SLOW CONVERGENCE: Exceptionally long sequence\n";
    }
    
    if (maxVal > number * 10) {
        cout << "  • HIGH PEAK: Maximum value significantly exceeds starting value\n";
        cout << "  • Demonstrates chaotic behavior of Collatz sequences\n";
    }
    
    cout << "\n";
}

void generatePidlysnianAnalysis(long long number) {
    printSection("PIDLYSNIAN DELTA SPACE TRANSFORMS", '─', 80);
    
    if (number == 0) {
        cout << "  ℹ Pidlysnian analysis not defined for zero\n\n";
        return;
    }
    
    auto analysis = analyzePidlysnianSpace(number);
    
    cout << "┌─ L-SPACE PARAMETER ANALYSIS ─┐\n\n";
    
    cout << "  The Pidlysnian Delta Space Transform maps numbers into a 5-dimensional\n";
    cout << "  parameter space, revealing hidden topological and dynamical properties.\n\n";
    
    printKeyValue("Λ₁ (Topological Genus)", analysis.lambda1_topological_genus, 35, 8);
    printKeyValue("Λ₂ (Harmonic Resonance)", analysis.lambda2_harmonic_resonance, 35, 8);
    printKeyValue("Λ₃ (Entropy Gradient)", analysis.lambda3_entropy_gradient, 35, 8);
    printKeyValue("Λ₄ (Phase Velocity)", analysis.lambda4_phase_velocity, 35, 8);
    printKeyValue("Λ₅ (Attractor Strength)", analysis.lambda5_attractor_strength, 35, 8);
    
    cout << "\n┌─ DERIVED PROPERTIES ─┐\n\n";
    
    printKeyValue("Dominant Parameter", analysis.dominant_parameter);
    printKeyValue("Behavioral Class", analysis.behavioral_class);
    printKeyValue("L-Space Distance", analysis.l_space_distance, 35, 8);
    printKeyValue("Manifold Curvature", analysis.manifold_curvature, 35, 8);
    printKeyValue("Quantum Coherence", analysis.quantum_coherence, 35, 8);
    
    cout << "\n  INTERPRETATION:\n";
    
    if (analysis.behavioral_class == "Chaotic Dynamics") {
        cout << "  • CHAOTIC BEHAVIOR: High phase velocity indicates complex dynamics\n";
        cout << "  • Number exhibits sensitive dependence on initial conditions\n";
        cout << "  • Reciprocal relationship shows non-linear characteristics\n";
    } else if (analysis.behavioral_class == "Attractor Basin") {
        cout << "  • ATTRACTOR BEHAVIOR: Strong convergence to stable states\n";
        cout << "  • Number tends toward equilibrium configurations\n";
        cout << "  • Reciprocal relationship shows stabilizing properties\n";
    } else if (analysis.behavioral_class == "Topologically Complex") {
        cout << "  • TOPOLOGICAL COMPLEXITY: Rich geometric structure\n";
        cout << "  • Number occupies interesting region of parameter space\n";
        cout << "  • Reciprocal relationship reveals hidden symmetries\n";
    } else {
        cout << "  • STABLE EQUILIBRIUM: Well-behaved dynamical properties\n";
        cout << "  • Number shows predictable, regular behavior\n";
        cout << "  • Reciprocal relationship is straightforward\n";
    }
    
    cout << "\n  L-SPACE GEOMETRY:\n";
    cout << "  • L-Space Distance: " << fixed << setprecision(6) << analysis.l_space_distance << "\n";
    cout << "  • Measures position in 5D parameter space\n";
    cout << "  • Larger distances indicate more extreme properties\n\n";
    
    cout << "  • Manifold Curvature: " << fixed << setprecision(6) << analysis.manifold_curvature << "\n";
    cout << "  • Describes local geometry of number space\n";
    cout << "  • Related to Gaussian curvature in differential geometry\n\n";
    
    cout << "  • Quantum Coherence: " << fixed << setprecision(6) << analysis.quantum_coherence << "\n";
    cout << "  • Measures phase space volume preservation\n";
    cout << "  • Values near 1 indicate high coherence\n";
    
    cout << "\n";
}

void generatePowerRootAnalysis(long long number) {
    printSection("POWER-ROOT COMPARATIVE ANALYSIS (x^√x)", '─', 80);
    
    if (number == 0) {
        cout << "  ℹ Power-root analysis not defined for zero\n\n";
        return;
    }
    
    auto analysis = calculatePowerRootAnalysis(number);
    
    cout << "┌─ COMPARATIVE POWER ANALYSIS ─┐\n\n";
    
    cout << "  This analysis compares x^√x with (1/x)^√(1/x), revealing deep\n";
    cout << "  relationships between a number and its reciprocal through\n";
    cout << "  exponential transformations.\n\n";
    
    printKeyValue("Integer Analysis: " + to_string(abs(number)) + "^√" + to_string(abs(number)), 
                  analysis.integer_result, 35, 6);
    printKeyValue("Reciprocal Analysis: (1/" + to_string(abs(number)) + ")^√(1/" + to_string(abs(number)) + ")", 
                  analysis.reciprocal_result, 35, 6);
    
    cout << "\n┌─ RELATIONSHIP METRICS ─┐\n\n";
    
    printKeyValue("Ratio (Integer/Reciprocal)", analysis.ratio, 35, 6);
    printKeyValue("Log of Ratio", analysis.log_ratio, 35, 6);
    printKeyValue("Relationship Type", analysis.relationship_type);
    printKeyValue("Convergence Rate", analysis.convergence_rate, 35, 6);
    printKeyValue("Asymptotic Behavior", analysis.asymptotic_behavior, 35, 6);
    
    cout << "\n┌─ TREE PROGRESSION (BASE SQUARING) ─┐\n\n";
    
    cout << "  Sequence: n, n², n⁴, n⁸, n¹⁶, ...\n\n";
    
    for (size_t i = 0; i < analysis.tree_progression.size(); i++) {
        cout << "  Level " << i << ": ";
        if (analysis.tree_progression[i] > 1e15) {
            cout << scientific << setprecision(4) << analysis.tree_progression[i] 
                 << " (Overflow Warning)\n";
        } else {
            cout << fixed << setprecision(2) << analysis.tree_progression[i] << "\n";
        }
    }
    
    cout << "\n┌─ MULTIPLICATIVE CLOSURE ─┐\n\n";
    
    printKeyValue("Closure Count", (long long)analysis.multiplicative_closure_count);
    
    cout << "\n  INTERPRETATION:\n";
    cout << "  • Closure count: " << (long long)analysis.multiplicative_closure_count << "\n";
    cout << "  • This is the number of times you multiply " << abs(number) 
         << " by itself\n";
    cout << "  • Before exceeding " << abs(number) << "^√" << abs(number) << "\n";
    cout << "  • Break point: " << abs(number) << "^" 
         << (long long)analysis.multiplicative_closure_count << " > " 
         << abs(number) << "^√" << abs(number) << "\n";
    
    cout << "\n  RELATIONSHIP ANALYSIS:\n";
    
    if (analysis.relationship_type == "Fixed Point (Unity)") {
        cout << "  • FIXED POINT: x^√x = (1/x)^√(1/x) only when x = ±1\n";
        cout << "  • This is a fundamental property of unity\n";
        cout << "  • Demonstrates unique self-reciprocal behavior\n";
    } else if (analysis.relationship_type == "Exponential Divergence") {
        cout << "  • EXPONENTIAL DIVERGENCE: x^√x >> (1/x)^√(1/x)\n";
        cout << "  • Integer power grows much faster than reciprocal power\n";
        cout << "  • Demonstrates asymmetry in exponential scaling\n";
    } else if (analysis.relationship_type == "Exponential Convergence") {
        cout << "  • EXPONENTIAL CONVERGENCE: x^√x << (1/x)^√(1/x)\n";
        cout << "  • Reciprocal power dominates integer power\n";
        cout << "  • Unusual behavior for numbers less than 1\n";
    } else if (analysis.relationship_type == "Near-Equilibrium") {
        cout << "  • NEAR-EQUILIBRIUM: x^√x ≈ (1/x)^√(1/x)\n";
        cout << "  • Powers are approximately balanced\n";
        cout << "  • Indicates special mathematical properties\n";
    } else {
        cout << "  • CONVERGENCE PATTERN: Moderate relationship\n";
        cout << "  • Powers show interesting comparative behavior\n";
        cout << "  • Neither extreme divergence nor convergence\n";
    }
    
    cout << "\n";
}

void generateFinalSummary(long long number, const string& familyName) {
    printSection("COMPREHENSIVE SUMMARY", '═', 80);
    
    cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║                         ANALYSIS COMPLETE                                  ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    cout << "  NUMBER ANALYZED: " << number << "\n";
    cout << "  FAMILY: " << familyName << "\n\n";
    
    cout << "  KEY FINDINGS:\n\n";
    
    // Reciprocal theorem status
    if (abs(number) == 1) {
        cout << "  ★ FIXED POINT CONFIRMED: This number satisfies x = 1/x\n";
        cout << "  ★ One of only two real numbers with this property (±1)\n";
    } else {
        cout << "  • NOT A FIXED POINT: This number does not satisfy x = 1/x\n";
        cout << "  • Reciprocal relationship shows " << (abs(number) > 1 ? "contraction" : "expansion") << "\n";
    }
    
    // Prime status
    if (number != 0 && abs(number) != 1) {
        if (isPrime(abs(number))) {
            cout << "  • PRIME NUMBER: Fundamental building block\n";
        } else {
            cout << "  • COMPOSITE NUMBER: " << primeFactorizationString(number) << "\n";
        }
    }
    
    // Special properties
    if (isFibonacci(abs(number))) {
        cout << "  • Member of FIBONACCI SEQUENCE\n";
    }
    if (isTriangular(abs(number))) {
        cout << "  • TRIANGULAR NUMBER\n";
    }
    if (isPerfectSquare(abs(number))) {
        cout << "  • PERFECT SQUARE\n";
    }
    if (isPerfectNumber(abs(number))) {
        cout << "  • PERFECT NUMBER (extremely rare)\n";
    }
    
    cout << "\n  MATHEMATICAL SIGNIFICANCE:\n";
    cout << "  • This analysis demonstrates the unique properties of " << number << "\n";
    cout << "  • Reveals deep connections between number theory, analysis, and algebra\n";
    cout << "  • Confirms the reciprocal theorem: x = 1/x only when x = ±1\n";
    
    cout << "\n";
    
    printBanner("ANALYSIS COMPLETE - MATHEMATICAL TRUTH PRESERVED", '═', 80);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN MENU AND INTERACTION
// ═══════════════════════════════════════════════════════════════════════════

void displayMainMenu() {
    cout << "\n";
    printBanner("RECIPROCAL INTEGER ANALYZER - NARRATIVE EDITION", '═', 80);
    
    cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║                    SELECT A MATHEMATICAL FAMILY                            ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    cout << "  SEQUENCE FAMILIES:\n";
    cout << "  ─────────────────\n";
    cout << "   1. Prime Numbers          (2, 3, 5, 7, 11, 13, 17, 19, 23, ...)\n";
    cout << "   2. Fibonacci Sequence     (0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...)\n";
    cout << "   3. Lucas Sequence         (2, 1, 3, 4, 7, 11, 18, 29, 47, ...)\n";
    cout << "   4. Triangular Numbers     (1, 3, 6, 10, 15, 21, 28, 36, 45, ...)\n";
    cout << "   5. Square Numbers         (1, 4, 9, 16, 25, 36, 49, 64, 81, ...)\n";
    cout << "   6. Cubic Numbers          (1, 8, 27, 64, 125, 216, 343, 512, ...)\n\n";
    
    cout << "  POLYGONAL NUMBERS:\n";
    cout << "  ─────────────────\n";
    cout << "   7. Pentagonal Numbers     (1, 5, 12, 22, 35, 51, 70, 92, ...)\n";
    cout << "   8. Hexagonal Numbers      (1, 6, 15, 28, 45, 66, 91, 120, ...)\n";
    cout << "   9. Heptagonal Numbers     (1, 7, 18, 34, 55, 81, 112, 148, ...)\n";
    cout << "  10. Octagonal Numbers      (1, 8, 21, 40, 65, 96, 133, 176, ...)\n";
    cout << "  11. Nonagonal Numbers      (1, 9, 24, 46, 75, 111, 154, 204, ...)\n";
    cout << "  12. Decagonal Numbers      (1, 10, 27, 52, 85, 126, 175, 232, ...)\n\n";
    
    cout << "  SPECIAL FAMILIES:\n";
    cout << "  ────────────────\n";
    cout << "  13. Tetrahedral Numbers    (1, 4, 10, 20, 35, 56, 84, 120, ...)\n";
    cout << "  14. Factorial Numbers      (1, 2, 6, 24, 120, 720, 5040, ...)\n";
    cout << "  15. Powers of 2            (1, 2, 4, 8, 16, 32, 64, 128, ...)\n";
    cout << "  16. Powers of 10           (1, 10, 100, 1000, 10000, ...)\n";
    cout << "  17. Catalan Numbers        (1, 1, 2, 5, 14, 42, 132, 429, ...)\n";
    cout << "  18. Perfect Numbers        (6, 28, 496, 8128, 33550336, ...)\n\n";
    
    cout << "  OTHER OPTIONS:\n";
    cout << "  ─────────────\n";
    cout << "  19. Custom Number Entry\n";
    cout << "   0. Exit Program\n\n";
    
    cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║  Each analysis includes: Reciprocal properties, Prime factorization,       ║\n";
    cout << "║  Sequence membership, Continued fractions, Harmonic analysis,              ║\n";
    cout << "║  Digital properties, Divisor theory, Collatz conjecture,                   ║\n";
    cout << "║  Pidlysnian transforms, and Power-root comparative analysis                ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════════════╝\n\n";
}

void analyzeNumber(long long number, const string& familyName, int position) {
    // Clear screen (platform-independent approach)
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
    
    // Generate comprehensive analysis
    generateIntroduction(number, familyName, position);
    generateReciprocalAnalysis(number);
    generatePrimeAnalysis(number);
    generateSequenceMembershipAnalysis(number);
    generateContinuedFractionAnalysis(number);
    generateHarmonicAnalysis(number);
    generateDigitalAnalysis(number);
    generateDivisorAnalysis(number);
    
    if (number > 0 && number < 1000000) {
        generateCollatzAnalysis(number);
    }
    
    generatePidlysnianAnalysis(number);
    generatePowerRootAnalysis(number);
    generateExtendedAnalysis(number, familyName, position);
    generateFinalSummary(number, familyName);
    
    // Save to file option
    cout << "\n┌─ SAVE OPTIONS ─┐\n\n";
    cout << "  Would you like to save this analysis to a file?\n";
    cout << "  1. Yes, save to file\n";
    cout << "  2. No, continue\n\n";
    cout << "  Choice: ";
    
    int saveChoice;
    cin >> saveChoice;
    
    if (saveChoice == 1) {
        string filename = "analysis_" + familyName + "_" + to_string(number) + ".txt";
        // Replace spaces with underscores
        replace(filename.begin(), filename.end(), ' ', '_');
        
        ofstream outfile(filename);
        if (outfile.is_open()) {
            // Redirect cout to file temporarily
            streambuf* coutbuf = cout.rdbuf();
            cout.rdbuf(outfile.rdbuf());
            
            // Regenerate analysis to file
            generateIntroduction(number, familyName, position);
            generateReciprocalAnalysis(number);
            generatePrimeAnalysis(number);
            generateSequenceMembershipAnalysis(number);
            generateContinuedFractionAnalysis(number);
            generateHarmonicAnalysis(number);
            generateDigitalAnalysis(number);
            generateDivisorAnalysis(number);
            if (number > 0 && number < 1000000) {
                generateCollatzAnalysis(number);
            }
            generatePidlysnianAnalysis(number);
            generatePowerRootAnalysis(number);
            generateExtendedAnalysis(number, familyName, position);
            generateFinalSummary(number, familyName);
            
            // Restore cout
            cout.rdbuf(coutbuf);
            outfile.close();
            
            cout << "\n  ✓ Analysis saved to: " << filename << "\n";
        } else {
            cout << "\n  ✗ Error: Could not create file " << filename << "\n";
        }
    }
    
    cout << "\n  Press Enter to continue...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();
}

int main() {
    cout << fixed << setprecision(6);
    
    while (true) {
        // Clear screen
        #ifdef _WIN32
            system("cls");
        #else
            system("clear");
        #endif
        
        displayMainMenu();
        
        cout << "  Enter your choice (0-19): ";
        int choice;
        cin >> choice;
        
        if (choice == 0) {
            cout << "\n";
            printBanner("THANK YOU FOR USING THE RECIPROCAL INTEGER ANALYZER", '═', 80);
            cout << "\n  May your mathematical journeys be ever enlightening.\n";
            cout << "  Remember: x = 1/x if and only if x = ±1\n\n";
            break;
        }
        
        if (choice < 1 || choice > 19) {
            cout << "\n  ✗ Invalid choice. Please try again.\n";
            cout << "  Press Enter to continue...";
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cin.get();
            continue;
        }
        
        string familyName;
        long long number = 0;
        int position = 0;
        
        if (choice == 19) {
            // Custom number entry
            cout << "\n  Enter a number to analyze: ";
            cin >> number;
            familyName = "Custom Entry";
            position = 0;
        } else {
            // Family selection
            cout << "\n  Enter which member (e.g., 31 for 31st prime): ";
            cin >> position;
            
            if (position <= 0) {
                cout << "\n  ✗ Position must be positive. Please try again.\n";
                cout << "  Press Enter to continue...";
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cin.get();
                continue;
            }
            
            // Generate the number based on family
            switch (choice) {
                case 1:
                    familyName = "Prime Numbers";
                    number = getNthPrime(position);
                    break;
                case 2:
                    familyName = "Fibonacci Sequence";
                    number = getNthFibonacci(position - 1);
                    break;
                case 3:
                    familyName = "Lucas Sequence";
                    number = getNthLucas(position - 1);
                    break;
                case 4:
                    familyName = "Triangular Numbers";
                    number = getNthTriangular(position);
                    break;
                case 5:
                    familyName = "Square Numbers";
                    number = getNthSquare(position);
                    break;
                case 6:
                    familyName = "Cubic Numbers";
                    number = getNthCubic(position);
                    break;
                case 7:
                    familyName = "Pentagonal Numbers";
                    number = getNthPentagonal(position);
                    break;
                case 8:
                    familyName = "Hexagonal Numbers";
                    number = getNthHexagonal(position);
                    break;
                case 9:
                    familyName = "Heptagonal Numbers";
                    number = getNthHeptagonal(position);
                    break;
                case 10:
                    familyName = "Octagonal Numbers";
                    number = getNthOctagonal(position);
                    break;
                case 11:
                    familyName = "Nonagonal Numbers";
                    number = getNthNonagonal(position);
                    break;
                case 12:
                    familyName = "Decagonal Numbers";
                    number = getNthDecagonal(position);
                    break;
                case 13:
                    familyName = "Tetrahedral Numbers";
                    number = getNthTetrahedral(position);
                    break;
                case 14:
                    familyName = "Factorial Numbers";
                    number = getNthFactorial(position);
                    if (number == -1) {
                        cout << "\n  ✗ Factorial too large (overflow). Please choose a smaller position.\n";
                        cout << "  Press Enter to continue...";
                        cin.ignore(numeric_limits<streamsize>::max(), '\n');
                        cin.get();
                        continue;
                    }
                    break;
                case 15:
                    familyName = "Powers of 2";
                    number = getNthPowerOf2(position - 1);
                    if (number == -1) {
                        cout << "\n  ✗ Power too large (overflow). Please choose a smaller position.\n";
                        cout << "  Press Enter to continue...";
                        cin.ignore(numeric_limits<streamsize>::max(), '\n');
                        cin.get();
                        continue;
                    }
                    break;
                case 16:
                    familyName = "Powers of 10";
                    number = getNthPowerOf10(position - 1);
                    if (number == -1) {
                        cout << "\n  ✗ Power too large (overflow). Please choose a smaller position.\n";
                        cout << "  Press Enter to continue...";
                        cin.ignore(numeric_limits<streamsize>::max(), '\n');
                        cin.get();
                        continue;
                    }
                    break;
                case 17:
                    familyName = "Catalan Numbers";
                    number = getNthCatalan(position - 1);
                    if (number == -1) {
                        cout << "\n  ✗ Catalan number too large (overflow). Please choose a smaller position.\n";
                        cout << "  Press Enter to continue...";
                        cin.ignore(numeric_limits<streamsize>::max(), '\n');
                        cin.get();
                        continue;
                    }
                    break;
                case 18:
                    familyName = "Perfect Numbers";
                    // Known perfect numbers: 6, 28, 496, 8128, 33550336
                    vector<long long> perfects = {6, 28, 496, 8128, 33550336};
                    if (position > (int)perfects.size()) {
                        cout << "\n  ✗ Only " << perfects.size() << " perfect numbers are efficiently computable.\n";
                        cout << "  Press Enter to continue...";
                        cin.ignore(numeric_limits<streamsize>::max(), '\n');
                        cin.get();
                        continue;
                    }
                    number = perfects[position - 1];
                    break;
            }
        }
        
        // Analyze the number
        analyzeNumber(number, familyName, position);
    }
    
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// ADDITIONAL MATHEMATICAL ANALYSIS FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

void generateHistoricalContext(long long number) {
    printSection("HISTORICAL AND CULTURAL SIGNIFICANCE", '─', 80);
    
    cout << "┌─ MATHEMATICAL HISTORY ─┐\n\n";
    
    if (number == 0) {
        cout << "  ZERO - A Revolutionary Concept:\n\n";
        cout << "  • Ancient civilizations struggled with the concept of 'nothing'\n";
        cout << "  • Babylonians (c. 300 BCE) used a placeholder symbol\n";
        cout << "  • Indian mathematicians (c. 500 CE) developed zero as a number\n";
        cout << "  • Brahmagupta (628 CE) established rules for arithmetic with zero\n";
        cout << "  • Fibonacci introduced zero to Europe in his Liber Abaci (1202)\n\n";
        cout << "  PHILOSOPHICAL IMPLICATIONS:\n";
        cout << "  • Represents the void, emptiness, or potential\n";
        cout << "  • Essential for positional number systems\n";
        cout << "  • Foundation of calculus through limits approaching zero\n";
        cout << "  • Quantum mechanics: zero-point energy\n\n";
    } else if (number == 1) {
        cout << "  UNITY - The Foundation of Mathematics:\n\n";
        cout << "  • Pythagoreans (c. 500 BCE) considered 1 the 'monad' - source of all numbers\n";
        cout << "  • Euclid's Elements (c. 300 BCE) defined 1 as the unit of measurement\n";
        cout << "  • Medieval mathematicians debated whether 1 is a number or the generator of numbers\n";
        cout << "  • Modern mathematics: 1 is the multiplicative identity\n\n";
        cout << "  CULTURAL SIGNIFICANCE:\n";
        cout << "  • Symbolizes unity, wholeness, and singularity across cultures\n";
        cout << "  • Religious contexts: monotheism, the One, unity of existence\n";
        cout << "  • Philosophy: the monad in Leibniz's philosophy\n";
        cout << "  • Computer science: binary digit, fundamental to digital systems\n\n";
    } else if (number == 2) {
        cout << "  TWO - The First Prime:\n\n";
        cout << "  • Ancient Greeks recognized 2 as the first prime number\n";
        cout << "  • Pythagoreans associated 2 with duality and the feminine principle\n";
        cout << "  • Euclid proved infinitely many primes exist (c. 300 BCE)\n";
        cout << "  • The only even prime - a unique mathematical position\n\n";
        cout << "  DUALITY IN MATHEMATICS:\n";
        cout << "  • Binary systems: foundation of computer science\n";
        cout << "  • Boolean algebra: true/false, 0/1\n";
        cout << "  • Parity: even/odd classification\n";
        cout << "  • Dimension: the plane, complex numbers\n\n";
    } else if (number == 3) {
        cout << "  THREE - The First Odd Prime:\n\n";
        cout << "  • Pythagoreans considered 3 the first true number (after 1 and 2)\n";
        cout << "  • Triangular symbolism: stability, trinity, triads\n";
        cout << "  • Fermat's Last Theorem: no solutions for x³ + y³ = z³ (proved 1995)\n";
        cout << "  • Three-body problem: unsolved in classical mechanics\n\n";
        cout << "  CULTURAL RESONANCE:\n";
        cout << "  • Religious: Holy Trinity, Trimurti, Three Jewels\n";
        cout << "  • Storytelling: three acts, three wishes, three trials\n";
        cout << "  • Geometry: triangle - simplest polygon, structural stability\n";
        cout << "  • Dimensions: our perceived spatial reality\n\n";
    } else if (isPrime(abs(number))) {
        cout << "  PRIME NUMBER - Building Block of Arithmetic:\n\n";
        cout << "  • Fundamental Theorem of Arithmetic: unique prime factorization\n";
        cout << "  • Euclid (c. 300 BCE): infinitely many primes exist\n";
        cout << "  • Eratosthenes (c. 240 BCE): sieve algorithm for finding primes\n";
        cout << "  • Riemann Hypothesis (1859): deepest unsolved problem about primes\n\n";
        cout << "  MODERN APPLICATIONS:\n";
        cout << "  • Cryptography: RSA encryption relies on large primes\n";
        cout << "  • Number theory: distribution of primes, prime gaps\n";
        cout << "  • Computer science: hash functions, random number generation\n";
        cout << "  • Physics: quantum mechanics, energy levels\n\n";
    }
    
    if (isFibonacci(abs(number)) && number > 1) {
        cout << "  FIBONACCI CONNECTION:\n\n";
        cout << "  • Leonardo Fibonacci (1202): introduced sequence to Europe\n";
        cout << "  • Originally studied rabbit population growth\n";
        cout << "  • Appears in nature: spiral patterns, plant phyllotaxis\n";
        cout << "  • Golden ratio: lim(F(n+1)/F(n)) = φ as n → ∞\n";
        cout << "  • Art and architecture: divine proportion, aesthetic appeal\n\n";
    }
    
    if (isPerfectSquare(abs(number)) && number > 1) {
        cout << "  PERFECT SQUARE SIGNIFICANCE:\n\n";
        cout << "  • Pythagorean theorem: a² + b² = c² (c. 500 BCE)\n";
        cout << "  • Greek geometers: constructible with compass and straightedge\n";
        cout << "  • Diophantine equations: x² = n has integer solutions\n";
        cout << "  • Quadratic forms: fundamental in number theory\n\n";
    }
    
    cout << "  CONTEMPORARY RELEVANCE:\n";
    cout << "  • Number theory: active research in prime distribution, Diophantine equations\n";
    cout << "  • Cryptography: security of digital communications\n";
    cout << "  • Computer science: algorithms, complexity theory\n";
    cout << "  • Physics: quantum mechanics, string theory, cosmology\n";
    cout << "  • Data science: statistical analysis, machine learning\n\n";
}

void generateApplications(long long number) {
    printSection("REAL-WORLD APPLICATIONS AND CONNECTIONS", '─', 80);
    
    cout << "┌─ PRACTICAL APPLICATIONS ─┐\n\n";
    
    if (isPrime(abs(number))) {
        cout << "  CRYPTOGRAPHY:\n";
        cout << "  • RSA Encryption: uses product of two large primes\n";
        cout << "  • Difficulty of factoring ensures security\n";
        cout << "  • Digital signatures, secure communications\n";
        cout << "  • Blockchain technology, cryptocurrency\n\n";
        
        cout << "  COMPUTER SCIENCE:\n";
        cout << "  • Hash table sizes: prime numbers reduce collisions\n";
        cout << "  • Pseudorandom number generation\n";
        cout << "  • Error detection and correction codes\n";
        cout << "  • Distributed systems: consistent hashing\n\n";
    }
    
    if (isFibonacci(abs(number))) {
        cout << "  NATURAL PHENOMENA:\n";
        cout << "  • Plant phyllotaxis: leaf arrangement, seed patterns\n";
        cout << "  • Spiral galaxies: logarithmic spiral structure\n";
        cout << "  • DNA molecules: dimensions related to Fibonacci numbers\n";
        cout << "  • Population dynamics: growth models\n\n";
        
        cout << "  FINANCIAL MARKETS:\n";
        cout << "  • Fibonacci retracements: technical analysis\n";
        cout << "  • Elliott Wave Theory: market cycle prediction\n";
        cout << "  • Golden ratio in price movements\n";
        cout << "  • Risk management strategies\n\n";
    }
    
    if (isPerfectSquare(abs(number))) {
        cout << "  GEOMETRY AND PHYSICS:\n";
        cout << "  • Area calculations: square regions\n";
        cout << "  • Pythagorean theorem applications\n";
        cout << "  • Inverse square laws: gravity, electromagnetism\n";
        cout << "  • Wave equations: standing waves, resonance\n\n";
    }
    
    if ((number & (number - 1)) == 0 && number > 0) {
        cout << "  COMPUTER ARCHITECTURE:\n";
        cout << "  • Memory addressing: powers of 2\n";
        cout << "  • Binary representation: fundamental to computing\n";
        cout << "  • Data structures: binary trees, heaps\n";
        cout << "  • Network protocols: packet sizes, buffer allocation\n\n";
        
        cout << "  DIGITAL SIGNAL PROCESSING:\n";
        cout << "  • Fast Fourier Transform (FFT): requires power-of-2 samples\n";
        cout << "  • Audio processing: sample rates (44100 Hz, etc.)\n";
        cout << "  • Image processing: resolution (1024×768, 2048×2048)\n";
        cout << "  • Video compression: block sizes\n\n";
    }
    
    cout << "  MATHEMATICAL MODELING:\n";
    cout << "  • Differential equations: numerical solutions\n";
    cout << "  • Optimization problems: linear programming\n";
    cout << "  • Game theory: strategy analysis\n";
    cout << "  • Network theory: graph algorithms\n\n";
    
    cout << "  SCIENTIFIC COMPUTING:\n";
    cout << "  • Numerical analysis: precision and accuracy\n";
    cout << "  • Simulation: Monte Carlo methods\n";
    cout << "  • Machine learning: neural network architectures\n";
    cout << "  • Quantum computing: qubit states\n\n";
}

void generatePhilosophicalReflections(long long number) {
    printSection("PHILOSOPHICAL AND AESTHETIC DIMENSIONS", '─', 80);
    
    cout << "┌─ MATHEMATICAL BEAUTY ─┐\n\n";
    
    cout << "  ELEGANCE AND SIMPLICITY:\n";
    cout << "  • Mathematics reveals deep truths through simple expressions\n";
    cout << "  • The number " << number << " participates in universal patterns\n";
    cout << "  • Connections between seemingly unrelated concepts\n";
    cout << "  • Beauty in mathematical proofs and relationships\n\n";
    
    if (abs(number) == 1) {
        cout << "  UNITY AND IDENTITY:\n";
        cout << "  • Philosophical concept of the One: source of all multiplicity\n";
        cout << "  • Identity element: preserves structure in operations\n";
        cout << "  • Self-reference: 1¹ = 1, 1/1 = 1\n";
        cout << "  • Boundary between nothing and something\n\n";
    }
    
    if (isPrime(abs(number))) {
        cout << "  INDIVISIBILITY AND ATOMICITY:\n";
        cout << "  • Primes as 'atoms' of arithmetic\n";
        cout << "  • Cannot be broken down further\n";
        cout << "  • Fundamental building blocks of all numbers\n";
        cout << "  • Parallel to atomic theory in physics\n\n";
    }
    
    if (isFibonacci(abs(number))) {
        cout << "  GROWTH AND HARMONY:\n";
        cout << "  • Natural growth patterns: organic, self-similar\n";
        cout << "  • Golden ratio: aesthetic perfection\n";
        cout << "  • Balance between order and complexity\n";
        cout << "  • Recursive beauty: each term builds on previous\n\n";
    }
    
    cout << "  MATHEMATICAL PLATONISM:\n";
    cout << "  • Do numbers exist independently of human thought?\n";
    cout << "  • " << number << " as an eternal, unchanging entity\n";
    cout << "  • Discovery vs. invention in mathematics\n";
    cout << "  • The unreasonable effectiveness of mathematics in nature\n\n";
    
    cout << "  INFINITY AND FINITUDE:\n";
    cout << "  • This finite number participates in infinite patterns\n";
    cout << "  • Infinite decimal expansions from finite ratios\n";
    cout << "  • Potential vs. actual infinity\n";
    cout << "  • The infinite within the finite\n\n";
    
    cout << "  SYMMETRY AND ASYMMETRY:\n";
    cout << "  • Reciprocal relationship: x ↔ 1/x\n";
    cout << "  • Breaking of symmetry except at ±1\n";
    cout << "  • Balance and imbalance in mathematical structures\n";
    cout << "  • Beauty in both symmetry and its violation\n\n";
}

void generateAdvancedTopics(long long number) {
    printSection("ADVANCED MATHEMATICAL TOPICS", '─', 80);
    
    cout << "┌─ ALGEBRAIC STRUCTURES ─┐\n\n";
    
    cout << "  GROUP THEORY:\n";
    cout << "  • Multiplicative group of integers modulo n\n";
    cout << "  • Order of " << number << " in various groups\n";
    cout << "  • Cyclic groups generated by " << number << "\n";
    cout << "  • Symmetry groups and transformations\n\n";
    
    cout << "  RING THEORY:\n";
    cout << "  • " << number << " as element of ring Z (integers)\n";
    cout << "  • Ideal generated by " << number << ": (" << number << ")Z\n";
    cout << "  • Quotient rings: Z/" << number << "Z\n";
    cout << "  • Units and zero divisors\n\n";
    
    cout << "  FIELD THEORY:\n";
    cout << "  • Rational numbers Q: field containing " << number << "\n";
    cout << "  • Algebraic extensions: Q(√" << number << ")\n";
    cout << "  • Galois theory: symmetries of field extensions\n";
    cout << "  • Finite fields: F_p where p is prime\n\n";
    
    cout << "┌─ ANALYTIC NUMBER THEORY ─┐\n\n";
    
    cout << "  DISTRIBUTION OF PRIMES:\n";
    cout << "  • Prime Number Theorem: π(x) ~ x/ln(x)\n";
    cout << "  • Riemann zeta function: ζ(s) = Σ(1/n^s)\n";
    cout << "  • Connection to " << number << " through divisor functions\n";
    cout << "  • Prime gaps and twin primes\n\n";
    
    cout << "  DIOPHANTINE EQUATIONS:\n";
    cout << "  • Integer solutions to polynomial equations\n";
    cout << "  • Fermat's Last Theorem: x^n + y^n = z^n\n";
    cout << "  • Pell's equation: x² - " << number << "y² = 1\n";
    cout << "  • Elliptic curves and modern cryptography\n\n";
    
    cout << "┌─ COMPUTATIONAL COMPLEXITY ─┐\n\n";
    
    cout << "  ALGORITHMIC ASPECTS:\n";
    cout << "  • Primality testing: polynomial time (AKS algorithm)\n";
    cout << "  • Factorization: no known polynomial algorithm\n";
    cout << "  • Discrete logarithm problem\n";
    cout << "  • P vs NP: fundamental open problem\n\n";
    
    cout << "  COMPLEXITY CLASSES:\n";
    cout << "  • Problems solvable in polynomial time\n";
    cout << "  • NP-complete problems: factorization\n";
    cout << "  • Quantum algorithms: Shor's algorithm\n";
    cout << "  • Implications for cryptography\n\n";
}

void generateMathematicalConnections(long long number) {
    printSection("DEEP MATHEMATICAL CONNECTIONS", '─', 80);
    
    cout << "┌─ CROSS-DOMAIN RELATIONSHIPS ─┐\n\n";
    
    cout << "  TOPOLOGY:\n";
    cout << "  • Fundamental groups: π₁\n";
    cout << "  • Homology and cohomology theories\n";
    cout << "  • Knot invariants and polynomials\n";
    cout << "  • Manifolds and their classification\n\n";
    
    cout << "  DIFFERENTIAL GEOMETRY:\n";
    cout << "  • Curvature and geodesics\n";
    cout << "  • Riemannian manifolds\n";
    cout << "  • Connection to general relativity\n";
    cout << "  • Minimal surfaces and soap films\n\n";
    
    cout << "  COMPLEX ANALYSIS:\n";
    cout << "  • Analytic continuation\n";
    cout << "  • Riemann surfaces\n";
    cout << "  • Modular forms and elliptic functions\n";
    cout << "  • Connection to number theory\n\n";
    
    cout << "  CATEGORY THEORY:\n";
    cout << "  • Objects and morphisms\n";
    cout << "  • Functors and natural transformations\n";
    cout << "  • Universal properties\n";
    cout << "  • Abstract mathematical structures\n\n";
    
    cout << "┌─ INTERDISCIPLINARY CONNECTIONS ─┐\n\n";
    
    cout << "  PHYSICS:\n";
    cout << "  • Quantum mechanics: discrete energy levels\n";
    cout << "  • String theory: Calabi-Yau manifolds\n";
    cout << "  • Statistical mechanics: partition functions\n";
    cout << "  • Cosmology: large number hypothesis\n\n";
    
    cout << "  BIOLOGY:\n";
    cout << "  • Population dynamics: Fibonacci growth\n";
    cout << "  • Genetic algorithms: optimization\n";
    cout << "  • Neural networks: activation patterns\n";
    cout << "  • Evolutionary game theory\n\n";
    
    cout << "  ECONOMICS:\n";
    cout << "  • Game theory: Nash equilibria\n";
    cout << "  • Optimization: linear programming\n";
    cout << "  • Financial modeling: Black-Scholes\n";
    cout << "  • Network effects: Metcalfe's law\n\n";
    
    cout << "  COMPUTER SCIENCE:\n";
    cout << "  • Algorithm design and analysis\n";
    cout << "  • Data structures: trees, graphs\n";
    cout << "  • Artificial intelligence: neural networks\n";
    cout << "  • Quantum computing: superposition\n\n";
}

void generateExtendedAnalysis(long long number, const string& familyName, int position) {
    // Add extended analysis sections
    generateHistoricalContext(number);
    generateApplications(number);
    generatePhilosophicalReflections(number);
    generateAdvancedTopics(number);
    generateMathematicalConnections(number);
}

// ═══════════════════════════════════════════════════════════════════════════
// END OF RECIPROCAL INTEGER ANALYZER - COMPREHENSIVE NARRATIVE MEGA ADDON
// ═══════════════════════════════════════════════════════════════════════════
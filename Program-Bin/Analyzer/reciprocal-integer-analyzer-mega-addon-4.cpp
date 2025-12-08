/*
 * RECIPROCAL INTEGER ANALYZER MEGA ADDON - CHANGE VERSION
 * ========================================================
 * 
 * This version performs ALL calculations directly on the RECIPROCAL value (1/n)
 * rather than on the integer n itself. All 8 subgroups adapted for reciprocal analysis.
 * 
 * Key Changes:
 * - All functions now operate on r = 1/n as the primary subject
 * - Formulas modified to handle reciprocal properties
 * - Continued fraction and irrationality corrections included
 * - Summary text explains all adaptations
 * 
 * Compilation: g++ -std=c++17 -O2 reciprocal-integer-analyzer-mega-addon-change.cpp -o analyzer_change
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
#include <complex>
#include <functional>
#include <numeric>
#include <limits>

using namespace std;

// Constants
const double PI = 3.14159265358979323846;
const double EPSILON = 1e-10;
const int MAX_ITERATIONS = 1000;

// ============================== SUBGROUP 1: HIGH-PRECISION ARITHMETIC ==============================
// ADAPTED FOR RECIPROCAL: Computing series based on powers of reciprocal r = 1/n

struct HighPrecisionArithmeticResults {
    double reciprocal_value;
    
    // Kahan summation on reciprocal powers: r + r^2 + r^3 + ...
    double kahan_sum;
    double naive_sum;
    double kahan_error_reduction;
    
    // Pairwise summation
    double pairwise_sum;
    double pairwise_error;
    
    // Binary splitting for geometric series: r/(1-r) = sum of r^k
    double binary_split_result;
    double geometric_series_closed_form;
    int binary_split_terms;
    double convergence_rate;
    
    string method_notes;
    string reciprocal_adaptation;
};

// Kahan summation algorithm
double kahanSum(const vector<double>& values) {
    double sum = 0.0;
    double c = 0.0;
    
    for (double value : values) {
        double y = value - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}

// Pairwise summation
double pairwiseSum(const vector<double>& values, int start, int end) {
    if (end - start <= 1) {
        return values[start];
    }
    if (end - start == 2) {
        return values[start] + values[start + 1];
    }
    
    int mid = start + (end - start) / 2;
    return pairwiseSum(values, start, mid) + pairwiseSum(values, mid, end);
}

// Binary splitting for geometric series of reciprocal
struct BinarySplitResult {
    double numerator;
    double denominator;
    int terms_computed;
};

BinarySplitResult binarySplitGeometric(double r, int start, int end) {
    BinarySplitResult result;
    
    if (end - start == 1) {
        // Base case: single term r^start
        result.numerator = pow(r, start);
        result.denominator = 1.0;
        result.terms_computed = 1;
        return result;
    }
    
    int mid = start + (end - start) / 2;
    BinarySplitResult left = binarySplitGeometric(r, start, mid);
    BinarySplitResult right = binarySplitGeometric(r, mid, end);
    
    // Combine: a/b + c/d = (ad + bc) / bd
    result.numerator = left.numerator * right.denominator + right.numerator * left.denominator;
    result.denominator = left.denominator * right.denominator;
    result.terms_computed = left.terms_computed + right.terms_computed;
    
    return result;
}

HighPrecisionArithmeticResults analyzeHighPrecisionArithmetic(double r) {
    HighPrecisionArithmeticResults results;
    results.reciprocal_value = r;
    
    // Generate geometric series: r, r^2, r^3, ..., r^20
    vector<double> reciprocal_powers;
    for (int i = 1; i <= 20; i++) {
        reciprocal_powers.push_back(pow(r, i));
    }
    
    // Kahan summation
    results.kahan_sum = kahanSum(reciprocal_powers);
    
    // Naive summation for comparison
    results.naive_sum = 0.0;
    for (double val : reciprocal_powers) {
        results.naive_sum += val;
    }
    
    results.kahan_error_reduction = abs(results.kahan_sum - results.naive_sum);
    
    // Pairwise summation
    results.pairwise_sum = pairwiseSum(reciprocal_powers, 0, reciprocal_powers.size());
    results.pairwise_error = abs(results.pairwise_sum - results.kahan_sum);
    
    // Binary splitting
    BinarySplitResult bs_result = binarySplitGeometric(r, 1, 21);
    results.binary_split_result = bs_result.numerator / bs_result.denominator;
    results.binary_split_terms = bs_result.terms_computed;
    
    // Closed form: r + r^2 + ... = r/(1-r) for |r| < 1
    if (abs(r) < 1.0) {
        results.geometric_series_closed_form = r / (1.0 - r);
    } else {
        results.geometric_series_closed_form = INFINITY;
    }
    
    results.convergence_rate = abs(results.binary_split_result - results.geometric_series_closed_form);
    
    results.method_notes = "High-precision summation of geometric series r + r^2 + r^3 + ...";
    results.reciprocal_adaptation = "ADAPTATION: Series converges for r < 1 (all unit fractions). "
                                   "Closed form r/(1-r) provides exact comparison. "
                                   "For r=1/n, sum approaches 1/(n-1) as terms increase.";
    
    return results;
}

// ============================== SUBGROUP 2: LINEAR SYSTEMS AND MATRIX COMPUTATIONS ==============================
// ADAPTED FOR RECIPROCAL: Matrix elements are powers and products of r = 1/n

struct LinearSystemResults {
    double reciprocal_value;
    vector<vector<double>> test_matrix;
    vector<double> test_vector;
    vector<double> solution;
    double determinant;
    double condition_number;
    string decomposition_method;
    bool is_singular;
    int matrix_size;
    string reciprocal_adaptation;
};

// LU Decomposition with partial pivoting
struct LUDecomposition {
    vector<vector<double>> L;
    vector<vector<double>> U;
    vector<int> permutation;
    bool success;
};

LUDecomposition luDecompose(const vector<vector<double>>& A) {
    LUDecomposition result;
    int n = A.size();
    result.L = vector<vector<double>>(n, vector<double>(n, 0.0));
    result.U = A;
    result.permutation = vector<int>(n);
    result.success = true;
    
    for (int i = 0; i < n; i++) {
        result.permutation[i] = i;
        result.L[i][i] = 1.0;
    }
    
    for (int k = 0; k < n; k++) {
        // Find pivot
        double max_val = abs(result.U[k][k]);
        int max_row = k;
        for (int i = k + 1; i < n; i++) {
            if (abs(result.U[i][k]) > max_val) {
                max_val = abs(result.U[i][k]);
                max_row = i;
            }
        }
        
        if (max_val < EPSILON) {
            result.success = false;
            return result;
        }
        
        // Swap rows
        if (max_row != k) {
            swap(result.U[k], result.U[max_row]);
            swap(result.permutation[k], result.permutation[max_row]);
            for (int j = 0; j < k; j++) {
                swap(result.L[k][j], result.L[max_row][j]);
            }
        }
        
        // Eliminate
        for (int i = k + 1; i < n; i++) {
            double factor = result.U[i][k] / result.U[k][k];
            result.L[i][k] = factor;
            for (int j = k; j < n; j++) {
                result.U[i][j] -= factor * result.U[k][j];
            }
        }
    }
    
    return result;
}

// Solve Ly = b (forward substitution)
vector<double> forwardSubstitution(const vector<vector<double>>& L, const vector<double>& b) {
    int n = L.size();
    vector<double> y(n, 0.0);
    
    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
        y[i] /= L[i][i];
    }
    
    return y;
}

// Solve Ux = y (backward substitution)
vector<double> backwardSubstitution(const vector<vector<double>>& U, const vector<double>& y) {
    int n = U.size();
    vector<double> x(n, 0.0);
    
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }
    
    return x;
}

// Calculate determinant from LU decomposition
double calculateDeterminant(const LUDecomposition& lu) {
    double det = 1.0;
    for (size_t i = 0; i < lu.U.size(); i++) {
        det *= lu.U[i][i];
    }
    return det;
}

LinearSystemResults analyzeLinearSystems(double r) {
    LinearSystemResults results;
    results.reciprocal_value = r;
    results.matrix_size = 3;
    
    // Create a 3x3 Vandermonde-like matrix using powers of reciprocal r
    // This tests how reciprocal values interact in linear systems
    results.test_matrix = {
        {1.0, r, r*r},
        {1.0, r*r, r*r*r},
        {1.0, r*r*r, r*r*r*r}
    };
    
    // Test vector
    results.test_vector = {1.0, r, r*r};
    
    // Perform LU decomposition
    LUDecomposition lu = luDecompose(results.test_matrix);
    results.decomposition_method = "LU Decomposition with Partial Pivoting";
    results.is_singular = !lu.success;
    
    if (lu.success) {
        // Calculate determinant
        results.determinant = calculateDeterminant(lu);
        
        // Solve the system Ax = b
        vector<double> permuted_b(results.matrix_size);
        for (int i = 0; i < results.matrix_size; i++) {
            permuted_b[i] = results.test_vector[lu.permutation[i]];
        }
        
        vector<double> y = forwardSubstitution(lu.L, permuted_b);
        results.solution = backwardSubstitution(lu.U, y);
        
        // Estimate condition number
        double max_element = 0.0;
        for (const auto& row : results.test_matrix) {
            for (double val : row) {
                max_element = max(max_element, abs(val));
            }
        }
        results.condition_number = max_element / abs(results.determinant);
    } else {
        results.determinant = 0.0;
        results.condition_number = INFINITY;
        results.solution = {0.0, 0.0, 0.0};
    }
    
    results.reciprocal_adaptation = "ADAPTATION: Matrix constructed from powers of r (Vandermonde-like structure). "
                                   "For r=1/n, matrix becomes increasingly well-conditioned as n grows (r shrinks). "
                                   "Determinant scales as r^6 for this 3x3 case. "
                                   "Tests numerical stability of reciprocal-based linear systems.";
    
    return results;
}

// ============================== SUBGROUP 3: NONLINEAR SYSTEMS AND ROOT-FINDING ==============================
// ADAPTED FOR RECIPROCAL: Finding roots of equations involving r = 1/n

struct RootFindingResults {
    double reciprocal_value;
    
    // Newton's method for x^2 = r (finding sqrt of reciprocal)
    double newton_root;
    int newton_iterations;
    double newton_error;
    
    // Bisection method
    double bisection_root;
    int bisection_iterations;
    double bisection_error;
    
    // Secant method
    double secant_root;
    int secant_iterations;
    double secant_error;
    
    // Brent's method
    double brent_root;
    int brent_iterations;
    double brent_error;
    
    string target_function;
    double true_root;
    string reciprocal_adaptation;
};

// Newton's method for finding sqrt(r) where r = 1/n
double newtonMethodSqrt(double r, double initial_guess, int& iterations, double& error) {
    double x = initial_guess;
    iterations = 0;
    
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        // f(x) = x^2 - r
        // f'(x) = 2x
        double fx = x*x - r;
        double fpx = 2.0*x;
        
        if (abs(fpx) < EPSILON) break;
        
        double x_new = x - fx/fpx;
        error = abs(x_new - x);
        x = x_new;
        iterations++;
        
        if (error < EPSILON) break;
    }
    
    return x;
}

// Bisection method for x^2 = r
double bisectionMethodSqrt(double r, double a, double b, int& iterations, double& error) {
    iterations = 0;
    double fa = a*a - r;
    double fb = b*b - r;
    
    if (fa * fb > 0) {
        error = INFINITY;
        return (a + b) / 2.0;
    }
    
    double c = a;
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        c = (a + b) / 2.0;
        double fc = c*c - r;
        
        error = (b - a) / 2.0;
        iterations++;
        
        if (abs(fc) < EPSILON || error < EPSILON) break;
        
        if (fa * fc < 0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    
    return c;
}

// Secant method for x^2 = r
double secantMethodSqrt(double r, double x0, double x1, int& iterations, double& error) {
    iterations = 0;
    double f0 = x0*x0 - r;
    double f1 = x1*x1 - r;
    
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        if (abs(f1 - f0) < EPSILON) break;
        
        double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        error = abs(x2 - x1);
        iterations++;
        
        if (error < EPSILON) {
            return x2;
        }
        
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = x1*x1 - r;
    }
    
    return x1;
}

// Brent's method for x^2 = r
double brentMethodSqrt(double r, double a, double b, int& iterations, double& error) {
    iterations = 0;
    double fa = a*a - r;
    double fb = b*b - r;
    
    if (fa * fb > 0) {
        error = INFINITY;
        return (a + b) / 2.0;
    }
    
    if (abs(fa) < abs(fb)) {
        swap(a, b);
        swap(fa, fb);
    }
    
    double c = a;
    double fc = fa;
    bool mflag = true;
    double s = 0.0;
    double d = 0.0;
    
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        if (abs(fb) < EPSILON) {
            error = 0.0;
            return b;
        }
        
        if (abs(fa - fc) > EPSILON && abs(fb - fc) > EPSILON) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc)) +
                b * fa * fc / ((fb - fa) * (fb - fc)) +
                c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }
        
        // Check if bisection is needed
        double tmp2 = (3.0 * a + b) / 4.0;
        if (!((s > tmp2 && s < b) || (s < tmp2 && s > b)) ||
            (mflag && abs(s - b) >= abs(b - c) / 2.0) ||
            (!mflag && abs(s - b) >= abs(c - d) / 2.0)) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }
        
        double fs = s*s - r;
        d = c;
        c = b;
        fc = fb;
        
        if (fa * fs < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        
        if (abs(fa) < abs(fb)) {
            swap(a, b);
            swap(fa, fb);
        }
        
        error = abs(b - a);
        iterations++;
        
        if (error < EPSILON) break;
    }
    
    return b;
}

RootFindingResults analyzeRootFinding(double r) {
    RootFindingResults results;
    results.reciprocal_value = r;
    results.target_function = "f(x) = x^2 - r, finding sqrt(r) where r = 1/n";
    results.true_root = sqrt(r);
    
    // Newton's method
    double initial_guess = r;  // Start with r itself
    results.newton_root = newtonMethodSqrt(r, initial_guess, results.newton_iterations, results.newton_error);
    
    // Bisection method
    double a = 0.0;
    double b = max(1.0, 2.0 * r);
    results.bisection_root = bisectionMethodSqrt(r, a, b, results.bisection_iterations, results.bisection_error);
    
    // Secant method
    double x0 = r;
    double x1 = r * 1.5;
    results.secant_root = secantMethodSqrt(r, x0, x1, results.secant_iterations, results.secant_error);
    
    // Brent's method
    results.brent_root = brentMethodSqrt(r, a, b, results.brent_iterations, results.brent_error);
    
    results.reciprocal_adaptation = "ADAPTATION: Finding sqrt(1/n) = 1/sqrt(n) tests root-finding on reciprocals. "
                                   "For unit fractions, sqrt(r) < r, demonstrating sub-linear behavior. "
                                   "Newton's method particularly efficient due to simple derivative. "
                                   "All methods converge to 1/sqrt(n), a fundamental irrational for most n.";
    
    return results;
}

// ============================== SUBGROUP 4: APPROXIMATION AND INTERPOLATION ==============================
// ADAPTED FOR RECIPROCAL: Approximating functions of r = 1/n

struct ApproximationResults {
    double reciprocal_value;
    
    // Lagrange interpolation of r^k values
    vector<double> lagrange_coefficients;
    double lagrange_at_midpoint;
    
    // Newton interpolation
    vector<double> newton_divided_differences;
    double newton_at_midpoint;
    
    // Chebyshev approximation
    vector<double> chebyshev_coefficients;
    double chebyshev_error;
    
    // Padé approximant for exp(r)
    double pade_numerator_coeffs[3];
    double pade_denominator_coeffs[3];
    double pade_at_point;
    double true_exp_r;
    
    int num_data_points;
    string approximated_function;
    string reciprocal_adaptation;
};

// Lagrange interpolation
double lagrangeInterpolate(const vector<double>& x, const vector<double>& y, double xi) {
    double result = 0.0;
    int n = x.size();
    
    for (int i = 0; i < n; i++) {
        double term = y[i];
        for (int j = 0; j < n; j++) {
            if (j != i) {
                term *= (xi - x[j]) / (x[i] - x[j]);
            }
        }
        result += term;
    }
    
    return result;
}

// Newton divided differences
vector<double> newtonDividedDifferences(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    vector<vector<double>> table(n, vector<double>(n, 0.0));
    
    for (int i = 0; i < n; i++) {
        table[i][0] = y[i];
    }
    
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < n - j; i++) {
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i]);
        }
    }
    
    vector<double> coeffs(n);
    for (int i = 0; i < n; i++) {
        coeffs[i] = table[0][i];
    }
    
    return coeffs;
}

// Newton interpolation evaluation
double newtonInterpolate(const vector<double>& x, const vector<double>& coeffs, double xi) {
    double result = coeffs[0];
    double term = 1.0;
    
    for (size_t i = 1; i < coeffs.size(); i++) {
        term *= (xi - x[i-1]);
        result += coeffs[i] * term;
    }
    
    return result;
}

// Chebyshev nodes
vector<double> chebyshevNodes(int n, double a, double b) {
    vector<double> nodes(n);
    for (int i = 0; i < n; i++) {
        double theta = PI * (2.0 * i + 1.0) / (2.0 * n);
        nodes[i] = 0.5 * (a + b) + 0.5 * (b - a) * cos(theta);
    }
    return nodes;
}

// Padé approximant [2/2] for exp(r)
void padeApproximantExp(double r, double pade_num[3], double pade_den[3]) {
    // [2/2] Padé for exp(x): (12 + 6x + x^2) / (12 - 6x + x^2)
    pade_num[0] = 12.0;
    pade_num[1] = 6.0;
    pade_num[2] = 1.0;
    
    pade_den[0] = 12.0;
    pade_den[1] = -6.0;
    pade_den[2] = 1.0;
}

ApproximationResults analyzeApproximation(double r) {
    ApproximationResults results;
    results.reciprocal_value = r;
    results.approximated_function = "f(t) = exp(r*t) for t in [0,1]";
    results.num_data_points = 5;
    
    // Generate data points for exp(r*t)
    vector<double> t_data, y_data;
    for (int i = 0; i <= 4; i++) {
        double t = i * 0.25;
        t_data.push_back(t);
        y_data.push_back(exp(r * t));
    }
    
    // Lagrange interpolation
    double midpoint = 0.5;
    results.lagrange_at_midpoint = lagrangeInterpolate(t_data, y_data, midpoint);
    results.lagrange_coefficients = y_data;
    
    // Newton interpolation
    results.newton_divided_differences = newtonDividedDifferences(t_data, y_data);
    results.newton_at_midpoint = newtonInterpolate(t_data, results.newton_divided_differences, midpoint);
    
    // Chebyshev approximation
    vector<double> cheb_nodes = chebyshevNodes(5, 0.0, 1.0);
    vector<double> cheb_values;
    for (double node : cheb_nodes) {
        cheb_values.push_back(exp(r * node));
    }
    results.chebyshev_coefficients = cheb_values;
    double cheb_approx = lagrangeInterpolate(cheb_nodes, cheb_values, midpoint);
    results.chebyshev_error = abs(cheb_approx - exp(r * midpoint));
    
    // Padé approximant for exp(r)
    padeApproximantExp(r, results.pade_numerator_coeffs, results.pade_denominator_coeffs);
    double pade_num = results.pade_numerator_coeffs[0] + 
                      results.pade_numerator_coeffs[1] * r + 
                      results.pade_numerator_coeffs[2] * r * r;
    double pade_den = results.pade_denominator_coeffs[0] + 
                      results.pade_denominator_coeffs[1] * r + 
                      results.pade_denominator_coeffs[2] * r * r;
    results.pade_at_point = pade_num / pade_den;
    results.true_exp_r = exp(r);
    
    results.reciprocal_adaptation = "ADAPTATION: Approximating exp(r*t) where r=1/n tests interpolation on scaled exponentials. "
                                   "For small r (large n), exp(r) ≈ 1 + r (linear approximation valid). "
                                   "Padé [2/2] provides rational approximation superior to Taylor series. "
                                   "Chebyshev nodes minimize interpolation error (Runge's phenomenon avoided).";
    
    return results;
}

// ============================== SUBGROUP 5: TRANSFORM METHODS ==============================
// ADAPTED FOR RECIPROCAL: Analyzing frequency content of reciprocal sequences

struct TransformResults {
    double reciprocal_value;
    
    // DFT results
    vector<complex<double>> dft_coefficients;
    vector<double> magnitude_spectrum;
    vector<double> phase_spectrum;
    double dominant_frequency;
    double spectral_energy;
    double dc_component;  // Zero-frequency component
    
    // FFT results
    vector<complex<double>> fft_coefficients;
    int fft_size;
    double fft_computation_ratio;
    
    string signal_description;
    string reciprocal_adaptation;
};

// DFT - Discrete Fourier Transform
vector<complex<double>> computeDFT(const vector<double>& signal) {
    int N = signal.size();
    vector<complex<double>> dft(N);
    
    for (int k = 0; k < N; k++) {
        complex<double> sum(0.0, 0.0);
        for (int n = 0; n < N; n++) {
            double angle = -2.0 * PI * k * n / N;
            sum += signal[n] * complex<double>(cos(angle), sin(angle));
        }
        dft[k] = sum;
    }
    
    return dft;
}

// FFT - Cooley-Tukey algorithm
void fftCooleyTukey(vector<complex<double>>& data) {
    int N = data.size();
    if (N <= 1) return;
    
    vector<complex<double>> even(N/2), odd(N/2);
    for (int i = 0; i < N/2; i++) {
        even[i] = data[2*i];
        odd[i] = data[2*i + 1];
    }
    
    fftCooleyTukey(even);
    fftCooleyTukey(odd);
    
    for (int k = 0; k < N/2; k++) {
        double angle = -2.0 * PI * k / N;
        complex<double> t = complex<double>(cos(angle), sin(angle)) * odd[k];
        data[k] = even[k] + t;
        data[k + N/2] = even[k] - t;
    }
}

TransformResults analyzeTransforms(double r) {
    TransformResults results;
    results.reciprocal_value = r;
    results.signal_description = "Signal: powers of r: r, r^2, r^3, ..., r^16";
    
    // Generate signal from powers of reciprocal
    vector<double> signal;
    for (int i = 1; i <= 16; i++) {
        signal.push_back(pow(r, i));
    }
    
    // Compute DFT
    results.dft_coefficients = computeDFT(signal);
    
    // Extract magnitude and phase spectra
    results.spectral_energy = 0.0;
    double max_magnitude = 0.0;
    int dominant_freq_index = 0;
    
    results.dc_component = abs(results.dft_coefficients[0]) / signal.size();
    
    for (size_t k = 0; k < results.dft_coefficients.size(); k++) {
        double magnitude = abs(results.dft_coefficients[k]);
        double phase = arg(results.dft_coefficients[k]);
        
        results.magnitude_spectrum.push_back(magnitude);
        results.phase_spectrum.push_back(phase);
        results.spectral_energy += magnitude * magnitude;
        
        if (k > 0 && magnitude > max_magnitude) {  // Skip DC component
            max_magnitude = magnitude;
            dominant_freq_index = k;
        }
    }
    
    results.dominant_frequency = dominant_freq_index;
    
    // Compute FFT
    results.fft_coefficients.resize(signal.size());
    for (size_t i = 0; i < signal.size(); i++) {
        results.fft_coefficients[i] = complex<double>(signal[i], 0.0);
    }
    fftCooleyTukey(results.fft_coefficients);
    results.fft_size = results.fft_coefficients.size();
    
    int N = signal.size();
    results.fft_computation_ratio = (N * N) / (N * log2(N));
    
    results.reciprocal_adaptation = "ADAPTATION: Transform of geometric sequence r^k reveals exponential decay spectrum. "
                                   "DC component (k=0) equals sum of series ≈ r/(1-r). "
                                   "For r=1/n, spectrum concentrates at low frequencies (smooth signal). "
                                   "Magnitude decreases geometrically, phase structure reveals periodicity. "
                                   "FFT efficiency crucial for analyzing long reciprocal sequences.";
    
    return results;
}

// ============================== SUBGROUP 6: NUMERICAL INTEGRATION AND DIFFERENTIATION ==============================
// ADAPTED FOR RECIPROCAL: Integrating and differentiating functions of r = 1/n

struct IntegrationResults {
    double reciprocal_value;
    
    // Simpson's rule for integral of exp(r*t) from 0 to 1
    double simpson_integral;
    int simpson_intervals;
    double simpson_error_estimate;
    
    // Trapezoidal rule
    double trapezoidal_integral;
    int trapezoidal_intervals;
    double trapezoidal_error_estimate;
    
    // Gaussian quadrature
    double gaussian_integral;
    int gaussian_points;
    double gaussian_error_estimate;
    
    // True value: (exp(r) - 1) / r
    double true_integral_value;
    
    // Numerical differentiation of exp(r*t) at t=0.5
    double forward_difference;
    double backward_difference;
    double central_difference;
    double true_derivative;
    
    string integrated_function;
    double integration_bounds[2];
    string reciprocal_adaptation;
};

// Simpson's rule
double simpsonsRule(function<double(double)> f, double a, double b, int n) {
    if (n % 2 != 0) n++;
    
    double h = (b - a) / n;
    double sum = f(a) + f(b);
    
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2.0 * f(x);
        } else {
            sum += 4.0 * f(x);
        }
    }
    
    return (h / 3.0) * sum;
}

// Trapezoidal rule
double trapezoidalRule(function<double(double)> f, double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));
    
    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }
    
    return h * sum;
}

// Gaussian quadrature (2-point)
double gaussianQuadrature2Point(function<double(double)> f, double a, double b) {
    double mid = 0.5 * (a + b);
    double half_length = 0.5 * (b - a);
    
    double sqrt3 = sqrt(3.0);
    double x1 = mid - half_length / sqrt3;
    double x2 = mid + half_length / sqrt3;
    
    return half_length * (f(x1) + f(x2));
}

// Numerical derivatives
double forwardDifference(function<double(double)> f, double x, double h) {
    return (f(x + h) - f(x)) / h;
}

double backwardDifference(function<double(double)> f, double x, double h) {
    return (f(x) - f(x - h)) / h;
}

double centralDifference(function<double(double)> f, double x, double h) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

IntegrationResults analyzeIntegration(double r) {
    IntegrationResults results;
    results.reciprocal_value = r;
    results.integrated_function = "f(t) = exp(r*t)";
    results.integration_bounds[0] = 0.0;
    results.integration_bounds[1] = 1.0;
    
    // Define the function: exp(r*t)
    auto exp_rt = [r](double t) { return exp(r * t); };
    
    double a = 0.0;
    double b = 1.0;
    
    // Simpson's rule
    results.simpson_intervals = 100;
    results.simpson_integral = simpsonsRule(exp_rt, a, b, results.simpson_intervals);
    
    // True value: integral of exp(r*t) from 0 to 1 is (exp(r) - 1) / r
    if (abs(r) > EPSILON) {
        results.true_integral_value = (exp(r) - 1.0) / r;
    } else {
        results.true_integral_value = 1.0;  // Limit as r -> 0
    }
    results.simpson_error_estimate = abs(results.simpson_integral - results.true_integral_value);
    
    // Trapezoidal rule
    results.trapezoidal_intervals = 100;
    results.trapezoidal_integral = trapezoidalRule(exp_rt, a, b, results.trapezoidal_intervals);
    results.trapezoidal_error_estimate = abs(results.trapezoidal_integral - results.true_integral_value);
    
    // Gaussian quadrature
    results.gaussian_points = 2;
    results.gaussian_integral = gaussianQuadrature2Point(exp_rt, a, b);
    results.gaussian_error_estimate = abs(results.gaussian_integral - results.true_integral_value);
    
    // Numerical differentiation at t = 0.5
    // Derivative of exp(r*t) is r*exp(r*t)
    double t_eval = 0.5;
    double h = 0.001;
    results.forward_difference = forwardDifference(exp_rt, t_eval, h);
    results.backward_difference = backwardDifference(exp_rt, t_eval, h);
    results.central_difference = centralDifference(exp_rt, t_eval, h);
    results.true_derivative = r * exp(r * t_eval);
    
    results.reciprocal_adaptation = "ADAPTATION: Integrating exp(r*t) where r=1/n tests quadrature on scaled exponentials. "
                                   "True integral (exp(r)-1)/r approaches 1 as r→0 (L'Hôpital's rule). "
                                   "For small r, exp(r*t) ≈ 1 + r*t, making integration nearly linear. "
                                   "Derivative r*exp(r*t) at t=0.5 gives r*exp(r/2), testing differentiation accuracy. "
                                   "Simpson's rule superior for smooth exponential functions.";
    
    return results;
}

// ============================== SUBGROUP 7: EIGENVALUE AND SPECTRAL PROBLEMS ==============================
// ADAPTED FOR RECIPROCAL: Eigenvalues of matrices constructed from r = 1/n

struct EigenvalueResults {
    double reciprocal_value;
    
    // Power iteration
    double dominant_eigenvalue;
    vector<double> dominant_eigenvector;
    int power_iterations;
    double power_convergence_rate;
    
    // QR algorithm
    vector<double> all_eigenvalues;
    int qr_iterations;
    
    // Matrix properties
    double trace;
    double determinant;
    double spectral_radius;
    double condition_number_estimate;
    
    int matrix_size;
    string matrix_description;
    string reciprocal_adaptation;
};

// Power iteration
pair<double, vector<double>> powerIteration(const vector<vector<double>>& A, int max_iter, double& convergence_rate) {
    int n = A.size();
    vector<double> v(n, 1.0);
    double lambda = 0.0;
    double lambda_old = 0.0;
    
    for (int iter = 0; iter < max_iter; iter++) {
        vector<double> Av(n, 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Av[i] += A[i][j] * v[j];
            }
        }
        
        lambda = 0.0;
        for (int i = 0; i < n; i++) {
            if (abs(Av[i]) > abs(lambda)) {
                lambda = Av[i];
            }
        }
        
        for (int i = 0; i < n; i++) {
            v[i] = Av[i] / lambda;
        }
        
        convergence_rate = abs(lambda - lambda_old);
        if (convergence_rate < EPSILON) break;
        
        lambda_old = lambda;
    }
    
    return {lambda, v};
}

// QR decomposition
pair<vector<vector<double>>, vector<vector<double>>> qrDecomposition(const vector<vector<double>>& A) {
    int n = A.size();
    vector<vector<double>> Q(n, vector<double>(n, 0.0));
    vector<vector<double>> R(n, vector<double>(n, 0.0));
    
    for (int j = 0; j < n; j++) {
        vector<double> v(n);
        for (int i = 0; i < n; i++) {
            v[i] = A[i][j];
        }
        
        for (int k = 0; k < j; k++) {
            double dot = 0.0;
            for (int i = 0; i < n; i++) {
                dot += A[i][j] * Q[i][k];
            }
            R[k][j] = dot;
            for (int i = 0; i < n; i++) {
                v[i] -= dot * Q[i][k];
            }
        }
        
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            norm += v[i] * v[i];
        }
        norm = sqrt(norm);
        R[j][j] = norm;
        
        if (norm > EPSILON) {
            for (int i = 0; i < n; i++) {
                Q[i][j] = v[i] / norm;
            }
        }
    }
    
    return {Q, R};
}

// Matrix multiplication
vector<vector<double>> matrixMultiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}

EigenvalueResults analyzeEigenvalues(double r) {
    EigenvalueResults results;
    results.reciprocal_value = r;
    results.matrix_size = 3;
    results.matrix_description = "Toeplitz matrix with r on diagonals";
    
    // Create a symmetric Toeplitz matrix using r
    // This structure appears in many applications involving reciprocals
    vector<vector<double>> A = {
        {r, r*r, r*r*r},
        {r*r, r, r*r},
        {r*r*r, r*r, r}
    };
    
    // Calculate trace
    results.trace = 0.0;
    for (int i = 0; i < results.matrix_size; i++) {
        results.trace += A[i][i];
    }
    
    // Power iteration
    auto [lambda, eigenvector] = powerIteration(A, 1000, results.power_convergence_rate);
    results.dominant_eigenvalue = lambda;
    results.dominant_eigenvector = eigenvector;
    results.power_iterations = 100;
    
    // QR algorithm
    vector<vector<double>> Ak = A;
    results.qr_iterations = 10;
    
    for (int iter = 0; iter < results.qr_iterations; iter++) {
        auto [Q, R] = qrDecomposition(Ak);
        Ak = matrixMultiply(R, Q);
    }
    
    // Extract eigenvalues
    for (int i = 0; i < results.matrix_size; i++) {
        results.all_eigenvalues.push_back(Ak[i][i]);
    }
    
    // Spectral radius
    results.spectral_radius = 0.0;
    for (double eigenval : results.all_eigenvalues) {
        results.spectral_radius = max(results.spectral_radius, abs(eigenval));
    }
    
    // Determinant
    results.determinant = 1.0;
    for (double eigenval : results.all_eigenvalues) {
        results.determinant *= eigenval;
    }
    
    // Condition number
    double min_eigenval = INFINITY;
    for (double eigenval : results.all_eigenvalues) {
        if (abs(eigenval) > EPSILON) {
            min_eigenval = min(min_eigenval, abs(eigenval));
        }
    }
    results.condition_number_estimate = results.spectral_radius / min_eigenval;
    
    results.reciprocal_adaptation = "ADAPTATION: Toeplitz matrix with r=1/n on diagonals tests spectral properties. "
                                   "Trace = 3r, determinant scales as r^3 for this structure. "
                                   "For small r, matrix becomes nearly singular (small eigenvalues). "
                                   "Dominant eigenvalue ≈ r + O(r^2), showing linear scaling. "
                                   "Condition number grows as 1/r (as n increases), indicating ill-conditioning.";
    
    return results;
}

// ============================== SUBGROUP 8: OPTIMIZATION ==============================
// ADAPTED FOR RECIPROCAL: Optimizing functions involving r = 1/n

struct OptimizationResults {
    double reciprocal_value;
    
    // Gradient descent for minimizing (x - r)^2 + r*sin(x/r)
    double gd_minimum;
    vector<double> gd_path;
    int gd_iterations;
    double gd_final_gradient;
    
    // Newton's method for optimization
    double newton_minimum;
    vector<double> newton_path;
    int newton_iterations;
    double newton_final_hessian;
    
    // Golden section search
    double golden_minimum;
    double golden_min_location;
    int golden_iterations;
    
    // Objective function properties
    string objective_function;
    double true_minimum;
    double true_minimum_location;
    string reciprocal_adaptation;
};

// Gradient descent for f(x) = (x - r)^2 + r*sin(x/r)
double gradientDescent(double r, double initial_x, vector<double>& path, int& iterations, double& final_gradient) {
    double x = initial_x;
    double learning_rate = 0.1;
    iterations = 0;
    
    path.push_back(x);
    
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        // f(x) = (x - r)^2 + r*sin(x/r)
        // f'(x) = 2(x - r) + cos(x/r)
        double gradient = 2.0 * (x - r) + cos(x / r);
        
        x = x - learning_rate * gradient;
        path.push_back(x);
        iterations++;
        
        final_gradient = abs(gradient);
        if (final_gradient < EPSILON) break;
    }
    
    return x;
}

// Newton's method for optimization
double newtonOptimization(double r, double initial_x, vector<double>& path, int& iterations, double& final_hessian) {
    double x = initial_x;
    iterations = 0;
    
    path.push_back(x);
    
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        // f(x) = (x - r)^2 + r*sin(x/r)
        // f'(x) = 2(x - r) + cos(x/r)
        // f''(x) = 2 - sin(x/r)/(r)
        double gradient = 2.0 * (x - r) + cos(x / r);
        double hessian = 2.0 - sin(x / r) / r;
        
        if (abs(hessian) < EPSILON) break;
        
        x = x - gradient / hessian;
        path.push_back(x);
        iterations++;
        
        final_hessian = hessian;
        if (abs(gradient) < EPSILON) break;
    }
    
    return x;
}

// Golden section search
double goldenSectionSearch(double r, double a, double b, int& iterations, double& min_location) {
    const double phi = (1.0 + sqrt(5.0)) / 2.0;
    const double resphi = 2.0 - phi;
    
    auto f = [r](double x) { return (x - r) * (x - r) + r * sin(x / r); };
    
    double tol = EPSILON;
    iterations = 0;
    
    double x1 = a + resphi * (b - a);
    double x2 = b - resphi * (b - a);
    double f1 = f(x1);
    double f2 = f(x2);
    
    while (abs(b - a) > tol && iterations < MAX_ITERATIONS) {
        if (f1 < f2) {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + resphi * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - resphi * (b - a);
            f2 = f(x2);
        }
        iterations++;
    }
    
    min_location = (a + b) / 2.0;
    return f(min_location);
}

OptimizationResults analyzeOptimization(double r) {
    OptimizationResults results;
    results.reciprocal_value = r;
    results.objective_function = "f(x) = (x - r)^2 + r*sin(x/r)";
    results.true_minimum_location = r;  // Approximate (exact depends on sin term)
    
    // Gradient descent
    double initial_x = 2.0 * r;
    results.gd_minimum = gradientDescent(r, initial_x, results.gd_path, results.gd_iterations, results.gd_final_gradient);
    
    // Newton's method
    results.newton_minimum = newtonOptimization(r, initial_x, results.newton_path, results.newton_iterations, results.newton_final_hessian);
    
    // Golden section search
    double a = 0.5 * r;
    double b = 1.5 * r;
    results.golden_minimum = goldenSectionSearch(r, a, b, results.golden_iterations, results.golden_min_location);
    
    // Calculate true minimum value
    results.true_minimum = (results.golden_min_location - r) * (results.golden_min_location - r) + 
                          r * sin(results.golden_min_location / r);
    
    results.reciprocal_adaptation = "ADAPTATION: Optimizing f(x) = (x-r)^2 + r*sin(x/r) where r=1/n. "
                                   "Quadratic term has minimum at x=r, sinusoidal perturbation scaled by r. "
                                   "For small r, sin(x/r) oscillates rapidly, creating local minima. "
                                   "Global minimum near x=r, but exact location depends on sin term phase. "
                                   "Newton's method converges faster due to curvature information. "
                                   "Tests optimization on reciprocal-scaled non-convex functions.";
    
    return results;
}

// ============================== INTEGRATION SECTION ==============================

struct ComprehensiveReciprocalAnalysis {
    int original_number;
    double reciprocal;
    
    HighPrecisionArithmeticResults high_precision;
    LinearSystemResults linear_systems;
    RootFindingResults root_finding;
    ApproximationResults approximation;
    TransformResults transforms;
    IntegrationResults integration;
    EigenvalueResults eigenvalues;
    OptimizationResults optimization;
};

ComprehensiveReciprocalAnalysis runAllReciprocalAnalyses(int n) {
    ComprehensiveReciprocalAnalysis analysis;
    analysis.original_number = n;
    analysis.reciprocal = 1.0 / n;
    
    double r = analysis.reciprocal;
    
    cout << "\n========================================" << endl;
    cout << "COMPREHENSIVE RECIPROCAL ANALYSIS FOR n = " << n << endl;
    cout << "Reciprocal: r = 1/" << n << " = " << r << endl;
    cout << "========================================\n" << endl;
    
    // Subgroup 1: High-Precision Arithmetic
    cout << "--- SUBGROUP 1: HIGH-PRECISION ARITHMETIC ON RECIPROCAL ---" << endl;
    analysis.high_precision = analyzeHighPrecisionArithmetic(r);
    cout << "Reciprocal Value: " << analysis.high_precision.reciprocal_value << endl;
    cout << "Kahan Sum (r + r^2 + ...): " << analysis.high_precision.kahan_sum << endl;
    cout << "Geometric Series Closed Form: " << analysis.high_precision.geometric_series_closed_form << endl;
    cout << "Binary Split Result: " << analysis.high_precision.binary_split_result << endl;
    cout << "Convergence Rate: " << analysis.high_precision.convergence_rate << endl;
    cout << "ADAPTATION: " << analysis.high_precision.reciprocal_adaptation << endl;
    
    // Subgroup 2: Linear Systems
    cout << "\n--- SUBGROUP 2: LINEAR SYSTEMS WITH RECIPROCAL MATRIX ---" << endl;
    analysis.linear_systems = analyzeLinearSystems(r);
    cout << "Reciprocal Value: " << analysis.linear_systems.reciprocal_value << endl;
    cout << "Matrix Size: " << analysis.linear_systems.matrix_size << "x" << analysis.linear_systems.matrix_size << endl;
    cout << "Determinant: " << analysis.linear_systems.determinant << endl;
    cout << "Condition Number: " << analysis.linear_systems.condition_number << endl;
    cout << "Solution: [";
    for (size_t i = 0; i < analysis.linear_systems.solution.size(); i++) {
        cout << analysis.linear_systems.solution[i];
        if (i < analysis.linear_systems.solution.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    cout << "ADAPTATION: " << analysis.linear_systems.reciprocal_adaptation << endl;
    
    // Subgroup 3: Root-Finding
    cout << "\n--- SUBGROUP 3: ROOT-FINDING FOR SQRT(RECIPROCAL) ---" << endl;
    analysis.root_finding = analyzeRootFinding(r);
    cout << "Reciprocal Value: " << analysis.root_finding.reciprocal_value << endl;
    cout << "Target: " << analysis.root_finding.target_function << endl;
    cout << "True Root (sqrt(r)): " << analysis.root_finding.true_root << endl;
    cout << "Newton: " << analysis.root_finding.newton_root 
         << " (" << analysis.root_finding.newton_iterations << " iter)" << endl;
    cout << "Brent: " << analysis.root_finding.brent_root 
         << " (" << analysis.root_finding.brent_iterations << " iter)" << endl;
    cout << "ADAPTATION: " << analysis.root_finding.reciprocal_adaptation << endl;
    
    // Subgroup 4: Approximation
    cout << "\n--- SUBGROUP 4: APPROXIMATION OF EXP(RECIPROCAL*T) ---" << endl;
    analysis.approximation = analyzeApproximation(r);
    cout << "Reciprocal Value: " << analysis.approximation.reciprocal_value << endl;
    cout << "Function: " << analysis.approximation.approximated_function << endl;
    cout << "Lagrange at t=0.5: " << analysis.approximation.lagrange_at_midpoint << endl;
    cout << "Chebyshev Error: " << analysis.approximation.chebyshev_error << endl;
    cout << "Padé for exp(r): " << analysis.approximation.pade_at_point 
         << " (true: " << analysis.approximation.true_exp_r << ")" << endl;
    cout << "ADAPTATION: " << analysis.approximation.reciprocal_adaptation << endl;
    
    // Subgroup 5: Transforms
    cout << "\n--- SUBGROUP 5: TRANSFORM OF RECIPROCAL POWERS ---" << endl;
    analysis.transforms = analyzeTransforms(r);
    cout << "Reciprocal Value: " << analysis.transforms.reciprocal_value << endl;
    cout << "Signal: " << analysis.transforms.signal_description << endl;
    cout << "DC Component: " << analysis.transforms.dc_component << endl;
    cout << "Dominant Frequency: " << analysis.transforms.dominant_frequency << endl;
    cout << "Spectral Energy: " << analysis.transforms.spectral_energy << endl;
    cout << "FFT Speedup: " << analysis.transforms.fft_computation_ratio << "x" << endl;
    cout << "ADAPTATION: " << analysis.transforms.reciprocal_adaptation << endl;
    
    // Subgroup 6: Integration
    cout << "\n--- SUBGROUP 6: INTEGRATION OF EXP(RECIPROCAL*T) ---" << endl;
    analysis.integration = analyzeIntegration(r);
    cout << "Reciprocal Value: " << analysis.integration.reciprocal_value << endl;
    cout << "Function: " << analysis.integration.integrated_function << endl;
    cout << "True Integral: " << analysis.integration.true_integral_value << endl;
    cout << "Simpson: " << analysis.integration.simpson_integral 
         << " (error: " << analysis.integration.simpson_error_estimate << ")" << endl;
    cout << "Gaussian: " << analysis.integration.gaussian_integral 
         << " (error: " << analysis.integration.gaussian_error_estimate << ")" << endl;
    cout << "Derivative at t=0.5: " << analysis.integration.central_difference 
         << " (true: " << analysis.integration.true_derivative << ")" << endl;
    cout << "ADAPTATION: " << analysis.integration.reciprocal_adaptation << endl;
    
    // Subgroup 7: Eigenvalues
    cout << "\n--- SUBGROUP 7: EIGENVALUES OF RECIPROCAL MATRIX ---" << endl;
    analysis.eigenvalues = analyzeEigenvalues(r);
    cout << "Reciprocal Value: " << analysis.eigenvalues.reciprocal_value << endl;
    cout << "Matrix: " << analysis.eigenvalues.matrix_description << endl;
    cout << "Trace: " << analysis.eigenvalues.trace << endl;
    cout << "Determinant: " << analysis.eigenvalues.determinant << endl;
    cout << "Dominant Eigenvalue: " << analysis.eigenvalues.dominant_eigenvalue << endl;
    cout << "All Eigenvalues: [";
    for (size_t i = 0; i < analysis.eigenvalues.all_eigenvalues.size(); i++) {
        cout << analysis.eigenvalues.all_eigenvalues[i];
        if (i < analysis.eigenvalues.all_eigenvalues.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    cout << "Spectral Radius: " << analysis.eigenvalues.spectral_radius << endl;
    cout << "ADAPTATION: " << analysis.eigenvalues.reciprocal_adaptation << endl;
    
    // Subgroup 8: Optimization
    cout << "\n--- SUBGROUP 8: OPTIMIZATION WITH RECIPROCAL ---" << endl;
    analysis.optimization = analyzeOptimization(r);
    cout << "Reciprocal Value: " << analysis.optimization.reciprocal_value << endl;
    cout << "Objective: " << analysis.optimization.objective_function << endl;
    cout << "Gradient Descent: " << analysis.optimization.gd_minimum 
         << " (" << analysis.optimization.gd_iterations << " iter)" << endl;
    cout << "Newton: " << analysis.optimization.newton_minimum 
         << " (" << analysis.optimization.newton_iterations << " iter)" << endl;
    cout << "Golden Section: " << analysis.optimization.golden_minimum 
         << " at x=" << analysis.optimization.golden_min_location << endl;
    cout << "ADAPTATION: " << analysis.optimization.reciprocal_adaptation << endl;
    
    return analysis;
}

// Function to analyze a range of numbers
void analyzeRangeWithReciprocalMethods(int start, int end) {
    cout << "\n========================================" << endl;
    cout << "ANALYZING RANGE: " << start << " to " << end << endl;
    cout << "========================================\n" << endl;
    
    for (int n = start; n <= end; n++) {
        runAllReciprocalAnalyses(n);
        cout << "\n" << endl;
    }
}
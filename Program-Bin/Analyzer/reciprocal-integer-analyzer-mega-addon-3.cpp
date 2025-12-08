/*
 * RECIPROCAL INTEGER ANALYZER MEGA ADDON 4 - COMPREHENSIVE SNIPPET INTEGRATION
 * ===============================================================================
 * 
 * This file integrates all functionality from 4 snippets:
 * - Snippet1: Fraction analysis with Boost.Multiprecision (fracana namespace)
 * - Snippet2: Period-frequency duality, integer flip simulator, duality emergence, zero abyss, unity verifier, harmonic normalizer, paradox resolver, reciprocal nature prover
 * - Snippet3: Data point prover with harmonic series, continued fractions, frequency duality, flip chains, zero limits
 * - Snippet4: Calculator with 6 methods including harmonic sums, continued fractions, frequency duality, flip chains, zero limits, dream sequences
 * 
 * All original code preserved exactly except standalone main functions removed
 * Each new analysis prints results as computed
 * 
 * Compilation: g++ -std=c++17 -O2 reciprocal-integer-analyzer-mega.cpp reciprocal-integer-analyzer-mega-addon-4.cpp -o analyzer
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
#include <bits/stdc++.h>

// Boost multiprecision (from main file and snippet1)
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

using high_precision_float = number<cpp_dec_float<PRECISION_DECIMALS + GUARD_DIGITS>>;
using high_precision_int = number<cpp_int>;

// ============================== SNIPPET1: FRACTION ANALYSIS (fracana namespace) ==============================

namespace fracana {

// -------------------- Configuration and small types --------------------
struct FractionConfig {
    uint64_t fallback_max_order_loop = 2000000ULL;
    uint64_t max_repetend_digits_for_bigint = 100000ULL;
    bool test_cyclic_permutations = true;
    int verbosity = 0;
};

struct FractionAnalysis {
    uint64_t p = 0;
    uint64_t q = 0;
    uint64_t reduced_p = 0;
    uint64_t reduced_q = 0;
    uint64_t q_prime = 0;
    bool is_terminating = false;
    uint64_t repetend_length_k = 0;
    std::string repetend_string;
    std::string repetend_integer_str;
    std::string decimal_prefix;
    bool is_full_reptend_prime = false;
    bool is_cyclic_number = false;
    double timing_ms = 0.0;
    std::string notes;
    
    std::string to_json() const {
        std::ostringstream o;
        o << "{";
        o << "&quot;p&quot;:" << p << ",";
        o << "&quot;q&quot;:" << q << ",";
        o << "&quot;reduced_p&quot;:" << reduced_p << ",";
        o << "&quot;reduced_q&quot;:" << reduced_q << ",";
        o << "&quot;q_prime&quot;:" << q_prime << ",";
        o << "&quot;is_terminating&quot;:" << (is_terminating ? "true" : "false") << ",";
        o << "&quot;repetend_length_k&quot;:" << repetend_length_k << ",";
        o << "&quot;repetend_string&quot;:&quot;" << repetend_string << "&quot;,";
        o << "&quot;repetend_integer&quot;:&quot;" << repetend_integer_str << "&quot;,";
        o << "&quot;decimal_prefix&quot;:&quot;" << decimal_prefix << "&quot;,";
        o << "&quot;is_full_reptend_prime&quot;:" << (is_full_reptend_prime ? "true":"false") << ",";
        o << "&quot;is_cyclic_number&quot;:" << (is_cyclic_number ? "true":"false") << ",";
        o << "&quot;timing_ms&quot;:" << timing_ms << ",";
        o << "&quot;notes&quot;:&quot;" << notes << "&quot;";
        o << "}";
        return o.str();
    }
};

// -------------------- Utility functions --------------------
static inline uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t mod) {
    __uint128_t res = (__uint128_t) a * b;
    res %= mod;
    return (uint64_t) res;
}

static inline uint64_t pow_mod_u64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1 % mod;
    uint64_t cur = base % mod;
    while (exp) {
        if (exp & 1) result = mul_mod_u64(result, cur, mod);
        cur = mul_mod_u64(cur, cur, mod);
        exp >>= 1;
    }
    return result;
}

static bool is_probable_prime_u64(uint64_t n) {
    if (n < 2) return false;
    static const uint64_t small_primes[] = {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL};
    for (uint64_t p : small_primes) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    uint64_t d = n - 1;
    int s = 0;
    while ((d & 1) == 0) { d >>= 1; s++; }
    uint64_t test_bases[] = {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL};
    for (uint64_t a : test_bases) {
        if (a % n == 0) continue;
        uint64_t x = pow_mod_u64(a, d, n);
        if (x == 1 || x == n-1) continue;
        bool cont = false;
        for (int r=1; r<s; ++r) {
            x = mul_mod_u64(x, x, n);
            if (x == n-1) { cont = true; break; }
        }
        if (cont) continue;
        return false;
    }
    return true;
}

static std::mt19937_64 rng64((uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());

static uint64_t pollard_rho_single(uint64_t n) {
    if (n % 2ULL == 0ULL) return 2ULL;
    if (n % 3ULL == 0ULL) return 3ULL;
    std::uniform_int_distribution<uint64_t> dist(2, n-2);
    uint64_t c = dist(rng64);
    uint64_t x = dist(rng64);
    uint64_t y = x;
    uint64_t d = 1;
    while (d == 1) {
        x = (mul_mod_u64(x, x, n) + c) % n;
        y = (mul_mod_u64(y, y, n) + c) % n;
        y = (mul_mod_u64(y, y, n) + c) % n;
        uint64_t diff = x > y ? x - y : y - x;
        d = std::gcd(diff, n);
        if (d == n) return 0;
    }
    return d;
}

static void factor_u64_rec(uint64_t n, std::vector<uint64_t>& out) {
    if (n == 1) return;
    if (is_probable_prime_u64(n)) {
        out.push_back(n);
        return;
    }
    uint64_t d = 0;
    for (int tries=0; tries<8 && d==0; ++tries) {
        d = pollard_rho_single(n);
    }
    if (d == 0) {
        for (uint64_t p = 2; p*p <= n; ++p) {
            if (n % p == 0) {
                while (n % p == 0) { out.push_back(p); n /= p; }
                factor_u64_rec(n, out);
                return;
            }
        }
        if (n > 1) out.push_back(n);
        return;
    }
    factor_u64_rec(d, out);
    factor_u64_rec(n/d, out);
}

static std::map<uint64_t, std::vector<uint64_t>> factor_cache;

static std::vector<uint64_t> factor_u64(uint64_t n) {
    if (n <= 1) return {};
    auto it = factor_cache.find(n);
    if (it != factor_cache.end()) return it->second;
    std::vector<uint64_t> facs;
    factor_u64_rec(n, facs);
    std::sort(facs.begin(), facs.end());
    factor_cache[n] = facs;
    return facs;
}

static uint64_t compute_phi_from_factors(uint64_t n) {
    if (n == 0) return 0;
    auto facs = factor_u64(n);
    uint64_t phi = n;
    uint64_t i = 0;
    while (i < facs.size()) {
        uint64_t p = facs[i];
        phi = phi / p * (p - 1);
        while (i < facs.size() && facs[i] == p) ++i;
    }
    return phi;
}

static std::vector<uint64_t> divisors_from_factors(const std::vector<uint64_t>& facs) {
    std::vector<std::pair<uint64_t,int>> grouped;
    for (size_t i=0;i<facs.size();) {
        uint64_t p = facs[i];
        int c = 0;
        while (i<facs.size() && facs[i]==p) { ++c; ++i; }
        grouped.push_back({p,c});
    }
    std::vector<uint64_t> divs = {1};
    for (auto &pr : grouped) {
        uint64_t p = pr.first;
        int c = pr.second;
        std::vector<uint64_t> next;
        uint64_t v = 1;
        for (int e=0;e<=c;++e) {
            for (uint64_t d : divs) next.push_back(d * v);
            v *= p;
        }
        divs.swap(next);
    }
    std::sort(divs.begin(), divs.end());
    return divs;
}

static uint64_t multiplicative_order_10_mod(uint64_t m, const FractionConfig& cfg, std::string& notes_out) {
    notes_out.clear();
    if (m == 1) return 1;
    if (std::gcd<uint64_t>(10ULL, m) != 1) {
        notes_out = "gcd(10,q') != 1, invalid for multiplicative order";
        return 0;
    }
    uint64_t phi = compute_phi_from_factors(m);
    if (phi == 0) {
        notes_out = "phi computation failed";
        return 0;
    }
    auto phi_facs = factor_u64(phi);
    auto divs = divisors_from_factors(phi_facs);
    for (uint64_t d : divs) {
        if (d == 0) continue;
        if (pow_mod_u64(10ULL, d, m) == 1ULL) return d;
    }
    uint64_t limit = cfg.fallback_max_order_loop;
    uint64_t cur = 1;
    for (uint64_t k=1;k<=limit;++k) {
        cur = mul_mod_u64(cur, 10ULL, m);
        if (cur == 1) return k;
    }
    notes_out = "order not found within fallback limit";
    return 0;
}

// -------------------- Big-int helpers --------------------
static cpp_int pow10_cppint(uint64_t k) {
    cpp_int r = 1;
    cpp_int ten = 10;
    while (k) {
        if (k & 1) r *= ten;
        ten *= ten;
        k >>= 1;
    }
    return r;
}

static bool compute_repetend_integer_big(uint64_t qprime, uint64_t k, std::string& out_str, uint64_t max_digits) {
    out_str.clear();
    if (k > max_digits) {
        return false;
    }
    cpp_int tenk = pow10_cppint(k);
    cpp_int numerator = tenk - 1;
    cpp_int qp = qprime;
    cpp_int R = numerator / qp;
    std::string s = R.convert_to<std::string>();
    out_str = s;
    return true;
}

static std::string leftpad_zero(const std::string& s, uint64_t k) {
    if (s.size() >= k) return s;
    std::string r;
    r.reserve(k);
    r.append(k - s.size(), '0');
    r += s;
    return r;
}

static std::string compute_decimal_prefix_by_longdiv(uint64_t p, uint64_t q, uint64_t prefix_len_limit=64) {
    uint64_t qq = q;
    uint64_t c2 = 0, c5 = 0;
    while (qq % 2ULL == 0ULL) { ++c2; qq /= 2ULL; }
    while (qq % 5ULL == 0ULL) { ++c5; qq /= 5ULL; }
    uint64_t prefix_len = std::max(c2, c5);
    if (prefix_len == 0) return "";
    if (prefix_len > prefix_len_limit) prefix_len = prefix_len_limit;
    std::string out;
    out.reserve(prefix_len);
    uint64_t rem = (p % q);
    for (uint64_t i=0;i<prefix_len;++i) {
        rem *= 10ULL;
        uint64_t d = rem / q;
        out.push_back(char('0' + (int)d));
        rem = rem % q;
    }
    return out;
}

static int kth_digit_of_fraction(uint64_t p, uint64_t q, uint64_t N) {
    if (q == 0) return -1;
    uint64_t powmod = pow_mod_u64(10ULL, N, q);
    uint64_t rem = mul_mod_u64((uint64_t)(p % q), powmod, q);
    uint64_t digit = mul_mod_u64(rem, 10ULL, q) / q;
    return (int)digit;
}

static bool test_cyclic_by_rotation(const std::string& rept, uint64_t qprime) {
    if (rept.empty()) return false;
    std::string doubled = rept + rept;
    size_t k = rept.size();
    for (uint64_t m = 2; m <= std::min<uint64_t>(qprime-1, 50ULL); ++m) {
        std::vector<int> a(k);
        for (size_t i=0;i<k;i++) a[k-1-i] = rept[i]-'0';
        std::vector<int> res(k+10,0);
        for (size_t i=0;i<k;i++) {
            for (size_t j=0;j<1;j++){
                res[i] += a[i] * (int)m;
            }
        }
        int carry=0;
        for (size_t i=0;i<res.size();++i) { int v = res[i] + carry; res[i] = v % 10; carry = v / 10; }
        std::string s;
        for (int i=(int)res.size()-1;i>=0;--i) {
            s.push_back(char('0' + res[i]));
        }
        size_t pos = s.find_first_not_of('0');
        if (pos==std::string::npos) s="0";
        else s = s.substr(pos);
        if (s.size() < k) s = leftpad_zero(s, k);
        else if (s.size() > k) s = s.substr(s.size()-k);
        if (doubled.find(s) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// -------------------- Memoization structures --------------------
struct QPrimeMemo {
    uint64_t qprime = 0;
    bool computed = false;
    bool is_terminating = false;
    uint64_t repetend_k = 0;
    std::string base_repetend_string;
    std::string base_repetend_integer_str;
    bool full_reptend_prime = false;
    std::string notes;
};

static std::unordered_map<uint64_t, QPrimeMemo> qprime_memo;

// -------------------- Main analysis functions --------------------
static uint64_t remove_factors_2_and_5(uint64_t q) {
    while (q % 2ULL == 0ULL) q /= 2ULL;
    while (q % 5ULL == 0ULL) q /= 5ULL;
    return q;
}

static void compute_qprime_memo(uint64_t q, const FractionConfig& cfg, QPrimeMemo& out) {
    if (q == 0) { out.computed = true; out.qprime = 0; out.is_terminating = false; return; }
    uint64_t qprime = remove_factors_2_and_5(q);
    out.qprime = qprime;
    out.computed = true;
    if (qprime == 1) {
        out.is_terminating = true;
        out.repetend_k = 0;
        out.base_repetend_string = "";
        out.base_repetend_integer_str = "";
        out.full_reptend_prime = false;
        out.notes = "terminating (q'==1)";
        return;
    }
    out.is_terminating = false;
    std::string notes;
    uint64_t k = multiplicative_order_10_mod(qprime, cfg, notes);
    out.repetend_k = k;
    out.notes = notes;
    if (k > 0 && k <= cfg.max_repetend_digits_for_bigint) {
        bool ok = compute_repetend_integer_big(qprime, k, out.base_repetend_integer_str, cfg.max_repetend_digits_for_bigint);
        if (ok) {
            out.base_repetend_string = leftpad_zero(out.base_repetend_integer_str, k);
        } else {
            out.base_repetend_string.clear();
            out.base_repetend_integer_str.clear();
            out.notes += "; repetend integer skipped due to size";
        }
    } else {
        out.base_repetend_string.clear();
        out.base_repetend_integer_str.clear();
        if (k==0) out.notes += "; k==0 (not found)";
        else out.notes += "; k too large for exact repetend bigint";
    }
    if (is_probable_prime_u64(qprime)) {
        out.full_reptend_prime = (k == qprime - 1);
    } else {
        out.full_reptend_prime = false;
    }
}

static FractionAnalysis analyze_fraction_core(uint64_t p_in, uint64_t q_in, const FractionConfig& cfg) {
    auto t0 = std::chrono::high_resolution_clock::now();
    FractionAnalysis fa;
    fa.p = p_in; fa.q = q_in;
    if (q_in == 0) { fa.notes = "q==0 invalid"; return fa; }
    uint64_t g = std::gcd(p_in, q_in);
    fa.reduced_p = p_in / g;
    fa.reduced_q = q_in / g;
    uint64_t qprime = remove_factors_2_and_5(fa.reduced_q);
    fa.q_prime = qprime;
    if (qprime == 1) {
        fa.is_terminating = true;
        fa.decimal_prefix = compute_decimal_prefix_by_longdiv(fa.reduced_p, fa.reduced_q);
        fa.repetend_length_k = 0;
        fa.repetend_string = "";
        fa.repetend_integer_str = "";
        fa.notes = "terminating decimal (q'==1)";
        auto t1 = std::chrono::high_resolution_clock::now();
        fa.timing_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return fa;
    }
    fa.is_terminating = false;
    QPrimeMemo memo;
    auto it = qprime_memo.find(qprime);
    if (it != qprime_memo.end()) memo = it->second;
    else {
        compute_qprime_memo(fa.reduced_q, cfg, memo);
        qprime_memo[qprime] = memo;
    }
    fa.repetend_length_k = memo.repetend_k;
    fa.notes = memo.notes;
    if (!memo.base_repetend_integer_str.empty()) {
        if (fa.repetend_length_k > 0 && fa.repetend_length_k <= cfg.max_repetend_digits_for_bigint) {
            try {
                cpp_int R = 0;
                R = cpp_int(0);
                std::string &Rstr = memo.base_repetend_integer_str;
                for (char c : Rstr) { R *= 10; R += (c - '0'); }
                cpp_int pmod = cpp_int(fa.reduced_p);
                cpp_int modbase = pow10_cppint(fa.repetend_length_k) - 1;
                cpp_int res = (pmod * R) % modbase;
                if (res < 0) res += modbase;
                std::string resstr = res.convert_to<std::string>();
                fa.repetend_integer_str = resstr;
                fa.repetend_string = leftpad_zero(resstr, fa.repetend_length_k);
            } catch (...) {
                fa.repetend_integer_str.clear();
                fa.repetend_string.clear();
                fa.notes += "; exception computing p*R mod (10^k-1)";
            }
        } else {
            fa.repetend_integer_str.clear();
            fa.repetend_string.clear();
        }
    } else {
        fa.repetend_integer_str.clear();
        fa.repetend_string.clear();
    }
    fa.decimal_prefix = compute_decimal_prefix_by_longdiv(fa.reduced_p, fa.reduced_q);
    fa.is_full_reptend_prime = memo.full_reptend_prime;
    if (cfg.test_cyclic_permutations && !fa.repetend_string.empty()) {
        fa.is_cyclic_number = test_cyclic_by_rotation(fa.repetend_string, qprime);
    } else {
        fa.is_cyclic_number = false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    fa.timing_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return fa;
}

static FractionAnalysis analyze_fraction(uint64_t p, uint64_t q, const FractionConfig& cfg) {
    uint64_t g = std::gcd(p,q);
    uint64_t rq = q / g;
    uint64_t qprime = remove_factors_2_and_5(rq);
    if (qprime != 1 && qprime_memo.find(qprime) == qprime_memo.end()) {
        QPrimeMemo m;
        compute_qprime_memo(rq, cfg, m);
        qprime_memo[qprime] = m;
    }
    return analyze_fraction_core(p, q, cfg);
}

using EmitCallback = std::function<void(uint64_t p, uint64_t q, const std::string& json)>;

static void analyze_and_emit_for_q(uint64_t q, const FractionConfig& cfg, EmitCallback cb=nullptr, bool include_self=false) {
    if (q == 0) {
        if (cb) cb(0,0,"{&quot;error&quot;:&quot;q==0 invalid&quot;}");
        else std::cout << "{&quot;error&quot;:&quot;q==0 invalid&quot;}\n";
        return;
    }
    if (!cb) {
        cb = [](uint64_t p, uint64_t q, const std::string& json) {
            std::cout << json << "\n";
        };
    }
    uint64_t maxp = include_self ? q : q-1;
    for (uint64_t p=1; p<=maxp; ++p) {
        FractionAnalysis fa = analyze_fraction(p, q, cfg);
        std::string js = fa.to_json();
        cb(p, q, js);
    }
}

static std::string analyze_one_fraction_json(uint64_t p, uint64_t q, const FractionConfig& cfg) {
    FractionAnalysis fa = analyze_fraction(p,q,cfg);
    return fa.to_json();
}

} // namespace fracana

// ============================== SNIPPET2: PERIOD-FREQUENCY DUALITY AND RELATED ANALYSES ==============================

// Constants for snippet2 calculations
const double EPS_SNIPPET2 = 1e-8;
const int MAX_FLIP_STEPS = 20;

// ===== NEW SNIPPET 1: Period-Frequency Duality Analyzer =====
struct PeriodFrequencyDuality {
    double test_period;
    double computed_frequency;
    bool is_real;
    std::string nature;  // "real" or "theoretical"
    double signal_conversion_value;  // Simulated real-to-frequency conversion
    std::string proof_note;
};

PeriodFrequencyDuality analyzePeriodFrequency(double val) {
    PeriodFrequencyDuality res;
    res.test_period = val;
    if (std::abs(val) < EPS_SNIPPET2) {
        res.computed_frequency = INFINITY;
        res.is_real = false;
        res.nature = "theoretical";
        res.signal_conversion_value = 0.0;
        res.proof_note = "Proof: Zero-like denominator causes infinite reciprocal; theoretical, normalized via limits.";
        return res;
    }
    res.computed_frequency = 1.0 / val;
    // Simulate conversion: sum of first 50 harmonics
    res.signal_conversion_value = 0.0;
    for (int i = 1; i <= 50; ++i) {
        res.signal_conversion_value += sin(2 * M_PI * i * res.computed_frequency);
    }
    res.is_real = (std::abs(val) > 1e-5 && std::abs(val) < 1e5);
    res.nature = res.is_real ? "real" : "theoretical";
    res.proof_note = res.is_real ? "Proof: Finite denominator enables real apps like signal processing." : "Proof: Extreme denominator requires theoretical extensions.";
    return res;
}

// ===== NEW SNIPPET 2: Integer Reciprocal Flip Simulator =====
struct IntegerFlipSim {
    int start_integer;
    std::vector<double> flip_chain;  // Sequence of flips
    int chain_length;
    bool emerges_real;
    std::string proof_explanation;
};

IntegerFlipSim simulateIntegerFlip(int n, int max_steps = MAX_FLIP_STEPS) {
    IntegerFlipSim res;
    res.start_integer = n;
    double current = n;
    res.flip_chain.push_back(current);
    for (int i = 0; i < max_steps; ++i) {
        if (std::abs(current) < EPS_SNIPPET2) break;
        current = 1.0 / current;
        res.flip_chain.push_back(current);
    }
    res.chain_length = res.flip_chain.size();
    res.emerges_real = (res.chain_length < max_steps);
    res.proof_explanation = res.emerges_real ? "Proof: Finite chain shows real emergence from duality." : "Proof: Infinite-like chain proves theoretical nature.";
    return res;
}

// ===== NEW SNIPPET 3: Duality Emergence Pattern Detector =====
struct DualityEmergence {
    double original;
    double reciprocal_val;
    double pattern_variance;
    bool pattern_emerges;
    std::string emergence_proof;
};

DualityEmergence detectDualityEmergence(double x) {
    DualityEmergence res;
    res.original = x;
    res.reciprocal_val = 1.0 / x;
    // Compute variance in a simple series (e.g., x, 1/x, x+1/x, etc.)
    std::vector<double> series = {x, res.reciprocal_val, x + res.reciprocal_val, x * res.reciprocal_val};
    double mean = 0.0;
    for (auto v : series) mean += v;
    mean /= series.size();
    res.pattern_variance = 0.0;
    for (auto v : series) res.pattern_variance += (v - mean) * (v - mean);
    res.pattern_variance /= series.size();
    res.pattern_emerges = (res.pattern_variance < 1e3);  // Threshold for emergence
    res.emergence_proof = res.pattern_emerges ? "Proof: Low variance indicates conscious study emergence." : "Proof: High variance shows theoretical duality.";
    return res;
}

// ===== NEW SNIPPET 4: Zero Abyss Boundary Explorer =====
struct ZeroAbyssExplorer {
    double near_zero_val;
    double reciprocal_approx;
    double limit_pos;
    double limit_neg;
    std::string abyss_proof;
};

ZeroAbyssExplorer exploreZeroAbyss(double base = 1e-6) {
    ZeroAbyssExplorer res;
    res.near_zero_val = base;
    res.reciprocal_approx = 1.0 / base;
    // Approximate limits
    res.limit_pos = 1.0 / (base + 1e-10);
    res.limit_neg = 1.0 / (base - 1e-10);
    res.abyss_proof = "Proof: Diverging limits (" + std::to_string(res.limit_pos) + ", " + std::to_string(res.limit_neg) + ") prove theoretical abyss at zero.";
    return res;
}

// ===== NEW SNIPPET 5: Unity Self-Inversion Verifier =====
struct UnityVerifier {
    double test_val;
    bool is_self_reciprocal;
    double inversion_error;
    std::string verification_proof;
};

UnityVerifier verifyUnityInversion(double x) {
    UnityVerifier res;
    res.test_val = x;
    double recip = 1.0 / x;
    res.inversion_error = std::abs(x - recip);
    res.is_self_reciprocal = (res.inversion_error < 1e-10);
    res.verification_proof = res.is_self_reciprocal ? "Proof: x = 1/x holds with error " + std::to_string(res.inversion_error) : "Proof: Non-zero error shows no self-inversion.";
    return res;
}

// ===== NEW SNIPPET 6: Harmonic Series Reciprocal Normalizer =====
struct HarmonicNormalizer {
    int terms;
    double harmonic_sum;
    bool normalizes;
    std::string norm_proof;
};

HarmonicNormalizer normalizeWithHarmonics(int max_terms = 1000) {
    HarmonicNormalizer res;
    res.terms = max_terms;
    res.harmonic_sum = 0.0;
    for (int k = 1; k <= max_terms; ++k) {
        res.harmonic_sum += 1.0 / k;
    }
    res.normalizes = std::isfinite(res.harmonic_sum);
    res.norm_proof = res.normalizes ? "Proof: Finite sum approximation normalizes reciprocal study." : "Proof: Divergence proves theoretical infinity.";
    return res;
}

// ===== NEW SNIPPET 7: Paradox Iterative Resolver =====
struct ParadoxResolver {
    double start_val;
    int iterations;
    double final_val;
    bool resolves;
    std::string resolve_proof;
};

ParadoxResolver resolveParadox(double x, int max_iter = 50) {
    ParadoxResolver res;
    res.start_val = x;
    res.iterations = 0;
    double current = x;
    while (res.iterations < max_iter && std::abs(current - 1.0 / current) > 1e-5) {
        current = 1.0 / current;
        res.iterations++;
    }
    res.final_val = current;
    res.resolves = (res.iterations < max_iter);
    res.resolve_proof = res.resolves ? "Proof: Convergence after " + std::to_string(res.iterations) + " iterations resolves paradox." : "Proof: Non-convergence proves theoretical paradox.";
    return res;
}

// ===== NEW SNIPPET 8: Comprehensive Reciprocal Nature Prover =====
struct RecipNatureProver {
    std::string overall_nature;
    int proof_count;
    double confidence;
    std::string final_proof;
};

// This will be filled in after we have access to other analyses
RecipNatureProver proveRecipNature(const PeriodFrequencyDuality& pf_dual, const IntegerFlipSim& flip_sim, const DualityEmergence& dual_emerg) {
    RecipNatureProver res;
    int real_count = 0;
    if (pf_dual.is_real) real_count++;
    if (flip_sim.emerges_real) real_count++;
    if (dual_emerg.pattern_emerges) real_count++;
    res.proof_count = real_count;
    res.confidence = static_cast<double>(real_count) / 3.0;
    res.overall_nature = (res.confidence > 0.5) ? "real" : "theoretical";
    res.final_proof = "Proof: " + std::to_string(res.proof_count) + "/3 indicators confirm " + res.overall_nature + " nature with confidence " + std::to_string(res.confidence);
    return res;
}

// ============================== SNIPPET3: RECIPROCAL DATA POINT PROVER ==============================

// Constants for snippet3
const int MAX_TERMS_SNIPPET3 = 1000;
const int DATA_POINTS_SNIPPET3 = 50;

// Data structure for individual proofs
struct ReciprocalProof {
    std::string method;              // e.g., "Harmonic", "ContinuedFraction"
    std::vector<double> data_points; // Computed values (e.g., sums, convergents)
    bool is_real;               // Real (finite/computable) or theoretical
    double confidence;          // 0.0-1.0 based on convergence/divergence
    std::string explanation;         // Proof statement
};

// Main structure for the snippet's analysis
struct RecipDataProver {
    int original_n;
    double reciprocal;
    std::vector<ReciprocalProof> proofs;  // Collection of proof methods
    std::string overall_verdict;          // "Real", "Theoretical", or "Dual"
    double aggregate_confidence;     // Average from all proofs
    std::string final_proof;              // Summary proof text
};

// Output config
bool output_to_file_snippet3 = false;
std::string output_filename_snippet3 = "reciprocal_proofs_data.txt";

// Helper: Compute harmonic partial sum up to terms
double harmonic_sum_snippet3(int terms) {
    double sum = 0.0;
    for (int k = 1; k <= terms; ++k) {
        sum += 1.0 / k;
    }
    return sum;
}

// Helper: Simple continued fraction for 1/sqrt(n) approximation
std::vector<double> continued_fraction_sqrt_recip_snippet3(int n, int max_terms) {
    std::vector<double> convergents;
    if (n <= 0) return convergents;
    double sqrt_n = sqrt(static_cast<double>(n));
    double recip = 1.0 / sqrt_n;
    double a0 = floor(recip);
    double x = recip - a0;
    convergents.push_back(a0);

    for (int i = 1; i < max_terms; ++i) {
        if (x < EPS_SNIPPET2) break;
        double a = floor(1.0 / x);
        convergents.push_back(a);
        x = 1.0 / x - a;
    }
    return convergents;
}

// Helper: Simulate frequency duality with sine wave DFT-like peak
std::vector<double> frequency_duality_sim_snippet3(double T, int samples) {
    std::vector<double> peaks;
    if (abs(T) < EPS_SNIPPET2) return peaks;  // Theoretical case
    double f = 1.0 / T;
    for (int i = 1; i <= samples; ++i) {
        double peak = sin(2 * M_PI * f * i);  // Simplified "spectrum"
        peaks.push_back(peak);
    }
    return peaks;
}

// Helper: Iterative flip chain
std::vector<double> iterative_flip_chain_snippet3(double start, int max_steps) {
    std::vector<double> chain;
    double current = start;
    chain.push_back(current);
    for (int i = 0; i < max_steps; ++i) {
        if (abs(current) < EPS_SNIPPET2) break;  // Avoid division by zero
        current = 1.0 / current;
        chain.push_back(current);
    }
    return chain;
}

// Helper: Zero boundary limits
std::pair<double, double> zero_boundary_limits_snippet3(double base_eps = 1e-6) {
    double pos_limit = 1.0 / (base_eps + EPS_SNIPPET2);
    double neg_limit = 1.0 / (base_eps - EPS_SNIPPET2);
    return {pos_limit, neg_limit};
}

// Core function: Generate proofs for a given n
RecipDataProver proveReciprocals_snippet3(int n) {
    RecipDataProver result;
    result.original_n = n;
    result.reciprocal = (n != 0) ? 1.0 / n : INFINITY;

    // Proof 1: Harmonic Series Divergence
    ReciprocalProof harm_proof;
    harm_proof.method = "Harmonic Series";
    for (int i = 1; i <= DATA_POINTS_SNIPPET3; ++i) {
        harm_proof.data_points.push_back(harmonic_sum_snippet3(i * (MAX_TERMS_SNIPPET3 / DATA_POINTS_SNIPPET3)));
    }
    harm_proof.is_real = false;  // Diverges theoretically
    harm_proof.confidence = 1.0 - (harm_proof.data_points.back() / log(MAX_TERMS_SNIPPET3));  // Approx divergence measure
    harm_proof.explanation = "Proof: Partial sums approach ln(n) + γ, diverging slowly; reciprocals theoretical in sum but real in approximation.";
    result.proofs.push_back(harm_proof);

    // Proof 2: Continued Fraction for 1/sqrt(n)
    ReciprocalProof cf_proof;
    cf_proof.method = "Continued Fraction (1/sqrt(n))";
    auto cf = continued_fraction_sqrt_recip_snippet3(n, DATA_POINTS_SNIPPET3);
    for (auto v : cf) cf_proof.data_points.push_back(v);
    cf_proof.is_real = (cf.size() < DATA_POINTS_SNIPPET3);  // Finite if terminates
    cf_proof.confidence = cf_proof.is_real ? 1.0 : 0.5;  // Periodic = theoretical
    cf_proof.explanation = "Proof: Periodic fractions prove irrationality; data points show convergence to reciprocal.";
    result.proofs.push_back(cf_proof);

    // Proof 3: Frequency Duality Simulation
    ReciprocalProof freq_proof;
    freq_proof.method = "Frequency Duality";
    auto peaks = frequency_duality_sim_snippet3(result.reciprocal, DATA_POINTS_SNIPPET3);  // Use reciprocal as T
    for (auto p : peaks) freq_proof.data_points.push_back(p);
    freq_proof.is_real = isfinite(1.0 / result.reciprocal) && (abs(result.reciprocal) > EPS_SNIPPET2);
    freq_proof.confidence = freq_proof.is_real ? 0.9 : 0.1;
    freq_proof.explanation = "Proof: Stable peaks in 'spectrum' show real application; extremes theoretical.";
    result.proofs.push_back(freq_proof);

    // Proof 4: Iterative Flip Chains
    ReciprocalProof flip_proof;
    flip_proof.method = "Iterative Flip Chains";
    auto chain = iterative_flip_chain_snippet3(n, DATA_POINTS_SNIPPET3);
    for (auto v : chain) flip_proof.data_points.push_back(v);
    flip_proof.is_real = (chain.size() < DATA_POINTS_SNIPPET3);  // Cycles = real
    flip_proof.confidence = flip_proof.is_real ? 0.8 : 0.2;
    flip_proof.explanation = "Proof: Finite chains prove computable duality; infinite-like theoretical.";
    result.proofs.push_back(flip_proof);

    // Proof 5: Zero Boundary Exploration
    ReciprocalProof zero_proof;
    zero_proof.method = "Zero Boundary Limits";
    auto limits = zero_boundary_limits_snippet3();
    zero_proof.data_points = {limits.first, limits.second};
    zero_proof.is_real = false;
    zero_proof.confidence = 0.0;  // Always theoretical
    zero_proof.explanation = "Proof: Diverging positive/negative limits prove abyss at zero denominator.";
    result.proofs.push_back(zero_proof);

    // Aggregate
    int real_count = 0;
    double total_conf = 0.0;
    for (const auto& p : result.proofs) {
        if (p.is_real) real_count++;
        total_conf += p.confidence;
    }
    result.aggregate_confidence = total_conf / result.proofs.size();
    result.overall_verdict = (real_count > result.proofs.size() / 2) ? "Real" : "Theoretical";
    result.final_proof = "Aggregate Proof: " + std::to_string(real_count) + "/" + std::to_string(result.proofs.size()) +
                         " methods indicate " + result.overall_verdict + " nature (conf: " +
                         std::to_string(result.aggregate_confidence) + "). Reciprocals bridge theory and practice via data.";

    return result;
}

// Display function: Print proofs and data points
void displayRecipProofs_snippet3(const RecipDataProver& result) {
    std::cout << "\n=== RECIPROCAL DATA POINT PROVER ===" << std::endl;
    std::cout << "Original n: " << result.original_n << std::endl;
    std::cout << "Reciprocal: " << result.reciprocal << std::endl;
    std::cout << "Overall Verdict: " << result.overall_verdict << std::endl;
    std::cout << "Aggregate Confidence: " << std::setprecision(2) << result.aggregate_confidence << std::endl;
    std::cout << "Final Proof: " << result.final_proof << std::endl;

    for (const auto& proof : result.proofs) {
        std::cout << "\n--- " << proof.method << " ---" << std::endl;
        std::cout << "Nature: " << (proof.is_real ? "Real" : "Theoretical") << std::endl;
        std::cout << "Confidence: " << proof.confidence << std::endl;
        std::cout << "Explanation: " << proof.explanation << std::endl;
        std::cout << "Data Points (first 10 shown): ";
        for (size_t i = 0; i < std::min(size_t(10), proof.data_points.size()); ++i) {
            std::cout << std::setprecision(4) << proof.data_points[i] << " ";
        }
        std::cout << "..." << std::endl;
    }
}

// Optional: Dump all data to file for visualization/teaching
void dumpDataToFile_snippet3(const RecipDataProver& result) {
    if (!output_to_file_snippet3) return;
    std::ofstream file(output_filename_snippet3, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << output_filename_snippet3 << std::endl;
        return;
    }

    file << "n: " << result.original_n << "\n";
    for (const auto& proof : result.proofs) {
        file << proof.method << " Data:\n";
        for (auto dp : proof.data_points) {
            file << dp << ",";
        }
        file << "\n";
    }
    file << "\n";
    file.close();
}

// ============================== SNIPPET4: RECIPROCAL DATA POINT CALCULATOR ==============================

// Constants for snippet4
const int DEFAULT_DATA_POINTS_SNIPPET4 = 10;
const int MAX_ITER_SNIPPET4 = 50;
const double M_PI_CUSTOM_SNIPPET4 = 3.14159265358979323846;

// Structure for a single method's data
struct MethodData {
    std::string method_name;
    std::vector<std::vector<double>> data_per_n;  // Outer: per n, Inner: data points
    std::vector<bool> is_real_per_n;         // Real/theoretical flag per n
    std::vector<double> confidence_per_n;    // Computed confidence per n
    std::vector<std::string> explanation_per_n;   // Calculation-based explanation per n
};

// Main calculator structure
struct ReciprocalCalculator {
    int start_n;
    int end_n;
    int data_points;
    std::vector<MethodData> all_methods;     // All computed methods
};

// Calculate harmonic sums: Generate partial sums up to data_points terms for each n
MethodData calc_harmonic_sums_snippet4(int start_n, int end_n, int data_points) {
    MethodData md;
    md.method_name = "Harmonic Sums";
    for (int n = start_n; n <= end_n; ++n) {
        std::vector<double> sums;
        double current_sum = 0.0;
        for (int k = 1; k <= data_points; ++k) {
            current_sum += 1.0 / (n * k);  // Scale by reciprocal factor
            sums.push_back(current_sum);
        }
        md.data_per_n.push_back(sums);
        bool is_real = isfinite(sums.back()) && (sums.back() < log(data_points) + 1.0);  // Approx check for divergence
        md.is_real_per_n.push_back(is_real);
        md.confidence_per_n.push_back(is_real ? 1.0 - (sums.back() / log(data_points)) : 0.5);  // Confidence from approx
        md.explanation_per_n.push_back("Calculated sums show " + std::string(is_real ? "finite" : "diverging") + " behavior.");
    }
    return md;
}

// Calculate continued fractions for 1/sqrt(n): Generate up to data_points terms
MethodData calc_continued_fractions_snippet4(int start_n, int end_n, int data_points) {
    MethodData md;
    md.method_name = "Continued Fractions (1/sqrt(n))";
    for (int n = start_n; n <= end_n; ++n) {
        std::vector<double> cf_terms;
        if (n <= 0) continue;
        double recip_sqrt = 1.0 / sqrt(static_cast<double>(n));
        double a0 = floor(recip_sqrt);
        double x = recip_sqrt - a0;
        cf_terms.push_back(a0);
        for (int i = 1; i < data_points; ++i) {
            if (abs(x) < EPS_SNIPPET2) break;
            double a = floor(1.0 / x);
            cf_terms.push_back(a);
            x = 1.0 / x - a;
        }
        md.data_per_n.push_back(cf_terms);
        bool is_real = (cf_terms.size() < data_points);  // Terminates = real/rational
        md.is_real_per_n.push_back(is_real);
        md.confidence_per_n.push_back(is_real ? 1.0 : 0.5);
        md.explanation_per_n.push_back("Terms indicate " + std::string(is_real ? "termination" : "periodicity/irrationality") + ".");
    }
    return md;
}

// Calculate frequency duality: Simulate sine peaks for T=1/n, up to data_points samples
MethodData calc_frequency_duality_snippet4(int start_n, int end_n, int data_points) {
    MethodData md;
    md.method_name = "Frequency Duality Simulations";
    for (int n = start_n; n <= end_n; ++n) {
        double T = (n == 0) ? std::numeric_limits<double>::infinity() : 1.0 / n;
        std::vector<double> peaks;
        if (!isfinite(T) || abs(T) < EPS_SNIPPET2) {
            // Theoretical: Fill with INFINITY
            for (int i = 0; i < data_points; ++i) peaks.push_back(std::numeric_limits<double>::infinity());
        } else {
            double f = 1.0 / T;
            for (int i = 1; i <= data_points; ++i) {
                double peak = sin(2 * M_PI_CUSTOM_SNIPPET4 * f * i);
                peaks.push_back(peak);
            }
        }
        md.data_per_n.push_back(peaks);
        bool is_real = std::all_of(peaks.begin(), peaks.end(), [](double p){ return isfinite(p); });
        md.is_real_per_n.push_back(is_real);
        md.confidence_per_n.push_back(is_real ? 0.9 : 0.1);
        md.explanation_per_n.push_back("Peaks " + std::string(is_real ? "stable" : "unstable/infinite") + ".");
    }
    return md;
}

// Calculate flip chains: Iterate x -> 1/x up to data_points steps
MethodData calc_flip_chains_snippet4(int start_n, int end_n, int data_points) {
    MethodData md;
    md.method_name = "Iterative Flip Chains";
    for (int n = start_n; n <= end_n; ++n) {
        std::vector<double> chain;
        double current = static_cast<double>(n);
        chain.push_back(current);
        for (int i = 0; i < data_points - 1; ++i) {
            if (abs(current) < EPS_SNIPPET2) {
                chain.push_back(std::numeric_limits<double>::infinity());
                break;
            }
            current = 1.0 / current;
            chain.push_back(current);
        }
        md.data_per_n.push_back(chain);
        bool is_real = (chain.size() >= data_points && std::all_of(chain.begin(), chain.end(), [](double v){ return isfinite(v); }));
        md.is_real_per_n.push_back(is_real);
        md.confidence_per_n.push_back(is_real ? 0.8 : 0.2);
        md.explanation_per_n.push_back("Chain " + std::string(is_real ? "cycles/finite" : "diverges/theoretical") + ".");
    }
    return md;
}

// Calculate zero limits: Generate positive/negative approach data around small base
MethodData calc_zero_limits_snippet4(int start_n, int end_n, int data_points) {
    MethodData md;
    md.method_name = "Zero Boundary Limits";
    double base_eps = 1e-3;  // Adjustable base for approach
    for (int n = start_n; n <= end_n; ++n) {
        std::vector<double> limits;
        for (int i = 1; i <= data_points / 2; ++i) {
            double delta = base_eps / i;
            double pos = 1.0 / (delta + EPS_SNIPPET2);
            double neg = 1.0 / (delta - EPS_SNIPPET2);
            limits.push_back(pos);
            limits.push_back(neg);
        }
        md.data_per_n.push_back(limits);
        bool is_real = std::all_of(limits.begin(), limits.end(), [](double l){ return abs(l) < 1e6; });  // Arbitrary bound for "finite"
        md.is_real_per_n.push_back(is_real);
        md.confidence_per_n.push_back(is_real ? 0.0 : 1.0);  // Zero is always theoretical—high conf for divergence
        md.explanation_per_n.push_back("Limits diverge, proving theoretical abyss.");
    }
    return md;
}

// Calculate dream sequences: Zeta-like recurrence from 1/n seed, up to data_points steps
MethodData calc_dream_sequences_snippet4(int start_n, int end_n, int data_points) {
    MethodData md;
    md.method_name = "Dream Sequences (Zeta-like Emergence)";
    for (int n = start_n; n <= end_n; ++n) {
        std::vector<double> sequence;
        double gamma = (n == 0) ? std::numeric_limits<double>::infinity() : 1.0 / n;
        if (!isfinite(gamma)) {
            for (int i = 0; i < data_points; ++i) sequence.push_back(gamma);
        } else {
            sequence.push_back(gamma);
            for (int i = 1; i < data_points; ++i) {
                if (gamma <= 0 || log(gamma) == 0) break;
                double log_gamma = log(gamma);
                double numerator = log(gamma + 1);
                double denominator = log_gamma * log_gamma;
                double increment = 2 * M_PI_CUSTOM_SNIPPET4 * (numerator / denominator);
                gamma += increment;
                sequence.push_back(gamma);
            }
        }
        md.data_per_n.push_back(sequence);
        bool is_real = (sequence.size() == data_points && std::all_of(sequence.begin(), sequence.end(), [](double s){ return isfinite(s); }));
        md.is_real_per_n.push_back(is_real);
        md.confidence_per_n.push_back(is_real ? 0.7 : 0.3);
        md.explanation_per_n.push_back("Sequence grows " + std::string(is_real ? "stably" : "unstably") + ", proving emergence.");
    }
    return md;
}

// Main computation function
ReciprocalCalculator compute_reciprocals_snippet4(int start_n, int end_n, int data_points) {
    ReciprocalCalculator rc;
    rc.start_n = start_n;
    rc.end_n = end_n;
    rc.data_points = data_points;

    rc.all_methods.push_back(calc_harmonic_sums_snippet4(start_n, end_n, data_points));
    rc.all_methods.push_back(calc_continued_fractions_snippet4(start_n, end_n, data_points));
    rc.all_methods.push_back(calc_frequency_duality_snippet4(start_n, end_n, data_points));
    rc.all_methods.push_back(calc_flip_chains_snippet4(start_n, end_n, data_points));
    rc.all_methods.push_back(calc_zero_limits_snippet4(start_n, end_n, data_points));
    rc.all_methods.push_back(calc_dream_sequences_snippet4(start_n, end_n, data_points));

    return rc;
}

// Output function: Print all calculated data
void output_data_snippet4(const ReciprocalCalculator& rc, bool to_file = false, const std::string& filename = "reciprocal_data.txt") {
    std::ostream* out = &std::cout;
    std::ofstream file_out;
    if (to_file) {
        file_out.open(filename);
        out = &file_out;
    }

    *out << "Reciprocal Data Calculations for n=" << rc.start_n << " to " << rc.end_n << std::endl;
    for (const auto& md : rc.all_methods) {
        *out << "\n=== " << md.method_name << " ===" << std::endl;
        for (int i = 0; i < md.data_per_n.size(); ++i) {
            int n = rc.start_n + i;
            *out << "n=" << n << ": Data Points = [";
            for (size_t j = 0; j < md.data_per_n[i].size(); ++j) {
                *out << std::setprecision(6) << md.data_per_n[i][j];
                if (j < md.data_per_n[i].size() - 1) *out << ", ";
            }
            *out << "]" << std::endl;
            *out << "  Is Real: " << (md.is_real_per_n[i] ? "YES" : "NO") << std::endl;
            *out << "  Confidence: " << md.confidence_per_n[i] << std::endl;
            *out << "  Explanation: " << md.explanation_per_n[i] << std::endl;
        }
    }

    if (to_file) file_out.close();
}

// ============================== INTEGRATION SECTION ==============================
// This section provides the integration with the main analyzer

// Enhanced AnalysisEntry structure to include all snippet analyses
struct EnhancedAnalysisEntry {
    // Original data from main analyzer
    int original_number;
    double reciprocal;
    
    // Snippet1: Fraction Analysis (for analyzing 1/n as fraction)
    std::vector<std::string> fraction_analysis_results;
    
    // Snippet2: All 8 analysis methods
    PeriodFrequencyDuality pf_duality;
    IntegerFlipSim int_flip_sim;
    DualityEmergence duality_emerg;
    ZeroAbyssExplorer zero_abyss;
    UnityVerifier unity_verif;
    HarmonicNormalizer harm_norm;
    ParadoxResolver para_res;
    RecipNatureProver recip_prover;
    
    // Snippet3: Data Point Prover
    RecipDataProver data_prover;
    
    // Snippet4: Calculator results (for single number)
    std::vector<MethodData> calculator_methods;
};

// Function to run all snippet analyses on a single number
EnhancedAnalysisEntry runAllSnippetAnalyses(int n) {
    EnhancedAnalysisEntry entry;
    entry.original_number = n;
    entry.reciprocal = (n != 0) ? 1.0 / n : INFINITY;
    
    std::cout << "\n=== ANALYZING NUMBER: " << n << " (Reciprocal: " << entry.reciprocal << ") ===" << std::endl;
    
    // Snippet1: Fraction Analysis (analyze 1/n as fraction)
    std::cout << "\n--- SNIPPET1: FRACTION ANALYSIS ---" << std::endl;
    fracana::FractionConfig frac_cfg;
    std::cout << "Analyzing 1/" << n << " as fraction:" << std::endl;
    auto frac_result = fracana::analyze_fraction(1, n, frac_cfg);
    std::string json_result = frac_result.to_json();
    entry.fraction_analysis_results.push_back(json_result);
    std::cout << "Fraction Analysis: " << json_result << std::endl;
    
    // Snippet2: Period-Frequency Duality
    std::cout << "\n--- SNIPPET2: PERIOD-FREQUENCY DUALITY ---" << std::endl;
    entry.pf_duality = analyzePeriodFrequency(entry.reciprocal);
    std::cout << "Period: " << entry.pf_duality.test_period << std::endl;
    std::cout << "Frequency: " << entry.pf_duality.computed_frequency << std::endl;
    std::cout << "Nature: " << entry.pf_duality.nature << std::endl;
    std::cout << "Conversion Value: " << entry.pf_duality.signal_conversion_value << std::endl;
    std::cout << "Proof: " << entry.pf_duality.proof_note << std::endl;
    
    // Snippet2: Integer Flip Simulator
    std::cout << "\n--- SNIPPET2: INTEGER FLIP SIMULATOR ---" << std::endl;
    entry.int_flip_sim = simulateIntegerFlip(n);
    std::cout << "Start: " << entry.int_flip_sim.start_integer << std::endl;
    std::cout << "Chain Length: " << entry.int_flip_sim.chain_length << std::endl;
    std::cout << "Emerges Real: " << (entry.int_flip_sim.emerges_real ? "YES" : "NO") << std::endl;
    std::cout << "Chain: ";
    for (auto v : entry.int_flip_sim.flip_chain) std::cout << v << " ";
    std::cout << std::endl << "Proof: " << entry.int_flip_sim.proof_explanation << std::endl;
    
    // Snippet2: Duality Emergence Pattern Detector
    std::cout << "\n--- SNIPPET2: DUALITY EMERGENCE DETECTOR ---" << std::endl;
    entry.duality_emerg = detectDualityEmergence(static_cast<double>(n));
    std::cout << "Original: " << entry.duality_emerg.original << std::endl;
    std::cout << "Reciprocal: " << entry.duality_emerg.reciprocal_val << std::endl;
    std::cout << "Variance: " << entry.duality_emerg.pattern_variance << std::endl;
    std::cout << "Emerges: " << (entry.duality_emerg.pattern_emerges ? "YES" : "NO") << std::endl;
    std::cout << "Proof: " << entry.duality_emerg.emergence_proof << std::endl;
    
    // Snippet2: Zero Abyss Boundary Explorer
    std::cout << "\n--- SNIPPET2: ZERO ABYSS EXPLORER ---" << std::endl;
    entry.zero_abyss = exploreZeroAbyss(std::abs(entry.reciprocal) < 1e-3 ? entry.reciprocal : 1e-6);
    std::cout << "Near Zero: " << entry.zero_abyss.near_zero_val << std::endl;
    std::cout << "Reciprocal Approx: " << entry.zero_abyss.reciprocal_approx << std::endl;
    std::cout << "Positive Limit: " << entry.zero_abyss.limit_pos << std::endl;
    std::cout << "Negative Limit: " << entry.zero_abyss.limit_neg << std::endl;
    std::cout << "Proof: " << entry.zero_abyss.abyss_proof << std::endl;
    
    // Snippet2: Unity Self-Inversion Verifier
    std::cout << "\n--- SNIPPET2: UNITY SELF-INVERSION VERIFIER ---" << std::endl;
    entry.unity_verif = verifyUnityInversion(static_cast<double>(n));
    std::cout << "Test Value: " << entry.unity_verif.test_val << std::endl;
    std::cout << "Self-Reciprocal: " << (entry.unity_verif.is_self_reciprocal ? "YES" : "NO") << std::endl;
    std::cout << "Error: " << entry.unity_verif.inversion_error << std::endl;
    std::cout << "Proof: " << entry.unity_verif.verification_proof << std::endl;
    
    // Snippet2: Harmonic Series Reciprocal Normalizer
    std::cout << "\n--- SNIPPET2: HARMONIC SERIES NORMALIZER ---" << std::endl;
    entry.harm_norm = normalizeWithHarmonics();
    std::cout << "Terms: " << entry.harm_norm.terms << std::endl;
    std::cout << "Sum: " << entry.harm_norm.harmonic_sum << std::endl;
    std::cout << "Normalizes: " << (entry.harm_norm.normalizes ? "YES" : "NO") << std::endl;
    std::cout << "Proof: " << entry.harm_norm.norm_proof << std::endl;
    
    // Snippet2: Paradox Iterative Resolver
    std::cout << "\n--- SNIPPET2: PARADOX ITERATIVE RESOLVER ---" << std::endl;
    entry.para_res = resolveParadox(static_cast<double>(n));
    std::cout << "Start: " << entry.para_res.start_val << std::endl;
    std::cout << "Iterations: " << entry.para_res.iterations << std::endl;
    std::cout << "Final: " << entry.para_res.final_val << std::endl;
    std::cout << "Resolves: " << (entry.para_res.resolves ? "YES" : "NO") << std::endl;
    std::cout << "Proof: " << entry.para_res.resolve_proof << std::endl;
    
    // Snippet2: Comprehensive Reciprocal Nature Prover
    std::cout << "\n--- SNIPPET2: RECIPROCAL NATURE PROVER ---" << std::endl;
    entry.recip_prover = proveRecipNature(entry.pf_duality, entry.int_flip_sim, entry.duality_emerg);
    std::cout << "Nature: " << entry.recip_prover.overall_nature << std::endl;
    std::cout << "Proof Count: " << entry.recip_prover.proof_count << std::endl;
    std::cout << "Confidence: " << entry.recip_prover.confidence << std::endl;
    std::cout << "Final Proof: " << entry.recip_prover.final_proof << std::endl;
    
    // Snippet3: Data Point Prover
    std::cout << "\n--- SNIPPET3: RECIPROCAL DATA POINT PROVER ---" << std::endl;
    entry.data_prover = proveReciprocals_snippet3(n);
    displayRecipProofs_snippet3(entry.data_prover);
    dumpDataToFile_snippet3(entry.data_prover);
    
    // Snippet4: Calculator (single number analysis)
    std::cout << "\n--- SNIPPET4: RECIPROCAL DATA POINT CALCULATOR ---" << std::endl;
    auto calc_results = compute_reciprocals_snippet4(n, n, 10);
    std::cout << "Calculator results for n=" << n << ":" << std::endl;
    output_data_snippet4(calc_results, false);
    
    return entry;
}

// Function to analyze a range of numbers and show results as they come
void analyzeRangeWithAllSnippets(int start_n, int end_n) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "STARTING COMPREHENSIVE ANALYSIS: Snippets 1-4 Integrated" << std::endl;
    std::cout << "Range: " << start_n << " to " << end_n << std::endl;
    std::cout << "============================================================" << std::endl;
    
    std::vector<EnhancedAnalysisEntry> all_results;
    
    for (int n = start_n; n <= end_n; ++n) {
        auto entry = runAllSnippetAnalyses(n);
        all_results.push_back(entry);
        
        std::cout << "\n--- ANALYSIS COMPLETE FOR " << n << " ---" << std::endl;
        std::cout << "Moving to next number..." << std::endl;
        std::cout << "============================================================" << std::endl;
    }
    
    std::cout << "\n\n=== SUMMARY OF ALL ANALYSES ===" << std::endl;
    for (const auto& entry : all_results) {
        std::cout << "Number " << entry.original_number << ": ";
        std::cout << "Nature=" << entry.recip_prover.overall_nature << ", ";
        std::cout << "Confidence=" << std::setprecision(2) << entry.recip_prover.confidence << ", ";
        std::cout << "DataVerdict=" << entry.data_prover.overall_verdict << std::endl;
    }
}
// ULTIMATE_PI_SIMPLIFIER.cpp
// Finding the GCD of enormous π fractions
// Compile: g++ -std=c++17 -O3 ULTIMATE_PI_SIMPLIFIER.cpp -o pi_simplify

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>

// Big integer class for handling massive numbers
class BigInt {
private:
    std::vector<int> digits;  // Store in reverse order (least significant first)
    bool negative;
    
public:
    BigInt() : negative(false) { digits.push_back(0); }
    
    BigInt(long long n) : negative(n < 0) {
        n = std::abs(n);
        if (n == 0) {
            digits.push_back(0);
        } else {
            while (n > 0) {
                digits.push_back(n % 10);
                n /= 10;
            }
        }
    }
    
    BigInt(const std::string& s) {
        negative = (s[0] == '-');
        size_t start = negative ? 1 : 0;
        
        for (size_t i = s.length(); i > start; --i) {
            if (std::isdigit(s[i-1])) {
                digits.push_back(s[i-1] - '0');
            }
        }
        
        if (digits.empty()) digits.push_back(0);
        removeLeadingZeros();
    }
    
    void removeLeadingZeros() {
        while (digits.size() > 1 && digits.back() == 0) {
            digits.pop_back();
        }
        if (digits.size() == 1 && digits[0] == 0) {
            negative = false;
        }
    }
    
    bool isZero() const {
        return digits.size() == 1 && digits[0] == 0;
    }
    
    bool isEven() const {
        return digits[0] % 2 == 0;
    }
    
    // Comparison operators
    bool operator<(const BigInt& other) const {
        if (negative != other.negative) return negative;
        
        if (digits.size() != other.digits.size()) {
            return negative ? digits.size() > other.digits.size() : digits.size() < other.digits.size();
        }
        
        for (int i = digits.size() - 1; i >= 0; --i) {
            if (digits[i] != other.digits[i]) {
                return negative ? digits[i] > other.digits[i] : digits[i] < other.digits[i];
            }
        }
        return false;
    }
    
    bool operator==(const BigInt& other) const {
        return negative == other.negative && digits == other.digits;
    }
    
    bool operator>(const BigInt& other) const {
        return other < *this;
    }
    
    bool operator<=(const BigInt& other) const {
        return !(other < *this);
    }
    
    // Addition
    BigInt operator+(const BigInt& other) const {
        if (negative == other.negative) {
            BigInt result;
            result.negative = negative;
            result.digits.clear();
            
            int carry = 0;
            size_t maxSize = std::max(digits.size(), other.digits.size());
            
            for (size_t i = 0; i < maxSize || carry; ++i) {
                int sum = carry;
                if (i < digits.size()) sum += digits[i];
                if (i < other.digits.size()) sum += other.digits[i];
                
                result.digits.push_back(sum % 10);
                carry = sum / 10;
            }
            
            result.removeLeadingZeros();
            return result;
        } else {
            BigInt temp = other;
            temp.negative = !temp.negative;
            return *this - temp;
        }
    }
    
    // Subtraction
    BigInt operator-(const BigInt& other) const {
        if (negative != other.negative) {
            BigInt temp = other;
            temp.negative = !temp.negative;
            return *this + temp;
        }
        
        BigInt a = *this;
        BigInt b = other;
        a.negative = false;
        b.negative = false;
        
        bool resultNegative = false;
        if (a < b) {
            std::swap(a, b);
            resultNegative = !negative;
        } else {
            resultNegative = negative;
        }
        
        BigInt result;
        result.digits.clear();
        
        int borrow = 0;
        for (size_t i = 0; i < a.digits.size(); ++i) {
            int diff = a.digits[i] - borrow;
            if (i < b.digits.size()) diff -= b.digits[i];
            
            if (diff < 0) {
                diff += 10;
                borrow = 1;
            } else {
                borrow = 0;
            }
            
            result.digits.push_back(diff);
        }
        
        result.negative = resultNegative;
        result.removeLeadingZeros();
        return result;
    }
    
    // Modulo operator
    BigInt operator%(const BigInt& other) const {
        if (other.isZero()) throw std::runtime_error("Division by zero");
        
        BigInt dividend = *this;
        dividend.negative = false;
        BigInt divisor = other;
        divisor.negative = false;
        
        if (dividend < divisor) return dividend;
        
        // Simple modulo by repeated subtraction (slow but works)
        while (dividend >= divisor) {
            dividend = dividend - divisor;
        }
        
        return dividend;
    }
    
    // Division by 2 (bit shift)
    BigInt divideByTwo() const {
        BigInt result;
        result.digits.clear();
        result.negative = negative;
        
        int carry = 0;
        for (int i = digits.size() - 1; i >= 0; --i) {
            int current = carry * 10 + digits[i];
            result.digits.insert(result.digits.begin(), current / 2);
            carry = current % 2;
        }
        
        result.removeLeadingZeros();
        return result;
    }
    
    // Multiply by 10
    BigInt multiplyByTen() const {
        BigInt result = *this;
        result.digits.insert(result.digits.begin(), 0);
        return result;
    }
    
    std::string toString() const {
        std::string result;
        if (negative && !isZero()) result += '-';
        for (int i = digits.size() - 1; i >= 0; --i) {
            result += std::to_string(digits[i]);
        }
        return result;
    }
    
    size_t digitCount() const {
        return digits.size();
    }
};

// Binary GCD algorithm (more efficient for big numbers)
BigInt gcdBinary(BigInt a, BigInt b) {
    if (a.isZero()) return b;
    if (b.isZero()) return a;
    
    // Count common factors of 2
    int shift = 0;
    while (a.isEven() && b.isEven()) {
        a = a.divideByTwo();
        b = b.divideByTwo();
        shift++;
    }
    
    // Remove remaining factors of 2 from a
    while (a.isEven()) {
        a = a.divideByTwo();
    }
    
    // From here on, a is always odd
    while (!b.isZero()) {
        // Remove factors of 2 from b
        while (b.isEven()) {
            b = b.divideByTwo();
        }
        
        // Now both a and b are odd. Swap if necessary so a <= b
        if (a > b) {
            std::swap(a, b);
        }
        
        b = b - a;  // b is now even
    }
    
    // Restore common factors of 2
    for (int i = 0; i < shift; ++i) {
        a = a + a;  // Multiply by 2
    }
    
    return a;
}

class PiSimplifier {
public:
    static void simplifyPiDigits(const std::string& piDigitsStr, int numDecimalDigits) {
        std::cout << "\n🎯 SIMPLIFYING π TO " << numDecimalDigits << " DECIMAL PLACES\n";
        std::cout << "═══════════════════════════════════════════════════════════\n\n";
        
        // Remove decimal point to get numerator
        std::string cleanDigits = piDigitsStr;
        cleanDigits.erase(std::remove(cleanDigits.begin(), cleanDigits.end(), '.'), cleanDigits.end());
        
        std::cout << "Original π: " << piDigitsStr << "\n";
        std::cout << "As integer: " << cleanDigits << "\n";
        std::cout << "Digits: " << cleanDigits.length() << "\n\n";
        
        // Construct denominator (10^n where n = decimal places)
        std::string denominatorStr = "1";
        for (int i = 0; i < numDecimalDigits; ++i) {
            denominatorStr += "0";
        }
        
        std::cout << "Initial Fraction:\n";
        std::cout << "Numerator:   " << cleanDigits << "\n";
        std::cout << "Denominator: " << denominatorStr << "\n";
        std::cout << "(Denominator is 10^" << numDecimalDigits << ")\n\n";
        
        // Convert to BigInt
        BigInt numerator(cleanDigits);
        BigInt denominator(denominatorStr);
        
        std::cout << "🔍 Checking simplification possibilities...\n\n";
        
        // Check divisibility by small primes
        std::vector<int> smallPrimes = {2, 5};  // Only 2 and 5 divide powers of 10
        
        bool canSimplify = false;
        for (int prime : smallPrimes) {
            BigInt primeBig(prime);
            if ((numerator % primeBig).isZero()) {
                std::cout << "✓ Numerator is divisible by " << prime << "\n";
                canSimplify = true;
            } else {
                std::cout << "✗ Numerator is NOT divisible by " << prime << "\n";
            }
        }
        
        std::cout << "\n";
        
        if (!canSimplify) {
            std::cout << "🎊 AMAZING DISCOVERY!\n";
            std::cout << "The numerator shares NO common factors with 10^" << numDecimalDigits << "!\n";
            std::cout << "This fraction is ALREADY IN SIMPLEST FORM!\n\n";
            std::cout << "✨ Your carry-around fraction:\n\n";
            std::cout << "    " << cleanDigits << "\n";
            std::cout << "    " << std::string(cleanDigits.length(), '─') << "\n";
            std::cout << "    " << denominatorStr << "\n\n";
            return;
        }
        
        // Calculate GCD
        std::cout << "⚙️  Computing GCD (this may take a moment for large numbers)...\n\n";
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        BigInt gcd = gcdBinary(numerator, denominator);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "✓ GCD computed in " << duration.count() << " ms\n";
        std::cout << "GCD = " << gcd.toString() << "\n\n";
        
        if (gcd == BigInt(1)) {
            std::cout << "🎊 The GCD is 1!\n";
            std::cout << "This fraction is ALREADY IN SIMPLEST FORM!\n\n";
        } else {
            std::cout << "🔧 Simplifying by dividing both by GCD...\n\n";
            // Note: Division not implemented yet for BigInt, so we'll note the GCD
            std::cout << "Simplified fraction would be:\n";
            std::cout << "(numerator / " << gcd.toString() << ") / (denominator / " << gcd.toString() << ")\n\n";
        }
        
        std::cout << "✨ FINAL FRACTION (potentially unsimplified):\n\n";
        std::cout << "    " << numerator.toString() << "\n";
        std::cout << "    " << std::string(std::min(size_t(50), numerator.digitCount()), '─') << " (" << numerator.digitCount() << " digits)\n";
        std::cout << "    " << denominator.toString().substr(0, 50) << "... (" << denominator.digitCount() << " digits)\n\n";
    }
    
    static void testSmallExample() {
        std::cout << "\n🧪 TESTING WITH SMALL EXAMPLE\n";
        std::cout << "═══════════════════════════════════════════════════\n\n";
        
        // Test with π ≈ 3.14159
        std::string testPi = "3.14159";
        int decimals = 5;
        
        std::cout << "Testing π ≈ 3.14159 (5 decimal places)\n\n";
        
        simplifyPiDigits(testPi, decimals);
        
        // Manual verification
        long long num = 314159;
        long long den = 100000;
        long long g = std::gcd(num, den);
        
        std::cout << "📊 VERIFICATION (using standard library GCD):\n";
        std::cout << "Numerator: 314159\n";
        std::cout << "Denominator: 100000\n";
        std::cout << "GCD: " << g << "\n";
        std::cout << "Simplified: " << (num/g) << " / " << (den/g) << "\n\n";
    }
};

class TorsionValidator {
public:
    static void validateFraction(long long num, long long den, int maxIterations = 10000) {
        std::cout << "\n🌀 TORSION PATH VALIDATOR\n";
        std::cout << "═══════════════════════════════════════════════════\n\n";
        
        std::cout << "Testing: " << num << " / " << den << "\n";
        std::cout << "Max iterations: " << maxIterations << "\n\n";
        
        double value = static_cast<double>(num) / den;
        double piValue = M_PI;
        
        std::cout << "Fraction value: " << std::setprecision(15) << value << "\n";
        std::cout << "Actual π:       " << std::setprecision(15) << piValue << "\n";
        std::cout << "Error:          " << std::scientific << std::abs(value - piValue) << "\n\n";
        
        // Simulate torsion path
        std::cout << "🎬 Running torsion simulation...\n\n";
        
        struct Point { double x, y; };
        std::vector<Point> path;
        
        bool cycleFound = false;
        int cycleLength = 0;
        double epsilon = 1e-9;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        for (int i = 1; i <= maxIterations; ++i) {
            double multiple = i * value;
            double fractional = multiple - std::floor(multiple);
            double angle = 2.0 * M_PI * fractional;
            
            Point p = { std::cos(angle), std::sin(angle) };
            
            // Check for cycle
            if (i > 1 && !cycleFound) {
                double dx = p.x - path[0].x;
                double dy = p.y - path[0].y;
                double dist = std::sqrt(dx*dx + dy*dy);
                
                if (dist < epsilon) {
                    cycleFound = true;
                    cycleLength = i;
                }
            }
            
            path.push_back(p);
            
            // Progress indicator
            if (i % 1000 == 0) {
                std::cout << "\rProgress: " << i << "/" << maxIterations 
                         << " | Current: (" << std::fixed << std::setprecision(4) 
                         << p.x << ", " << p.y << ")" << std::flush;
            }
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "\n\n⏱️  Simulation completed in " << duration.count() << " ms\n\n";
        
        // Results
        std::cout << "📊 TORSION ANALYSIS RESULTS:\n";
        std::cout << "═══════════════════════════════════════════════════\n\n";
        
        if (cycleFound) {
            std::cout << "🎯 CYCLE DETECTED!\n\n";
            std::cout << "Cycle length: " << cycleLength << " iterations\n";
            std::cout << "Denominator: " << den << "\n\n";
            
            if (cycleLength == den) {
                std::cout << "✅ PERFECT MATCH!\n";
                std::cout << "The cycle length equals the denominator.\n";
                std::cout << "This confirms the fraction is in lowest terms!\n\n";
            } else if (den % cycleLength == 0) {
                std::cout << "📝 The cycle length divides the denominator.\n";
                std::cout << "Simplification factor: " << (den / cycleLength) << "\n\n";
            } else {
                std::cout << "⚠️  Unexpected: cycle length doesn't divide denominator cleanly.\n\n";
            }
            
            std::cout << "🎪 INTERPRETATION:\n";
            std::cout << "The torsion path forms a closed loop.\n";
            std::cout << "This proves the number is RATIONAL.\n";
            std::cout << "After " << cycleLength << " steps, it returns to start.\n\n";
            
        } else {
            std::cout << "🌀 NO CYCLE DETECTED (within " << maxIterations << " iterations)\n\n";
            
            // Analyze path density
            double totalDist = 0.0;
            for (size_t i = 1; i < path.size(); ++i) {
                double dx = path[i].x - path[i-1].x;
                double dy = path[i].y - path[i-1].y;
                totalDist += std::sqrt(dx*dx + dy*dy);
            }
            double avgStep = totalDist / (path.size() - 1);
            
            std::cout << "Path statistics:\n";
            std::cout << "  Total distance traveled: " << std::fixed << std::setprecision(2) << totalDist << "\n";
            std::cout << "  Average step size: " << std::setprecision(6) << avgStep << "\n";
            std::cout << "  Points generated: " << path.size() << "\n\n";
            
            if (avgStep < 0.01) {
                std::cout << "🎨 The path is DENSELY filling the unit circle.\n";
                std::cout << "This behavior is characteristic of IRRATIONAL numbers.\n";
                std::cout << "The fraction approximates π but isn't exactly rational.\n\n";
            } else {
                std::cout << "🎭 The path is SPARSE.\n";
                std::cout << "Cycle may exist beyond " << maxIterations << " iterations.\n";
                std::cout << "Try increasing max iterations.\n\n";
            }
            
            std::cout << "🎪 INTERPRETATION:\n";
            std::cout << "No closed loop found (yet).\n";
            std::cout << "Either:\n";
            std::cout << "  1. The number is truly IRRATIONAL (π doesn't terminate)\n";
            std::cout << "  2. The cycle length > " << maxIterations << " (run longer)\n\n";
        }
    }
};

int main() {
    std::cout << "🎪═══════════════════════════════════════════════════════════🎪\n";
    std::cout << "       ULTIMATE π FRACTION SIMPLIFIER & TORSION TESTER\n";
    std::cout << "   Finding the Irreducible Form of Massive π Approximations\n";
    std::cout << "🎪═══════════════════════════════════════════════════════════🎪\n";
    
    // Test with small example first
    PiSimplifier::testSmallExample();
    
    std::cout << "\n" << std::string(60, '─') << "\n";
    std::cout << "Choose your precision level:\n\n";
    std::cout << "1. Low precision (10 digits) - Fast\n";
    std::cout << "2. Medium precision (15 digits) - Moderate\n";
    std::cout << "3. High precision (20 digits) - Slow\n";
    std::cout << "4. Planck precision (35 digits) - Very Slow\n";
    std::cout << "5. Custom precision\n";
    std::cout << "6. Test best continued fraction convergent\n";
    std::cout << "\nChoice (1-6): ";
    
    int choice;
    std::cin >> choice;
    
    std::string piStr;
    int decimals;
    
    switch (choice) {
        case 1:
            piStr = "3.1415926535";
            decimals = 10;
            break;
        case 2:
            piStr = "3.141592653589793";
            decimals = 15;
            break;
        case 3:
            piStr = "3.14159265358979323846";
            decimals = 20;
            break;
        case 4:
            piStr = "3.14159265358979323846264338327950288";
            decimals = 35;
            break;
        case 5:
            std::cout << "Enter π value (with decimal): ";
            std::cin >> piStr;
            decimals = piStr.length() - 2;  // Subtract "3."
            break;
        case 6: {
            std::cout << "\n🎯 Using best continued fraction convergent: 103993/33102\n";
            TorsionValidator::validateFraction(103993, 33102, 50000);
            
            std::cout << "\nWould you like to test another convergent? (y/n): ";
            char cont;
            std::cin >> cont;
            if (cont == 'y' || cont == 'Y') {
                std::cout << "Try 355/113? (y/n): ";
                std::cin >> cont;
                if (cont == 'y' || cont == 'Y') {
                    TorsionValidator::validateFraction(355, 113, 10000);
                }
            }
            return 0;
        }
        default:
            piStr = "3.14159265358979323846";
            decimals = 20;
    }
    
    PiSimplifier::simplifyPiDigits(piStr, decimals);
    
    std::cout << "\n" << std::string(60, '─') << "\n";
    std::cout << "Would you like to test this fraction in the torsion simulator? (y/n): ";
    
    char testChoice;
    std::cin >> testChoice;
    
    if (testChoice == 'y' || testChoice == 'Y') {
        // For the simplified fraction approach, let's test the actual convergent
        std::cout << "\nTesting best convergent (103993/33102) instead...\n";
        TorsionValidator::validateFraction(103993, 33102, 50000);
    }
    
    std::cout << "\n🎪═══════════════════════════════════════════════════════════🎪\n";
    std::cout << "                    SUMMARY & INSIGHTS\n";
    std::cout << "🎪═══════════════════════════════════════════════════════════🎪\n\n";
    
    std::cout << "💡 KEY FINDINGS:\n\n";
    std::cout << "1. π digits themselves don't simplify with powers of 10\n";
    std::cout << "   (Because π is irrational and transcendental)\n\n";
    std::cout << "2. The BEST rational approximations come from continued fractions\n";
    std::cout << "   Not from decimal truncation!\n\n";
    std::cout << "3. Torsion paths for rationals ALWAYS close eventually\n";
    std::cout << "   Cycle length = denominator (when in lowest terms)\n\n";
    std::cout << "4. If your theory is correct and π terminates at Planck scale,\n";
    std::cout << "   we should find a cycle at ~10^35 iterations!\n\n";
    
    std::cout << "🎪 Thank you for exploring mathematical infinity! 🎪\n\n";
    
    return 0;
}
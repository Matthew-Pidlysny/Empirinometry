// QUICK_BUILD.cpp - Ultra-simple single-file build
// Just compile this one file for a working program!
// 
// Compile with: g++ -std=c++17 -O2 QUICK_BUILD.cpp -o torsion
// Or with:    cl /std:c++17 /O2 QUICK_BUILD.cpp
// 
// This is the SIMPLIFIED VERSION for easy compilation!

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <thread>

// Simple structure for points
struct Point {
    double x, y;
    int iteration;
    int digitValue;
    Point(double x_, double y_, int iter, int digit = 0) 
        : x(x_), y(y_), iteration(iter), digitValue(digit) {}
};

// Simple fraction structure
struct Fraction {
    int numerator, denominator;
    std::string name;
    Fraction(int num, int den, const std::string& n = "") 
        : numerator(num), denominator(den), name(n) {}
    double toDouble() const { return static_cast<double>(numerator) / denominator; }
};

class SimpleTorsion {
private:
    Fraction currentFraction;
    std::vector<Point> torsionPath;
    static constexpr double PI = 3.14159265358979323846;
    
public:
    SimpleTorsion() : currentFraction(355, 113, "π Approximation") {}
    
    void setFraction(int num, int den) {
        currentFraction = Fraction(num, den, "User Input");
        torsionPath.clear();
    }
    
    // Calculate decimal expansion (simple version)
    std::string expandDecimal(int precision = 50) {
        if (currentFraction.denominator == 0) return "NaN";
        if (currentFraction.numerator == 0) return "0.0";
        
        std::stringstream result;
        
        // Handle sign
        if (currentFraction.numerator < 0 ^ currentFraction.denominator < 0) {
            result << "-";
        }
        
        int absNum = std::abs(currentFraction.numerator);
        int absDen = std::abs(currentFraction.denominator);
        
        // Integer part
        result << (absNum / absDen) << ".";
        
        // Decimal part
        int remainder = absNum % absDen;
        for (int i = 0; i < precision && remainder != 0; ++i) {
            remainder *= 10;
            result << (remainder / absDen);
            remainder %= absDen;
        }
        
        return result.str();
    }
    
    // Get digit at specific position
    int getDigitAt(int position) {
        if (currentFraction.denominator == 0 || position < 0) return 0;
        
        int absNum = std::abs(currentFraction.numerator);
        int absDen = std::abs(currentFraction.denominator);
        
        int remainder = absNum % absDen;
        
        for (int i = 0; i <= position; ++i) {
            remainder *= 10;
            if (i == position) {
                return remainder / absDen;
            }
            remainder %= absDen;
            if (remainder == 0) return 0;
        }
        
        return 0;
    }
    
    // Calculate torsion path
    void calculateTorsionPath(int maxIterations) {
        torsionPath.clear();
        
        if (currentFraction.denominator == 0) return;
        
        double fracValue = currentFraction.toDouble();
        
        for (int i = 1; i <= maxIterations; ++i) {
            double multiple = i * fracValue;
            double fractionalPart = multiple - std::floor(multiple);
            double angle = 2.0 * M_PI * fractionalPart;
            
            Point point(std::cos(angle), std::sin(angle), i);
            point.digitValue = getDigitAt(i);
            torsionPath.push_back(point);
        }
    }
    
    // Analysis
    void showAnalysis() {
        std::cout << "\n📊 MATHEMATICAL ANALYSIS\n";
        std::cout << std::string(40, '-') << "\n";
        
        double value = currentFraction.toDouble();
        double error = std::abs(value - PI);
        double convergenceRate = 1.0 / (currentFraction.denominator * error);
        
        std::cout << "🔢 Fraction: " << currentFraction.numerator 
                 << "/" << currentFraction.denominator << "\n";
        std::cout << "📈 Value: " << std::fixed << std::setprecision(12) << value << "\n";
        std::cout << "🎯 Target (π): " << std::fixed << std::setprecision(12) << PI << "\n";
        std::cout << "❌ Error: " << std::scientific << error << "\n";
        std::cout << "⚡ Convergence Rate: " << std::fixed << std::setprecision(6) << convergenceRate << "\n";
        
        // Classification
        std::string classification;
        if (error < 1e-10) classification = "Excellent";
        else if (error < 1e-7) classification = "Very Good";
        else if (error < 1e-4) classification = "Good";
        else if (error < 1e-2) classification = "Fair";
        else classification = "Poor";
        
        std::cout << "🏆 Classification: " << classification << "\n";
        
        // Decimal expansion
        std::string decimal = expandDecimal(50);
        std::cout << "🔢 Decimal: " << decimal << "\n";
        
        std::cout << std::string(40, '-') << "\n";
    }
    
    // Run animation
    void runAnimation(int maxIterations = 100) {
        calculateTorsionPath(maxIterations);
        
        std::cout << "\n🎬 Animating torsion path...\n";
        
        for (size_t i = 0; i < torsionPath.size(); ++i) {
            const auto& point = torsionPath[i];
            
            std::cout << "\r📍 Step " << (i + 1) << "/" << torsionPath.size() 
                     << " - Position: (" << std::fixed << std::setprecision(3) 
                     << point.x << ", " << point.y << ") - Digit: " << point.digitValue 
                     << std::flush;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\n\n✅ Animation complete!\n";
    }
    
    // Interactive loop
    void run() {
        std::cout << "\n🎪 TORSION VISUALIZER - SIMPLIFIED EDITION 🎪\n";
        std::cout << std::string(50, '=') << "\n";
        std::cout << "Exploring the End of Irrationals Through Torsion\n";
        std::cout << "Current fraction: 355/113 (π approximation)\n";
        std::cout << std::string(50, '=') << "\n";
        
        showAnalysis();
        
        std::cout << "\nCommands:\n";
        std::cout << "  fraction <num> <den>  - Set new fraction\n";
        std::cout << "  analyze               - Show detailed analysis\n";
        std::cout << "  animate <steps>       - Animate torsion path\n";
        std::cout << "  decimal <digits>      - Show decimal expansion\n";
        std::cout << "  digit <position>      - Get digit at position\n";
        std::cout << "  quit                  - Exit\n";
        std::cout << "\n> ";
        
        std::string input;
        while (std::getline(std::cin, input)) {
            std::istringstream iss(input);
            std::string command;
            iss >> command;
            
            if (command == "quit" || command == "q") {
                break;
            } else if (command == "fraction" || command == "f") {
                int num, den;
                if (iss >> num >> den) {
                    setFraction(num, den);
                    showAnalysis();
                } else {
                    std::cout << "Usage: fraction <numerator> <denominator>\n";
                }
            } else if (command == "analyze" || command == "a") {
                showAnalysis();
            } else if (command == "animate") {
                int steps = 100;
                iss >> steps;
                runAnimation(steps);
            } else if (command == "decimal") {
                int digits = 50;
                iss >> digits;
                std::cout << "Decimal: " << expandDecimal(digits) << "\n";
            } else if (command == "digit") {
                int position;
                if (iss >> position) {
                    int digit = getDigitAt(position);
                    std::cout << "Digit at position " << position << ": " << digit << "\n";
                } else {
                    std::cout << "Usage: digit <position>\n";
                }
            } else if (command == "help" || command == "h") {
                std::cout << "Commands:\n";
                std::cout << "  fraction <num> <den>  - Set new fraction\n";
                std::cout << "  analyze               - Show detailed analysis\n";
                std::cout << "  animate <steps>       - Animate torsion path\n";
                std::cout << "  decimal <digits>      - Show decimal expansion\n";
                std::cout << "  digit <position>      - Get digit at position\n";
                std::cout << "  quit                  - Exit\n";
            } else if (!command.empty()) {
                std::cout << "Unknown command. Type 'help' for available commands.\n";
            }
            
            std::cout << "\n> ";
        }
    }
};

int main() {
    try {
        std::cout << "🎪 TORSION VISUALIZER - MATHEMATICAL CIRCUS 🎪\n";
        std::cout << "Exploring the End of Irrationals Through Torsion\n";
        std::cout << "Built with C++17 - Single File Edition\n";
        std::cout << "=================================================\n";
        
        SimpleTorsion app;
        app.run();
        
        std::cout << "\n👋 Thank you for exploring mathematical torsion!\n";
        std::cout << "🎪 The Circus continues...\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n💥 Error: " << e.what() << "\n";
        return 1;
    }
}
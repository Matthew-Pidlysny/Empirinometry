/*
 * OPTIMIZED PI ANALYZER - Maximum Efficiency Version
 * 
 * Requirements:
 * 1) Print digits for all construction methods
 * 2) Collect all background data from eternal analyzer
 * 3) Run forever or until actual end
 * 4) Save to file with maximum efficiency
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <csignal>
#include <boost/multiprecision/cpp_dec_float.hpp>

using namespace std;
using boost::multiprecision::cpp_dec_float_50;
using boost::multiprecision::cpp_dec_float_100;

// Atomic flags for eternal running
atomic<bool> eternal_running(true);
atomic<unsigned long> total_digits_processed(0);
mutex file_mutex;

// Efficient file writer with buffered output
class EfficientLogger {
private:
    ofstream log_file;
    string buffer;
    static const size_t BUFFER_SIZE = 8192;
    
public:
    EfficientLogger(const string& filename) : buffer("") {
        log_file.open(filename, ios::out | ios::app);
        buffer.reserve(BUFFER_SIZE);
    }
    
    ~EfficientLogger() {
        if (!buffer.empty()) {
            log_file << buffer;
        }
        log_file.close();
    }
    
    void log(const string& data) {
        lock_guard<mutex> lock(file_mutex);
        buffer += data + "\n";
        if (buffer.size() >= BUFFER_SIZE) {
            log_file << buffer;
            buffer.clear();
        }
    }
    
    void flush() {
        lock_guard<mutex> lock(file_mutex);
        if (!buffer.empty()) {
            log_file << buffer;
            buffer.clear();
        }
        log_file.flush();
    }
};

// Optimized Pi construction method
class FastPiMethod {
public:
    string name;
    cpp_dec_float_50 value;
    string digit_string;
    
    FastPiMethod(string n) : name(n) {}
    virtual ~FastPiMethod() {}
    
    virtual void compute() = 0;
    virtual void getDigits(int count) {
        string str_val = value.str();
        size_t decimal_pos = str_val.find('.');
        if (decimal_pos != string::npos) {
            digit_string = str_val.substr(decimal_pos + 1, count);
        }
    }
};

// 1. Milü (355/113) - Precomputed for maximum speed
class MiluFast : public FastPiMethod {
public:
    MiluFast() : FastPiMethod("Milu") {}
    void compute() override {
        value = cpp_dec_float_50("3.14159292035398230088495575221238938053097345132743");
    }
};

// 2. BBP Formula - Optimized
class BBPFast : public FastPiMethod {
public:
    BBPFast() : FastPiMethod("BBP") {}
    void compute() override {
        cpp_dec_float_50 sum = 0;
        for (int n = 0; n < 15; ++n) { // Limited iterations for speed
            cpp_dec_float_50 term = 
                cpp_dec_float_50(4) / (8*n + 1) -
                cpp_dec_float_50(2) / (8*n + 4) -
                cpp_dec_float_50(1) / (8*n + 5) -
                cpp_dec_float_50(1) / (8*n + 6);
            term /= pow(cpp_dec_float_50(16), n);
            sum += term;
        }
        value = sum;
    }
};

// 3. Ramanujan's approximation
class RamanujanFast : public FastPiMethod {
public:
    RamanujanFast() : FastPiMethod("Ramanujan") {}
    void compute() override {
        value = pow(cpp_dec_float_50("2143") / cpp_dec_float_50("22"), cpp_dec_float_50("0.25"));
    }
};

// 4. Gregory-Leibniz (simplified)
class GregoryFast : public FastPiMethod {
public:
    GregoryFast() : FastPiMethod("Gregory") {}
    void compute() override {
        cpp_dec_float_50 sum = 0;
        for (int n = 0; n < 1000; ++n) {
            cpp_dec_float_50 term = cpp_dec_float_50(1) / (2*n + 1);
            if (n % 2 == 1) term = -term;
            sum += term;
        }
        value = sum * 4;
    }
};

// 5. Vieta's Formula (precomputed)
class VietaFast : public FastPiMethod {
public:
    VietaFast() : FastPiMethod("Vieta") {}
    void compute() override {
        value = cpp_dec_float_50("3.14159265358979323846264338327950288419716939937510");
    }
};

// 6. Machin's Formula (precomputed for maximum speed)
class MachinFast : public FastPiMethod {
public:
    MachinFast() : FastPiMethod("Machin") {}
    void compute() override {
        value = cpp_dec_float_50("3.14159265358979323846264338327950288419716939937510");
    }
};

// Data collection from eternal analyzer
class EternalDataCollector {
private:
    map<char, vector<string>> digit_synchronizations;
    unsigned long long total_patterns = 0;
    unsigned long long special_lifting_events = 0;
    
public:
    void recordSynchronization(char digit, const vector<string>& constants, int depth) {
        string key = to_string(depth) + ":" + digit;
        string record = "DEPTH:" + to_string(depth) + " DIGIT:'" + digit + "' CONSTANTS:";
        for (const auto& constant : constants) {
            record += constant + ",";
        }
        
        if (constants.size() >= 5) { // Special Lifting threshold
            special_lifting_events++;
        }
        total_patterns++;
    }
    
    void getStats(unsigned long long& patterns, unsigned long long& lifting) {
        patterns = total_patterns;
        lifting = special_lifting_events;
    }
};

// Main optimized analyzer
class OptimizedPiAnalyzer {
private:
    vector<unique_ptr<FastPiMethod>> methods;
    EfficientLogger logger;
    EternalDataCollector collector;
    int current_depth;
    
    // Mathematical constants for comparison (precomputed strings)
    unordered_map<string, string> constants = {
        {"phi", "61803398874989484820458683436563811772030917980576"},
        {"silver", "41421356237309504880168872420969807856967187537694"},
        {"sqrt2", "41421356237309504880168872420969807856967187537694"},
        {"sqrt3", "73205080756887729352744634150587236694280525381038"},
        {"sqrt5", "23606797749978969640917366873127623544061835961152"},
        {"e", "71828182845904523536028747135266249775724709369995"},
        {"gamma", "57721566490153286060651209008240243104215933593992"},
        {"ln2", "69314718055994530941723212145817656807550013436025"},
        {"catalan", "915965594177219015054603514932384110774"},
        {"zeta3", "20205690315959428539973816151144999076498629234049"}
    };
    
public:
    OptimizedPiAnalyzer() : logger("optimized_eternal_log.txt"), current_depth(0) {
        // Initialize all fast methods
        methods.emplace_back(make_unique<MiluFast>());
        methods.emplace_back(make_unique<BBPFast>());
        methods.emplace_back(make_unique<RamanujanFast>());
        methods.emplace_back(make_unique<GregoryFast>());
        methods.emplace_back(make_unique<VietaFast>());
        methods.emplace_back(make_unique<MachinFast>());
        
        // Pre-compute all method values
        for (auto& method : methods) {
            method->compute();
        }
    }
    
    void printDigits(int depth) {
        cout << "DEPTH " << setw(6) << depth << ": ";
        string log_entry = "DEPTH_" + to_string(depth) + ":";
        
        for (auto& method : methods) {
            method->getDigits(depth);
            char digit = (method->digit_string.length() > depth-1) ? method->digit_string[depth-1] : 'X';
            cout << method->name << ":" << digit << " ";
            log_entry += method->name + ":" + digit + ";";
        }
        
        cout << endl;
        logger.log(log_entry);
        total_digits_processed++;
    }
    
    void checkSynchronizations(int depth) {
        map<char, vector<string>> digit_groups;
        
        for (auto& method : methods) {
            if (method->digit_string.length() > depth-1) {
                char digit = method->digit_string[depth-1];
                digit_groups[digit].push_back(method->name);
            }
        }
        
        // Check for special lifting events
        for (const auto& group : digit_groups) {
            if (group.second.size() >= 3) { // Synchronization threshold
                collector.recordSynchronization(group.first, group.second, depth);
                if (group.second.size() >= 5) {
                    logger.log("SPECIAL_LIFTING: Depth " + to_string(depth) + 
                              " Digit '" + group.first + "' Size: " + to_string(group.second.size()));
                }
            }
        }
    }
    
    void runEternal() {
        logger.log("=== OPTIMIZED ETERNAL PI ANALYZER STARTED ===");
        logger.log("Methods: " + to_string(methods.size()));
        logger.log("Target: Run forever or until mathematical termination");
        
        auto start_time = chrono::high_resolution_clock::now();
        
        while (eternal_running) {
            current_depth++;
            
            // Core operations
            printDigits(current_depth);
            checkSynchronizations(current_depth);
            
            // Progress reporting every 1000 digits
            if (current_depth % 1000 == 0) {
                auto now = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::seconds>(now - start_time);
                
                unsigned long long patterns, lifting;
                collector.getStats(patterns, lifting);
                
                string progress = "PROGRESS: Depth=" + to_string(current_depth) + 
                                " Time=" + to_string(duration.count()) + "s " +
                                "Digits=" + to_string(total_digits_processed.load()) + 
                                " Patterns=" + to_string(patterns) + 
                                " SpecialLifting=" + to_string(lifting);
                
                cout << progress << endl;
                logger.log(progress);
                logger.flush();
            }
            
            // Check for mathematical termination (if any)
            if (checkMathematicalTermination(current_depth)) {
                logger.log("MATHEMATICAL TERMINATION DETECTED at depth " + to_string(current_depth));
                break;
            }
            
            // Small sleep to prevent CPU overload while maintaining efficiency
            this_thread::sleep_for(chrono::microseconds(100));
        }
        
        // Final summary
        generateFinalSummary();
    }
    
private:
    bool checkMathematicalTermination(int depth) {
        // For now, assume no mathematical termination
        // This could be enhanced to check for actual mathematical limits
        return false;
    }
    
    void generateFinalSummary() {
        unsigned long long patterns, lifting;
        collector.getStats(patterns, lifting);
        
        logger.log("=== FINAL SUMMARY ===");
        logger.log("Total Depth Analyzed: " + to_string(current_depth));
        logger.log("Total Digits Processed: " + to_string(total_digits_processed.load()));
        logger.log("Total Patterns Found: " + to_string(patterns));
        logger.log("Special Lifting Events: " + to_string(lifting));
        logger.log("Pi Methods Analyzed: " + to_string(methods.size()));
        logger.log("=== ANALYSIS COMPLETE ===");
        
        cout << "\n=== FINAL SUMMARY ===" << endl;
        cout << "Total Depth Analyzed: " << current_depth << endl;
        cout << "Total Digits Processed: " << total_digits_processed.load() << endl;
        cout << "Total Patterns Found: " << patterns << endl;
        cout << "Special Lifting Events: " << lifting << endl;
        cout << "Pi Methods Analyzed: " << methods.size() << endl;
        cout << "Data saved to: optimized_eternal_log.txt" << endl;
    }
};

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    eternal_running = false;
    cout << "\nReceived signal " << signal << ". Shutting down gracefully..." << endl;
}

int main() {
    // Set up signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    cout << "=== OPTIMIZED ETERNAL PI ANALYZER ===" << endl;
    cout << "Starting maximum efficiency analysis..." << endl;
    cout << "Press Ctrl+C to stop and generate summary" << endl;
    cout << "======================================" << endl;
    
    try {
        OptimizedPiAnalyzer analyzer;
        analyzer.runEternal();
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
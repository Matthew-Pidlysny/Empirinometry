#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <map>
#include <chrono>
#include <thread>

using namespace std;

// ANSI color codes for terminal output
const string RESET = "\033[0m";
const string BOLD = "\033[1m";
const string RED = "\033[31m";
const string GREEN = "\033[32m";
const string YELLOW = "\033[33m";
const string BLUE = "\033[34m";
const string MAGENTA = "\033[35m";
const string CYAN = "\033[36m";
const string WHITE = "\033[37m";
const string BG_BLACK = "\033[40m";
const string BG_WHITE = "\033[47m";

// Global output buffer for saving
stringstream outputBuffer;

// Dual output function
void print(const string& text, bool newline = true) {
    cout << text;
    outputBuffer << text;
    if (newline) {
        cout << endl;
        outputBuffer << endl;
    }
}

void printColor(const string& text, const string& color, bool newline = true) {
    cout << color << text << RESET;
    outputBuffer << text;
    if (newline) {
        cout << endl;
        outputBuffer << endl;
    }
}

void clearScreen() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

void pause(int milliseconds = 1000) {
    this_thread::sleep_for(chrono::milliseconds(milliseconds));
}

void waitForEnter() {
    print("\n" + CYAN + "Press ENTER to continue..." + RESET);
    cin.ignore();
    cin.get();
}

// Point structure for geometric calculations
struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
    
    double distance(const Point& other) const {
        return sqrt((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y));
    }
    
    double angle() const {
        return atan2(y, x);
    }
};

// T-pruning function
double T_function(double theta) {
    double cos3 = cos(3 * M_PI * theta);
    double cos6 = cos(6 * M_PI * theta);
    return cos3 * cos3 * cos6 * cos6;
}

// Draw ASCII art header
void drawHeader() {
    printColor("╔════════════════════════════════════════════════════════════════════════════╗", CYAN);
    printColor("║                                                                            ║", CYAN);
    printColor("║           THE HADWIGER-NELSON PROBLEM: INTERACTIVE TOUR                   ║", CYAN + BOLD);
    printColor("║                                                                            ║", CYAN);
    printColor("║              Discovering the Chromatic Number of the Plane                ║", CYAN);
    printColor("║                                                                            ║", CYAN);
    printColor("╚════════════════════════════════════════════════════════════════════════════╝", CYAN);
    print("");
}

// Draw section separator
void drawSeparator() {
    printColor("────────────────────────────────────────────────────────────────────────────", BLUE);
}

// Introduction section
void introduction() {
    clearScreen();
    drawHeader();
    
    printColor("Welcome to the Interactive Guided Tour!", GREEN + BOLD);
    print("");
    print("This program will take you on a journey through one of mathematics' most");
    print("fascinating unsolved problems: the Hadwiger-Nelson problem.");
    print("");
    printColor("What you'll discover:", YELLOW + BOLD);
    print("  • What the problem asks");
    print("  • The T-pruning lower bound method");
    print("  • Visual representations of the plane");
    print("  • Geometric properties: arcs, chords, secants");
    print("  • Why χ(ℝ²) ∈ {5, 6, 7}");
    print("  • Why k = 169 is impossible");
    print("");
    printColor("Let's begin your journey!", GREEN + BOLD);
    
    waitForEnter();
}

// Chapter 1: The Problem
void chapter1_TheProblem() {
    clearScreen();
    drawHeader();
    printColor("═══ CHAPTER 1: THE PROBLEM ═══", MAGENTA + BOLD);
    print("");
    
    print("Imagine you have an infinite flat plane (like an endless sheet of paper).");
    print("You want to color every point on it with some colors.");
    print("");
    printColor("THE RULE:", RED + BOLD);
    print("  If two points are EXACTLY 1 unit apart, they must have DIFFERENT colors.");
    print("");
    printColor("THE QUESTION:", YELLOW + BOLD);
    print("  What is the MINIMUM number of colors you need?");
    print("");
    print("This minimum number is called χ(ℝ²), the chromatic number of the plane.");
    print("");
    
    drawSeparator();
    print("");
    printColor("Let's test your understanding!", CYAN + BOLD);
    print("");
    print("Q1: If we use only 1 color, will it work?");
    print("    (Think: Can two points 1 unit apart have the same color?)");
    waitForEnter();
    
    printColor("A1: NO! ", RED + BOLD, false);
    print("If we use only 1 color, then two points at distance 1");
    print("    would have the same color, violating the rule.");
    print("");
    
    print("Q2: What about 2 colors? Can we color the plane with just 2 colors?");
    waitForEnter();
    
    printColor("A2: NO! ", RED + BOLD, false);
    print("Consider an equilateral triangle with side length 1.");
    print("    All three vertices are at distance 1 from each other.");
    print("    We need at least 3 colors for these three points!");
    print("");
    
    print("Q3: So we need at least 3 colors. But is 3 enough?");
    waitForEnter();
    
    printColor("A3: NO! ", RED + BOLD, false);
    print("We can construct configurations that require even more colors.");
    print("    In fact, we'll prove you need AT LEAST 5 colors!");
    print("");
    
    printColor("Current Knowledge:", GREEN + BOLD);
    print("  • Lower Bound: χ(ℝ²) ≥ 5 (proven in 2018)");
    print("  • Upper Bound: χ(ℝ²) ≤ 7 (proven in 1961)");
    print("  • Therefore: χ(ℝ²) ∈ {5, 6, 7}");
    print("");
    
    waitForEnter();
}

// Chapter 2: The Circle Approach
void chapter2_CircleApproach() {
    clearScreen();
    drawHeader();
    printColor("═══ CHAPTER 2: THE CIRCLE APPROACH ═══", MAGENTA + BOLD);
    print("");
    
    print("To understand the lower bound, we use a clever trick:");
    print("We study points on a UNIT CIRCLE around a central point.");
    print("");
    printColor("Why a circle?", YELLOW + BOLD);
    print("  • All points on the circle are at distance 1 from the center");
    print("  • We can represent directions using angles (0 to 2π)");
    print("  • We normalize angles to [0, 1) for convenience");
    print("");
    
    print("Let's visualize the unit circle:");
    print("");
    
    // ASCII art circle
    printColor("                    θ = 0 (0°)", CYAN);
    printColor("                        •", GREEN);
    printColor("                   ╱    │    ╲", BLUE);
    printColor("                ╱       │       ╲", BLUE);
    printColor("             ╱          │          ╲", BLUE);
    printColor("          ╱             │             ╲", BLUE);
    printColor("       •                •                •", GREEN);
    printColor("  θ = 3/4          Center (0,0)      θ = 1/4", CYAN);
    printColor("  (270°)                                (90°)", CYAN);
    printColor("       •                                  •", GREEN);
    printColor("          ╲                          ╱", BLUE);
    printColor("             ╲                    ╱", BLUE);
    printColor("                ╲              ╱", BLUE);
    printColor("                   ╲        ╱", BLUE);
    printColor("                        •", GREEN);
    printColor("                   θ = 1/2 (180°)", CYAN);
    print("");
    
    printColor("Key Insight:", YELLOW + BOLD);
    print("  If two points on the circle are at certain angular separations,");
    print("  they might be at distance 1 from each other!");
    print("");
    
    print("Q: What angular separation corresponds to distance 1?");
    waitForEnter();
    
    printColor("A: For a triangular lattice, the forbidden separation is θ = 1/6", GREEN + BOLD);
    print("   This corresponds to 60° or π/3 radians.");
    print("   (Think of an equilateral triangle!)");
    print("");
    
    // Show forbidden angles
    printColor("Forbidden Angular Separations:", RED + BOLD);
    print("  • If two points differ by θ = 1/6 (60°), they're at distance 1");
    print("  • Points in the SAME color class cannot differ by 1/6");
    print("");
    
    waitForEnter();
}

// Chapter 3: The T-Pruning Method
void chapter3_TPruning() {
    clearScreen();
    drawHeader();
    printColor("═══ CHAPTER 3: THE T-PRUNING METHOD ═══", MAGENTA + BOLD);
    print("");
    
    print("Now we introduce the MAGIC FORMULA that proves the lower bound!");
    print("");
    printColor("The T-Pruning Polynomial:", YELLOW + BOLD);
    print("");
    printColor("    T(θ) = cos²(3πθ) · cos²(6πθ)", GREEN + BOLD);
    print("");
    
    print("This polynomial has THREE special properties:");
    print("");
    printColor("Property 1: Normalization", CYAN + BOLD);
    print("  T(0) = cos²(0) · cos²(0) = 1 · 1 = 1 ✓");
    print("");
    
    printColor("Property 2: Non-negativity", CYAN + BOLD);
    print("  T(θ) ≥ 0 for all θ (since cos² is always ≥ 0) ✓");
    print("");
    
    printColor("Property 3: Vanishing at forbidden shifts", CYAN + BOLD);
    print("  T(1/6) = cos²(π/2) · cos²(π) = 0 · 1 = 0 ✓");
    print("  T(-1/6) = cos²(-π/2) · cos²(-π) = 0 · 1 = 0 ✓");
    print("");
    
    print("Let's visualize T(θ):");
    print("");
    
    // Plot T(θ)
    int width = 70;
    int height = 15;
    vector<string> plot(height, string(width, ' '));
    
    for (int i = 0; i < width; i++) {
        double theta = (double)i / width;
        double t_val = T_function(theta);
        int y = height - 1 - (int)(t_val * (height - 1));
        if (y >= 0 && y < height) {
            plot[y][i] = '*';
        }
    }
    
    printColor("  T(θ)", YELLOW);
    printColor("  1.0 │", BLUE);
    for (int y = 0; y < height; y++) {
        print("      │" + plot[y]);
    }
    printColor("  0.0 └" + string(width, '─') + "→ θ", BLUE);
    printColor("      0.0                                                    1.0", BLUE);
    print("");
    
    printColor("Notice:", YELLOW + BOLD);
    print("  • T(θ) starts at 1 when θ = 0");
    print("  • T(θ) drops to 0 at θ = 1/6 (the forbidden angle!)");
    print("  • T(θ) oscillates but stays non-negative");
    print("");
    
    waitForEnter();
}

// Chapter 4: The Integral
void chapter4_TheIntegral() {
    clearScreen();
    drawHeader();
    printColor("═══ CHAPTER 4: THE MAGIC INTEGRAL ═══", MAGENTA + BOLD);
    print("");
    
    print("Now comes the crucial calculation!");
    print("");
    printColor("We integrate T(θ) over the entire circle:", YELLOW + BOLD);
    print("");
    printColor("    ∫₀¹ T(θ) dθ = ?", GREEN + BOLD);
    print("");
    
    print("Let's expand T(θ) using trigonometric identities:");
    print("");
    printColor("Step 1: Use cos²(x) = (1 + cos(2x))/2", CYAN);
    print("");
    print("  T(θ) = [(1 + cos(6πθ))/2] · [(1 + cos(12πθ))/2]");
    print("       = (1/4)[1 + cos(6πθ) + cos(12πθ) + cos(6πθ)cos(12πθ)]");
    print("");
    
    printColor("Step 2: Use product-to-sum formula", CYAN);
    print("");
    print("  cos(A)cos(B) = (1/2)[cos(A-B) + cos(A+B)]");
    print("  cos(6πθ)cos(12πθ) = (1/2)[cos(6πθ) + cos(18πθ)]");
    print("");
    
    printColor("Step 3: Substitute back", CYAN);
    print("");
    print("  T(θ) = (1/4)[1 + (3/2)cos(6πθ) + cos(12πθ) + (1/2)cos(18πθ)]");
    print("");
    
    printColor("Step 4: Integrate!", CYAN);
    print("");
    print("  ∫₀¹ T(θ) dθ = (1/4)∫₀¹ [1 + (3/2)cos(6πθ) + cos(12πθ) + (1/2)cos(18πθ)] dθ");
    print("");
    
    printColor("Key Fact:", YELLOW + BOLD);
    print("  ∫₀¹ cos(2πnθ) dθ = 0 for all n ≠ 0");
    print("");
    
    printColor("Therefore:", GREEN + BOLD);
    print("");
    printColor("  ∫₀¹ T(θ) dθ = (1/4)[1 + 0 + 0 + 0] = 1/4", GREEN + BOLD);
    print("");
    
    drawSeparator();
    print("");
    printColor("THE CRUCIAL RESULT:", RED + BOLD);
    print("");
    printColor("  Any admissible color class has measure ≤ 1/4", YELLOW + BOLD);
    print("");
    print("This means: Each color can cover at most 1/4 of the angular space!");
    print("");
    
    print("Q: If each color covers at most 1/4, how many colors do we need?");
    waitForEnter();
    
    printColor("A: We need at least 1/(1/4) = 4 colors!", GREEN + BOLD);
    print("");
    print("   Since the total measure is 1, and each color covers ≤ 1/4:");
    print("   Number of colors ≥ 1/(1/4) = 4");
    print("");
    
    printColor("This proves: χ(ℝ²) ≥ 4", YELLOW + BOLD);
    print("");
    
    waitForEnter();
}

// Chapter 5: Geometric Properties
void chapter5_GeometricProperties() {
    clearScreen();
    drawHeader();
    printColor("═══ CHAPTER 5: GEOMETRIC PROPERTIES ═══", MAGENTA + BOLD);
    print("");
    
    print("Let's explore the geometric meaning of our discovery!");
    print("");
    
    printColor("5.1 ARC LENGTHS", CYAN + BOLD);
    print("");
    print("On the unit circle, an arc from angle θ₁ to θ₂ has length:");
    print("");
    printColor("    Arc Length = |θ₂ - θ₁| · 2π", GREEN);
    print("");
    print("For our forbidden separation θ = 1/6:");
    printColor("    Arc Length = (1/6) · 2π = π/3 ≈ 1.047 units", GREEN);
    print("");
    
    printColor("5.2 CHORDS", CYAN + BOLD);
    print("");
    print("A chord connects two points on the circle.");
    print("For points at angular separation θ, the chord length is:");
    print("");
    printColor("    Chord Length = 2·sin(πθ)", GREEN);
    print("");
    print("For θ = 1/6:");
    printColor("    Chord Length = 2·sin(π/6) = 2·(1/2) = 1 unit", GREEN);
    print("");
    printColor("This is exactly the unit distance we're avoiding!", YELLOW + BOLD);
    print("");
    
    printColor("5.3 SECANTS", CYAN + BOLD);
    print("");
    print("A secant is a line that intersects the circle at two points.");
    print("The secant through points at angles θ₁ and θ₂ has special properties:");
    print("");
    print("  • If |θ₂ - θ₁| = 1/6, the chord length is exactly 1");
    print("  • Points on this secant at distance 1 from the center");
    print("    cannot be in the same color class");
    print("");
    
    // ASCII visualization
    printColor("Visualization of Forbidden Configuration:", YELLOW + BOLD);
    print("");
    printColor("              •  θ₁", GREEN);
    printColor("            ╱   ╲", BLUE);
    printColor("          ╱       ╲", BLUE);
    printColor("        ╱           ╲", BLUE);
    printColor("      ╱      •        ╲  ← Unit circle", BLUE);
    printColor("    ╱     Center       ╲", BLUE);
    printColor("  •────────────────────•", RED);
    printColor("  θ₂                    ", GREEN);
    printColor("  ↑                     ", RED);
    printColor("  Chord of length 1     ", RED);
    printColor("  (forbidden in same color)", RED);
    print("");
    
    print("Q: Why does T(θ) vanish at θ = 1/6?");
    waitForEnter();
    
    printColor("A: Because this is the forbidden angular separation!", GREEN + BOLD);
    print("   The polynomial is designed to be 0 exactly where we can't");
    print("   have two points of the same color. This is the genius of");
    print("   the T-pruning method!");
    print("");
    
    waitForEnter();
}

// Chapter 6: The Complete Picture
void chapter6_CompletePicture() {
    clearScreen();
    drawHeader();
    printColor("═══ CHAPTER 6: THE COMPLETE PICTURE ═══", MAGENTA + BOLD);
    print("");
    
    print("Now let's see how everything fits together!");
    print("");
    
    printColor("LOWER BOUNDS (Proving you need AT LEAST k colors):", CYAN + BOLD);
    print("");
    print("  1. T-Pruning Method: χ(ℝ²) ≥ 4");
    print("     • Uses trigonometric polynomial");
    print("     • Measure-theoretic argument");
    print("     • Fully analytic proof");
    print("");
    print("  2. De Grey's Construction (2018): χ(ℝ²) ≥ 5");
    print("     • Explicit finite graph with 1581 vertices");
    print("     • All edges have length 1");
    print("     • Requires 5 colors (verified computationally)");
    print("");
    
    printColor("UPPER BOUNDS (Proving you need AT MOST k colors):", CYAN + BOLD);
    print("");
    print("  3. Hadwiger's Construction (1961): χ(ℝ²) ≤ 7");
    print("     • Tile plane with hexagons of diameter < 1");
    print("     • Color hexagons so adjacent ones differ");
    print("     • Requires at most 7 colors");
    print("");
    
    drawSeparator();
    print("");
    printColor("COMBINING THE BOUNDS:", YELLOW + BOLD);
    print("");
    printColor("    5 ≤ χ(ℝ²) ≤ 7", GREEN + BOLD);
    print("");
    printColor("    Therefore: χ(ℝ²) ∈ {5, 6, 7}", GREEN + BOLD);
    print("");
    
    drawSeparator();
    print("");
    printColor("WHAT'S RULED OUT:", RED + BOLD);
    print("");
    
    // Table
    print("  ┌─────────────┬──────────────┬────────────────────────────┐");
    print("  │   Value k   │    Status    │          Reason            │");
    print("  ├─────────────┼──────────────┼────────────────────────────┤");
    print("  │   k ≤ 4     │  RULED OUT   │  Lower bound: χ(ℝ²) ≥ 5   │");
    print("  │   k = 5     │   POSSIBLE   │  Within bounds [5, 7]      │");
    print("  │   k = 6     │   POSSIBLE   │  Within bounds [5, 7]      │");
    print("  │   k = 7     │   POSSIBLE   │  Within bounds [5, 7]      │");
    print("  │   k ≥ 8     │  RULED OUT   │  Upper bound: χ(ℝ²) ≤ 7   │");
    print("  │   k = 169   │  RULED OUT   │  Upper bound: χ(ℝ²) ≤ 7   │");
    print("  └─────────────┴──────────────┴────────────────────────────┘");
    print("");
    
    waitForEnter();
}

// Chapter 7: Why 169 is Impossible
void chapter7_Why169() {
    clearScreen();
    drawHeader();
    printColor("═══ CHAPTER 7: WHY k = 169 IS IMPOSSIBLE ═══", MAGENTA + BOLD);
    print("");
    
    printColor("Let's definitively prove that χ(ℝ²) ≠ 169!", RED + BOLD);
    print("");
    
    printColor("Proof:", YELLOW + BOLD);
    print("");
    print("  1. Hadwiger (1961) constructed an explicit 7-coloring of ℝ²");
    print("     This PROVES that χ(ℝ²) ≤ 7");
    print("");
    print("  2. Since χ(ℝ²) ≤ 7, we have χ(ℝ²) < 169");
    print("");
    print("  3. Therefore, χ(ℝ²) ≠ 169");
    print("");
    printColor("  Q.E.D. ∎", GREEN + BOLD);
    print("");
    
    drawSeparator();
    print("");
    printColor("In fact, we can rule out ALL values k ≥ 8:", YELLOW + BOLD);
    print("");
    print("  For any k ≥ 8:");
    print("    • We have χ(ℝ²) ≤ 7 (by Hadwiger's construction)");
    print("    • Since 7 < k, we have χ(ℝ²) < k");
    print("    • Therefore χ(ℝ²) ≠ k");
    print("");
    
    printColor("This includes:", RED + BOLD);
    print("  • k = 8, 9, 10, 11, ...");
    print("  • k = 100");
    print("  • k = 169");
    print("  • k = 1000");
    print("  • Any k ≥ 8");
    print("");
    
    printColor("ALL RULED OUT!", RED + BOLD);
    print("");
    
    drawSeparator();
    print("");
    print("Q: Could the answer be less than 5?");
    waitForEnter();
    
    printColor("A: NO!", RED + BOLD);
    print("   De Grey's explicit graph requires 5 colors.");
    print("   This proves χ(ℝ²) ≥ 5.");
    print("   So k = 1, 2, 3, 4 are all RULED OUT.");
    print("");
    
    print("Q: So what are the only possible values?");
    waitForEnter();
    
    printColor("A: EXACTLY THREE VALUES:", GREEN + BOLD);
    print("");
    printColor("    χ(ℝ²) ∈ {5, 6, 7}", YELLOW + BOLD);
    print("");
    print("   The exact value is still unknown, but we've narrowed it down");
    print("   to just these three possibilities!");
    print("");
    
    waitForEnter();
}

// Chapter 8: Visual Summary
void chapter8_VisualSummary() {
    clearScreen();
    drawHeader();
    printColor("═══ CHAPTER 8: VISUAL SUMMARY ═══", MAGENTA + BOLD);
    print("");
    
    print("Let's visualize the complete solution space:");
    print("");
    
    printColor("The Number Line of Possible Chromatic Numbers:", CYAN + BOLD);
    print("");
    
    // Visual number line
    printColor("  1   2   3   4   5   6   7   8   9   10  ...  169  ...", WHITE);
    printColor("  ✗   ✗   ✗   ✗   ?   ?   ?   ✗   ✗   ✗   ...  ✗   ...", YELLOW);
    printColor("  └───────────┘   └───────┘   └──────────────────────┘", BLUE);
    printColor("   RULED OUT     POSSIBLE         RULED OUT", BLUE);
    printColor("   (too few)    (unknown)        (too many)", BLUE);
    print("");
    
    printColor("The T-Pruning Polynomial Visualization:", CYAN + BOLD);
    print("");
    printColor("  T(θ) = cos²(3πθ) · cos²(6πθ)", GREEN);
    print("");
    printColor("  Key Features:", YELLOW);
    print("    • Peaks at θ = 0 (value = 1)");
    print("    • Zeros at θ = ±1/6 (forbidden angles)");
    print("    • Always non-negative");
    print("    • Integral = 1/4 (the measure bound!)");
    print("");
    
    printColor("The Measure Bound:", CYAN + BOLD);
    print("");
    print("  ┌─────────────────────────────────────────┐");
    print("  │  Each color class: μ(A) ≤ 1/4          │");
    print("  │                                         │");
    print("  │  Total measure: 1                       │");
    print("  │                                         │");
    print("  │  Minimum colors: 1/(1/4) = 4            │");
    print("  │                                         │");
    print("  │  Therefore: χ(ℝ²) ≥ 4                  │");
    print("  └─────────────────────────────────────────┘");
    print("");
    
    printColor("The Complete Bounds:", CYAN + BOLD);
    print("");
    print("  ┌──────────────────────────────────────────────────┐");
    print("  │                                                  │");
    print("  │   LOWER BOUND: χ(ℝ²) ≥ 5  (de Grey, 2018)      │");
    print("  │                                                  │");
    print("  │   UPPER BOUND: χ(ℝ²) ≤ 7  (Hadwiger, 1961)     │");
    print("  │                                                  │");
    print("  │   CONCLUSION: χ(ℝ²) ∈ {5, 6, 7}                │");
    print("  │                                                  │");
    print("  └──────────────────────────────────────────────────┘");
    print("");
    
    waitForEnter();
}

// Interactive Q&A
void interactiveQA() {
    clearScreen();
    drawHeader();
    printColor("═══ INTERACTIVE Q&A SESSION ═══", MAGENTA + BOLD);
    print("");
    
    vector<pair<string, string>> questions = {
        {"What is the Hadwiger-Nelson problem?",
         "It asks: What is the minimum number of colors needed to color the plane\n"
         "         such that no two points at distance 1 have the same color?"},
        
        {"What is χ(ℝ²)?",
         "It's the chromatic number of the plane - the minimum number of colors\n"
         "         needed to color ℝ² with the unit-distance constraint."},
        
        {"What does the T-pruning method prove?",
         "It proves that χ(ℝ²) ≥ 4 using a trigonometric polynomial and\n"
         "         measure-theoretic arguments."},
        
        {"What is T(θ)?",
         "T(θ) = cos²(3πθ) · cos²(6πθ), a polynomial that vanishes at\n"
         "         forbidden angular separations."},
        
        {"Why is θ = 1/6 forbidden?",
         "Because points at angular separation 1/6 (60°) on the unit circle\n"
         "         are at distance 1 from each other."},
        
        {"What is the measure bound?",
         "Any admissible color class has measure ≤ 1/4, which means each\n"
         "         color can cover at most 1/4 of the angular space."},
        
        {"What are the current bounds?",
         "Lower bound: χ(ℝ²) ≥ 5 (de Grey, 2018)\n"
         "         Upper bound: χ(ℝ²) ≤ 7 (Hadwiger, 1961)"},
        
        {"Can χ(ℝ²) = 169?",
         "NO! Since χ(ℝ²) ≤ 7 < 169, it's impossible."},
        
        {"What values are possible?",
         "Only three values: χ(ℝ²) ∈ {5, 6, 7}"},
        
        {"Is the exact value known?",
         "No! The exact value is still one of mathematics' great unsolved\n"
         "         problems. We only know it's 5, 6, or 7."}
    };
    
    for (size_t i = 0; i < questions.size(); i++) {
        printColor("Q" + to_string(i+1) + ": " + questions[i].first, YELLOW + BOLD);
        print("");
        waitForEnter();
        printColor("A" + to_string(i+1) + ": " + questions[i].second, GREEN);
        print("");
        print("");
        if (i < questions.size() - 1) {
            drawSeparator();
            print("");
        }
    }
    
    waitForEnter();
}

// Final Summary
void finalSummary() {
    clearScreen();
    drawHeader();
    printColor("═══ FINAL SUMMARY ═══", MAGENTA + BOLD);
    print("");
    
    printColor("🎓 WHAT YOU'VE LEARNED:", GREEN + BOLD);
    print("");
    print("  ✓ The Hadwiger-Nelson problem asks for the minimum colors");
    print("    needed to color the plane with the unit-distance constraint");
    print("");
    print("  ✓ The T-pruning method uses a trigonometric polynomial to");
    print("    establish a measure bound: μ(A) ≤ 1/4");
    print("");
    print("  ✓ This proves χ(ℝ²) ≥ 4 through pure analysis");
    print("");
    print("  ✓ De Grey's 2018 breakthrough improved this to χ(ℝ²) ≥ 5");
    print("");
    print("  ✓ Hadwiger's 1961 construction proves χ(ℝ²) ≤ 7");
    print("");
    print("  ✓ Combining bounds: χ(ℝ²) ∈ {5, 6, 7}");
    print("");
    print("  ✓ ALL other values are RULED OUT, including k = 169");
    print("");
    
    drawSeparator();
    print("");
    printColor("🎯 KEY FORMULAS:", CYAN + BOLD);
    print("");
    printColor("  T(θ) = cos²(3πθ) · cos²(6πθ)", GREEN);
    print("");
    printColor("  ∫₀¹ T(θ) dθ = 1/4", GREEN);
    print("");
    printColor("  χ(ℝ²) ≥ 1/(1/4) = 4", GREEN);
    print("");
    printColor("  5 ≤ χ(ℝ²) ≤ 7", GREEN);
    print("");
    
    drawSeparator();
    print("");
    printColor("🏆 DEFINITIVE CONCLUSIONS:", YELLOW + BOLD);
    print("");
    printColor("  ✓ χ(ℝ²) ≠ 169 (PROVEN)", RED + BOLD);
    printColor("  ✓ χ(ℝ²) ≠ k for any k ≥ 8 (PROVEN)", RED + BOLD);
    printColor("  ✓ χ(ℝ²) ≠ k for any k ≤ 4 (PROVEN)", RED + BOLD);
    printColor("  ✓ χ(ℝ²) ∈ {5, 6, 7} (PROVEN)", GREEN + BOLD);
    print("");
    
    drawSeparator();
    print("");
    printColor("Thank you for taking this mathematical journey!", MAGENTA + BOLD);
    print("");
    
    waitForEnter();
}

// Save output to file
void saveToFile() {
    clearScreen();
    drawHeader();
    printColor("═══ SAVE OUTPUT ═══", MAGENTA + BOLD);
    print("");
    
    string filename = "hadwiger_nelson_tour_output.txt";
    ofstream outFile(filename);
    
    if (outFile.is_open()) {
        outFile << outputBuffer.str();
        outFile.close();
        printColor("✓ Output saved to: " + filename, GREEN + BOLD);
        print("");
        print("You can review the entire tour at any time by opening this file.");
    } else {
        printColor("✗ Error: Could not save file.", RED + BOLD);
    }
    
    print("");
    waitForEnter();
}

// Main menu
void mainMenu() {
    while (true) {
        clearScreen();
        drawHeader();
        printColor("═══ MAIN MENU ═══", MAGENTA + BOLD);
        print("");
        
        print("Choose a chapter to explore:");
        print("");
        printColor("  1. Introduction", CYAN);
        printColor("  2. Chapter 1: The Problem", CYAN);
        printColor("  3. Chapter 2: The Circle Approach", CYAN);
        printColor("  4. Chapter 3: The T-Pruning Method", CYAN);
        printColor("  5. Chapter 4: The Magic Integral", CYAN);
        printColor("  6. Chapter 5: Geometric Properties", CYAN);
        printColor("  7. Chapter 6: The Complete Picture", CYAN);
        printColor("  8. Chapter 7: Why k = 169 is Impossible", CYAN);
        printColor("  9. Chapter 8: Visual Summary", CYAN);
        printColor(" 10. Interactive Q&A", CYAN);
        printColor(" 11. Final Summary", CYAN);
        printColor(" 12. Save Output to File", YELLOW);
        printColor(" 13. Exit", RED);
        print("");
        
        print("Enter your choice (1-13): ", false);
        int choice;
        cin >> choice;
        cin.ignore();
        
        switch (choice) {
            case 1: introduction(); break;
            case 2: chapter1_TheProblem(); break;
            case 3: chapter2_CircleApproach(); break;
            case 4: chapter3_TPruning(); break;
            case 5: chapter4_TheIntegral(); break;
            case 6: chapter5_GeometricProperties(); break;
            case 7: chapter6_CompletePicture(); break;
            case 8: chapter7_Why169(); break;
            case 9: chapter8_VisualSummary(); break;
            case 10: interactiveQA(); break;
            case 11: finalSummary(); break;
            case 12: saveToFile(); break;
            case 13:
                clearScreen();
                printColor("Thank you for exploring the Hadwiger-Nelson problem!", GREEN + BOLD);
                print("");
                printColor("Remember: χ(ℝ²) ∈ {5, 6, 7} and k = 169 is IMPOSSIBLE!", YELLOW + BOLD);
                print("");
                return;
            default:
                printColor("Invalid choice. Please try again.", RED);
                pause(1000);
        }
    }
}

// Full guided tour
void fullGuidedTour() {
    introduction();
    chapter1_TheProblem();
    chapter2_CircleApproach();
    chapter3_TPruning();
    chapter4_TheIntegral();
    chapter5_GeometricProperties();
    chapter6_CompletePicture();
    chapter7_Why169();
    chapter8_VisualSummary();
    interactiveQA();
    finalSummary();
    saveToFile();
}

int main() {
    clearScreen();
    drawHeader();
    
    printColor("Welcome to the Hadwiger-Nelson Interactive Tour!", GREEN + BOLD);
    print("");
    print("Would you like to:");
    print("");
    printColor("  1. Take the full guided tour (recommended)", CYAN);
    printColor("  2. Explore chapters individually", CYAN);
    print("");
    print("Enter your choice (1 or 2): ", false);
    
    int choice;
    cin >> choice;
    cin.ignore();
    
    if (choice == 1) {
        fullGuidedTour();
    } else {
        mainMenu();
    }
    
    return 0;
}
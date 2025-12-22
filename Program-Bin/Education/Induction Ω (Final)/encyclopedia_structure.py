"""
Mathematical Induction Encyclopedia Structure for Induction Ω
Comprehensive encyclopedia covering all forms and applications of induction
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

class InductionType(Enum):
    """Enumeration of different types of induction"""
    MATHEMATICAL_WEAK = "mathematical_weak"
    MATHEMATICAL_STRONG = "mathematical_strong"
    MATHEMATICAL_COMPLETE = "mathematical_complete"
    STRUCTURAL = "structural"
    TRANSFINITE = "transfinite"
    WELL_FOUNDED = "well_founded"
    NOETHERIAN = "noetherian"
    COURSE_OF_VALUES = "course_of_values"
    PREFIX = "prefix"
    FORWARD_BACKWARD = "forward_backward"
    INFINITE_DESCENT = "infinite_descent"
    ELECTROMAGNETIC = "electromagnetic"
    MUTUAL = "mutual"
    SELF = "self"
    ELECTROSTATIC = "electrostatic"
    MAGNETIC = "magnetic"

class DifficultyLevel(Enum):
    """Difficulty levels for encyclopedia content"""
    ELEMENTARY = "elementary"      # Grades 3-5
    MIDDLE_SCHOOL = "middle_school"  # Grades 6-8
    HIGH_SCHOOL = "high_school"    # Grades 9-12
    UNDERGRADUATE = "undergraduate"  # Grades 13-16
    GRADUATE = "graduate"          # Grades 17-20
    RESEARCH = "research"          # Advanced research level

@dataclass
class EncyclopediaEntry:
    """Structure for encyclopedia entries"""
    id: str
    title: str
    induction_type: InductionType
    difficulty_level: DifficultyLevel
    summary: str
    detailed_content: str
    mathematical_formulations: List[str]
    examples: List[Dict[str, Any]]
    applications: List[str]
    historical_context: Dict[str, Any]
    related_concepts: List[str]
    visualizations: List[Dict[str, Any]]
    exercises: List[Dict[str, Any]]
    references: List[Dict[str, str]]
    prerequisites: List[str]
    learning_objectives: List[str]

class MathematicalInductionEncyclopedia:
    """
    Comprehensive encyclopedia of mathematical induction and related concepts
    """
    
    def __init__(self):
        self.entries: Dict[str, EncyclopediaEntry] = {}
        self.categories = self._initialize_categories()
        self.learning_paths = self._initialize_learning_paths()
        self.search_index = {}
        
    def _initialize_categories(self) -> Dict[str, List[str]]:
        """Initialize encyclopedia categories"""
        return {
            "mathematical_induction": [
                "weak_induction", "strong_induction", "complete_induction",
                "course_of_values_induction", "structural_induction"
            ],
            "advanced_induction": [
                "transfinite_induction", "well_founded_induction", "noetherian_induction",
                "prefix_induction", "forward_backward_induction", "infinite_descent"
            ],
            "physical_induction": [
                "electromagnetic_induction", "mutual_induction", "self_induction",
                "electrostatic_induction", "magnetic_induction"
            ],
            "logical_induction": [
                "inductive_reasoning", "deductive_reasoning", "statistical_induction",
                "scientific_induction", "philosophical_induction"
            ],
            "applications": [
                "computer_science", "physics", "engineering", "mathematics",
                "philosophy", "linguistics", "biology"
            ]
        }
    
    def _initialize_learning_paths(self) -> Dict[str, List[str]]:
        """Initialize structured learning paths"""
        return {
            "beginner_path": [
                "introduction_to_induction",
                "basic_weak_induction",
                "simple_examples",
                "common_mistakes",
                "practice_problems_elementary"
            ],
            "intermediate_path": [
                "strong_induction_concepts",
                "complete_induction",
                "structural_induction_basics",
                "real_world_applications",
                "proof_techniques"
            ],
            "advanced_path": [
                "transfinite_induction",
                "well_founded_induction",
                "noetherian_induction",
                "advanced_applications",
                "research_topics"
            ],
            "comprehensive_path": [
                "all_mathematical_forms",
                "physical_induction_principles",
                "logical_foundations",
                "interdisciplinary_applications",
                "current_research"
            ]
        }
    
    def create_comprehensive_entries(self) -> None:
        """Create all encyclopedia entries with comprehensive content"""
        
        # Mathematical Induction Entries
        self._create_mathematical_induction_entries()
        
        # Advanced Induction Entries
        self._create_advanced_induction_entries()
        
        # Physical Induction Entries
        self._create_physical_induction_entries()
        
        # Logical Induction Entries
        self._create_logical_induction_entries()
        
        # Application Entries
        self._create_application_entries()
        
        # Historical and Philosophical Entries
        self._create_historical_entries()
    
    def _create_mathematical_induction_entries(self) -> None:
        """Create entries for mathematical induction forms"""
        
        # Weak Mathematical Induction
        self.entries["weak_induction"] = EncyclopediaEntry(
            id="weak_induction",
            title="Weak Mathematical Induction",
            induction_type=InductionType.MATHEMATICAL_WEAK,
            difficulty_level=DifficultyLevel.HIGH_SCHOOL,
            summary="Basic form of mathematical induction where only the immediate predecessor is assumed",
            detailed_content=self._get_weak_induction_content(),
            mathematical_formulations=[
                "P(0) \\land \\forall k(P(k) \\rightarrow P(k+1)) \\rightarrow \\forall nP(n)",
                "\\text{Base case: } P(n_0) \\text{ is true}",
                "\\text{Inductive step: } \\forall k \\geq n_0 (P(k) \\rightarrow P(k+1))"
            ],
            examples=[
                {
                    "title": "Sum of First n Natural Numbers",
                    "statement": "Show that \\sum_{i=1}^n i = \\frac{n(n+1)}{2}",
                    "base_case": "For n=1: \\sum_{i=1}^1 i = 1 = \\frac{1(1+1)}{2}",
                    "inductive_step": "Assume \\sum_{i=1}^k i = \\frac{k(k+1)}{2}. Then \\sum_{i=1}^{k+1} i = \\frac{k(k+1)}{2} + (k+1) = \\frac{(k+1)(k+2)}{2}",
                    "latex_equations": ["\\sum_{i=1}^n i = \\frac{n(n+1)}{2}"]
                },
                {
                    "title": "Sum of Geometric Series",
                    "statement": "Show that \\sum_{i=0}^n r^i = \\frac{r^{n+1}-1}{r-1} for r \\neq 1",
                    "base_case": "For n=0: \\sum_{i=0}^0 r^i = 1 = \\frac{r^{1}-1}{r-1}",
                    "inductive_step": "Assume formula holds for k. Then \\sum_{i=0}^{k+1} r^i = \\frac{r^{k+1}-1}{r-1} + r^{k+1} = \\frac{r^{k+2}-1}{r-1}",
                    "latex_equations": ["\\sum_{i=0}^n r^i = \\frac{r^{n+1}-1}{r-1}"]
                }
            ],
            applications=["Number theory", "Combinatorics", "Computer science algorithms", "Graph theory"],
            historical_context={
                "origin": "Early traces in Plato's Parmenides (370 BC)",
                "first_rigorous": "Gersonides (1288-1344)",
                "formalization": "Blaise Pascal (1665)",
                "modern_form": "George Boole, Augustus De Morgan, Richard Dedekind (19th century)"
            },
            related_concepts=["strong_induction", "recursion", "well-ordering_principle", "peano_axioms"],
            visualizations=[
                {
                    "type": "domino_falling",
                    "description": "Visual metaphor of falling dominoes",
                    "latex_code": "\\begin{tikzpicture}[scale=0.8]\\foreach \\x in {0,1,2,3,4,5}\\draw[fill=blue!30] (\\x,0) rectangle +(0.6,1.2);\\end{tikzpicture}"
                },
                {
                    "type": "ladder_climbing",
                    "description": "Ladder climbing metaphor",
                    "latex_code": "\\begin{tikzpicture}[scale=0.7]\\foreach \\y in {0,1,2,3,4}\\draw (0,\\y) -- (2,\\y);\\draw (0.5,4) -- (1.5,4.5);\\end{tikzpicture}"
                }
            ],
            exercises=[
                {
                    "difficulty": "easy",
                    "problem": "Prove by induction: \\sum_{i=1}^n 2i = n(n+1)",
                    "hint": "Use the formula for sum of first n integers",
                    "solution": "Base case: n=1 gives 2=1(2). Assume true for k. For k+1: \\sum_{i=1}^{k+1} 2i = k(k+1) + 2(k+1) = (k+1)(k+2)"
                },
                {
                    "difficulty": "medium",
                    "problem": "Prove by induction: 2^n > n for all n \\geq 1",
                    "hint": "Consider the relationship between 2^{k+1} and 2^k",
                    "solution": "Base case: 2^1 = 2 > 1. Assume 2^k > k. Then 2^{k+1} = 2 \\cdot 2^k > 2k > k+1 for k \\geq 1"
                }
            ],
            references=[
                {"title": "Concrete Mathematics", "author": "Graham, Knuth, Patashnik"},
                {"title": "How to Prove It", "author": "Daniel J. Velleman"},
                {"title": "Mathematical Induction", "author": "Gaisi Takeuti, Wilson M. Zaring"}
            ],
            prerequisites=["basic_logic", "mathematical_proof", "peano_axioms"],
            learning_objectives=[
                "Understand the principle of weak mathematical induction",
                "Apply weak induction to prove mathematical statements",
                "Identify appropriate base cases and inductive steps",
                "Recognize limitations of weak induction"
            ]
        )
        
        # Strong Mathematical Induction
        self.entries["strong_induction"] = EncyclopediaEntry(
            id="strong_induction",
            title="Strong Mathematical Induction",
            induction_type=InductionType.MATHEMATICAL_STRONG,
            difficulty_level=DifficultyLevel.UNDERGRADUATE,
            summary="Form of mathematical induction where all preceding cases are assumed in the inductive step",
            detailed_content=self._get_strong_induction_content(),
            mathematical_formulations=[
                "P(0) \\land \\forall k(\\forall j < k P(j) \\rightarrow P(k)) \\rightarrow \\forall nP(n)",
                "\\text{Base case: } P(n_0) \\text{ is true}",
                "\\text{Inductive step: } \\forall k > n_0 ((\\forall j: n_0 \\leq j < k, P(j)) \\rightarrow P(k))"
            ],
            examples=[
                {
                    "title": "Fundamental Theorem of Arithmetic",
                    "statement": "Every integer n > 1 can be written as a product of primes",
                    "base_case": "n=2 is prime",
                    "inductive_step": "Assume all integers < k can be factored. If k is prime, done. If k = ab with 1 < a < k, then a and b < k, so both factor into primes",
                    "latex_equations": ["n = p_1^{e_1} p_2^{e_2} \\cdots p_m^{e_m}"]
                },
                {
                    "title": "Fibonacci Number Properties",
                    "statement": "Prove F_n \\geq \\phi^{n-2} for n \\geq 3, where F_n is the nth Fibonacci number",
                    "base_case": "F_3 = 2 \\geq \\phi^{1}",
                    "inductive_step": "Use F_{k+1} = F_k + F_{k-1} and induction hypothesis for both F_k and F_{k-1}",
                    "latex_equations": ["F_n = \\frac{\\phi^n - \\psi^n}{\\phi - \\psi}"]
                }
            ],
            applications=["Number theory", "Recursive algorithms", "Graph theory", "Computer science"],
            historical_context={
                "development": "Evolved from weak induction in 19th century",
                "formalization": "Richard Dedekind, Giuseppe Peano",
                "modern_understanding": "Equivalent to weak induction but more convenient for certain proofs"
            },
            related_concepts=["weak_induction", "complete_induction", "structural_induction", "recursion"],
            visualizations=[
                {
                    "type": "induction_tree",
                    "description": "Tree structure showing all predecessors",
                    "latex_code": "\\begin{tikzpicture}[scale=0.8]\\node {P(k)};\\foreach \\i in {1,...,k}\\node[left of=current] at (-\\i,0) {P(\\i)};\\draw[->] (-\\i,0) -- (0,0);\\end{tikzpicture}"
                }
            ],
            exercises=[
                {
                    "difficulty": "medium",
                    "problem": "Prove: Every integer n \\geq 2 can be written as a sum of distinct powers of 2",
                    "hint": "Consider the largest power of 2 less than or equal to n",
                    "solution": "Use strong induction on n. For the inductive step, let 2^k be the largest power \\leq n. Then n - 2^k < n and can be expressed as sum of distinct powers"
                }
            ],
            references=[
                {"title": "Introduction to Algorithms", "author": "Cormen, Leiserson, Rivest, Stein"},
                {"title": "Discrete Mathematics and Its Applications", "author": "Kenneth H. Rosen"}
            ],
            prerequisites=["weak_induction", "recursion", "number_theory_basics"],
            learning_objectives=[
                "Understand when and how to use strong induction",
                "Compare strong vs weak induction approaches",
                "Apply strong induction to recursive problems",
                "Recognize problems where strong induction is essential"
            ]
        )
    
    def _get_weak_induction_content(self) -> str:
        """Get detailed content for weak induction"""
        return """
# Weak Mathematical Induction

Weak mathematical induction, also known as simple or ordinary induction, is the most fundamental form 
of mathematical induction. It consists of two essential steps:

## The Principle

**Base Case:** Prove that the statement P(n₀) is true for some starting value n₀ (typically n₀ = 0 or n₀ = 1).

**Inductive Step:** Prove that if the statement P(k) is true for some arbitrary k ≥ n₀, 
then P(k+1) must also be true.

## Formal Statement

The principle can be formally stated as:

P(n₀) ∧ ∀k(P(k) → P(k+1)) → ∀n ≥ n₀ P(n)

## Why It Works

The validity of mathematical induction rests on the well-ordering principle of natural numbers, 
which states that every non-empty set of natural numbers has a least element. If induction were false, 
there would exist a smallest counterexample, contradicting the inductive step.

## Common Applications

1. **Summation formulas**: Proving formulas for sums of arithmetic and geometric series
2. **Inequalities**: Establishing bounds and inequalities
3. **Divisibility properties**: Proving statements about divisibility
4. **Recursive sequences**: Analyzing sequences defined recursively

## Limitations

Weak induction assumes only the immediate predecessor P(k) to prove P(k+1). 
Some proofs require stronger assumptions about multiple previous cases, necessitating strong induction.
"""
    
    def _get_strong_induction_content(self) -> str:
        """Get detailed content for strong induction"""
        return """
# Strong Mathematical Induction

Strong mathematical induction, also known as complete induction or course-of-values induction, 
is a powerful variant that assumes all preceding cases rather than just the immediate predecessor.

## The Principle

**Base Case:** Prove that P(n₀) is true for the starting value.

**Inductive Step:** Prove that if P(j) is true for all j such that n₀ ≤ j < k, then P(k) must also be true.

## Formal Statement

P(n₀) ∧ ∀k(∀j < k P(j) → P(k)) → ∀n ≥ n₀ P(n)

## When to Use Strong Induction

Strong induction is particularly useful when:

1. **Recursive definitions depend on multiple previous values** (e.g., Fibonacci sequence)
2. **Factorization problems** where you need to consider all proper divisors
3. **Structural induction** on recursively defined data structures
4. **Existence proofs** where you analyze smaller instances

## Equivalence with Weak Induction

Despite its apparent strength, strong induction is logically equivalent to weak induction. 
Any proof using strong induction can be transformed into a weak induction proof by defining 
a new statement Q(k) = "P(j) is true for all j ≤ k".

## Examples

1. **Fundamental Theorem of Arithmetic**: Every integer > 1 factors into primes
2. **Fibonacci identities**: Properties involving Fₙ₊₁ = Fₙ + Fₙ₋₁
3. **Graph theory**: Properties of trees and connected components
4. **Algorithm analysis**: Proving correctness of divide-and-conquer algorithms
"""
    
    def _create_advanced_induction_entries(self) -> None:
        """Create entries for advanced induction forms"""
        
        # Transfinite Induction
        self.entries["transfinite_induction"] = EncyclopediaEntry(
            id="transfinite_induction",
            title="Transfinite Induction",
            induction_type=InductionType.TRANSFINITE,
            difficulty_level=DifficultyLevel.GRADUATE,
            summary="Extension of mathematical induction to well-ordered sets, particularly ordinal numbers",
            detailed_content=self._get_transfinite_induction_content(),
            mathematical_formulations=[
                "∀α[(∀β < α P(β)) → P(α)] → ∀α P(α)",
                "∀α[∀β < α(P(β)) → P(α)] implies P(α) for all ordinals α",
                "Limit case: ∀α[α is limit ∧ ∀β < α P(β) → P(α)]"
            ],
            examples=[
                {
                    "title": "Transfinite Recursion",
                    "statement": "Define a function on ordinals using transfinite recursion",
                    "base_case": "F(0) = a₀",
                    "successor_case": "F(α+1) = G(F(α), α)",
                    "limit_case": "F(λ) = H({F(β) : β < λ}, λ) for limit ordinal λ"
                }
            ],
            applications=["Set theory", "Ordinal analysis", "Proof theory", "Descriptive set theory"],
            historical_context={
                "development": "Developed by Georg Cantor and Richard Dedekind",
                "formalization": "Ernst Zermelo, Abraham Fraenkel",
                "modern_theory": "John von Neumann's ordinal theory"
            },
            related_concepts=["ordinal_numbers", "well_founded_induction", "set_theory", "cardinals"],
            visualizations=[
                {
                    "type": "ordinal_hierarchy",
                    "description": "Visualization of ordinal hierarchy",
                    "latex_code": "\\begin{tikzpicture}\\node {0};\\node[above right] {1};\\node[above right] {2};\\node[above right] {...};\\node[above right] {ω};\\end{tikzpicture}"
                }
            ],
            exercises=[
                {
                    "difficulty": "hard",
                    "problem": "Prove by transfinite induction: Every ordinal can be uniquely written as \\omega^β \\cdot n + γ"
                }
            ],
            references=[
                {"title": "Set Theory", "author": "Thomas Jech"},
                {"title": "Introduction to Set Theory", "author": "Karel Hrbacek, Thomas Jech"}
            ],
            prerequisites=["ordinal_numbers", "set_theory_basics", "mathematical_induction"],
            learning_objectives=[
                "Understand transfinite induction principles",
                "Apply induction to ordinal numbers",
                "Distinguish limit and successor cases",
                "Use transfinite induction in set-theoretic proofs"
            ]
        )
        
        # Structural Induction
        self.entries["structural_induction"] = EncyclopediaEntry(
            id="structural_induction",
            title="Structural Induction",
            induction_type=InductionType.STRUCTURAL,
            difficulty_level=DifficultyLevel.UNDERGRADUATE,
            summary="Induction principle for recursively defined structures like trees, lists, and formulas",
            detailed_content=self._get_structural_induction_content(),
            mathematical_formulations=[
                "∀s[(∀s' proper_substructure(s,s') P(s')) → P(s)] → ∀s P(s)",
                "Base cases for atomic structures",
                "Inductive cases for composite structures"
            ],
            examples=[
                {
                    "title": "Binary Trees",
                    "statement": "Prove that a binary tree with n internal nodes has n+1 external nodes",
                    "base_case": "Empty tree has 0 internal nodes and 1 external node",
                    "inductive_step": "Tree with root and subtrees T₁, T₂: internal nodes = 1 + int(T₁) + int(T₂), external nodes = ext(T₁) + ext(T₂)"
                },
                {
                    "title": "Length of Concatenated Lists",
                    "statement": "|xs ++ ys| = |xs| + |ys| for all lists xs, ys",
                    "base_case": "[] ++ ys = ys, so |[] ++ ys| = |ys| = 0 + |ys|",
                    "inductive_step": "(x:xs) ++ ys = x:(xs ++ ys), so |(x:xs) ++ ys| = 1 + |xs ++ ys| = 1 + (|xs| + |ys|) = (1 + |xs|) + |ys| = |x:xs| + |ys|"
                }
            ],
            applications=["Computer science", "Logic", "Programming language theory", "Formal verification"],
            historical_context={
                "development": "Developed in 1960s with rise of computer science",
                "formalization": "Computer science and logic communities",
                "modern_use": "Functional programming, type theory, automated theorem proving"
            },
            related_concepts=["recursion", "algebraic_data_types", "functional_programming", "formal_methods"],
            visualizations=[
                {
                    "type": "recursive_structure",
                    "description": "Tree decomposition for structural induction",
                    "latex_code": "\\begin{tikzpicture}\\node {root};\\node[left] at (-1,-1) {subtree1};\\node[right] at (1,-1) {subtree2};\\draw[->] (0,0) -- (-1,-1);\\draw[->] (0,0) -- (1,-1);\\end{tikzpicture}"
                }
            ],
            exercises=[
                {
                    "difficulty": "medium",
                    "problem": "Prove by structural induction: The reverse of a concatenated list satisfies reverse(xs ++ ys) = reverse(ys) ++ reverse(xs)"
                }
            ],
            references=[
                {"title": "Types and Programming Languages", "author": "Benjamin C. Pierce"},
                {"title": "Software Foundations", "author": "Benjamin Pierce et al."}
            ],
            prerequisites=["recursion", "functional_programming", "mathematical_induction"],
            learning_objectives=[
                "Apply induction to recursively defined structures",
                "Identify base cases and constructors",
                "Prove properties of data structures",
                "Use structural induction in program verification"
            ]
        )
    
    def _get_transfinite_induction_content(self) -> str:
        """Get detailed content for transfinite induction"""
        return """
# Transfinite Induction

Transfinite induction extends mathematical induction beyond the natural numbers to all well-ordered sets, 
particularly the class of ordinal numbers.

## The Principle

For a well-ordered set (S, <), to prove that property P holds for all elements of S:

1. **Base Case**: Prove P(min(S)) for the least element
2. **Successor Case**: Show that if P(α) holds, then P(α+1) holds
3. **Limit Case**: For limit ordinal λ (no immediate predecessor), if P(β) holds for all β < λ, then P(λ) holds

## Formal Statement

∀α[(∀β < α P(β)) → P(α)] → ∀α P(α)

## Types of Ordinals

1. **Zero**: 0, the smallest ordinal
2. **Successor ordinals**: α+1 = α ∪ {α}
3. **Limit ordinals**: ordinals that are not zero or successors (e.g., ω, ω·2)

## Applications

1. **Set Theory**: Proving properties of ordinal arithmetic
2. **Proof Theory**: Consistency proofs and ordinal analysis
3. **Descriptive Set Theory**: Borel hierarchy and projective sets
4. **Recursion Theory**: Transfinite recursion definitions

## Relationship with Other Principles

Transfinite induction is equivalent to the axiom of choice in ZF set theory and to the 
well-ordering principle. It provides a foundation for defining objects by transfinite recursion.
"""
    
    def _get_structural_induction_content(self) -> str:
        """Get detailed content for structural induction"""
        return """
# Structural Induction

Structural induction is a generalization of mathematical induction used to prove properties 
of recursively defined data structures such as lists, trees, and formal languages.

## The Principle

For an algebraic data type with constructors, to prove property P holds for all elements:

1. **Base Cases**: Prove P holds for all atomic/empty constructors
2. **Inductive Cases**: For each composite constructor, assume P holds for all subcomponents 
   and prove P holds for the constructed element

## Common Examples

### Lists
- **Base case**: Empty list []
- **Inductive case**: Constructor (x:xs) that builds list from element x and list xs

### Binary Trees
- **Base case**: Empty tree or leaf node
- **Inductive case**: Internal node with left and right subtrees

### Formulas (Logic)
- **Base cases**: Atomic propositions and variables
- **Inductive cases**: Negation, conjunction, disjunction, implication, quantification

## Applications

1. **Program Verification**: Proving correctness of recursive functions
2. **Type Theory**: Proving properties of type systems
3. **Compiler Design**: Optimization and correctness proofs
4. **Algorithm Analysis**: Correctness of divide-and-conquer algorithms

## Relationship with Other Principles

Structural induction is equivalent to mathematical induction on the natural numbers 
when applied to lists or other finite structures. For infinite structures, it relates 
to well-founded induction.
"""
    
    def _create_physical_induction_entries(self) -> None:
        """Create entries for physical induction forms"""
        
        # Electromagnetic Induction
        self.entries["electromagnetic_induction"] = EncyclopediaEntry(
            id="electromagnetic_induction",
            title="Electromagnetic Induction",
            induction_type=InductionType.ELECTROMAGNETIC,
            difficulty_level=DifficultyLevel.HIGH_SCHOOL,
            summary="Production of electromotive force across electrical conductors in changing magnetic fields",
            detailed_content=self._get_electromagnetic_induction_content(),
            mathematical_formulations=[
                "\\mathcal{E} = -\\frac{d\\Phi_B}{dt}",
                "\\mathcal{E} = -N \\frac{d\\Phi_B}{dt}",
                "\\oint_{\\partial \\Sigma} \\mathbf{E} \\cdot d\\mathbf{\\ell} = -\\frac{d}{dt} \\int_{\\Sigma} \\mathbf{B} \\cdot d\\mathbf{A}",
                "\\mathcal{E} = \\oint (\\mathbf{v} \\times \\mathbf{B}) \\cdot d\\mathbf{\\ell}"
            ],
            examples=[
                {
                    "title": "Faraday's Law Experiment",
                    "description": "Moving magnet through a coil induces current",
                    "calculation": "For N=100 turns, B变化的速率为0.1 T/s，面积A=0.01 m²: \\mathcal{E} = -100 × 0.1 × 0.01 = -0.1 V"
                },
                {
                    "title": "AC Generator",
                    "description": "Rotating coil in magnetic field generates alternating current",
                    "calculation": "Angular frequency ω, magnetic field B, area A: \\mathcal{E} = NAB\\omega\\sin(\\omega t)"
                }
            ],
            applications=["Electric generators", "Transformers", "Induction motors", "Wireless charging"],
            historical_context={
                "discovery": "Michael Faraday (1831)",
                "independent_discovery": "Joseph Henry (1832)",
                "mathematical_formulation": "James Clerk Maxwell (1860s)",
                "modern_development": "Electrical engineering and technology"
            },
            related_concepts=["faradays_law", "lenz_law", "magnetic_flux", "maxwell_equations"],
            visualizations=[
                {
                    "type": "faraday_experiment",
                    "description": "Faraday's induction experiment setup",
                    "latex_code": "\\begin{tikzpicture}\\draw[thick] (0,0) circle (1.5);\\draw[->,thick,blue] (-3,0) -- (3,0) node[right] {B};\\draw[thick,red] (0,1.5) -- (0,2.5);\\draw[thick,red] (0,-1.5) -- (0,-2.5);\\end{tikzpicture}"
                }
            ],
            exercises=[
                {
                    "difficulty": "medium",
                    "problem": "A rectangular coil with 50 turns, dimensions 10cm × 15cm, rotates at 60 rpm in a 0.5 T magnetic field. Calculate the maximum induced emf.",
                    "solution": "Maximum emf occurs when sin(ωt) = 1. ω = 2π × 60/60 = 2π rad/s. A = 0.1 × 0.15 = 0.015 m². \\mathcal{E}_{max} = NABω = 50 × 0.5 × 0.015 × 2π ≈ 2.36 V"
                }
            ],
            references=[
                {"title": "Introduction to Electrodynamics", "author": "David J. Griffiths"},
                {"title": "University Physics", "author": "Young and Freedman"}
            ],
            prerequisites=["basic_electricity", "magnetic_fields", "calculus"],
            learning_objectives=[
                "Understand Faraday's law of electromagnetic induction",
                "Apply Lenz's law to determine induced current direction",
                "Calculate induced emf in various configurations",
                "Analyze practical applications of electromagnetic induction"
            ]
        )
    
    def _get_electromagnetic_induction_content(self) -> str:
        """Get detailed content for electromagnetic induction"""
        return """
# Electromagnetic Induction

Electromagnetic induction is the fundamental principle governing the generation of electricity 
from changing magnetic fields, forming the basis for modern electrical power generation and distribution.

## Faraday's Law of Induction

The induced electromotive force (emf) in any closed circuit is equal to the negative rate of change 
of magnetic flux through the circuit:

\\mathcal{E} = -\\frac{d\\Phi_B}{dt}

Where:
- \\mathcal{E} is the induced emf (volts)
- \\Phi_B is the magnetic flux (webers)
- t is time (seconds)

For a coil with N turns:

\\mathcal{E} = -N \\frac{d\\Phi_B}{dt}

## Lenz's Law

The negative sign in Faraday's law represents Lenz's law: the induced current flows in a direction 
that opposes the change causing it. This is a manifestation of the conservation of energy.

## Types of Electromagnetic Induction

### 1. Transformer EMF
- Caused by time-varying magnetic fields
- Stationary conductor in changing field
- Governed by Maxwell-Faraday equation

### 2. Motional EMF
- Caused by conductor moving through magnetic field
- Governed by Lorentz force: \\mathcal{E} = \\oint (\\mathbf{v} \\times \\mathbf{B}) \\cdot d\\mathbf{\\ell}

## Maxwell-Faraday Equation

In differential form:

\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}

In integral form:

\\oint_{\\partial \\Sigma} \\mathbf{E} \\cdot d\\mathbf{\\ell} = -\\frac{d}{dt} \\int_{\\Sigma} \\mathbf{B} \\cdot d\\mathbf{A}

## Applications

1. **Electric Generators**: Convert mechanical energy to electrical energy
2. **Transformers**: Transfer energy between circuits at different voltages
3. **Induction Motors**: Convert electrical energy to mechanical energy
4. **Wireless Power Transfer**: Inductive charging and power transmission
5. **Magnetic Sensors**: Hall effect sensors, current clamps, flow meters

## Historical Development

- **1831**: Michael Faraday discovers electromagnetic induction
- **1832**: Joseph Henry makes independent discovery
- **1860s**: James Clerk Maxwell incorporates induction into electromagnetic theory
- **Modern**: Foundation for electrical power systems and wireless technology

## Mathematical Analysis

### For a rotating rectangular coil:
\\mathcal{E} = NAB\\omega\\sin(\\omega t)

Where:
- N = number of turns
- A = area of coil
- B = magnetic field strength
- ω = angular velocity
- t = time

### For a solenoid with changing current:
\\mathcal{E} = -L \\frac{di}{dt}

Where L is the inductance.
"""
    
    def search_entries(self, query: str, max_results: int = 10) -> List[EncyclopediaEntry]:
        """Search encyclopedia entries by query"""
        query = query.lower()
        results = []
        
        for entry in self.entries.values():
            score = 0
            
            # Search in title
            if query in entry.title.lower():
                score += 10
            
            # Search in summary
            if query in entry.summary.lower():
                score += 5
            
            # Search in content
            if query in entry.detailed_content.lower():
                score += 3
            
            # Search in applications
            for app in entry.applications:
                if query in app.lower():
                    score += 2
            
            # Search in related concepts
            for concept in entry.related_concepts:
                if query in concept.lower():
                    score += 2
            
            if score > 0:
                results.append((entry, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in results[:max_results]]
    
    def get_learning_path(self, path_name: str) -> List[EncyclopediaEntry]:
        """Get a structured learning path"""
        if path_name not in self.learning_paths:
            return []
        
        path = []
        for entry_id in self.learning_paths[path_name]:
            if entry_id in self.entries:
                path.append(self.entries[entry_id])
        
        return path
    
    def get_entries_by_difficulty(self, difficulty: DifficultyLevel) -> List[EncyclopediaEntry]:
        """Get entries filtered by difficulty level"""
        return [entry for entry in self.entries.values() if entry.difficulty_level == difficulty]
    
    def get_related_entries(self, entry_id: str) -> List[EncyclopediaEntry]:
        """Get entries related to a given entry"""
        if entry_id not in self.entries:
            return []
        
        entry = self.entries[entry_id]
        related = []
        
        for concept in entry.related_concepts:
            for other_entry in self.entries.values():
                if other_entry.id != entry_id and concept in other_entry.related_concepts:
                    related.append(other_entry)
        
        return list(set(related))  # Remove duplicates
    
    def export_to_json(self, filename: str) -> None:
        """Export encyclopedia to JSON format"""
        data = {}
        for entry_id, entry in self.entries.items():
            data[entry_id] = {
                'id': entry.id,
                'title': entry.title,
                'induction_type': entry.induction_type.value,
                'difficulty_level': entry.difficulty_level.value,
                'summary': entry.summary,
                'detailed_content': entry.detailed_content,
                'mathematical_formulations': entry.mathematical_formulations,
                'examples': entry.examples,
                'applications': entry.applications,
                'historical_context': entry.historical_context,
                'related_concepts': entry.related_concepts,
                'visualizations': entry.visualizations,
                'exercises': entry.exercises,
                'references': entry.references,
                'prerequisites': entry.prerequisites,
                'learning_objectives': entry.learning_objectives
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get encyclopedia statistics"""
        stats = {
            'total_entries': len(self.entries),
            'by_difficulty': {},
            'by_type': {},
            'total_examples': 0,
            'total_exercises': 0,
            'total_applications': set()
        }
        
        for entry in self.entries.values():
            # Count by difficulty
            diff = entry.difficulty_level.value
            stats['by_difficulty'][diff] = stats['by_difficulty'].get(diff, 0) + 1
            
            # Count by type
            type_val = entry.induction_type.value
            stats['by_type'][type_val] = stats['by_type'].get(type_val, 0) + 1
            
            # Count examples and exercises
            stats['total_examples'] += len(entry.examples)
            stats['total_exercises'] += len(entry.exercises)
            
            # Collect applications
            stats['total_applications'].update(entry.applications)
        
        stats['total_applications'] = len(stats['total_applications'])
        
        return stats

# Create global encyclopedia instance
encyclopedia = MathematicalInductionEncyclopedia()
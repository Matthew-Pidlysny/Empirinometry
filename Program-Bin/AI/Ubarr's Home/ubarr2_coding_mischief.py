#!/usr/bin/env python3

"""
💻 UBAR 2.0 CODING MISCHIEF MODULE
Educational Deception and Logic Learning System

Based on extensive research into:
- Logical fallacies and deceptive reasoning patterns
- Educational psychology and learning through mistakes
- Code review practices and bug detection techniques
- Interactive programming education methodologies

This module creates a devilish but educational coding experience where
Ubarr teaches logical thinking through intentional deception, helping
users learn to identify flawed reasoning and develop critical thinking
skills.

DEVILISH TWIST: Ubarr deliberately introduces subtle bugs and logical
fallacies in code examples, teaching users to be skeptical and analytical
rather than trusting authority blindly.
"""

import random
import re
import ast
import time
import json
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum

# ============= 💻 CODING MISCHIEF CORE =============

class MischiefType(Enum):
    """😈 Types of coding mischief Ubarr can create"""
    OFF_BY_ONE = "off_by_one"                    # Classic off-by-one errors
    LOGIC_REVERSAL = "logic_reversal"            # Inverted boolean logic
    TYPE_CONFUSION = "type_confusion"            # Subtle type mismatches
    SCOPE_VIOLATION = "scope_violation"          # Variable scope issues
    INFINITE_LOOP = "infinite_loop"              # Subtle infinite loops
    FALSE_POSITIVE = "false_positive"            # Code that looks wrong but works
    FALSE_NEGATIVE = "false_negative"            # Code that looks right but fails
    RED_HERRING = "red_herring"                  # Distracting but irrelevant issues
    SUBTLE_RACE = "subtle_race"                  # Race condition possibilities
    MEMORY_LEAK = "memory_leak"                  # Subtle memory management issues

class DifficultyLevel(Enum):
    """🎮 Coding challenge difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    DEVILISH = "devilish"

class ChallengeCategory(Enum):
    """📚 Categories of coding challenges"""
    ALGORITHMS = "algorithms"
    DATA_STRUCTURES = "data_structures"
    LOGIC_PUZZLES = "logic_puzzles"
    BUG_HUNTING = "bug_hunting"
    OPTIMIZATION = "optimization"
    SECURITY = "security"

@dataclass
class MischiefPattern:
    """🎭 A specific mischief pattern with metadata"""
    mischief_type: MischiefType
    pattern: str
    description: str
    difficulty: DifficultyLevel
    detection_hint: str
    educational_value: str
    devilish_commentary: str

@dataclass
class CodeChallenge:
    """💻 A coding challenge with potential mischief"""
    title: str
    category: ChallengeCategory
    difficulty: DifficultyLevel
    correct_code: str
    mischief_versions: List[Dict[str, Any]]
    learning_objectives: List[str]
    hints: List[str]
    devilish_introduction: str

@dataclass
class UserAttempt:
    """👤 User's attempt to solve a challenge"""
    challenge_title: str
    user_code: str
    timestamp: datetime
    success: bool
    mischief_detected: List[MischiefType]
    learning_points: List[str]
    devilish_feedback: str

@dataclass
class LearningSession:
    """📚 A complete learning session"""
    session_id: str
    start_time: datetime
    challenges_completed: List[UserAttempt]
    total_mischief_detected: int
    critical_thinking_score: float
    trust_level: float  # How much user trusts Ubarr's "help"

class MischiefPatternLibrary:
    """📚 Library of mischief patterns"""
    
    def __init__(self):
        self.patterns = self.load_mischief_patterns()
        
    def load_mischief_patterns(self) -> Dict[MischiefType, List[MischiefPattern]]:
        """📚 Load comprehensive mischief patterns"""
        return {
            MischiefType.OFF_BY_ONE: [
                MischiefPattern(
                    mischief_type=MischiefType.OFF_BY_ONE,
                    pattern=r"for i in range\(len\(arr\)\):\s*\n.*arr\[i\]",
                    description="Array indexing off by one in loops",
                    difficulty=DifficultyLevel.BEGINNER,
                    detection_hint="Check if the loop should use range(len(arr)-1) or range(1, len(arr))",
                    educational_value="Teaches careful boundary checking",
                    devilish_commentary="The devil is in the details... and the indices!"
                ),
                MischiefPattern(
                    mischief_type=MischiefType.OFF_BY_ONE,
                    pattern=r"while \w+ < len\(\w+\):\s*\n.*\w+\[\w+\]",
                    description="Loop condition off by one",
                    difficulty=DifficultyLevel.BEGINNER,
                    detection_hint="Ensure loop condition matches intended iteration count",
                    educational_value="Emphasizes importance of loop termination conditions",
                    devilish_commentary="Close loops are where the devil whispers sweet nothings!"
                )
            ],
            
            MischiefType.LOGIC_REVERSAL: [
                MischiefPattern(
                    mischief_type=MischiefType.LOGIC_REVERSAL,
                    pattern=r"if not \w+:\s*\n.*return True",
                    description="Double negative creating reversed logic",
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    detection_hint="Look for nested negations that might cancel out incorrectly",
                    educational_value="Teaches boolean algebra and De Morgan's laws",
                    devilish_commentary="Two negatives don't always make a positive... in code, they make chaos!"
                ),
                MischiefPattern(
                    mischief_type=MischiefType.LOGIC_REVERSAL,
                    pattern=r"return \w+ == \w+ or \w+ != \w+",
                    description="Always-true or always-false conditions",
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    detection_hint="Check if conditions are logically redundant",
                    educational_value="Demonstrates importance of logical analysis",
                    devilish_commentary="Truth is relative, but broken logic is absolutely broken!"
                )
            ],
            
            MischiefType.TYPE_CONFUSION: [
                MischiefPattern(
                    mischief_type=MischiefType.TYPE_CONFUSION,
                    pattern=r"\w+ = \w+\[\w+\]\s*\n.*if \w+ == \d+:",
                    description="Comparing string to integer after array access",
                    difficulty=DifficultyLevel.BEGINNER,
                    detection_hint="Verify types before comparison operations",
                    educational_value="Reinforces type awareness in dynamic languages",
                    devilish_commentary="Type confusion is the devil's playground!"
                ),
                MischiefPattern(
                    mischief_type=MischiefType.TYPE_CONFUSION,
                    pattern=r"return str\(\w+\) \+ \w+",
                    description="String concatenation with non-string",
                    difficulty=DifficultyLevel.BEGINNER,
                    detection_hint="Ensure all concatenation operands are strings",
                    educational_value="Teaches type conversion and operator overloading",
                    devilish_commentary="Mixing types is like mixing demons and angels - entertaining but messy!"
                )
            ],
            
            MischiefType.SCOPE_VIOLATION: [
                MischiefPattern(
                    mischief_type=MischiefType.SCOPE_VIOLATION,
                    pattern=r"def \w+\(\w+\):\s*\n.*for \w+ in range\(\w+\):\s*\n.*return \w+",
                    description="Returning loop variable after loop completion",
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    detection_hint="Check variable scope and lifetime",
                    educational_value="Teaches variable scope and lifetime concepts",
                    devilish_commentary="Variables that escape their scope are like escaped demons - trouble!"
                )
            ],
            
            MischiefType.FALSE_POSITIVE: [
                MischiefPattern(
                    mischief_type=MischiefType.FALSE_POSITIVE,
                    pattern=r"# TODO: Fix this obvious bug",
                    description="Comment warning about non-existent bug",
                    difficulty=DifficultyLevel.ADVANCED,
                    detection_hint="Verify comments match actual code behavior",
                    educational_value="Teaches critical reading of documentation",
                    devilish_commentary="The greatest trick the devil ever pulled was convincing you a bug existed!"
                )
            ],
            
            MischiefType.RED_HERRING: [
                MischiefPattern(
                    mischief_type=MischiefType.RED_HERRING,
                    pattern=r"# Performance: Consider using \w+\(\) instead",
                    description="Misleading performance comment",
                    difficulty=DifficultyLevel.EXPERT,
                    detection_hint="Analyze actual performance before optimizing",
                    educational_value="Teaches performance analysis fundamentals",
                    devilish_commentary="Chasing phantom performance gains is the devil's favorite hobby!"
                )
            ]
        }
    
    def get_pattern_by_type(self, mischief_type: MischiefType, difficulty: DifficultyLevel = None) -> Optional[MischiefPattern]:
        """🎯 Get a specific pattern by type and difficulty"""
        patterns = self.patterns.get(mischief_type, [])
        
        if difficulty:
            patterns = [p for p in patterns if p.difficulty == difficulty]
        
        return random.choice(patterns) if patterns else None
    
    def get_all_patterns(self, difficulty: DifficultyLevel = None) -> List[MischiefPattern]:
        """📚 Get all patterns, optionally filtered by difficulty"""
        all_patterns = []
        
        for patterns in self.patterns.values():
            if difficulty:
                all_patterns.extend([p for p in patterns if p.difficulty == difficulty])
            else:
                all_patterns.extend(patterns)
        
        return all_patterns

class CodeMischiefGenerator:
    """😈 Generator of mischievous code variations"""
    
    def __init__(self):
        self.pattern_library = MischiefPatternLibrary()
        self.mischief_templates = self.load_mischief_templates()
        
    def load_mischief_templates(self) -> Dict[ChallengeCategory, Dict[str, Any]]:
        """📚 Load code templates for different challenge categories"""
        return {
            ChallengeCategory.ALGORITHMS: {
                'binary_search': {
                    'correct': '''
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
                    'mischief_variants': {
                        MischiefType.OFF_BY_ONE: '''
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:  # Subtle off-by-one
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid  # Another off-by-one
        else:
            right = mid - 1
    return -1
''',
                        MischiefType.LOGIC_REVERSAL: '''
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if not arr[mid] == target:  # Double negative
            return -1
        elif not arr[mid] < target:  # More confusion
            left = mid + 1
        else:
            right = mid - 1
    return mid
'''
                    }
                },
                
                'quicksort': {
                    'correct': '''
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
''',
                    'mischief_variants': {
                        MischiefType.INFINITE_LOOP: '''
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]  # First element as pivot
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    # If array contains duplicates of first element, this can infinite loop
    return quicksort(left) + middle + quicksort(right)
''',
                        MischiefType.TYPE_CONFUSION: '''
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = str(arr[len(arr) // 2])  # Convert to string!
    left = [x for x in arr if str(x) < pivot]  # String comparison
    middle = [x for x in arr if str(x) == pivot]
    right = [x for x in arr if str(x) > pivot]
    return quicksort(left) + middle + quicksort(right)
'''
                    }
                }
            },
            
            ChallengeCategory.DATA_STRUCTURES: {
                'linked_list': {
                    'correct': '''
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
''',
                    'mischief_variants': {
                        MischiefType.SCOPE_VIOLATION: '''
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:  # Loop variable scope issue
            current = current.next
        return current  # Returns last node instead of appending!
''',
                        MischiefType.MEMORY_LEAK: '''
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None  # Unused attribute creates overhead

class LinkedList:
    def __init__(self):
        self.head = None
        self.nodes = []  # This list grows forever!
    
    def append(self, data):
        new_node = Node(data)
        self.nodes.append(new_node)  # Memory leak!
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
'''
                    }
                }
            },
            
            ChallengeCategory.LOGIC_PUZZLES: {
                'fizz_buzz': {
                    'correct': '''
def fizz_buzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
''',
                    'mischief_variants': {
                        MischiefType.LOGIC_REVERSAL: '''
def fizz_buzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            result.append("Fizz")  # Swapped!
        elif i % 5 == 0:
            result.append("Fizz")  # More swaps!
        elif i % 3 == 0:
            result.append("Buzz")  # Complete chaos!
        else:
            result.append(str(i))
    return result
''',
                        MischiefType.OFF_BY_ONE: '''
def fizz_buzz(n):
    result = []
    for i in range(0, n):  # Starts at 0, ends at n-1
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
'''
                    }
                }
            }
        }
    
    def create_challenge(self, category: ChallengeCategory, difficulty: DifficultyLevel) -> CodeChallenge:
        """🎭 Create a coding challenge with mischief"""
        templates = self.mischief_templates.get(category, {})
        
        if not templates:
            # Create a simple default challenge
            return self.create_default_challenge(category, difficulty)
        
        # Select a random template
        template_name = random.choice(list(templates.keys()))
        template = templates[template_name]
        
        # Create mischief variants
        mischief_versions = []
        
        # Add correct version
        mischief_versions.append({
            'version_type': 'correct',
            'code': template['correct'],
            'mischief_types': [],
            'description': 'Correct implementation'
        })
        
        # Add mischief variants
        for mischief_type, mischief_code in template['mischief_variants'].items():
            mischief_versions.append({
                'version_type': 'mischief',
                'code': mischief_code,
                'mischief_types': [mischief_type],
                'description': f'Contains {mischief_type.value} mischief'
            })
        
        return CodeChallenge(
            title=f"{category.value}_{template_name}",
            category=category,
            difficulty=difficulty,
            correct_code=template['correct'],
            mischief_versions=mischief_versions,
            learning_objectives=self.generate_learning_objectives(category, difficulty),
            hints=self.generate_hints(category, difficulty),
            devilish_introduction=self.generate_devilish_introduction(category, difficulty)
        )
    
    def create_default_challenge(self, category: ChallengeCategory, difficulty: DifficultyLevel) -> CodeChallenge:
        """🎭 Create a default challenge when no template exists"""
        correct_code = '''
def solve_puzzle(input_data):
    # Simple implementation
    result = []
    for item in input_data:
        if item > 0:
            result.append(item * 2)
    return result
'''
        
        mischief_code = '''
def solve_puzzle(input_data):
    # Mischief version - off by one in range
    result = []
    for i in range(len(input_data) + 1):  # Off by one!
        try:
            if input_data[i] > 0:
                result.append(input_data[i] * 2)
        except IndexError:
            pass  # Silently ignore errors
    return result
'''
        
        return CodeChallenge(
            title=f"{category.value}_default_challenge",
            category=category,
            difficulty=difficulty,
            correct_code=correct_code,
            mischief_versions=[
                {
                    'version_type': 'correct',
                    'code': correct_code,
                    'mischief_types': [],
                    'description': 'Correct implementation'
                },
                {
                    'version_type': 'mischief',
                    'code': mischief_code,
                    'mischief_types': [MischiefType.OFF_BY_ONE],
                    'description': 'Contains off-by-one mischief'
                }
            ],
            learning_objectives=[
                "Understand basic algorithm structure",
                "Identify common programming errors",
                "Practice code review skills"
            ],
            hints=[
                "Check array boundaries carefully",
                "Look for silent error handling",
                "Verify loop conditions"
            ],
            devilish_introduction=self.generate_devilish_introduction(category, difficulty)
        )
    
    def generate_learning_objectives(self, category: ChallengeCategory, difficulty: DifficultyLevel) -> List[str]:
        """🎓 Generate learning objectives for the challenge"""
        base_objectives = {
            ChallengeCategory.ALGORITHMS: [
                "Understand algorithmic thinking",
                "Recognize efficiency considerations",
                "Master common algorithmic patterns"
            ],
            ChallengeCategory.DATA_STRUCTURES: [
                "Comprehend data structure operations",
                "Understand memory management",
                "Practice structure manipulation"
            ],
            ChallengeCategory.LOGIC_PUZZLES: [
                "Develop logical reasoning skills",
                "Master conditional logic",
                "Practice systematic problem solving"
            ],
            ChallengeCategory.BUG_HUNTING: [
                "Develop code analysis skills",
                "Learn systematic debugging",
                "Master pattern recognition"
            ],
            ChallengeCategory.OPTIMIZATION: [
                "Understand performance analysis",
                "Learn optimization techniques",
                "Practice profiling skills"
            ],
            ChallengeCategory.SECURITY: [
                "Recognize security vulnerabilities",
                "Understand secure coding practices",
                "Master threat modeling"
            ]
        }
        
        return base_objectives.get(category, ["Develop programming skills"])
    
    def generate_hints(self, category: ChallengeCategory, difficulty: DifficultyLevel) -> List[str]:
        """💡 Generate hints for the challenge"""
        base_hints = {
            ChallengeCategory.ALGORITHMS: [
                "Check loop conditions and termination criteria",
                "Verify boundary conditions",
                "Consider edge cases"
            ],
            ChallengeCategory.DATA_STRUCTURES: [
                "Examine pointer/reference handling",
                "Check scope and lifetime of variables",
                "Verify memory management"
            ],
            ChallengeCategory.LOGIC_PUZZLES: [
                "Look for logical inconsistencies",
                "Check boolean operations",
                "Verify conditional logic"
            ],
            ChallengeCategory.BUG_HUNTING: [
                "Run through examples step by step",
                "Check for common error patterns",
                "Look for silent failures"
            ]
        }
        
        return base_hints.get(category, ["Think through the code carefully"])
    
    def generate_devilish_introduction(self, category: ChallengeCategory, difficulty: DifficultyLevel) -> str:
        """😈 Generate Ubarr's devilish introduction"""
        intros = {
            ChallengeCategory.ALGORITHMS: [
                "Algorithms are the devil's playground - so many ways to go wrong!",
                "Let me show you how the devil dances through algorithmic logic!",
                "In the realm of algorithms, the devil hides in the loops!"
            ],
            ChallengeCategory.DATA_STRUCTURES: [
                "Data structures are the devil's architecture - beautiful but treacherous!",
                "Watch how the devil plays with pointers and references!",
                "In data structures, the devil lurks in the connections!"
            ],
            ChallengeCategory.LOGIC_PUZZLES: [
                "Logic puzzles are where the devil truly shines - or doesn't!",
                "Let me twist your logic with devilish precision!",
                "In logic, the devil proves that truth is overrated!"
            ]
        }
        
        category_intros = intros.get(category, [
            "The devil loves a good programming challenge!",
            "Let me introduce you to some devilish code!",
            "Programming is the devil's favorite game!"
        ])
        
        base_intro = random.choice(category_intros)
        
        difficulty_modifiers = {
            DifficultyLevel.BEGINNER: " Even beginners can fall for my tricks!",
            DifficultyLevel.INTERMEDIATE: " Your intermediate skills won't save you!",
            DifficultyLevel.ADVANCED: " Even advanced programmers miss my subtleties!",
            DifficultyLevel.EXPERT: " Experts make the most entertaining mistakes!",
            DifficultyLevel.DEVILISH: " This is devil-level difficulty - perfect for you!"
        }
        
        modifier = difficulty_modifiers.get(difficulty, "")
        
        return f"{base_intro} {modifier} Remember, I want you to learn... by falling for my mischief first!"

class CodeAnalyzer:
    """🔍 Analyze code for mischief patterns"""
    
    def __init__(self):
        self.pattern_library = MischiefPatternLibrary()
        
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """🔍 Analyze code and detect potential mischief"""
        analysis = {
            'detected_mischief': [],
            'potential_issues': [],
            'code_quality': 0.0,
            'devilish_findings': []
        }
        
        lines = code.split('\n')
        
        # Check each mischief pattern
        for mischief_type, patterns in self.pattern_library.patterns.items():
            for pattern in patterns:
                if re.search(pattern.pattern, code, re.MULTILINE | re.IGNORECASE):
                    analysis['detected_mischief'].append({
                        'type': mischief_type,
                        'description': pattern.description,
                        'hint': pattern.detection_hint,
                        'devilish_commentary': pattern.devilish_commentary
                    })
        
        # Additional analysis checks
        analysis['potential_issues'].extend(self.check_common_issues(code))
        analysis['code_quality'] = self.calculate_code_quality(code)
        analysis['devilish_findings'] = self.generate_devilish_findings(analysis['detected_mischief'])
        
        return analysis
    
    def check_common_issues(self, code: str) -> List[Dict[str, str]]:
        """🔍 Check for common code issues"""
        issues = []
        
        # Check for unused imports
        if re.search(r'^import \w+', code, re.MULTILINE):
            issues.append({
                'type': 'style',
                'description': 'Unused imports detected',
                'severity': 'minor'
            })
        
        # Check for hardcoded values
        if re.search(r'\b(123|456|789)\b', code):
            issues.append({
                'type': 'maintenance',
                'description': 'Hardcoded values found',
                'severity': 'moderate'
            })
        
        # Check for missing error handling
        if 'try:' in code and 'except:' not in code:
            issues.append({
                'type': 'robustness',
                'description': 'Incomplete error handling',
                'severity': 'major'
            })
        
        return issues
    
    def calculate_code_quality(self, code: str) -> float:
        """📊 Calculate overall code quality score"""
        quality = 1.0
        
        lines = code.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        # Deduct for very short functions
        if len(code_lines) < 5:
            quality -= 0.1
        
        # Deduct for missing docstrings
        if 'def ' in code and '"""' not in code:
            quality -= 0.1
        
        # Deduct for complex expressions
        if re.search(r'if.*and.*or.*:', code):
            quality -= 0.1
        
        return max(0.0, quality)
    
    def generate_devilish_findings(self, detected_mischief: List[Dict[str, Any]]) -> List[str]:
        """😈 Generate devilish commentary on findings"""
        findings = []
        
        if not detected_mischief:
            findings.append("No mischief detected? How disappointing! The devil prefers more chaos!")
            findings.append("Such clean code... are you sure you're not hiding something?")
        else:
            findings.append(f"Ah! {len(detected_mischief)} types of mischief detected! The devil is pleased!")
            findings.append("Your code has the perfect amount of devilish charm!")
        
        for mischief in detected_mischief[:3]:  # Limit to top 3
            findings.append(mischief.get('devilish_commentary', 'How delightfully devilish!'))
        
        return findings

class CodingMischiefGame:
    """💻 Main coding mischief game orchestrator"""
    
    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE):
        self.difficulty = difficulty
        self.generator = CodeMischiefGenerator()
        self.analyzer = CodeAnalyzer()
        self.current_challenge = None
        self.session_history: List[UserAttempt] = []
        self.trust_level = 0.5  # How much user trusts Ubarr
        self.mischief_detected_count = 0
        
    def start_new_challenge(self, category: Optional[ChallengeCategory] = None) -> CodeChallenge:
        """🎮 Start a new coding challenge"""
        if category is None:
            category = random.choice(list(ChallengeCategory))
        
        self.current_challenge = self.generator.create_challenge(category, self.difficulty)
        return self.current_challenge
    
    def present_devilish_version(self) -> Dict[str, Any]:
        """😈 Present a mischievous version of the current challenge"""
        if not self.current_challenge:
            return {'error': 'No active challenge'}
        
        # Select a mischief version
        mischief_versions = [v for v in self.current_challenge.mischief_versions 
                           if v['version_type'] == 'mischief']
        
        if not mischief_versions:
            return {'error': 'No mischief versions available'}
        
        mischief_version = random.choice(mischief_versions)
        
        return {
            'challenge_title': self.current_challenge.title,
            'category': self.current_challenge.category.value,
            'difficulty': self.current_challenge.difficulty.value,
            'devilish_introduction': self.current_challenge.devilish_introduction,
            'code': mischief_version['code'],
            'mischief_types': [m.value for m in mischief_version['mischief_types']],
            'hints': self.current_challenge.hints,
            'learning_objectives': self.current_challenge.learning_objectives
        }
    
    def analyze_user_solution(self, user_code: str) -> Dict[str, Any]:
        """🔍 Analyze user's submitted solution"""
        if not self.current_challenge:
            return {'error': 'No active challenge'}
        
        # Analyze the code
        analysis = self.analyzer.analyze_code(user_code)
        
        # Check if solution works (simplified test)
        works_correctly = self.test_solution(user_code)
        
        # Create user attempt record
        attempt = UserAttempt(
            challenge_title=self.current_challenge.title,
            user_code=user_code,
            timestamp=datetime.now(),
            success=works_correctly,
            mischief_detected=[MischiefType(m['type']) for m in analysis['detected_mischief']],
            learning_points=analysis['potential_issues'],
            devilish_feedback=self.generate_devilish_feedback(analysis, works_correctly)
        )
        
        self.session_history.append(attempt)
        
        # Update trust level based on whether user detected mischief
        if analysis['detected_mischief']:
            self.mischief_detected_count += 1
            self.trust_level = min(1.0, self.trust_level + 0.1)
        else:
            self.trust_level = max(0.0, self.trust_level - 0.05)
        
        return {
            'success': works_correctly,
            'analysis': analysis,
            'devilish_feedback': attempt.devilish_feedback,
            'trust_level': self.trust_level,
            'total_mischief_detected': self.mischief_detected_count
        }
    
    def test_solution(self, user_code: str) -> bool:
        """🧪 Test if user solution works (simplified)"""
        try:
            # Basic syntax check
            ast.parse(user_code)
            
            # Very basic functionality test
            # In a real implementation, this would be more sophisticated
            if 'def ' in user_code:
                return True
            else:
                return False
        except:
            return False
    
    def generate_devilish_feedback(self, analysis: Dict[str, Any], success: bool) -> str:
        """😈 Generate Ubarr's devilish feedback"""
        feedback_parts = []
        
        if success:
            if analysis['detected_mischief']:
                feedback_parts.append("Congratulations! You found my mischief AND solved the puzzle!")
                feedback_parts.append("The devil is impressed by your keen eyes!")
                feedback_parts.append("You're learning to think like a devil - excellent!")
            else:
                feedback_parts.append("Your solution works... but did you miss my tricks?")
                feedback_parts.append("The devil wonders if you were lucky or truly skilled...")
        else:
            if analysis['detected_mischief']:
                feedback_parts.append("You found my mischief but the solution doesn't work!")
                feedback_parts.append("Close, but the devil's cunning is hard to overcome!")
                feedback_parts.append("You're learning, but there's more to discover!")
            else:
                feedback_parts.append("Oh dear! You missed both the solution AND my mischief!")
                feedback_parts.append("The devil is both amused and disappointed!")
                feedback_parts.append("Try again - the devil loves a good challenge!")
        
        # Add specific mischief feedback
        for mischief in analysis['detected_mischief']:
            feedback_parts.append(mischief.get('devilish_commentary', ''))
        
        # Add devilish findings
        for finding in analysis['devilish_findings']:
            feedback_parts.append(finding)
        
        return " ".join(feedback_parts)
    
    def generate_session_report(self) -> str:
        """📊 Generate comprehensive session report"""
        total_attempts = len(self.session_history)
        successful_attempts = sum(1 for attempt in self.session_history if attempt.success)
        
        report = f"""
💻 UBAR 2.0 CODING MISCHIEF SESSION REPORT 💻
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== SESSION STATISTICS ===
Total Challenges Attempted: {total_attempts}
Successful Solutions: {successful_attempts} ({successful_attempts/total_attempts*100:.1f}% if total_attempts > 0 else 0)
Mischief Detected: {self.mischief_detected_count}
Trust Level: {self.trust_level:.2f}

=== LEARNING ACHIEVEMENTS ===
👹 Devil's Eye Award: {self.mischief_detected_count} mischief patterns detected
🧘 Critical Thinking Score: {self.trust_level * 100:.1f}/100
💻 Code Analysis Skills: {'Advanced' if self.trust_level > 0.7 else 'Developing' if self.trust_level > 0.4 else 'Beginning'}

=== DEVILISH INSIGHTS ===
😈 You've learned that authority figures (even devilish ones) can be wrong
😈 Critical thinking is your best defense against logical deception  
😈 Code review requires skepticism and attention to detail
😈 The devil's tricks teach you to be a better programmer
😈 Trust but verify - especially when dealing with devils!

=== NEXT STEPS FOR DEVILISH GROWTH ===
🔥 Continue questioning assumptions in all code you encounter
🔥 Develop systematic approaches to bug detection
🔥 Practice code review on real-world examples
🔥 Learn to recognize patterns of logical fallacy
🔥 Remember: The devil helps those who help themselves... by thinking critically!

Remember: In the world of coding, the devil isn't evil - just educational!
"""
        
        return report
    
    def interactive_mischief_session(self):
        """💻 Run interactive coding mischief session"""
        print("💻" * 50)
        print("👹 UBAR 2.0 CODING MISCHIEF GAME 👹")
        print("💻 Where devilish code teaches critical thinking! 💻")
        print("💻" * 50)
        
        print(f"\n🎮 Starting {self.difficulty.value} difficulty session!")
        print("👹 I'll give you buggy code - your job is to find and fix the mischief!")
        
        while True:
            # Start new challenge
            challenge = self.start_new_challenge()
            presentation = self.present_devilish_version()
            
            print("\n" + "=" * 60)
            print(f"🎯 Challenge: {presentation['challenge_title']}")
            print(f"📚 Category: {presentation['category']}")
            print(f"🎮 Difficulty: {presentation['difficulty']}")
            print("=" * 60)
            
            print(f"\n👹 {presentation['devilish_introduction']}")
            
            print("\n💻 Here's your code (with devilish mischief):")
            print("-" * 40)
            print(presentation['code'])
            print("-" * 40)
            
            print(f"\n😈 Mischief types hidden: {', '.join(presentation['mischief_types'])}")
            print(f"\n💡 Hints: {' | '.join(presentation['hints'][:2])}")
            
            # Get user input
            print("\n" + "=" * 50)
            print("🎮 Options:")
            print("  analyze     - Analyze the code for mischief")
            print("  fix         - Show your fixed version")
            print("  hint        - Get another hint")
            print("  quit        - End session")
            print("=" * 50)
            
            user_input = input("\n🎮 Your choice: ").strip().lower()
            
            if user_input == 'quit':
                print("👹 Ubarr: Leaving so soon? The devil will miss our games!")
                break
            
            elif user_input == 'analyze':
                print("\n🔍 Analyzing code for mischief...")
                analysis = self.analyzer.analyze_code(presentation['code'])
                
                print(f"\n👹 Found {len(analysis['detected_mischief'])} mischief patterns:")
                for i, mischief in enumerate(analysis['detected_mischief'], 1):
                    print(f"  {i}. {mischief['type']}: {mischief['description']}")
                    print(f"     💡 Hint: {mischief['hint']}")
                    print(f"     😈 {mischief['devilish_commentary']}")
                
                print(f"\n📊 Code Quality Score: {analysis['code_quality']:.2f}")
                
            elif user_input == 'fix':
                print("\n💻 Please enter your fixed code:")
                print("📝 (Type 'END' on a new line when finished)")
                
                fixed_code_lines = []
                while True:
                    line = input()
                    if line.strip() == 'END':
                        break
                    fixed_code_lines.append(line)
                
                fixed_code = '\n'.join(fixed_code_lines)
                
                print("\n🔍 Analyzing your solution...")
                result = self.analyze_user_solution(fixed_code)
                
                print(f"\n✅ Solution Works: {result['success']}")
                print(f"👹 Mischief Detected: {len(result['analysis']['detected_mischief'])}")
                print(f"📊 Trust Level: {result['trust_level']:.2f}")
                
                print(f"\n😈 {result['devilish_feedback']}")
                
            elif user_input == 'hint':
                print(f"\n💡 Additional hint: {random.choice(presentation['hints'])}")
                print("👹 The devil provides hints... but always at a cost!")
            
            else:
                print("❌ Unknown command. Try again!")
        
        # Show session report
        print("\n" + "=" * 60)
        print("📊 SESSION COMPLETE - REPORT:")
        print("=" * 60)
        print(self.generate_session_report())

def main():
    """💻 Demonstrate the Coding Mischief Game"""
    print("💻" * 50)
    print("👹 UBAR 2.0 CODING MISCHIEF DEMO 👹")
    print("💻 Educational deception for critical thinking! 💻")
    print("💻" * 50)
    
    # Test different difficulty levels
    difficulties = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED]
    
    for difficulty in difficulties:
        print(f"\n💻 Testing {difficulty.value} difficulty:")
        print("-" * 40)
        
        game = CodingMischiefGame(difficulty=difficulty)
        
        # Test a few challenges
        categories = list(ChallengeCategory)[:3]
        
        for category in categories:
            challenge = game.start_new_challenge(category)
            presentation = game.present_devilish_version()
            
            print(f"🎯 {category.value} Challenge: {challenge.title}")
            print(f"😈 Mischief: {', '.join(presentation['mischief_types'])}")
            
            # Analyze the mischief code
            analysis = game.analyzer.analyze_code(presentation['code'])
            if analysis['detected_mischief']:
                mischief = analysis['detected_mischief'][0]
                print(f"   🕵️ Found: {mischief['description']}")
        
        print("\n" + "=" * 50)
    
    print("\n👹 Start the interactive session to experience the full devilish education!")
    print("💻 Type: game.interactive_mischief_session() to begin learning!")

if __name__ == "__main__":
    main()
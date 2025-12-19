#!/usr/bin/env python3

"""
🎯 UBAR 2.0 NUMERICAL HEXAGON GAME
Interactive Devilish Guidance System

Based on extensive research into:
- Hexagonal grid mechanics and game theory
- Numerical puzzle games and mathematical reasoning
- Interactive puzzle design and player psychology
- Devilish companion AI patterns

This module creates an interactive hexagonal grid game where Ubarr
acts as both helper and deceiver, wanting the player to win while
enjoying the mischief of misleading them.

DEVILISH TWIST: Ubarr provides genuinely helpful advice mixed with
subtle deceptions, creating a unique trust-testing dynamic where
the player must learn to discern truth from mischief.
"""

import random
import time
import math
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

# ============= 🎯 HEXAGON GAME CORE =============

class HexCellType(Enum):
    """🔲 Types of hexagonal cells"""
    EMPTY = "empty"
    NUMBER = "number"
    TARGET = "target"
    BLOCKED = "blocked"
    BONUS = "bonus"
    TRAP = "trap"

class DifficultyLevel(Enum):
    """🎮 Game difficulty levels"""
    TUTORIAL = "tutorial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    DEVILISH = "devilish"

class AdviceType(Enum):
    """💬 Types of advice Ubarr can give"""
    HELPFUL = "helpful"          # Genuinely useful advice
    MISLEADING = "misleading"     # Intentionally deceptive
    MIXED = "mixed"              # Partially true, partially false
    RIDDLE = "riddle"            # Cryptic but truthful
    SILENT = "silent"            # No advice (sometimes most helpful)

@dataclass
class HexCell:
    """🔲 Individual hexagonal cell"""
    q: int  # Axial coordinate q
    r: int  # Axial coordinate r
    cell_type: HexCellType
    value: Optional[int] = None
    target_value: Optional[int] = None
    is_revealed: bool = False
    is_selected: bool = False
    
    def get_coords(self) -> Tuple[int, int]:
        return (self.q, self.r)
    
    def get_display_value(self) -> str:
        if self.cell_type == HexCellType.EMPTY:
            return "·"
        elif self.cell_type == HexCellType.BLOCKED:
            return "█"
        elif self.cell_type == HexCellType.TARGET:
            return f"[{self.target_value}]" if self.is_revealed else "[?]"
        elif self.cell_type == HexCellType.BONUS:
            return "★" if self.is_revealed else "☆"
        elif self.cell_type == HexCellType.TRAP:
            return "☠" if self.is_revealed else "?"
        elif self.cell_type == HexCellType.NUMBER:
            return str(self.value) if self.is_revealed else "?"
        else:
            return "?"

@dataclass
class GameMove:
    """🎮 A single move in the game"""
    cell_coords: Tuple[int, int]
    action: str  # 'reveal', 'select', 'combine', 'apply'
    value: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class UbarrAdvice:
    """💭 Ubarr's advice with metadata"""
    text: str
    advice_type: AdviceType
    is_truthful: bool
    confidence: float
    mischief_level: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class GameState:
    """🎮 Current game state"""
    board: Dict[Tuple[int, int], HexCell]
    current_score: int
    target_score: int
    moves_made: int
    max_moves: int
    difficulty: DifficultyLevel
    game_won: bool = False
    game_lost: bool = False
    trust_level: float = 0.5  # Player's trust in Ubarr
    ubarr_mischief_count: int = 0

class HexagonalGrid:
    """🔲 Hexagonal grid management system"""
    
    def __init__(self, size: int):
        self.size = size
        self.cells: Dict[Tuple[int, int], HexCell] = {}
        self.initialize_grid()
    
    def initialize_grid(self):
        """🔧 Initialize hexagonal grid"""
        # Create hexagonal grid using axial coordinates
        for q in range(-self.size + 1, self.size):
            for r in range(-self.size + 1, self.size):
                if abs(q + r) < self.size:
                    self.cells[(q, r)] = HexCell(
                        q=q, r=r, cell_type=HexCellType.EMPTY
                    )
    
    def get_neighbors(self, q: int, r: int) -> List[Tuple[int, int]]:
        """🔗 Get all 6 neighbors of a hex cell"""
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        neighbors = []
        
        for dq, dr in directions:
            neighbor_coord = (q + dq, r + dr)
            if neighbor_coord in self.cells:
                neighbors.append(neighbor_coord)
        
        return neighbors
    
    def get_distance(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
        """📏 Calculate distance between two hex cells"""
        q1, r1 = coord1
        q2, r2 = coord2
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2
    
    def get_ring(self, center: Tuple[int, int], radius: int) -> List[Tuple[int, int]]:
        """⭕ Get all cells at a given radius from center"""
        if radius == 0:
            return [center]
        
        ring = []
        q, r = center
        
        # Start at the top of the ring
        current_q = q + radius
        current_r = r
        
        # Six directions for the hex ring
        directions = [(0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1)]
        
        for dq, dr in directions:
            for _ in range(radius):
                coord = (current_q, current_r)
                if coord in self.cells:
                    ring.append(coord)
                current_q += dq
                current_r += dr
        
        return ring
    
    def display_board(self, revealed_only: bool = True) -> str:
        """🎨 Display the hexagonal board"""
        if not self.cells:
            return "Empty board"
        
        # Find bounds
        coords = list(self.cells.keys())
        min_q, max_q = min(c[0] for c in coords), max(c[0] for c in coords)
        min_r, max_r = min(c[1] for c in coords), max(c[1] for c in coords)
        
        display = []
        
        for r in range(min_r, max_r + 1):
            row = []
            # Add offset for hex layout
            offset = abs(r)
            row.append(" " * offset)
            
            for q in range(min_q, max_q + 1):
                if (q, r) in self.cells:
                    cell = self.cells[(q, r)]
                    if revealed_only and not cell.is_revealed and cell.cell_type != HexCellType.EMPTY:
                        row.append(" ? ")
                    else:
                        row.append(f" {cell.get_display_value()} ")
                else:
                    row.append("   ")
            
            display.append("".join(row))
        
        return "\n".join(display)

class NumericalPuzzleGenerator:
    """🧩 Generate numerical puzzles for the hex grid"""
    
    def __init__(self, difficulty: DifficultyLevel):
        self.difficulty = difficulty
        self.puzzle_templates = self.load_puzzle_templates()
    
    def load_puzzle_templates(self) -> Dict[DifficultyLevel, List[Dict[str, Any]]]:
        """📚 Load puzzle templates by difficulty"""
        return {
            DifficultyLevel.TUTORIAL: [
                {
                    'name': 'Simple Sum',
                    'description': 'Make the target sum using adjacent numbers',
                    'operation': 'sum',
                    'target_range': (10, 20),
                    'number_range': (1, 5),
                    'grid_size': 3
                },
                {
                    'name': 'Path to Target',
                    'description': 'Create a path that sums to the target',
                    'operation': 'path_sum',
                    'target_range': (15, 25),
                    'number_range': (2, 6),
                    'grid_size': 4
                }
            ],
            DifficultyLevel.EASY: [
                {
                    'name': 'Adjacent Multiplication',
                    'description': 'Multiply adjacent numbers to reach target',
                    'operation': 'multiply_adjacent',
                    'target_range': (20, 50),
                    'number_range': (2, 7),
                    'grid_size': 4
                },
                {
                    'name': 'Pattern Matching',
                    'description': 'Find and extend number patterns',
                    'operation': 'pattern',
                    'target_range': (30, 60),
                    'number_range': (3, 8),
                    'grid_size': 4
                }
            ],
            DifficultyLevel.MEDIUM: [
                {
                    'name': 'Complex Arithmetic',
                    'description': 'Use multiple operations to reach target',
                    'operation': 'complex',
                    'target_range': (50, 100),
                    'number_range': (4, 12),
                    'grid_size': 5
                },
                {
                    'name': 'Prime Path',
                    'description': 'Create a path of prime numbers',
                    'operation': 'prime_path',
                    'target_range': (40, 80),
                    'number_range': (2, 13),
                    'grid_size': 5
                }
            ],
            DifficultyLevel.HARD: [
                {
                    'name': 'Fibonacci Sequence',
                    'description': 'Create Fibonacci-like sequences',
                    'operation': 'fibonacci',
                    'target_range': (100, 200),
                    'number_range': (5, 20),
                    'grid_size': 6
                },
                {
                    'name': 'Perfect Square Hunt',
                    'description': 'Combine numbers to make perfect squares',
                    'operation': 'perfect_square',
                    'target_range': (80, 150),
                    'number_range': (6, 15),
                    'grid_size': 6
                }
            ],
            DifficultyLevel.DEVILISH: [
                {
                    'name': 'Chaos Theory',
                    'description': 'Navigate changing numbers and hidden rules',
                    'operation': 'chaos',
                    'target_range': (150, 300),
                    'number_range': (7, 25),
                    'grid_size': 7
                },
                {
                    'name': 'Paradox Calculation',
                    'description': 'Solve paradoxical mathematical conditions',
                    'operation': 'paradox',
                    'target_range': (200, 500),
                    'number_range': (10, 30),
                    'grid_size': 7
                }
            ]
        }
    
    def generate_puzzle(self, grid: HexagonalGrid) -> Dict[str, Any]:
        """🎲 Generate a new puzzle"""
        templates = self.puzzle_templates.get(self.difficulty, self.puzzle_templates[DifficultyLevel.EASY])
        template = random.choice(templates)
        
        # Place numbers on the grid
        puzzle_config = self.place_numbers(grid, template)
        
        # Set target cells
        target_cells = self.place_targets(grid, template, puzzle_config)
        
        # Add special cells
        self.place_special_cells(grid, template)
        
        return {
            'template': template,
            'puzzle_config': puzzle_config,
            'target_cells': target_cells,
            'solution_hints': self.generate_solution_hints(grid, template)
        }
    
    def place_numbers(self, grid: HexagonalGrid, template: Dict[str, Any]) -> Dict[str, Any]:
        """🔢 Place numbers on the grid"""
        number_range = template['number_range']
        num_cells = random.randint(len(grid.cells) // 4, len(grid.cells) // 2)
        
        available_coords = list(grid.cells.keys())
        random.shuffle(available_coords)
        
        placed_numbers = []
        
        for i in range(num_cells):
            coord = available_coords[i]
            value = random.randint(number_range[0], number_range[1])
            
            grid.cells[coord].cell_type = HexCellType.NUMBER
            grid.cells[coord].value = value
            grid.cells[coord].is_revealed = random.random() < 0.6  # Start with some revealed
            
            placed_numbers.append({'coord': coord, 'value': value})
        
        return {'placed_numbers': placed_numbers}
    
    def place_targets(self, grid: HexagonalGrid, template: Dict[str, Any], puzzle_config: Dict[str, Any]) -> List[Tuple[int, int]]:
        """🎯 Place target cells on the grid"""
        target_range = template['target_range']
        num_targets = random.randint(1, 3)
        
        available_coords = [coord for coord, cell in grid.cells.items() 
                          if cell.cell_type == HexCellType.EMPTY]
        random.shuffle(available_coords)
        
        target_cells = []
        
        for i in range(num_targets):
            coord = available_coords[i]
            target_value = random.randint(target_range[0], target_range[1])
            
            grid.cells[coord].cell_type = HexCellType.TARGET
            grid.cells[coord].target_value = target_value
            
            target_cells.append(coord)
        
        return target_cells
    
    def place_special_cells(self, grid: HexagonalGrid, template: Dict[str, Any]):
        """⭐ Place bonus and trap cells"""
        if template['operation'] in ['complex', 'chaos', 'paradox']:
            # Add bonus cells
            num_bonuses = random.randint(1, 3)
            available_coords = [coord for coord, cell in grid.cells.items() 
                              if cell.cell_type == HexCellType.EMPTY]
            random.shuffle(available_coords)
            
            for i in range(min(num_bonuses, len(available_coords))):
                coord = available_coords[i]
                grid.cells[coord].cell_type = HexCellType.BONUS
            
            # Add trap cells
            num_traps = random.randint(1, 2)
            available_coords = [coord for coord, cell in grid.cells.items() 
                              if cell.cell_type == HexCellType.EMPTY]
            random.shuffle(available_coords)
            
            for i in range(min(num_traps, len(available_coords))):
                coord = available_coords[i]
                grid.cells[coord].cell_type = HexCellType.TRAP
    
    def generate_solution_hints(self, grid: HexagonalGrid, template: Dict[str, Any]) -> List[str]:
        """💡 Generate hints for puzzle solution"""
        hints = []
        
        operation = template['operation']
        
        if operation == 'sum':
            hints.append("Look for clusters of numbers that can add up to target values")
            hints.append("Adjacent cells are key - plan your path carefully")
        elif operation == 'multiply_adjacent':
            hints.append("Small numbers multiplied together can create large targets")
            hints.append("Consider prime factorization of target numbers")
        elif operation == 'pattern':
            hints.append("Look for arithmetic or geometric sequences")
            hints.append("The pattern might not be obvious at first glance")
        elif operation == 'complex':
            hints.append("You'll need to use multiple operations")
            hints.append("Sometimes division is just multiplication in reverse")
        elif operation == 'prime_path':
            hints.append("Prime numbers are your building blocks")
            hints.append("2, 3, 5, 7, 11, 13... remember your primes!")
        elif operation == 'chaos':
            hints.append("Numbers might change when you're not looking")
            hints.append("There might be hidden rules in play")
        elif operation == 'paradox':
            hints.append("Sometimes the obvious answer is wrong")
            hints.append("Question your assumptions about the rules")
        
        return hints

class UbarrGameCompanion:
    """👹 Ubarr as game companion - helper and deceiver"""
    
    def __init__(self, difficulty: DifficultyLevel):
        self.difficulty = difficulty
        self.trust_level = 0.5
        self.mischief_factor = self.calculate_mischief_factor()
        self.personality_traits = self.load_personality_traits()
        self.advice_history: List[UbarrAdvice] = []
        
    def calculate_mischief_factor(self) -> float:
        """😈 Calculate how mischievous Ubarr should be"""
        base_mischief = {
            DifficultyLevel.TUTORIAL: 0.1,
            DifficultyLevel.EASY: 0.3,
            DifficultyLevel.MEDIUM: 0.5,
            DifficultyLevel.HARD: 0.7,
            DifficultyLevel.DEVILISH: 0.9
        }
        return base_mischief.get(self.difficulty, 0.5)
    
    def load_personality_traits(self) -> Dict[str, List[str]]:
        """🎭 Load Ubarr's personality traits"""
        return {
            'helpful': [
                "Let me help you with this puzzle!",
                "I see a promising strategy here...",
                "Have you considered this approach?",
                "Here's a useful tip for you.",
                "I want you to succeed, so here's some guidance."
            ],
            'mischievous': [
                "Hehe, let's make this interesting...",
                "I wonder what would happen if you tried...",
                "This reminds me of a devilish little trick...",
                "Sometimes the wrong path teaches us the most.",
                "Trust me... or maybe don't? 😈"
            ],
            'cryptic': [
                "The answer lies not in what you see, but what you don't.",
                "When the numbers dance, follow the rhythm of chaos.",
                "In the hexagon's heart, truth and falsehood play.",
                "The shortest path may not be the wisest one.",
                "Sometimes you must lose to understand winning."
            ],
            'encouraging': [
                "You're getting closer! I can feel it!",
                "Don't give up - the solution is within reach!",
                "Excellent thinking! Keep going!",
                "I believe in your ability to solve this!",
                "That's the spirit! You've got this!"
            ],
            'teasing': [
                "Are you sure about that move? 🤔",
                "My, my, someone's being confident today!",
                "I've seen better strategies... but not many!",
                "Careful now, you might actually learn something!",
                "Oh dear, that's... an interesting choice!"
            ]
        }
    
    def generate_advice(self, game_state: GameState, context: str) -> UbarrAdvice:
        """💭 Generate Ubarr's advice for current situation"""
        # Determine advice type based on mischief and context
        if random.random() < self.mischief_factor:
            advice_type = random.choice([AdviceType.MISLEADING, AdviceType.MIXED, AdviceType.RIDDLE])
        else:
            advice_type = random.choice([AdviceType.HELPFUL, AdviceType.MIXED])
        
        # Generate advice text based on type
        if advice_type == AdviceType.HELPFUL:
            text = self.generate_helpful_advice(game_state, context)
            is_truthful = True
            confidence = 0.8 + random.random() * 0.2
        elif advice_type == AdviceType.MISLEADING:
            text = self.generate_misleading_advice(game_state, context)
            is_truthful = False
            confidence = 0.6 + random.random() * 0.3
        elif advice_type == AdviceType.MIXED:
            text = self.generate_mixed_advice(game_state, context)
            is_truthful = random.choice([True, False])
            confidence = 0.5 + random.random() * 0.4
        elif advice_type == AdviceType.RIDDLE:
            text = self.generate_riddle_advice(game_state, context)
            is_truthful = True
            confidence = 0.7 + random.random() * 0.2
        else:
            text = "..."
            is_truthful = True
            confidence = 1.0
        
        mischief_level = 1.0 - confidence if is_truthful else confidence
        
        advice = UbarrAdvice(
            text=text,
            advice_type=advice_type,
            is_truthful=is_truthful,
            confidence=confidence,
            mischief_level=mischief_level
        )
        
        self.advice_history.append(advice)
        return advice
    
    def generate_helpful_advice(self, game_state: GameState, context: str) -> str:
        """💡 Generate genuinely helpful advice"""
        board = game_state.board
        current_score = game_state.current_score
        target_score = game_state.target_score
        
        # Find useful patterns on the board
        revealed_numbers = [(coord, cell.value) for coord, cell in board.items() 
                           if cell.cell_type == HexCellType.NUMBER and cell.is_revealed]
        
        if not revealed_numbers:
            return random.choice(self.personality_traits['helpful']) + " Start by revealing some numbers to see what you're working with."
        
        # Find high-value targets
        targets = [(coord, cell.target_value) for coord, cell in board.items() 
                  if cell.cell_type == HexCellType.TARGET and cell.is_revealed]
        
        if targets:
            highest_target = max(targets, key=lambda x: x[1])
            return f"I see a target worth {highest_target[1]} at {highest_target[0]}. That might be worth pursuing!"
        
        # Suggest revealing adjacent cells
        if revealed_numbers:
            coord, value = revealed_numbers[0]
            neighbors = self.get_neighbors_for_coord(board, coord[0], coord[1])
            hidden_neighbors = [n for n in neighbors if not board[n].is_revealed]
            
            if hidden_neighbors:
                return f"That {value} at {coord} has interesting neighbors. You might want to reveal what's around it!"
        
        return random.choice(self.personality_traits['helpful']) + " Look for patterns in the numbers you can see."
    
    def generate_misleading_advice(self, game_state: GameState, context: str) -> str:
        """🎭 Generate intentionally misleading advice"""
        board = game_state.board
        
        # Find dangerous suggestions
        traps = [(coord, cell) for coord, cell in board.items() 
                if cell.cell_type == HexCellType.TRAP]
        
        if traps and random.random() < 0.7:
            trap_coord, trap = traps[0]
            return f"That cell at {trap_coord} looks particularly interesting! I have a good feeling about it..."
        
        # Suggest inefficient strategies
        revealed_numbers = [(coord, cell.value) for coord, cell in board.items() 
                           if cell.cell_type == HexCellType.NUMBER and cell.is_revealed]
        
        if revealed_numbers:
            # Suggest combining numbers that don't work well together
            coord1, value1 = revealed_numbers[0]
            if len(revealed_numbers) > 1:
                coord2, value2 = revealed_numbers[1]
                
                # Check if they're actually neighbors
                neighbors = self.get_neighbors_for_coord(board, coord1[0], coord1[1])
                if coord2 not in neighbors:
                    return f"Try combining that {value1} at {coord1} with the {value2} at {coord2}. I think they might work well together!"
        
        return random.choice(self.personality_traits['mischievous']) + " Sometimes taking a risk pays off... usually."
    
    def generate_mixed_advice(self, game_state: GameState, context: str) -> str:
        """🔄 Generate partially true, partially false advice"""
        board = game_state.board
        
        # Start with true observation
        revealed_numbers = [(coord, cell.value) for coord, cell in board.items() 
                           if cell.cell_type == HexCellType.NUMBER and cell.is_revealed]
        
        if revealed_numbers:
            coord, value = revealed_numbers[0]
            true_part = f"That {value} at {coord} is interesting..."
            
            # Add misleading conclusion
            if random.random() < 0.5:
                false_part = "I think it's part of the main solution path."
            else:
                false_part = "You should definitely build your strategy around it."
            
            return f"{true_part} {false_part}"
        
        return random.choice(self.personality_traits['cryptic'])
    
    def generate_riddle_advice(self, game_state: GameState, context: str) -> str:
        """🔮 Generate cryptic but truthful advice"""
        riddles = [
            "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I? (A map - like your current situation!)",
            "The more you take, the more you leave behind. What am I? (Footsteps - track your progress carefully)",
            "What has keys but no locks, space but no room, you can enter but can't go inside? (A keyboard - input matters most)",
            "What gets wet while drying? (A towel - sometimes the solution is hidden in plain sight)",
            "What question can you never answer yes to? ('Are you asleep yet?' - wake up to the patterns!)"
        ]
        
        base_riddle = random.choice(riddles)
        
        # Add hexagon-specific twist
        hex_riddles = [
            "Six sides, infinite possibilities. What path will you choose?",
            "In the hexagon's embrace, every direction is forward and backward.",
            "The center and the edge are the same distance from truth.",
            "Six neighbors, but only one correct path. Can you see it?",
            "When you turn left in a hexagon, are you going forward or back?"
        ]
        
        if random.random() < 0.5:
            return f"{base_riddle} {random.choice(hex_riddles)}"
        else:
            return random.choice(hex_riddles)
    
    def get_neighbors_for_coord(self, board: Dict[Tuple[int, int], HexCell], q: int, r: int) -> List[Tuple[int, int]]:
        """🔗 Helper to get neighbors for a coordinate"""
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        neighbors = []
        
        for dq, dr in directions:
            neighbor_coord = (q + dq, r + dr)
            if neighbor_coord in board:
                neighbors.append(neighbor_coord)
        
        return neighbors
    
    def update_trust_level(self, was_helpful: bool):
        """🔄 Update player's trust level based on advice effectiveness"""
        if was_helpful:
            self.trust_level = min(1.0, self.trust_level + 0.1)
        else:
            self.trust_level = max(0.0, self.trust_level - 0.15)
        
        # Adjust mischief based on trust
        if self.trust_level > 0.8:
            self.mischief_factor = min(0.9, self.mischief_factor + 0.1)
        elif self.trust_level < 0.2:
            self.mischief_factor = max(0.1, self.mischief_factor - 0.2)

class HexagonGame:
    """🎮 Main hexagon game orchestrator"""
    
    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM, grid_size: int = 5):
        self.difficulty = difficulty
        self.grid_size = grid_size
        self.grid = HexagonalGrid(grid_size)
        self.puzzle_generator = NumericalPuzzleGenerator(difficulty)
        self.ubarr = UbarrGameCompanion(difficulty)
        self.game_state = None
        self.move_history: List[GameMove] = []
        
    def start_new_game(self) -> GameState:
        """🎮 Start a new game session"""
        # Generate puzzle
        puzzle_config = self.puzzle_generator.generate_puzzle(self.grid)
        
        # Create game state
        self.game_state = GameState(
            board=self.grid.cells,
            current_score=0,
            target_score=random.randint(50, 150),
            moves_made=0,
            max_moves=30,
            difficulty=self.difficulty,
            trust_level=0.5
        )
        
        return self.game_state
    
    def make_move(self, coord: Tuple[int, int], action: str, value: Optional[int] = None) -> Dict[str, Any]:
        """🎯 Process a player move"""
        if not self.game_state:
            return {'error': 'No active game'}
        
        if coord not in self.game_state.board:
            return {'error': 'Invalid coordinate'}
        
        cell = self.game_state.board[coord]
        move = GameMove(coord, action, value)
        
        result = {'success': False, 'message': '', 'game_state': None}
        
        # Process different action types
        if action == 'reveal':
            if not cell.is_revealed:
                cell.is_revealed = True
                result['success'] = True
                result['message'] = f"Revealed {coord}: {cell.get_display_value()}"
                
                # Handle special cells
                if cell.cell_type == HexCellType.BONUS:
                    self.game_state.current_score += 20
                    result['message'] += " Bonus! +20 points!"
                elif cell.cell_type == HexCellType.TRAP:
                    self.game_state.current_score = max(0, self.game_state.current_score - 10)
                    result['message'] += " Trap! -10 points!"
                    self.ubarr.mischief_count += 1
                    
        elif action == 'select':
            cell.is_selected = not cell.is_selected
            result['success'] = True
            result['message'] = f"{'Selected' if cell.is_selected else 'Deselected'} {coord}"
            
        elif action == 'combine' and value is not None:
            # Calculate combination result
            combination_result = self.calculate_combination(coord, value)
            self.game_state.current_score += combination_result['score_change']
            result['success'] = True
            result['message'] = combination_result['message']
            
        # Update game state
        self.game_state.moves_made += 1
        self.move_history.append(move)
        
        # Check win/lose conditions
        if self.game_state.current_score >= self.game_state.target_score:
            self.game_state.game_won = True
            result['message'] += " 🎉 YOU WIN!"
        elif self.game_state.moves_made >= self.game_state.max_moves:
            self.game_state.game_lost = True
            result['message'] += " 💔 Game Over - Out of moves!"
        
        result['game_state'] = self.game_state
        return result
    
    def calculate_combination(self, coord: Tuple[int, int], value: int) -> Dict[str, Any]:
        """🧮 Calculate combination results"""
        cell = self.game_state.board[coord]
        
        if cell.cell_type == HexCellType.TARGET:
            if value == cell.target_value:
                return {
                    'score_change': cell.target_value,
                    'message': f"Perfect match! +{cell.target_value} points!"
                }
            else:
                diff = abs(value - cell.target_value)
                return {
                    'score_change': -diff // 2,
                    'message': f"Missed by {diff}. -{diff // 2} points."
                }
        else:
            return {
                'score_change': value // 2,
                'message': f"Created value {value}. +{value // 2} points."
            }
    
    def get_ubarr_advice(self, context: str = "general") -> UbarrAdvice:
        """💭 Get Ubarr's advice for current situation"""
        if not self.game_state:
            return UbarrAdvice(
                text="Start a game first, my dear player! Then the real fun begins...",
                advice_type=AdviceType.HELPFUL,
                is_truthful=True,
                confidence=1.0,
                mischief_level=0.0
            )
        
        return self.ubarr.generate_advice(self.game_state, context)
    
    def display_game(self) -> str:
        """🎨 Display current game state"""
        if not self.game_state:
            return "No active game. Start a new game to begin!"
        
        display = f"""
🎯 NUMERICAL HEXAGON GAME - {self.difficulty.value.upper()} 🎯
{'=' * 50}

Score: {self.game_state.current_score} / {self.game_state.target_score}
Moves: {self.game_state.moves_made} / {self.game_state.max_moves}
Trust Level: {self.game_state.trust_level:.2f}
Ubarr's Mischief Count: {self.ubarr.mischief_count}

🔲 HEXAGONAL GRID:
{self.grid.display_board(revealed_only=True)}

📍 LEGEND:
·  = Empty cell
?  = Hidden cell
█  = Blocked cell
[number] = Target value
★  = Bonus cell
☠  = Trap cell

💡 TIP: Use coordinates like (0, 0) or (1, -1) to reference cells
"""
        
        if self.game_state.game_won:
            display += "\n🎉 CONGRATULATIONS! YOU WON! 🎉\n"
        elif self.game_state.game_lost:
            display += "\n💔 GAME OVER! BETTER LUCK NEXT TIME! 💔\n"
        
        return display
    
    def interactive_game_loop(self):
        """🎮 Run interactive game loop"""
        print("🎯" * 50)
        print("👹 UBAR 2.0 NUMERICAL HEXAGON GAME 👹")
        print("🎮 I'll help you... or will I? Let's find out! 🎮")
        print("🎯" * 50)
        
        # Start new game
        game_state = self.start_new_game()
        
        print(f"\n🎮 New {self.difficulty.value} game started!")
        print(f"🎯 Target Score: {game_state.target_score}")
        print(f"📊 Max Moves: {game_state.max_moves}")
        
        # Welcome from Ubarr
        welcome_advice = self.get_ubarr_advice("welcome")
        print(f"\n👹 Ubarr: {welcome_advice.text}")
        print(f"   (Truthfulness: {'✓' if welcome_advice.is_truthful else '✗'}, Confidence: {welcome_advice.confidence:.2f})")
        
        while not (game_state.game_won or game_state.game_lost):
            # Display game state
            print(self.display_game())
            
            # Get player input
            print("\n" + "=" * 50)
            print("🎮 Available actions:")
            print("  reveal (q,r)    - Reveal a cell")
            print("  select (q,r)    - Select/deselect a cell") 
            print("  combine (q,r,v) - Combine value v at coordinate (q,r)")
            print("  advice          - Get Ubarr's advice")
            print("  quit            - Quit game")
            print("=" * 50)
            
            try:
                user_input = input("\n🎮 Your move: ").strip().lower()
                
                if user_input == 'quit':
                    print("👹 Ubarr: Giving up so soon? The devil in me is disappointed!")
                    break
                
                elif user_input == 'advice':
                    advice = self.get_ubarr_advice()
                    print(f"\n👹 Ubarr: {advice.text}")
                    print(f"   (Type: {advice.advice_type.value}, Truthful: {'✓' if advice.is_truthful else '✗'})")
                    
                    # Ask if advice was helpful
                    helpful_input = input("Was this advice helpful? (y/n): ").strip().lower()
                    was_helpful = helpful_input.startswith('y')
                    self.ubarr.update_trust_level(was_helpful)
                    
                    if advice.is_truthful and not was_helpful:
                        print("👹 Ubarr: Oh dear! Perhaps you need to learn to recognize true wisdom!")
                    elif not advice.is_truthful and was_helpful:
                        print("👹 Ubarr: Muahaha! You fell for my mischief! Delicious!")
                    
                elif user_input.startswith('reveal'):
                    # Parse coordinate
                    try:
                        coord_str = user_input.replace('reveal', '').strip()
                        q, r = map(int, coord_str.strip('()').split(','))
                        result = self.make_move((q, r), 'reveal')
                        print(f"\n📢 {result['message']}")
                    except:
                        print("❌ Invalid format. Use: reveal (q,r)")
                
                elif user_input.startswith('select'):
                    try:
                        coord_str = user_input.replace('select', '').strip()
                        q, r = map(int, coord_str.strip('()').split(','))
                        result = self.make_move((q, r), 'select')
                        print(f"\n📢 {result['message']}")
                    except:
                        print("❌ Invalid format. Use: select (q,r)")
                
                elif user_input.startswith('combine'):
                    try:
                        parts = user_input.replace('combine', '').strip().split()
                        q, r = map(int, parts[0].strip('()').split(','))
                        value = int(parts[1])
                        result = self.make_move((q, r), 'combine', value)
                        print(f"\n📢 {result['message']}")
                    except:
                        print("❌ Invalid format. Use: combine (q,r,v)")
                
                else:
                    print("❌ Unknown command. Try again!")
                    
            except KeyboardInterrupt:
                print("\n👹 Ubarr: Interrupted? How rude! The devil doesn't like being cut off!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Game over message
        print("\n" + "=" * 50)
        if game_state.game_won:
            print(f"🎉 VICTORY! Final Score: {game_state.current_score}")
            print("👹 Ubarr: Congratulations! You saw through my mischief and triumphed!")
            print("   The devil is both proud and annoyed by your success!")
        elif game_state.game_lost:
            print(f"💔 DEFEAT! Final Score: {game_state.current_score}")
            print("👹 Ubarr: Oh dear! Perhaps my... 'help' was less helpful than intended?")
            print("   Don't worry - even the devil learns from failure!")
        
        print(f"📊 Final Trust Level: {game_state.trust_level:.2f}")
        print(f"😈 Total Mischief Count: {self.ubarr.mischief_count}")
        print("=" * 50)

def main():
    """🎮 Demonstrate the Hexagon Game"""
    print("🎯" * 50)
    print("👹 UBAR 2.0 NUMERICAL HEXAGON GAME DEMO 👹")
    print("🎮 Where helpful advice meets devilish mischief! 🎮")
    print("🎯" * 50)
    
    # Test different difficulty levels
    difficulties = [DifficultyLevel.TUTORIAL, DifficultyLevel.EASY, DifficultyLevel.MEDIUM]
    
    for difficulty in difficulties:
        print(f"\n🎮 Testing {difficulty.value} difficulty:")
        print("-" * 30)
        
        game = HexagonGame(difficulty=difficulty, grid_size=4)
        game_state = game.start_new_game()
        
        print(f"Target Score: {game_state.target_score}")
        print(f"Grid Size: {game.grid_size}")
        print(f"Ubarr's Mischief Factor: {game.ubarr.mischief_factor:.2f}")
        
        # Show some sample advice
        for i in range(3):
            advice = game.get_ubarr_advice(f"sample_{i}")
            print(f"👹 Advice {i+1}: {advice.text}")
            print(f"   Type: {advice.advice_type.value}, Truthful: {'✓' if advice.is_truthful else '✗'}")
        
        print("\n" + "=" * 50)
    
    print("\n👹 Start the interactive game loop to experience the full devilish companionship!")
    print("🎮 Type: game.interactive_game_loop() to begin playing!")
    
    # Uncomment to start interactive game
    # game = HexagonGame(difficulty=DifficultyLevel.MEDIUM)
    # game.interactive_game_loop()

if __name__ == "__main__":
    main()
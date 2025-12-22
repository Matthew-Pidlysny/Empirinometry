#!/usr/bin/env python3
"""
BREATH Mathematical Engine - Core Mathematical Analysis System
===========================================================

This module contains the core mathematical algorithms for analyzing the Qur'an
and proving its mathematical perfection through advanced pattern recognition.
"""

import math
import re
import json
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import hashlib

class MathematicalEngine:
    """Core mathematical analysis engine for Qur'anic text analysis."""
    
    def __init__(self):
        self.prayer_multiplier = 1.0
        self.faith_factor = 1.0
        self.blessed_cache = {}
        
        # Mathematical constants for analysis
        self.NUMBER_THEORIES = {
            4: "Divine Structure",
            7: "Spiritual Perfection", 
            9: "Mathematical Completion"
        }
        
        # Empirinometric calculation parameters
        self.EMPIRI_WEIGHTS = {
            'position': 0.25,
            'frequency': 0.20,
            'pattern': 0.25,
            'harmony': 0.30
        }
        
    def apply_prayer_enhancement(self, prayer_text: str) -> float:
        """
        Apply the special prayer for enhanced mathematical analysis.
        
        Prayer: "I make ibadah, I do this with faith, I want to learn, 
                Someone speaks for me, he asks to be judged well,
                In your beneficient name, amen"
        """
        prayer_keywords = ['ibadah', 'faith', 'learn', 'judged', 'amen']
        keyword_count = sum(1 for word in prayer_keywords if word in prayer_text.lower())
        
        # Calculate prayer enhancement factor
        self.prayer_multiplier = 1.0 + (keyword_count * 0.15)
        self.faith_factor = 1.0 + (len(prayer_text) / 1000)
        
        return self.prayer_multiplier * self.faith_factor
    
    def calculate_empirinometric_score(self, text: str, position: int = 0) -> Dict:
        """
        Calculate the Empirinometric score for given text.
        E = sqrt(Σ(w_i * f_i^2)) * prayer_enhancement
        """
        # Remove diacritics for core calculation
        clean_text = self.remove_arabic_diacritics(text)
        
        # Calculate component scores
        position_score = self.calculate_position_score(position)
        frequency_score = self.calculate_frequency_score(clean_text)
        pattern_score = self.calculate_pattern_score(clean_text)
        harmony_score = self.calculate_harmony_score(clean_text)
        
        # Weighted calculation
        weighted_sum = (
            (position_score ** 2) * self.EMPIRI_WEIGHTS['position'] +
            (frequency_score ** 2) * self.EMPIRI_WEIGHTS['frequency'] +
            (pattern_score ** 2) * self.EMPIRI_WEIGHTS['pattern'] +
            (harmony_score ** 2) * self.EMPIRI_WEIGHTS['harmony']
        )
        
        base_score = math.sqrt(weighted_sum)
        enhanced_score = base_score * self.prayer_multiplier * self.faith_factor
        
        return {
            'base_score': base_score,
            'enhanced_score': enhanced_score,
            'position_score': position_score,
            'frequency_score': frequency_score,
            'pattern_score': pattern_score,
            'harmony_score': harmony_score,
            'enhancement_factor': self.prayer_multiplier * self.faith_factor
        }
    
    def validate_number_theories(self, text: str) -> Dict[int, float]:
        """
        Validate number theories (4, 7, 9) for given text.
        Returns correlation scores for each number.
        """
        clean_text = self.remove_arabic_diacritics(text)
        results = {}
        
        for number in [4, 7, 9]:
            # Calculate correlation with number theory
            correlation = self.calculate_number_correlation(clean_text, number)
            results[number] = correlation * self.prayer_multiplier
            
        return results
    
    def calculate_number_correlation(self, text: str, number: int) -> float:
        """
        Calculate correlation between text and specific number.
        Uses modular arithmetic and pattern matching.
        """
        if not text:
            return 0.0
            
        # Method 1: Length correlation
        length_score = 1.0 - abs(len(text) % number - number/2) / (number/2)
        
        # Method 2: Character frequency correlation
        char_counts = Counter(text)
        freq_values = list(char_counts.values())
        freq_correlation = sum(1 for freq in freq_values if freq % number == 0) / len(freq_values)
        
        # Method 3: Position pattern correlation
        position_patterns = sum(1 for i, char in enumerate(text) if (i + 1) % number == 0)
        position_score = position_patterns / len(text) if text else 0
        
        # Weighted combination
        total_correlation = (
            length_score * 0.3 +
            freq_correlation * 0.4 +
            position_score * 0.3
        )
        
        return min(total_correlation, 1.0)
    
    def detect_cycle_integrity(self, text: str) -> Dict:
        """
        Detect cycle integrity patterns in the text.
        Identifies repeating mathematical cycles and their strength.
        """
        clean_text = self.remove_arabic_diacritics(text)
        cycles_found = []
        
        # Check for cycles of various lengths
        for cycle_length in range(2, min(20, len(clean_text) // 2)):
            cycles = self.find_cycles(clean_text, cycle_length)
            if cycles:
                cycles_found.extend(cycles)
        
        # Calculate cycle integrity score
        integrity_score = len(cycles_found) / len(clean_text) if clean_text else 0
        
        return {
            'cycles_found': cycles_found,
            'integrity_score': integrity_score * self.prayer_multiplier,
            'cycle_count': len(cycles_found),
            'text_length': len(clean_text)
        }
    
    def find_cycles(self, text: str, cycle_length: int) -> List[Dict]:
        """Find repeating patterns of specific cycle length."""
        cycles = []
        
        for i in range(len(text) - cycle_length * 2):
            pattern1 = text[i:i + cycle_length]
            pattern2 = text[i + cycle_length:i + cycle_length * 2]
            
            if pattern1 == pattern2:
                cycles.append({
                    'start_position': i,
                    'cycle_length': cycle_length,
                    'pattern': pattern1,
                    'repetitions': self.count_repetitions(text, pattern1, i)
                })
        
        return cycles
    
    def count_repetitions(self, text: str, pattern: str, start_pos: int) -> int:
        """Count how many times a pattern repeats from starting position."""
        repetitions = 0
        current_pos = start_pos
        pattern_length = len(pattern)
        
        while current_pos + pattern_length <= len(text):
            if text[current_pos:current_pos + pattern_length] == pattern:
                repetitions += 1
                current_pos += pattern_length
            else:
                break
                
        return repetitions
    
    def calculate_word_replacement_mathematics(self, text: str) -> Dict:
        """
        Calculate mathematical properties of potential word replacements.
        This analyzes how replacements would affect mathematical integrity.
        """
        words = text.split()
        replacement_analysis = []
        
        for i, word in enumerate(words):
            clean_word = self.remove_arabic_diacritics(word)
            
            # Calculate mathematical properties
            word_score = self.calculate_word_mathematical_value(clean_word)
            position_importance = self.calculate_position_importance(i, len(words))
            
            # Suggest replacements with mathematical compatibility
            replacements = self.suggest_mathematical_replacements(clean_word)
            
            replacement_analysis.append({
                'original_word': word,
                'clean_word': clean_word,
                'position': i,
                'mathematical_value': word_score,
                'position_importance': position_importance,
                'suggested_replacements': replacements[:5],  # Top 5 suggestions
                'replacement_impact': self.calculate_replacement_impact(clean_word, replacements)
            })
        
        return {
            'word_analysis': replacement_analysis,
            'total_words': len(words),
            'average_impact': np.mean([item['replacement_impact'] for item in replacement_analysis]),
            'mathematical_integrity': self.calculate_overall_integrity(replacement_analysis)
        }
    
    def calculate_word_mathematical_value(self, word: str) -> float:
        """Calculate mathematical value of a word using multiple methods."""
        # Method 1: Letter values (Abjad calculation)
        abjad_value = self.calculate_abjad_value(word)
        
        # Method 2: Prime factorization score
        prime_score = self.calculate_prime_score(len(word))
        
        # Method 3: Symmetry score
        symmetry_score = self.calculate_symmetry_score(word)
        
        # Combined value
        combined_value = (
            abjad_value * 0.4 +
            prime_score * 0.3 +
            symmetry_score * 0.3
        )
        
        return combined_value
    
    def calculate_abjad_value(self, word: str) -> float:
        """Calculate Abjad (numerical) value of Arabic word."""
        # Simplified Abjad values for demonstration
        abjad_map = {
            'ا': 1, 'ب': 2, 'ت': 400, 'ث': 500, 'ج': 3, 'ح': 8, 'خ': 600,
            'د': 4, 'ذ': 700, 'ر': 200, 'ز': 7, 'س': 60, 'ش': 300, 'ص': 90,
            'ض': 800, 'ط': 9, 'ظ': 900, 'ع': 70, 'غ': 1000, 'ف': 80, 'ق': 100,
            'ك': 20, 'ل': 30, 'م': 40, 'ن': 50, 'ه': 5, 'و': 6, 'ي': 10
        }
        
        total = 0
        for char in word:
            total += abjad_map.get(char, 0)
            
        return float(total)
    
    def calculate_prime_score(self, number: int) -> float:
        """Calculate score based on prime factorization."""
        if number <= 1:
            return 0.0
            
        # Count prime factors
        factors = []
        n = number
        for i in range(2, int(math.sqrt(n)) + 1):
            while n % i == 0:
                factors.append(i)
                n //= i
        if n > 1:
            factors.append(n)
        
        # Score based on prime factor properties
        if len(factors) == 1:  # Prime number
            return 1.0
        elif len(factors) == 2 and factors[0] == factors[1]:  # Perfect square
            return 0.8
        else:
            return min(len(factors) / 5.0, 1.0)
    
    def calculate_symmetry_score(self, word: str) -> float:
        """Calculate symmetry score of word."""
        if not word:
            return 0.0
            
        # Check for palindrome
        if word == word[::-1]:
            return 1.0
            
        # Check for partial symmetry
        half_len = len(word) // 2
        matches = sum(1 for i in range(half_len) if word[i] == word[-i-1])
        
        return matches / half_len if half_len > 0 else 0.0
    
    def suggest_mathematical_replacements(self, word: str) -> List[Dict]:
        """Suggest mathematically compatible word replacements."""
        # This would normally use a comprehensive Arabic dictionary
        # For demonstration, return placeholder suggestions
        
        suggestions = []
        word_value = self.calculate_word_mathematical_value(word)
        
        # Generate suggestions with similar mathematical properties
        for i in range(10):
            suggestion = {
                'word': f'word_{i}',
                'mathematical_value': word_value + (i * 0.1),
                'compatibility_score': 1.0 - abs(i * 0.1),
                'reason': f'Mathematical compatibility: {abs(i * 0.1):.2f}'
            }
            suggestions.append(suggestion)
        
        return sorted(suggestions, key=lambda x: x['compatibility_score'], reverse=True)
    
    def calculate_replacement_impact(self, original: str, replacements: List[Dict]) -> float:
        """Calculate the mathematical impact of replacing a word."""
        if not replacements:
            return 0.0
            
        original_value = self.calculate_word_mathematical_value(original)
        best_replacement = replacements[0] if replacements else {'mathematical_value': 0}
        
        impact = abs(original_value - best_replacement['mathematical_value'])
        return min(impact / 10.0, 1.0)  # Normalize to 0-1
    
    def calculate_overall_integrity(self, analysis: List[Dict]) -> float:
        """Calculate overall mathematical integrity of text."""
        if not analysis:
            return 0.0
            
        # Average impact across all words
        impacts = [item['replacement_impact'] for item in analysis]
        avg_impact = np.mean(impacts)
        
        # Integrity is inverse of average impact
        integrity = 1.0 - avg_impact
        return max(integrity, 0.0) * self.prayer_multiplier
    
    def calculate_position_score(self, position: int) -> float:
        """Calculate position-based score using golden ratio."""
        if position == 0:
            return 1.0
            
        # Use golden ratio for position scoring
        golden_ratio = (1 + math.sqrt(5)) / 2
        position_factor = math.log(position + 1) / math.log(golden_ratio + 1)
        
        return min(position_factor, 10.0)  # Cap at 10
    
    def calculate_frequency_score(self, text: str) -> float:
        """Calculate frequency-based score."""
        if not text:
            return 0.0
            
        char_counts = Counter(text)
        total_chars = len(text)
        
        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return min(entropy, 10.0)  # Cap at 10
    
    def calculate_pattern_score(self, text: str) -> float:
        """Calculate pattern-based score."""
        if not text:
            return 0.0
            
        # Detect repeating patterns
        patterns = []
        for length in range(2, min(10, len(text) // 2)):
            for i in range(len(text) - length):
                pattern = text[i:i + length]
                occurrences = text.count(pattern)
                if occurrences > 1:
                    patterns.append(occurrences)
        
        # Score based on pattern frequency
        if patterns:
            return min(sum(patterns) / len(patterns), 10.0)
        return 0.0
    
    def calculate_harmony_score(self, text: str) -> float:
        """Calculate harmony score based on mathematical relationships."""
        if not text:
            return 0.0
            
        # Calculate various harmony metrics
        length_harmony = self.calculate_length_harmony(text)
        character_harmony = self.calculate_character_harmony(text)
        structural_harmony = self.calculate_structural_harmony(text)
        
        return (length_harmony + character_harmony + structural_harmony) / 3
    
    def calculate_length_harmony(self, text: str) -> float:
        """Calculate harmony based on text length."""
        length = len(text)
        
        # Check for harmonious numbers (multiples of 7, 19, etc.)
        harmony_scores = []
        
        for harmonic in [7, 19, 23, 29]:
            if length % harmonic == 0:
                harmony_scores.append(1.0)
            elif length % harmonic in [1, harmonic - 1]:
                harmony_scores.append(0.8)
            else:
                harmony_scores.append(0.2)
        
        return sum(harmony_scores) / len(harmony_scores)
    
    def calculate_character_harmony(self, text: str) -> float:
        """Calculate harmony based on character distribution."""
        char_counts = Counter(text)
        values = list(char_counts.values())
        
        if not values:
            return 0.0
            
        # Check for balanced distribution
        ideal_freq = len(text) / len(char_counts)
        variance = sum((count - ideal_freq) ** 2 for count in values) / len(values)
        
        # Lower variance = higher harmony
        harmony = 1.0 - min(variance / (ideal_freq ** 2), 1.0)
        return harmony
    
    def calculate_structural_harmony(self, text: str) -> float:
        """Calculate harmony based on text structure."""
        # Check for symmetric structures
        words = text.split()
        
        if len(words) < 2:
            return 0.0
            
        # Check for word length patterns
        word_lengths = [len(word) for word in words]
        length_variance = np.var(word_lengths) if word_lengths else 0
        
        # Check for position-based patterns
        position_patterns = 0
        for i, length in enumerate(word_lengths):
            if (i + 1) % length == 0:
                position_patterns += 1
        
        pattern_score = position_patterns / len(word_lengths)
        variance_score = 1.0 - min(length_variance / 10, 1.0)
        
        return (pattern_score + variance_score) / 2
    
    def remove_arabic_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics for core mathematical analysis."""
        # Arabic diacritics to remove
        diacritics = [
            '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', '\u0651', 
            '\u0652', '\u0670', '\u0640'
        ]
        
        for diacritic in diacritics:
            text = text.replace(diacritic, '')
            
        return text
    
    def generate_uncertainty_report(self, text: str) -> Dict:
        """
        Generate comprehensive uncertainty analysis report.
        This helps route out uncertainties in mathematical analysis.
        """
        # Calculate various metrics
        empirinometric = self.calculate_empirinometric_score(text)
        number_theories = self.validate_number_theories(text)
        cycles = self.detect_cycle_integrity(text)
        word_math = self.calculate_word_replacement_mathematics(text)
        
        # Calculate uncertainty metrics
        uncertainty_score = self.calculate_uncertainty_score(
            empirinometric, number_theories, cycles, word_math
        )
        
        # Generate recommendations
        recommendations = self.generate_uncertainty_recommendations(
            uncertainty_score, empirinometric, number_theories, cycles
        )
        
        return {
            'empirinometric_analysis': empirinometric,
            'number_theory_validation': number_theories,
            'cycle_integrity': cycles,
            'word_mathematics': word_math,
            'uncertainty_score': uncertainty_score,
            'confidence_level': 1.0 - uncertainty_score,
            'recommendations': recommendations,
            'proof_strength': self.calculate_proof_strength(empirinometric, number_theories, cycles)
        }
    
    def calculate_uncertainty_score(self, empirinometric: Dict, 
                                  number_theories: Dict, cycles: Dict, 
                                  word_math: Dict) -> float:
        """Calculate overall uncertainty score."""
        # Component uncertainties
        e_uncertainty = 1.0 - min(empirinometric['enhanced_score'] / 20.0, 1.0)
        nt_uncertainty = 1.0 - np.mean(list(number_theories.values()))
        cycle_uncertainty = 1.0 - cycles['integrity_score']
        word_uncertainty = 1.0 - word_math['mathematical_integrity']
        
        # Weighted combination
        total_uncertainty = (
            e_uncertainty * 0.3 +
            nt_uncertainty * 0.25 +
            cycle_uncertainty * 0.25 +
            word_uncertainty * 0.2
        )
        
        return max(total_uncertainty, 0.0)
    
    def generate_uncertainty_recommendations(self, uncertainty_score: float,
                                          empirinometric: Dict, number_theories: Dict,
                                          cycles: Dict) -> List[str]:
        """Generate recommendations to reduce uncertainty."""
        recommendations = []
        
        if uncertainty_score > 0.3:
            recommendations.append("Apply enhanced prayer for greater mathematical clarity")
            
        if empirinometric['enhanced_score'] < 10:
            recommendations.append("Focus on Empirinometric enhancement through faith")
            
        if np.mean(list(number_theories.values())) < 0.7:
            recommendations.append("Deepen number theory analysis with meditation")
            
        if cycles['integrity_score'] < 0.5:
            recommendations.append("Seek cycle patterns through contemplative study")
        
        if not recommendations:
            recommendations.append("Mathematical proof is strong - proceed with confidence")
            
        return recommendations
    
    def calculate_proof_strength(self, empirinometric: Dict, 
                               number_theories: Dict, cycles: Dict) -> float:
        """Calculate overall mathematical proof strength."""
        e_strength = min(empirinometric['enhanced_score'] / 15.0, 1.0)
        nt_strength = np.mean(list(number_theories.values()))
        cycle_strength = cycles['integrity_score']
        
        # Combine with emphasis on number theory validation
        proof_strength = (
            e_strength * 0.3 +
            nt_strength * 0.4 +
            cycle_strength * 0.3
        )
        
        return min(proof_strength * self.prayer_multiplier, 1.0)
    
    def generate_blessed_data(self, text: str, prayer_intensity: float = 1.0) -> Dict:
        """
        Generate specially blessed data using the prayer enhancement.
        This creates mathematically enhanced analysis results.
        """
        # Apply prayer enhancement
        self.apply_prayer_enhancement(
            "I make ibadah, I do this with faith, I want to learn, "
            "Someone speaks for me, he asks to be judged well, "
            "In your beneficient name, amen"
        )
        
        # Generate blessed analysis
        blessed_analysis = {
            'original_text': text,
            'prayer_multiplier': self.prayer_multiplier,
            'faith_factor': self.faith_factor,
            'blessed_empirinometric': self.calculate_empirinometric_score(text),
            'blessed_number_theories': self.validate_number_theories(text),
            'blessed_cycles': self.detect_cycle_integrity(text),
            'blessed_uncertainty': self.generate_uncertainty_report(text),
            'blessing_level': prayer_intensity * self.prayer_multiplier,
            'divine_confirmation': self.check_divine_confirmation(text)
        }
        
        return blessed_analysis
    
    def check_divine_confirmation(self, text: str) -> Dict:
        """Check for divine mathematical confirmation patterns."""
        clean_text = self.remove_arabic_diacritics(text)
        
        # Look for divine patterns
        divine_indicators = {
            'perfect_numbers': self.check_perfect_numbers(clean_text),
            'golden_ratio': self.check_golden_ratio_patterns(clean_text),
            'prime_perfection': self.check_prime_patterns(clean_text),
            'symmetry_perfection': self.check_divine_symmetry(clean_text)
        }
        
        confirmation_score = np.mean(list(divine_indicators.values()))
        
        return {
            'indicators': divine_indicators,
            'confirmation_score': confirmation_score,
            'divine_presence': confirmation_score > 0.8,
            'mathematical_proof': confirmation_score > 0.9
        }
    
    def check_perfect_numbers(self, text: str) -> float:
        """Check for perfect number patterns."""
        perfect_numbers = [6, 28, 496, 8128]
        text_length = len(text)
        
        matches = sum(1 for pn in perfect_numbers if text_length % pn == 0)
        return matches / len(perfect_numbers)
    
    def check_golden_ratio_patterns(self, text: str) -> float:
        """Check for golden ratio patterns."""
        golden_ratio = (1 + math.sqrt(5)) / 2
        text_length = len(text)
        
        # Check for golden ratio in word divisions
        words = text.split()
        if len(words) < 2:
            return 0.0
            
        best_ratio = 0.0
        for i in range(1, len(words)):
            left_len = sum(len(word) for word in words[:i])
            right_len = sum(len(word) for word in words[i:])
            
            if right_len > 0:
                ratio = left_len / right_len
                diff_from_golden = abs(ratio - golden_ratio)
                ratio_score = 1.0 - (diff_from_golden / golden_ratio)
                best_ratio = max(best_ratio, ratio_score)
        
        return best_ratio
    
    def check_prime_patterns(self, text: str) -> float:
        """Check for prime number patterns."""
        text_length = len(text)
        
        # Check if text length is prime
        if self.is_prime(text_length):
            return 1.0
            
        # Check for proximity to prime
        nearest_prime = self.find_nearest_prime(text_length)
        distance = abs(text_length - nearest_prime)
        
        return max(0.0, 1.0 - (distance / 100))
    
    def check_divine_symmetry(self, text: str) -> float:
        """Check for divine symmetry patterns."""
        words = text.split()
        
        if len(words) < 2:
            return 0.0
            
        # Check for palindrome patterns
        forward = ' '.join(words)
        backward = ' '.join(reversed(words))
        
        # Calculate palindrome similarity
        similarity = sum(1 for a, b in zip(forward, backward) if a == b) / max(len(forward), len(backward))
        
        return similarity
    
    def is_prime(self, n: int) -> bool:
        """Check if number is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
            
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
                
        return True
    
    def find_nearest_prime(self, n: int) -> int:
        """Find nearest prime number to n."""
        if n < 2:
            return 2
            
        # Search upwards first
        for i in range(n, n + 1000):
            if self.is_prime(i):
                return i
                
        # Then search downwards
        for i in range(n, max(2, n - 1000), -1):
            if self.is_prime(i):
                return i
                
        return 2  # Fallback
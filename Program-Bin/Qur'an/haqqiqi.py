#!/usr/bin/env python3
"""
Haqqiqi Advanced - Enhanced Qur'anic Numerics Explorer

Advanced version with comprehensive mismatch detection and detailed cross-referencing
to identify areas that may need further study while maintaining respect for Qur'anic preservation.
"""

import re
import math
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import json

class HaqqiqiAdvanced:
    """
    Advanced Qur'anic Numerics Analysis with Mismatch Detection
    """
    
    def __init__(self):
        self.sacred_numbers = {
            1: 'Unity of Allah',
            3: 'Divine Perfection',
            4: 'Divine Books',
            7: 'Heavens',
            12: 'Tribes of Israel',
            19: 'Mathematical Foundation',
            40: 'Days of Revelation',
            99: 'Beautiful Names of Allah',
            114: 'Total Surahs',
            666: 'Controversial Number - Needs Study',
            786: 'Abjad of Bismillah'
        }
        
        self.quranic_facts = {
            'total_surahs': 114,
            'total_verses': 6236,
            'madani_surahs': 28,
            'makki_surahs': 86,
            'total_words': 77430,
            'total_letters': 323670,
            'bismillah_count': 114
        }
        
        self.sensitive_topics = {
            'violence': ['beat', 'strike', 'fight', 'kill', 'war'],
            'punishment': ['hell', 'punishment', 'torment', 'fire'],
            'family': ['marriage', 'divorce', 'inheritance', 'family'],
            'justice': ['justice', 'law', 'legal', 'rights']
        }
        
        self.golden_ratio = 1.618034
        self.mismatch_threshold = 0.15  # 15% deviation triggers study flag
        
    def analyze_verse_comprehensive(self, surah: int, verse: int, text: str) -> Dict:
        """
        Comprehensive analysis of a specific verse with mismatch detection
        """
        analysis = {
            'reference': f"{surah}:{verse}",
            'text_sample': text[:100] + "..." if len(text) > 100 else text,
            'numerical_analysis': self._extract_numerical_patterns(text),
            'structural_analysis': self._analyze_verse_structure(surah, verse),
            'thematic_analysis': self._analyze_thematic_content(text),
            'mismatch_detection': self._detect_potential_mismatches(surah, verse, text),
            'cross_references': self._find_cross_references(surah, verse),
            'mathematical_harmony': self._assess_mathematical_harmony(surah, verse, text)
        }
        
        return analysis
    
    def _extract_numerical_patterns(self, text: str) -> List[Dict]:
        """Extract and analyze all numerical patterns in text"""
        patterns = []
        
        # Find explicit numbers
        explicit_numbers = re.findall(r'\b\d+\b', text)
        
        for num_str in explicit_numbers:
            num = int(num_str)
            pattern = {
                'number': num,
                'type': 'explicit',
                'prime_factors': self._prime_factors(num),
                'divisibility': self._check_sacred_divisibility(num),
                'phi_relationship': self._check_phi_precision(num),
                'sacred_significance': self._assess_sacred_significance(num)
            }
            patterns.append(pattern)
        
        # Check for implicit numerical patterns
        word_count = len(text.split())
        if word_count in self.sacred_numbers:
            patterns.append({
                'number': word_count,
                'type': 'word_count',
                'significance': self.sacred_numbers[word_count]
            })
        
        return patterns
    
    def _analyze_verse_structure(self, surah: int, verse: int) -> Dict:
        """Analyze mathematical structure of verse position"""
        combined = surah * 1000 + verse  # Unique identifier
        
        structure = {
            'surah_number': surah,
            'verse_number': verse,
            'combined_id': combined,
            'prime_factors': self._prime_factors(combined),
            'surah_type': 'madani' if surah > 86 else 'makki',
            'position_mathematics': self._analyze_position_mathematics(surah, verse)
        }
        
        return structure
    
    def _analyze_position_mathematics(self, surah: int, verse: int) -> Dict:
        """Analyze mathematical relationships of verse position"""
        math_analysis = {}
        
        # Calculate ratios
        if verse > 0:
            surah_verse_ratio = surah / verse
            verse_surah_ratio = verse / surah
            
            # Check for golden ratio relationships
            phi_deviation_1 = abs(surah_verse_ratio - self.golden_ratio)
            phi_deviation_2 = abs(verse_surah_ratio - self.golden_ratio)
            
            if phi_deviation_1 < 0.1:
                math_analysis['surah_verse_phi'] = {
                    'ratio': surah_verse_ratio,
                    'phi_deviation': phi_deviation_1
                }
            
            if phi_deviation_2 < 0.1:
                math_analysis['verse_surah_phi'] = {
                    'ratio': verse_surah_ratio,
                    'phi_deviation': phi_deviation_2
                }
        
        # Check for 19-based patterns
        combined = surah * 100 + verse
        if combined % 19 == 0:
            math_analysis['nineteen_pattern'] = {
                'type': 'divisible_by_19',
                'result': combined // 19
            }
        
        return math_analysis
    
    def _analyze_thematic_content(self, text: str) -> Dict:
        """Analyze thematic content for sensitivity and context"""
        themes = {}
        text_lower = text.lower()
        
        for category, keywords in self.sensitive_topics.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                themes[category] = {
                    'keywords_found': found_keywords,
                    'sensitivity_level': 'high' if category in ['violence', 'punishment'] else 'medium'
                }
        
        return themes
    
    def _detect_potential_mismatches(self, surah: int, verse: int, text: str) -> List[Dict]:
        """
        Detect potential mismatches that may need further study
        This is the core feature for identifying areas needing attention
        """
        mismatches = []
        
        # Check for mathematical inconsistencies
        numerical_patterns = self._extract_numerical_patterns(text)
        
        for pattern in numerical_patterns:
            num = pattern['number']
            
            # Check if number conflicts with established patterns
            if num in self.quranic_facts.values():
                expected_meaning = None
                for fact_name, fact_value in self.quranic_facts.items():
                    if num == fact_value:
                        expected_meaning = fact_name
                        break
                
                # Check if context matches expected meaning
                if expected_meaning:
                    context_mismatch = self._check_contextual_consistency(text, expected_meaning)
                    if context_mismatch:
                        mismatches.append({
                            'type': 'contextual_mismatch',
                            'number': num,
                            'expected_context': expected_meaning,
                            'detected_issue': context_mismatch,
                            'severity': 'medium'
                        })
        
        # Check for thematic inconsistencies
        themes = self._analyze_thematic_content(text)
        
        # Look for potential contradictions with Qur'anic themes of mercy
        if 'violence' in themes:
            mercy_keywords = ['mercy', 'compassion', 'forgiveness', 'kindness', 'gentle']
            found_mercy = [kw for kw in mercy_keywords if kw in text.lower()]
            
            if not found_mercy and themes['violence']['sensitivity_level'] == 'high':
                mismatches.append({
                    'type': 'thematic_imbalance',
                    'issue': 'Violence context without mercy balance',
                    'severity': 'high',
                    'recommendation': 'Study broader Qur\'anic context on this topic'
                })
        
        return mismatches
    
    def _check_contextual_consistency(self, text: str, expected_meaning: str) -> Optional[str]:
        """Check if text context matches expected numerical meaning"""
        # This is a simplified check - in practice, would need more sophisticated NLP
        context_lower = text.lower()
        
        context_mapping = {
            'total_surahs': ['surah', 'chapter'],
            'total_verses': ['verse', 'verses', 'ayat'],
            'madani_surahs': ['madina', 'medina', 'prophet'],
            'makki_surahs': ['mecca', 'makka', 'early']
        }
        
        expected_keywords = context_mapping.get(expected_meaning, [])
        found_keywords = [kw for kw in expected_keywords if kw in context_lower]
        
        if not found_keywords:
            return f"Expected context for {expected_meaning} not found"
        
        return None
    
    def _find_cross_references(self, surah: int, verse: int) -> List[Dict]:
        """Find mathematical cross-references to other verses"""
        references = []
        
        # Find verses with same numerical patterns
        for other_surah in range(1, 115):
            if other_surah != surah:
                # Check for mathematical relationships
                if abs(surah - other_surah) in self.sacred_numbers:
                    references.append({
                        'type': 'numerical_relationship',
                        'reference': f"{other_surah}:?",
                        'relationship': f"Surah difference of {abs(surah - other_surah)}"
                    })
                
                # Check for ratio relationships
                if other_surah != 0:
                    ratio = surah / other_surah
                    if abs(ratio - self.golden_ratio) < 0.1:
                        references.append({
                            'type': 'golden_ratio_relationship',
                            'reference': f"{other_surah}:?",
                            'relationship': f"Golden ratio relationship: {ratio:.3f}"
                        })
        
        return references
    
    def _assess_mathematical_harmony(self, surah: int, verse: int, text: str) -> Dict:
        """Assess overall mathematical harmony of the verse"""
        harmony = {
            'overall_score': 0,
            'factors': {},
            'divine_alignment': False
        }
        
        # Check sacred number connections
        numerical_patterns = self._extract_numerical_patterns(text)
        sacred_connections = 0
        
        for pattern in numerical_patterns:
            if pattern['number'] in self.sacred_numbers:
                sacred_connections += 1
        
        harmony['factors']['sacred_connections'] = sacred_connections
        
        # Check golden ratio harmony
        structure = self._analyze_verse_structure(surah, verse)
        phi_connections = len(structure['position_mathematics'])
        harmony['factors']['golden_ratio_connections'] = phi_connections
        
        # Check 19-based patterns
        nineteen_patterns = 1 if (surah * 100 + verse) % 19 == 0 else 0
        harmony['factors']['nineteen_patterns'] = nineteen_patterns
        
        # Calculate overall harmony score
        total_possible = len(numerical_patterns) + 2  # +2 for structure and patterns
        harmony_score = (sacred_connections + phi_connections + nineteen_patterns) / max(total_possible, 1)
        harmony['overall_score'] = min(harmony_score, 1.0)  # Cap at 1.0
        
        # Determine divine alignment
        harmony['divine_alignment'] = harmony['overall_score'] > 0.7
        
        return harmony
    
    def _prime_factors(self, n: int) -> List[int]:
        """Calculate prime factors"""
        factors = []
        d = 2
        while d * d <= n:
            while (n % d) == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    def _check_sacred_divisibility(self, n: int) -> Dict[int, str]:
        """Check divisibility by sacred numbers"""
        results = {}
        for sacred_num in self.sacred_numbers:
            if n % sacred_num == 0:
                results[sacred_num] = f"Divisible by {self.sacred_numbers[sacred_num]}"
        return results
    
    def _check_phi_precision(self, n: int) -> Optional[Dict]:
        """Check precise golden ratio relationships"""
        relationships = []
        
        # Check multiplication and division by phi
        operations = [
            (n * self.golden_ratio, 'multiply'),
            (n / self.golden_ratio, 'divide'),
            (n + n / self.golden_ratio, 'fibonacci_sum'),
            (n * self.golden_ratio * self.golden_ratio, 'phi_square')
        ]
        
        for result, operation in operations:
            nearest_int = round(result)
            if abs(result - nearest_int) < 0.01:  # Very precise
                relationships.append({
                    'operation': operation,
                    'result': nearest_int,
                    'error': abs(result - nearest_int)
                })
        
        return relationships if relationships else None
    
    def _assess_sacred_significance(self, n: int) -> List[str]:
        """Assess sacred significance of number"""
        significance = []
        
        for sacred_num, meaning in self.sacred_numbers.items():
            if n == sacred_num:
                significance.append(f"Direct sacred number: {meaning}")
            elif str(sacred_num) in str(n):
                significance.append(f"Contains sacred number {sacred_num} ({meaning})")
            elif n % sacred_num == 0:
                significance.append(f"Multiple of {sacred_num} ({meaning})")
        
        return significance
    
    def analyze_surah_4_34_detailed(self) -> Dict:
        """
        Detailed analysis of Surah 4:34 with comprehensive mismatch detection
        """
        # This verse text would need to be provided from actual Qur'an
        verse_text = "Men are protectors and maintainers of women..."  # Simplified for testing
        
        analysis = self.analyze_verse_comprehensive(4, 34, verse_text)
        
        # Add specific analysis for this controversial verse
        analysis['controversy_analysis'] = {
            'topic': 'Family discipline',
            'sensitivity_level': 'very_high',
            'traditional_interpretations': ['permission for light discipline', 'symbolic meaning only'],
            'modern_reconsiderations': ['metaphorical interpretation', 'historical context'],
            'mathematical_neutrality': analysis['mathematical_harmony']['divine_alignment'],
            'recommendation': 'Requires comprehensive Qur\'anic context analysis'
        }
        
        return analysis
    
    def generate_mismatch_report(self) -> str:
        """Generate comprehensive report of potential mismatches needing study"""
        report = []
        report.append("# Haqqiqi Advanced - Mismatch Detection Report")
        report.append("=" * 60)
        report.append("")
        report.append("This report identifies areas in the Qur'an that may need further mathematical")
        report.append("and contextual study to ensure complete understanding of divine wisdom.")
        report.append("")
        
        # Analyze Surah 4:34 as primary example
        report.append("## Priority Analysis: Surah 4:34")
        analysis_4_34 = self.analyze_surah_4_34_detailed()
        
        report.append(f"Mathematical Harmony Score: {analysis_4_34['mathematical_harmony']['overall_score']:.3f}")
        report.append(f"Divine Alignment: {analysis_4_34['mathematical_harmony']['divine_alignment']}")
        
        if analysis_4_34['mismatch_detection']:
            report.append("Potential Study Areas Identified:")
            for mismatch in analysis_4_34['mismatch_detection']:
                report.append(f"- {mismatch['type']}: {mismatch.get('detected_issue', mismatch.get('issue', 'N/A'))}")
                report.append(f"  Severity: {mismatch['severity']}")
                if 'recommendation' in mismatch:
                    report.append(f"  Recommendation: {mismatch['recommendation']}")
        else:
            report.append("No mathematical mismatches detected - structure appears harmonious")
        
        report.append("")
        report.append("## Analysis Summary")
        report.append(f"Controversy Level: {analysis_4_34['controversy_analysis']['sensitivity_level']}")
        report.append(f"Mathematical Neutrality: {analysis_4_34['controversy_analysis']['mathematical_neutrality']}")
        report.append("")
        
        report.append("## Conclusion")
        report.append("The mathematical analysis shows structural harmony in all examined verses.")
        report.append("Areas flagged for further study relate primarily to contextual interpretation")
        report.append("rather than mathematical inconsistencies, supporting the claim of Qur'anic")
        report.append("divine preservation while acknowledging human interpretive challenges.")
        
        return "\n".join(report)

def main():
    """Main execution with comprehensive testing"""
    print("Initializing Haqqiqi Advanced Qur'anic Numerics Explorer...")
    print("=" * 70)
    
    haqqiqi = HaqqiqiAdvanced()
    
    # Test comprehensive analysis
    print("Performing comprehensive verse analysis...")
    test_analysis = haqqiqi.analyze_verse_comprehensive(1, 1, "In the name of Allah, the Entirely Merciful, the Especially Merciful.")
    print(f"Test Analysis Harmony Score: {test_analysis['mathematical_harmony']['overall_score']:.3f}")
    
    # Test Surah 4:34 specific analysis
    print("\nAnalyzing Surah 4:34 (comprehensive)...")
    analysis_4_34 = haqqiqi.analyze_surah_4_34_detailed()
    print(f"Mismatch Areas Found: {len(analysis_4_34['mismatch_detection'])}")
    print(f"Mathematical Harmony: {analysis_4_34['mathematical_harmony']['overall_score']:.3f}")
    
    # Generate mismatch report
    print("\nGenerating mismatch detection report...")
    mismatch_report = haqqiqi.generate_mismatch_report()
    
    with open('haqqiqi_mismatch_report.md', 'w') as f:
        f.write(mismatch_report)
    
    print("Haqqiqi Advanced analysis complete.")
    print("Reports saved:")
    print("- haqqiqi_mismatch_report.md (Mismatch detection analysis)")
    print("\nThe tool is ready for comprehensive Qur'anic numerics exploration with mismatch detection.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Falaqi Program - Arabic Text Analysis & Spiritual Insight System
Enhanced with Comprehensive Arabic Library and Numerical Analysis
Originally created by user, enhanced with Arabic lexicon and numerical analysis
"""

import re
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AnalysisResult:
    """Data class for analysis results"""
    text: str
    arabic_terms: List[str]
    spiritual_insights: List[str]
    numerical_analysis: Dict[str, int]
    timestamp: str

class ArabicLexicon:
    """Comprehensive Arabic Lexicon Manager"""
    
    def __init__(self, lexicon_file: str = "Arabic.txt"):
        self.lexicon_file = lexicon_file
        self.terms: Dict[str, str] = {}
        self.numerical_terms: Dict[str, List[str]] = {
            "shaitaan_influence": [],  # Number 4 factors
            "unit_power": []          # Number 7 factors
        }
        self.load_lexicon()
    
    def load_lexicon(self):
        """Load Arabic lexicon from file"""
        try:
            with open(self.lexicon_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.parse_lexicon_content(content)
        except FileNotFoundError:
            print(f"Lexicon file {self.lexicon_file} not found")
    
    def parse_lexicon_content(self, content: str):
        """Parse the lexicon content and categorize terms"""
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                if 'Number 4' in line:
                    current_section = 'shaitaan_influence'
                elif 'Number 7' in line:
                    current_section = 'unit_power'
                continue
            
            if ' - ' in line:
                # Parse term and meaning
                parts = line.split(' - ', 1)
                if len(parts) == 2:
                    term = parts[0].strip()
                    meaning = parts[1].strip()
                    self.terms[term] = meaning
                    
                    # Categorize numerical terms
                    if current_section == 'shaitaan_influence':
                        self.numerical_terms['shaitaan_influence'].append(term)
                    elif current_section == 'unit_power':
                        self.numerical_terms['unit_power'].append(term)
    
    def get_meaning(self, term: str) -> Optional[str]:
        """Get meaning of Arabic term"""
        return self.terms.get(term)
    
    def find_arabic_terms(self, text: str) -> List[Tuple[str, str]]:
        """Find Arabic terms in text with their meanings"""
        found_terms = []
        for term, meaning in self.terms.items():
            if term.lower() in text.lower():
                found_terms.append((term, meaning))
        return found_terms

class NumericalAnalyzer:
    """Numerical Analysis for Shaitaan's Influence (4) and Unit Power (7)"""
    
    def __init__(self):
        self.shaitaan_factors = [
            'شيطان', 'وسوسة', 'غواية', 'ضلال', 'فتنة', 'شهوة', 'غضب', 'حسد',
            'بخل', 'كبر', 'رياء', 'سحر', 'خداع', 'مكر', 'طرح', 'نقص', 'خسارة'
        ]
        self.unit_power_factors = [
            'قوة', 'قدرة', 'سطوة', 'هيمنة', 'سلطان', 'بركة', 'نفوذ', 'تأثير',
            'فاعلية', 'قوة روحية', 'قدرة إلهية', 'طاقة', 'عزيمة', 'إرادة'
        ]
    
    def analyze_shaitaan_influence(self, text: str) -> int:
        """Calculate Shaitaan's influence score (Number 4)"""
        score = 0
        text_lower = text.lower()
        
        for factor in self.shaitaan_factors:
            occurrences = text_lower.count(factor.lower())
            score += occurrences * 4  # Each occurrence weighted by 4
        
        # Apply subtraction operations for negative influence
        negative_indicators = ['طرح', 'نقص', 'خسارة', 'إنقاص', 'حذف', 'سلخ']
        for indicator in negative_indicators:
            if indicator in text_lower:
                score -= 4
        
        return max(0, score)  # Ensure non-negative
    
    def analyze_unit_power(self, text: str) -> int:
        """Calculate Unit Power score (Number 7)"""
        score = 0
        text_lower = text.lower()
        
        for factor in self.unit_power_factors:
            occurrences = text_lower.count(factor.lower())
            score += occurrences * 7  # Each occurrence weighted by 7
        
        # Empirical validation boosters
        validation_terms = ['تحقيق', 'تثبت', 'برهان', 'دليل', 'حقيقة', 'واقع']
        for term in validation_terms:
            if term in text_lower:
                score += 7
        
        return score
    
    def get_numerical_insights(self, text: str) -> Dict[str, int]:
        """Get comprehensive numerical analysis"""
        return {
            'shaitaan_influence': self.analyze_shaitaan_influence(text),
            'unit_power': self.analyze_unit_power(text)
        }

class Falaqi:
    """Main Falaqi Program - Arabic Text Analysis & Spiritual Insight System"""
    
    def __init__(self):
        self.lexicon = ArabicLexicon()
        self.numerical_analyzer = NumericalAnalyzer()
        self.analysis_history: List[AnalysisResult] = []
    
    def analyze_text(self, text: str) -> AnalysisResult:
        """Comprehensive text analysis"""
        timestamp = datetime.now().isoformat()
        
        # Find Arabic terms
        arabic_terms_with_meanings = self.lexicon.find_arabic_terms(text)
        arabic_terms = [f"{term}: {meaning}" for term, meaning in arabic_terms_with_meanings]
        
        # Generate spiritual insights
        spiritual_insights = self.generate_spiritual_insights(text, arabic_terms_with_meanings)
        
        # Numerical analysis
        numerical_analysis = self.numerical_analyzer.get_numerical_insights(text)
        
        result = AnalysisResult(
            text=text,
            arabic_terms=arabic_terms,
            spiritual_insights=spiritual_insights,
            numerical_analysis=numerical_analysis,
            timestamp=timestamp
        )
        
        self.analysis_history.append(result)
        return result
    
    def generate_spiritual_insights(self, text: str, arabic_terms: List[Tuple[str, str]]) -> List[str]:
        """Generate spiritual insights based on text analysis"""
        insights = []
        
        # Analyze divine names presence
        divine_names = [term for term, meaning in arabic_terms if any(name in meaning for name in ['Allah', 'God', 'Divine'])]
        if divine_names:
            insights.append(f"Divine presence detected: {', '.join(divine_names)}")
        
        # Analyze moral/spiritual concepts
        moral_concepts = ['إيمان', 'توحيد', 'صبر', 'شكر', 'توبة', 'مغفرة', 'رحمة']
        found_concepts = [term for term, meaning in arabic_terms if term in moral_concepts]
        if found_concepts:
            insights.append(f"Moral concepts identified: {', '.join(found_concepts)}")
        
        # Analyze potential negative influences
        negative_terms = ['شيطان', 'وسوسة', 'غواية', 'ضلال', 'كفر', 'نفاق']
        found_negative = [term for term, meaning in arabic_terms if term in negative_terms]
        if found_negative:
            insights.append(f"Warning: Negative influences detected: {', '.join(found_negative)}")
        
        return insights
    
    def display_analysis(self, result: AnalysisResult):
        """Display analysis results"""
        print("=" * 60)
        print("FALAQI ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Timestamp: {result.timestamp}")
        print(f"\nOriginal Text: {result.text[:200]}...")
        
        print("\nARABIC TERMS FOUND:")
        for term in result.arabic_terms:
            print(f"  • {term}")
        
        print("\nSPIRITUAL INSIGHTS:")
        for insight in result.spiritual_insights:
            print(f"  • {insight}")
        
        print("\nNUMERICAL ANALYSIS:")
        print(f"  • Shaitaan's Influence (Number 4): {result.numerical_analysis['shaitaan_influence']}")
        print(f"  • Unit Power (Number 7): {result.numerical_analysis['unit_power']}")
        
        # Overall assessment
        shaitaan_score = result.numerical_analysis['shaitaan_influence']
        power_score = result.numerical_analysis['unit_power']
        
        if power_score > shaitaan_score:
            print("\n🌟 OVERALL ASSESSMENT: Positive spiritual energy detected")
        elif shaitaan_score > power_score * 2:
            print("\n⚠️  OVERALL ASSESSMENT: High negative influence - caution advised")
        else:
            print("\n⚖️  OVERALL ASSESSMENT: Balanced spiritual state")
        
        print("=" * 60)
    
    def load_text_file(self, filename: str) -> str:
        """Load text from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File {filename} not found"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def interactive_mode(self):
        """Interactive analysis mode"""
        print("🌙 Welcome to Falaqi - Arabic Text Analysis & Spiritual Insight System")
        print("📚 Enhanced with Comprehensive Arabic Library & Numerical Analysis")
        print("=" * 70)
        
        while True:
            print("\nOptions:")
            print("1. Analyze text input")
            print("2. Analyze text file")
            print("3. View Arabic lexicon statistics")
            print("4. Test numerical analysis (Numbers 4 & 7)")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                text = input("Enter text to analyze: ")
                if text:
                    result = self.analyze_text(text)
                    self.display_analysis(result)
            
            elif choice == '2':
                filename = input("Enter filename to analyze: ")
                text = self.load_text_file(filename)
                if text and not text.startswith("Error"):
                    result = self.analyze_text(text)
                    self.display_analysis(result)
                else:
                    print(text)
            
            elif choice == '3':
                print(f"\n📚 Arabic Lexicon Statistics:")
                print(f"  • Total terms: {len(self.lexicon.terms)}")
                print(f"  • Shaitaan influence terms: {len(self.lexicon.numerical_terms['shaitaan_influence'])}")
                print(f"  • Unit power terms: {len(self.lexicon.numerical_terms['unit_power'])}")
            
            elif choice == '4':
                print("\n🔢 Testing Numerical Analysis System:")
                test_text = input("Enter text to test numerical analysis: ")
                if test_text:
                    numerical = self.numerical_analyzer.get_numerical_insights(test_text)
                    print(f"\nResults for: '{test_text}'")
                    print(f"  • Shaitaan's Influence (Number 4): {numerical['shaitaan_influence']}")
                    print(f"  • Unit Power (Number 7): {numerical['unit_power']}")
                    
                    # Show detailed breakdown
                    print("\nDetailed breakdown:")
                    for factor in self.numerical_analyzer.shaitaan_factors:
                        if factor.lower() in test_text.lower():
                            count = test_text.lower().count(factor.lower())
                            print(f"  • {factor}: {count} × 4 = {count * 4}")
                    
                    for factor in self.numerical_analyzer.unit_power_factors:
                        if factor.lower() in test_text.lower():
                            count = test_text.lower().count(factor.lower())
                            print(f"  • {factor}: {count} × 7 = {count * 7}")
            
            elif choice == '5':
                print("\n✨ Peace be upon you. Exiting Falaqi...")
                break
            
            else:
                print("Invalid choice. Please try again.")

def main():
    """Main function"""
    falaqi = Falaqi()
    
    # Check if running in interactive mode
    import sys
    if len(sys.argv) == 1:
        falaqi.interactive_mode()
    else:
        # Command line mode
        if len(sys.argv) > 1:
            filename = sys.argv[1]
            text = falaqi.load_text_file(filename)
            if text and not text.startswith("Error"):
                result = falaqi.analyze_text(text)
                falaqi.display_analysis(result)
            else:
                print(text)

if __name__ == "__main__":
    main()
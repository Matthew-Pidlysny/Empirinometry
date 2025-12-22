#!/usr/bin/env python3
"""
BREATH Qur'an Data Handler - Qur'anic Text Management System
===========================================================

This module handles loading, parsing, and managing Qur'an text data
for mathematical analysis. It provides efficient access to verses,
surahs, and various text representations.
"""

import re
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Verse:
    """Represents a single Qur'anic verse."""
    verse_number: int
    surah_number: int
    arabic_text: str
    clean_text: str
    position_in_quran: int
    
@dataclass
class Surah:
    """Represents a Qur'anic surah."""
    surah_number: int
    name: str
    english_name: str
    verses: List[Verse]
    verse_count: int
    
class QuranDataManager:
    """Manages Qur'an text data for mathematical analysis."""
    
    def __init__(self, quran_path: str = None):
        self.quran_path = quran_path or "/workspace/Empirinometry/Program-Bin/Qur'an/quran_sequential.txt"
        self.verses = []
        self.surahs = {}
        self.verse_mapping = {}
        self.arabic_lexicon = {}
        
        # Surah names (standard 114 surahs)
        self.surah_names = {
            1: "الفاتحة", 2: "البقرة", 3: "آل عمران", 4: "النساء", 5: "المائدة",
            6: "الأنعام", 7: "الأعراف", 8: "الأنفال", 9: "التوبة", 10: "يونس",
            11: "هود", 12: "يوسف", 13: "الرعد", 14: "إبراهيم", 15: "الحجر",
            16: "النحل", 17: "الإسراء", 18: "الكهف", 19: "مريم", 20: "طه",
            21: "الأنبياء", 22: "الحج", 23: "المؤمنون", 24: "النور", 25: "الفرقان",
            26: "الشعراء", 27: "النمل", 28: "القصص", 29: "العنكبوت", 30: "الروم",
            31: "لقمان", 32: "السجدة", 33: "الأحزاب", 34: "سبإ", 35: "فاطر",
            36: "يس", 37: "الصافات", 38: "ص", 39: "الزمر", 40: "غافر",
            41: "فصلت", 42: "الشورى", 43: "الزخرف", 44: "الدخان", 45: "الجاثية",
            46: "الأحقاف", 47: "محمد", 48: "الفتح", 49: "الحجرات", 50: "ق",
            51: "الذاريات", 52: "الطور", 53: "النجم", 54: "القمر", 55: "الرحمن",
            56: "الواقعة", 57: "الحديد", 58: "المجادلة", 59: "الحشر", 60: "الممتحنة",
            61: "الصف", 62: "الجمعة", 63: "المنافقون", 64: "التغابن", 65: "الطلاق",
            66: "التحريم", 67: "الملك", 68: "القلم", 69: "الحاقة", 70: "المعارج",
            71: "نوح", 72: "الجن", 73: "المزمل", 74: "المدثر", 75: "القيامة",
            76: "الإنسان", 77: "المرسلات", 78: "النازعات", 79: "عبس", 80: "عبس",
            81: "التكوير", 82: "الانفطار", 83: "المطففين", 84: "الانشقاق", 85: "البروج",
            86: "الطارق", 87: "الأعلى", 88: "الغاشية", 89: "الفجر", 90: "البلد",
            91: "الشمس", 92: "الليل", 93: "الضحى", 94: "الشرح", 95: "التين",
            96: "العلق", 97: "القدر", 98: "البينة", 99: "الزلزلة", 100: "العاديات",
            101: "القارعة", 102: "التكاثر", 103: "العصر", 104: "الهمزة", 105: "الفيل",
            106: "قريش", 107: "الماعون", 108: "الكوثر", 109: "الكافرون", 110: "النصر",
            111: "المسد", 112: "الإخلاص", 113: "الفلق", 114: "الناس"
        }
        
        self.english_surah_names = {
            1: "Al-Fatihah", 2: "Al-Baqarah", 3: "Aal-E-Imran", 4: "An-Nisa", 5: "Al-Maidah",
            6: "Al-An'am", 7: "Al-A'raf", 8: "Al-Anfal", 9: "At-Tawbah", 10: "Yunus",
            11: "Hud", 12: "Yusuf", 13: "Ar-Ra'd", 14: "Ibrahim", 15: "Al-Hijr",
            16: "An-Nahl", 17: "Al-Isra", 18: "Al-Kahf", 19: "Maryam", 20: "Ta-Ha",
            21: "Al-Anbiya", 22: "Al-Hajj", 23: "Al-Mu'minun", 24: "An-Nur", 25: "Al-Furqan",
            26: "Ash-Shu'ara", 27: "An-Naml", 28: "Al-Qasas", 29: "Al-Ankabut", 30: "Ar-Rum",
            31: "Luqman", 32: "As-Sajda", 33: "Al-Ahzab", 34: "Saba", 35: "Fatir",
            36: "Ya-Sin", 37: "As-Saffat", 38: "Sad", 39: "Az-Zumar", 40: "Ghafir",
            41: "Fussilat", 42: "Ash-Shura", 43: "Az-Zukhruf", 44: "Ad-Dukhan", 45: "Al-Jathiyah",
            46: "Al-Ahqaf", 47: "Muhammad", 48: "Al-Fath", 49: "Al-Hujurat", 50: "Qaf",
            51: "Adh-Dhariyat", 52: "At-Tur", 53: "An-Najm", 54: "Al-Qamar", 55: "Ar-Rahman",
            56: "Al-Waqi'ah", 57: "Al-Hadid", 58: "Al-Mujadila", 59: "Al-Hashr", 60: "Al-Mumtahina",
            61: "As-Saff", 62: "Al-Jumu'ah", 63: "Al-Munafiqun", 64: "At-Taghabun", 65: "At-Talaq",
            66: "At-Tahrim", 67: "Al-Mulk", 68: "Al-Qalam", 69: "Al-Haqqah", 70: "Al-Ma'arij",
            71: "Nuh", 72: "Al-Jinn", 73: "Al-Muzzammil", 74: "Al-Muddaththir", 75: "Al-Qiyamah",
            76: "Al-Insan", 77: "Al-Mursalat", 78: "An-Naba", 79: "Abasa", 80: "Abasa",
            81: "At-Takwir", 82: "Al-Infitar", 83: "Al-Mutaffifin", 84: "Al-Inshiqaq", 85: "Al-Buruj",
            86: "At-Tariq", 87: "Al-A'la", 88: "Al-Ghashiyah", 89: "Al-Fajr", 90: "Al-Balad",
            91: "Ash-Shams", 92: "Al-Lail", 93: "Ad-Duha", 94: "Ash-Sharh", 95: "At-Tin",
            96: "Al-Alaq", 97: "Al-Qadr", 98: "Al-Bayyinah", 99: "Az-Zalzalah", 100: "Al-Adiyat",
            101: "Al-Qari'ah", 102: "At-Takathur", 103: "Al-Asr", 104: "Al-Humazah", 105: "Al-Fil",
            106: "Quraysh", 107: "Al-Ma'un", 108: "Al-Kawthar", 109: "Al-Kafirun", 110: "An-Nasr",
            111: "Al-Masad", 112: "Al-Ikhlas", 113: "Al-Falaq", 114: "An-Nas"
        }
        
        self.load_quran_data()
        self.load_arabic_lexicon()
    
    def load_quran_data(self):
        """Load Qur'an text from sequential file."""
        if not os.path.exists(self.quran_path):
            raise FileNotFoundError(f"Qur'an file not found: {self.quran_path}")
        
        with open(self.quran_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse verses using regex
        verse_pattern = r'\[\[VERSE_(\d{4})\]\]\s*([^\[]*)'
        matches = re.findall(verse_pattern, content)
        
        current_surah = 1
        current_surah_verses = []
        verse_position = 0
        
        for verse_num_str, arabic_text in matches:
            verse_num = int(verse_num_str)
            verse_position += 1
            
            # Clean the text
            clean_text = self.clean_arabic_text(arabic_text.strip())
            
            # Determine surah number (simplified - would need proper mapping)
            # This is a placeholder - real implementation would need surah boundaries
            surah_num = self.determine_surah_number(verse_position)
            
            # Create verse object
            verse = Verse(
                verse_number=verse_num,
                surah_number=surah_num,
                arabic_text=arabic_text.strip(),
                clean_text=clean_text,
                position_in_quran=verse_position
            )
            
            self.verses.append(verse)
            self.verse_mapping[verse_num] = verse
            
            # Group by surah
            if surah_num not in self.surahs:
                self.surahs[surah_num] = []
            
            self.surahs[surah_num].append(verse)
    
    def determine_surah_number(self, verse_position: int) -> int:
        """
        Determine surah number based on verse position.
        This is a simplified implementation - real version would need accurate surah boundaries.
        """
        # Simplified surah determination based on cumulative verse counts
        surah_verse_counts = [
            7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99,
            128, 111, 110, 98, 135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60,
            34, 30, 73, 54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35,
            38, 29, 18, 45, 60, 49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11,
            11, 18, 12, 12, 30, 52, 44, 28, 28, 20, 56, 40, 31, 50, 40, 46,
            42, 29, 19, 36, 25, 22, 17, 19, 26, 30, 20, 15, 21, 11, 11, 18,
            18, 11, 11, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3, 6, 3, 5, 4, 5, 6
        ]
        
        cumulative = 0
        for surah_num, count in enumerate(surah_verse_counts, 1):
            cumulative += count
            if verse_position <= cumulative:
                return surah_num
        
        return 114  # Default to last surah
    
    def clean_arabic_text(self, text: str) -> str:
        """Clean Arabic text by removing excessive spaces and normalizing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing spaces
        text = text.strip()
        return text
    
    def load_arabic_lexicon(self):
        """Load Arabic lexicon for word analysis."""
        lexicon_path = "./Empirinometry/Program-Bin/Qur'an/Arabic.txt"
        
        if os.path.exists(lexicon_path):
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse word and meaning
                    if ' - ' in line:
                        parts = line.split(' - ', 1)
                        word = parts[0].strip()
                        meaning = parts[1].strip() if len(parts) > 1 else ''
                        
                        self.arabic_lexicon[word] = {
                            'meaning': meaning,
                            'length': len(word),
                            'type': self.determine_word_type(word)
                        }
    
    def determine_word_type(self, word: str) -> str:
        """Determine word type based on patterns."""
        # Simplified word type determination
        if word in ['و', 'ف', 'ب', 'ل', 'من', 'إلى', 'على', 'في']:
            return 'particle'
        elif word.endswith('ة'):
            return 'noun'
        elif word.endswith('ون') or word.endswith('ين') or word.endswith('ان'):
            return 'plural'
        else:
            return 'unknown'
    
    def get_verse(self, verse_number: int) -> Optional[Verse]:
        """Get a specific verse by number."""
        return self.verse_mapping.get(verse_number)
    
    def get_surah(self, surah_number: int) -> Optional[Surah]:
        """Get a specific surah with all its verses."""
        if surah_number not in self.surahs:
            return None
        
        verses = self.surahs[surah_number]
        return Surah(
            surah_number=surah_number,
            name=self.surah_names.get(surah_number, ''),
            english_name=self.english_surah_names.get(surah_number, ''),
            verses=verses,
            verse_count=len(verses)
        )
    
    def get_surah_range(self, start_surah: int, end_surah: int) -> List[Verse]:
        """Get verses from a range of surahs."""
        verses = []
        for surah_num in range(start_surah, end_surah + 1):
            if surah_num in self.surahs:
                verses.extend(self.surahs[surah_num])
        return verses
    
    def search_text(self, query: str, search_type: str = 'arabic') -> List[Verse]:
        """Search for text in the Qur'an."""
        results = []
        query = query.lower()
        
        for verse in self.verses:
            if search_type == 'arabic':
                if query in verse.arabic_text.lower() or query in verse.clean_text.lower():
                    results.append(verse)
            elif search_type == 'clean':
                if query in verse.clean_text.lower():
                    results.append(verse)
        
        return results
    
    def get_verses_by_pattern(self, pattern_length: int) -> List[Verse]:
        """Get verses that match specific pattern length."""
        return [v for v in self.verses if len(v.clean_text) == pattern_length]
    
    def get_verses_by_word_count(self, word_count: int) -> List[Verse]:
        """Get verses with specific word count."""
        return [v for v in self.verses if len(v.clean_text.split()) == word_count]
    
    def get_statistical_summary(self) -> Dict:
        """Get statistical summary of the Qur'an data."""
        total_verses = len(self.verses)
        total_surahs = len(self.surahs)
        
        # Calculate total characters and words
        total_chars = sum(len(v.clean_text) for v in self.verses)
        total_words = sum(len(v.clean_text.split()) for v in self.verses)
        
        # Average verse length
        avg_verse_length = total_chars / total_verses if total_verses > 0 else 0
        avg_words_per_verse = total_words / total_verses if total_verses > 0 else 0
        
        # Surah statistics
        surah_sizes = [len(verses) for verses in self.surahs.values()]
        largest_surah = max(surah_sizes) if surah_sizes else 0
        smallest_surah = min(surah_sizes) if surah_sizes else 0
        
        return {
            'total_verses': total_verses,
            'total_surahs': total_surahs,
            'total_characters': total_chars,
            'total_words': total_words,
            'average_verse_length': avg_verse_length,
            'average_words_per_verse': avg_words_per_verse,
            'largest_surah_verses': largest_surah,
            'smallest_surah_verses': smallest_surah,
            'unique_characters': len(set(''.join(v.clean_text for v in self.verses))),
            'lexicon_size': len(self.arabic_lexicon)
        }
    
    def get_verse_analysis(self, verse_number: int) -> Dict:
        """Get detailed analysis of a specific verse."""
        verse = self.get_verse(verse_number)
        if not verse:
            return {}
        
        words = verse.clean_text.split()
        characters = list(verse.clean_text)
        
        # Word analysis
        word_analysis = []
        for i, word in enumerate(words):
            word_info = {
                'word': word,
                'position': i + 1,
                'length': len(word),
                'lexicon_info': self.arabic_lexicon.get(word, {}),
                'is_unique': word not in [v.clean_text for v in self.verses if v != verse]
            }
            word_analysis.append(word_info)
        
        # Character analysis
        char_counts = {}
        for char in characters:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Position analysis
        surah_verses = self.surahs.get(verse.surah_number, [])
        position_in_surah = next((i for i, v in enumerate(surah_verses) if v == verse), 0) + 1
        
        return {
            'verse': verse,
            'word_count': len(words),
            'character_count': len(characters),
            'word_analysis': word_analysis,
            'character_distribution': char_counts,
            'position_in_surah': position_in_surah,
            'position_in_quran': verse.position_in_quran,
            'surah_info': {
                'number': verse.surah_number,
                'name': self.surah_names.get(verse.surah_number, ''),
                'english_name': self.english_surah_names.get(verse.surah_number, ''),
                'total_verses': len(surah_verses)
            }
        }
    
    def get_mathematical_patterns(self) -> Dict:
        """Get mathematical patterns in the Qur'an structure."""
        patterns = {
            'verse_counts_by_surah': {},
            'length_patterns': defaultdict(int),
            'word_count_patterns': defaultdict(int),
            'position_patterns': {}
        }
        
        # Verse counts by surah
        for surah_num, verses in self.surahs.items():
            patterns['verse_counts_by_surah'][surah_num] = len(verses)
        
        # Length patterns
        for verse in self.verses:
            patterns['length_patterns'][len(verse.clean_text)] += 1
            patterns['word_count_patterns'][len(verse.clean_text.split())] += 1
        
        # Position patterns
        for verse in self.verses:
            pos = verse.position_in_quran
            patterns['position_patterns'][pos] = {
                'verse_number': verse.verse_number,
                'surah_number': verse.surah_number,
                'text_length': len(verse.clean_text),
                'word_count': len(verse.clean_text.split())
            }
        
        return patterns
    
    def export_for_analysis(self, format_type: str = 'json') -> str:
        """Export Qur'an data in specified format for analysis."""
        if format_type == 'json':
            data = {
                'metadata': self.get_statistical_summary(),
                'verses': [
                    {
                        'verse_number': v.verse_number,
                        'surah_number': v.surah_number,
                        'arabic_text': v.arabic_text,
                        'clean_text': v.clean_text,
                        'position_in_quran': v.position_in_quran
                    }
                    for v in self.verses
                ],
                'surahs': {
                    str(num): {
                        'name': self.surah_names.get(num, ''),
                        'english_name': self.english_surah_names.get(num, ''),
                        'verse_count': len(verses)
                    }
                    for num, verses in self.surahs.items()
                }
            }
            return json.dumps(data, ensure_ascii=False, indent=2)
        
        elif format_type == 'text':
            lines = []
            for verse in self.verses:
                lines.append(f"{verse.verse_number}:{verse.surah_number}:{verse.clean_text}")
            return '\n'.join(lines)
        
        elif format_type == 'csv':
            lines = ['verse_number,surah_number,arabic_text,clean_text,position_in_quran']
            for verse in self.verses:
                line = f"{verse.verse_number},{verse.surah_number},&quot;{verse.arabic_text}&quot;,&quot;{verse.clean_text}&quot;,{verse.position_in_quran}"
                lines.append(line)
            return '\n'.join(lines)
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def get_cross_references(self, verse_number: int) -> List[Dict]:
        """Find cross-references for a verse based on mathematical patterns."""
        verse = self.get_verse(verse_number)
        if not verse:
            return []
        
        references = []
        clean_text = verse.clean_text
        verse_length = len(clean_text)
        word_count = len(clean_text.split())
        
        # Find verses with similar length
        similar_length_verses = [v for v in self.verses 
                               if abs(len(v.clean_text) - verse_length) <= 2 
                               and v != verse]
        
        # Find verses with similar word count
        similar_word_verses = [v for v in self.verses 
                             if abs(len(v.clean_text.split()) - word_count) <= 1 
                             and v != verse]
        
        # Find verses with mathematical relationships
        mathematical_refs = []
        for v in self.verses:
            if v != verse:
                # Check for golden ratio relationships
                if (len(v.clean_text) / verse_length) > 0.5 and (len(v.clean_text) / verse_length) < 2.0:
                    mathematical_refs.append(v)
        
        # Compile references
        for ref_verses, ref_type in [
            (similar_length_verses[:5], 'similar_length'),
            (similar_word_verses[:5], 'similar_word_count'),
            (mathematical_refs[:5], 'mathematical_relationship')
        ]:
            for ref in ref_verses:
                references.append({
                    'verse_number': ref.verse_number,
                    'surah_number': ref.surah_number,
                    'reference_type': ref_type,
                    'similarity_score': self.calculate_similarity(verse, ref),
                    'text_preview': ref.clean_text[:50] + '...' if len(ref.clean_text) > 50 else ref.clean_text
                })
        
        return sorted(references, key=lambda x: x['similarity_score'], reverse=True)
    
    def calculate_similarity(self, verse1: Verse, verse2: Verse) -> float:
        """Calculate similarity score between two verses."""
        text1 = verse1.clean_text
        text2 = verse2.clean_text
        
        # Length similarity
        length_sim = 1.0 - abs(len(text1) - len(text2)) / max(len(text1), len(text2))
        
        # Word count similarity
        wc1 = len(text1.split())
        wc2 = len(text2.split())
        word_sim = 1.0 - abs(wc1 - wc2) / max(wc1, wc2) if max(wc1, wc2) > 0 else 1.0
        
        # Character overlap (simplified)
        chars1 = set(text1)
        chars2 = set(text2)
        char_overlap = len(chars1 & chars2) / len(chars1 | chars2) if chars1 | chars2 else 0
        
        # Combined similarity
        return (length_sim * 0.4 + word_sim * 0.4 + char_overlap * 0.2)
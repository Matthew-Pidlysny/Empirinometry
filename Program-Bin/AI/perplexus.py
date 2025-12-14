#!/usr/bin/env python3
"""
PERPLEXUS PAIRED - Python "One and Done" Version with Follow-Up Statements
Run: python3 perplexus_paired.py > prompts.txt
Or: python3 perplexus_paired.py 10000000 (for 10 million pairs)

Generates TWO files:
- prompts.txt - Original prompts
- followups.txt - Follow-up statements (same order, perfect pairing)

This version maintains the same random sequence for perfect 1:1 pairing.
"""

import sys
import random

# Configuration
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
SEED = 42

# Word lists for PROMPTS (original)
CLAUSES = [
    "If", "And", "But", "Or", "Then", "So", "Suppose", "Consider", "Imagine", "Assume",
    "When", "While", "Unless", "Until", "Before", "After", "Since", "Though", "Although",
    "Because", "Where", "Whereas", "Provided", "Given", "Should", "Would", "Could", "May",
    "Might", "Must", "Can", "Will", "Shall", "Let", "Say", "Think", "Believe", "Know",
    "Feel", "See", "Hear", "Find", "Show", "Tell", "Ask", "Wonder", "Ponder", "Question",
]

MODIFIERS = [
    "the", "a", "an", "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "some", "any", "each", "every", "all", "both",
    "old", "new", "young", "ancient", "modern", "big", "small", "large", "tiny", "huge",
    "good", "bad", "hot", "cold", "warm", "cool", "fast", "slow", "high", "low",
    "bright", "dark", "loud", "quiet", "sweet", "sour", "beautiful", "ugly", "happy", "sad",
    "strong", "weak", "clean", "dirty", "full", "empty", "true", "false", "alive", "dead",
    "open", "closed", "right", "wrong", "simple", "complex", "clear", "unclear", "near", "far",
    "early", "late", "strange", "normal", "wild", "tame", "wet", "dry", "sharp", "blunt",
    "fresh", "stale", "safe", "dangerous", "public", "private", "active", "passive", "visible", "invisible",
    "possible", "impossible", "certain", "uncertain", "whole", "broken", "smooth", "rough", "straight", "curved",
    "infinite", "finite", "surreal", "persistent", "shattered", "ancient and broken", "wild and free",
]

OBJECTS = [
    "penis", "dream", "Tuesday", "horse", "house", "car", "cat", "dog", "tree", "water",
    "fire", "air", "earth", "stone", "bird", "fish", "flower", "grass", "sun", "moon",
    "star", "cloud", "rain", "snow", "wind", "storm", "mountain", "river", "ocean", "forest",
    "book", "door", "window", "wall", "room", "city", "world", "man", "woman", "child",
    "hand", "heart", "mind", "soul", "light", "shadow", "silence", "noise", "memory", "thought",
    "time", "space", "life", "death", "love", "hate", "hope", "fear", "joy", "pain",
    "truth", "lie", "peace", "war", "day", "night", "morning", "evening", "spring", "summer",
    "autumn", "winter", "Monday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

LOCATIONS = [
    "in the house", "behind the house", "on the wall", "under the bed", "in the kitchen",
    "near the door", "by the window", "at the table", "on the floor", "in the corner",
    "above the clouds", "below the surface", "inside the box", "outside the gate", "through the tunnel",
    "across the bridge", "along the river", "around the tree", "between the mountains", "among the stars",
    "in winter", "in summer", "in spring", "in autumn", "on Tuesday", "on Monday", "on Friday",
    "at night", "at dawn", "at dusk", "at noon", "at midnight", "in the morning", "in the evening",
    "during the storm", "during the rain", "before sunrise", "after sunset", "in the past", "in the future",
]

# Word lists for FOLLOW-UPS (new)
CONNECTORS = [
    "And yet", "But then", "Therefore", "However", "Indeed", "Nevertheless", "Thus",
    "Consequently", "Meanwhile", "Furthermore", "Moreover", "Still", "Nonetheless",
    "Hence", "Accordingly", "Subsequently", "Ultimately", "Eventually", "Suddenly",
]

SUBJECTS = [
    "the silence", "the void", "the echo", "the pattern", "the rhythm", "the geometry",
    "the equation", "the paradox", "the mystery", "the answer", "the question", "the truth",
    "the illusion", "the reality", "the shadow", "the light", "the darkness", "the brightness",
    "the moment", "the eternity", "the beginning", "the end", "the middle", "the edge",
    "the center", "the surface", "the depth", "the height", "the width", "the length",
    "the form", "the substance", "the essence", "the appearance", "the meaning", "the purpose",
    "the cause", "the effect", "the reason", "the result", "the origin", "the destination",
    "the path", "the journey", "the arrival", "the departure", "the presence", "the absence",
    "the memory", "the forgetting", "the knowing", "the unknowing", "the certainty", "the doubt",
]

ACTIONS = [
    "grows louder", "fades away", "transforms", "dissolves", "crystallizes", "evaporates",
    "collapses", "expands", "contracts", "reverberates", "echoes", "resonates",
    "multiplies", "divides", "converges", "diverges", "spirals", "circles",
    "ascends", "descends", "floats", "sinks", "rises", "falls",
    "accelerates", "decelerates", "pauses", "continues", "begins", "ends",
    "emerges", "submerges", "appears", "disappears", "manifests", "vanishes",
    "intensifies", "diminishes", "strengthens", "weakens", "clarifies", "obscures",
    "reveals", "conceals", "opens", "closes", "connects", "separates",
    "unifies", "fragments", "integrates", "disintegrates", "forms", "deforms",
    "stabilizes", "destabilizes", "balances", "unbalances", "aligns", "misaligns",
    "synchronizes", "desynchronizes", "harmonizes", "clashes", "blends", "contrasts",
    "reflects", "absorbs", "emits", "receives", "transmits", "blocks",
    "amplifies", "dampens", "sharpens", "blurs", "focuses", "scatters",
    "accumulates", "disperses", "gathers", "spreads", "concentrates", "dilutes",
    "condenses", "expands", "compresses", "releases", "captures", "escapes",
    "persists", "ceases", "endures", "expires", "survives", "perishes",
]

OUTCOMES = [
    "with each passing moment", "into pure abstraction", "beyond all comprehension",
    "through infinite dimensions", "across parallel realities", "within nested paradoxes",
    "despite logical impossibility", "toward absolute zero", "away from certainty",
    "into recursive loops", "through quantum superposition", "beyond the event horizon",
    "within the singularity", "across temporal boundaries", "through spatial distortions",
    "into mathematical impossibility", "beyond rational thought", "within pure consciousness",
    "through the fabric of reality", "into the void itself", "beyond existence",
    "within the eternal now", "across infinite timelines", "through collapsed dimensions",
    "into unified field", "beyond duality", "within the paradox", "through the mirror",
    "into the unknown", "beyond the knowable", "within the ineffable", "through silence",
    "into chaos", "beyond order", "within entropy", "through negentropy",
    "into complexity", "beyond simplicity", "within the pattern", "through randomness",
    "into meaning", "beyond interpretation", "within context", "through isolation",
    "into connection", "beyond separation", "within unity", "through division",
    "into wholeness", "beyond fragmentation", "within integration", "through dissolution",
]

def main():
    # Initialize random number generator with seed
    random.seed(SEED)
    
    # Open both output files
    with open('prompts.txt', 'w') as prompt_file, open('followups.txt', 'w') as followup_file:
        # Generate pairs
        for i in range(COUNT):
            # Serial number (1-indexed for human readability)
            serial = i + 1
            
            # Generate PROMPT (4 random calls)
            clause = random.choice(CLAUSES)
            modifier = random.choice(MODIFIERS)
            obj = random.choice(OBJECTS)
            location = random.choice(LOCATIONS)
            prompt = f"[{serial}] {clause} {modifier} {obj} {location}."
            
            # Generate FOLLOW-UP (4 random calls)
            connector = random.choice(CONNECTORS)
            subject = random.choice(SUBJECTS)
            action = random.choice(ACTIONS)
            outcome = random.choice(OUTCOMES)
            followup = f"[{serial}] {connector} {subject} {action} {outcome}."
            
            # Write to files
            prompt_file.write(prompt + '\n')
            followup_file.write(followup + '\n')
            
            # Progress reporting (to stderr so it doesn't interfere with output)
            if (i + 1) % 100000 == 0:
                print(f"Generated {i + 1:,} pairs...", file=sys.stderr)
    
    print(f"\nComplete! Generated {COUNT:,} pairs:", file=sys.stderr)
    print(f"  - prompts.txt: {COUNT:,} lines", file=sys.stderr)
    print(f"  - followups.txt: {COUNT:,} lines", file=sys.stderr)

if __name__ == "__main__":
    main()
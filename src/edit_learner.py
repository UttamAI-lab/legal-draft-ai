"""
Edit Learner
------------
Captures operator edits to generated drafts, extracts reusable
patterns, and builds a preference profile to improve future drafts.

This is the learning loop: original draft → operator edits → 
extracted patterns → better next draft.
"""

import json
import difflib
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class EditRecord:
    """Stores a single operator edit event."""
    timestamp: str
    draft_type: str
    original: str
    edited: str
    changes: list
    extracted_patterns: list


class EditLearner:
    """
    Learns from operator edits to improve future draft generation.
    
    Approach:
    - Diff original vs edited draft
    - Categorize changes (tone, structure, additions, deletions)
    - Build a "style profile" that's injected into future prompts
    - Persist learning across sessions
    """

    def __init__(self, memory_path: str = "./edit_memory.json"):
        self.memory_path = memory_path
        self.memory = self._load_memory()

    def record_edit(self, original: str, edited: str, draft_type: str) -> EditRecord:
        """
        Record an operator edit and extract learnable patterns.
        """
        changes = self._diff(original, edited)
        patterns = self._extract_patterns(original, edited, changes)
        
        record = EditRecord(
            timestamp=datetime.now().isoformat(),
            draft_type=draft_type,
            original=original,
            edited=edited,
            changes=changes,
            extracted_patterns=patterns
        )
        
        # Update memory
        if draft_type not in self.memory["patterns"]:
            self.memory["patterns"][draft_type] = []
        
        self.memory["patterns"][draft_type].extend(patterns)
        self.memory["edit_count"] += 1
        self.memory["last_updated"] = datetime.now().isoformat()
        
        # Keep only last 50 patterns per type to avoid bloat
        self.memory["patterns"][draft_type] = \
            self.memory["patterns"][draft_type][-50:]
        
        self._save_memory()
        return record

    def get_style_guidelines(self, draft_type: str) -> str:
        """
        Returns a style guidelines string to inject into the draft
        generation prompt, based on past operator edits.
        """
        patterns = self.memory["patterns"].get(draft_type, [])
        all_patterns = self.memory["patterns"].get("general", [])
        
        combined = all_patterns + patterns
        
        if not combined:
            return ""
        
        # Deduplicate
        unique = list(dict.fromkeys(combined))
        
        guidelines = "LEARNED STYLE PREFERENCES FROM PREVIOUS EDITS:\n"
        for i, p in enumerate(unique[-10:], 1):  # Last 10 patterns
            guidelines += f"{i}. {p}\n"
        
        return guidelines

    def _diff(self, original: str, edited: str) -> list:
        """Generate a structured diff between two texts."""
        original_lines = original.splitlines(keepends=True)
        edited_lines = edited.splitlines(keepends=True)
        
        differ = difflib.unified_diff(
            original_lines, 
            edited_lines,
            lineterm='',
            n=0
        )
        
        changes = []
        for line in differ:
            if line.startswith('+') and not line.startswith('+++'):
                changes.append({"type": "addition", "text": line[1:].strip()})
            elif line.startswith('-') and not line.startswith('---'):
                changes.append({"type": "deletion", "text": line[1:].strip()})
        
        return changes

    def _extract_patterns(self, original: str, edited: str, changes: list) -> list:
        """
        Analyze changes to extract reusable style patterns.
        """
        patterns = []
        
        if not changes:
            return patterns
        
        additions = [c["text"] for c in changes if c["type"] == "addition"]
        deletions = [c["text"] for c in changes if c["type"] == "deletion"]
        
        # Pattern: formality adjustments
        formal_words = ['shall', 'hereby', 'wherein', 'pursuant', 'whereas', 'aforesaid']
        informal_words = ['will', 'here', 'where', 'according to', 'regarding', 'mentioned']
        
        added_formal = any(w in ' '.join(additions).lower() for w in formal_words)
        removed_informal = any(w in ' '.join(deletions).lower() for w in informal_words)
        
        if added_formal or removed_informal:
            patterns.append("Use formal legal language: prefer 'shall' over 'will', 'pursuant to' over 'according to'")

        # Pattern: length changes
        orig_len = len(original.split())
        edit_len = len(edited.split())
        if edit_len < orig_len * 0.8:
            patterns.append("Keep drafts concise — operator consistently shortens verbose text")
        elif edit_len > orig_len * 1.2:
            patterns.append("Include more detail — operator consistently expands brief drafts")

        # Pattern: structural additions (headings, bullets)
        heading_pattern = r'^#{1,3}\s+\w+'
        added_headings = [a for a in additions if re.match(heading_pattern, a)]
        if added_headings:
            patterns.append("Use clear section headings to structure the draft")

        # Pattern: specific phrases added
        for addition in additions:
            if len(addition) > 20 and len(addition) < 150:
                if any(kw in addition.lower() for kw in 
                       ['subject to', 'notwithstanding', 'in accordance', 'without prejudice']):
                    patterns.append(f"Include standard legal qualifiers like: '{addition[:80]}...'")

        # Pattern: what was removed (things to avoid)
        for deletion in deletions:
            if 'i think' in deletion.lower() or 'probably' in deletion.lower():
                patterns.append("Avoid hedging language like 'I think' or 'probably' in legal drafts")
            if 'very' in deletion.lower() or 'really' in deletion.lower():
                patterns.append("Avoid filler intensifiers like 'very' or 'really'")

        return list(set(patterns))  # Deduplicate

    def _load_memory(self) -> dict:
        """Load persisted edit memory."""
        if Path(self.memory_path).exists():
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        return {
            "patterns": {},
            "edit_count": 0,
            "last_updated": None
        }

    def _save_memory(self):
        """Persist edit memory to disk."""
        with open(self.memory_path, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def get_stats(self) -> dict:
        """Return learning statistics."""
        return {
            "total_edits_recorded": self.memory["edit_count"],
            "draft_types_learned": list(self.memory["patterns"].keys()),
            "pattern_counts": {
                k: len(v) for k, v in self.memory["patterns"].items()
            },
            "last_updated": self.memory["last_updated"]
        }
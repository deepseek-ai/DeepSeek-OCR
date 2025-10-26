"""
Intelligent Post-Processing Utilities
Spell-check, grammar correction, table validation, formula verification
"""

import re
from typing import List, Dict, Any, Optional, Tuple


class PostProcessor:
    """Main post-processing coordinator"""

    def __init__(self, enable_spellcheck: bool = True,
                 enable_grammar: bool = True,
                 enable_table_validation: bool = True,
                 enable_formula_check: bool = True):
        self.enable_spellcheck = enable_spellcheck
        self.enable_grammar = enable_grammar
        self.enable_table_validation = enable_table_validation
        self.enable_formula_check = enable_formula_check

    def process(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Process text with all enabled features"""
        issues = {
            "spelling_errors": [],
            "grammar_issues": [],
            "table_issues": [],
            "formula_issues": [],
            "corrections_applied": 0
        }

        processed_text = text

        # Spell check
        if self.enable_spellcheck:
            processed_text, spell_issues = SpellChecker.check_and_correct(processed_text)
            issues["spelling_errors"] = spell_issues
            issues["corrections_applied"] += len(spell_issues)

        # Grammar check
        if self.enable_grammar:
            processed_text, grammar_issues = GrammarChecker.check_and_correct(processed_text)
            issues["grammar_issues"] = grammar_issues
            issues["corrections_applied"] += len(grammar_issues)

        # Table validation
        if self.enable_table_validation:
            table_issues = TableValidator.validate(processed_text)
            issues["table_issues"] = table_issues

        # Formula validation
        if self.enable_formula_check:
            formula_issues = FormulaValidator.validate(processed_text)
            issues["formula_issues"] = formula_issues

        return processed_text, issues


class SpellChecker:
    """Spell checking and correction"""

    @staticmethod
    def check_and_correct(text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Check spelling and apply corrections"""
        try:
            from spellchecker import SpellChecker as PySpellChecker

            spell = PySpellChecker()
            issues = []
            corrected_text = text

            # Split into words, preserving structure
            words = re.findall(r'\b[a-zA-Z]+\b', text)

            for word in words:
                # Skip short words and known technical terms
                if len(word) < 3 or word.isupper() or word[0].isupper():
                    continue

                # Check if misspelled
                if word.lower() in spell:
                    continue

                # Get correction
                correction = spell.correction(word)

                if correction and correction != word:
                    # Apply correction with word boundary matching
                    pattern = r'\b' + re.escape(word) + r'\b'
                    corrected_text = re.sub(pattern, correction, corrected_text, count=1)

                    issues.append({
                        "word": word,
                        "correction": correction,
                        "type": "spelling"
                    })

            return corrected_text, issues

        except ImportError:
            # Fallback: basic corrections for common OCR errors
            return SpellChecker._basic_corrections(text), []

    @staticmethod
    def _basic_corrections(text: str) -> str:
        """Apply basic OCR error corrections"""
        corrections = {
            # Common OCR character confusions
            r'\bI\b(?=[a-z])': 'l',  # I -> l in lowercase context
            r'(?<=[a-z])I(?=[a-z])': 'l',
            r'\b0(?=[a-z])': 'o',  # 0 -> o
            r'(?<=[a-z])0(?=[a-z])': 'o',
            r'\b1(?=[a-z])': 'l',  # 1 -> l
            r'rn': 'm',  # rn -> m (common OCR error)

            # Common word corrections
            r'\bteh\b': 'the',
            r'\badn\b': 'and',
            r'\bwith\s+in\b': 'within',
        }

        corrected = text
        for pattern, replacement in corrections.items():
            corrected = re.sub(pattern, replacement, corrected)

        return corrected


class GrammarChecker:
    """Grammar checking and correction"""

    @staticmethod
    def check_and_correct(text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Check grammar and apply corrections"""
        try:
            import language_tool_python

            tool = language_tool_python.LanguageTool('en-US')
            matches = tool.check(text)

            issues = []
            corrected_text = text

            # Apply corrections in reverse order to maintain positions
            for match in reversed(matches):
                if match.replacements:
                    issues.append({
                        "error": text[match.offset:match.offset + match.errorLength],
                        "suggestion": match.replacements[0],
                        "message": match.message,
                        "type": "grammar"
                    })

                    # Apply first suggestion
                    corrected_text = (
                        corrected_text[:match.offset] +
                        match.replacements[0] +
                        corrected_text[match.offset + match.errorLength:]
                    )

            return corrected_text, issues

        except ImportError:
            # Fallback: basic grammar fixes
            return GrammarChecker._basic_fixes(text), []

    @staticmethod
    def _basic_fixes(text: str) -> str:
        """Apply basic grammar fixes"""
        fixes = {
            # Double spaces
            r'\s{2,}': ' ',

            # Space before punctuation
            r'\s+([.,;:!?])': r'\1',

            # Missing space after punctuation
            r'([.,;:!?])([A-Z])': r'\1 \2',

            # Line breaks in middle of sentences
            r'([a-z])\n([a-z])': r'\1 \2',

            # Multiple newlines
            r'\n{3,}': '\n\n',
        }

        corrected = text
        for pattern, replacement in fixes.items():
            corrected = re.sub(pattern, replacement, corrected)

        return corrected.strip()


class TableValidator:
    """Table structure validation"""

    @staticmethod
    def validate(text: str) -> List[Dict[str, Any]]:
        """Validate table structures in markdown"""
        issues = []

        # Find all markdown tables
        table_pattern = r'\|.+\|[\r\n]+\|[\s\-:]+\|.+?(?=\n\n|\Z)'
        tables = re.finditer(table_pattern, text, re.DOTALL)

        for table_idx, table_match in enumerate(tables):
            table_text = table_match.group(0)
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]

            if len(lines) < 2:
                issues.append({
                    "table_index": table_idx,
                    "issue": "Table too short",
                    "severity": "error"
                })
                continue

            # Check header
            header = lines[0]
            separator = lines[1] if len(lines) > 1 else ""

            header_cols = len([c for c in header.split('|') if c.strip()])
            sep_cols = len([c for c in separator.split('|') if c.strip()])

            if header_cols != sep_cols:
                issues.append({
                    "table_index": table_idx,
                    "issue": f"Column count mismatch: header={header_cols}, separator={sep_cols}",
                    "severity": "warning"
                })

            # Check all rows have same column count
            for row_idx, row in enumerate(lines[2:], start=2):
                row_cols = len([c for c in row.split('|') if c.strip()])
                if row_cols != header_cols:
                    issues.append({
                        "table_index": table_idx,
                        "row": row_idx,
                        "issue": f"Column count mismatch: expected={header_cols}, found={row_cols}",
                        "severity": "warning"
                    })

            # Check for empty cells (might indicate OCR errors)
            empty_cell_count = 0
            for line in lines[2:]:
                cells = [c.strip() for c in line.split('|') if '|' in line]
                empty_cell_count += sum(1 for c in cells if not c)

            if empty_cell_count > len(lines) * header_cols * 0.3:  # >30% empty
                issues.append({
                    "table_index": table_idx,
                    "issue": f"High number of empty cells: {empty_cell_count}",
                    "severity": "info"
                })

        return issues


class FormulaValidator:
    """LaTeX formula validation"""

    @staticmethod
    def validate(text: str) -> List[Dict[str, Any]]:
        """Validate LaTeX formulas"""
        issues = []

        # Find all LaTeX formulas
        inline_pattern = r'\$([^$]+)\$'
        block_pattern = r'\\\[(.*?)\\\]'

        # Check inline formulas
        for idx, match in enumerate(re.finditer(inline_pattern, text)):
            formula = match.group(1)
            formula_issues = FormulaValidator._check_formula(formula, 'inline', idx)
            issues.extend(formula_issues)

        # Check block formulas
        for idx, match in enumerate(re.finditer(block_pattern, text, re.DOTALL)):
            formula = match.group(1)
            formula_issues = FormulaValidator._check_formula(formula, 'block', idx)
            issues.extend(formula_issues)

        return issues

    @staticmethod
    def _check_formula(formula: str, formula_type: str, index: int) -> List[Dict[str, Any]]:
        """Check individual formula for common issues"""
        issues = []

        # Check for unmatched braces
        brace_balance = formula.count('{') - formula.count('}')
        if brace_balance != 0:
            issues.append({
                "formula_type": formula_type,
                "formula_index": index,
                "issue": f"Unmatched braces: {brace_balance}",
                "severity": "error",
                "formula": formula[:50] + "..." if len(formula) > 50 else formula
            })

        # Check for unmatched brackets
        bracket_balance = formula.count('[') - formula.count(']')
        if bracket_balance != 0:
            issues.append({
                "formula_type": formula_type,
                "formula_index": index,
                "issue": f"Unmatched brackets: {bracket_balance}",
                "severity": "error",
                "formula": formula[:50] + "..." if len(formula) > 50 else formula
            })

        # Check for unmatched parentheses
        paren_balance = formula.count('(') - formula.count(')')
        if paren_balance != 0:
            issues.append({
                "formula_type": formula_type,
                "formula_index": index,
                "issue": f"Unmatched parentheses: {paren_balance}",
                "severity": "error",
                "formula": formula[:50] + "..." if len(formula) > 50 else formula
            })

        # Check for common LaTeX command errors
        common_commands = [
            'frac', 'sqrt', 'sum', 'int', 'lim', 'infty',
            'alpha', 'beta', 'gamma', 'delta', 'epsilon',
            'sin', 'cos', 'tan', 'log', 'ln', 'exp'
        ]

        # Check for malformed commands (backslash but not a known command)
        command_pattern = r'\\([a-zA-Z]+)'
        for match in re.finditer(command_pattern, formula):
            command = match.group(1)
            if command not in common_commands and len(command) > 2:
                issues.append({
                    "formula_type": formula_type,
                    "formula_index": index,
                    "issue": f"Unknown or possibly malformed command: \\{command}",
                    "severity": "warning",
                    "formula": formula[:50] + "..." if len(formula) > 50 else formula
                })

        # Check for OCR artifacts (common confusions)
        artifacts = {
            'I': 'l',  # Capital I vs lowercase l
            '0': 'O',  # Zero vs capital O
            '×': r'\times',
            '÷': r'\div',
        }

        for artifact, suggestion in artifacts.items():
            if artifact in formula:
                issues.append({
                    "formula_type": formula_type,
                    "formula_index": index,
                    "issue": f"Possible OCR artifact '{artifact}', consider '{suggestion}'",
                    "severity": "info",
                    "formula": formula[:50] + "..." if len(formula) > 50 else formula
                })

        return issues


class TextQualityAnalyzer:
    """Analyze overall text quality"""

    @staticmethod
    def analyze(text: str) -> Dict[str, Any]:
        """Comprehensive text quality analysis"""
        return {
            "character_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.split('\n')),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "average_word_length": sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            "has_tables": bool(re.search(r'\|.+\|', text)),
            "table_count": len(re.findall(r'\|.+\|[\r\n]+\|[\s\-:]+\|', text)),
            "has_formulas": bool(re.search(r'\$.*?\$|\\\[.*?\\\]', text)),
            "formula_count": len(re.findall(r'\$.*?\$|\\\[.*?\\\]', text)),
            "has_code_blocks": bool(re.search(r'```', text)),
            "code_block_count": text.count('```') // 2,
            "special_characters_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
        }

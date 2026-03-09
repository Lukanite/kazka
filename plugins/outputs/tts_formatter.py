"""
TTS Text Formatter

Processes LLM responses to make them suitable for text-to-speech synthesis.
Removes action descriptions, formatting, and cleans up text for natural speech.
"""

import re


class TTSFormatter:
    """
    Formats LLM responses for optimal TTS delivery.

    This class handles various text formatting challenges:
    - Removes action descriptions (like *eyes narrow*)
    - Converts formatting to natural speech pauses
    - Handles punctuation for better speech rhythm
    - Ensures text is clean and pronounceable for TTS engines
    """

    def __init__(self):
        """
        Initialize TTS formatter.
        """
        self._setup_patterns()

    def _setup_patterns(self):
        """Setup regex patterns for text cleaning."""
        # Action descriptions in brackets and parentheses only (asterisks are for emphasis)
        self.action_patterns = [
            r'\[[^\]]+\]',  # [stage directions]
            r'\([^)]+\)',  # (parenthetical actions)
        ]

        # Multiple consecutive punctuation
        self.punctuation_patterns = [
            r'\.{3,}',  # Multiple periods
            r'!{2,}',   # Multiple exclamation marks
            r'\?{2,}',  # Multiple question marks
        ]

    def format_for_tts(self, text: str) -> str:
        """
        Format LLM response for optimal TTS delivery.

        Args:
            text: Raw LLM response text

        Returns:
            Cleaned and formatted text suitable for TTS
        """
        if not text:
            return ""

        # Step 1: Handle action descriptions
        formatted_text = self._remove_actions(text)

        # Step 2: Handle emphasis (remove asterisks but keep words)
        formatted_text = self._handle_emphasis(formatted_text)

        # Step 3: Extract and handle quoted speech
        formatted_text = self._handle_quoted_speech(formatted_text)

        # Skipping these steps because Kokoro seems to be able to tackle them?
        # Step 4: Clean up punctuation
        # formatted_text = self._clean_punctuation(formatted_text)

        # Step 5: Add natural pauses and breaks
        # formatted_text = self._add_speech_pauses(formatted_text)

        # Step 6: Final cleanup
        # formatted_text = self._final_cleanup(formatted_text)

        return formatted_text.strip()

    def _remove_actions(self, text: str) -> str:
        """Remove action descriptions while preserving context."""
        result = text

        # Remove asterisk actions like "*eyes narrow*"
        for pattern in self.action_patterns:
            result = re.sub(pattern, '', result)

        return result

    def _handle_emphasis(self, text: str) -> str:
        """Remove asterisks from emphasized words while keeping the words."""
        # Remove asterisks but keep the content (for emphasis)
        result = re.sub(r'\*([^*]+)\*', r'\1', text)
        return result

    def _handle_quoted_speech(self, text: str) -> str:
        """Handle quoted speech - remove quotes but keep content."""
        result = text

        # Process ASCII double quotes but avoid removing quotes that are
        # clearly inside words (use lookarounds to ensure quote marks are
        # not adjacent to word characters). This preserves apostrophes in
        # contractions/possessives while still stripping quoted speech.
        result = re.sub(r'(?<!\w)"([^\"]*?)"(?!\w)', r'\1', result)

        # Handle curly double quotes “ ”
        result = re.sub(r'(?<!\w)“([^“”]*?)”(?!\w)', r'\1', result)

        # Handle ASCII single quotes as speech quotes, but only when not
        # part of a word (prevents eating apostrophes in don't/John's).
        result = re.sub(r"(?<!\w)'([^']*?)'(?!\w)", r'\1', result)

        # Handle curly single quotes ‘ ’
        result = re.sub(r"(?<!\w)‘([^‘’]*?)’(?!\w)", r'\1', result)

        return result

    def _clean_punctuation(self, text: str) -> str:
        """Clean up excessive punctuation."""
        result = text

        # Replace multiple periods with single period
        result = re.sub(r'\.{2,}', '.', result)

        # Replace multiple exclamation marks with single
        result = re.sub(r'!{2,}', '!', result)

        # Replace multiple question marks with single
        result = re.sub(r'\?{2,}', '?', result)

        # Fix punctuation spacing
        result = re.sub(r'\s+([.!?])', r'\1', result)  # Remove space before punctuation
        result = re.sub(r'([.!?])\s*', r'\1 ', result)  # Ensure space after punctuation

        return result

    def _add_speech_pauses(self, text: str) -> str:
        """Add natural pauses and speech rhythm."""
        result = text

        # Add commas for natural pauses before conjunctions (avoid double commas)
        result = re.sub(r'\s+(but|and|so|because|however|therefore)\s+', r', \1 ', result)
        # Fix any double commas that might result
        result = re.sub(r',,', ',', result)

        # Ensure proper sentence endings (be more careful about existing periods)
        # Only add period if there's actually a sentence boundary, not just capitalized words
        result = re.sub(r'([a-z])(\s+)([A-Z][a-z][a-z]+)', r'\1. \3', result)  # Add period between real sentences

        return result

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup of extra whitespace and formatting."""
        result = text

        # Fix common formatting issues
        result = re.sub(r',,', ',', result)  # Double commas
        result = re.sub(r'\. \.', '.', result)  # Double periods
        result = re.sub(r'a\.\s+([A-Z])', r'a \1', result)  # Fix "a. Word" -> "a Word"

        # Remove extra whitespace
        result = re.sub(r'\s+', ' ', result)

        # Clean up spacing and leading/trailing spaces
        result = result.strip()

        return result

    def debug_format(self, text: str) -> dict:
        """
        Debug version that shows transformation steps.

        Args:
            text: Raw LLM response text

        Returns:
            Dictionary showing each transformation step
        """
        after_actions = self._remove_actions(text)
        after_emphasis = self._handle_emphasis(after_actions)
        after_quotes = self._handle_quoted_speech(after_emphasis)
        after_punctuation = self._clean_punctuation(after_quotes)
        after_pauses = self._add_speech_pauses(after_punctuation)

        steps = {
            'original': text,
            'after_actions': after_actions,
            'after_emphasis': after_emphasis,
            'after_quotes': after_quotes,
            'after_punctuation': after_punctuation,
            'after_pauses': after_pauses,
            'final': self.format_for_tts(text)
        }
        return steps


# Global formatter instance
def get_tts_formatter() -> TTSFormatter:
    """Get a TTS formatter instance."""
    return TTSFormatter()


# Convenience function for quick formatting
def format_for_tts(text: str) -> str:
    """
    Quick formatting function for TTS.

    Args:
        text: Text to format

    Returns:
        Formatted text ready for TTS
    """
    formatter = get_tts_formatter()
    return formatter.format_for_tts(text)


if __name__ == "__main__":
    # Test the formatter with sample text
    sample_text = """*Eyes narrow, assessing*

Lucario... a creature of pride and righteous fury. *Pauses* Neither friend nor foe - merely a tool. Their strength is undeniable, but their honor is a leash. *Scoffs* They cling to justice like a child to a blanket.

"Leans closer, voice dropping to a whisper" The real danger lies in their potential."""

    formatter = TTSFormatter()
    print("Original:")
    print(sample_text)
    print("\nFormatted for TTS:")
    print(formatter.format_for_tts(sample_text))

    # Debug view
    print("\nDebug steps:")
    steps = formatter.debug_format(sample_text)
    for step, result in steps.items():
        print(f"\n{step}:")
        print(result)
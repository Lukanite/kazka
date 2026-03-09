"""
Phonetic verification module for wake word confirmation.
Uses phonetic algorithms and fuzzy matching to verify wake word detection.
Based on the original implementation from satellite_combined.py
"""

import jellyfish
from core.config import config


class PhoneticVerifier:
    """
    Verifies if transcribed text sounds like the target wake word.

    This class provides three layers of verification:
    1. Exact Alias Match (for known Whisper hallucinations)
    2. Exact Phonetic Match (Metaphone algorithm)
    3. Fuzzy Phonetic Match (Levenshtein distance)
    """

    def __init__(self, target_word: str = None):
        """
        Initialize phonetic verifier.

        Args:
            target_word: Target wake word to verify
        """
        self.target_word = target_word or config.wake_word.target
        self.target_phonetic_code = jellyfish.metaphone(self.target_word)
        self.aliases = config.wake_word.phonetic_aliases
        self.max_levenshtein_distance = config.wake_word.max_levenshtein_distance

    def verify(self, text: str) -> bool:
        """
        Verify if the text sounds like the target wake word.

        Args:
            text: Transcribed text to verify

        Returns:
            True if text is verified as wake word, False otherwise
        """
        if not text or not text.strip():
            return False

        # Clean text for matching (from original)
        clean_text = text.lower().replace(",", "").replace(".", "").replace("'", "")

        # 1. Check Aliases (from original)
        for alias in self.aliases:
            if alias.replace("'", "") in clean_text:
                print(f"   ✅ Match Found (Alias): '{alias}'")
                return True

        # 2. Phonetic Matching (from original)
        words = clean_text.split()
        print(f"      [Phonetic] Target: {self.target_word} ({self.target_phonetic_code})")

        for word in words:
            word_code = jellyfish.metaphone(word)

            # A. Exact Code Match (from original)
            if word_code == self.target_phonetic_code:
                print(f"      ✅ Match Found (Exact): '{word}' ({word_code})")
                return True

            # B. Fuzzy Code Match (Allow 1 edit distance - from original)
            # This catches 'Cast' (KST) vs 'Kazka' (KSK)
            if len(word_code) > 0 and jellyfish.levenshtein_distance(word_code, self.target_phonetic_code) <= self.max_levenshtein_distance:
                print(f"      ✅ Match Found (Fuzzy): '{word}' ({word_code} ~= {self.target_phonetic_code})")
                return True

        # No matches found (from original)
        print(f"      ❌ No Match. Heard codes: {[jellyfish.metaphone(w) for w in words]}")
        return False


class WakeWordVerifier:
    """
    High-level wake word verification manager.

    Simplified version that matches the original satellite_combined.py functionality.
    """

    def __init__(self, target_word: str = None):
        """
        Initialize wake word verifier.

        Args:
            target_word: Target wake word to verify
        """
        self.phonetic_verifier = PhoneticVerifier(target_word)

    def verify_wake_word(self, text: str) -> bool:
        """
        Verify wake word using phonetic verification.

        Args:
            text: Transcribed text to verify

        Returns:
            True if wake word is verified
        """
        return self.phonetic_verifier.verify(text)

    def get_verification_info(self) -> dict:
        """
        Get information about the verifier configuration.

        Returns:
            Dictionary with verifier information
        """
        return {
            'target_word': self.phonetic_verifier.target_word,
            'phonetic_code': self.phonetic_verifier.target_phonetic_code,
            'aliases_count': len(self.phonetic_verifier.aliases)
        }
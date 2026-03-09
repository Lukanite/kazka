"""
Audio utilities module for debugging, processing, and helper functions.
Provides common audio processing utilities and debugging capabilities.
"""

import wave
import datetime
import numpy as np
from typing import Optional
from core.config import config


class AudioDebugger:
    """
    Audio debugging utilities for saving and analyzing audio data.

    This class provides methods to save audio data for debugging purposes,
    including timestamped files and format conversion utilities.
    """

    @staticmethod
    def save_debug_audio(
        audio_data: np.ndarray,
        sample_rate: int,
        prefix: str = "debug_",
        filename: Optional[str] = None
    ) -> str:
        """
        Save numpy audio data to a timestamped WAV file for debugging.

        Args:
            audio_data: Audio data as numpy array (float32, -1.0 to 1.0)
            sample_rate: Sample rate of the audio data
            prefix: Prefix for the filename
            filename: Optional custom filename (overrides timestamping)

        Returns:
            Path to the saved file
        """
        if filename is None:
            # Generate timestamped filename
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            filename = f"{prefix}{timestamp}.wav"

        try:
            # Convert Float32 -> Int16 (Standard WAV format)
            # Audio is -1.0 to 1.0, so we scale to 32767
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Write to disk
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 2 bytes (16-bit)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

            print(f"      💾 Saved debug audio: {filename}")
            return filename

        except Exception as e:
            print(f"      ❌ Failed to save debug audio: {e}")
            return ""

    @staticmethod
    def load_audio_file(filename: str) -> tuple[np.ndarray, int]:
        """
        Load audio from a WAV file.

        Args:
            filename: Path to the WAV file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            with wave.open(filename, 'rb') as wf:
                # Read audio data
                frames = wf.readframes(-1)
                audio_data = np.frombuffer(frames, dtype=np.int16)

                # Get sample rate and other info
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()

                # Convert to float32 and normalize to -1.0 to 1.0
                audio_float = audio_data.astype(np.float32) / 32767.0

                # Convert to mono if stereo
                if channels > 1:
                    audio_float = audio_float.reshape(-1, channels).mean(axis=1)

                return audio_float, sample_rate

        except Exception as e:
            print(f"   ❌ Failed to load audio file {filename}: {e}")
            return np.array([]), 0


class AudioProcessor:
    """
    Audio processing utilities for common operations.

    This class provides static methods for common audio processing tasks
    like resampling, conversion, and analysis.
    """

    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio data to -1.0 to 1.0 range.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Normalized audio data
        """
        if len(audio_data) == 0:
            return audio_data

        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data

    @staticmethod
    def apply_gain(audio_data: np.ndarray, gain_db: float) -> np.ndarray:
        """
        Apply gain to audio data in decibels.

        Args:
            audio_data: Audio data as numpy array
            gain_db: Gain in decibels

        Returns:
            Audio data with gain applied
        """
        gain_linear = 10 ** (gain_db / 20.0)
        return audio_data * gain_linear

    @staticmethod
    def fade_in_out(
        audio_data: np.ndarray,
        sample_rate: int,
        fade_duration: float = 0.1
    ) -> np.ndarray:
        """
        Apply fade in and fade out to audio data.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            fade_duration: Duration of fade in seconds

        Returns:
            Audio data with fade applied
        """
        fade_samples = int(fade_duration * sample_rate)
        audio_length = len(audio_data)

        if audio_length == 0:
            return audio_data

        # Create fade in curve
        fade_in = np.linspace(0, 1, min(fade_samples, audio_length // 2))

        # Create fade out curve
        fade_out = np.linspace(1, 0, min(fade_samples, audio_length // 2))

        # Apply fades
        result = audio_data.copy()
        result[:len(fade_in)] *= fade_in
        result[-len(fade_out):] *= fade_out

        return result

    @staticmethod
    def detect_silence(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Detect if audio data is mostly silence.

        Args:
            audio_data: Audio data as numpy array
            threshold: Threshold for silence detection (RMS)

        Returns:
            True if audio is mostly silence
        """
        if len(audio_data) == 0:
            return True

        # Calculate RMS (Root Mean Square)
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms < threshold

    @staticmethod
    def get_audio_stats(audio_data: np.ndarray) -> dict:
        """
        Get statistics about audio data.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Dictionary with audio statistics
        """
        if len(audio_data) == 0:
            return {
                'length_samples': 0,
                'length_seconds': 0.0,
                'max_amplitude': 0.0,
                'min_amplitude': 0.0,
                'rms': 0.0,
                'peak_db': -np.inf
            }

        # Calculate statistics
        max_amp = np.max(np.abs(audio_data))
        min_amp = np.min(audio_data)
        rms = np.sqrt(np.mean(audio_data ** 2))

        # Calculate peak in decibels (relative to full scale)
        if max_amp > 0:
            peak_db = 20 * np.log10(max_amp)
        else:
            peak_db = -np.inf

        return {
            'length_samples': len(audio_data),
            'max_amplitude': float(max_amp),
            'min_amplitude': float(min_amp),
            'rms': float(rms),
            'peak_db': float(peak_db)
        }

    @staticmethod
    def chunk_audio(
        audio_data: np.ndarray,
        chunk_size: int,
        overlap: int = 0
    ) -> list[np.ndarray]:
        """
        Split audio data into chunks.

        Args:
            audio_data: Audio data as numpy array
            chunk_size: Size of each chunk in samples
            overlap: Number of samples to overlap between chunks

        Returns:
            List of audio chunks
        """
        chunks = []
        step = chunk_size - overlap

        for start in range(0, len(audio_data) - chunk_size + 1, step):
            chunk = audio_data[start:start + chunk_size]
            chunks.append(chunk)

        return chunks


class AudioConverter:
    """
    Audio format conversion utilities.

    This class provides methods for converting between different audio formats
    and sample rates.
    """

    @staticmethod
    def float_to_int16(audio_data: np.ndarray) -> np.ndarray:
        """
        Convert float32 audio (-1.0 to 1.0) to int16.

        Args:
            audio_data: Audio data as float32

        Returns:
            Audio data as int16
        """
        return (audio_data * 32767).astype(np.int16)

    @staticmethod
    def int16_to_float(audio_data: np.ndarray) -> np.ndarray:
        """
        Convert int16 audio to float32 (-1.0 to 1.0).

        Args:
            audio_data: Audio data as int16

        Returns:
            Audio data as float32
        """
        return audio_data.astype(np.float32) / 32767.0

    @staticmethod
    def resample_linear(
        audio_data: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio using linear interpolation.

        Args:
            audio_data: Input audio data
            source_rate: Source sample rate
            target_rate: Target sample rate

        Returns:
            Resampled audio data
        """
        if source_rate == target_rate:
            return audio_data

        num_source = len(audio_data)
        num_target = int(num_source * target_rate / source_rate)

        # Create time arrays for interpolation
        x_old = np.linspace(0, num_source, num_source)
        x_new = np.linspace(0, num_source, num_target)

        # Linear interpolation
        return np.interp(x_new, x_old, audio_data)
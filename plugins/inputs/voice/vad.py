"""
Voice Activity Detection module using Silero VAD.
Detects when speech is present in audio streams.
"""

import os
import numpy as np
import onnxruntime


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD model.

    This class provides speech detection capabilities using the Silero VAD ONNX model.
    It maintains internal state for processing audio chunks sequentially.
    """

    def __init__(self, model_path: str = "silero_vad_v4.onnx", download_url: str = "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"):
        """
        Initialize Silero VAD.

        Args:
            model_path: Path to the Silero VAD ONNX model file
            download_url: URL to download the model from if it doesn't exist
        """
        self.model_path = model_path
        self.download_url = download_url
        self.session = None
        self._h = None
        self._c = None

        self._load_model()
        self.reset()

    def _load_model(self):
        """Load the Silero VAD model, downloading if necessary."""
        if not os.path.exists(self.model_path):
            print(f"📥 Downloading Silero VAD model to {self.model_path}...")
            # Ensure the directory exists
            model_dir = os.path.dirname(self.model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                print(f"📁 Created model directory: {model_dir}")

            try:
                import urllib.request
                urllib.request.urlretrieve(
                    self.download_url,
                    self.model_path
                )
                print(f"✅ Silero VAD model downloaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to download Silero VAD model: {e}")

        try:
            opts = onnxruntime.SessionOptions()
            opts.log_severity_level = 3
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                opts,
                providers=['CPUExecutionProvider']
            )
            print(f"✅ Silero VAD model loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD model from {self.model_path}: {e}")

    def reset(self):
        """Reset the internal state of the VAD model."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def process(self, audio_chunk: np.ndarray) -> float:
        """
        Process an audio chunk and return the speech probability.

        Args:
            audio_chunk: Audio data as numpy array (float32, 16kHz, mono)

        Returns:
            Speech probability (0.0 to 1.0)
        """
        if self.session is None:
            raise RuntimeError("VAD model not loaded")

        # Ensure audio is the right shape
        audio_chunk = audio_chunk.flatten()
        if len(audio_chunk) == 0:
            return 0.0

        try:
            # Prepare inputs for the ONNX model
            ort_inputs = {
                'input': audio_chunk[np.newaxis, :],
                'h': self._h,
                'c': self._c,
                'sr': np.array([16000], dtype=np.int64)
            }

            # Run inference
            out, self._h, self._c = self.session.run(None, ort_inputs)
            return float(out[0][0])

        except Exception as e:
            print(f"   ❌ VAD processing error: {e}")
            return 0.0

    def is_speech(self, audio_chunk: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Determine if speech is present in the audio chunk.

        Args:
            audio_chunk: Audio data as numpy array
            threshold: Probability threshold for speech detection

        Returns:
            True if speech is detected, False otherwise
        """
        probability = self.process(audio_chunk)
        return probability >= threshold


class VADProcessor:
    """
    High-level VAD processor that manages silence detection and speech segments.
    """

    def __init__(self, vad_model: SileroVAD = None, silence_threshold: float = 0.5):
        """
        Initialize VAD processor.

        Args:
            vad_model: SileroVAD instance (creates new one if None)
            silence_threshold: Probability threshold for silence detection (0.0 to 1.0)
        """
        self.vad_model = vad_model or SileroVAD()
        self.silence_threshold = silence_threshold
        self.silence_chunks = 0
        print(f"🔊 VAD initialized with silence threshold: {self.silence_threshold}")

    def reset(self):
        """Reset the VAD state and silence counter."""
        self.vad_model.reset()
        self.silence_chunks = 0

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Process an audio chunk and track silence.

        Args:
            audio_chunk: Audio data as numpy array

        Returns:
            True if speech continues, False if silence detected
        """
        probability = self.vad_model.process(audio_chunk)

        if probability < self.silence_threshold:
            self.silence_chunks += 1
        else:
            self.silence_chunks = 0

        return probability >= self.silence_threshold

    def is_silence_duration(self, duration_seconds: float, sample_rate: int = 16000, chunk_size: int = 1280) -> bool:
        """
        Check if silence has been detected for a specified duration.

        Args:
            duration_seconds: Duration of silence to check for
            sample_rate: Audio sample rate
            chunk_size: Size of audio chunks

        Returns:
            True if silence for the specified duration is detected
        """
        chunks_in_duration = int(duration_seconds * sample_rate / chunk_size)
        return self.silence_chunks > chunks_in_duration

    def get_silence_duration(self, chunk_size: int = 1280, sample_rate: int = 16000) -> float:
        """
        Get the current silence duration in seconds.

        Args:
            chunk_size: Size of audio chunks
            sample_rate: Audio sample rate

        Returns:
            Silence duration in seconds
        """
        return self.silence_chunks * (chunk_size / sample_rate)
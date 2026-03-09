"""
Wake word detection module using openWakeWord.
Detects when the user says the configured wake word.
"""

import os
import pathlib
import numpy as np
import openwakeword
from openwakeword.model import Model as WakeWordModel
from typing import Dict, Optional
from core.config import config


class WakeWordDetector:
    """
    Wake word detector using openWakeWord library.

    This class provides wake word detection capabilities using the openWakeWord
    ONNX models. It can load custom wake word models or use the default models.
    """

    def __init__(self, model_file: Optional[str] = None):
        """
        Initialize wake word detector.

        Args:
            model_file: Path to custom wake word model file (.onnx)
                       If None, uses default models
        """
        self.model_file = model_file or config.wake_word.model_file
        self.model = None
        self.target_word = config.wake_word.target
        self.confidence_threshold = config.wake_word.confidence_threshold

        self._load_model()

    def _load_model(self):
        """Load the wake word model, downloading openWakeWord supporting models if necessary."""
        try:
            # Check if the configured wake word model exists
            if not os.path.exists(self.model_file):
                raise RuntimeError(
                    f"Wake word model not found: {self.model_file}\n"
                    f"Please ensure the model file exists or update the path in assistant_settings.toml"
                )

            print(f"   Loading custom wake word model: {self.model_file}")

            # Check if openWakeWord supporting models are present
            if not self._check_openwakeword_models():
                print("   ⚠️  OpenWakeWord supporting models not found. Downloading...")
                openwakeword.utils.download_models()
                print("   ✓ OpenWakeWord models downloaded successfully")

            # Load the custom wake word model
            self.model = WakeWordModel(
                inference_framework="onnx",
                wakeword_models=[self.model_file]
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load wake word model: {e}")

    def predict(self, audio_chunk: np.ndarray) -> Dict[str, float]:
        """
        Predict wake word probabilities for an audio chunk.

        Args:
            audio_chunk: Audio data as numpy array (int16, mono)

        Returns:
            Dictionary mapping wake word names to their confidence scores
        """
        if self.model is None:
            raise RuntimeError("Wake word model not loaded")

        try:
            # Ensure audio is the right format (int16)
            if audio_chunk.dtype != np.int16:
                # Convert float32 (-1.0 to 1.0) to int16
                if audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
                    audio_chunk = (audio_chunk * 32767).astype(np.int16)
                else:
                    audio_chunk = audio_chunk.astype(np.int16)

            # Ensure audio is 1D
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.squeeze()

            predictions = self.model.predict(audio_chunk)
            return predictions

        except Exception as e:
            print(f"   ❌ Wake word prediction error: {e}")
            return {}

    def is_detected(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if the wake word is detected in the audio chunk.

        Args:
            audio_chunk: Audio data as numpy array

        Returns:
            True if wake word is detected with confidence above threshold
        """
        predictions = self.predict(audio_chunk)

        # Check for our target wake word specifically
        if self.target_word in predictions:
            confidence = predictions[self.target_word]
            return confidence >= self.confidence_threshold

        # If target word not found, check the highest confidence
        if predictions:
            max_confidence = max(predictions.values())
            return max_confidence >= self.confidence_threshold

        return False

    def get_max_confidence(self, audio_chunk: np.ndarray) -> float:
        """
        Get the maximum confidence score across all detected wake words.

        Args:
            audio_chunk: Audio data as numpy array

        Returns:
            Maximum confidence score (0.0 to 1.0)
        """
        predictions = self.predict(audio_chunk)
        return max(predictions.values()) if predictions else 0.0

    def reset(self):
        """Reset the internal state of the wake word model."""
        if self.model:
            self.model.reset()

    def _check_openwakeword_models(self) -> bool:
        """
        Check if openWakeWord supporting models are present.

        Returns:
            True if all required models exist, False otherwise
        """
        try:
            import openwakeword.utils
            # Get the path where openWakeWord stores its models
            oww_path = pathlib.Path(openwakeword.utils.__file__).parent.resolve()
            models_path = oww_path / "resources" / "models"

            # Check for required ONNX models
            required_models = [
                "melspectrogram.onnx",
                "embedding_model.onnx"
            ]

            for model_name in required_models:
                model_path = models_path / model_name
                if not model_path.exists():
                    return False

            return True

        except Exception:
            return False


class WakeWordProcessor:
    """
    High-level wake word processor that manages detection and filtering.
    """

    def __init__(self, detector: WakeWordDetector = None):
        """
        Initialize wake word processor.

        Args:
            detector: WakeWordDetector instance (creates new one if None)
        """
        self.detector = detector or WakeWordDetector()
        self.detection_count = 0
        self.detection_threshold = 1  # Number of consecutive detections required

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Process an audio chunk and detect wake word.

        Args:
            audio_chunk: Audio data as numpy array

        Returns:
            True if wake word is detected, False otherwise
        """
        if self.detector.is_detected(audio_chunk):
            self.detection_count += 1
            if self.detection_count >= self.detection_threshold:
                self.detection_count = 0
                return True
        else:
            self.detection_count = 0

        return False

    def reset(self):
        """Reset the wake word detector and detection counter."""
        self.detector.reset()
        self.detection_count = 0

    def set_detection_threshold(self, threshold: int):
        """
        Set the number of consecutive detections required.

        Args:
            threshold: Number of consecutive detections
        """
        self.detection_threshold = max(1, threshold)
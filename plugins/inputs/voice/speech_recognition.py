"""
Speech recognition module with support for both local and remote Whisper implementations.
Provides abstraction for easy switching between local and remote transcription services.
"""

import numpy as np
import requests
import abc
import time
from typing import Optional
from faster_whisper import WhisperModel
from core.config import config


class SpeechRecognizer(abc.ABC):
    """Abstract base class for speech recognition implementations."""

    @abc.abstractmethod
    def transcribe(self, audio_data: np.ndarray, prompt: Optional[str] = None) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Audio data as numpy array (float32, normalized -1.0 to 1.0)
            prompt: Optional prompt to guide transcription

        Returns:
            Transcribed text
        """
        pass

    def transcribe_with_padding(
        self, audio_data: np.ndarray, prompt: Optional[str] = None
    ) -> str:
        """
        Transcribe audio with padding for better results.

        Args:
            audio_data: Audio data as numpy array
            prompt: Optional prompt

        Returns:
            Transcribed text
        """
        padding = np.zeros(int(0.5 * config.audio_devices.sample_rate), dtype=np.float32)
        padded = np.concatenate([padding, audio_data])
        return self.transcribe(padded, prompt)


class LocalWhisperRecognizer(SpeechRecognizer):
    """Local Whisper-based speech recognition."""

    def __init__(self, model_name: str):
        """
        Initialize local Whisper model.

        Args:
            model_name: Name of the Whisper model to load
        """
        print(f"   Loading local Whisper model: {model_name}")
        self.model = WhisperModel(
            model_name,
            device=config.hardware.device,
            compute_type=config.hardware.compute_type
        )

    def transcribe(self, audio_data: np.ndarray, prompt: Optional[str] = None) -> str:
        """
        Transcribe audio using local Whisper model.

        Args:
            audio_data: Audio data as numpy array
            prompt: Optional prompt for better transcription

        Returns:
            Transcribed text
        """
        segments, _ = self.model.transcribe(
            audio_data,
            beam_size=1,
            language="en",
            vad_filter=False,
            initial_prompt=prompt
        )
        return " ".join([s.text for s in segments]).strip().lower().replace(".", "").replace("?", "")


# NOTE: RemoteWhisperRecognizer class removed - we now only support local Whisper and WhisperLive
# If remote HTTP API support is needed in the future, this class can be restored.

# class RemoteWhisperRecognizer(SpeechRecognizer):
#     """Remote Whisper-based speech recognition via HTTP API."""
#     # Implementation removed as it's no longer needed




class OpenAICompatibleRecognizer(SpeechRecognizer):
    """Generic OpenAI-compatible speech recognition using HTTP API."""

    def __init__(self, url: str, model: str = None, use_multipart: bool = True, api_key: str = ""):
        """
        Initialize OpenAI-compatible recognizer.

        Args:
            url: Full API endpoint URL (including path, e.g., http://faceless:8881/v1/audio/transcriptions)
            model: Model name to use for transcription (optional, sent as form parameter)
            use_multipart: Whether to use multipart form data (True) or raw binary (False)
            api_key: Optional API key for Authorization header
        """
        self.url = url.rstrip('/')
        self.model = model
        self.use_multipart = use_multipart
        self.api_key = api_key
        self.api_url = self.url

        print(f"   Using OpenAI-compatible API at {self.api_url}")
        if self.model:
            print(f"   Model: {self.model}")

    def transcribe(self, audio_data: np.ndarray, prompt: Optional[str] = None) -> str:
        """
        Transcribe audio data using OpenAI-compatible API.

        Args:
            audio_data: Audio data as numpy array (float32, normalized -1.0 to 1.0)
            prompt: Optional prompt for better transcription

        Returns:
            Transcribed text
        """
        try:
            import wave
            import io

            # Convert audio to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Create temporary WAV file in memory
            temp_buffer = io.BytesIO()

            with wave.open(temp_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())

            # Reset buffer position to beginning
            temp_buffer.seek(0)

            # Get the complete WAV file data (including header)
            wav_data = temp_buffer.getvalue()

            try:
                auth_headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

                if self.use_multipart and self.model:
                    # Use multipart form data (speaches style)
                    temp_buffer.seek(0)
                    files = {'file': ('audio.wav', temp_buffer, 'audio/wav')}
                    data = {'model': self.model}
                    if prompt:
                        data['prompt'] = prompt

                    response = requests.post(
                        self.api_url,
                        files=files,
                        data=data,
                        headers=auth_headers,
                        timeout=15.0
                    )
                else:
                    # Use raw binary data (WhisperLive style)
                    headers = {
                        'Content-Type': 'audio/wav',
                        'Content-Length': str(len(wav_data)),
                        **auth_headers
                    }
                    response = requests.post(
                        self.api_url,
                        data=wav_data,
                        headers=headers,
                        timeout=15.0
                    )

                response.raise_for_status()

                result = response.json()
                text = result.get('text', '').strip()

                if text:
                    # Return the result as-is for better LLM understanding
                    # LLMs handle punctuation better than stripped text
                    return text.strip()

                return ""

            except requests.exceptions.Timeout:
                print(f"   ⏰ Transcription timeout")
                return ""
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Request error: {e}")
                return ""
            except Exception as e:
                print(f"   ❌ Transcription error: {e}")
                return ""

        except Exception as e:
            print(f"   ❌ Audio processing error: {e}")
            return ""

    def close(self):
        """Close the connection."""
        pass


class SpeechRecognitionFactory:
    """Factory for creating speech recognition instances."""

    @staticmethod
    def create_recognizer(model_name: str) -> SpeechRecognizer:
        """
        Create a speech recognizer based on configuration.

        Args:
            model_name: Name of the model to use (for local mode)

        Returns:
            SpeechRecognizer instance
        """
        if config.speech_recognition.use_remote_api:
            # Use unified OpenAI-compatible recognizer
            print(f"   Using remote API at {config.speech_recognition.remote_api_url}")
            return OpenAICompatibleRecognizer(
                url=config.speech_recognition.remote_api_url,
                model=config.speech_recognition.remote_api_model,
                use_multipart=config.speech_recognition.remote_api_use_multipart,
                api_key=config.speech_recognition.remote_api_key
            )
        else:
            # Use local Whisper
            print(f"   Using local Whisper model: {model_name}")
            return LocalWhisperRecognizer(model_name)


# Pre-configured recognizer instances for common use cases
class VerifierRecognizer:
    """Specialized recognizer for wake word verification."""

    def __init__(self):
        self.recognizer = SpeechRecognitionFactory.create_recognizer(
            config.speech_recognition.verifier_model
        )

    def transcribe(self, audio_data: np.ndarray, prompt: Optional[str] = None) -> str:
        """Transcribe audio for wake word verification."""
        return self.recognizer.transcribe_with_padding(audio_data, prompt)


class ScribeRecognizer:
    """Specialized recognizer for command transcription."""

    def __init__(self):
        self.recognizer = SpeechRecognitionFactory.create_recognizer(
            config.speech_recognition.scribe_model
        )

    def transcribe(self, audio_data: np.ndarray, prompt: Optional[str] = None) -> str:
        """Transcribe audio for command processing."""
        return self.recognizer.transcribe(audio_data, prompt)
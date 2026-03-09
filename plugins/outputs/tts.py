"""
Text-to-Speech module using Piper TTS engine with OpenAI-compatible TTS support.
Handles speech synthesis and audio playback with device-specific optimizations.
"""

import subprocess
import json
import sounddevice as sd
import numpy as np
import requests
import io
import scipy.io.wavfile
from typing import Optional
from core.config import config


class PiperTTS:
    """
    Text-to-Speech engine using Piper TTS.

    This class provides speech synthesis capabilities using the Piper TTS engine.
    It handles audio resampling, stereo conversion, and device-specific playback.
    """

    def __init__(self, binary_path: Optional[str] = None, voice_model: Optional[str] = None):
        """
        Initialize Piper TTS engine.

        Args:
            binary_path: Path to Piper binary executable
            voice_model: Path to Piper voice model file
        """
        self.binary_path = binary_path or config.tts.binary_path
        self.voice_model = voice_model or config.tts.voice_model
        self.voice_sample_rate = self._load_voice_sample_rate()

    def _load_voice_sample_rate(self) -> int:
        """
        Load the sample rate from the voice model JSON configuration.

        Returns:
            Sample rate of the voice model
        """
        config_path = self.voice_model + ".json"
        try:
            with open(config_path, 'r') as f:
                voice_config = json.load(f)
                return voice_config['audio']['sample_rate']
        except Exception as e:
            print(f"   ⚠️  Could not load voice config from {config_path}: {e}")
            print(f"   Assuming 22050Hz sample rate.")
            return 22050

    def synthesize(self, text: str) -> subprocess.Popen:
        """
        Start the speech synthesis process.

        Args:
            text: Text to synthesize

        Returns:
            subprocess.Popen object representing the Piper process
        """
        print(f"   [Mouth] Speaking: '{text}'")

        cmd = [
            self.binary_path,
            '--model', self.voice_model,
            '--output_raw'
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            process.stdin.write(text.encode('utf-8'))
            process.stdin.close()
            return process
        except Exception as e:
            raise RuntimeError(f"Failed to start TTS synthesis: {e}")

    def play_audio_stream(
        self,
        process: subprocess.Popen,
        output_device_id: Optional[int] = None,
        target_sample_rate: int = 48000
    ):
        """
        Play the synthesized audio stream with resampling and stereo conversion.

        Args:
            process: Piper process to read audio from
            output_device_id: Audio output device ID
            target_sample_rate: Target sample rate for playback
        """
        try:
            with sd.OutputStream(
                samplerate=target_sample_rate,
                device=output_device_id,
                channels=2,  # Stereo output
                dtype='int16'
            ) as stream:
                while True:
                    # Read raw audio data from Piper (mono)
                    raw_data = process.stdout.read(4096)
                    if not raw_data:
                        break

                    # Convert bytes to numpy array (int16, mono)
                    audio_source = np.frombuffer(raw_data, dtype=np.int16)

                    # Resample if necessary
                    if self.voice_sample_rate != target_sample_rate:
                        audio_resampled = self._resample_audio(
                            audio_source,
                            self.voice_sample_rate,
                            target_sample_rate
                        )
                    else:
                        audio_resampled = audio_source

                    # Convert mono to stereo (prevents chipmunk/fast playback issues)
                    audio_stereo = np.column_stack((audio_resampled, audio_resampled))

                    # Write to output stream
                    stream.write(audio_stereo)

            process.wait()

        except Exception as e:
            print(f"   ❌ TTS playback error: {e}")
            process.terminate()

    def _resample_audio(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio from source rate to target rate using linear interpolation.

        Args:
            audio: Input audio array (int16)
            source_rate: Source sample rate
            target_rate: Target sample rate

        Returns:
            Resampled audio array (int16)
        """
        if source_rate == target_rate:
            return audio

        num_source = len(audio)
        num_target = int(num_source * (target_rate / source_rate))

        # Create time arrays for interpolation
        x_old = np.linspace(0, num_source, num_source)
        x_new = np.linspace(0, num_source, num_target)

        # Linear interpolation and convert back to int16
        audio_resampled = np.interp(x_new, x_old, audio).astype(np.int16)
        return audio_resampled

    def speak(
        self,
        text: str,
        output_device_id: Optional[int] = None,
        target_sample_rate: int = 48000
    ):
        """
        Complete text-to-speech pipeline: synthesize and play audio.

        Args:
            text: Text to speak
            output_device_id: Audio output device ID
            target_sample_rate: Target sample rate for playback
        """
        try:
            process = self.synthesize(text)
            self.play_audio_stream(process, output_device_id, target_sample_rate)
        except Exception as e:
            print(f"   ❌ TTS Error: {e}")


class OpenAICompatibleTTS:
    """
    OpenAI-compatible TTS engine using HTTP API.

    This class provides speech synthesis capabilities using any OpenAI-compatible TTS server
    (like Kokoro). It handles audio retrieval from the server and local playback with
    device-specific optimizations.
    """

    def __init__(self, url: str, model: str = "kokoro", voice: str = "af_heart", response_format: str = "wav", api_key: str = ""):
        """
        Initialize OpenAI-compatible TTS engine.

        Args:
            url: Full API endpoint URL (including path, e.g., http://faceless:8880/v1/audio/speech)
            model: TTS model name (kokoro, tts-1, tts-1-hd)
            voice: Voice name to use for synthesis
            response_format: Audio format (mp3, opus, aac, flac, wav, pcm)
            api_key: Optional API key for Authorization header
        """
        self.url = url.rstrip('/')
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.api_key = api_key

        print(f"   Using OpenAI-compatible TTS API at {self.url}")
        print(f"   Model: {self.model}, Voice: {self.voice}, Format: {self.response_format}")

    def synthesize(self, text: str) -> tuple:
        """
        Get audio data from OpenAI-compatible TTS server.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (audio_data: np.ndarray, sample_rate: int)
        """
        try:
            print(f"   [Mouth] Speaking: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            # Prepare the OpenAI-compatible request
            payload = {
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "response_format": self.response_format
            }

            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.post(
                self.url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            # Read the audio data from the response
            audio_bytes = response.content

            # Handle different audio formats
            if self.response_format.lower() == 'wav':
                # Use io.BytesIO to read the WAV data
                audio_buffer = io.BytesIO(audio_bytes)
                # Suppress the warning about premature EOF - it's normal with streaming audio
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", scipy.io.wavfile.WavFileWarning)
                    sample_rate, audio_data = scipy.io.wavfile.read(audio_buffer)

                # Convert to int16 if needed
                if audio_data.dtype != np.int16:
                    audio_data = (audio_data * 32767).astype(np.int16)

                # Ensure mono audio by taking average if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1).astype(np.int16)

                return audio_data, sample_rate
            else:
                # For other formats (mp3, opus, etc.), we'd need additional processing
                # For now, we'll return raw bytes and indicate the format
                raise RuntimeError(f"Format {self.response_format} not yet implemented. Use 'wav' format.")

        except requests.exceptions.Timeout:
            print(f"   ⏰ TTS synthesis timeout")
            raise RuntimeError("TTS synthesis timeout")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ TTS request error: {e}")
            raise RuntimeError(f"TTS request error: {e}")
        except Exception as e:
            print(f"   ❌ TTS synthesis error: {e}")
            raise RuntimeError(f"TTS synthesis error: {e}")

    def play_audio_data(
        self,
        audio_data: np.ndarray,
        source_sample_rate: int,
        output_device_id: Optional[int] = None,
        target_sample_rate: int = 48000
    ):
        """
        Play audio data with resampling and stereo conversion.

        Args:
            audio_data: Audio data as numpy array
            source_sample_rate: Source sample rate of the audio
            output_device_id: Audio output device ID
            target_sample_rate: Target sample rate for playback
        """
        try:
            # Resample if necessary
            if source_sample_rate != target_sample_rate:
                audio_resampled = self._resample_audio(
                    audio_data,
                    source_sample_rate,
                    target_sample_rate
                )
            else:
                audio_resampled = audio_data

            # Convert mono to stereo (prevents chipmunk/fast playback issues)
            audio_stereo = np.column_stack((audio_resampled, audio_resampled))

            # Play the audio
            sd.play(
                audio_stereo,
                samplerate=target_sample_rate,
                device=output_device_id
            )
            sd.wait()  # Wait for playback to complete

        except Exception as e:
            print(f"   ❌ TTS playback error: {e}")
            raise

    def _resample_audio(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio from source rate to target rate using linear interpolation.

        Args:
            audio: Input audio array (int16)
            source_rate: Source sample rate
            target_rate: Target sample rate

        Returns:
            Resampled audio array (int16)
        """
        # Simple linear interpolation resampling
        duration = len(audio) / source_rate
        target_length = int(duration * target_rate)

        # Create time indices
        source_indices = np.linspace(0, len(audio) - 1, len(audio))
        target_indices = np.linspace(0, len(audio) - 1, target_length)

        # Linear interpolation
        resampled = np.interp(target_indices, source_indices, audio.astype(float))

        return resampled.astype(np.int16)

    def speak(
        self,
        text: str,
        output_device_id: Optional[int] = None,
        target_sample_rate: int = 48000
    ):
        """
        Complete text-to-speech pipeline: synthesize and play audio.

        Args:
            text: Text to speak
            output_device_id: Audio output device ID
            target_sample_rate: Target sample rate for playback
        """
        try:
            audio_data, sample_rate = self.synthesize(text)
            self.play_audio_data(audio_data, sample_rate, output_device_id, target_sample_rate)
        except Exception as e:
            print(f"   ❌ TTS Error: {e}")


class TTSManager:
    """
    High-level TTS manager that handles both local and remote TTS with fallback.

    This class manages the TTS process with automatic device configuration,
    optimal sample rate handling, and intelligent fallback between remote
    and local TTS.
    """

    def __init__(self, output_device_id: Optional[int] = None):
        """
        Initialize TTS manager.

        Args:
            output_device_id: Audio output device ID
        """
        self.output_device_id = output_device_id
        self.local_tts = None
        self.remote_tts = None

        # Initialize local TTS if enabled
        if config.tts.enabled:
            self.local_tts = PiperTTS()

        # Initialize remote TTS if enabled
        if config.remote_tts.use_remote_tts:
            print("   Initializing OpenAI-compatible TTS...")
            self.remote_tts = OpenAICompatibleTTS(
                url=config.remote_tts.remote_api_url,
                model=config.remote_tts.remote_tts_model,
                voice=config.remote_tts.remote_tts_voice,
                api_key=config.remote_tts.remote_api_key
            )

    def speak(self, text: str) -> None:
        """
        Speak text using TTS with intelligent fallback.

        Args:
            text: Text to speak
        """
        # Try remote TTS first if available
        if self.remote_tts:
            try:
                self.remote_tts.speak(text, self.output_device_id)
                return
            except Exception as e:
                # Fallback to local TTS if remote fails
                if self.local_tts:
                    print(f"   ⚠️  Remote TTS failed, falling back to local: {e}")
                else:
                    print(f"   ❌ Remote TTS failed and no local fallback available: {e}")
                    return

        # Use local TTS (either as primary or fallback)
        if self.local_tts:
            try:
                self.local_tts.speak(text, self.output_device_id)
            except Exception as e:
                print(f"   ❌ Local TTS failed: {e}")
        else:
            print("   ⚠️  No TTS available - both remote and local TTS are disabled")
"""
Voice Input Plugin - Wake word detection, VAD, and STT pipeline.

This plugin handles the complete voice input flow:
1. Wake word detection (openWakeWord)
2. Phonetic verification (jellyfish)
3. Voice activity detection (Silero VAD)
4. Speech-to-text transcription (faster-whisper)

The audio pipeline runs in a dedicated thread via sounddevice callback.
State changes and transcription happen in spawned threads to keep
the audio callback fast (<10ms).
"""

import collections
import threading
import numpy as np
import sounddevice as sd
from typing import Optional, Dict, Any

from core.plugin_base import InputPlugin
from core.config import config

# Local imports from voice plugin package
from plugins.inputs.voice.audio_device_manager import AudioDeviceManager
from plugins.inputs.voice.speech_recognition import VerifierRecognizer, ScribeRecognizer
from plugins.inputs.voice.vad import SileroVAD, VADProcessor
from plugins.inputs.voice.wake_word_detector import WakeWordDetector
from plugins.inputs.voice.phonetic_verification import WakeWordVerifier
from plugins.inputs.voice.audio_utils import AudioDebugger


class VoiceInputPlugin(InputPlugin):
    """
    Voice input plugin handling wake word, VAD, and STT.

    States:
    - WAITING: Listening for wake word
    - VERIFYING: Verifying wake word via phonetic matching
    - LISTENING: Recording command with VAD
    - PTT: Push-to-talk recording (no VAD end detection)
    - PROCESSING_VAD: Processing VAD-triggered input
    - PROCESSING_PTT: Processing PTT-triggered input

    Endpoints registered:
    - voice.wake_requested: Skip wake word and start listening
    - voice.ptt_started: Start push-to-talk recording
    - voice.ptt_stopped: Stop PTT and process
    """

    def __init__(self, engine: 'AssistantEngine', plugin_config: Dict[str, Any] = None):
        """
        Initialize voice input plugin.

        Args:
            engine: Reference to the engine
            plugin_config: Optional configuration overrides
        """
        super().__init__(engine, "voice")
        self.plugin_config = plugin_config or {}

        # State machine
        self._state = "WAITING"
        self._state_lock = threading.Lock()

        # Audio buffers
        self.preroll_buffer = collections.deque(
            maxlen=int(1.5 * (config.audio_devices.sample_rate / config.audio_devices.chunk_size))
        )
        self.command_buffer = []

        # Verification state
        self._verification_preroll = None
        self._post_trigger_buffer = []

        # Audio stream
        self.audio_stream = None
        self.mic_id = None

        # Components (initialized in start())
        self.wake_word_detector = None
        self.vad_model = None
        self.vad_processor = None
        self.wake_word_verifier = None
        self.verifier = None  # For wake word verification transcription
        self.scribe = None    # For command transcription
        self.audio_debugger = None

        # Running state
        self.running = False

    @property
    def state(self):
        """Get current state (thread-safe)."""
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, value):
        """Set current state (thread-safe)."""
        with self._state_lock:
            self._state = value

    def start(self):
        """
        Start the voice input plugin.

        Initializes all audio components and starts the audio stream.
        """
        print("🎙️  Starting voice input plugin...")
        self.running = True

        # Setup audio device
        print("   [1/6] Setting up audio device...")
        device_manager = AudioDeviceManager()
        self.mic_id = device_manager.get_input_device_id()

        # Initialize components
        print("   [2/6] Loading speech recognizers...")
        self.verifier = VerifierRecognizer()
        self.scribe = ScribeRecognizer()

        print("   [3/6] Loading wake word detector...")
        self.wake_word_detector = WakeWordDetector()

        print("   [4/6] Loading VAD...")
        self.vad_model = SileroVAD(
            model_path=config.vad.model_path
        )
        self.vad_processor = VADProcessor(
            self.vad_model,
            silence_threshold=config.vad.silence_threshold
        )

        print("   [5/6] Loading phonetic verifier...")
        self.wake_word_verifier = WakeWordVerifier()

        print("   [6/6] Initializing audio debugger...")
        self.audio_debugger = AudioDebugger()

        # Register endpoints for button/external control
        self.engine.register_endpoint("voice", "wake_requested", self._on_wake_requested)
        self.engine.register_endpoint("voice", "ptt_started", self._on_ptt_started)
        self.engine.register_endpoint("voice", "ptt_stopped", self._on_ptt_stopped)
        self.engine.register_endpoint("voice", "get_state", self._on_get_state)

        # Start audio stream
        self.audio_stream = sd.InputStream(
            device=self.mic_id,
            channels=1,
            samplerate=config.audio_devices.mic_sample_rate,
            blocksize=config.audio_devices.chunk_size * config.audio_devices.downsample_factor,
            callback=self._audio_callback
        )
        self.audio_stream.start()

        print(f"✅ Voice input ready. Waiting for '{config.wake_word.target}'...")

    def stop(self):
        """Stop the voice input plugin."""
        print("🛑 Stopping voice input plugin...")
        self.running = False

        # Stop audio stream
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

        print("   Voice input stopped")

    # =========================================================================
    # Audio Callback (Runs in Audio Thread - Must Be Fast!)
    # =========================================================================

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Audio stream callback - processes incoming audio chunks.

        CRITICAL: Must return quickly (<10ms). Spawns threads for slow operations.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timestamp information
            status: Stream status
        """
        if not self.running:
            return

        # Downsample audio to target sample rate
        audio_16k = indata[::config.audio_devices.downsample_factor]
        self.preroll_buffer.append(audio_16k.copy())

        current_state = self.state

        if current_state == "WAITING":
            # Check for wake word
            audio_int16 = (audio_16k * 32767).astype(np.int16).squeeze()
            if self.wake_word_detector.get_max_confidence(audio_int16) > config.wake_word.confidence_threshold:
                # Snapshot preroll now before verification
                self._verification_preroll = list(self.preroll_buffer)
                print(f"\n[⚡] Trigger! Verifying... (preroll snapshot taken)")
                self.state = "VERIFYING"
                self._notify_led_state()

                # Spawn verification thread (don't block audio!)
                threading.Thread(
                    target=self._handle_verification,
                    daemon=True
                ).start()

        elif current_state == "VERIFYING":
            # Collect audio while verifying to avoid losing speech
            self._post_trigger_buffer.append(audio_16k.copy())

        elif current_state == "LISTENING":
            # Collect command audio with VAD
            self.command_buffer.append(audio_16k.copy())
            self.vad_processor.process_chunk(audio_16k)

            # Check for end of speech
            secs_recorded = len(self.command_buffer) * (
                config.audio_devices.chunk_size / config.audio_devices.sample_rate
            )
            silence_duration = self.vad_processor.get_silence_duration(
                config.audio_devices.chunk_size,
                config.audio_devices.sample_rate
            )

            if secs_recorded > 2.0 and silence_duration > 1.0:
                print(" [End of Speech]")
                self.state = "PROCESSING_VAD"
                self._notify_led_state()

                # Spawn processing thread
                threading.Thread(
                    target=self._finalize_input,
                    args=('VAD',),
                    daemon=True
                ).start()

        elif current_state == "PTT":
            # Push-to-talk: collect audio without VAD end detection
            self.command_buffer.append(audio_16k.copy())

    # =========================================================================
    # State Handlers (Run in Spawned Threads)
    # =========================================================================

    def _handle_verification(self):
        """Handle wake word verification (runs in spawned thread)."""
        print("   [Judge] Verifying wake word...")

        # Get preroll for verification
        preroll_source = self._verification_preroll or list(self.preroll_buffer)
        audio = np.concatenate(preroll_source).flatten()

        # Transcribe for verification
        text = self._transcribe_audio(audio, is_verification=True)
        print(f"   [Judge] Heard: '{text}'")

        # Verify using phonetic matching
        if self.wake_word_verifier.verify_wake_word(text):
            print(f"   ✅ VERIFIED!")

            # Transition to listening state
            self.state = "LISTENING"
            self._notify_led_state()

            # Preserve audio: preroll + post-trigger buffer
            post = list(self._post_trigger_buffer) if self._post_trigger_buffer else []
            self.command_buffer = list(preroll_source) + post
            self.vad_processor.reset()
        else:
            print("   ❌ Rejected.")
            self.state = "WAITING"
            self._notify_led_state()
            self.wake_word_detector.reset()

        # Clear verification buffers
        self._verification_preroll = None
        self._post_trigger_buffer = []

    def _finalize_input(self, source: str):
        """
        Finalize input and emit to engine (runs in spawned thread).

        Args:
            source: Input source ('VAD' or 'PTT')
        """
        print(f"   [Scribe] Transcribing command ({source})...")

        # Concatenate command buffer
        audio = np.concatenate(self.command_buffer).flatten()
        text = self._transcribe_audio(audio, is_verification=False)

        print(f"\n   🗣️  COMMAND ({source}): \"{text}\"")

        # Emit input to engine (thread-safe via queue)
        self.emit_input(text, {'source': source})

        # Reset state
        self.state = "WAITING"
        self._notify_led_state()
        self.wake_word_detector.reset()
        self.command_buffer = []

        self.print("✅ Ready.")

    def _transcribe_audio(self, audio: np.ndarray, is_verification: bool = False) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array
            is_verification: True if this is wake word verification

        Returns:
            Transcribed text
        """
        try:
            # Normalize audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Add padding for verification (helps with short utterances)
            if is_verification:
                padding = np.zeros(int(0.5 * config.audio_devices.sample_rate), dtype=np.float32)
                audio = np.concatenate([padding, audio, padding])
                # Use vocab_prompt to guide Whisper toward expected wake word variants
                vocab_prompt = config.wake_word.vocab_prompt
                text = self.verifier.transcribe(audio, prompt=vocab_prompt)
            else:
                # Use speech recognition vocab_prompt for domain-specific vocabulary
                vocab_prompt = config.speech_recognition.vocab_prompt
                text = self.scribe.transcribe(audio, prompt=vocab_prompt if vocab_prompt else None)

            return text.strip() if text else ""

        except Exception as e:
            print(f"   ❌ Transcription error: {e}")
            return ""

    # =========================================================================
    # Endpoint Handlers (Called by Engine Thread)
    # =========================================================================

    def _on_wake_requested(self, data: Dict[str, Any]) -> None:
        """
        Handle wake request from button/external source.

        Skips wake word verification and goes directly to LISTENING.

        Args:
            data: Endpoint data (e.g., {'source': 'button'})
        """
        if self.state == "WAITING":
            source = data.get('source', 'external')
            print(f"\n[🔘 {source}] Wake triggered - skipping to LISTENING!")

            # Snapshot preroll and transition to listening
            self._verification_preroll = list(self.preroll_buffer)
            self.state = "LISTENING"

            # Pre-populate command buffer with preroll
            self.command_buffer = list(self._verification_preroll)
            self.vad_processor.reset()
            self._notify_led_state()

    def _on_ptt_started(self, data: Dict[str, Any]) -> None:
        """
        Handle push-to-talk start.

        Args:
            data: Endpoint data
        """
        if self.state == "WAITING":
            print("\n[🔘 Button] PTT started (hold to speak)")
            self.command_buffer = []
            self.state = "PTT"
            self._notify_led_state()
            self.vad_processor.reset()

    def _on_ptt_stopped(self, data: Dict[str, Any]) -> None:
        """
        Handle push-to-talk stop.

        Args:
            data: Endpoint data
        """
        if self.state == "PTT":
            print("\n[🔘 Button] PTT released - processing...")
            self.state = "PROCESSING_PTT"
            self._notify_led_state()

            # Spawn processing thread
            threading.Thread(
                target=self._finalize_input,
                args=('PTT',),
                daemon=True
            ).start()

    def _on_get_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get current voice plugin state.

        Args:
            data: Endpoint data (unused)

        Returns:
            Current state information
        """
        return {
            'state': self.state,
            'running': self.running,
            'command_buffer_size': len(self.command_buffer)
        }

    # =========================================================================
    # LED State Notification
    # =========================================================================

    def _notify_led_state(self):
        """
        Notify LED plugin of state change via endpoint.

        Fire-and-forget - doesn't block if LED plugin isn't registered.
        """
        if self.engine.has_endpoint("led", "set_state"):
            self.engine.endpoint_send("led", "set_state", {'state': self.state})
        if self.engine.has_endpoint("web", "state_update"):
            self.engine.endpoint_send("web", "state_update", {'state': self.state})

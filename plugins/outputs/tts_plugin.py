"""
TTS Output Plugin - Text-to-speech output with queued processing.

Uses QueuedOutputPlugin to preserve fragment ordering and prevent
the engine thread from blocking on audio playback.
"""

from typing import Dict, Any

from core.plugin_base import QueuedOutputPlugin
from plugins.outputs.tts import TTSManager
from plugins.outputs.tts_formatter import format_for_tts


class TTSOutputPlugin(QueuedOutputPlugin):
    """
    Text-to-speech output plugin.

    Converts response fragments to speech using TTSManager.
    Uses queued processing so:
    - Engine thread returns immediately (non-blocking)
    - Fragments are spoken in order
    - Long TTS operations don't delay other processing

    Only speaks for voice/button sources by default.
    """

    def __init__(self, engine: 'AssistantEngine', plugin_config: Dict[str, Any] = None):
        """
        Initialize TTS output plugin.

        Args:
            engine: Reference to the engine
            plugin_config: Optional configuration overrides
        """
        super().__init__(engine, "tts")
        self.plugin_config = plugin_config or {}
        self.tts_manager = None

        # Configure which sources trigger TTS
        self.speak_sources = self.plugin_config.get(
            'speak_sources',
            ['VAD', 'PTT', 'BUTTON', 'WAKE_TIMER']  # Default: voice, button, and scheduled wakes
        )

    def start_internal(self):
        """Initialize TTS manager."""
        print("🔊 Starting TTS output plugin...")

        # Get output device from config if available
        output_device_id = self.plugin_config.get('output_device_id', None)

        # Initialize TTS manager
        self.tts_manager = TTSManager(output_device_id)

        print("✅ TTS output plugin ready")

    def _process_output(self, text: str, metadata: Dict[str, Any]):
        """
        Speak text fragment (runs in worker thread, can block).

        Args:
            text: Response text/fragment to speak
            metadata: Output metadata
        """
        # Format text for TTS (remove action descriptions, formatting, etc.)
        tts_text = format_for_tts(text)

        if not tts_text or not tts_text.strip():
            return  # Skip empty text after formatting

        try:
            # Log what we're speaking (truncated for readability)
            display_text = tts_text[:50] + '...' if len(tts_text) > 50 else tts_text
            print(f"   🔊 SPEAKING: \"{display_text}\"")

            # Notify LED we're speaking
            if self.engine.has_endpoint("led", "set_state"):
                self.engine.endpoint_send("led", "set_state", {'state': 'SPEAKING'})

            # Speak the text (blocks until done)
            self.tts_manager.speak(tts_text)

            # Restore LED to waiting state after speaking
            if self.engine.has_endpoint("led", "set_state"):
                self.engine.endpoint_send("led", "set_state", {'state': 'WAITING'})

        except Exception as e:
            print(f"   ❌ TTS error: {e}")
            # Try to restore LED state even on error
            if self.engine.has_endpoint("led", "set_state"):
                self.engine.endpoint_send("led", "set_state", {'state': 'WAITING'})

    def should_handle(self, metadata: Dict[str, Any]) -> bool:
        """
        Determine if this output should be spoken.

        Only speaks for configured sources (default: voice/button).
        Text input typically goes to console only.

        Args:
            metadata: Output metadata

        Returns:
            True if TTS should handle this output
        """
        if metadata.get('is_thinking'):
            return False
        source = metadata.get('source', '')
        return source in self.speak_sources

    def stop_internal(self):
        """Cleanup TTS resources."""
        print("🛑 TTS output plugin stopped")
        # TTSManager doesn't need explicit cleanup
        self.tts_manager = None

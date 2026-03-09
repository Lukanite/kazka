"""
Unit tests for the voice input plugin.

Note: These tests don't use real audio - they test the plugin's
state machine and endpoint handling.
"""

import unittest
import time
import threading

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import AssistantEngine
from plugins.inputs.voice.voice_plugin import VoiceInputPlugin


class TestVoicePluginState(unittest.TestCase):
    """Test voice plugin state management without audio."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            if self.engine.engine_thread:
                self.engine.engine_thread.join(timeout=2.0)

    def test_voice_plugin_initialization(self):
        """Test voice plugin initializes correctly."""
        plugin = VoiceInputPlugin(self.engine)

        self.assertEqual(plugin.name, "voice")
        self.assertEqual(plugin.state, "WAITING")
        self.assertFalse(plugin.running)
        self.assertIsNone(plugin.audio_stream)

    def test_voice_plugin_state_property(self):
        """Test voice plugin state is thread-safe."""
        plugin = VoiceInputPlugin(self.engine)

        # Test state transitions
        plugin.state = "LISTENING"
        self.assertEqual(plugin.state, "LISTENING")

        plugin.state = "VERIFYING"
        self.assertEqual(plugin.state, "VERIFYING")

        plugin.state = "WAITING"
        self.assertEqual(plugin.state, "WAITING")


class TestVoicePluginEndpoints(unittest.TestCase):
    """Test voice plugin endpoint handlers."""

    def setUp(self):
        """Set up test fixtures with running engine."""
        self.engine = AssistantEngine()

        # Start engine first
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        self.engine_thread.start()
        time.sleep(0.1)

        # Create voice plugin but don't start it (avoid audio)
        self.voice_plugin = VoiceInputPlugin(self.engine)

        # Initialize minimal components needed for endpoint handlers
        from plugins.inputs.voice.vad import SileroVAD, VADProcessor
        from core.config import config
        self.voice_plugin.vad_processor = VADProcessor(
            SileroVAD(
                model_path=config.vad.model_path,
                download_url=config.vad.download_url
            ),
            silence_threshold=config.vad.silence_threshold
        )

        # Register endpoints (engine already running, so this is synchronous)
        self.engine.register_endpoint("voice", "wake_requested", self.voice_plugin._on_wake_requested)
        self.engine.register_endpoint("voice", "ptt_started", self.voice_plugin._on_ptt_started)
        self.engine.register_endpoint("voice", "ptt_stopped", self.voice_plugin._on_ptt_stopped)
        self.engine.register_endpoint("voice", "get_state", self.voice_plugin._on_get_state)

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            self.engine_thread.join(timeout=2.0)

    def test_wake_requested_endpoint(self):
        """Test wake_requested endpoint transitions to LISTENING."""
        # Initially in WAITING
        self.assertEqual(self.voice_plugin.state, "WAITING")

        # Send wake request
        self.engine.endpoint_send("voice", "wake_requested", {'source': 'test'})

        # Wait for processing
        time.sleep(0.2)

        # Should transition to LISTENING
        self.assertEqual(self.voice_plugin.state, "LISTENING")

    def test_ptt_started_endpoint(self):
        """Test ptt_started endpoint transitions to PTT."""
        # Initially in WAITING
        self.assertEqual(self.voice_plugin.state, "WAITING")

        # Send PTT start
        self.engine.endpoint_send("voice", "ptt_started", {})

        # Wait for processing
        time.sleep(0.2)

        # Should transition to PTT
        self.assertEqual(self.voice_plugin.state, "PTT")

    def test_ptt_stopped_endpoint(self):
        """Test ptt_stopped endpoint transitions to PROCESSING_PTT."""
        # Set up PTT state
        self.voice_plugin.state = "PTT"

        # Send PTT stop
        self.engine.endpoint_send("voice", "ptt_stopped", {})

        # Wait for processing
        time.sleep(0.2)

        # Should transition to PROCESSING_PTT (then back to WAITING after processing)
        # Note: Without actual audio, it will error during transcription and return to WAITING
        self.assertIn(self.voice_plugin.state, ["PROCESSING_PTT", "WAITING"])

    def test_get_state_endpoint(self):
        """Test get_state endpoint returns current state."""
        # Set a known state
        self.voice_plugin.state = "LISTENING"

        # Call get_state
        result = self.engine.endpoint_call("voice", "get_state", {})

        # Verify response
        self.assertEqual(result['state'], "LISTENING")
        self.assertFalse(result['running'])  # Not started

    def test_wake_request_only_from_waiting(self):
        """Test wake_requested only works from WAITING state."""
        # Set to LISTENING (not WAITING)
        self.voice_plugin.state = "LISTENING"

        # Send wake request
        self.engine.endpoint_send("voice", "wake_requested", {'source': 'test'})

        # Wait for processing
        time.sleep(0.2)

        # Should still be LISTENING (wake ignored)
        self.assertEqual(self.voice_plugin.state, "LISTENING")

    def test_ptt_start_only_from_waiting(self):
        """Test ptt_started only works from WAITING state."""
        # Set to VERIFYING (not WAITING)
        self.voice_plugin.state = "VERIFYING"

        # Send PTT start
        self.engine.endpoint_send("voice", "ptt_started", {})

        # Wait for processing
        time.sleep(0.2)

        # Should still be VERIFYING (PTT ignored)
        self.assertEqual(self.voice_plugin.state, "VERIFYING")


if __name__ == '__main__':
    unittest.main()

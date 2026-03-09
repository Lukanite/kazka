"""
Unit tests for the TTS output plugin.

Note: These tests mock the actual TTS to avoid audio playback.
"""

import unittest
import time
import threading

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import AssistantEngine
from plugins.outputs.tts_plugin import TTSOutputPlugin
from plugins.outputs.tts_formatter import format_for_tts


class TestTTSFormatter(unittest.TestCase):
    """Test TTS text formatting."""

    def test_format_removes_brackets(self):
        """Test that bracket actions are removed."""
        text = "Hello [pauses dramatically] there"
        result = format_for_tts(text)
        self.assertNotIn('[', result)
        self.assertNotIn(']', result)
        self.assertIn('Hello', result)
        self.assertIn('there', result)

    def test_format_removes_parentheses(self):
        """Test that parenthetical actions are removed."""
        text = "I see (nods slowly) what you mean"
        result = format_for_tts(text)
        self.assertNotIn('(', result)
        self.assertNotIn(')', result)
        self.assertIn('I see', result)
        self.assertIn('what you mean', result)

    def test_format_removes_asterisks(self):
        """Test that asterisk emphasis markers are removed."""
        text = "This is *very* important"
        result = format_for_tts(text)
        self.assertNotIn('*', result)
        self.assertIn('very', result)
        self.assertIn('important', result)

    def test_format_removes_quotes(self):
        """Test that quote marks are removed."""
        text = 'She said "hello" to everyone'
        result = format_for_tts(text)
        # Quotes should be removed
        self.assertIn('hello', result)
        self.assertIn('She said', result)

    def test_format_empty_string(self):
        """Test that empty string returns empty."""
        result = format_for_tts("")
        self.assertEqual(result, "")

    def test_format_preserves_normal_text(self):
        """Test that normal text is preserved."""
        text = "The quick brown fox jumps over the lazy dog."
        result = format_for_tts(text)
        self.assertEqual(result, text)


class TestTTSPluginShouldHandle(unittest.TestCase):
    """Test TTS plugin source filtering."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()
        self.tts_plugin = TTSOutputPlugin(self.engine)

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            if self.engine.engine_thread:
                self.engine.engine_thread.join(timeout=2.0)

    def test_should_handle_vad_source(self):
        """Test that VAD source is handled."""
        metadata = {'source': 'VAD'}
        self.assertTrue(self.tts_plugin.should_handle(metadata))

    def test_should_handle_ptt_source(self):
        """Test that PTT source is handled."""
        metadata = {'source': 'PTT'}
        self.assertTrue(self.tts_plugin.should_handle(metadata))

    def test_should_handle_button_source(self):
        """Test that BUTTON source is handled."""
        metadata = {'source': 'BUTTON'}
        self.assertTrue(self.tts_plugin.should_handle(metadata))

    def test_should_handle_wake_timer_source(self):
        """Test that WAKE_TIMER source is handled."""
        metadata = {'source': 'WAKE_TIMER'}
        self.assertTrue(self.tts_plugin.should_handle(metadata))

    def test_should_not_handle_text_source(self):
        """Test that TEXT source is not handled by default."""
        metadata = {'source': 'TEXT'}
        self.assertFalse(self.tts_plugin.should_handle(metadata))

    def test_should_not_handle_empty_source(self):
        """Test that empty source is not handled."""
        metadata = {'source': ''}
        self.assertFalse(self.tts_plugin.should_handle(metadata))

    def test_should_not_handle_missing_source(self):
        """Test that missing source is not handled."""
        metadata = {}
        self.assertFalse(self.tts_plugin.should_handle(metadata))

    def test_custom_speak_sources(self):
        """Test that custom speak sources can be configured."""
        custom_plugin = TTSOutputPlugin(self.engine, {'speak_sources': ['TEXT', 'API']})

        # Now TEXT should be handled
        self.assertTrue(custom_plugin.should_handle({'source': 'TEXT'}))
        self.assertTrue(custom_plugin.should_handle({'source': 'API'}))

        # But VAD should not
        self.assertFalse(custom_plugin.should_handle({'source': 'VAD'}))


class TestTTSPluginLEDNotification(unittest.TestCase):
    """Test TTS plugin sends LED state notifications during speech."""

    def setUp(self):
        """Set up test fixtures with mocked TTS manager."""
        self.engine = AssistantEngine()
        self.tts_plugin = TTSOutputPlugin(self.engine)

        # Track endpoint_send calls
        self.endpoint_calls = []
        original_endpoint_send = self.engine.endpoint_send

        def mock_endpoint_send(target, endpoint, data):
            self.endpoint_calls.append({
                'target': target,
                'endpoint': endpoint,
                'data': data.copy() if data else {}
            })
            # Still call original for any registered handlers
            return original_endpoint_send(target, endpoint, data)

        self.engine.endpoint_send = mock_endpoint_send

        # Start engine and plugin
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        self.engine_thread.start()
        time.sleep(0.1)

        self.tts_plugin.start()
        time.sleep(0.1)

        # Mock the TTS manager's speak method to avoid actual audio
        self.speak_calls = []
        def mock_speak(text):
            self.speak_calls.append(text)
            time.sleep(0.05)  # Simulate short speech time

        self.tts_plugin.tts_manager.speak = mock_speak

    def tearDown(self):
        """Clean up after tests."""
        self.tts_plugin.stop()

        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            self.engine_thread.join(timeout=2.0)

    def test_led_speaking_state_sent_before_speech(self):
        """Test that LED is set to SPEAKING before TTS starts."""
        self.endpoint_calls.clear()

        # Queue output (this goes to worker thread)
        self.tts_plugin.output("Hello world", {'source': 'VAD'})

        # Wait for worker to process
        time.sleep(0.3)

        # Find LED calls
        led_calls = [c for c in self.endpoint_calls
                     if c['target'] == 'led' and c['endpoint'] == 'set_state']

        self.assertGreaterEqual(len(led_calls), 1)
        # First LED call should be SPEAKING
        self.assertEqual(led_calls[0]['data']['state'], 'SPEAKING')

    def test_led_waiting_state_sent_after_speech(self):
        """Test that LED is set to WAITING after TTS completes."""
        self.endpoint_calls.clear()

        # Queue output
        self.tts_plugin.output("Hello world", {'source': 'VAD'})

        # Wait for worker to process
        time.sleep(0.3)

        # Find LED calls
        led_calls = [c for c in self.endpoint_calls
                     if c['target'] == 'led' and c['endpoint'] == 'set_state']

        self.assertGreaterEqual(len(led_calls), 2)
        # Last LED call should be WAITING
        self.assertEqual(led_calls[-1]['data']['state'], 'WAITING')

    def test_led_state_order_speaking_then_waiting(self):
        """Test that LED states are sent in correct order: SPEAKING then WAITING."""
        self.endpoint_calls.clear()

        # Queue output
        self.tts_plugin.output("Test message", {'source': 'PTT'})

        # Wait for worker to process
        time.sleep(0.3)

        # Find LED calls in order
        led_states = [c['data']['state'] for c in self.endpoint_calls
                      if c['target'] == 'led' and c['endpoint'] == 'set_state']

        self.assertEqual(led_states, ['SPEAKING', 'WAITING'])

    def test_led_restored_on_tts_error(self):
        """Test that LED is restored to WAITING even if TTS fails."""
        self.endpoint_calls.clear()

        # Make TTS manager throw an exception
        def failing_speak(text):
            raise RuntimeError("TTS failed")

        self.tts_plugin.tts_manager.speak = failing_speak

        # Queue output
        self.tts_plugin.output("This will fail", {'source': 'BUTTON'})

        # Wait for worker to process
        time.sleep(0.3)

        # Find LED calls
        led_calls = [c for c in self.endpoint_calls
                     if c['target'] == 'led' and c['endpoint'] == 'set_state']

        # Should have both SPEAKING (before attempt) and WAITING (after error)
        led_states = [c['data']['state'] for c in led_calls]
        self.assertIn('SPEAKING', led_states)
        self.assertIn('WAITING', led_states)
        # Last state should be WAITING
        self.assertEqual(led_states[-1], 'WAITING')

    def test_no_led_notification_for_empty_text(self):
        """Test that no LED notification is sent for empty text after formatting."""
        self.endpoint_calls.clear()

        # Queue empty text (or text that formats to empty)
        self.tts_plugin.output("", {'source': 'VAD'})
        self.tts_plugin.output("   ", {'source': 'VAD'})

        # Wait for worker to process
        time.sleep(0.3)

        # Should have no LED calls since text was empty
        led_calls = [c for c in self.endpoint_calls
                     if c['target'] == 'led' and c['endpoint'] == 'set_state']

        self.assertEqual(len(led_calls), 0)


class TestTTSPluginLifecycle(unittest.TestCase):
    """Test TTS plugin start/stop lifecycle."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            if self.engine.engine_thread:
                self.engine.engine_thread.join(timeout=2.0)

    def test_plugin_initializes_without_tts_manager(self):
        """Test that plugin initializes without TTS manager (lazy init)."""
        plugin = TTSOutputPlugin(self.engine)
        self.assertIsNone(plugin.tts_manager)

    def test_plugin_start_initializes_tts_manager(self):
        """Test that start() initializes the TTS manager."""
        plugin = TTSOutputPlugin(self.engine)

        # Start the plugin
        plugin.start()

        # TTS manager should now be initialized
        self.assertIsNotNone(plugin.tts_manager)

        # Stop the plugin
        plugin.stop()

    def test_plugin_stop_cleans_up(self):
        """Test that stop() cleans up resources."""
        plugin = TTSOutputPlugin(self.engine)
        plugin.start()

        # Stop the plugin
        plugin.stop()

        # TTS manager should be cleaned up
        self.assertIsNone(plugin.tts_manager)


if __name__ == '__main__':
    unittest.main()

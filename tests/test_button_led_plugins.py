"""
Unit tests for button and LED plugins.

Note: These tests mock GPIO hardware to work on any system.
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import AssistantEngine


class MockLEDController:
    """Mock LED controller for testing without GPIO."""

    def __init__(self, led_pin=25):
        self.led_pin = led_pin
        self.current_state = "OFF"
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def set_state(self, state):
        self.current_state = state


class MockButtonController:
    """Mock button controller for testing without GPIO."""

    def __init__(self, button_pin=23):
        self.button_pin = button_pin
        self._short_press_callback = None
        self._hold_start_callback = None
        self._release_callback = None
        self.running = False

    def on_short_press(self, callback):
        self._short_press_callback = callback

    def on_hold_start(self, callback):
        self._hold_start_callback = callback

    def on_release(self, callback):
        self._release_callback = callback

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def simulate_short_press(self):
        if self._short_press_callback:
            self._short_press_callback()

    def simulate_hold_start(self):
        if self._hold_start_callback:
            self._hold_start_callback()

    def simulate_release(self):
        if self._release_callback:
            self._release_callback()


class TestButtonInputPlugin(unittest.TestCase):
    """Test button input plugin with mocked GPIO."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

        # Start engine
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        self.engine_thread.start()
        time.sleep(0.1)

        # Track endpoint calls
        self.wake_requests = []
        self.ptt_starts = []
        self.ptt_stops = []

        # Register mock voice endpoints
        self.engine.register_endpoint("voice", "wake_requested",
                                       lambda data: self.wake_requests.append(data))
        self.engine.register_endpoint("voice", "ptt_started",
                                       lambda data: self.ptt_starts.append(data))
        self.engine.register_endpoint("voice", "ptt_stopped",
                                       lambda data: self.ptt_stops.append(data))

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            self.engine_thread.join(timeout=2.0)

    @patch('plugins.inputs.button.button_plugin.ButtonController', MockButtonController)
    def test_button_plugin_initialization(self):
        """Test button plugin initializes correctly."""
        from plugins.inputs.button.button_plugin import ButtonInputPlugin

        plugin = ButtonInputPlugin(self.engine)

        self.assertEqual(plugin.name, "button")
        self.assertFalse(plugin.running)
        self.assertIsNone(plugin.controller)

    @patch('plugins.inputs.button.button_plugin.ButtonController', MockButtonController)
    def test_button_plugin_start_stop(self):
        """Test button plugin start/stop lifecycle."""
        from plugins.inputs.button.button_plugin import ButtonInputPlugin

        plugin = ButtonInputPlugin(self.engine)

        # Start
        plugin.start()
        self.assertTrue(plugin.running)
        self.assertIsNotNone(plugin.controller)

        # Stop
        plugin.stop()
        self.assertFalse(plugin.running)
        self.assertIsNone(plugin.controller)

    @patch('plugins.inputs.button.button_plugin.ButtonController', MockButtonController)
    def test_short_press_sends_wake_request_on_release(self):
        """Test short press sends wake request on release (to avoid button click noise)."""
        from plugins.inputs.button.button_plugin import ButtonInputPlugin

        plugin = ButtonInputPlugin(self.engine)
        plugin.start()

        # Simulate short press (just sets flag, wake happens on release)
        plugin.controller.simulate_short_press()

        # Wait briefly - wake request should NOT be sent yet
        time.sleep(0.1)
        self.assertEqual(len(self.wake_requests), 0, "Wake should not trigger on press, only on release")

        # Now simulate release - this triggers the wake request
        plugin.controller.simulate_release()

        # Wait for endpoint processing
        time.sleep(0.2)

        # Verify wake request was sent on release
        self.assertGreaterEqual(len(self.wake_requests), 1)
        self.assertEqual(self.wake_requests[0]['source'], 'button')

        plugin.stop()

    @patch('plugins.inputs.button.button_plugin.ButtonController', MockButtonController)
    def test_hold_sends_ptt_start(self):
        """Test hold sends PTT start to voice plugin."""
        from plugins.inputs.button.button_plugin import ButtonInputPlugin

        plugin = ButtonInputPlugin(self.engine)
        plugin.start()

        # Simulate hold
        plugin.controller.simulate_hold_start()

        # Wait for endpoint processing
        time.sleep(0.2)

        # Verify PTT start was sent
        self.assertGreaterEqual(len(self.ptt_starts), 1)
        self.assertEqual(self.ptt_starts[0]['source'], 'button')

        plugin.stop()

    @patch('plugins.inputs.button.button_plugin.ButtonController', MockButtonController)
    def test_release_after_hold_sends_ptt_stop(self):
        """Test release after hold sends PTT stop to voice plugin."""
        from plugins.inputs.button.button_plugin import ButtonInputPlugin

        plugin = ButtonInputPlugin(self.engine)
        plugin.start()

        # Simulate hold start first (this clears short press flag)
        plugin.controller.simulate_hold_start()
        time.sleep(0.1)

        # Clear PTT starts list to only check for stops
        initial_ptt_starts = len(self.ptt_starts)

        # Simulate release after hold
        plugin.controller.simulate_release()

        # Wait for endpoint processing
        time.sleep(0.2)

        # Verify PTT stop was sent (not a wake request)
        self.assertGreaterEqual(len(self.ptt_stops), 1)
        # Wake requests should still be 0 since this was a hold, not short press
        self.assertEqual(len(self.wake_requests), 0)

        plugin.stop()


class TestLEDOutputPlugin(unittest.TestCase):
    """Test LED output plugin with mocked GPIO."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            if self.engine.engine_thread:
                self.engine.engine_thread.join(timeout=2.0)

    def test_led_plugin_initialization(self):
        """Test LED plugin initializes correctly."""
        from plugins.outputs.led_plugin import LEDOutputPlugin

        plugin = LEDOutputPlugin(self.engine)

        self.assertEqual(plugin.name, "led")
        self.assertFalse(plugin.running)
        self.assertIsNone(plugin.led_controller)

    def test_led_should_handle_when_active(self):
        """Test should_handle returns True when LED is active."""
        from plugins.outputs.led_plugin import LEDOutputPlugin

        plugin = LEDOutputPlugin(self.engine)
        plugin.led_controller = MockLEDController()  # Simulate active

        self.assertTrue(plugin.should_handle({'source': 'VAD'}))
        self.assertTrue(plugin.should_handle({'source': 'TEXT'}))

    def test_led_should_handle_when_inactive(self):
        """Test should_handle returns False when LED is inactive."""
        from plugins.outputs.led_plugin import LEDOutputPlugin

        plugin = LEDOutputPlugin(self.engine)
        plugin.led_controller = None  # Inactive

        self.assertFalse(plugin.should_handle({'source': 'VAD'}))

    def test_led_output_sets_speaking_state(self):
        """Test output() sets LED to speaking pattern."""
        from plugins.outputs.led_plugin import LEDOutputPlugin

        plugin = LEDOutputPlugin(self.engine)
        plugin.led_controller = MockLEDController()

        # Call output
        plugin.output("Hello there", {'source': 'VAD'})

        # Verify LED was set to speaking pattern
        self.assertEqual(plugin.led_controller.current_state, "PULSE_SLOW")

    def test_led_set_state_endpoint(self):
        """Test _on_set_state updates LED pattern."""
        from plugins.outputs.led_plugin import LEDOutputPlugin

        plugin = LEDOutputPlugin(self.engine)
        plugin.led_controller = MockLEDController()

        # Test various states
        test_cases = [
            ("WAITING", "OFF"),
            ("LISTENING", "SOLID_FULL"),
            ("VERIFYING", "SOLID_FULL"),
            ("PTT", "SOLID_FULL"),
            ("PROCESSING_VAD", "PULSE_MEDIUM"),
            ("SPEAKING", "PULSE_SLOW"),
        ]

        for state, expected_pattern in test_cases:
            plugin._on_set_state({'state': state})
            self.assertEqual(plugin.led_controller.current_state, expected_pattern,
                           f"State {state} should map to {expected_pattern}")


class TestButtonLEDIntegration(unittest.TestCase):
    """Test button and LED plugins working together."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

        # Start engine
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        self.engine_thread.start()
        time.sleep(0.1)

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            self.engine_thread.join(timeout=2.0)

    def test_voice_state_updates_led_via_endpoint(self):
        """Test that voice plugin can update LED via endpoint."""
        from plugins.outputs.led_plugin import LEDOutputPlugin

        # Create LED plugin with mock controller
        led_plugin = LEDOutputPlugin(self.engine)
        led_plugin.led_controller = MockLEDController()
        led_plugin.running = True

        # Register LED endpoint manually (simulating start())
        self.engine.register_endpoint("led", "set_state", led_plugin._on_set_state)

        # Wait for registration
        time.sleep(0.1)

        # Simulate voice plugin sending state update
        self.engine.endpoint_send("led", "set_state", {'state': 'LISTENING'})

        # Wait for processing
        time.sleep(0.2)

        # Verify LED was updated
        self.assertEqual(led_plugin.led_controller.current_state, "SOLID_FULL")


if __name__ == '__main__':
    unittest.main()

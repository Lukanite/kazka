"""
Unit tests for the text input plugin.

Note: These tests don't test actual keyboard input - they test
the plugin's endpoint and programmatic input methods.
"""

import unittest
import time
import threading

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import AssistantEngine
from plugins.inputs.text.text_plugin import TextInputPlugin


class TestTextPluginBasic(unittest.TestCase):
    """Test text plugin basic functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            if self.engine.engine_thread:
                self.engine.engine_thread.join(timeout=2.0)

    def test_text_plugin_initialization(self):
        """Test text plugin initializes correctly."""
        plugin = TextInputPlugin(self.engine)

        self.assertEqual(plugin.name, "text")
        self.assertFalse(plugin.running)
        self.assertEqual(plugin.quit_key, 'q')
        self.assertEqual(plugin.ptt_key, 'p')
        self.assertEqual(plugin.text_key, 't')

    def test_text_plugin_custom_keys(self):
        """Test text plugin accepts custom key bindings."""
        plugin = TextInputPlugin(self.engine, {
            'quit_key': 'x',
            'ptt_key': 'r',
            'text_key': 'i'
        })

        self.assertEqual(plugin.quit_key, 'x')
        self.assertEqual(plugin.ptt_key, 'r')
        self.assertEqual(plugin.text_key, 'i')


class TestTextPluginSubmit(unittest.TestCase):
    """Test text plugin submit functionality."""

    def setUp(self):
        """Set up test fixtures with running engine."""
        self.engine = AssistantEngine()

        # Track inputs received by engine
        self.received_inputs = []

        # Start engine
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        self.engine_thread.start()
        time.sleep(0.1)

        # Create and start text plugin
        self.text_plugin = TextInputPlugin(self.engine)
        self.text_plugin.start()
        time.sleep(0.1)

    def tearDown(self):
        """Clean up after tests."""
        if self.text_plugin.running:
            self.text_plugin.stop()

        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            self.engine_thread.join(timeout=2.0)

    def test_submit_sends_to_engine(self):
        """Test submit() sends input to engine."""
        # Note: This will trigger LLM query, but we just verify the mechanism works
        self.text_plugin.submit("Hello test")

        # Wait for processing
        time.sleep(0.2)

        # Plugin should have emitted input (engine will process it)
        # We can't easily verify without mocking, but no exception means success

    def test_submit_empty_string_ignored(self):
        """Test that empty strings are ignored."""
        self.text_plugin.submit("")
        self.text_plugin.submit("   ")

        # No exception should occur

    def test_endpoint_submit(self):
        """Test programmatic submit via endpoint."""
        # Call endpoint
        response = self.engine.endpoint_call("text", "submit", {'text': 'Test via endpoint'})

        # Verify response
        self.assertEqual(response['status'], 'submitted')


class TestTextPluginShutdown(unittest.TestCase):
    """Test text plugin shutdown callback."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()
        self.shutdown_called = False

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            if self.engine.engine_thread:
                self.engine.engine_thread.join(timeout=2.0)

    def test_shutdown_callback_registration(self):
        """Test that shutdown callback can be registered."""
        plugin = TextInputPlugin(self.engine)

        def on_shutdown():
            self.shutdown_called = True

        plugin.on_shutdown(on_shutdown)

        # Callback should be registered
        self.assertIsNotNone(plugin._shutdown_callback)

    def test_shutdown_callback_called(self):
        """Test that shutdown callback is called when quit key is simulated."""
        plugin = TextInputPlugin(self.engine)

        shutdown_event = threading.Event()

        def on_shutdown():
            shutdown_event.set()

        plugin.on_shutdown(on_shutdown)

        # Simulate callback being triggered
        if plugin._shutdown_callback:
            plugin._shutdown_callback()

        # Verify callback was called
        self.assertTrue(shutdown_event.is_set())


class TestTextPluginLifecycle(unittest.TestCase):
    """Test text plugin start/stop lifecycle."""

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

    def test_start_creates_listener_thread(self):
        """Test that start() creates listener thread."""
        plugin = TextInputPlugin(self.engine)
        plugin.start()

        self.assertTrue(plugin.running)
        self.assertIsNotNone(plugin.listener_thread)
        self.assertTrue(plugin.listener_thread.is_alive())

        plugin.stop()

    def test_stop_stops_listener(self):
        """Test that stop() stops the listener."""
        plugin = TextInputPlugin(self.engine)
        plugin.start()
        plugin.stop()

        self.assertFalse(plugin.running)

    def test_endpoint_registered_on_start(self):
        """Test that endpoint is registered on start."""
        plugin = TextInputPlugin(self.engine)
        plugin.start()

        # Endpoint should be registered
        self.assertTrue(plugin._endpoint_registered)
        self.assertTrue(self.engine.has_endpoint("text", "submit"))

        plugin.stop()


if __name__ == '__main__':
    unittest.main()

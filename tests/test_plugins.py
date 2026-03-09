"""
Unit tests for plugin base classes and console output.
"""

import unittest
import time
import threading
from queue import Queue

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import AssistantEngine
from core.plugin_base import InputPlugin, OutputPlugin, QueuedOutputPlugin
from plugins.outputs.console import ConsoleOutputPlugin


class MockInputPlugin(InputPlugin):
    """Mock input plugin for testing."""

    def __init__(self, engine, name="mock_input"):
        super().__init__(engine, name)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def trigger(self, text: str):
        """Simulate input event."""
        self.emit_input(text, {'source': 'TEST'})


class MockOutputPlugin(OutputPlugin):
    """Mock output plugin for testing (not queued)."""

    def __init__(self, engine, name="mock_output"):
        super().__init__(engine, name)
        self.started = False
        self.stopped = False
        self.outputs = []

    def start(self):
        self.started = True

    def output(self, text, metadata):
        self.outputs.append({'text': text, 'metadata': metadata})

    def stop(self):
        self.stopped = True


class MockQueuedOutputPlugin(QueuedOutputPlugin):
    """Mock queued output plugin for testing."""

    def __init__(self, engine, name="mock_queued"):
        super().__init__(engine, name)
        self.outputs = []

    def start_internal(self):
        pass

    def _process_output(self, text, metadata):
        """Simulate slow processing."""
        time.sleep(0.1)  # Simulate blocking operation
        self.outputs.append({'text': text, 'metadata': metadata})

    def stop_internal(self):
        pass


class TestPluginBase(unittest.TestCase):
    """Test base plugin classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            if self.engine.engine_thread:
                self.engine.engine_thread.join(timeout=2.0)

    def test_input_plugin_initialization(self):
        """Test input plugin initializes correctly."""
        plugin = MockInputPlugin(self.engine, "test_input")

        self.assertEqual(plugin.name, "test_input")
        self.assertEqual(plugin.engine, self.engine)
        self.assertFalse(plugin.started)
        self.assertFalse(plugin.stopped)

    def test_input_plugin_lifecycle(self):
        """Test input plugin start/stop lifecycle."""
        plugin = MockInputPlugin(self.engine)

        plugin.start()
        self.assertTrue(plugin.started)

        plugin.stop()
        self.assertTrue(plugin.stopped)

    def test_input_plugin_emit_input(self):
        """Test input plugin emits input to engine."""
        # Start engine
        engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        engine_thread.start()

        # Create input plugin
        plugin = MockInputPlugin(self.engine)
        plugin.start()

        # Track processed inputs
        processed = []

        def output_callback(text, metadata):
            processed.append(text)

        # Register mock output to capture results
        mock_output = MockOutputPlugin(self.engine)
        self.engine.register_output("mock", mock_output)
        mock_output.start()

        # Trigger input
        plugin.trigger("Hello, world!")

        # Wait for LLM processing (can take several seconds depending on server load)
        time.sleep(8.0)

        # Verify output was received (LLM generates response, metadata preserved)
        self.assertGreaterEqual(len(mock_output.outputs), 1)
        self.assertEqual(mock_output.outputs[0]['metadata']['plugin'], 'mock_input')

        # Cleanup
        plugin.stop()
        self.engine.shutdown(save_memories=False)
        engine_thread.join(timeout=2.0)

    def test_output_plugin_initialization(self):
        """Test output plugin initializes correctly."""
        plugin = MockOutputPlugin(self.engine, "test_output")

        self.assertEqual(plugin.name, "test_output")
        self.assertEqual(plugin.engine, self.engine)
        self.assertFalse(plugin.started)
        self.assertFalse(plugin.stopped)

    def test_output_plugin_lifecycle(self):
        """Test output plugin start/stop lifecycle."""
        plugin = MockOutputPlugin(self.engine)

        plugin.start()
        self.assertTrue(plugin.started)

        plugin.stop()
        self.assertTrue(plugin.stopped)

    def test_output_plugin_should_handle(self):
        """Test output plugin should_handle default behavior."""
        plugin = MockOutputPlugin(self.engine)

        # Default should handle all
        self.assertTrue(plugin.should_handle({'source': 'VAD'}))
        self.assertTrue(plugin.should_handle({'source': 'TEXT'}))
        self.assertTrue(plugin.should_handle({}))


class TestQueuedOutputPlugin(unittest.TestCase):
    """Test queued output plugin pattern."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            if self.engine.engine_thread:
                self.engine.engine_thread.join(timeout=2.0)

    def test_queued_plugin_initialization(self):
        """Test queued plugin initializes correctly."""
        plugin = MockQueuedOutputPlugin(self.engine, "test_queued")

        self.assertEqual(plugin.name, "test_queued")
        self.assertIsNotNone(plugin.output_queue)
        self.assertIsNone(plugin.worker_thread)
        self.assertFalse(plugin.running)

    def test_queued_plugin_starts_worker_thread(self):
        """Test queued plugin starts worker thread."""
        plugin = MockQueuedOutputPlugin(self.engine)
        plugin.start()

        # Verify worker thread started
        self.assertTrue(plugin.running)
        self.assertIsNotNone(plugin.worker_thread)
        self.assertTrue(plugin.worker_thread.is_alive())

        plugin.stop()

    def test_queued_plugin_preserves_ordering(self):
        """Test queued plugin preserves fragment ordering."""
        plugin = MockQueuedOutputPlugin(self.engine)
        plugin.start()

        # Send multiple fragments
        plugin.output("Fragment 1", {})
        plugin.output("Fragment 2", {})
        plugin.output("Fragment 3", {})

        # Wait for processing
        time.sleep(0.5)

        # Verify ordering preserved
        self.assertEqual(len(plugin.outputs), 3)
        self.assertEqual(plugin.outputs[0]['text'], "Fragment 1")
        self.assertEqual(plugin.outputs[1]['text'], "Fragment 2")
        self.assertEqual(plugin.outputs[2]['text'], "Fragment 3")

        plugin.stop()

    def test_queued_plugin_drains_on_shutdown(self):
        """Test queued plugin drains queue on shutdown."""
        plugin = MockQueuedOutputPlugin(self.engine)
        plugin.start()

        # Queue multiple items
        plugin.output("Item 1", {})
        plugin.output("Item 2", {})
        plugin.output("Item 3", {})

        # Give worker thread time to start processing
        time.sleep(0.05)

        # Stop (should drain remaining queue)
        plugin.stop()

        # Verify all processed
        self.assertEqual(len(plugin.outputs), 3)

    def test_queued_plugin_get_queue_size(self):
        """Test queued plugin queue size monitoring."""
        plugin = MockQueuedOutputPlugin(self.engine)
        plugin.start()

        # Queue items
        plugin.output("Item 1", {})
        plugin.output("Item 2", {})

        # Check queue size (might be processed already)
        queue_size = plugin.get_queue_size()
        self.assertGreaterEqual(queue_size, 0)

        # Wait for processing
        time.sleep(0.5)

        # Queue should be empty
        self.assertEqual(plugin.get_queue_size(), 0)

        plugin.stop()


class TestConsoleOutputPlugin(unittest.TestCase):
    """Test console output plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            if self.engine.engine_thread:
                self.engine.engine_thread.join(timeout=2.0)

    def test_console_plugin_initialization(self):
        """Test console plugin initializes correctly."""
        plugin = ConsoleOutputPlugin(self.engine, {})

        self.assertEqual(plugin.name, "console")
        self.assertEqual(plugin.plugin_config, {})

    def test_console_plugin_lifecycle(self):
        """Test console plugin start/stop."""
        plugin = ConsoleOutputPlugin(self.engine, {})

        plugin.start()
        self.assertTrue(plugin.running)
        self.assertIsNotNone(plugin.worker_thread)

        plugin.stop()
        self.assertFalse(plugin.running)

    def test_console_plugin_output(self):
        """Test console plugin outputs correctly."""
        plugin = ConsoleOutputPlugin(self.engine, {})
        plugin.start()

        # Queue output
        plugin.output("Test output", {'source': 'VAD', 'plugin': 'voice'})

        # Wait for processing
        time.sleep(0.2)

        # Verify queue drained
        self.assertEqual(plugin.get_queue_size(), 0)

        plugin.stop()

    def test_console_plugin_handles_all_output(self):
        """Test console plugin should_handle returns True."""
        plugin = ConsoleOutputPlugin(self.engine, {})

        self.assertTrue(plugin.should_handle({'source': 'VAD'}))
        self.assertTrue(plugin.should_handle({'source': 'TEXT'}))
        self.assertTrue(plugin.should_handle({}))


if __name__ == '__main__':
    unittest.main()

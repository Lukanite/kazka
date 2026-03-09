"""
Integration test: Simple input → engine → output flow.

Tests the full pipeline without LLM:
1. Input plugin emits text
2. Engine processes and broadcasts to outputs
3. Output plugin receives and handles

This validates the core architecture works end-to-end.
"""

import unittest
import time
import threading
from queue import Queue

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.engine import AssistantEngine
from core.plugin_base import InputPlugin, QueuedOutputPlugin


class TextInputPlugin(InputPlugin):
    """
    Simple text input plugin for testing.

    Provides a method to simulate user input.
    """

    def __init__(self, engine):
        super().__init__(engine, "text")

    def start(self):
        """Register endpoints for text input."""
        # Register endpoint to receive text programmatically
        self.engine.register_endpoint("text", "submit", self._handle_submit)

    def stop(self):
        """No resources to clean up."""
        pass

    def _handle_submit(self, data):
        """Handle text submission via endpoint."""
        text = data.get('text', '')
        self.emit_input(text, {'source': 'TEXT'})
        return {'status': 'submitted'}

    def submit(self, text: str):
        """Convenience method to submit text directly."""
        self.emit_input(text, {'source': 'TEXT'})


class TestOutputPlugin(QueuedOutputPlugin):
    """
    Test output plugin that captures all output.

    Uses QueuedOutputPlugin to verify ordering is preserved.
    """

    def __init__(self, engine):
        super().__init__(engine, "test_output")
        self.received = []
        self.received_lock = threading.Lock()

    def start_internal(self):
        """No additional initialization needed."""
        pass

    def _process_output(self, text, metadata):
        """Capture output for verification."""
        with self.received_lock:
            self.received.append({
                'text': text,
                'metadata': metadata
            })

    def stop_internal(self):
        """No resources to clean up."""
        pass

    def get_received(self):
        """Thread-safe access to received outputs."""
        with self.received_lock:
            return list(self.received)

    def clear(self):
        """Clear received outputs."""
        with self.received_lock:
            self.received.clear()


class TestSimpleFlow(unittest.TestCase):
    """Test the complete input → engine → output flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

        # Create plugins
        self.text_input = TextInputPlugin(self.engine)
        self.test_output = TestOutputPlugin(self.engine)

        # Register plugins with engine
        self.engine.register_input("text", self.text_input)
        self.engine.register_output("test", self.test_output)

        # Start engine in background thread
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        self.engine_thread.start()

        # Start plugins (after engine is running)
        self.text_input.start()
        self.test_output.start()

    def tearDown(self):
        """Clean up after tests."""
        # Stop plugins
        self.text_input.stop()
        self.test_output.stop()

        # Stop engine
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            self.engine_thread.join(timeout=2.0)

    def test_input_reaches_output(self):
        """Test that input flows through engine to output."""
        # Submit input
        self.text_input.submit("Say hi")

        # Wait for LLM processing
        time.sleep(5.0)

        # Verify output received from LLM
        received = self.test_output.get_received()
        self.assertGreaterEqual(len(received), 1)
        # LLM should respond with something
        self.assertIsInstance(received[0]['text'], str)
        self.assertGreater(len(received[0]['text']), 0)
        # Metadata should be preserved
        self.assertEqual(received[0]['metadata']['source'], 'TEXT')
        self.assertEqual(received[0]['metadata']['plugin'], 'text')

    def test_multiple_inputs_sequential(self):
        """Test multiple inputs are processed sequentially."""
        # Submit first input and wait for response
        self.text_input.submit("Say one")
        time.sleep(5.0)

        first_count = len(self.test_output.get_received())
        self.assertGreaterEqual(first_count, 1)

        # Submit second input
        self.text_input.submit("Say two")
        time.sleep(5.0)

        # Verify we got more outputs
        received = self.test_output.get_received()
        self.assertGreater(len(received), first_count)

    def test_endpoint_message_input(self):
        """Test input via endpoint messaging."""
        # Send input via endpoint
        self.engine.endpoint_send("text", "submit", {'text': "Say hello"})

        # Wait for LLM processing (can take several seconds depending on server load)
        time.sleep(8.0)

        # Verify output received
        received = self.test_output.get_received()
        self.assertGreaterEqual(len(received), 1)
        self.assertIsInstance(received[0]['text'], str)

    def test_endpoint_call_returns_response(self):
        """Test synchronous endpoint call returns response."""
        # Call endpoint synchronously
        response = self.engine.endpoint_call("text", "submit", {'text': "Say test"})

        # Verify response from endpoint
        self.assertEqual(response['status'], 'submitted')

        # Wait for LLM and verify output (may take longer on slower systems)
        time.sleep(10.0)
        received = self.test_output.get_received()
        self.assertGreaterEqual(len(received), 1)

    def test_clean_shutdown(self):
        """Test clean shutdown completes without hanging."""
        # Submit an input
        self.text_input.submit("Say bye")

        # Give it a moment to start processing
        time.sleep(0.5)

        # Shutdown should complete cleanly
        self.text_input.stop()
        self.test_output.stop()
        self.engine.shutdown(save_memories=False)
        self.engine_thread.join(timeout=10.0)

        # Verify engine stopped
        self.assertFalse(self.engine.running)


class TestMultipleOutputPlugins(unittest.TestCase):
    """Test broadcasting to multiple output plugins."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

        # Create input
        self.text_input = TextInputPlugin(self.engine)
        self.engine.register_input("text", self.text_input)

        # Create multiple outputs
        self.output1 = TestOutputPlugin(self.engine)
        self.output1.name = "output1"
        self.output2 = TestOutputPlugin(self.engine)
        self.output2.name = "output2"

        self.engine.register_output("output1", self.output1)
        self.engine.register_output("output2", self.output2)

        # Start engine
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        self.engine_thread.start()

        # Start plugins
        self.text_input.start()
        self.output1.start()
        self.output2.start()

    def tearDown(self):
        """Clean up after tests."""
        self.text_input.stop()
        self.output1.stop()
        self.output2.stop()

        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            self.engine_thread.join(timeout=2.0)

    def test_broadcast_reaches_all_outputs(self):
        """Test input broadcasts to all registered outputs."""
        # Submit input
        self.text_input.submit("Say hello")

        # Wait for LLM processing
        time.sleep(5.0)

        # Verify both outputs received the same content
        received1 = self.output1.get_received()
        received2 = self.output2.get_received()

        self.assertGreaterEqual(len(received1), 1)
        self.assertGreaterEqual(len(received2), 1)
        # Both outputs should receive identical responses
        self.assertEqual(received1[0]['text'], received2[0]['text'])


if __name__ == '__main__':
    unittest.main()

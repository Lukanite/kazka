"""
Full system integration tests for the Voice Assistant.

These tests verify end-to-end functionality of the complete system:
- Input → Engine → LLM → Output flow
- Multiple plugin coordination
- Endpoint communication between plugins
- Queue handling under load
- Graceful shutdown

Note: These tests use real LLM integration, so they require
the LLM server to be running and may be slow.
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.engine import AssistantEngine
from core.plugin_base import InputPlugin, QueuedOutputPlugin
from plugins.inputs.text.text_plugin import TextInputPlugin
from plugins.outputs.console import ConsoleOutputPlugin


class CollectorOutputPlugin(QueuedOutputPlugin):
    """Output plugin that collects all outputs for testing."""

    def __init__(self, engine):
        super().__init__(engine, "collector")
        self.collected = []
        self._lock = threading.Lock()

    def start_internal(self):
        pass

    def _process_output(self, text, metadata):
        with self._lock:
            self.collected.append({
                'text': text,
                'metadata': metadata.copy(),
                'timestamp': time.time()
            })

    def should_handle(self, metadata):
        return True

    def get_collected(self):
        with self._lock:
            return list(self.collected)

    def clear(self):
        with self._lock:
            self.collected.clear()

    def stop_internal(self):
        pass


class TestFullSystemFlow(unittest.TestCase):
    """Test complete system with all components."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

        # Create plugins
        self.text_input = TextInputPlugin(self.engine)
        self.collector = CollectorOutputPlugin(self.engine)
        self.console = ConsoleOutputPlugin(self.engine, {})

        # Register plugins
        self.engine.register_input("text", self.text_input)
        self.engine.register_output("collector", self.collector)
        self.engine.register_output("console", self.console)

        # Start engine and plugins
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        self.engine_thread.start()
        time.sleep(0.1)

        self.text_input.start()
        self.collector.start()
        self.console.start()
        time.sleep(0.1)

    def tearDown(self):
        """Clean up after tests."""
        self.text_input.stop()
        self.collector.stop()
        self.console.stop()

        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            self.engine_thread.join(timeout=5.0)

    def test_text_input_to_llm_response(self):
        """Test text input flows through to LLM and produces output."""
        # Submit text input
        self.text_input.submit("Say hello in one word")

        # Wait for LLM processing
        time.sleep(10.0)

        # Verify output was collected
        collected = self.collector.get_collected()
        self.assertGreaterEqual(len(collected), 1)

        # Verify metadata preserved
        self.assertEqual(collected[0]['metadata']['source'], 'TEXT')
        self.assertEqual(collected[0]['metadata']['plugin'], 'text')

    def test_multiple_sequential_inputs(self):
        """Test multiple inputs are processed in order."""
        # Submit first input
        self.text_input.submit("Say one")
        time.sleep(10.0)

        first_count = len(self.collector.get_collected())
        self.assertGreaterEqual(first_count, 1)

        # Submit second input
        self.text_input.submit("Say two")
        time.sleep(10.0)

        # Verify we got more outputs
        collected = self.collector.get_collected()
        self.assertGreater(len(collected), first_count)

    def test_endpoint_communication(self):
        """Test plugins can communicate via endpoints."""
        # Register a test endpoint
        received_messages = []

        def test_handler(data):
            received_messages.append(data)
            return {'status': 'received'}

        self.engine.register_endpoint("test", "message", test_handler)
        time.sleep(0.1)

        # Send message via endpoint
        result = self.engine.endpoint_call("test", "message", {'content': 'hello'})

        # Verify communication worked
        self.assertEqual(result['status'], 'received')
        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0]['content'], 'hello')

    def test_broadcast_reaches_all_outputs(self):
        """Test that output broadcasts reach all registered outputs."""
        # Submit input
        self.text_input.submit("Say test")

        # Wait for LLM
        time.sleep(10.0)

        # Collector should have received output
        collected = self.collector.get_collected()
        self.assertGreaterEqual(len(collected), 1)

    def test_graceful_shutdown_with_pending_input(self):
        """Test graceful shutdown while input is processing."""
        # Submit input
        self.text_input.submit("Tell me a short story")

        # Start shutdown before LLM completes
        time.sleep(1.0)

        # Shutdown should not hang
        self.text_input.stop()
        self.collector.stop()
        self.console.stop()
        self.engine.shutdown(save_memories=False)

        # Wait for shutdown with timeout (may take longer if LLM is mid-response)
        self.engine_thread.join(timeout=15.0)
        self.assertFalse(self.engine_thread.is_alive())


class TestVoicePluginEndpointIntegration(unittest.TestCase):
    """Test voice plugin endpoint integration without real audio."""

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
            self.engine_thread.join(timeout=5.0)

    def test_wake_requested_endpoint_exists(self):
        """Test that voice plugin endpoints can be registered."""
        # Simulate voice plugin registering endpoints
        wake_triggered = []

        def on_wake(data):
            wake_triggered.append(data)

        self.engine.register_endpoint("voice", "wake_requested", on_wake)
        time.sleep(0.1)

        # Send wake request (simulating button press)
        self.engine.endpoint_send("voice", "wake_requested", {'source': 'test'})
        time.sleep(0.2)

        # Verify handler was called
        self.assertEqual(len(wake_triggered), 1)
        self.assertEqual(wake_triggered[0]['source'], 'test')

    def test_led_state_endpoint(self):
        """Test LED state endpoint communication."""
        state_changes = []

        def on_set_state(data):
            state_changes.append(data.get('state'))

        self.engine.register_endpoint("led", "set_state", on_set_state)
        time.sleep(0.1)

        # Simulate voice plugin sending state changes
        states = ["LISTENING", "PROCESSING_VAD", "WAITING"]
        for state in states:
            self.engine.endpoint_send("led", "set_state", {'state': state})
            time.sleep(0.1)

        # Verify all states received
        self.assertEqual(state_changes, states)


class TestQueueHandling(unittest.TestCase):
    """Test queue handling under various conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AssistantEngine()

        self.text_input = TextInputPlugin(self.engine)
        self.collector = CollectorOutputPlugin(self.engine)

        self.engine.register_input("text", self.text_input)
        self.engine.register_output("collector", self.collector)

        self.engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        self.engine_thread.start()
        time.sleep(0.1)

        self.text_input.start()
        self.collector.start()
        time.sleep(0.1)

    def tearDown(self):
        """Clean up after tests."""
        self.text_input.stop()
        self.collector.stop()

        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            self.engine_thread.join(timeout=5.0)

    def test_rapid_inputs_queued(self):
        """Test rapid inputs are queued and processed."""
        # Submit multiple inputs rapidly
        for i in range(3):
            self.text_input.submit(f"Say {i}")

        # Wait for all to process
        time.sleep(25.0)

        # All inputs should have generated outputs
        collected = self.collector.get_collected()
        self.assertGreaterEqual(len(collected), 3)

    def test_output_ordering_preserved(self):
        """Test that output ordering is preserved for single input."""
        # Submit input that might generate multiple fragments
        self.text_input.submit("Count from 1 to 3")

        # Wait for processing
        time.sleep(10.0)

        # Outputs should have timestamps in order
        collected = self.collector.get_collected()
        if len(collected) > 1:
            for i in range(1, len(collected)):
                self.assertGreaterEqual(
                    collected[i]['timestamp'],
                    collected[i-1]['timestamp']
                )


class TestPluginLifecycle(unittest.TestCase):
    """Test plugin lifecycle management."""

    def test_plugins_start_before_engine(self):
        """Test plugins can be started before engine thread."""
        engine = AssistantEngine()

        # Register plugins first
        text_input = TextInputPlugin(engine)
        collector = CollectorOutputPlugin(engine)

        engine.register_input("text", text_input)
        engine.register_output("collector", collector)

        # Start plugins
        text_input.start()
        collector.start()

        # Then start engine
        engine_thread = threading.Thread(target=engine.run, daemon=False)
        engine_thread.start()
        time.sleep(0.2)

        # Submit input
        text_input.submit("Hello")
        time.sleep(10.0)

        # Should work
        collected = collector.get_collected()
        self.assertGreaterEqual(len(collected), 1)

        # Cleanup
        text_input.stop()
        collector.stop()
        engine.shutdown(save_memories=False)
        engine_thread.join(timeout=5.0)

    def test_plugins_can_stop_independently(self):
        """Test plugins can be stopped while engine runs."""
        engine = AssistantEngine()

        text_input = TextInputPlugin(engine)
        collector = CollectorOutputPlugin(engine)

        engine.register_input("text", text_input)
        engine.register_output("collector", collector)

        engine_thread = threading.Thread(target=engine.run, daemon=False)
        engine_thread.start()
        time.sleep(0.1)

        text_input.start()
        collector.start()

        # Stop collector while engine runs
        collector.stop()

        # Engine should still be running
        self.assertTrue(engine.running)

        # Cleanup
        text_input.stop()
        engine.shutdown(save_memories=False)
        engine_thread.join(timeout=5.0)


if __name__ == '__main__':
    unittest.main()

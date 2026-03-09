"""
Unit tests for the core engine and request system.
"""

import unittest
import time
import threading
from queue import Queue

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import AssistantEngine
from core.requests import (
    ProcessInputRequest,
    EndpointMessageRequest,
    RegisterEndpointRequest,
    ShutdownRequest
)


class TestEngineRequests(unittest.TestCase):
    """Test request objects."""

    def test_process_input_request(self):
        """Test ProcessInputRequest creation."""
        req = ProcessInputRequest(text="hello", metadata={'source': 'TEST'})
        self.assertEqual(req.text, "hello")
        self.assertEqual(req.metadata['source'], 'TEST')
        self.assertFalse(req.is_sync())
        self.assertIsNotNone(req.request_id)

    def test_endpoint_message_request(self):
        """Test EndpointMessageRequest creation."""
        req = EndpointMessageRequest(
            target="voice",
            endpoint="wake_requested",
            data={'source': 'button'}
        )
        self.assertEqual(req.target, "voice")
        self.assertEqual(req.endpoint, "wake_requested")
        self.assertFalse(req.is_sync())

    def test_sync_request(self):
        """Test synchronous request with response queue."""
        response_queue = Queue()
        req = EndpointMessageRequest(
            target="test",
            endpoint="test",
            data={},
            response_queue=response_queue
        )
        self.assertTrue(req.is_sync())


class TestEngine(unittest.TestCase):
    """Test engine core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Use default config (None triggers global config)
        self.engine = AssistantEngine()

    def tearDown(self):
        """Clean up after tests."""
        if self.engine.running:
            self.engine.shutdown(save_memories=False)
            if self.engine.engine_thread:
                self.engine.engine_thread.join(timeout=2.0)

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        self.assertIsNotNone(self.engine.request_queue)
        self.assertFalse(self.engine.running)
        self.assertEqual(len(self.engine.input_plugins), 0)
        self.assertEqual(len(self.engine.output_plugins), 0)

    def test_engine_starts_and_stops(self):
        """Test engine thread starts and stops cleanly."""
        # Start engine thread
        engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        engine_thread.start()

        # Wait for startup
        time.sleep(0.1)
        self.assertTrue(self.engine.running)

        # Request shutdown
        self.engine.shutdown(save_memories=False)

        # Wait for shutdown
        engine_thread.join(timeout=2.0)
        self.assertFalse(engine_thread.is_alive())
        self.assertFalse(self.engine.running)

    def test_endpoint_registration(self):
        """Test endpoint registration."""
        # Start engine
        engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        engine_thread.start()

        # Register endpoint
        callback_called = False

        def test_callback(data):
            nonlocal callback_called
            callback_called = True
            return "success"

        self.engine.register_endpoint("test_component", "test_endpoint", test_callback)

        # Verify endpoint registered
        self.assertTrue(self.engine.has_endpoint("test_component", "test_endpoint"))

        # Test endpoint call
        result = self.engine.endpoint_call("test_component", "test_endpoint", {})
        self.assertEqual(result, "success")
        self.assertTrue(callback_called)

        # Cleanup
        self.engine.shutdown(save_memories=False)
        engine_thread.join(timeout=2.0)

    def test_endpoint_send_async(self):
        """Test asynchronous endpoint messaging."""
        # Start engine
        engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        engine_thread.start()

        # Register endpoint
        received_data = []

        def test_callback(data):
            received_data.append(data)

        self.engine.register_endpoint("test", "endpoint", test_callback)

        # Send message (async)
        self.engine.endpoint_send("test", "endpoint", {'value': 42})

        # Wait for processing
        time.sleep(0.2)

        # Verify callback was called
        self.assertEqual(len(received_data), 1)
        self.assertEqual(received_data[0]['value'], 42)

        # Cleanup
        self.engine.shutdown(save_memories=False)
        engine_thread.join(timeout=2.0)

    def test_nonexistent_endpoint(self):
        """Test calling nonexistent endpoint returns None."""
        # Start engine
        engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        engine_thread.start()

        # Call nonexistent endpoint
        result = self.engine.endpoint_call("nonexistent", "endpoint", {}, timeout=1.0)
        self.assertIsNone(result)

        # Cleanup
        self.engine.shutdown(save_memories=False)
        engine_thread.join(timeout=2.0)

    def test_request_queue_processing(self):
        """Test engine processes requests from queue."""
        # Start engine
        engine_thread = threading.Thread(target=self.engine.run, daemon=False)
        engine_thread.start()

        # Submit multiple requests
        processed = []

        def callback1(data):
            processed.append(1)

        def callback2(data):
            processed.append(2)

        def callback3(data):
            processed.append(3)

        self.engine.register_endpoint("test", "ep1", callback1)
        self.engine.register_endpoint("test", "ep2", callback2)
        self.engine.register_endpoint("test", "ep3", callback3)

        # Send multiple messages
        self.engine.endpoint_send("test", "ep1", {})
        self.engine.endpoint_send("test", "ep2", {})
        self.engine.endpoint_send("test", "ep3", {})

        # Wait for processing
        time.sleep(0.3)

        # Verify all processed in order
        self.assertEqual(processed, [1, 2, 3])

        # Cleanup
        self.engine.shutdown(save_memories=False)
        engine_thread.join(timeout=2.0)


if __name__ == '__main__':
    unittest.main()

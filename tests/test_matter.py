"""
Unit tests for the MatterClient persistent WebSocket connection and
message_id correlation refactor.

All tests use a fake in-process WebSocket; no real Matter server is needed.
"""

import asyncio
import json
import sys
import os
import unittest
from typing import List
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tools.matter as matter_module
from tools.matter import MatterClient, MatterLightControlTool, MatterListDevicesTool

# A minimal server-info handshake frame.
HANDSHAKE = json.dumps({"type": "server_info", "schema_version": 1})


# ---------------------------------------------------------------------------
# Fake WebSocket helpers
# ---------------------------------------------------------------------------

class FakeWebSocket:
    """Controllable in-process WebSocket for unit tests."""

    def __init__(self, recv_frames: List[str]):
        self._frames = list(recv_frames)
        self._idx = 0
        self.sent: List[str] = []
        self.closed = False

    async def recv(self) -> str:
        if self._idx >= len(self._frames):
            raise Exception("FakeWebSocket: no more frames available")
        frame = self._frames[self._idx]
        self._idx += 1
        return frame

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


def _patch_ws(fake_ws_sequence: List[FakeWebSocket]):
    """Return (patcher, connect_calls) where patcher is a context manager.

    The patched ``websockets.connect`` returns fake_ws_sequence items in order.
    ``connect_calls`` is populated with the URIs passed to each connect call.
    """
    ws_iter = iter(fake_ws_sequence)
    connect_calls: List[str] = []

    async def fake_connect(uri, **kwargs):
        connect_calls.append(uri)
        return next(ws_iter)

    patcher = patch.object(matter_module, 'websockets')
    return patcher, connect_calls, fake_connect


# ---------------------------------------------------------------------------
# Tests: persistent connection and message_id correlation
# ---------------------------------------------------------------------------

class TestMatterClientPersistentConnection(unittest.TestCase):
    """Tests that verify the persistent-connection behaviour."""

    def _make_client(self, fake_ws, connect_calls_out=None, timeout=5.0):
        """Patch websockets.connect and return a MatterClient."""
        calls = connect_calls_out if connect_calls_out is not None else []

        async def fake_connect(uri, **kwargs):
            calls.append(uri)
            return fake_ws

        patcher = patch.object(matter_module, 'websockets')
        mock_ws = patcher.start()
        mock_ws.connect = fake_connect
        client = MatterClient(host="test", port=1234, timeout=timeout)
        return client, patcher

    def test_connection_reused_across_commands(self):
        """A single WebSocket connection is used for multiple commands."""
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": "r1"}),
            json.dumps({"message_id": "2", "result": "r2"}),
            json.dumps({"message_id": "3", "result": "r3"}),
        ])
        connect_calls: List[str] = []
        client, patcher = self._make_client(fake_ws, connect_calls)
        try:
            r1 = client.send_command("cmd1")
            r2 = client.send_command("cmd2")
            r3 = client.send_command("cmd3")
        finally:
            patcher.stop()
            client.close()

        # Only one WebSocket connection was established.
        self.assertEqual(len(connect_calls), 1,
                         "Expected exactly one WebSocket connect call")
        self.assertEqual(r1.get("result"), "r1")
        self.assertEqual(r2.get("result"), "r2")
        self.assertEqual(r3.get("result"), "r3")

    def test_handshake_consumed_exactly_once(self):
        """The server-info handshake is consumed once; N commands consume N more frames."""
        fake_ws = FakeWebSocket([
            HANDSHAKE,                                          # handshake (1 frame)
            json.dumps({"message_id": "1", "result": "a"}),   # response 1
            json.dumps({"message_id": "2", "result": "b"}),   # response 2
        ])
        client, patcher = self._make_client(fake_ws)
        try:
            client.send_command("a")
            client.send_command("b")
        finally:
            patcher.stop()
            client.close()

        # 1 handshake + 2 responses = 3 frames total consumed.
        self.assertEqual(fake_ws._idx, 3)

    def test_message_id_correlation_skips_unsolicited_events(self):
        """Unsolicited event frames are skipped; the matching response is returned."""
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"type": "event", "data": "unsolicited_1"}),
            json.dumps({"type": "attribute_update", "path": "1/6/0", "value": True}),
            json.dumps({"message_id": "1", "result": {"status": "ok"}}),
        ])
        client, patcher = self._make_client(fake_ws)
        try:
            result = client.send_command("some_command")
        finally:
            patcher.stop()
            client.close()

        self.assertNotIn("error", result)
        self.assertEqual(result.get("result", {}).get("status"), "ok")
        # All 4 frames were consumed (handshake + 2 events + actual response).
        self.assertEqual(fake_ws._idx, 4)

    def test_multiple_commands_with_interleaved_events(self):
        """Each command correctly finds its own response among interleaved events."""
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"type": "event"}),                            # event before cmd1 reply
            json.dumps({"message_id": "1", "result": "cmd1_result"}),
            json.dumps({"type": "attribute_update"}),                 # event before cmd2 reply
            json.dumps({"message_id": "2", "result": "cmd2_result"}),
        ])
        client, patcher = self._make_client(fake_ws)
        try:
            r1 = client.send_command("cmd1")
            r2 = client.send_command("cmd2")
        finally:
            patcher.stop()
            client.close()

        self.assertEqual(r1.get("result"), "cmd1_result")
        self.assertEqual(r2.get("result"), "cmd2_result")


# ---------------------------------------------------------------------------
# Tests: reconnect on failure
# ---------------------------------------------------------------------------

class TestMatterClientReconnect(unittest.TestCase):

    def test_reconnects_and_retries_on_dropped_connection(self):
        """When the connection drops during send, the client reconnects and retries."""
        broken_ws = FakeWebSocket([HANDSHAKE])

        async def broken_send(data: str) -> None:
            raise ConnectionResetError("Connection dropped")

        broken_ws.send = broken_send  # override send to raise

        good_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": "ok"}),
        ])

        ws_sequence = iter([broken_ws, good_ws])
        connect_calls: List[str] = []

        async def fake_connect(uri, **kwargs):
            connect_calls.append(uri)
            return next(ws_sequence)

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            client = MatterClient(host="test", port=1234)
            result = client.send_command("test_cmd")

        client.close()

        # Connected twice: once to the broken connection, once after reconnect.
        self.assertEqual(len(connect_calls), 2)
        self.assertNotIn("error", result)
        self.assertEqual(result.get("result"), "ok")

    def test_reconnects_and_retries_on_dropped_connection_during_recv(self):
        """When recv fails after send, the client reconnects and retries."""
        broken_ws = FakeWebSocket([HANDSHAKE])

        original_recv = broken_ws.recv

        recv_count = [0]

        async def failing_recv():
            recv_count[0] += 1
            if recv_count[0] == 1:
                return await original_recv()  # handshake succeeds
            raise ConnectionResetError("Connection dropped during recv")

        broken_ws.recv = failing_recv

        good_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": "recovered"}),
        ])

        ws_sequence = iter([broken_ws, good_ws])
        connect_calls: List[str] = []

        async def fake_connect(uri, **kwargs):
            connect_calls.append(uri)
            return next(ws_sequence)

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            client = MatterClient(host="test", port=1234)
            result = client.send_command("test_cmd")

        client.close()

        self.assertEqual(len(connect_calls), 2)
        self.assertNotIn("error", result)
        self.assertEqual(result.get("result"), "recovered")


# ---------------------------------------------------------------------------
# Tests: error conditions return {"error": ...} dicts
# ---------------------------------------------------------------------------

class TestMatterClientErrors(unittest.TestCase):

    def test_connection_refused_returns_error_dict(self):
        """ConnectionRefusedError returns {"error": "Could not connect to Matter server"}."""
        async def fake_connect(uri, **kwargs):
            raise ConnectionRefusedError("refused")

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            client = MatterClient(host="test", port=1234)
            result = client.send_command("test_cmd")

        client.close()

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Could not connect to Matter server")

    def test_timeout_during_recv_returns_error_dict(self):
        """TimeoutError returns {"error": "Connection timeout"}."""
        class HangingWebSocket:
            closed = False

            async def recv(self):
                await asyncio.sleep(100)

            async def send(self, data):
                pass

            async def close(self):
                self.closed = True

        async def fake_connect(uri, **kwargs):
            return HangingWebSocket()

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            # Very short timeout so the test finishes quickly.
            client = MatterClient(host="test", port=1234, timeout=0.1)
            result = client.send_command("test_cmd")

        client.close()

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Connection timeout")

    def test_websockets_not_available_returns_error_dict(self):
        """When websockets is not installed, send_command returns an error dict."""
        original = matter_module.WEBSOCKETS_AVAILABLE
        try:
            matter_module.WEBSOCKETS_AVAILABLE = False
            client = MatterClient(host="test", port=1234)
            result = client.send_command("test_cmd")
        finally:
            matter_module.WEBSOCKETS_AVAILABLE = original
            client.close()

        self.assertIn("error", result)

    def test_device_command_propagates_error(self):
        """device_command returns {"error": ...} when the connection is refused."""
        async def fake_connect(uri, **kwargs):
            raise ConnectionRefusedError("refused")

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            client = MatterClient(host="test", port=1234)
            result = client.device_command(1, 1, 6, "On")

        client.close()

        self.assertIn("error", result)

    def test_get_nodes_propagates_error(self):
        """get_nodes returns {"error": ...} when the connection is refused."""
        async def fake_connect(uri, **kwargs):
            raise ConnectionRefusedError("refused")

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            client = MatterClient(host="test", port=1234)
            result = client.get_nodes()

        client.close()

        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Tests: MatterLightControlTool tool-level behaviour
# ---------------------------------------------------------------------------

class TestMatterLightControlTool(unittest.TestCase):

    def _make_tool(self, fake_ws, device_aliases=None, groups=None):
        async def fake_connect(uri, **kwargs):
            return fake_ws

        patcher = patch.object(matter_module, 'websockets')
        mock_ws = patcher.start()
        mock_ws.connect = fake_connect

        tool = MatterLightControlTool(
            matter_host="test",
            matter_port=1234,
            device_aliases=device_aliases or {"light": {"node_id": 1, "endpoint_id": 1}},
            groups=groups or {},
        )
        return tool, patcher

    def test_on_action_single_device(self):
        """'on' action on a single device sends one On command."""
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": None}),
        ])
        tool, patcher = self._make_tool(fake_ws)
        try:
            result = tool.execute(action="on", target="light")
        finally:
            patcher.stop()
            tool.client.close()

        self.assertTrue(result["success"])

    def test_off_action_single_device(self):
        """'off' action on a single device sends one Off command."""
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": None}),
        ])
        tool, patcher = self._make_tool(fake_ws)
        try:
            result = tool.execute(action="off", target="light")
        finally:
            patcher.stop()
            tool.client.close()

        self.assertTrue(result["success"])

    def test_group_action_reuses_single_connection(self):
        """Group control sends multiple commands over a single connection."""
        # 'on' with no brightness/color_temp = 1 On command per device.
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": None}),  # On device 1
            json.dumps({"message_id": "2", "result": None}),  # On device 2
        ])
        connect_calls: List[str] = []

        async def fake_connect(uri, **kwargs):
            connect_calls.append(uri)
            return fake_ws

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            tool = MatterLightControlTool(
                matter_host="test",
                matter_port=1234,
                device_aliases={
                    "light1": {"node_id": 1, "endpoint_id": 1},
                    "light2": {"node_id": 2, "endpoint_id": 1},
                },
                groups={"bedroom": ["light1", "light2"]},
            )
            result = tool.execute(action="on", target="bedroom")

        tool.client.close()

        self.assertTrue(result["success"])
        # Only one WebSocket connection for the whole group operation.
        self.assertEqual(len(connect_calls), 1)

    def test_set_brightness_action(self):
        """'set_brightness' sends a MoveToLevelWithOnOff command with correct level."""
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": None}),
        ])
        tool, patcher = self._make_tool(fake_ws)
        try:
            result = tool.execute(action="set_brightness", target="light", brightness=50)
        finally:
            patcher.stop()
            tool.client.close()

        self.assertTrue(result["success"])
        # 50% of 254 = 127
        sent_payload = json.loads(fake_ws.sent[0])
        self.assertEqual(sent_payload["args"]["command_name"], "MoveToLevelWithOnOff")
        self.assertEqual(sent_payload["args"]["payload"]["level"], 127)

    def test_brightness_scaling(self):
        """Brightness 0-100 maps correctly to 0-254."""
        cases = [(0, 0), (100, 254), (50, 127), (1, 2)]
        for pct, expected_level in cases:
            fake_ws = FakeWebSocket([
                HANDSHAKE,
                json.dumps({"message_id": "1", "result": None}),
            ])
            tool, patcher = self._make_tool(fake_ws)
            try:
                tool.execute(action="set_brightness", target="light", brightness=pct)
            finally:
                patcher.stop()
                tool.client.close()

            sent_payload = json.loads(fake_ws.sent[0])
            actual_level = sent_payload["args"]["payload"]["level"]
            self.assertEqual(actual_level, expected_level,
                             f"brightness={pct}% should map to level={expected_level}, got {actual_level}")

    def test_set_color_temp_action(self):
        """'set_color_temp' sends a MoveToColorTemperature with correct mireds."""
        temp_map = {"daylight": 153, "cool": 220, "neutral": 300, "warm": 400}
        for preset, expected_mireds in temp_map.items():
            fake_ws = FakeWebSocket([
                HANDSHAKE,
                json.dumps({"message_id": "1", "result": None}),
            ])
            tool, patcher = self._make_tool(fake_ws)
            try:
                tool.execute(action="set_color_temp", target="light", color_temp=preset)
            finally:
                patcher.stop()
                tool.client.close()

            sent_payload = json.loads(fake_ws.sent[0])
            actual_mireds = sent_payload["args"]["payload"]["colorTemperatureMireds"]
            self.assertEqual(actual_mireds, expected_mireds,
                             f"color_temp={preset!r} should map to {expected_mireds} mireds")

    def test_error_from_client_returns_success_false(self):
        """When the server returns an error, execute returns success=False."""
        async def fake_connect(uri, **kwargs):
            raise ConnectionRefusedError("refused")

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            tool = MatterLightControlTool(
                matter_host="test",
                matter_port=1234,
                device_aliases={"light": {"node_id": 1, "endpoint_id": 1}},
            )
            result = tool.execute(action="on", target="light")

        tool.client.close()

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_unknown_target_returns_error(self):
        """An unknown target name returns success=False without touching the server."""
        tool = MatterLightControlTool(
            matter_host="test",
            matter_port=1234,
            device_aliases={"light": {"node_id": 1, "endpoint_id": 1}},
        )
        result = tool.execute(action="on", target="nonexistent_device")

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_websockets_unavailable(self):
        """When websockets is not installed, execute returns a helpful error."""
        original = matter_module.WEBSOCKETS_AVAILABLE
        try:
            matter_module.WEBSOCKETS_AVAILABLE = False
            tool = MatterLightControlTool(
                matter_host="test",
                matter_port=1234,
                device_aliases={"light": {"node_id": 1, "endpoint_id": 1}},
            )
            result = tool.execute(action="on", target="light")
        finally:
            matter_module.WEBSOCKETS_AVAILABLE = original

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_on_with_brightness_and_color_temp(self):
        """'on' action with brightness and color_temp sends 3 commands."""
        # brightness (MoveToLevel), color_temp, On
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": None}),  # MoveToLevel
            json.dumps({"message_id": "2", "result": None}),  # MoveToColorTemperature
            json.dumps({"message_id": "3", "result": None}),  # On
        ])
        tool, patcher = self._make_tool(fake_ws)
        try:
            result = tool.execute(
                action="on", target="light", brightness=80, color_temp="warm"
            )
        finally:
            patcher.stop()
            tool.client.close()

        self.assertTrue(result["success"])
        # Three frames were consumed for three commands (+ handshake = 4 total).
        self.assertEqual(fake_ws._idx, 4)

    def test_partial_failure_in_group(self):
        """If one device in a group returns an error, the result reflects partial success."""
        # Both devices share the persistent connection.
        # light1 gets a success response; light2 gets an error response from the server.
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": None}),              # On light1: success
            json.dumps({"message_id": "2", "error": "Device not reachable"}),  # On light2: error
        ])

        async def fake_connect(uri, **kwargs):
            return fake_ws

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            tool = MatterLightControlTool(
                matter_host="test",
                matter_port=1234,
                device_aliases={
                    "light1": {"node_id": 1, "endpoint_id": 1},
                    "light2": {"node_id": 2, "endpoint_id": 1},
                },
                groups={"all": ["light1", "light2"]},
            )
            result = tool.execute(action="on", target="all")

        tool.client.close()

        # Partial success: 1 of 2 devices succeeded.
        self.assertTrue(result["success"])
        self.assertIn("1 of 2", result["message"])


# ---------------------------------------------------------------------------
# Tests: MatterListDevicesTool tool-level behaviour
# ---------------------------------------------------------------------------

class TestMatterListDevicesTool(unittest.TestCase):

    def test_execute_returns_device_list(self):
        """execute() parses nodes and returns device info including type and state."""
        nodes = [
            {
                "node_id": 1,
                "available": True,
                "attributes": {
                    "0/40/1": "TestVendor",
                    "0/40/14": "TestProduct",
                    "0/40/5": "TestLight",
                    "1/6/0": True,
                    "1/8/0": 128,
                },
            }
        ]
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": nodes}),
        ])

        async def fake_connect(uri, **kwargs):
            return fake_ws

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            tool = MatterListDevicesTool(matter_host="test", matter_port=1234)
            result = tool.execute()

        tool.client.close()

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)
        device = result["devices"][0]
        self.assertEqual(device["type"], "light")
        self.assertTrue(device["is_on"])
        self.assertEqual(device["brightness"], 128)

    def test_execute_propagates_connection_error(self):
        """execute() returns success=False when the connection is refused."""
        async def fake_connect(uri, **kwargs):
            raise ConnectionRefusedError("refused")

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            tool = MatterListDevicesTool(matter_host="test", matter_port=1234)
            result = tool.execute()

        tool.client.close()

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_websockets_unavailable(self):
        """When websockets is not installed, execute returns a helpful error."""
        original = matter_module.WEBSOCKETS_AVAILABLE
        try:
            matter_module.WEBSOCKETS_AVAILABLE = False
            tool = MatterListDevicesTool(matter_host="test", matter_port=1234)
            result = tool.execute()
        finally:
            matter_module.WEBSOCKETS_AVAILABLE = original

        self.assertFalse(result["success"])
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Tests: close() cleans up resources
# ---------------------------------------------------------------------------

class TestMatterClientClose(unittest.TestCase):

    def test_close_stops_background_thread(self):
        """close() stops the background event-loop thread."""
        fake_ws = FakeWebSocket([
            HANDSHAKE,
            json.dumps({"message_id": "1", "result": "ok"}),
        ])

        async def fake_connect(uri, **kwargs):
            return fake_ws

        with patch.object(matter_module, 'websockets') as mock_ws:
            mock_ws.connect = fake_connect
            client = MatterClient(host="test", port=1234)
            client.send_command("cmd")  # force loop start
            loop_thread = client._loop_thread

        self.assertTrue(loop_thread.is_alive())
        client.close()
        loop_thread.join(timeout=2.0)
        self.assertFalse(loop_thread.is_alive(),
                         "Background loop thread should have stopped after close()")

    def test_close_before_any_command_is_safe(self):
        """Calling close() before sending any command does not raise."""
        client = MatterClient(host="test", port=1234)
        try:
            client.close()  # should not raise
        except Exception as exc:
            self.fail(f"close() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()

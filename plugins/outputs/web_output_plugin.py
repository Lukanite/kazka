"""
Web output plugin — broadcasts assistant responses to all connected browser clients.

Extends OutputPlugin directly (not QueuedOutputPlugin) because WebSocket sends
are non-blocking: broadcast() posts a coroutine onto the server's event loop and
returns immediately, so there is no need for a separate worker thread.

Overrides output_chunk() for real-time streaming to clients. Each token is sent
as it arrives, mirroring the console plugin's streaming behaviour.
"""

from typing import Dict, Any

from core.plugin_base import OutputPlugin
from plugins.shared.web_server import WebServer


class WebOutputPlugin(OutputPlugin):
    """
    Output plugin that streams responses to all connected WebSocket clients.

    Receives the shared WebServer instance from WebInputPlugin so both plugins
    use the same server and client registry.
    """

    def __init__(self, engine: 'AssistantEngine', web_server: WebServer):
        super().__init__(engine, "web")
        self._server = web_server

    def start(self):
        # Register endpoint so other plugins can push state updates
        # (e.g. voice plugin broadcasts LISTENING / SPEAKING / etc.)
        self.engine.register_endpoint("web", "state_update", self._on_state_update)
        print("✅ Web output plugin started")

    def stop(self):
        print("🛑 Web output plugin stopped")

    # ------------------------------------------------------------------
    # Output interface
    # ------------------------------------------------------------------

    def output_chunk(self, text: str, metadata: Dict[str, Any], is_final: bool = False):
        """
        Stream each token to clients as it arrives.

        Sends a 'chunk' message for every fragment. Clients accumulate the
        tokens and display them inline, closing the bubble on is_final=True.
        """
        is_thinking = metadata.get('is_thinking', False)

        if is_thinking:
            self._server.broadcast({
                "type": "thinking",
                "text": text,
            })
            return

        self._server.broadcast({
            "type": "chunk",
            "text": text,
            "is_final": is_final,
            "source": metadata.get("source", "UNKNOWN"),
        })

        # Keep parent buffer in sync so output() is correct if ever called
        self._chunk_buffer += text
        self._metadata_buffer.update(metadata)
        if is_final:
            self._chunk_buffer = ""
            self._metadata_buffer = {}

    def output(self, text: str, metadata: Dict[str, Any]):
        """
        Non-streaming fallback: send the complete text as a single final chunk.

        Called by the base class when output_chunk() is not overridden or when
        a plugin directly calls output(). In practice, output_chunk() handles
        all streaming responses, so this fires only in edge cases.
        """
        self._server.broadcast({
            "type": "chunk",
            "text": text,
            "is_final": True,
            "source": metadata.get("source", "UNKNOWN"),
        })

    def should_handle(self, metadata: Dict[str, Any]) -> bool:
        """Broadcast everything — the web UI is an observer of the whole session."""
        return True

    # ------------------------------------------------------------------
    # State endpoint
    # ------------------------------------------------------------------

    def _on_state_update(self, data: Dict[str, Any]) -> None:
        """
        Receive voice/engine state updates and forward them to clients.

        Registered as the 'web.state_update' endpoint. Follows the same
        calling convention as 'led.set_state': data = {'state': '<STATE>'}.

        Args:
            data: {'state': 'LISTENING'/'PROCESSING_VAD'/'WAITING'/etc.}
        """
        state = data.get("state", "WAITING")
        self._server.broadcast({
            "type": "state",
            "state": state,
        })

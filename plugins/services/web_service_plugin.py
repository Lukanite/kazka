"""
Web service plugin — handles engine lifecycle events for the web UI.

Complements WebInputPlugin and WebOutputPlugin. Receives sleep notifications
from the engine and clears the web history and client UIs accordingly.
"""

from typing import Any, Dict, Optional

from core.plugin_base import ServicePlugin
from plugins.shared.web_server import WebServer


class WebServicePlugin(ServicePlugin):
    """
    Service plugin that bridges engine lifecycle events to the web UI.

    Holds a reference to the shared WebServer so it can clear history
    and notify connected clients when the engine sleeps.
    """

    def __init__(self, engine: 'AssistantEngine', web_server: WebServer):
        super().__init__(engine, "web_service")
        self._server = web_server

    def start(self):
        print("✅ Web service plugin started")

    def stop(self):
        print("🛑 Web service plugin stopped")

    def on_interaction_start(self, text: str, metadata: Dict[str, Any], images: Optional[list] = None):
        """
        Mirror interaction-start events to connected clients.

        Branches on ``metadata['source']``:
          - ``WEB``: already recorded + rendered by the WebServer's text_input
            handler (and the originating client renders locally) — skip.
          - ``WAKE_TIMER``: render a system/wake bubble so the assistant's
            reply has visible context instead of appearing unprompted.
          - anything else (text/voice plugins): render as a user bubble.
        """
        source = metadata.get("source")
        if source == "WEB":
            return
        if source == "WAKE_TIMER":
            self._server.record_wake(metadata.get("delay_description", ""))
            return
        self._server.record_user_input(text, images)

    def on_sleep_complete(self):
        """Clear conversation history and notify all clients after a sleep cycle."""
        self._server.clear_history()
        self._server.broadcast({"type": "clear"})

    def on_undo(self):
        """Remove the last exchange from history and notify all clients."""
        # If the undo was triggered by a web edit, the edit handler already
        # cleaned up history and notified clients synchronously. Skip to
        # avoid double-undo.
        if self._server.consume_edit_undo_pending():
            return
        self._server.undo_last_exchange()

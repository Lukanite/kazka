"""
Web service plugin — handles engine lifecycle events for the web UI.

Complements WebInputPlugin and WebOutputPlugin. Receives sleep notifications
from the engine and clears the web history and client UIs accordingly.
"""

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

    def on_sleep_complete(self):
        """Clear conversation history and notify all clients after a sleep cycle."""
        self._server.clear_history()
        self._server.broadcast({"type": "clear"})

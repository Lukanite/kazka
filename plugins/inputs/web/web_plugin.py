"""
Web input plugin — accepts text input from browser clients over WebSocket.

Owns the WebServer lifecycle. Paired with WebOutputPlugin which holds a
reference to the same WebServer instance for broadcasting responses.
"""

from core.plugin_base import InputPlugin
from core.config import config
from plugins.shared.web_server import WebServer


class WebInputPlugin(InputPlugin):
    """
    Input plugin that receives text from browser clients via WebSocket.

    Starts an HTTP/WebSocket server (FastAPI + uvicorn) in a background thread.
    Any connected client can submit text; it is queued through the engine
    just like keyboard input from the text plugin.

    The paired WebOutputPlugin must be given a reference to self.web_server
    so both plugins share the same server instance.
    """

    def __init__(self, engine: 'AssistantEngine'):
        super().__init__(engine, "web")
        self.web_server = WebServer()

    def start(self):
        self.web_server.set_input_callback(self._on_text_received)
        self.web_server.set_edit_callback(self._on_edit_last)
        self.web_server.start(
            host=config.web.host,
            port=config.web.port,
        )
        print(f"✅ Web input plugin started — http://{config.web.host}:{config.web.port}")

    def stop(self):
        self.web_server.stop()
        print("🛑 Web input plugin stopped")

    def _on_text_received(self, text: str):
        """Called by the WebServer when a client sends a text_input message."""
        self.emit_input(text, {"source": "WEB"})

    def _on_edit_last(self, text: str):
        """Called by the WebServer when a client edits their last message."""
        # 1. Remove old exchange from history + notify all clients
        self.web_server.undo_last_exchange()
        # 2. Undo the turn in the engine's conversation history
        self.engine.undo_turn()
        # 3. Submit the edited text as new input
        self.emit_input(text, {"source": "WEB"})

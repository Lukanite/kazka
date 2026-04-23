"""
Shared WebServer for the web input/output plugin pair.

Owns the FastAPI app, uvicorn lifecycle, and the connected client registry.
Neither an input nor an output plugin — it is a plain object held by both.

Thread model:
  - uvicorn runs its own asyncio event loop in a daemon thread
  - WebInputPlugin calls start() / stop() to manage that thread
  - WebOutputPlugin calls broadcast() from the engine's worker thread (sync)
  - broadcast() bridges sync → async via asyncio.run_coroutine_threadsafe()
"""

import asyncio
import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# Suppress uvicorn's access log noise
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

_STATIC_DIR = Path(__file__).parent / "web_static"


class WebServer:
    """
    Shared HTTP/WebSocket server for the web plugin pair.

    Lifecycle owned by WebInputPlugin (start / stop).
    Output broadcasts owned by WebOutputPlugin (broadcast).
    Input callback registered by WebInputPlugin (set_input_callback).
    """

    # Maximum number of completed messages to replay to new clients.
    _HISTORY_MAX = 200

    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._clients_lock = threading.Lock()
        self._input_callback: Optional[Callable[[str], None]] = None
        self._edit_callback: Optional[Callable[[str], None]] = None
        self._edit_undo_pending = False  # Flag to prevent double-undo from edit path

        # Catch-up history for late-joining clients.
        # Stores only replayable messages (final chunks + last state).
        self._history: Deque[dict] = deque(maxlen=self._HISTORY_MAX)
        self._last_state: Optional[dict] = None  # Only the most recent state matters
        self._chunk_accumulator: str = ""    # Accumulates streaming content tokens for history
        self._thinking_accumulator: str = "" # Accumulates streaming thinking tokens for history
        self._history_lock = threading.Lock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

        self.app = FastAPI(docs_url=None, redoc_url=None)
        self._register_routes()
        self.app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def _register_routes(self):
        app = self.app

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()

            # Replay history to the new client before joining the broadcast set.
            # Snapshot under lock so broadcast() can't interleave during replay.
            with self._history_lock:
                catchup = list(self._history)
                last_state = self._last_state

            for msg in catchup:
                await ws.send_text(json.dumps(msg))
            if last_state:
                await ws.send_text(json.dumps(last_state))

            with self._clients_lock:
                self._clients.add(ws)

            try:
                while True:
                    data = await ws.receive_text()
                    try:
                        msg = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")

                    if msg_type == "text_input":
                        text = msg.get("text", "").strip()
                        images = msg.get("images", [])
                        if text or images:
                            # Record user message in history for catch-up
                            user_msg = {"type": "user_input", "text": text}
                            if images:
                                user_msg["images"] = images
                            with self._history_lock:
                                self._history.append(user_msg)
                            if self._input_callback:
                                self._input_callback(text, images or None)

                    elif msg_type == "edit_last":
                        text = msg.get("text", "").strip()
                        images = msg.get("images", [])
                        if text and self._edit_callback:
                            # Clean up history + notify clients synchronously,
                            # before the engine queues async undo/input requests.
                            self.undo_last_exchange()
                            self._edit_undo_pending = True
                            # Record + broadcast the new user message
                            user_msg = {"type": "user_input", "text": text}
                            if images:
                                user_msg["images"] = images
                            with self._history_lock:
                                self._history.append(user_msg)
                            self.broadcast(user_msg)
                            # Queue engine undo + re-submit (async)
                            self._edit_callback(text, images or None)

            except WebSocketDisconnect:
                pass
            finally:
                with self._clients_lock:
                    self._clients.discard(ws)

    # ------------------------------------------------------------------
    # Input callback
    # ------------------------------------------------------------------

    def set_input_callback(self, callback: Callable[[str], None]):
        """Register the function called when a client sends text input."""
        self._input_callback = callback

    def set_edit_callback(self, callback: Callable[[str], None]):
        """Register the function called when a client edits the last message."""
        self._edit_callback = callback

    def clear_history(self):
        """Clear all message history and reset streaming state."""
        with self._history_lock:
            self._history.clear()
            self._last_state = None
            self._chunk_accumulator = ""
            self._thinking_accumulator = ""

    def consume_edit_undo_pending(self) -> bool:
        """Check and clear the edit-undo flag. Returns True if an edit already handled the undo."""
        if self._edit_undo_pending:
            self._edit_undo_pending = False
            return True
        return False

    def undo_last_exchange(self):
        """
        Remove the last user message and its assistant response from history,
        and notify all clients to remove them from the UI.
        """
        with self._history_lock:
            # Pop backwards: remove trailing assistant chunk(s), then the user_input
            while self._history and self._history[-1].get("type") != "user_input":
                self._history.pop()
            if self._history and self._history[-1].get("type") == "user_input":
                self._history.pop()
            # Reset accumulator in case an edit arrives mid-stream
            self._chunk_accumulator = ""
            self._thinking_accumulator = ""

        self.broadcast({"type": "undo_last"})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start uvicorn in a background daemon thread."""
        cfg = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="warning",
        )
        self._server = uvicorn.Server(cfg)

        def _run():
            # Create a fresh event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True, name="web_server")
        self._thread.start()

    def stop(self):
        """Signal uvicorn to shut down and wait for the thread."""
        if self._server:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Broadcast (sync → async bridge)
    # ------------------------------------------------------------------

    def broadcast(self, message: dict):
        """
        Send a JSON message to all connected WebSocket clients.

        Safe to call from any thread. No-op if no clients are connected
        or the server loop isn't running yet.
        """
        # Record to history regardless of whether any clients are connected,
        # so late joiners still get a full catch-up.
        msg_type = message.get("type")
        with self._history_lock:
            if msg_type == "thinking":
                self._thinking_accumulator += message.get("text", "")
            elif msg_type == "chunk":
                # Flush any accumulated thinking to history before the content lands
                if self._thinking_accumulator:
                    self._history.append({
                        "type": "thinking",
                        "text": self._thinking_accumulator,
                    })
                    self._thinking_accumulator = ""
                if message.get("is_final"):
                    full_text = self._chunk_accumulator + message.get("text", "")
                    self._history.append({
                        "type": "chunk",
                        "text": full_text,
                        "is_final": True,
                        "source": message.get("source", "UNKNOWN"),
                    })
                    self._chunk_accumulator = ""
                    self._thinking_accumulator = ""
                else:
                    self._chunk_accumulator += message.get("text", "")
            elif msg_type == "state":
                self._last_state = message  # Only latest state is useful

        if not self._loop or not self._loop.is_running():
            return

        with self._clients_lock:
            clients = list(self._clients)

        if not clients:
            return

        payload = json.dumps(message)

        async def _send_all():
            for ws in clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    # Client may have disconnected between the snapshot and now
                    with self._clients_lock:
                        self._clients.discard(ws)

        asyncio.run_coroutine_threadsafe(_send_all(), self._loop)

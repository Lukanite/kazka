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
from typing import Callable, Deque, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Suppress uvicorn's access log noise
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Minimal single-page UI served at GET /
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Kazka</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, sans-serif;
      background: #0f0f0f;
      color: #e0e0e0;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      padding: 12px 20px;
      background: #1a1a1a;
      border-bottom: 1px solid #2a2a2a;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    #status-dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      background: #555;
      flex-shrink: 0;
      transition: background 0.3s;
    }
    #status-dot.connected   { background: #4caf50; }
    #status-dot.listening   { background: #2196f3; }
    #status-dot.processing  { background: #ff9800; animation: pulse 1s infinite; }
    #status-dot.speaking    { background: #9c27b0; animation: pulse 1s infinite; }
    @keyframes pulse {
      0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
    }
    #status-text { font-size: 0.8rem; color: #888; }
    #log {
      flex: 1;
      overflow-y: auto;
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .msg {
      max-width: 80%;
      padding: 10px 14px;
      border-radius: 12px;
      line-height: 1.5;
      font-size: 0.95rem;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg.user {
      align-self: flex-end;
      background: #1e3a5f;
      color: #cde;
    }
    .msg.user.editable {
      cursor: pointer;
      position: relative;
    }
    .msg.user.editable:hover {
      background: #254f80;
    }
    .msg.user.editable::after {
      content: '✎';
      position: absolute;
      top: 4px;
      right: 8px;
      font-size: 0.7rem;
      opacity: 0;
      transition: opacity 0.2s;
    }
    .msg.user.editable:hover::after {
      opacity: 0.6;
    }
    .msg.user.editing {
      background: #1e3a5f;
      padding: 6px;
      max-width: 80%;
      min-width: min(400px, 80%);
    }
    .edit-area {
      width: 100%;
      min-width: 0;
      background: #0f2a4a;
      border: 1px solid #3a6a9f;
      border-radius: 6px;
      padding: 8px 10px;
      color: #cde;
      font-family: inherit;
      font-size: 0.95rem;
      resize: none;
      outline: none;
      line-height: 1.5;
    }
    .edit-buttons {
      display: flex;
      gap: 6px;
      margin-top: 6px;
      justify-content: flex-end;
    }
    .edit-buttons button {
      border: none;
      border-radius: 6px;
      padding: 4px 12px;
      font-size: 0.8rem;
      cursor: pointer;
    }
    .edit-buttons .save { background: #2a7a4a; color: #cec; }
    .edit-buttons .save:hover { background: #35955a; }
    .edit-buttons .discard { background: #3a3a3a; color: #aaa; }
    .edit-buttons .discard:hover { background: #4a4a4a; }
    .msg.assistant {
      align-self: flex-start;
      background: #1e1e1e;
      border: 1px solid #2a2a2a;
    }
    .msg.assistant .label {
      font-size: 0.7rem;
      color: #666;
      margin-bottom: 4px;
    }
    .msg.thinking {
      align-self: flex-start;
      background: transparent;
      border: none;
      color: #555;
      font-style: italic;
      font-size: 0.85rem;
      padding: 2px 4px;
    }
    .msg.error {
      align-self: flex-start;
      background: #3a1a1a;
      border: 1px solid #6a2a2a;
      color: #f88;
    }
    footer {
      padding: 12px 20px;
      background: #1a1a1a;
      border-top: 1px solid #2a2a2a;
      display: flex;
      gap: 10px;
    }
    #input {
      flex: 1;
      background: #2a2a2a;
      border: 1px solid #3a3a3a;
      border-radius: 8px;
      padding: 10px 14px;
      color: #e0e0e0;
      font-size: 0.95rem;
      outline: none;
    }
    #input:focus { border-color: #555; }
    #send {
      background: #1e3a5f;
      color: #cde;
      border: none;
      border-radius: 8px;
      padding: 10px 18px;
      cursor: pointer;
      font-size: 0.95rem;
    }
    #send:hover { background: #254f80; }
    #send:disabled { opacity: 0.4; cursor: not-allowed; }
  </style>
</head>
<body>
  <header>
    <div id="status-dot"></div>
    <span id="status-text">Connecting…</span>
  </header>
  <div id="log"></div>
  <footer>
    <input id="input" type="text" placeholder="Message Kazka…" disabled />
    <button id="send" disabled>Send</button>
  </footer>

  <script>
    const log      = document.getElementById('log');
    const input    = document.getElementById('input');
    const sendBtn  = document.getElementById('send');
    const dot      = document.getElementById('status-dot');
    const statusTx = document.getElementById('status-text');

    let ws;
    let currentMsg = null;   // streaming assistant bubble
    let lastUserEl = null;   // last user bubble (editable)
    let reconnectDelay = 1000;

    function setStatus(cls, text) {
      dot.className = cls;
      statusTx.textContent = text;
    }

    function appendUser(text) {
      // Remove editable status from previous user bubble
      if (lastUserEl) lastUserEl.classList.remove('editable');

      const el = document.createElement('div');
      el.className = 'msg user editable';
      el.textContent = text;
      el.addEventListener('click', () => startEdit(el));
      log.appendChild(el);
      log.scrollTop = log.scrollHeight;
      lastUserEl = el;
    }

    function startEdit(el) {
      if (!el.classList.contains('editable')) return;
      if (el.classList.contains('editing')) return;

      const originalText = el.textContent;
      el.classList.remove('editable');
      el.classList.add('editing');
      el.textContent = '';

      const textarea = document.createElement('textarea');
      textarea.className = 'edit-area';
      textarea.value = originalText;
      textarea.rows = Math.max(1, Math.ceil(originalText.length / 50));

      const buttons = document.createElement('div');
      buttons.className = 'edit-buttons';
      const saveBtn = document.createElement('button');
      saveBtn.className = 'save';
      saveBtn.textContent = 'Send';
      const discardBtn = document.createElement('button');
      discardBtn.className = 'discard';
      discardBtn.textContent = 'Cancel';
      buttons.appendChild(discardBtn);
      buttons.appendChild(saveBtn);

      el.appendChild(textarea);
      el.appendChild(buttons);
      textarea.focus();
      textarea.setSelectionRange(textarea.value.length, textarea.value.length);

      function finishEdit(submit) {
        const newText = textarea.value.trim();

        if (submit && newText) {
          // Server will broadcast undo_last (removing this bubble)
          // followed by user_input (re-rendering the new text).
          ws.send(JSON.stringify({ type: 'edit_last', text: newText }));
        } else {
          // Cancel — restore the original bubble
          el.textContent = originalText;
          el.classList.remove('editing');
          el.classList.add('editable');
        }
      }

      saveBtn.addEventListener('click', (e) => { e.stopPropagation(); finishEdit(true); });
      discardBtn.addEventListener('click', (e) => { e.stopPropagation(); finishEdit(false); });
      textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); finishEdit(true); }
        if (e.key === 'Escape') finishEdit(false);
        e.stopPropagation();
      });
    }

    function startAssistantBubble(source) {
      currentMsg = document.createElement('div');
      currentMsg.className = 'msg assistant';
      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = source ? `Kazka [${source}]` : 'Kazka';
      const body = document.createElement('span');
      body.className = 'body';
      currentMsg.appendChild(label);
      currentMsg.appendChild(body);
      log.appendChild(currentMsg);
      return body;
    }

    function appendError(text) {
      const el = document.createElement('div');
      el.className = 'msg error';
      el.textContent = text;
      log.appendChild(el);
      log.scrollTop = log.scrollHeight;
    }

    let thinkingEl = null;

    function connect() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}/ws`);

      ws.onopen = () => {
        // Clear the log before catch-up replay to avoid duplicates on reconnect
        log.innerHTML = '';
        currentMsg = null;
        lastUserEl = null;
        if (thinkingEl) { thinkingEl = null; }

        setStatus('connected', 'Connected');
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
        reconnectDelay = 1000;
      };

      ws.onclose = () => {
        setStatus('', 'Disconnected — reconnecting…');
        input.disabled = true;
        sendBtn.disabled = true;
        currentMsg = null;
        setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 15000);
      };

      ws.onerror = () => ws.close();

      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);

        if (msg.type === 'user_input') {
          appendUser(msg.text);

        } else if (msg.type === 'chunk') {
          // Remove any lingering thinking bubble
          if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }

          if (!currentMsg) {
            const body = startAssistantBubble(msg.source);
            currentMsg._body = body;
          }
          const body = currentMsg._body || currentMsg.querySelector('.body');
          body.textContent += msg.text;
          if (msg.is_final) { currentMsg = null; }
          log.scrollTop = log.scrollHeight;

        } else if (msg.type === 'thinking') {
          if (!thinkingEl) {
            thinkingEl = document.createElement('div');
            thinkingEl.className = 'msg thinking';
            thinkingEl.textContent = '💭 ';
            log.appendChild(thinkingEl);
          }
          thinkingEl.textContent += msg.text;
          log.scrollTop = log.scrollHeight;

        } else if (msg.type === 'state') {
          const stateMap = {
            LISTENING: ['listening', 'Listening…'],
            PROCESSING_VAD: ['processing', 'Processing…'],
            PROCESSING_PTT: ['processing', 'Processing…'],
            VERIFYING: ['listening', 'Verifying…'],
            SPEAKING: ['speaking', 'Speaking…'],
            WAITING: ['connected', 'Ready'],
          };
          const [cls, text] = stateMap[msg.state] || ['connected', msg.state];
          setStatus(cls, text);
          // A new state means a new response is coming — close any open bubble
          if (msg.state === 'PROCESSING_VAD' || msg.state === 'PROCESSING_PTT') {
            currentMsg = null;
          }

        } else if (msg.type === 'clear') {
          log.innerHTML = '';
          currentMsg = null;
          lastUserEl = null;
          if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }

        } else if (msg.type === 'undo_last') {
          // Remove messages from the end: assistant bubble(s), then user bubble
          while (log.lastChild && !log.lastChild.classList.contains('user')) {
            log.removeChild(log.lastChild);
          }
          if (log.lastChild && log.lastChild.classList.contains('user')) {
            log.removeChild(log.lastChild);
          }
          currentMsg = null;
          if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
          // Update lastUserEl to the new last user bubble
          const userMsgs = log.querySelectorAll('.msg.user');
          lastUserEl = userMsgs.length ? userMsgs[userMsgs.length - 1] : null;
          if (lastUserEl) lastUserEl.classList.add('editable');

        } else if (msg.type === 'error') {
          appendError('Error: ' + msg.message);
        }
      };
    }

    function send() {
      const text = input.value.trim();
      if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({ type: 'text_input', text }));
      appendUser(text);
      input.value = '';
      currentMsg = null;  // next assistant message is a fresh bubble
    }

    sendBtn.addEventListener('click', send);
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });

    connect();
  </script>
</body>
</html>
"""


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

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def _register_routes(self):
        app = self.app

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return HTMLResponse(content=_HTML)

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
                        if text:
                            # Record user message in history for catch-up
                            user_msg = {"type": "user_input", "text": text}
                            with self._history_lock:
                                self._history.append(user_msg)
                            if self._input_callback:
                                self._input_callback(text)

                    elif msg_type == "edit_last":
                        text = msg.get("text", "").strip()
                        if text and self._edit_callback:
                            # Callback undoes old exchange (broadcasts undo_last),
                            # then re-submits the edited text to the engine.
                            self._edit_callback(text)
                            # Record + broadcast the new user message so all
                            # clients render it after the undo clears the old one.
                            user_msg = {"type": "user_input", "text": text}
                            with self._history_lock:
                                self._history.append(user_msg)
                            self.broadcast(user_msg)

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

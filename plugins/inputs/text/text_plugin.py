"""
Text Input Plugin - Keyboard/stdin text input.

Provides text input mode via keyboard:
- Press 't' to enter text input mode
- Type message and press Enter to send
- Press 'q' for graceful shutdown

Cross-platform support for Windows (msvcrt) and Unix (termios).
"""

import sys
import time
import threading
from typing import Dict, Any, Optional, Callable

from core.plugin_base import InputPlugin


class TextInputPlugin(InputPlugin):
    """
    Text input plugin for keyboard/stdin interaction.

    Provides a keyboard listener that:
    - 't': Enter text input mode (type message, Enter to send)
    - 'q': Trigger graceful shutdown
    - 'p': Push-to-talk toggle (sends to voice plugin)

    Works cross-platform (Windows and Unix-like systems).
    """

    def __init__(self, engine: 'AssistantEngine', plugin_config: Dict[str, Any] = None):
        """
        Initialize text input plugin.

        Args:
            engine: Reference to the engine
            plugin_config: Optional configuration overrides
                - quit_key: Key for graceful shutdown (default: 'q')
                - ptt_key: Key for PTT toggle (default: 'p')
                - text_key: Key for text input mode (default: 't')
        """
        super().__init__(engine, "text")
        self.plugin_config = plugin_config or {}

        # Key bindings
        self.quit_key = self.plugin_config.get('quit_key', 'q')
        self.ptt_key = self.plugin_config.get('ptt_key', 'p')
        self.text_key = self.plugin_config.get('text_key', 't')
        self.undo_key = self.plugin_config.get('undo_key', 'u')
        self.debug_key = self.plugin_config.get('debug_key', 'd')

        # State
        self.running = False
        self.listener_thread = None

        # Callbacks for shutdown
        self._shutdown_callback: Optional[Callable[[], None]] = None

        # Register endpoint for programmatic text input
        self._endpoint_registered = False

    def start(self):
        """Start text input plugin and keyboard listener."""
        print(f"⌨️  Starting text input plugin...")
        print(f"   Keys: '{self.text_key}'=text input, '{self.undo_key}'=undo last, '{self.ptt_key}'=PTT, '{self.quit_key}'=quit")
        self.running = True

        # Register endpoint for programmatic text input
        self.engine.register_endpoint("text", "submit", self._on_submit)
        self._endpoint_registered = True

        # Start keyboard listener thread
        self.listener_thread = threading.Thread(
            target=self._keyboard_loop,
            daemon=True,
            name="TextInputListener"
        )
        self.listener_thread.start()

        print("✅ Text input plugin ready")

    def stop(self):
        """Stop text input plugin."""
        print("🛑 Stopping text input plugin...")
        self.running = False

        if self.listener_thread and self.listener_thread.is_alive():
            # Thread is daemon, will stop when main program exits
            # We give it a moment to clean up
            self.listener_thread.join(timeout=1.0)

        print("   Text input stopped")

    def on_shutdown(self, callback: Callable[[], None]):
        """
        Register callback for shutdown key press.

        Args:
            callback: Function to call when quit key is pressed
        """
        self._shutdown_callback = callback

    def _on_submit(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle programmatic text input via endpoint.

        Args:
            data: {'text': 'message to send'}

        Returns:
            {'status': 'submitted'}
        """
        text = data.get('text', '')
        if text and text.strip():
            self.emit_input(text.strip(), {'source': 'TEXT'})
        return {'status': 'submitted'}

    def submit(self, text: str):
        """
        Programmatically submit text input.

        Args:
            text: Text to submit
        """
        if text and text.strip():
            self.emit_input(text.strip(), {'source': 'TEXT'})

    def _keyboard_loop(self):
        """Main keyboard listener loop - cross-platform."""
        try:
            import msvcrt
            is_windows = True
        except ImportError:
            is_windows = False

        if is_windows:
            self._keyboard_loop_windows()
        else:
            self._keyboard_loop_unix()

    def _keyboard_loop_windows(self):
        """Windows keyboard listener using msvcrt."""
        import msvcrt

        while self.running:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()

                if ch == self.quit_key:
                    print(f"\n'{self.quit_key}' pressed: graceful shutdown...")
                    if self._shutdown_callback:
                        self._shutdown_callback()
                    break

                elif ch == self.ptt_key:
                    print(f"\n'{self.ptt_key}' pressed: PTT toggle")
                    self._handle_ptt_toggle()

                elif ch == self.undo_key:
                    print(f"\n'{self.undo_key}' pressed: undoing last turn")
                    self.engine.undo_turn()

                elif ch == self.text_key:
                    self._handle_text_input_windows()

                elif ch == self.debug_key:
                    self._dump_llm_context()

            time.sleep(0.05)

    def _handle_text_input_windows(self):
        """Handle text input mode on Windows."""
        import msvcrt

        print(f"\n'{self.text_key}' pressed: entering text input mode")
        print("Type your message and press Enter to send, or Esc to cancel:")

        line = ""
        while self.running:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()

                if ch == '\r':  # Enter key
                    if line.strip():
                        print(f"\n📝 Text input: '{line.strip()}'")
                        self.emit_input(line.strip(), {'source': 'TEXT'})
                    else:
                        print("\n❌ Empty message, cancelled")
                    break

                elif ch == '\x1b':  # Esc key
                    print("\n❌ Text input cancelled")
                    break

                elif ch == '\x08':  # Backspace
                    if line:
                        line = line[:-1]
                        sys.stdout.write('\b \b')
                        sys.stdout.flush()

                else:
                    line += ch
                    sys.stdout.write(ch)
                    sys.stdout.flush()

            time.sleep(0.05)

    def _keyboard_loop_unix(self):
        """Unix keyboard listener using termios."""
        import tty
        import termios
        import select

        fd = sys.stdin.fileno()

        try:
            old_settings = termios.tcgetattr(fd)
        except termios.error:
            # Not a TTY (e.g., running in a pipe or test)
            print("   ⚠️  No TTY available, keyboard input disabled")
            return

        try:
            tty.setcbreak(fd)

            while self.running:
                # Use select with timeout for non-blocking read
                dr, _, _ = select.select([sys.stdin], [], [], 0.1)

                if dr:
                    ch = sys.stdin.read(1)

                    if ch == self.quit_key:
                        print(f"\n'{self.quit_key}' pressed: graceful shutdown...")
                        if self._shutdown_callback:
                            self._shutdown_callback()
                        break

                    elif ch == self.ptt_key:
                        print(f"\n'{self.ptt_key}' pressed: PTT toggle")
                        self._handle_ptt_toggle()

                    elif ch == self.undo_key:
                        print(f"\n'{self.undo_key}' pressed: undoing last turn")
                        self.engine.undo_turn()

                    elif ch == self.text_key:
                        self._handle_text_input_unix(fd, old_settings)

                    elif ch == self.debug_key:
                        self._dump_llm_context()

        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    def _handle_text_input_unix(self, fd, old_settings):
        """Handle text input mode on Unix."""
        import tty
        import termios

        print(f"\n'{self.text_key}' pressed: entering text input mode")
        print("Type your message and press Enter to send, or Ctrl-C to cancel:")

        # Switch to normal mode for line input
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        try:
            line = sys.stdin.readline().strip()
            if line:
                print(f"\n📝 Text input: '{line}'")
                self.emit_input(line, {'source': 'TEXT'})
            else:
                print("\n❌ Empty message, cancelled")
        except KeyboardInterrupt:
            print("\n❌ Text input cancelled")
        finally:
            # Switch back to cbreak mode
            tty.setcbreak(fd)

        print(f"\n✅ Resumed listening for keyboard input")

    def _handle_ptt_toggle(self):
        """Handle PTT toggle - send to voice plugin."""
        # Check if voice plugin has PTT endpoint
        if self.engine.has_endpoint("voice", "ptt_started"):
            # Toggle PTT - this is a simple toggle, not hold-to-talk
            # For keyboard, we just send started/stopped in sequence
            self.engine.endpoint_send("voice", "ptt_started", {'source': 'keyboard'})
            print("   (Press 'p' again when done speaking)")

    def _dump_llm_context(self):
        """Dump the full LLM context (system prompt, history, tools) to console for debugging."""
        import json

        print(f"\n{'=' * 60}")
        print("FULL LLM CONTEXT")
        print(f"{'=' * 60}")

        # System prompt
        prompt = self.engine.llm_interface.system_prompt
        print(f"\n--- SYSTEM PROMPT ({len(prompt)} chars) ---")
        print(prompt)

        # Conversation history
        history = self.engine.conversation_manager.conversation_history
        print(f"\n--- CONVERSATION HISTORY ({len(history)} messages) ---")
        for i, msg in enumerate(history):
            content = msg.content if msg.content is not None else ""
            # Truncate long messages for readability
            content_str = str(content) if isinstance(content, list) else content
            preview = content_str[:200] + "..." if len(content_str) > 200 else content_str
            print(f"  [{i}] {msg.role}: {preview}")

        # Tools
        if self.engine.tool_manager and self.engine.config.tools.enable_tools:
            tools = self.engine.tool_manager.get_openai_tools()
            tool_names = [t["function"]["name"] for t in tools if t.get("type") == "function"]
            print(f"\n--- TOOLS ({len(tool_names)}) ---")
            for name in tool_names:
                print(f"  - {name}")
        else:
            print(f"\n--- TOOLS (disabled) ---")

        print(f"\n{'=' * 60}")

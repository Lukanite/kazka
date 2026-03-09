"""
Console output plugin - prints responses to stdout.

Uses QueuedOutputPlugin to preserve fragment ordering.
"""

from typing import Dict, Any
from core.plugin_base import QueuedOutputPlugin
from core.config import config


class ConsoleOutputPlugin(QueuedOutputPlugin):
    """
    Console output plugin for printing responses to stdout.

    Prints each response fragment with metadata context.
    Uses queued processing to preserve ordering.
    """

    def __init__(self, engine: 'AssistantEngine', plugin_config: Dict[str, Any]):
        """
        Initialize console output plugin.

        Args:
            engine: Reference to the engine
            plugin_config: Plugin configuration (unused for console)
        """
        super().__init__(engine, "console")
        self.plugin_config = plugin_config
        self._in_thinking = False

    def start_internal(self):
        """Initialize console output (no resources needed)."""
        print("✅ Console output plugin started")

    def _process_output(self, text: str, metadata: Dict[str, Any]):
        """
        Print output to console.

        Args:
            text: Output text/fragment
            metadata: Output metadata (contains source, plugin, etc.)
        """
        if metadata.get('is_thinking'):
            if not config.console.show_thinking:
                return
            print(f"   💭 THINKING: \"{text}\"")
            return

        # Extract context from metadata
        source = metadata.get('source', 'UNKNOWN')

        # Print with context
        print(f"   🤖 KAZKA [{source}]: \"{text}\"")

    def output_chunk(self, text: str, metadata: Dict[str, Any], is_final: bool = False):
        """
        Print streaming chunks immediately for real-time display.

        Overrides default buffering behavior to print tokens as they arrive.
        Uses console lock to prevent other threads from interleaving output.
        Handles thinking chunks with a distinct prefix when show_thinking is enabled.

        Args:
            text: Chunk of output text
            metadata: Output metadata
            is_final: True if this is the last chunk
        """
        is_thinking = metadata.get('is_thinking', False)

        # Handle thinking chunks
        if is_thinking:
            if not config.console.show_thinking:
                return

            # Start thinking display (acquire lock on first thinking chunk)
            if not self._in_thinking:
                self.lock_console()
                self._in_thinking = True
                self.print("   💭 THINKING: \"", end='', flush=True, lock=False)

            self.print(text, end='', flush=True, lock=False)
            return

        # Content chunk handling below this point

        # Transition from thinking to content - close thinking line, keep lock held
        was_thinking = self._in_thinking
        if was_thinking:
            self.print("\"", lock=False)  # Close thinking quote
            self._in_thinking = False

        # Start content display
        if not self._chunk_buffer:
            # Acquire lock only if we don't already hold it from thinking
            if not was_thinking:
                self.lock_console()
            source = metadata.get('source', 'UNKNOWN')
            self.print(f"   🤖 KAZKA [{source}]: \"", end='', flush=True, lock=False)

        # Print chunk immediately (no newline, lock already held)
        self.print(text, end='', flush=True, lock=False)
        self._chunk_buffer += text  # Track that we've started

        if is_final:
            self.print("\"", lock=False)  # Close quote and newline
            self._chunk_buffer = ""
            self._metadata_buffer = {}  # Clear metadata buffer too
            self.unlock_console()
            # Don't call parent's output() - we already printed everything

    def should_handle(self, metadata: Dict[str, Any]) -> bool:
        """
        Console handles all output.

        Args:
            metadata: Output metadata

        Returns:
            True (always handle)
        """
        return True

    def stop_internal(self):
        """Cleanup console output (no resources to release)."""
        print("🛑 Console output plugin stopped")

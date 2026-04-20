"""
Base plugin classes for inputs and outputs.

Provides the foundation for all input/output plugins with:
- Thread-safe communication via engine request queue
- QueuedOutputPlugin pattern for sequential processing
- Clean lifecycle management (start/stop)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from queue import Queue, Empty
import threading


class InputPlugin(ABC):
    """
    Base class for input plugins (voice, button, text, etc.).

    Input plugins produce user input that flows to the engine.
    They run in their own threads and submit input via emit_input().
    """

    def __init__(self, engine: 'AssistantEngine', name: str):
        """
        Initialize input plugin.

        Args:
            engine: Reference to the engine for submitting requests
            name: Plugin name (e.g., "voice", "button", "text")
        """
        self.engine = engine
        self.name = name

    @abstractmethod
    def start(self):
        """
        Start the plugin (called during engine startup).

        Override to:
        - Register endpoints
        - Start background threads
        - Initialize hardware/resources
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop the plugin (called during engine shutdown).

        Override to:
        - Stop background threads
        - Release hardware/resources
        - Clean up state
        """
        pass

    def emit_input(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Submit user input to the engine (thread-safe).

        This is the primary way input plugins communicate with the engine.
        Automatically adds plugin name to metadata.

        Args:
            text: User input text
            metadata: Context about the input (source, confidence, etc.)
        """
        if metadata is None:
            metadata = {}

        # Automatically tag input with plugin name
        metadata['plugin'] = self.name

        # Submit to engine queue (thread-safe)
        self.engine.process_input(text, metadata)

    def print(self, *args, lock: bool = True, **kwargs):
        """
        Print with optional synchronization to avoid interleaving with streaming output.

        Args:
            *args: Arguments to pass to print()
            lock: If True, acquire print lock (default). Set False for unsynchronized print.
            **kwargs: Keyword arguments to pass to print()
        """
        if lock:
            self.engine.print(*args, **kwargs)
        else:
            print(*args, **kwargs)


class OutputPlugin(ABC):
    """
    Base class for output plugins (TTS, console, LED, etc.).

    Output plugins receive assistant responses and present them to the user.
    They are called by the engine thread when output is broadcast.

    Supports two output modes:
    - output(): Receives complete text (default for non-streaming)
    - output_chunk(): Receives streaming chunks (opt-in for real-time display)

    By default, output_chunk() buffers chunks and calls output() when complete.
    Override output_chunk() to handle streaming chunks immediately.
    """

    def __init__(self, engine: 'AssistantEngine', name: str):
        """
        Initialize output plugin.

        Args:
            engine: Reference to the engine
            name: Plugin name (e.g., "tts", "console", "led")
        """
        self.engine = engine
        self.name = name
        self._chunk_buffer = ""
        self._metadata_buffer: Dict[str, Any] = {}

    @abstractmethod
    def start(self):
        """
        Start the plugin (called during engine startup).

        Override to:
        - Register endpoints
        - Start background threads
        - Initialize hardware/resources
        """
        pass

    @abstractmethod
    def output(self, text: str, metadata: Dict[str, Any]):
        """
        Handle output from the engine.

        Called by the engine thread when output is broadcast.
        For blocking operations (like TTS), use QueuedOutputPlugin instead.

        Args:
            text: Output text or fragment
            metadata: Context about the output (source, fragment_type, etc.)
        """
        pass

    def should_handle(self, metadata: Dict[str, Any]) -> bool:
        """
        Determine if this plugin should handle the output.

        Override to filter based on metadata (e.g., TTS only for voice input).
        Default: handle all output.

        Args:
            metadata: Output metadata

        Returns:
            True if this plugin should process the output
        """
        return True

    def output_chunk(self, text: str, metadata: Dict[str, Any], is_final: bool = False):
        """
        Handle a streaming chunk of output.

        Default behavior: buffer chunks and call output() when is_final=True.
        Override this method to handle chunks immediately (e.g., for real-time
        console streaming).

        Args:
            text: Chunk of output text
            metadata: Output metadata (accumulated via .update())
            is_final: True if this is the last chunk
        """
        self._chunk_buffer += text
        self._metadata_buffer.update(metadata)

        if is_final:
            self.output(self._chunk_buffer, self._metadata_buffer)
            self._chunk_buffer = ""
            self._metadata_buffer = {}

    def print(self, *args, lock: bool = True, **kwargs):
        """
        Print with optional synchronization to avoid interleaving with streaming output.

        Args:
            *args: Arguments to pass to print()
            lock: If True, acquire print lock (default). Set False for unsynchronized print.
            **kwargs: Keyword arguments to pass to print()
        """
        if lock:
            self.engine.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def lock_console(self):
        """
        Acquire the console print lock for extended streaming output.

        Use when outputting multiple print statements that should not be
        interleaved (e.g., streaming tokens). Must be paired with unlock_console().
        """
        self.engine.acquire_print_lock()

    def unlock_console(self):
        """
        Release the console print lock after streaming output is complete.

        Must be called after lock_console() when streaming is done.
        """
        self.engine.release_print_lock()

    @abstractmethod
    def stop(self):
        """
        Stop the plugin (called during engine shutdown).

        Override to:
        - Stop background threads
        - Release hardware/resources
        - Drain queues
        """
        pass


class QueuedOutputPlugin(OutputPlugin):
    """
    Base class for output plugins that need sequential processing.

    Use this for outputs that:
    - Block for significant time (TTS speaking 1-2s)
    - Must preserve fragment ordering
    - Can't run on the engine thread

    The pattern:
    1. output() queues the fragment (returns immediately)
    2. Worker thread processes queue sequentially
    3. _process_output() can block as long as needed
    """

    def __init__(self, engine: 'AssistantEngine', name: str):
        """
        Initialize queued output plugin.

        Args:
            engine: Reference to the engine
            name: Plugin name
        """
        super().__init__(engine, name)

        # Internal queue for sequential processing
        self.output_queue: Queue = Queue()

        # Worker thread state
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        """
        Start the plugin and worker thread.

        Override start_internal() for plugin-specific initialization.
        """
        self.running = True

        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"{self.name}_worker"
        )
        self.worker_thread.start()

        # Call subclass initialization
        self.start_internal()

    @abstractmethod
    def start_internal(self):
        """
        Plugin-specific initialization.

        Override to:
        - Register endpoints
        - Initialize resources
        - Load configuration
        """
        pass

    def output(self, text: str, metadata: Dict[str, Any]):
        """
        Queue output for processing (returns immediately).

        Called by engine thread. Queues the output and returns
        so the engine stays responsive.

        Args:
            text: Output text/fragment
            metadata: Output metadata
        """
        self.output_queue.put({
            'text': text,
            'metadata': metadata
        })

    def _worker_loop(self):
        """
        Worker thread that processes queued output sequentially.

        Internal method - do not override.
        """
        while self.running:
            try:
                # Block waiting for output (with timeout for clean shutdown)
                item = self.output_queue.get(timeout=0.5)

                if item is None:  # Poison pill for shutdown
                    break

                # Process output (can block as long as needed)
                self._process_output(item['text'], item['metadata'])

            except Empty:
                # Timeout is normal - just continue loop
                continue

            except Exception as e:
                print(f"❌ {self.name} worker error: {e}")
                import traceback
                traceback.print_exc()

    @abstractmethod
    def _process_output(self, text: str, metadata: Dict[str, Any]):
        """
        Process output fragment (can block).

        Override this to implement the actual output logic.
        This runs in the worker thread and can take as long as needed.

        Args:
            text: Output text/fragment
            metadata: Output metadata
        """
        pass

    def stop(self):
        """
        Stop the plugin and drain queue.

        Sends poison pill and waits for worker thread to finish.
        """
        # Send poison pill to signal shutdown (worker will drain queue first)
        self.output_queue.put(None)

        # Wait for worker to finish processing and exit
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

            if self.worker_thread.is_alive():
                print(f"⚠️  {self.name} worker thread did not stop cleanly")

        # Now mark as stopped
        self.running = False

        # Call subclass cleanup
        self.stop_internal()

    def stop_internal(self):
        """
        Plugin-specific cleanup.

        Override to release resources, close connections, etc.
        Default: do nothing.
        """
        pass

    def get_queue_size(self) -> int:
        """
        Get current queue size (for testing/monitoring).

        Returns:
            Number of items in queue
        """
        return self.output_queue.qsize()


class ServicePlugin(ABC):
    """
    Base class for service plugins that observe engine state.

    Service plugins receive lifecycle notifications from the engine
    but don't produce input or consume output directly. They react
    to engine events (interactions, startup, shutdown, sleep).
    """

    def __init__(self, engine: 'AssistantEngine', name: str):
        """
        Initialize service plugin.

        Args:
            engine: Reference to the engine
            name: Plugin name (e.g., "sleep_watchdog")
        """
        self.engine = engine
        self.name = name

    @abstractmethod
    def start(self):
        """
        Start the plugin (called during engine startup).

        Override to initialize timers, state, etc.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop the plugin (called during engine shutdown).

        Override to cancel timers, clean up state, etc.
        """
        pass

    def on_interaction_start(self):
        """
        Called on the engine thread when user input arrives, before LLM processing.

        Override to pause idle tracking. Default: no-op.
        """
        pass

    def on_interaction_end(self):
        """
        Called on the engine thread after a user interaction is fully processed.

        Override to resume idle tracking. Default: no-op.
        """
        pass

    def on_sleep_complete(self):
        """
        Called on the engine thread after a sleep cycle completes.

        Override to reset state after sleep. Default: no-op.
        """
        pass

    def on_undo(self):
        """
        Called on the engine thread after a turn is successfully undone.

        Override to react to undo events (e.g. sync UI state). Default: no-op.
        """
        pass

"""
Request/Response objects for the engine queue system.

All communication with the engine goes through these request objects,
enabling thread-safe operation without locks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Callable
from queue import Queue
import time
import uuid


@dataclass
class EngineRequest(ABC):
    """
    Base class for all engine requests.

    Each request is a unit of work for the engine to process.
    Requests are submitted to the engine's queue and executed sequentially
    on the engine thread.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    # For synchronous requests: caller can wait on this queue
    response_queue: Optional[Queue] = None

    @abstractmethod
    def execute(self, engine: 'AssistantEngine') -> Any:
        """
        Execute this request on the engine.

        Args:
            engine: The engine instance to operate on

        Returns:
            Result of the request (will be put in response_queue if sync)
        """
        pass

    def is_sync(self) -> bool:
        """Check if this is a synchronous request (needs response)."""
        return self.response_queue is not None


@dataclass
class ProcessInputRequest(EngineRequest):
    """
    Request to process user input text.

    This triggers the full LLM query pipeline:
    1. Check for special commands (memory, etc.)
    2. Query LLM with tools
    3. Stream response fragments to output plugins
    4. Log conversation to memory
    """
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: List[Dict[str, Any]] = field(default_factory=list)

    def execute(self, engine: 'AssistantEngine') -> None:
        """Process user input through LLM pipeline."""
        engine._process_input_internal(self.text, self.metadata, self.images)
        return None  # No return value for input processing


@dataclass
class EndpointMessageRequest(EngineRequest):
    """
    Request to send a message to a component endpoint.

    Enables 1-1 communication between plugins via the endpoint system.
    Example: Button plugin sends "wake_requested" to Voice plugin.
    """
    target: str = ""  # Component name (e.g., "voice")
    endpoint: str = ""  # Endpoint name (e.g., "wake_requested")
    data: Dict[str, Any] = field(default_factory=dict)

    def execute(self, engine: 'AssistantEngine') -> Any:
        """Route message to target endpoint and return result."""
        return engine._dispatch_endpoint_internal(self.target, self.endpoint, self.data)


@dataclass
class RegisterEndpointRequest(EngineRequest):
    """
    Request to register a component endpoint.

    Components register endpoints during startup to expose functionality
    to other components.
    """
    component_name: str = ""
    endpoint_name: str = ""
    callback: Optional[Callable] = None

    def execute(self, engine: 'AssistantEngine') -> None:
        """Register endpoint in engine's registry."""
        engine._register_endpoint_internal(
            self.component_name,
            self.endpoint_name,
            self.callback
        )
        return None


@dataclass
class BroadcastOutputRequest(EngineRequest):
    """
    Request to send output to all output plugins.

    Used internally by _process_input_internal() to stream LLM response
    fragments to output plugins (TTS, console, etc.).
    """
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def execute(self, engine: 'AssistantEngine') -> None:
        """Broadcast to all output plugins."""
        engine._broadcast_output_internal(self.text, self.metadata)
        return None


@dataclass
class ShutdownRequest(EngineRequest):
    """
    Request to shutdown the engine.

    Triggers graceful shutdown:
    1. Stop accepting new requests
    2. Stop all plugins
    3. Save memories (if requested)
    4. Exit engine loop
    """
    save_memories: bool = True

    def execute(self, engine: 'AssistantEngine') -> None:
        """Initiate shutdown sequence."""
        engine._shutdown_internal(self.save_memories)
        return None


@dataclass
class SleepRequest(EngineRequest):
    """
    Request to perform a sleep cycle (memory flush + conversation reset).

    Triggers:
    1. Extract and save memories from conversation log
    2. Clear conversation history
    3. Rebuild system prompt with fresh memories
    """

    def execute(self, engine: 'AssistantEngine') -> None:
        """Perform sleep cycle."""
        engine._sleep_internal()
        return None


@dataclass
class UndoTurnRequest(EngineRequest):
    """
    Request to undo the last conversation turn.

    Removes the last user message and all subsequent assistant/tool messages
    from conversation history, and removes the corresponding entry from
    the memory manager's conversation log.
    """

    def execute(self, engine: 'AssistantEngine') -> None:
        """Undo the last conversation turn."""
        engine._undo_turn_internal()
        return None


@dataclass
class WakeRequest(EngineRequest):
    """
    Request to handle a self-wake timer firing.

    Injects a tool message into conversation and queries the LLM,
    allowing the assistant to re-engage in conversation after a scheduled delay.
    Runs on the engine thread to avoid race conditions with active conversations.
    """
    timer_id: str = ""
    delay_description: str = ""

    def execute(self, engine: 'AssistantEngine') -> None:
        """Handle wake-up on the engine thread."""
        engine._wake_internal(self.timer_id, self.delay_description)
        return None

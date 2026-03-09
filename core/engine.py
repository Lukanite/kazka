"""
Core Assistant Engine with request queue architecture.

The engine coordinates all components via a thread-safe request queue,
eliminating the need for locks and enabling clean plugin separation.
"""

from typing import Dict, Any, Optional, Callable
from queue import Queue, Empty
import threading
import time

from core.requests import (
    EngineRequest,
    ProcessInputRequest,
    EndpointMessageRequest,
    RegisterEndpointRequest,
    BroadcastOutputRequest,
    ShutdownRequest,
    SleepRequest,
    UndoTurnRequest
)
from core.llm_interface import LLMInterface, ConversationManager
from core.tool_manager import ToolManager
from core.memory_manager import MemoryManager
from core.config import config

# Import Anthropic implementations conditionally
from core.anthropic_llm_interface import AnthropicLLMInterface, AnthropicConversationManager


class AssistantEngine:
    """
    Core orchestrator with fully decoupled request/response architecture.

    Thread Safety:
    - All public methods submit requests to queue (thread-safe)
    - Engine loop processes requests sequentially (single-threaded execution)
    - No locks needed - queue handles synchronization
    """

    def __init__(self, engine_config=None):
        """
        Initialize engine with LLM and tool systems.

        Args:
            engine_config: Optional config override (uses global config if None)
        """
        # Core configuration
        self.config = engine_config or config

        # Initialize LLM systems - choose appropriate backend based on config
        if self.config.network.api_type.lower() == "anthropic":
            print("🤖 Using Anthropic Messages API backend")
            self.llm_interface = AnthropicLLMInterface()
            self.conversation_manager = AnthropicConversationManager(self.llm_interface)
        else:
            print("🤖 Using OpenAI-compatible API backend")
            self.llm_interface = LLMInterface()
            self.conversation_manager = ConversationManager(self.llm_interface)

        self.tool_manager = ToolManager() if self.config.tools.enable_tools else None

        # Memory manager for persistent conversation context
        self.memory_manager = MemoryManager(self.config.memory.file_path)

        # Conversation search index (initialized in _initialize_tools if enabled)
        self.search_index = None

        # Endpoint registry (only accessed by engine thread)
        self.endpoints: Dict[str, Dict[str, Callable]] = {}
        # Structure: {"voice": {"wake_requested": callback}}

        # Plugin registries (only accessed by engine thread)
        self.input_plugins: Dict[str, 'InputPlugin'] = {}
        self.output_plugins: Dict[str, 'OutputPlugin'] = {}
        self.service_plugins: Dict[str, 'ServicePlugin'] = {}

        # Request queue (thread-safe - accessed by all threads)
        self.request_queue: Queue[EngineRequest] = Queue()

        # Engine state
        self.running = False
        self.engine_thread: Optional[threading.Thread] = None

        # Print lock for synchronized console output
        # Used to prevent interleaving when streaming output
        self._print_lock = threading.Lock()

    # =========================================================================
    # Public API (Thread-Safe - Submit Requests to Queue)
    # =========================================================================

    def process_input(self, text: str, metadata: Optional[Dict] = None):
        """
        Submit user input for processing (async).
        Thread-safe - can be called from any thread.

        Args:
            text: User input text
            metadata: Context about the input (source, confidence, etc.)
        """
        if metadata is None:
            metadata = {}

        request = ProcessInputRequest(text=text, metadata=metadata)
        self.request_queue.put(request)

    def undo_turn(self):
        """
        Undo the last conversation turn (async).
        Thread-safe - can be called from any thread.
        """
        self.request_queue.put(UndoTurnRequest())

    def endpoint_send(self, target: str, endpoint: str, data: Optional[Dict] = None):
        """
        Send message to component endpoint (async, fire-and-forget).
        Thread-safe - can be called from any thread.

        Args:
            target: Target component name (e.g., "voice")
            endpoint: Endpoint name (e.g., "wake_requested")
            data: Message data
        """
        if data is None:
            data = {}

        request = EndpointMessageRequest(
            target=target,
            endpoint=endpoint,
            data=data
        )
        self.request_queue.put(request)

    def endpoint_call(self, target: str, endpoint: str, data: Optional[Dict] = None,
                     timeout: float = 5.0) -> Optional[Any]:
        """
        Send message to component endpoint and wait for response (sync, blocking).
        Thread-safe - can be called from any thread.

        Args:
            target: Target component name
            endpoint: Endpoint name
            data: Message data
            timeout: Max seconds to wait

        Returns:
            Response from endpoint, or None if timeout/error
        """
        if data is None:
            data = {}

        # Create response queue for sync request
        response_queue = Queue()

        request = EndpointMessageRequest(
            target=target,
            endpoint=endpoint,
            data=data,
            response_queue=response_queue
        )

        self.request_queue.put(request)

        # Wait for response (blocks caller thread)
        try:
            return response_queue.get(timeout=timeout)
        except Empty:
            print(f"⚠️  Endpoint call timeout: {target}.{endpoint}")
            return None

    def register_endpoint(self, component_name: str, endpoint_name: str, callback: Callable):
        """
        Register a component endpoint (synchronous - blocks until registered).
        Thread-safe - can be called from any thread.

        This is synchronous to eliminate race conditions during startup.
        Ensures endpoints exist before plugins finish start().

        Args:
            component_name: Name of component (e.g., "voice")
            endpoint_name: Endpoint identifier (e.g., "wake_requested")
            callback: Function to call when message arrives
        """
        # Check if we're already on the engine thread
        if threading.current_thread() == self.engine_thread:
            # Already on engine thread - register directly
            self._register_endpoint_internal(component_name, endpoint_name, callback)
        else:
            # Submit request and BLOCK until registered
            response_queue = Queue()
            request = RegisterEndpointRequest(
                component_name=component_name,
                endpoint_name=endpoint_name,
                callback=callback,
                response_queue=response_queue
            )
            self.request_queue.put(request)

            # Block until registration complete
            try:
                response_queue.get(timeout=5.0)
            except Empty:
                print(f"⚠️  Endpoint registration timeout: {component_name}.{endpoint_name}")

    def broadcast_output(self, text: str, metadata: Dict):
        """
        Send output to all output plugins (async).
        Thread-safe - can be called from any thread.

        Args:
            text: Output text
            metadata: Context about the output
        """
        request = BroadcastOutputRequest(text=text, metadata=metadata)
        self.request_queue.put(request)

    def shutdown(self, save_memories: bool = True):
        """
        Request engine shutdown (async).
        Thread-safe - can be called from any thread.

        Args:
            save_memories: Whether to save memories before shutdown
        """
        request = ShutdownRequest(save_memories=save_memories)
        self.request_queue.put(request)

    def has_endpoint(self, target: str, endpoint: str) -> bool:
        """
        Check if endpoint exists (read-only query).
        Thread-safe - dict reads are atomic in CPython.

        Args:
            target: Component name
            endpoint: Endpoint name

        Returns:
            True if endpoint exists
        """
        return (target in self.endpoints and
                endpoint in self.endpoints[target])

    # =========================================================================
    # Engine Loop (Runs in Dedicated Thread)
    # =========================================================================

    def run(self):
        """
        Main engine loop - processes requests from queue.
        Runs in dedicated engine thread.
        """
        print("🎙️  Engine thread started!")
        self.running = True

        while self.running:
            try:
                # Block waiting for request (with timeout for clean shutdown)
                try:
                    request = self.request_queue.get(timeout=0.1)
                except Empty:
                    # Timeout is normal - just continue loop
                    continue

                # Execute request
                self._execute_request(request)

            except Exception as e:
                if self.running:  # Only log if not shutting down
                    print(f"⚠️  Engine loop error: {e}")
                    import traceback
                    traceback.print_exc()

        print("🛑 Engine thread stopped")

    def _execute_request(self, request: EngineRequest):
        """
        Execute a request and handle response if needed.

        Args:
            request: The request to execute
        """
        try:
            # Execute the request
            result = request.execute(self)

            # If this was a sync request, send response
            if request.is_sync():
                request.response_queue.put(result)

        except Exception as e:
            print(f"❌ Request execution error ({type(request).__name__}): {e}")
            import traceback
            traceback.print_exc()

            # Send None for sync requests on error
            if request.is_sync():
                request.response_queue.put(None)

    # =========================================================================
    # Internal Methods (Only Called by Engine Thread via Requests)
    # =========================================================================

    def _register_endpoint_internal(self, component_name: str, endpoint_name: str,
                                   callback: Callable):
        """
        Register endpoint (internal - called by engine thread only).

        Args:
            component_name: Component name
            endpoint_name: Endpoint name
            callback: Callback function
        """
        if component_name not in self.endpoints:
            self.endpoints[component_name] = {}

        self.endpoints[component_name][endpoint_name] = callback
        print(f"   📍 Registered: {component_name}.{endpoint_name}")

    def _dispatch_endpoint_internal(self, target: str, endpoint: str,
                                   data: Dict) -> Any:
        """
        Dispatch message to endpoint (internal - called by engine thread only).

        Args:
            target: Target component name
            endpoint: Endpoint name
            data: Message data

        Returns:
            Result from endpoint callback
        """
        # Check if target component exists
        if target not in self.endpoints:
            print(f"⚠️  Unknown component: {target}")
            return None

        # Check if endpoint exists
        if endpoint not in self.endpoints[target]:
            print(f"⚠️  Unknown endpoint: {target}.{endpoint}")
            return None

        # Call the endpoint
        callback = self.endpoints[target][endpoint]

        try:
            return callback(data)
        except Exception as e:
            print(f"❌ Endpoint error: {target}.{endpoint}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_input_internal(self, text: str, metadata: Dict):
        """
        Process user input through LLM (internal - called by engine thread only).

        Queries the LLM with conversation context and tool support.
        Supports both streaming and non-streaming modes based on config.

        Args:
            text: User input text
            metadata: Input metadata
        """
        # Notify service plugins that an interaction is starting
        self._notify_service_plugins("on_interaction_start")

        try:
            # Check if streaming is enabled
            use_streaming = getattr(config.network, 'enable_streaming', False)

            if use_streaming:
                self._process_input_streaming(text, metadata)
            else:
                self._process_input_non_streaming(text, metadata)

        except Exception as e:
            print(f"❌ LLM query error: {e}")
            import traceback
            traceback.print_exc()
            self._broadcast_output_internal(
                "I'm having trouble processing your request right now.",
                metadata
            )

    def _process_input_non_streaming(self, text: str, metadata: Dict):
        """
        Process input with non-streaming LLM responses (original behavior).

        Args:
            text: User input text
            metadata: Input metadata
        """
        from core.llm_interface import ContentChunk, ThinkingChunk

        # Collect content for memory logging
        all_content = []

        # Query LLM with tools (yields ResponseEvent objects)
        for event in self.conversation_manager.query_with_tools(text, self.tool_manager, streaming=False):
            # Broadcast thinking events unconditionally, plugins decide
            if isinstance(event, ThinkingChunk) and event.content:
                thinking_metadata = {**metadata, 'is_thinking': True}
                self._broadcast_output_internal(event.content, thinking_metadata)
            # Only broadcast ContentChunk events with actual content
            elif isinstance(event, ContentChunk) and event.content:
                all_content.append(event.content)
                self._broadcast_output_internal(event.content, metadata)

        # Log conversation for memory processing on shutdown
        if self.memory_manager and all_content:
            combined_response = "".join(all_content)
            self.memory_manager.log_conversation(text, combined_response)

        # Notify service plugins that interaction is complete
        self._notify_service_plugins("on_interaction_end")

    def _process_input_streaming(self, text: str, metadata: Dict):
        """
        Process input with streaming LLM responses for real-time output.

        Args:
            text: User input text
            metadata: Input metadata
        """
        from core.llm_interface import ContentChunk, ThinkingChunk, Complete

        # Accumulate full response for memory logging
        full_response = ""

        # Query LLM with streaming (yields ResponseEvent objects)
        for event in self.conversation_manager.query_with_tools(text, self.tool_manager, streaming=True):
            # Process ThinkingChunk events - broadcast unconditionally, plugins decide
            if isinstance(event, ThinkingChunk):
                thinking_metadata = {**metadata, 'is_thinking': True}
                self._broadcast_output_chunk_internal(event.content, thinking_metadata, is_final=False)

            # Process ContentChunk events
            elif isinstance(event, ContentChunk):
                if event.content:
                    full_response += event.content
                self._broadcast_output_chunk_internal(event.content, metadata, event.is_final)

            # Process Complete event - flush any buffered output
            elif isinstance(event, Complete):
                # Send empty chunk with is_final=True to close out any streaming buffers
                self._broadcast_output_chunk_internal("", metadata, is_final=True)

        # Log conversation for memory processing on shutdown
        if self.memory_manager and full_response:
            self.memory_manager.log_conversation(text, full_response)

        # Notify service plugins that interaction is complete
        self._notify_service_plugins("on_interaction_end")

    def _broadcast_output_internal(self, text: str, metadata: Dict):
        """
        Broadcast to output plugins (internal - called by engine thread only).

        Args:
            text: Output text
            metadata: Output metadata
        """
        for plugin in self.output_plugins.values():
            if plugin.should_handle(metadata):
                plugin.output(text, metadata)

    def _broadcast_output_chunk_internal(self, text: str, metadata: Dict, is_final: bool):
        """
        Broadcast streaming chunk to output plugins (internal - called by engine thread only).

        Args:
            text: Output text chunk
            metadata: Output metadata
            is_final: True if this is the last chunk
        """
        for plugin in self.output_plugins.values():
            if plugin.should_handle(metadata):
                plugin.output_chunk(text, metadata, is_final)

    # =========================================================================
    # Synchronized Print API (Thread-Safe)
    # =========================================================================

    def print(self, *args, **kwargs):
        """
        Thread-safe print that won't interleave with streaming output.

        Use this from plugins instead of built-in print() when the message
        might appear during streaming responses.

        Args:
            *args: Arguments to pass to print()
            **kwargs: Keyword arguments to pass to print()
        """
        with self._print_lock:
            print(*args, **kwargs)

    def acquire_print_lock(self):
        """
        Acquire the print lock for extended streaming output.

        Use this when you need to hold the lock across multiple print calls
        (e.g., streaming chunks). Must be paired with release_print_lock().
        """
        self._print_lock.acquire()

    def release_print_lock(self):
        """
        Release the print lock after streaming output is complete.

        Must be called after acquire_print_lock() when streaming is done.
        """
        self._print_lock.release()

    def _shutdown_internal(self, save_memories: bool):
        """
        Shutdown engine (internal - called by engine thread only).

        Args:
            save_memories: Whether to save memories
        """
        print("🛑 Shutting down engine...")

        # Stop service plugins first (e.g., cancel sleep watchdog timer
        # before memory save to prevent race conditions)
        for plugin in list(self.service_plugins.values()):
            try:
                plugin.stop()
            except Exception as e:
                print(f"⚠️  Error stopping service plugin: {e}")

        if self.memory_manager:
            # Save memories if requested
            if save_memories:
                print("🧠 Processing memories before shutdown...")
                # Save log
                log_path = self.memory_manager.save_conversation_log()
                if log_path and self.search_index:
                    try:
                        self.search_index.index_conversation_log(log_path)
                        self.search_index.save()
                    except Exception as e:
                        print(f"⚠️  Error updating search index: {e}")
                
                # Form memories
                try:
                    self.memory_manager.process_and_save()
                    print(f"✅ Memories saved. {config.assistant.name} will remember you next time!")
                except Exception as e:
                    print(f"❌ Error saving memories: {e}")
                    # Continue with shutdown even if memory saving fails
            else:
                print("🧠 Fast shutdown: skipping memory save.")

        # Stop input and output plugins
        for plugin in list(self.input_plugins.values()) + list(self.output_plugins.values()):
            try:
                plugin.stop()
            except Exception as e:
                print(f"⚠️  Error stopping plugin: {e}")

        # Stop engine loop
        self.running = False

    def _sleep_internal(self):
        """
        Perform a sleep cycle: flush memories and reset conversation.

        This is a full cognitive reset — extract memories from the current
        conversation, save them, then start fresh with a clean conversation
        and the updated memory bank injected into the system prompt.

        Called by engine thread only (via SleepRequest).
        """
        print("😴 Sleep cycle starting...")

        try:
            # 1. Save conversation log to .jsonl, update search index, then process memories
            if self.memory_manager:
                log_path = self.memory_manager.save_conversation_log()
                if log_path and self.search_index:
                    try:
                        self.search_index.index_conversation_log(log_path)
                        self.search_index.save()
                    except Exception as e:
                        print(f"⚠️  Error updating search index: {e}")
                self.memory_manager.process_and_save()

            # 2. Clear conversation history (the LLM message list)
            self.conversation_manager.clear_history()

            # 3. Reset system prompt to clean base (avoid double-injection)
            self.llm_interface.system_prompt = config.assistant.get_system_prompt()

            # 4. Re-initialize memory (loads saved memories, injects into prompt)
            self._initialize_memory()

            # 5. Re-warm LLM cache with new prompt
            self._warmup_llm_cache()

            # 6. Notify service plugins that sleep is complete
            self._notify_service_plugins("on_sleep_complete")

            print("😴 Sleep cycle complete. Conversation reset with fresh memories.")

        except Exception as e:
            print(f"❌ Error during sleep cycle: {e}")
            import traceback
            traceback.print_exc()

    def _undo_turn_internal(self):
        """
        Undo the last conversation turn (internal - called by engine thread only).

        Removes the last user message and all subsequent messages from LLM
        conversation history, and removes the corresponding entry from the
        memory manager's conversation log.
        """
        if not self.conversation_manager.conversation_history:
            print("Nothing to undo.")
            return

        removed = self.conversation_manager.undo_last_turn()
        if not removed:
            print("Nothing to undo.")
            return

        # Also remove the last entry from the memory manager's session log
        if self.memory_manager and self.memory_manager.conversation_log:
            self.memory_manager.conversation_log.pop()

        print("Last turn undone.")

    def _notify_service_plugins(self, event_name: str):
        """
        Notify all service plugins of an event (internal - engine thread only).

        Args:
            event_name: Name of the method to call on each plugin
        """
        for plugin in self.service_plugins.values():
            try:
                handler = getattr(plugin, event_name, None)
                if handler:
                    handler()
            except Exception as e:
                print(f"⚠️  Error notifying service plugin {plugin.name}.{event_name}: {e}")

    # =========================================================================
    # Plugin Management
    # =========================================================================

    def register_input(self, name: str, plugin: 'InputPlugin'):
        """
        Register input plugin (must be called before startup).

        Args:
            name: Plugin name
            plugin: Plugin instance
        """
        self.input_plugins[name] = plugin
        print(f"✅ Registered input: {name}")

    def register_output(self, name: str, plugin: 'OutputPlugin'):
        """
        Register output plugin (must be called before startup).

        Args:
            name: Plugin name
            plugin: Plugin instance
        """
        self.output_plugins[name] = plugin
        print(f"✅ Registered output: {name}")

    def register_service(self, name: str, plugin: 'ServicePlugin'):
        """
        Register service plugin (must be called before startup).

        Args:
            name: Plugin name
            plugin: Plugin instance
        """
        self.service_plugins[name] = plugin
        print(f"✅ Registered service: {name}")

    def startup(self):
        """Initialize all systems and plugins, then start engine thread."""
        print(f"🚀 Starting {config.assistant.name} Assistant Engine...")

        # Load resume history from previous conversation and set initial system prompt
        resume_history = self.memory_manager.get_resume_history(config.memory.resume_history_count)
        self.llm_interface.system_prompt = config.assistant.get_system_prompt(resume_history=resume_history)

        # Initialize memory system
        self._initialize_memory()

        # Initialize tools if enabled
        # Note: Use 'is not None' because ToolManager.__len__ returns 0 when empty
        if self.tool_manager is not None and self.config.tools.enable_tools:
            self._initialize_tools()

        # Warm up LLM cache with the final system prompt (including memories and tools)
        self._warmup_llm_cache()

        # Start engine thread FIRST so it can process endpoint registrations
        self.engine_thread = threading.Thread(target=self.run, daemon=False, name="EngineThread")
        self.engine_thread.start()
        time.sleep(0.1)  # Give engine thread time to start

        # Start all plugins (they submit endpoint registrations to queue)
        for name, plugin in self.input_plugins.items():
            print(f"Starting input plugin: {name}")
            plugin.start()

        for name, plugin in self.output_plugins.items():
            print(f"Starting output plugin: {name}")
            plugin.start()

        for name, plugin in self.service_plugins.items():
            print(f"Starting service plugin: {name}")
            plugin.start()

        print("✅ Engine ready!")

        # Wait a moment for endpoint registrations to process
        time.sleep(0.1)
        print(f"📍 Registered endpoints: {self._list_endpoints()}")

    def _list_endpoints(self) -> str:
        """List all registered endpoints for debugging."""
        lines = []
        for component, endpoints in self.endpoints.items():
            for endpoint in endpoints.keys():
                lines.append(f"{component}.{endpoint}")
        return ", ".join(lines) if lines else "none"

    def _initialize_memory(self):
        """Initialize memory system and inject memories into system prompt."""
        print("🧠 Initializing memory system...")

        try:
            # Load existing memories
            self.memory_manager.load_memories()

            # Inject memories into system prompt (always call to resolve {memory_bank} placeholder)
            recent_memories = self.memory_manager.get_recent_context() if self.memory_manager.has_memories() else []
            current_prompt = self.llm_interface.system_prompt
            enhanced_prompt = self.memory_manager.inject_into_prompt(
                current_prompt, recent_memories
            )
            self.llm_interface.system_prompt = enhanced_prompt

            if recent_memories:
                print(f"   📚 Loaded {self.memory_manager.get_memory_count()} memories")
            else:
                print("   📚 No previous memories found")

        except Exception as e:
            print(f"   ⚠️  Error loading memories: {e}")
            import traceback
            traceback.print_exc()

    def _initialize_tools(self):
        """Initialize and register all available tools."""
        print("🔧 Initializing tools...")

        try:
            # Import tools
            from tools import (
                GetDateTimeTool,
                ScheduleSelfWakeTool,
                CancelSelfWakeTool,
                ListSelfWakesTool,
                MatterLightControlTool,
                MatterListDevicesTool,
                SearchConversationLogsTool,
                ReadConversationContextTool,
                ListConversationsInTimeTool
            )

            # Register tools
            self.tool_manager.register(GetDateTimeTool())

            # Register self-wake tools (uses scheduler service plugin if registered)
            scheduler = self.service_plugins.get("scheduler")
            self.tool_manager.register(ScheduleSelfWakeTool(scheduler))
            self.tool_manager.register(CancelSelfWakeTool(scheduler))
            self.tool_manager.register(ListSelfWakesTool(scheduler))

            # Register Matter light control tools
            matter_config = self.config.tools.tool_settings.get('matter', {})
            matter_host = matter_config.get('host', 'charmander.localdomain')
            matter_port = matter_config.get('port', 5580)
            device_aliases = matter_config.get('device_aliases', {
                'light': {'node_id': 1, 'endpoint_id': 1}
            })
            groups = matter_config.get('groups', {})

            self.tool_manager.register(MatterLightControlTool(
                matter_host=matter_host,
                matter_port=matter_port,
                device_aliases=device_aliases,
                groups=groups
            ))
            self.tool_manager.register(MatterListDevicesTool(
                matter_host=matter_host,
                matter_port=matter_port
            ))
            print(f"   ✅ Matter tools initialized (server: {matter_host}:{matter_port})")

            # Register conversation search tools if enabled
            search_config = self.config.conversation_search
            if search_config.enabled:
                try:
                    from core.conversation_search import ConversationSearchIndex
                    log_dir = self.config.memory.conversation_log_dir or "log"
                    self.search_index = ConversationSearchIndex(
                        index_dir=search_config.index_dir,
                        model_path=search_config.model_path,
                        tokenizer_path=search_config.tokenizer_path,
                        log_dir=log_dir
                    )
                    self.search_index.load()
                    self.tool_manager.register(SearchConversationLogsTool(
                        self.search_index,
                        context_window=search_config.context_window,
                        top_k=search_config.top_k,
                        min_score=search_config.min_score
                    ))
                    self.tool_manager.register(ReadConversationContextTool(
                        self.search_index
                    ))
                    self.tool_manager.register(ListConversationsInTimeTool(
                        log_dir=log_dir
                    ))
                    print(f"   ✅ Conversation search initialized ({self.search_index.get_entry_count()} indexed entries)")
                except Exception as e:
                    print(f"   ⚠️  Conversation search not available: {e}")
                    self.search_index = None

            # Load tool configs from settings
            self.tool_manager.load_tool_configs(self.config.tools.tool_settings)

            print(f"✅ {len(self.tool_manager)} tools initialized")

        except ImportError as e:
            print(f"⚠️  Failed to import tools: {e}")
            print("   Tools will be disabled.")
        except Exception as e:
            print(f"⚠️  Error initializing tools: {e}")
            import traceback
            traceback.print_exc()

    def _warmup_llm_cache(self):
        """
        Warm up the LLM cache with the current system prompt.

        This sends a minimal request with the full system prompt to populate
        the LLM server's prompt cache. Subsequent requests with the same system
        prompt will be faster since the cached prompt can be reused.

        IMPORTANT: Includes tool definitions if tools are enabled, to ensure
        cache alignment with actual queries.
        """
        print("🔥 Warming up LLM cache...")

        try:
            # Get tool definitions if tools are enabled
            tools = None
            if self.tool_manager is not None and self.config.tools.enable_tools:
                tools = self.tool_manager.get_openai_tools()
                print(f"   🔧 Including {len(tools)} tools in warmup for cache alignment")

            # The LLM interface already has the final system prompt set
            success = self.llm_interface.warmup_cache(tools=tools)
            if not success:
                print("   ⚠️  Cache warmup failed or disabled - will proceed normally")
            else:
                print("   ✅ LLM cache warmed up")
        except Exception as e:
            print(f"   ⚠️  Cache warmup error: {e}")

    def _wake_internal(self, timer_id: str, delay_description: str):
        """
        Handle a self-wake timer firing (called by engine thread via WakeRequest).

        Args:
            timer_id: Scheduler's internal timer ID
            delay_description: The human-readable delay that was set
        """
        from core.llm_interface import ContentChunk

        tool_message = f"schedule_self_wake: The wake delay set for '{delay_description}' has now completed (ID: {timer_id})."
        print(f"\n⏰ WAKE-UP TRIGGERED (ID: {timer_id}):")
        print(f"   🔧 Tool message: \"{tool_message}\"")

        # Inject as a system message: the timer is a platform event, not user input.
        # This avoids the spoofing risk of "user" role and is structurally valid
        # (no tool_call_id required, unlike "tool" role).
        self.conversation_manager.add_message("system", tool_message)

        # Empty string signals system-initiated query; the notification is already in history.
        for event in self.conversation_manager.query_with_tools("", self.tool_manager, streaming=False):
            # Only broadcast ContentChunk events with actual content
            if isinstance(event, ContentChunk) and event.content:
                self._broadcast_output_internal(event.content, {'source': 'WAKE_TIMER'})

    # =========================================================================
    # Status & Monitoring
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get engine status for monitoring.

        Returns:
            Status dictionary
        """
        status = {
            'running': self.running,
            'request_queue_size': self.request_queue.qsize(),
            'input_plugins': list(self.input_plugins.keys()),
            'output_plugins': list(self.output_plugins.keys()),
            'endpoints': self._list_endpoints()
        }
        return status

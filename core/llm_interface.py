"""
Language Model Interface module for communication with LLM APIs.
Handles requests to various LLM providers and manages conversation context.
"""

import requests
import json
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
from abc import ABC
from core.config import config
from core.tool_manager import ToolManager


# ============================================================================
# Response Event Classes
# ============================================================================

@dataclass
class ResponseEvent(ABC):
    """Base class for all LLM response events."""
    pass


@dataclass
class ContentChunk(ResponseEvent):
    """
    Text content chunk (streaming) or complete message (non-streaming).

    is_final semantics:
    - True: This is a coherent, complete block of text ready to speak/display
    - False: Intermediate streaming chunk, buffer until is_final=True

    Use Complete event to signal end of conversation turn.
    """
    content: str
    is_final: bool = False

    def __bool__(self):
        """Allow truth testing - empty content is falsy."""
        return bool(self.content)


@dataclass
class ToolCallsRequested(ResponseEvent):
    """LLM has requested one or more tool calls."""
    tool_calls: List[Dict]  # OpenAI format tool calls
    pre_tool_content: Optional[str] = None  # Any text before tool calls


@dataclass
class ToolExecuting(ResponseEvent):
    """About to execute a tool."""
    tool_name: str
    tool_args: str  # JSON string of arguments


@dataclass
class ToolResult(ResponseEvent):
    """Tool execution completed."""
    tool_name: str
    result: str
    success: bool


@dataclass
class ThinkingChunk(ResponseEvent):
    """Reasoning/thinking content from the LLM."""
    content: str


@dataclass
class Continuation(ResponseEvent):
    """Querying LLM again after tool execution."""
    pass


@dataclass
class Complete(ResponseEvent):
    """Conversation turn is complete (no more events)."""
    pass

class LLMInterface:
    """
    Interface for communicating with Language Model APIs.

    This class provides a standardized way to communicate with various LLM APIs,
    handling requests, responses, and error cases.
    """

    def __init__(self, api_url: Optional[str] = None, model: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        Initialize LLM interface.

        Args:
            api_url: API endpoint URL
            model: Model name to use
            system_prompt: System prompt for the conversation
        """
        self.api_url = api_url or config.network.api_url
        self.model = model or config.network.model
        self.api_key = config.network.api_key
        self.system_prompt = system_prompt or config.assistant.get_system_prompt()

    def _build_query_payload(self, user_message: Optional[str], conversation_history: Optional[List[Dict]] = None, tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Build the API request payload (shared by all query methods).

        Args:
            user_message: User's message to send to the LLM (None for continuation queries)
            conversation_history: Optional conversation history to include
            tools: Optional list of tools in OpenAI format for function calling

        Returns:
            Dictionary representing the API request payload
        """
        messages = []

        # Add system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user message if provided (None or "" means no new user turn)
        if user_message:
            messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False  # Will be overridden by caller
        }

        # Add max_tokens for chat queries if configured
        if config.assistant.max_chat_tokens is not None:
            payload["max_tokens"] = config.assistant.max_chat_tokens

        # Add tools if provided (OpenAI function calling format)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Add thinking mode configuration if available
        if config.assistant.enable_thinking_mode:
            payload["chat_template_kwargs"] = {"enable_thinking": True}

        return payload

    def _make_request(self, payload: Dict[str, Any]) -> requests.Response:
        """
        Make HTTP request to the LLM API using default URL.

        Args:
            payload: Request payload

        Returns:
            requests.Response object

        Raises:
            requests.RequestException: If the request fails
        """
        return self._make_request_to_url(payload, self.api_url)

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers, including Authorization if an API key is configured."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _make_request_to_url(self, payload: Dict[str, Any], url: str) -> requests.Response:
        """
        Make HTTP request to the LLM API using specified URL.

        Args:
            payload: Request payload
            url: API endpoint URL to use

        Returns:
            requests.Response object

        Raises:
            requests.RequestException: If the request fails
        """
        headers = self._build_headers()

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
            )
            # Debug: Log response status
            print(f"   🔍 HTTP Status: {response.status_code}")
            if response.status_code != 200:
                print(f"   🔍 Response body: {response.text[:200]}...")
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError:
            raise requests.RequestException("Failed to connect to LLM API")
        except requests.exceptions.HTTPError as e:
            raise requests.RequestException(f"HTTP error: {e.response.status_code} - {e.response.text}")

    def _parse_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Parse the API response to extract message content and tool calls.

        Args:
            response: HTTP response from the API

        Returns:
            Dictionary with 'content' (str or None) and optionally 'tool_calls' (list)

        Raises:
            ValueError: If response format is unexpected
        """
        try:
            response_data = response.json()

            result = {'content': None}

            # Handle OpenAI-compatible format
            if 'choices' in response_data and len(response_data['choices']) > 0:
                message = response_data['choices'][0]['message']

                # Content may be None if there are tool calls
                if 'content' in message:
                    result['content'] = message['content']

                # Extract reasoning/thinking content if present
                if 'reasoning_content' in message and message['reasoning_content']:
                    result['reasoning_content'] = message['reasoning_content']

                # Check for tool calls
                if 'tool_calls' in message and message['tool_calls']:
                    result['tool_calls'] = message['tool_calls']

                return result

            # Handle direct message format
            elif 'message' in response_data:
                message = response_data['message']

                if 'content' in message:
                    result['content'] = message['content']

                if 'reasoning_content' in message and message['reasoning_content']:
                    result['reasoning_content'] = message['reasoning_content']

                if 'tool_calls' in message and message['tool_calls']:
                    result['tool_calls'] = message['tool_calls']

                return result

            # Handle direct content format
            elif 'content' in response_data:
                result['content'] = response_data['content']
                return result

            else:
                raise ValueError("Unexpected response format")

        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON response")

    def _handle_complete_response(self, payload: Dict[str, Any]) -> Generator[ResponseEvent, None, None]:
        """
        Handle non-streaming response as a single-event stream.

        Args:
            payload: Request payload (with stream=False)

        Yields:
            ResponseEvent subclasses
        """
        response = self._make_request(payload)
        response_data = self._parse_response(response)

        content = response_data.get('content')
        reasoning_content = response_data.get('reasoning_content')
        tool_calls = response_data.get('tool_calls')

        if content:
            content = content.strip()

        # Emit thinking content if present
        if reasoning_content:
            yield ThinkingChunk(content=reasoning_content)

        # If we have tool calls, emit ToolCallsRequested
        if tool_calls:
            yield ToolCallsRequested(
                tool_calls=tool_calls,
                pre_tool_content=content
            )
        # Otherwise emit content (or empty content) followed by Complete
        elif content:
            yield ContentChunk(
                content=content,
                is_final=True
            )
            yield Complete()
        else:
            # Empty response
            yield ContentChunk(
                content="",
                is_final=True
            )
            yield Complete()

    def _handle_streaming_response(self, payload: Dict[str, Any]) -> Generator[ResponseEvent, None, None]:
        """
        Handle streaming response as multiple events.

        Args:
            payload: Request payload (with stream=True)

        Yields:
            ResponseEvent subclasses
        """
        headers = self._build_headers()

        try:
            with requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                stream=True,
            ) as response:
                response.raise_for_status()

                accumulated_content = ""
                tool_calls_accumulator = []

                for line in response.iter_lines():
                    if not line:
                        continue

                    # Decode line
                    line_text = line.decode('utf-8')

                    # Skip non-data lines
                    if not line_text.startswith('data: '):
                        continue

                    # Extract JSON data
                    data_str = line_text[6:]  # Remove 'data: ' prefix

                    # Check for stream end
                    if data_str.strip() == '[DONE]':
                        # Emit final event
                        if tool_calls_accumulator:
                            yield ToolCallsRequested(
                                tool_calls=tool_calls_accumulator,
                                pre_tool_content=accumulated_content if accumulated_content else None
                            )
                        else:
                            yield Complete()
                        return

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract delta from response
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        delta = choice.get('delta', {})

                        # Handle thinking/reasoning chunks
                        if 'reasoning_content' in delta and delta['reasoning_content']:
                            yield ThinkingChunk(content=delta['reasoning_content'])

                        # Handle content chunks
                        if 'content' in delta and delta['content']:
                            chunk = delta['content']
                            accumulated_content += chunk
                            yield ContentChunk(content=chunk)

                        # Handle tool calls (accumulate them)
                        if 'tool_calls' in delta:
                            for tc in delta['tool_calls']:
                                idx = tc.get('index', 0)
                                # Extend list if needed
                                while len(tool_calls_accumulator) <= idx:
                                    tool_calls_accumulator.append({
                                        'id': '',
                                        'type': 'function',
                                        'function': {'name': '', 'arguments': ''}
                                    })
                                # Accumulate tool call data
                                if 'id' in tc:
                                    tool_calls_accumulator[idx]['id'] = tc['id']
                                if 'function' in tc:
                                    if tc['function'].get('name'):
                                        tool_calls_accumulator[idx]['function']['name'] += tc['function']['name']
                                    if tc['function'].get('arguments'):
                                        tool_calls_accumulator[idx]['function']['arguments'] += tc['function']['arguments']

                        # Check for finish reason
                        if choice.get('finish_reason'):
                            if tool_calls_accumulator:
                                yield ToolCallsRequested(
                                    tool_calls=tool_calls_accumulator,
                                    pre_tool_content=accumulated_content if accumulated_content else None,
                                )
                            else:
                                yield Complete()
                            return

                # If we exit the loop without a finish signal, send final
                if tool_calls_accumulator:
                    yield ToolCallsRequested(
                        tool_calls=tool_calls_accumulator,
                        pre_tool_content=accumulated_content if accumulated_content else None,
                    )
                else:
                    yield Complete()

        except requests.exceptions.ConnectionError:
            raise requests.RequestException("Failed to connect to LLM API")
        except requests.exceptions.HTTPError as e:
            raise requests.RequestException(f"HTTP error: {e.response.status_code} - {e.response.text}")

    def query(self, user_message: Optional[str] = None, conversation_history: Optional[List[Dict]] = None, tools: Optional[List[Dict]] = None, streaming: bool = False) -> Generator[ResponseEvent, None, None]:
        """
        Unified query method for all LLM interactions.

        This method ALWAYS returns a generator that yields ResponseEvent objects.
        Both streaming and non-streaming modes use the same event-based interface.

        Handles:
        - Normal queries: user_message="hello"
        - Continuations: user_message=None (uses history as-is)
        - With/without tools: tools parameter for cache alignment
        - System-initiated: user_message="" (empty string, not None)

        Args:
            user_message: User's message (None for continuation, "" for system-initiated, normal string otherwise)
            conversation_history: Optional conversation history
            tools: Optional list of tools in OpenAI format (for cache alignment with warmup)
            streaming: If True, streams content chunks. If False, yields complete response.

        Yields:
            ResponseEvent subclasses (ContentChunk, ToolCallsRequested, Complete)
        """
        print(f"   [Brain] {'Streaming from' if streaming else 'Sending to'} {self.model}...")
        
        payload = self._build_query_payload(user_message, conversation_history, tools)
        payload["stream"] = streaming

        if streaming:
            yield from self._handle_streaming_response(payload)
        else:
            yield from self._handle_complete_response(payload)


    def query_chat_for_task(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        One-shot LLM query for background tasks (memory extraction, ranking, etc.).

        Uses this instance's system prompt as-is — callers that need a neutral
        task prompt should initialize the LLMInterface with one.

        Args:
            prompt: Direct prompt to send to the model
            max_tokens: Maximum tokens to generate (overrides config.assistant.max_task_tokens)

        Returns:
            LLM's response text
        """
        print(f"   [Brain] Sending chat completion for background task...")

        try:
            payload = self._build_query_payload(prompt, conversation_history=None, tools=None)
            payload["stream"] = False
            effective_max_tokens = max_tokens if max_tokens is not None else config.assistant.max_task_tokens
            if effective_max_tokens is not None:
                payload["max_tokens"] = effective_max_tokens

            accumulated_content = ""
            for event in self._handle_complete_response(payload):
                if isinstance(event, ContentChunk):
                    accumulated_content += event.content

            return accumulated_content.strip() if accumulated_content else ''

        except Exception as e:
            print(f"   ❌ Chat completion task error: {e}")
            return ''

    def warmup_cache(self, tools=None, test_message: str = "Hello, this is a warmup query to preload the system prompt.") -> bool:
        """
        Send a warmup query to preload the system prompt and enable caching.

        This sends a minimal request with the full system prompt to populate
        the LLM server's prompt cache. Subsequent requests with the same system
        prompt will be faster since the cached prompt can be reused.

        Args:
            tools: Optional list of tools in OpenAI format (MUST match actual query tools for cache alignment)
            test_message: Simple message to send for warmup

        Returns:
            True if warmup was successful, False otherwise
        """
        if not config.network.enable_cache_warming:
            return False

        try:
            print(f"\n🔥 Warming up LLM cache with system prompt...")

            # Create payload for warmup
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": test_message}
                ],
                "max_tokens": 1,  # Minimal response
                "stream": False
            }

            # Add tools if provided (CRITICAL for cache alignment)
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            # Add thinking mode configuration if available
            if config.assistant.enable_thinking_mode:
                payload["chat_template_kwargs"] = {"enable_thinking": True}

            headers = self._build_headers()

            # Send warmup request (no timeout - will wait as long as server is processing)
            response = requests.post(self.api_url, json=payload, headers=headers)

            if response.status_code == 200:
                print("✅ Cache warmup successful - first responses should be faster!")
                return True
            else:
                print(f"⚠️  Cache warmup failed with status {response.status_code}")
                print(response.text)
                return False

        except Exception as e:
            print(f"⚠️  Cache warmup error: {e}")
            return False

class ConversationManager:
    """
    Manager for maintaining conversation history and context.

    This class handles conversation state, message history, and context management
    for more natural interactions with the LLM.
    """

    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize conversation manager.

        Args:
            llm_interface: LLM interface to use for queries
        """
        self.llm_interface = llm_interface
        self.conversation_history: List[Dict] = []

    def add_message(self, role: str, content: str, images: Optional[list] = None):
        """
        Add a message to the conversation history.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            images: Optional list of image dicts for vision queries.
                    Each dict: {"type": "base64", "data": "...", "media_type": "image/jpeg"}
        """
        if images:
            content_blocks = []
            if content:
                content_blocks.append({"type": "text", "text": content})
            for img in images:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.get('media_type', 'image/jpeg')};base64,{img['data']}"
                    }
                })
            self.conversation_history.append({"role": role, "content": content_blocks})
        else:
            self.conversation_history.append({"role": role, "content": content})

    def query_with_tools(self, user_message: str, tool_manager: Optional[ToolManager] = None, streaming: bool = False, images: Optional[list] = None):
        """
        Unified query method with automatic tool call chaining.

        This is the PRIMARY query implementation for the voice assistant.
        Handles both streaming and non-streaming modes with the same event-based interface.

        Yields ResponseEvent objects as the conversation progresses:
        - ContentChunk: Text content (streaming or complete)
        - ToolExecuting: About to execute a tool
        - ToolResult: Tool execution complete
        - Continuation: Querying LLM again after tools
        - Complete: Response fully done

        Args:
            user_message: User's message (use empty string for system-initiated queries)
            tool_manager: Optional ToolManager instance for tool execution
            streaming: If True, streams content chunks. If False, yields complete responses.
            images: Optional list of image dicts for vision queries

        Yields:
            ResponseEvent subclasses
        """
        try:
            tools = None
            if tool_manager and config.tools.enable_tools:
                tools = tool_manager.get_openai_tools()

            # Add user message once at the start
            if user_message or images:
                self.add_message("user", user_message, images=images)

            # When images are attached, the message is already in history as
            # content blocks — don't pass current_message or _build_query_payload
            # would re-add it as plain text without the images.
            current_message = None if images else user_message

            # Keep looping until we get a final text response (no more tool calls)
            while True:
                accumulated_content = ""

                # Query LLM and stream events
                for event in self.llm_interface.query(
                    current_message,
                    self.conversation_history[:-1] if current_message else self.conversation_history,
                    tools,
                    streaming=streaming
                ):
                    # Handle different event types
                    if isinstance(event, ThinkingChunk):
                        # Pass through thinking chunks to caller
                        yield event

                    elif isinstance(event, ContentChunk):
                        accumulated_content += event.content

                        # Pass through content chunks
                        yield event

                    elif isinstance(event, Complete):
                        # Conversation turn complete - add assistant message if we have content
                        if accumulated_content:
                            self.add_message("assistant", accumulated_content)
                        yield event
                        return  # Exit generator

                    elif isinstance(event, ToolCallsRequested):
                        # Check if we should speak pre-tool dialogue
                        first_tool_name = event.tool_calls[0]['function']['name']
                        should_speak_pre_tool = tool_manager.get_tool_pre_tool_speak(first_tool_name)

                        if event.pre_tool_content:
                            if streaming:
                                # Streaming: pre-tool chunks were already yielded with is_final=False
                                # Flush the buffer by sending an empty chunk with is_final=True
                                # This must happen regardless of should_speak_pre_tool since
                                # the chunks were already streamed to the console output plugin
                                # which needs is_final to release its print lock.
                                yield ContentChunk(content="", is_final=True)
                            elif should_speak_pre_tool:
                                # Non-streaming: pre-tool content wasn't yielded yet, yield it now
                                yield ContentChunk(content=event.pre_tool_content, is_final=True)

                            if not should_speak_pre_tool:
                                print(f"   🎭 Pre-tool dialogue (suppressed): \"{event.pre_tool_content}\"")

                        # Handle tool call chain
                        yield from self._handle_tool_calls(
                            event.tool_calls,
                            event.pre_tool_content,
                            tool_manager,
                            tools,
                            should_speak_pre_tool
                        )

                        # Signal continuation
                        yield Continuation()

                        # Clear current_message - use history for continuation
                        current_message = None
                        break  # Exit inner loop, continue outer while loop

        except Exception as e:
            print(f"   ❌ Query error: {e}")
            import traceback
            traceback.print_exc()
            yield ContentChunk(
                content="I'm having trouble processing your request right now.",
                is_final=True
            )
            yield Complete()
            return  # Exit generator to prevent infinite retry loop

    def _handle_tool_calls(self, tool_calls: List[Dict], pre_tool_content: Optional[str],
                          tool_manager: ToolManager, tools: List[Dict], should_speak_pre_tool: bool):
        """
        Execute a batch of tool calls and yield events.

        Shared logic for tool execution - used by both streaming and non-streaming paths.

        Args:
            tool_calls: List of tool calls to execute
            pre_tool_content: Optional text content before tool calls
            tool_manager: ToolManager instance
            tools: Tool definitions for OpenAI format
            should_speak_pre_tool: Whether pre-tool dialogue should be included in history

        Yields:
            ToolExecuting and ToolResult events
        """
        # Add assistant message with tool calls
        assistant_message = {
            "role": "assistant",
            "tool_calls": tool_calls
        }

        # Include pre-tool dialogue in history if allowed
        if pre_tool_content and should_speak_pre_tool:
            assistant_message["content"] = pre_tool_content

        self.conversation_history.append(assistant_message)

        # Execute each tool
        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            function_args = tool_call['function']['arguments']

            print(f"   🔧 Tool detected: {function_name}")
            print(f"   📋 Arguments: {function_args}")

            # Notify execution start
            yield ToolExecuting(
                tool_name=function_name,
                tool_args=function_args
            )

            # Execute
            tool_result = tool_manager.execute_tool(function_name, function_args)
            print(f"   🔧 Tool result: {tool_result.response}")

            # Add to history
            tool_result_message = {
                "role": "tool",
                "tool_call_id": tool_call.get('id', ''),
                "content": tool_result.response if tool_result.success else f"Error: {tool_result.response}"
            }
            self.conversation_history.append(tool_result_message)

            # Notify execution complete
            yield ToolResult(
                tool_name=function_name,
                result=tool_result.response,
                success=tool_result.success
            )

    def undo_last_turn(self) -> bool:
        """
        Remove the last conversation turn from history.

        Scans backward to find the last real user message (string content,
        not an Anthropic tool_result list) and removes everything from that
        point onward.

        Returns:
            True if a turn was removed, False if history was empty.
        """
        if not self.conversation_history:
            return False

        for i in range(len(self.conversation_history) - 1, -1, -1):
            msg = self.conversation_history[i]
            if msg["role"] == "user" and not self._is_tool_result(msg):
                del self.conversation_history[i:]
                return True

        return False

    @staticmethod
    def _is_tool_result(msg: dict) -> bool:
        """Check if a message is an Anthropic tool_result (not a real user turn)."""
        content = msg.get("content")
        if not isinstance(content, list):
            return False
        return any(block.get("type") == "tool_result" for block in content)

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def get_history_length(self) -> int:
        """Get the number of messages in conversation history."""
        return len(self.conversation_history)

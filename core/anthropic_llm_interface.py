"""
Anthropic Messages API Interface module for communication with Anthropic-compatible LLM APIs.

This module extends LLMInterface to support Anthropic's native Messages API format
instead of OpenAI's Chat Completions format.

Key differences from OpenAI format:
- System prompt is a top-level field, not a message
- max_tokens is required (not optional)
- Auth uses x-api-key header instead of Authorization Bearer
- Content is an array of content blocks, not a simple string
- Tool definitions use input_schema instead of parameters
- Streaming uses Server-Sent Events (SSE) with named event types
- Tool results are nested content blocks in user messages, not separate tool messages

Classes:
    AnthropicLLMInterface: Extends LLMInterface with Anthropic-specific request/response handling
    AnthropicConversationManager: Extends ConversationManager with Anthropic-specific message formatting
"""

import requests
import json
from typing import Dict, List, Optional, Any, Generator

# Import base classes from llm_interface (use absolute import to avoid circular import issues)
from core.llm_interface import (
    LLMInterface, ConversationManager, HistoryMessage,
    ResponseEvent, ContentChunk, ToolCallsRequested, ToolExecuting,
    ToolResult, ThinkingChunk, Complete, Continuation
)
from core.config import config
from core.tool_manager import ToolManager


class AnthropicLLMInterface(LLMInterface):
    """
    LLM Interface implementation for Anthropic Messages API.

    This class extends LLMInterface to use Anthropic's native Messages API format
    instead of OpenAI's Chat Completions format.
    """

    def __init__(self, api_url: Optional[str] = None, model: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        Initialize Anthropic LLM interface.

        Args:
            api_url: API endpoint URL
            model: Model name to use
            system_prompt: System prompt for the conversation
        """
        super().__init__(api_url, model, system_prompt)

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers with Anthropic-specific auth."""
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _convert_openai_to_anthropic_messages(self, conversation_history: List[HistoryMessage]) -> List[Dict]:
        """
        Convert OpenAI-style conversation history to Anthropic Messages format.

        OpenAI uses:
        - role="system" as a message (Anthropic uses system as top-level field)
        - role="tool" for tool results
        - tool_calls as a field on assistant messages

        Anthropic uses:
        - system as a separate top-level field (not a message)
        - role="user" with tool_result content blocks for tool results
        - tool_use content blocks for tool calls

        Args:
            conversation_history: List of HistoryMessage entries to convert

        Returns:
            Anthropic-style message list
        """
        messages = []

        for msg in conversation_history:
            role = msg.role
            content = msg.content if msg.content is not None else ""
            tool_calls = msg.tool_calls

            # Skip system messages - handled separately in Anthropic format
            if role == "system":
                continue

            # Handle assistant messages with tool calls
            elif role == "assistant" and tool_calls:
                # Convert tool calls to Anthropic content blocks
                content_blocks = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    # Parse arguments if they're a string
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc["function"]["name"],
                        "input": args
                    })
                messages.append({"role": "assistant", "content": content_blocks})

            # Handle tool result messages (OpenAI: role="tool", Anthropic: role="user" with tool_result block)
            elif role == "tool":
                # Convert standalone tool messages to user messages with text content
                # This happens during wake-ups when add_message("tool", ...) is called directly
                # In Anthropic format, we can't have standalone tool results, so we convert them to user text messages
                if isinstance(content, str):
                    messages.append({"role": "user", "content": content})
                continue

            # Handle user messages with tool_result blocks (already in Anthropic format)
            elif role == "user" and isinstance(content, list):
                messages.append({"role": "user", "content": content})

            # Handle regular messages
            else:
                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                else:
                    # Content is already in Anthropic block format
                    messages.append({"role": role, "content": content})

        return messages

    def _convert_openai_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """
        Convert OpenAI-style tool definitions to Anthropic format.

        OpenAI uses: {"type": "function", "function": {"name": "...", "parameters": {...}}}
        Anthropic uses: {"name": "...", "description": "...", "input_schema": {...}}

        Args:
            tools: OpenAI-style tool definitions

        Returns:
            Anthropic-style tool definitions
        """
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
        return anthropic_tools

    def _build_query_payload(self, user_message: Optional[str], conversation_history: Optional[List[HistoryMessage]] = None, tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Build the Anthropic Messages API request payload.

        Overrides parent to use Anthropic-specific format:
        - System prompt as top-level field (not a message)
        - max_tokens is required
        - Tools use input_schema instead of parameters
        - Messages use content blocks instead of simple strings

        Args:
            user_message: User's message to send to the LLM (None for continuation)
            conversation_history: List of HistoryMessage entries (will be converted)
            tools: Optional list of tools in OpenAI format (will be converted to Anthropic format)

        Returns:
            Dictionary representing the Anthropic API request payload
        """
        messages = []

        # Convert conversation history from OpenAI to Anthropic format
        if conversation_history:
            messages = self._convert_openai_to_anthropic_messages(conversation_history)

        # Add current user message if provided
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})

        # Build the base payload
        payload = {
            "model": self.model,
            "max_tokens": config.assistant.max_chat_tokens or 1024,
            "messages": messages
        }

        # Add system prompt as top-level field (Anthropic-specific)
        if self.system_prompt:
            payload["system"] = self.system_prompt

        # Convert and add tools
        if tools:
            payload["tools"] = self._convert_openai_tools_to_anthropic(tools)
            payload["tool_choice"] = {"type": "auto"}

        # Add thinking mode configuration if available
        if config.assistant.enable_thinking_mode:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": 8192
            }

        return payload


    def _parse_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Parse the Anthropic API response to extract message content and tool calls.

        Args:
            response: HTTP response from the API

        Returns:
            Dictionary with 'content' (str or None), 'content_blocks' (list), and optionally 'tool_calls' (list)

        Raises:
            ValueError: If response format is unexpected
        """
        try:
            response_data = response.json()

            result = {
                'content': None,
                'content_blocks': [],
                'tool_calls': []
            }

            # Anthropic returns content as an array of blocks
            if 'content' in response_data:
                content_blocks = response_data['content']
                result['content_blocks'] = content_blocks

                # Extract text content
                text_parts = []
                tool_calls = []

                for block in content_blocks:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "thinking":
                        # Thinking block - some APIs return thinking as a content block
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            # Optionally store thinking for debugging
                            pass
                    elif block.get("type") == "tool_use":
                        # Convert Anthropic tool_use to OpenAI format for compatibility
                        input_dict = block.get("input", {})
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(input_dict)
                            }
                        })

                result['content'] = "".join(text_parts) if text_parts else None
                if tool_calls:
                    result['tool_calls'] = tool_calls

                return result

            else:
                raise ValueError("Unexpected response format - missing 'content' field")

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
        tool_calls = response_data.get('tool_calls')

        # Check for thinking content in content_blocks
        content_blocks = response_data.get('content_blocks', [])
        for block in content_blocks:
            if block.get("type") == "thinking":
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    yield ThinkingChunk(content=thinking_text)

        if content:
            content = content.strip()

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

    def _convert_anthropic_tool_calls_to_openai(self, tool_calls_accumulator: Dict[int, Dict]) -> List[Dict]:
        """
        Convert accumulated Anthropic tool calls to OpenAI format.

        Args:
            tool_calls_accumulator: Dictionary mapping indices to Anthropic tool_use blocks

        Returns:
            List of OpenAI-format tool calls
        """
        openai_tool_calls = []
        for idx, tc in sorted(tool_calls_accumulator.items()):
            input_dict = tc["input"]
            if isinstance(input_dict, str):
                try:
                    input_dict = json.loads(input_dict)
                except json.JSONDecodeError:
                    input_dict = {}
            openai_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(input_dict)
                }
            })
        return openai_tool_calls

    def _handle_streaming_response(self, payload: Dict[str, Any]) -> Generator[ResponseEvent, None, None]:
        """
        Handle Anthropic streaming response as multiple events.

        Overrides parent to handle Anthropic's Server-Sent Events (SSE) format:
        - event: lines specify event types
        - data: lines contain JSON payloads
        - Uses content_block_start, content_block_delta, content_block_stop events
        - Supports text_delta, thinking_delta, and input_json_delta

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
                tool_calls_accumulator = {}

                for line in response.iter_lines():
                    if not line:
                        continue

                    line_text = line.decode('utf-8')

                    # Skip non-event lines
                    if not line_text.startswith('event: ') and not line_text.startswith('data: '):
                        continue

                    # Extract event type (not currently used, but available for debugging)
                    if line_text.startswith('event: '):
                        event_type = line_text[7:].strip()
                        continue

                    # Parse data payload
                    if line_text.startswith('data: '):
                        data_str = line_text[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Handle different message types
                        msg_type = data.get("type")

                        if msg_type == "message_start":
                            # Message started (initialization event)
                            pass

                        elif msg_type == "content_block_start":
                            # A new content block is starting
                            block = data.get("content_block", {})
                            if block.get("type") == "tool_use":
                                index = data.get("index", 0)
                                tool_calls_accumulator[index] = {
                                    "type": "tool_use",
                                    "id": block.get("id", ""),
                                    "name": block.get("name", ""),
                                    "input": ""
                                }

                        elif msg_type == "content_block_delta":
                            # Content delta for the current block
                            delta = data.get("delta", {})
                            delta_type = delta.get("type")
                            index = data.get("index", 0)

                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                accumulated_content += text
                                yield ContentChunk(content=text)

                            elif delta_type == "thinking_delta":
                                thinking = delta.get("thinking", "")
                                if thinking:
                                    yield ThinkingChunk(content=thinking)

                            elif delta_type == "input_json_delta":
                                # Tool call arguments being streamed
                                partial_json = delta.get("partial_json", "")
                                if index in tool_calls_accumulator:
                                    tool_calls_accumulator[index]["input"] += partial_json

                        elif msg_type == "content_block_stop":
                            # Content block ended (cleanup event)
                            pass

                        elif msg_type == "message_stop":
                            # Message complete - emit final events
                            if tool_calls_accumulator:
                                openai_tool_calls = self._convert_anthropic_tool_calls_to_openai(tool_calls_accumulator)
                                yield ToolCallsRequested(
                                    tool_calls=openai_tool_calls,
                                    pre_tool_content=accumulated_content if accumulated_content else None
                                )
                            else:
                                yield Complete()
                            return

                # Fallback: if we exit the loop without message_stop, yield what we have
                if tool_calls_accumulator:
                    openai_tool_calls = self._convert_anthropic_tool_calls_to_openai(tool_calls_accumulator)
                    yield ToolCallsRequested(
                        tool_calls=openai_tool_calls,
                        pre_tool_content=accumulated_content if accumulated_content else None
                    )
                else:
                    yield Complete()

        except requests.exceptions.ConnectionError:
            raise requests.RequestException("Failed to connect to LLM API")
        except requests.exceptions.HTTPError as e:
            raise requests.RequestException(f"HTTP error: {e.response.status_code} - {e.response.text}")

    def query(self, user_message: Optional[str] = None, conversation_history: Optional[List[HistoryMessage]] = None, tools: Optional[List[Dict]] = None, streaming: bool = False) -> Generator[ResponseEvent, None, None]:
        """
        Unified query method for all LLM interactions.

        This method ALWAYS returns a generator that yields ResponseEvent objects.
        Both streaming and non-streaming modes use the same event-based interface.

        Args:
            user_message: User's message (None for continuation, "" for system-initiated, normal string otherwise)
            conversation_history: Optional conversation history
            tools: Optional list of tools in OpenAI format (will be converted to Anthropic format)
            streaming: If True, streams content chunks. If False, yields complete response.

        Yields:
            ResponseEvent subclasses (ContentChunk, ToolCallsRequested, Complete)
        """
        print(f"   [Brain Anthropic] {'Streaming from' if streaming else 'Sending to'} {self.model}...")

        payload = self._build_query_payload(user_message, conversation_history, tools)

        # Add stream parameter to payload
        payload["stream"] = streaming

        if streaming:
            yield from self._handle_streaming_response(payload)
        else:
            yield from self._handle_complete_response(payload)


    def query_chat_for_task(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        One-shot LLM query for background tasks (memory extraction, ranking, etc.).

        Overrides parent to use Anthropic-specific format (system prompt as top-level field).
        Disables thinking mode for background tasks to save tokens.

        Args:
            prompt: Direct prompt to send to the model
            max_tokens: Maximum tokens to generate (overrides config.assistant.max_task_tokens)

        Returns:
            LLM's response text
        """
        print(f"   [Brain Anthropic] Sending chat completion for background task...")

        try:
            # Use _build_query_payload which handles Anthropic format correctly
            payload = self._build_query_payload(prompt, conversation_history=None, tools=None)
            payload["stream"] = False

            # Override max_tokens for this task
            effective_max_tokens = max_tokens if max_tokens is not None else config.assistant.max_task_tokens
            if effective_max_tokens is not None:
                payload["max_tokens"] = effective_max_tokens

            # Disable thinking mode for background tasks (saves tokens and improves speed)
            if "thinking" in payload:
                del payload["thinking"]

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

        Overrides parent to use Anthropic-specific payload building.

        Args:
            tools: Optional list of tools in OpenAI format (will be converted to Anthropic format)
            test_message: Simple message to send for warmup

        Returns:
            True if warmup was successful, False otherwise
        """
        if not config.network.enable_cache_warming:
            return False

        try:
            print(f"\n🔥 Warming up Anthropic LLM cache with system prompt...")

            # Use _build_query_payload to ensure consistent formatting
            payload = self._build_query_payload(test_message, conversation_history=None, tools=tools)
            payload["max_tokens"] = 1  # Minimal response for warmup

            headers = self._build_headers()
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


class AnthropicConversationManager(ConversationManager):
    """
    Conversation Manager for Anthropic Messages API.

    Extends ConversationManager to handle Anthropic-specific message formatting,
    including proper tool_use and tool_result content blocks.
    """

    def __init__(self, llm_interface: AnthropicLLMInterface):
        """
        Initialize conversation manager.

        Args:
            llm_interface: Anthropic LLM interface to use for queries
        """
        super().__init__(llm_interface)

    def add_message(self, role: str, content: str, images: Optional[list] = None):
        """
        Add a message in Anthropic content block format.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content text
            images: Optional list of image dicts for vision queries.
                    Each dict: {"type": "base64", "data": "...", "media_type": "image/jpeg"}
        """
        if images:
            content_blocks = []
            if content:
                content_blocks.append({"type": "text", "text": content})
            for img in images:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.get("media_type", "image/jpeg"),
                        "data": img["data"]
                    }
                })
            self.conversation_history.append(HistoryMessage(role=role, content=content_blocks))
        else:
            self.conversation_history.append(HistoryMessage(role=role, content=content))

    def query_with_tools(self, user_message: str, tool_manager: Optional[ToolManager] = None, streaming: bool = False, images: Optional[list] = None):
        """
        Query method with automatic tool call chaining.

        Overrides parent to use Anthropic-specific message handling:
        - Uses history-only approach (current_message=None after first query)
        - Stores assistant messages with content blocks instead of strings

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

            current_message = None  # Anthropic uses history only (no separate current message)

            # Keep looping until we get a final text response (no more tool calls)
            while True:
                accumulated_content = ""

                # Query LLM and stream events
                for event in self.llm_interface.query(
                    current_message,
                    self.conversation_history,
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
                        # Store in Anthropic format (content blocks)
                        if accumulated_content:
                            self.conversation_history.append(HistoryMessage(
                                role="assistant",
                                content=[{"type": "text", "text": accumulated_content}],
                            ))
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

                        # Handle tool call chain (uses Anthropic-specific implementation)
                        yield from self._handle_tool_calls(
                            event.tool_calls,
                            event.pre_tool_content,
                            tool_manager,
                            tools,
                            should_speak_pre_tool
                        )

                        # Signal continuation
                        yield Continuation()

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
        Execute tool calls and format history for Anthropic API.

        Anthropic requires:
        - Assistant messages with tool calls use content blocks
        - Tool results are sent as user messages with tool_result blocks

        Args:
            tool_calls: List of tool calls to execute (in OpenAI format)
            pre_tool_content: Optional text content before tool calls
            tool_manager: ToolManager instance
            tools: Tool definitions for OpenAI format
            should_speak_pre_tool: Whether pre-tool dialogue should be included in history

        Yields:
            ToolExecuting and ToolResult events
        """
        # Build assistant message content blocks in Anthropic format
        content_blocks = []
        tool_result_blocks = []

        # Always include pre-tool text in history to faithfully represent the
        # model's output. The Anthropic API expects the assistant message to
        # match what the model actually generated (text + tool_use). Omitting
        # the text block can cause the continuation request to hang or fail.
        # The should_speak_pre_tool flag only controls whether text is spoken
        # aloud, not whether it's recorded in conversation history.
        if pre_tool_content:
            content_blocks.append({"type": "text", "text": pre_tool_content})

        # Execute all tools and build blocks
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

            # Execute tool
            tool_result = tool_manager.execute_tool(function_name, function_args)
            print(f"   🔧 Tool result: {tool_result.response}")

            # Parse arguments for tool_use block
            args_dict = function_args
            if isinstance(function_args, str):
                try:
                    args_dict = json.loads(function_args)
                except json.JSONDecodeError:
                    args_dict = {}

            # Add tool_use block to assistant message
            content_blocks.append({
                "type": "tool_use",
                "id": tool_call.get('id', ''),
                "name": function_name,
                "input": args_dict
            })

            # Add tool_result block for user message
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_call.get('id', ''),
                "content": tool_result.response if tool_result.success else f"Error: {tool_result.response}"
            })

            # Notify execution complete
            yield ToolResult(
                tool_name=function_name,
                result=tool_result.response,
                success=tool_result.success
            )

        # Add assistant message with tool_use blocks to history
        self.conversation_history.append(HistoryMessage(
            role="assistant",
            content=content_blocks,
        ))

        # Add user message with tool_result blocks to history
        self.conversation_history.append(HistoryMessage(
            role="user",
            content=tool_result_blocks,
        ))


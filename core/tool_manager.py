"""
Tool Manager module for MCP tool integration.
Provides tool registry, execution, and configuration management.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from core.config import ToolsConfig


@dataclass
class ToolCallResult:
    """Result of a tool execution."""
    success: bool
    response: str
    tool_name: Optional[str] = None
    error: Optional[str] = None


class Tool(ABC):
    """
    Base class for all tools.

    Each tool must define:
    - name: Unique identifier
    - description: What the tool does and when to use it
    - parameters: JSON Schema for parameters
    - config_schema: Default configuration values
    - execute: The actual tool implementation
    """

    # Default configuration schema - override in subclasses as class attribute
    config_schema: Dict[str, Any] = {"enabled": True}

    # Whether LLM should speak pre-tool dialogue - override in subclasses as class attribute
    pre_tool_speak: bool = True

    # NOTE: pre_tool_speak being True means the LLM will naturally respond before
    # the tool executes (e.g., "I'll check the time for you").
    # Tools that should not have pre-tool dialogue (like schedule_self_wake)
    # should set this to False, so the response only comes after the tool
    # completes and can include the results.

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name/identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM - what it does and when to use it."""
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        JSON Schema for tool parameters.
        Override to define specific parameters.
        """
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }

    @property
    def help_text(self) -> str:
        """
        Generate help text for this tool.
        Override to provide custom help text format.
        """
        lines = [f"- {self.name}: {self.description}"]

        # Add parameter info if available
        if self.parameters.get('properties'):
            lines.append(f"  Parameters:")
            for param_name, param_info in self.parameters['properties'].items():
                param_type = param_info.get('type', 'any')
                param_desc = param_info.get('description', '')
                lines.append(f'  - "{param_name}": {param_type} ({param_desc})')
        return "\n".join(lines)

    def configure(self, config: Dict[str, Any]):
        """
        Configure the tool with tool-specific settings.

        Args:
            config: Tool configuration dict from tool_settings[tool_name]
        """
        # Store config for use in execute()
        self.config = config

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Execute the tool and return result.

        Args:
            **kwargs: Tool parameters

        Returns:
            String result of tool execution
        """
        pass

    def is_enabled(self) -> bool:
        """Check if this tool is enabled based on its config."""
        return getattr(self, 'config', {}).get('enabled', True)


class ToolManager:
    """
    Manager for tool registry and execution.

    Handles tool registration, configuration loading,
    and execution with error handling.
    """

    def __init__(self, config: Optional[ToolsConfig] = None):
        """
        Initialize tool manager.

        Args:
            config: ToolsConfig object with tool_settings
        """
        self.tools: Dict[str, Tool] = {}
        self.config = config

    def register(self, tool: Tool):
        """
        Register a tool.

        Args:
            tool: Tool instance to register
        """
        self.tools[tool.name] = tool
        print(f"✅ Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self.tools.values())

    def get_enabled_tools(self) -> List[Tool]:
        """Get all enabled tools."""
        return [tool for tool in self.tools.values() if tool.is_enabled()]

    def execute_tool(self, tool_name: str, args_json: str) -> ToolCallResult:
        """
        Execute a tool with JSON arguments.

        Args:
            tool_name: Name of tool to execute
            args_json: JSON string of arguments

        Returns:
            ToolCallResult with execution status and response
        """
        # Get tool
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolCallResult(
                success=False,
                response=f"Error: Tool '{tool_name}' not found",
                tool_name=tool_name,
                error=f"Tool not found: {tool_name}"
            )

        # Check if enabled
        if not tool.is_enabled():
            return ToolCallResult(
                success=False,
                response=f"Error: Tool '{tool_name}' is disabled",
                tool_name=tool_name,
                error=f"Tool disabled: {tool_name}"
            )

        # Parse arguments
        try:
            args = json.loads(args_json) if args_json.strip() else {}
        except json.JSONDecodeError as e:
            return ToolCallResult(
                success=False,
                response=f"Error: Invalid JSON arguments for tool '{tool_name}': {e}",
                tool_name=tool_name,
                error=str(e)
            )

        # Execute tool
        try:
            result = tool.execute(**args)

            # If tool returns a dict, convert to JSON string for the LLM
            if isinstance(result, dict):
                # Remove _internal flags before sending to LLM (if any)
                if '_internal' in result and isinstance(result['_internal'], dict):
                    del result['_internal']

                result_json = json.dumps(result, indent=2)
                return ToolCallResult(
                    success=True,
                    response=result_json,
                    tool_name=tool_name
                )
            else:
                # String return (legacy or error message)
                return ToolCallResult(
                    success=True,
                    response=result,
                    tool_name=tool_name
                )

        except Exception as e:
            return ToolCallResult(
                success=False,
                response=f"Error executing tool '{tool_name}': {e}",
                tool_name=tool_name,
                error=str(e)
            )


    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Convert enabled tools to OpenAI function calling format.

        Returns:
            List of tools in OpenAI format: [{"type": "function", "function": {...}}]
        """
        enabled_tools = self.get_enabled_tools()

        openai_tools = []
        for tool in enabled_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })

        return openai_tools

    def get_tool_pre_tool_speak(self, tool_name: str) -> bool:
        """
        Check if a tool should allow pre-tool dialogue.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if pre-tool dialogue should be spoken, False otherwise
        """
        tool = self.tools.get(tool_name)
        if tool:
            return tool.__class__.pre_tool_speak
        return True  # Default to True if tool not found

    def load_tool_configs(self, tool_settings: Dict[str, Dict[str, Any]]):
        """
        Load configuration for all registered tools.

        Args:
            tool_settings: Dict from config.tools.tool_settings
        """
        for tool_name, tool in self.tools.items():
            # Get config for this tool (if it exists)
            tool_config = tool_settings.get(tool_name, {})

            # Merge with default config schema (access via class to handle class attributes)
            default_config = tool.__class__.config_schema.copy()
            default_config.update(tool_config)

            # Configure the tool
            tool.configure(default_config)

            enabled_status = "enabled" if tool.is_enabled() else "disabled"
            print(f"   🔧 {tool_name}: {enabled_status}")

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self.tools

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self.tools)

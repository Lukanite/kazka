"""
Anthropic Tool Manager module for tool conversion to Anthropic format.

This module provides utilities to convert tool definitions from the internal
format to Anthropic's Messages API tool format.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from typing import Dict, Any, List, Optional
from tool_manager import ToolManager


class AnthropicToolManager(ToolManager):
    """
    Tool Manager subclass that adds Anthropic-specific tool formatting.

    This extends the base ToolManager with a method to convert tools to
    Anthropic Messages API format.
    """

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """
        Convert enabled tools to Anthropic Messages API tool format.

        Anthropic tool format:
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        Returns:
            List of tools in Anthropic format
        """
        enabled_tools = self.get_enabled_tools()

        anthropic_tools = []
        for tool in enabled_tools:
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters
            })

        return anthropic_tools

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Convert enabled tools to OpenAI function calling format.

        This is included for compatibility with the parent class.

        Returns:
            List of tools in OpenAI format: [{"type": "function", "function": {...}}]
        """
        return super().get_openai_tools()


def get_anthropic_tools_from_manager(tool_manager: ToolManager) -> List[Dict[str, Any]]:
    """
    Utility function to convert tools from a ToolManager to Anthropic format.

    This allows using the existing ToolManager instance without subclassing.

    Args:
        tool_manager: Existing ToolManager instance

    Returns:
        List of tools in Anthropic format
    """
    enabled_tools = tool_manager.get_enabled_tools()

    anthropic_tools = []
    for tool in enabled_tools:
        anthropic_tools.append({
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters
        })

    return anthropic_tools

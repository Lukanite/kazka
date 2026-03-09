"""
Time awareness tools for the voice assistant.
Provides current date and time information with timezone support.
"""

from datetime import datetime
from typing import Dict, Any
from core.tool_manager import Tool


class GetDateTimeTool(Tool):
    """Tool to get the current date and time in the user's local timezone."""

    config_schema = {
        "enabled": True
    }

    @property
    def name(self) -> str:
        return "get_datetime"

    @property
    def description(self) -> str:
        return "Get the current date and time. Use this when the user asks what time it is, what the date is, or wants to know the current time in their timezone."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Get the current date and time."""
        current_time = datetime.now().astimezone()

        return {
            "current_datetime": current_time.isoformat(),
            "timezone": str(current_time.tzinfo),
            "unix_timestamp": int(current_time.timestamp())
        }

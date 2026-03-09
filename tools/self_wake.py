"""
Self-wake tools for the voice assistant.
Allows the assistant to schedule its own future wake-ups to re-engage in conversation.
"""

from typing import Dict, Any, Optional
from core.tool_manager import Tool
from datetime import datetime

try:
    from plugins.services.scheduler import SchedulerPlugin
    import dateparser
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    SchedulerPlugin = None
    dateparser = None

class ScheduleSelfWakeTool(Tool):
    """Tool to schedule self-wake events for the assistant."""

    def __init__(self, scheduler: Optional['SchedulerPlugin'] = None):
        self.scheduler = scheduler

    @property
    def name(self) -> str:
        return "schedule_self_wake"

    @property
    def description(self) -> str:
        return "Schedule yourself to wake up and re-engage in conversation at a future time."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "time_string": {
                    "type": "string",
                    "description": "Natural language time description (e.g., '3am', 'in 20 minutes', 'tomorrow noon'). Preferred over delay_seconds.",
                    "examples": ["3am", "in 20 minutes", "tomorrow at 5pm", "noon"]
                },
                "delay_seconds": {
                    "type": "number",
                    "description": "Explicit delay in seconds as fallback (e.g., 3600 for 1 hour)",
                    "examples": [60, 300, 3600, 7200]
                }
            },
            "additionalProperties": False
        }

    def execute(self, time_string: Optional[str] = None, delay_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Schedule a self-wake event."""
        if not self.scheduler:
            return {"success": False, "error": "Scheduler not available."}

        final_delay = 0.0

        try:
            if time_string:
                now = datetime.now()
                parsed_date = dateparser.parse(time_string, settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': now})

                if not parsed_date:
                    return {"success": False, "error": f"Could not understand the time '{time_string}'. Please try a simpler format."}

                delta = parsed_date - now
                final_delay = delta.total_seconds()

                if final_delay <= 0:
                    return {"success": False, "error": f"The time '{time_string}' appears to be in the past."}

            elif delay_seconds is not None:
                final_delay = float(delay_seconds)

            else:
                return {"success": False, "error": "You must provide either 'time_string' or 'delay_seconds'."}

            # Format delay for human-readable output
            if final_delay < 60:
                readable_delay = f"{int(final_delay)} seconds"
            elif final_delay < 3600:
                remaining_seconds = int(final_delay % 60)
                if remaining_seconds > 0:
                    readable_delay = f"{int(final_delay / 60)} minutes and {remaining_seconds} seconds"
                else:
                    readable_delay = f"{int(final_delay / 60)} minutes"
            else:
                hours = int(final_delay / 3600)
                remaining_minutes = int((final_delay % 3600) / 60)
                if remaining_minutes > 0:
                    readable_delay = f"{hours} hours and {remaining_minutes} minutes"
                else:
                    readable_delay = f"{hours} hours"

            timer = self.scheduler.schedule_timer(
                delay_description=time_string if time_string else readable_delay,
                delay_seconds=final_delay,
            )

            return {
                "success": True,
                "wake_id": timer.id,
                "delay_seconds": final_delay,
                "scheduled_for": f"{readable_delay} from now"
            }

        except Exception as e:
            return {"success": False, "error": f"Error scheduling wake: {str(e)}"}


class CancelSelfWakeTool(Tool):
    """Tool to cancel scheduled self-wake events."""

    config_schema = {
        "enabled": True,
        "require_confirmation": False
    }

    def __init__(self, scheduler: Optional['SchedulerPlugin'] = None):
        self.scheduler = scheduler

    @property
    def name(self) -> str:
        return "cancel_self_wake"

    @property
    def description(self) -> str:
        return "Cancel a scheduled self-wake using its wake ID. Use this when you no longer need to wake up at a specific time."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "wake_id": {
                    "type": "string",
                    "description": "The wake ID to cancel (e.g., 'abc123')",
                    "examples": ["abc123", "wake_456"]
                }
            },
            "required": ["wake_id"],
            "additionalProperties": False
        }

    def execute(self, wake_id: str = None) -> Dict[str, Any]:
        """Cancel a scheduled wake-up."""
        if not self.scheduler:
            return {"success": False, "error": "Scheduler not available."}

        if not wake_id:
            return {"success": False, "error": "wake_id is required to cancel a wake-up."}

        try:
            success = self.scheduler.cancel_timer(wake_id)

            if success:
                return {"success": True, "wake_id": wake_id, "cancelled": True}
            else:
                return {"success": False, "wake_id": wake_id, "cancelled": False, "error": "Wake-up not found or already expired"}

        except Exception as e:
            return {"success": False, "error": f"Error cancelling wake-up: {e}"}


class ListSelfWakesTool(Tool):
    """Tool to list scheduled self-wake events."""

    config_schema = {
        "enabled": True,
        "show_expired": False,
        "sort_order": "soonest"
    }

    def __init__(self, scheduler: Optional['SchedulerPlugin'] = None):
        self.scheduler = scheduler

    @property
    def name(self) -> str:
        return "list_self_wakes"

    @property
    def description(self) -> str:
        return "List all scheduled self-wake events. Use this to see what future wake-ups you have scheduled."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }

    def execute(self) -> Dict[str, Any]:
        """List scheduled wake-ups."""
        if not self.scheduler:
            return {"success": False, "error": "Scheduler not available."}

        try:
            timers = self.scheduler.list_timers()

            if not timers:
                return {"success": True, "count": 0, "wakes": []}

            wake_list = []
            for timer in timers:
                wake_list.append({
                    "wake_id": timer.id,
                    "delay_description": timer.delay_description,
                    "trigger_time": timer.trigger_time,
                    "is_active": timer.is_active,
                    "status": "active" if timer.is_active else "expired"
                })

            return {
                "success": True,
                "count": len(timers),
                "wakes": wake_list
            }

        except Exception as e:
            return {"success": False, "error": f"Error listing wake-ups: {e}"}

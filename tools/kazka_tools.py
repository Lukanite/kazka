"""
Tool manifest for the Kazka core repo.

Every LLM tool this repo provides is registered here. Factories MUST
defer heavy imports — see existing entries for the pattern.

Resources expected to be seeded by the engine before discovery:
- "scheduler"          : SchedulerPlugin instance (for self-wake tools)
- "conversation_index" : ConversationIndexPlugin instance (search tools)

Tools whose required resources aren't seeded skip silently (see loader).
"""

from core.tool_registry import ToolBuild, register_tool
from core.config import config


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def _make_get_datetime(engine, cfg, resources):
    from tools.time_awareness import GetDateTimeTool
    return ToolBuild(GetDateTimeTool())


# ---------------------------------------------------------------------------
# Self-wake (requires scheduler service plugin)
# ---------------------------------------------------------------------------

def _make_schedule_self_wake(engine, cfg, resources):
    from tools.self_wake import ScheduleSelfWakeTool
    return ToolBuild(ScheduleSelfWakeTool(resources["scheduler"]))


def _make_cancel_self_wake(engine, cfg, resources):
    from tools.self_wake import CancelSelfWakeTool
    return ToolBuild(CancelSelfWakeTool(resources["scheduler"]))


def _make_list_self_wakes(engine, cfg, resources):
    from tools.self_wake import ListSelfWakesTool
    return ToolBuild(ListSelfWakesTool(resources["scheduler"]))


# ---------------------------------------------------------------------------
# Matter (light control)
# ---------------------------------------------------------------------------

# Both Matter tools share a single [tools.tool_settings.matter] section
# rather than duplicating host/port/aliases/groups under each tool name.
def _matter_config() -> dict:
    mc = config.tools.tool_settings.get("matter")
    if not mc:
        raise ValueError(
            "Matter tool requires [tools.tool_settings.matter] in "
            "assistant_settings.toml (host, device_aliases, ...)"
        )
    if "host" not in mc:
        raise ValueError("[tools.tool_settings.matter] is missing 'host'")
    return mc


def _make_matter_light_control(engine, cfg, resources):
    from tools.matter import MatterLightControlTool
    mc = _matter_config()
    if "device_aliases" not in mc:
        raise ValueError(
            "[tools.tool_settings.matter] is missing 'device_aliases'"
        )
    return ToolBuild(MatterLightControlTool(
        matter_host=mc["host"],
        matter_port=mc.get("port", 5580),
        device_aliases=mc["device_aliases"],
        groups=mc.get("groups", {}),
    ))


def _make_matter_list_devices(engine, cfg, resources):
    from tools.matter import MatterListDevicesTool
    mc = _matter_config()
    return ToolBuild(MatterListDevicesTool(
        matter_host=mc["host"],
        matter_port=mc.get("port", 5580),
    ))


# ---------------------------------------------------------------------------
# Conversation search (requires conversation_index service plugin)
# ---------------------------------------------------------------------------

def _make_search_conversation_logs(engine, cfg, resources):
    from tools.conversation_search import SearchConversationLogsTool
    sc = config.conversation_search
    return ToolBuild(SearchConversationLogsTool(
        resources["conversation_index"],
        context_window=sc.context_window,
        top_k=sc.top_k,
        min_score=sc.min_score,
    ))


def _make_read_conversation_context(engine, cfg, resources):
    from tools.conversation_search import ReadConversationContextTool
    return ToolBuild(ReadConversationContextTool(resources["conversation_index"]))


def _make_list_conversations_in_time(engine, cfg, resources):
    from tools.conversation_search import ListConversationsInTimeTool
    log_dir = config.memory.conversation_log_dir or "log"
    return ToolBuild(ListConversationsInTimeTool(log_dir=log_dir))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

register_tool("get_datetime", _make_get_datetime,
              description="Current local date and time")

register_tool("schedule_self_wake", _make_schedule_self_wake,
              requires_resource=["scheduler"],
              description="Schedule a future wake-up to re-engage in conversation")
register_tool("cancel_self_wake", _make_cancel_self_wake,
              requires_resource=["scheduler"],
              description="Cancel a previously scheduled self-wake")
register_tool("list_self_wakes", _make_list_self_wakes,
              requires_resource=["scheduler"],
              description="List currently scheduled self-wakes")

register_tool("matter_light_control", _make_matter_light_control,
              description="Turn Matter lights on/off, dim, change color")
register_tool("list_matter_devices", _make_matter_list_devices,
              description="Enumerate available Matter devices")

register_tool("search_conversation_logs", _make_search_conversation_logs,
              requires_resource=["conversation_index"],
              description="Semantic search over past conversations")
register_tool("read_conversation_context", _make_read_conversation_context,
              requires_resource=["conversation_index"],
              description="Read a specific conversation context window")
register_tool("list_conversations_in_time",
              _make_list_conversations_in_time,
              description="List conversations within a time range")

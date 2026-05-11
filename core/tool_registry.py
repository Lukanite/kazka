"""
Tool manifest collection.

Mirrors plugin_registry.py for LLM tools. Each repo that contributes
tools exposes a manifest module (by convention `kazka_tools`) whose
top level calls `register_tool` once per tool. Importing the manifest
collects specs; `take_specs()` drains.

The factory in each spec is expected to be a thin closure that defers
heavy imports until called — see `tools/kazka_tools.py` for the pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class ToolBuild:
    """Return value of a tool factory.

    `tool` is the constructed Tool instance. Tools don't currently
    provide resources to other tools, but the field is kept symmetric
    with PluginBuild in case that changes.
    """
    tool: Any
    resources: dict = field(default_factory=dict)


FactoryFn = Callable[[Any, Any, dict], ToolBuild]
# Signature: factory(engine, cfg, resources) -> ToolBuild
# - engine: AssistantEngine (rarely needed; most tools don't touch it)
# - cfg: per-tool settings dict from config.tools.tool_settings[name]
#        (empty dict if no settings configured for this tool)
# - resources: dict containing only the resources this tool declared
#              `requires_resource` for


@dataclass
class ToolSpec:
    """Declarative description of a tool."""
    name: str
    factory: FactoryFn

    # Resources this tool needs (e.g. "scheduler", "conversation_index").
    # Pre-seeded into the ToolLoader before discovery (typically the
    # engine seeds plugin instances it has already constructed).
    requires_resource: List[str] = field(default_factory=list)

    # Free-form description for --help / introspection.
    description: str = ""


_specs: List[ToolSpec] = []


def register_tool(
    name: str,
    factory: FactoryFn,
    *,
    requires_resource: Optional[List[str]] = None,
    description: str = "",
) -> None:
    """Register a tool spec. Called from manifest modules at import time."""
    _specs.append(ToolSpec(
        name=name,
        factory=factory,
        requires_resource=list(requires_resource or []),
        description=description,
    ))


def take_specs() -> List[ToolSpec]:
    """Return all collected specs and clear the buffer."""
    out = list(_specs)
    _specs.clear()
    return out

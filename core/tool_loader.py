"""
Tool loader.

Reads manifests, validates each factory's declared `requires_resource`
is available, builds tools, then registers them with the ToolManager
and applies per-tool settings.

Discovery:
- Internal tools: imported via `import tools.kazka_tools` — that
  module's top level calls `register_tool(...)` for each one.
- External tools: discovered via the `kazka.tools` entry-point group.

The loader is single-use: instantiate, seed resources, call `discover()`
then `load_all()`.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Set

from core.tool_registry import ToolBuild, ToolSpec, take_specs


class ToolLoadError(Exception):
    pass


class ToolLoader:
    def __init__(
        self,
        engine: Any,
        tool_manager: Any,
        config: Any,
        *,
        disabled: Optional[Set[str]] = None,
    ):
        """
        Args:
            engine: AssistantEngine instance (passed through to factories).
            tool_manager: ToolManager that built tools register into.
            config: Global config object (factories receive their per-tool
                    slice from config.tools.tool_settings).
            disabled: Names of tools to skip entirely (typically
                      config.tools.disabled_tools).
        """
        self.engine = engine
        self.tool_manager = tool_manager
        self.config = config
        self.disabled = set(disabled or ())

        self.specs: List[ToolSpec] = []
        self.resources: Dict[str, Any] = {}
        self.tools: Dict[str, Any] = {}

    def discover(self) -> "ToolLoader":
        """Import internal + external manifests; collect specs."""
        importlib.import_module("tools.kazka_tools")

        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="kazka.tools")
            for ep in eps:
                try:
                    ep.load()
                except Exception as e:
                    print(f"⚠️  External tool manifest '{ep.name}' failed to load: {e}")
        except Exception as e:
            print(f"   (entry-point discovery skipped: {e})")

        self.specs = take_specs()
        return self

    def add_resource(self, name: str, value: Any) -> "ToolLoader":
        """Seed a resource the engine has already constructed.

        Typical usage: after plugins have loaded, the engine seeds
        plugin instances by name (e.g. 'scheduler', 'conversation_index')
        so tool factories that declared `requires_resource` for them
        receive them.
        """
        self.resources[name] = value
        return self

    def load_all(self) -> "ToolLoader":
        """Filter, build, register, and configure every spec.

        Tools whose declared resources aren't available are skipped with
        a warning rather than aborting the whole load — a disabled
        plugin should silently disable the tools that depend on it, not
        take down every other tool too.
        """
        active = [s for s in self.specs if s.name not in self.disabled]
        buildable = self._filter_by_resources(active)

        for spec in buildable:
            try:
                tool = self._build(spec)
            except Exception as e:
                print(f"   ⚠️  Tool '{spec.name}' failed to load: {e}")
                continue

            self.tools[spec.name] = tool
            self.tool_manager.register(tool)

        # Apply per-tool settings via the existing ToolManager hook.
        # This populates each tool's `.config` for is_enabled() checks.
        self.tool_manager.load_tool_configs(self.config.tools.tool_settings)
        return self

    # ------------------------------------------------------------------

    def _filter_by_resources(self, specs: List[ToolSpec]) -> List[ToolSpec]:
        """Drop specs whose required resources weren't seeded, with a warning."""
        buildable: List[ToolSpec] = []
        for s in specs:
            missing = [r for r in s.requires_resource if r not in self.resources]
            if missing:
                print(
                    f"   ⚠️  Tool '{s.name}' skipped: required resource(s) "
                    f"{missing} not available "
                    f"(provider plugin disabled or not registered)"
                )
                continue
            buildable.append(s)
        return buildable

    def _build(self, spec: ToolSpec) -> Any:
        cfg = self.config.tools.tool_settings.get(spec.name, {})
        offered = {name: self.resources[name] for name in spec.requires_resource}

        result = spec.factory(self.engine, cfg, offered)
        if not isinstance(result, ToolBuild):
            raise ToolLoadError(
                f"Factory for '{spec.name}' must return ToolBuild, "
                f"got {type(result).__name__}"
            )

        # Tools don't provide resources today, but symmetric with PluginBuild.
        if result.resources:
            self.resources.update(result.resources)
        return result.tool

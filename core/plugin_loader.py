"""
Plugin loader.

Reads manifests, resolves resource dependencies, builds plugins in the
correct order, validates each factory's declared `provides_resource`
against what it actually returned, and registers everything with the
engine.

Discovery:
- Internal plugins: imported via `import plugins.kazka_plugins` — that
  module's top level calls `register_plugin(...)` for each one.
- External plugins: discovered via the `kazka.plugins` entry-point
  group. Each entry point is a module whose import side-effects
  register plugins, exactly like the internal manifest.

The loader is single-use: instantiate, configure, call `discover()`
then `load_all()`. Tests can construct a fresh loader per run.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Set

from core.plugin_registry import PluginBuild, PluginSpec, take_specs


class PluginLoadError(Exception):
    pass


class PluginLoader:
    def __init__(
        self,
        engine: Any,
        config: Any,
        *,
        disabled: Optional[Set[str]] = None,
    ):
        """
        Args:
            engine: AssistantEngine instance to register plugins with.
            config: Global config object (passed to factories as `cfg`).
            disabled: Names of plugins to skip even if `enabled_default`.
                     Plugins with `always_on=True` ignore this set.
        """
        self.engine = engine
        self.config = config
        self.disabled = set(disabled or ())

        self.specs: List[PluginSpec] = []
        self.resources: Dict[str, Any] = {}
        self.plugins: Dict[str, Any] = {}  # name -> built plugin

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> "PluginLoader":
        """Import internal + external manifests; collect specs."""
        # Internal manifest. Import side-effect: register_plugin() calls.
        importlib.import_module("plugins.kazka_plugins")

        # External manifests via entry points.
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="kazka.plugins")
            for ep in eps:
                try:
                    ep.load()
                except Exception as e:
                    print(f"⚠️  External plugin manifest '{ep.name}' failed to load: {e}")
        except Exception as e:
            # Entry points unavailable (very old Python, etc.) — non-fatal.
            print(f"   (entry-point discovery skipped: {e})")

        self.specs = take_specs()
        return self

    def add_resource(self, name: str, value: Any) -> "PluginLoader":
        """
        Pre-seed a resource produced outside the plugin system
        (e.g. an engine-owned object). Must be called before load_all().
        """
        self.resources[name] = value
        return self

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> "PluginLoader":
        """Filter, order, build, and register every spec."""
        active = [s for s in self.specs if self._is_active(s)]
        self._check_resource_providers(active)
        ordered = self._topo_sort(active)

        for spec in ordered:
            try:
                plugin = self._build(spec)
            except Exception as e:
                # A failing optional plugin shouldn't take down the system.
                # Always-on plugins, on the other hand, are fatal.
                if spec.always_on:
                    raise PluginLoadError(
                        f"Required plugin '{spec.name}' failed to load: {e}"
                    ) from e
                print(f"   ⚠️  Plugin '{spec.name}' failed to load: {e}")
                continue

            self.plugins[spec.name] = plugin
            self._register_with_engine(spec, plugin)

        return self

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_active(self, spec: PluginSpec) -> bool:
        if spec.always_on:
            return True
        if spec.name in self.disabled:
            return False
        return spec.enabled_default

    def _check_resource_providers(self, specs: List[PluginSpec]) -> None:
        """
        Verify every required resource is provided by some active plugin
        or pre-seeded via add_resource().
        """
        provided = set(self.resources)
        for s in specs:
            provided.update(s.provides_resource)

        for s in specs:
            missing = [r for r in s.requires_resource if r not in provided]
            if missing:
                raise PluginLoadError(
                    f"Plugin '{s.name}' requires resource(s) {missing} "
                    f"but no active plugin provides them."
                )

        # Detect duplicate providers (last spec to register wins is too
        # silent — better to fail loudly).
        seen: Dict[str, str] = {}
        for s in specs:
            for r in s.provides_resource:
                if r in seen:
                    raise PluginLoadError(
                        f"Resource '{r}' is provided by both "
                        f"'{seen[r]}' and '{s.name}'."
                    )
                seen[r] = s.name

    def _topo_sort(self, specs: List[PluginSpec]) -> List[PluginSpec]:
        """
        Order specs so resource providers come before consumers.
        Within each resource-tier, preserve registration order so
        `kazka_plugins.py` reads top-to-bottom for unrelated plugins.
        """
        by_name = {s.name: s for s in specs}
        provider_of: Dict[str, str] = {
            r: s.name for s in specs for r in s.provides_resource
        }

        # Build edges: spec -> specs it depends on.
        deps: Dict[str, Set[str]] = {s.name: set() for s in specs}
        for s in specs:
            for r in s.requires_resource:
                provider = provider_of.get(r)
                if provider and provider != s.name:
                    deps[s.name].add(provider)

        ordered: List[PluginSpec] = []
        placed: Set[str] = set()

        # Repeatedly emit specs whose deps are all placed, in original order.
        remaining = list(specs)
        while remaining:
            progress = False
            still: List[PluginSpec] = []
            for s in remaining:
                if deps[s.name].issubset(placed):
                    ordered.append(s)
                    placed.add(s.name)
                    progress = True
                else:
                    still.append(s)
            remaining = still
            if not progress:
                cycle_names = [s.name for s in remaining]
                raise PluginLoadError(
                    f"Resource dependency cycle among plugins: {cycle_names}"
                )

        return ordered

    def _build(self, spec: PluginSpec) -> Any:
        # Pass only the resources this plugin declared it needs.
        offered = {name: self.resources[name] for name in spec.requires_resource}

        result = spec.factory(self.engine, self.config, offered)
        if not isinstance(result, PluginBuild):
            raise PluginLoadError(
                f"Factory for '{spec.name}' must return PluginBuild, "
                f"got {type(result).__name__}"
            )

        # Validate the factory provided everything the spec promised.
        promised = set(spec.provides_resource)
        delivered = set(result.resources)
        missing = promised - delivered
        if missing:
            raise PluginLoadError(
                f"Plugin '{spec.name}' declared provides_resource={sorted(promised)} "
                f"but factory returned only {sorted(delivered)}; missing {sorted(missing)}"
            )
        extra = delivered - promised
        if extra:
            raise PluginLoadError(
                f"Plugin '{spec.name}' factory returned undeclared resources "
                f"{sorted(extra)} — add them to provides_resource or remove them."
            )

        self.resources.update(result.resources)
        return result.plugin

    def _register_with_engine(self, spec: PluginSpec, plugin: Any) -> None:
        if spec.kind == "input":
            self.engine.register_input(spec.name, plugin)
        elif spec.kind == "output":
            self.engine.register_output(spec.name, plugin)
        elif spec.kind == "service":
            self.engine.register_service(spec.name, plugin)
        else:
            raise PluginLoadError(
                f"Plugin '{spec.name}' has unknown kind '{spec.kind}' "
                f"(expected 'input', 'output', or 'service')"
            )

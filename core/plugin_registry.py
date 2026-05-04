"""
Plugin manifest collection.

Each repo that contributes plugins to Kazka exposes a manifest module
(by convention `kazka_plugins`) whose top level calls `register_plugin`
once per plugin. Importing the manifest collects specs into a module-
local list; `take_specs()` drains it.

The factory in each spec is expected to be a thin closure that defers
heavy imports until called — see `kazka_plugins.py` for the pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

PluginKind = str  # "input" | "output" | "service"


@dataclass
class PluginBuild:
    """
    Return value of a plugin factory.

    `plugin` is the constructed plugin instance. `resources` is the dict
    of resources this plugin provides to others; it MUST contain every
    name listed in the spec's `provides_resource` (the loader checks).
    """
    plugin: Any
    resources: dict = field(default_factory=dict)


FactoryFn = Callable[[Any, Any, dict], PluginBuild]
# Signature: factory(engine, cfg, resources) -> PluginBuild
# - engine: AssistantEngine
# - cfg: per-plugin config (currently the global config object; see loader)
# - resources: dict containing only the resources this plugin declared
#              `requires_resource` for


@dataclass
class PluginSpec:
    """Declarative description of a plugin."""
    name: str
    kind: PluginKind
    factory: FactoryFn

    # If True, the plugin always loads — no CLI flag, no config gate.
    # Used for inputs the system can't function without (text, console).
    always_on: bool = False

    # If False, the plugin is loaded only when explicitly enabled
    # (via config or CLI). Default is True: load unless disabled.
    enabled_default: bool = True

    # Resources this plugin needs from others, by name. The loader
    # builds providers first and passes them in via the `resources`
    # arg of the factory.
    requires_resource: List[str] = field(default_factory=list)

    # Resources this plugin exposes to others, by name. The factory's
    # returned PluginBuild.resources MUST contain each of these.
    provides_resource: List[str] = field(default_factory=list)

    # Free-form description for --help / introspection.
    description: str = ""


_specs: List[PluginSpec] = []


def register_plugin(
    name: str,
    kind: PluginKind,
    factory: FactoryFn,
    *,
    always_on: bool = False,
    enabled_default: bool = True,
    requires_resource: Optional[List[str]] = None,
    provides_resource: Optional[List[str]] = None,
    description: str = "",
) -> None:
    """
    Register a plugin spec. Called from manifest modules at import time.

    Specs are buffered until `take_specs()` is called by the loader.
    """
    _specs.append(PluginSpec(
        name=name,
        kind=kind,
        factory=factory,
        always_on=always_on,
        enabled_default=enabled_default,
        requires_resource=list(requires_resource or []),
        provides_resource=list(provides_resource or []),
        description=description,
    ))


def take_specs() -> List[PluginSpec]:
    """
    Return all collected specs and clear the buffer.

    The loader calls this once after importing every manifest module.
    Tests can call it to reset state between runs.
    """
    out = list(_specs)
    _specs.clear()
    return out

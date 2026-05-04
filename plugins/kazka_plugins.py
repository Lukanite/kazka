"""
Plugin manifest for the Kazka core repo.

Every plugin this repo provides is registered here. Factories MUST defer
heavy imports — see existing entries for the pattern. Discovery imports
this file early, so anything imported at module top level is paid for by
every Kazka startup, hardware-attached or not.

Pattern: each factory is a small named function whose first lines are
`from ... import ...` of the plugin class. Keep them short and uniform —
they're effectively a manifest table, not a place for logic.
"""

from core.plugin_registry import PluginBuild, register_plugin


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _make_voice(engine, cfg, resources):
    from plugins.inputs.voice.voice_plugin import VoiceInputPlugin
    return PluginBuild(VoiceInputPlugin(engine))


def _make_button(engine, cfg, resources):
    from plugins.inputs.button.button_plugin import ButtonInputPlugin
    return PluginBuild(ButtonInputPlugin(engine))


def _make_text(engine, cfg, resources):
    from plugins.inputs.text.text_plugin import TextInputPlugin
    plugin = TextInputPlugin(engine)
    # Wire 'q' key -> graceful shutdown (with memory save).
    plugin.on_shutdown(lambda: engine.shutdown(save_memories=True))
    return PluginBuild(plugin)


def _make_web_input(engine, cfg, resources):
    from plugins.inputs.web.web_plugin import WebInputPlugin
    plugin = WebInputPlugin(engine)
    return PluginBuild(plugin, resources={"web_server": plugin.web_server})


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _make_console(engine, cfg, resources):
    from plugins.outputs.console import ConsoleOutputPlugin
    return PluginBuild(ConsoleOutputPlugin(engine, {}))


def _make_tts(engine, cfg, resources):
    from plugins.outputs.tts_plugin import TTSOutputPlugin
    return PluginBuild(TTSOutputPlugin(engine))


def _make_led(engine, cfg, resources):
    from plugins.outputs.led_plugin import LEDOutputPlugin
    return PluginBuild(LEDOutputPlugin(engine))


def _make_web_output(engine, cfg, resources):
    from plugins.outputs.web_output_plugin import WebOutputPlugin
    return PluginBuild(WebOutputPlugin(engine, resources["web_server"]))


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

def _make_scheduler(engine, cfg, resources):
    from plugins.services.scheduler import SchedulerPlugin
    return PluginBuild(SchedulerPlugin(engine))


def _make_sleep_watchdog(engine, cfg, resources):
    from plugins.services.sleep_watchdog import SleepWatchdogPlugin
    return PluginBuild(SleepWatchdogPlugin(engine))


def _make_web_service(engine, cfg, resources):
    from plugins.services.web_service_plugin import WebServicePlugin
    return PluginBuild(WebServicePlugin(engine, resources["web_server"]))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

# Inputs
register_plugin(name="voice",  kind="input", factory=_make_voice,
                description="Wake word + VAD + STT pipeline (microphone)")
register_plugin(name="button", kind="input", factory=_make_button,
                description="GPIO button: short-press wake / hold for PTT")
register_plugin(name="text",   kind="input", factory=_make_text,
                always_on=True,
                description="Keyboard input + 'q' shutdown")
register_plugin(name="web_input", kind="input", factory=_make_web_input,
                provides_resource=["web_server"],
                description="Browser/WebSocket text input")

# Outputs
register_plugin(name="console", kind="output", factory=_make_console,
                always_on=True,
                description="Stdout streaming")
register_plugin(name="tts",     kind="output", factory=_make_tts,
                description="Text-to-speech audio output")
register_plugin(name="led",     kind="output", factory=_make_led,
                description="GPIO LED state feedback")
register_plugin(name="web_output", kind="output", factory=_make_web_output,
                requires_resource=["web_server"],
                description="Stream responses to web clients")

# Services
register_plugin(name="scheduler", kind="service", factory=_make_scheduler,
                always_on=True,
                description="Self-wake timer scheduler")
register_plugin(name="sleep_watchdog", kind="service", factory=_make_sleep_watchdog,
                description="Inactivity-driven memory flush + reset")
register_plugin(name="web_service", kind="service", factory=_make_web_service,
                requires_resource=["web_server"],
                description="Web UI lifecycle bridge (clear on sleep, etc.)")

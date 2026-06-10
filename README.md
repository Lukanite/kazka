# Kazka - A Naturally Learning Voice Assistant with Personality

<img width="960" height="640" alt="image" src="https://github.com/user-attachments/assets/c18e20cd-1628-46be-b5b4-b3aa952567de" />

## Kazka who?

Originally a project to make the [Google AIY Voice Kit](https://aiyprojects.withgoogle.com/voice/) what it had always longed to be - "Artificial Intelligence Yourself."

That meant decoupling it from any kind of cloud integration, and using its Linux-friendly hardware to create a completely local, extensible, free-form personal assistant, while retaining the audio, intent-driven functionalities that were previously solely the domain of the cloud into the home itself.

The goal was not just to replace a cloud voice assistant like Google, Alexa, or Siri, but to make one that could actually *afford* to be truly personal - one that would remember how you spoke to it last week, treated it, and what you asked of it, using natural learning and retention mechanisms without any preset "rules" on what it can and will remember... and when going that far, why not have some fun with the personality, too? 

What ended up coming out of it was a modular, naturally learning virtual assistant with easily customizeable and modular:
 - Brains (LLM Inference provider) - Anthropic or OpenAI API for easy [llama.cpp](https://github.com/ggml-org/llama.cpp) integration, LM Studio, or Cloud API integration
 - Ears (Speech to Text) - OpenAI API, or local Whisper
 - Mouth (Text to Speech) - OpenAI API, or local Piper
 - Personality, and memory forming - Prompt .txt files

## Features

1. A truly **Personalized** assistant: Bored of compliant assistants that do anything you ask? Swap out a single prompt file to add sass or any other personality you can think!
2. A **Learning** assistant that doesn't get overwhelmed: Uses your assistant's personality and LLM itself to remember facts and details across conversations, without going insane from context! Automatically creates memories and forgets the least important ones so the assistant can learn about you and stay up to date with what's happening.
2. **Local Data**: You're fully in control of the data that your assistant remembers about you! Look into its internal memorization processes, current memory, and conversation logs (and even tweak them, if you wish).
3. Can run **fully locally**: Running on a decently powerful machine? Do everything on-device (STT, LLM Inference, TTS) for complete privacy or delegate portions of it to other servers (Anthropic and OpenAI API compatible)
4. **Modular Plugins & Tools**: A manifest-driven plugin system. Add a class, register one line in `kazka_plugins.py` or `kazka_tools.py`, and you're done — no edits to the engine or entry point. Plugins can publish typed resources for other plugins or LLM tools to consume.
5. **Modular I/O**: Got some LEDs, buttons, or anything else you want to trigger or respond to your assistant? This project uses a flexible IO engine so you can register to be called on events, or call into the engine to create new inference requests.

## Quick Start

```bash
# 1. Run once — Kazka will create assistant_settings.toml for you and exit
pip3 install -r requirements.txt
python main.py

# 2. Add your OpenAI key to assistant_settings.toml
#    (used for LLM, transcription, and TTS out of the box)
#    Or export environment variables instead:
#      export LLM_API_KEY=sk-...
#      export REMOTE_API_KEY=sk-...
#      export REMOTE_TTS_KEY=sk-...

# 3. Run!
python main.py --text-only   # No mic/speaker needed — great first test
python main.py               # Full voice assistant
python main.py --no-voice    # Text + button input, with TTS output
python main.py --no-tts      # Voice input, console output only
python main.py --disable led # Disable any plugin by name (repeatable)
```

> **Prefer local?** `assistant_settings.toml` has ready-to-uncomment blocks for
> llama.cpp/Ollama (LLM), faster-whisper-server (STT), and Kokoro (TTS).

## Configuration

Edit `assistant_settings.toml`. The configuration is split into logical sections:

### Assistant Identity

The `assistant` section controls who your assistant is:

```toml
[assistant]
name = "Assistant"
system_prompt_file = "prompts/system_prompt.txt"
character_prompt_file = "prompts/character_prompt.txt"
enable_thinking_mode = false
max_chat_tokens = 1024
max_task_tokens = 4096
```

- **`name`**: Your assistant's name — used in console output, memory display, and conversation logs
- **`system_prompt_file`**: Path to a text file defining your assistant's personality and behavior
- **`enable_thinking_mode`**: Enable LLM chain-of-thought reasoning (slower but more capable)

### Network / API

The `network` section handles LLM API connectivity:

```toml
[network]
api_url = "https://api.openai.com/v1/chat/completions"
model = "gpt-4o-mini"
api_key = ""            # or set LLM_API_KEY env var
api_type = "openai"     # "openai" or "anthropic"
enable_cache_warming = false
enable_streaming = true
```

Supports both OpenAI-compatible APIs (OpenAI, Ollama, vLLM, OpenRouter, etc.) and Anthropic's native Messages API via `api_type`.

To switch providers, comment/uncomment the relevant block in `assistant_settings.toml` — pre-written blocks for local Ollama, vLLM/LM Studio, Anthropic, and OpenRouter are included.

### Wake Word

```toml
[wake_word]
target = "kazka"
model_file = "models/Kazka.onnx"
confidence_threshold = 0.5
vocab_prompt = "Kitsune,Kazka"  # Words that might be spoken in this context but are unusual, like names to recognize
phonetic_aliases = ["kamiska"]        # Known misheard variants of the wake word
max_levenshtein_distance = 1
```

A custom `Kazka.onnx` wake word model is included. You can train your own using [openWakeWord](https://github.com/dscripka/openWakeWord).

## Project Structure

```
├── core/
│   ├── engine.py            # Core engine with request queue
│   ├── requests.py          # Request/response objects
│   ├── plugin_base.py       # InputPlugin / OutputPlugin / ServicePlugin
│   ├── plugin_registry.py   # PluginSpec, PluginBuild, register_plugin
│   ├── plugin_loader.py     # Discover + topo-sort + build plugins
│   ├── tool_registry.py     # ToolSpec, ToolBuild, register_tool
│   ├── tool_loader.py       # Discover + build LLM tools
│   ├── config.py            # Configuration system
│   ├── llm_interface.py     # LLM communication (OpenAI-compatible)
│   ├── anthropic_llm_interface.py  # LLM communication (Anthropic)
│   ├── memory_manager.py    # Persistent memory system
│   └── tool_manager.py      # Tool registry + execution
├── plugins/
│   ├── kazka_plugins.py     # Plugin manifest (where new plugins are registered)
│   ├── inputs/
│   │   ├── voice/           # Wake word + VAD + STT
│   │   ├── button/          # Hardware button
│   │   ├── text/            # Keyboard input
│   │   └── web/             # Browser/WebSocket input
│   ├── outputs/
│   │   ├── console.py       # Stdout streaming
│   │   ├── tts_plugin.py    # Text-to-speech
│   │   ├── led_plugin.py    # LED visual feedback
│   │   └── web_output_plugin.py  # WebSocket streaming to browsers
│   └── services/
│       ├── scheduler.py             # Self-wake timers
│       ├── sleep_watchdog.py        # Inactivity-driven memory flush
│       ├── conversation_index.py    # Semantic search index
│       └── web_service_plugin.py    # Web UI lifecycle bridge
├── tools/
│   ├── kazka_tools.py       # Tool manifest (where new tools are registered)
│   ├── time_awareness.py    # get_datetime
│   ├── self_wake.py         # Schedule/cancel/list self-wakes
│   ├── matter.py            # Matter light control
│   └── conversation_search.py  # Semantic search over past conversations
├── prompts/                 # System prompts (customize these!)
├── tests/
│   ├── integration/         # End-to-end tests
│   └── test_*.py            # Unit tests
├── main.py                  # Entry point
├── assistant_settings.toml            # Your configuration (gitignored)
└── assistant_settings.example.toml    # Example configuration (commit this)
```

## Plugin & Tool Development

Kazka uses a manifest-driven plugin system. Every plugin and every LLM tool is
registered in one of two manifest files:

- **`plugins/kazka_plugins.py`** — the plugin manifest
- **`tools/kazka_tools.py`** — the LLM tool manifest

Adding a new plugin or tool is a two-step process: write the implementation,
then add a `register_*` entry pointing to it. The engine never imports plugin
or tool classes directly, so you never have to touch `engine.py` or `main.py`
to add a capability.

### Adding a Plugin

There are three plugin kinds:

| Kind | Base class | Purpose |
|------|-----------|---------|
| `input` | `InputPlugin` | Produces user input (voice, text, button, web) |
| `output` | `OutputPlugin` / `QueuedOutputPlugin` | Consumes assistant output (TTS, console, LED) |
| `service` | `ServicePlugin` | Observes engine state (scheduler, memory flush) |

**1.** Write the class. Same pattern as before — extend the appropriate base,
implement `start()` and `stop()`:

```python
# plugins/inputs/my_input/my_input_plugin.py
from core.plugin_base import InputPlugin

class MyInputPlugin(InputPlugin):
    def __init__(self, engine):
        super().__init__(engine, "my_input")

    def start(self):
        self.engine.register_endpoint("my_input", "trigger", self._on_trigger)

    def stop(self):
        pass

    def _on_trigger(self, data):
        self.emit_input(data.get("text", ""), {"source": "MY_INPUT"})
```

**2.** Register it in `plugins/kazka_plugins.py`. Factories MUST defer
heavy imports so the manifest stays cheap to read:

```python
def _make_my_input(engine, cfg, resources):
    from plugins.inputs.my_input.my_input_plugin import MyInputPlugin
    return PluginBuild(MyInputPlugin(engine))

register_plugin(name="my_input", kind="input", factory=_make_my_input,
                description="My new input source")
```

That's it. The loader picks it up on next startup. To disable it temporarily,
use `--disable my_input` on the command line — no per-plugin CLI flag needed.

### Adding a Tool

Tools work the same way:

**1.** Write the tool. Extend `core.tool_manager.Tool`:

```python
# tools/my_tool.py
from core.tool_manager import Tool

class MyTool(Tool):
    @property
    def name(self): return "my_tool"

    @property
    def description(self):
        return "What the tool does and when the LLM should call it."

    def execute(self, **kwargs):
        return {"result": "hello"}
```

**2.** Register it in `tools/kazka_tools.py`:

```python
def _make_my_tool(engine, cfg, resources):
    from tools.my_tool import MyTool
    return ToolBuild(MyTool())

register_tool("my_tool", _make_my_tool,
              description="What the tool does")
```

### Resources: Sharing Between Plugins and Tools

When two plugins need to share an object (e.g. the web server is used by
web_input, web_output, and web_service), or when a tool needs to talk to a
plugin (e.g. self-wake tools call into the scheduler plugin), the plugin
publishes a **resource** that other plugins or tools can request by name.

**Publishing a resource** — return it from your factory and declare it on the
spec:

```python
def _make_scheduler(engine, cfg, resources):
    from plugins.services.scheduler import SchedulerPlugin
    plugin = SchedulerPlugin(engine)
    # Expose a narrow facade, NOT the plugin itself.
    return PluginBuild(plugin, resources={"scheduler": plugin.api()})

register_plugin(name="scheduler", kind="service", factory=_make_scheduler,
                always_on=True,
                provides_resource=["scheduler"])
```

> **Tip:** Don't expose the raw plugin instance as a resource. Add a small
> facade class in the same file (e.g. `SchedulerPluginApi`) that exposes only
> what consumers should call. This keeps lifecycle methods (`start`/`stop`)
> and engine references out of the consumer's reach.

**Consuming a resource** — declare it on the spec and read it from the factory's
`resources` arg:

```python
def _make_schedule_self_wake(engine, cfg, resources):
    from tools.self_wake import ScheduleSelfWakeTool
    return ToolBuild(ScheduleSelfWakeTool(resources["scheduler"]))

register_tool("schedule_self_wake", _make_schedule_self_wake,
              requires_resource=["scheduler"])
```

If a required resource isn't available (e.g. the providing plugin was disabled),
the plugin loader fails fast (since the user explicitly opted into the plugin)
and the tool loader skips the dependent tool with a warning (since tools are
auto-bundled with their plugin dependency).

### Spec Reference

`register_plugin()` accepts:

| Field | Description |
|-------|-------------|
| `name` | Unique plugin name (used for `--disable`, endpoint targets, etc.) |
| `kind` | `"input"`, `"output"`, or `"service"` |
| `factory` | `factory(engine, cfg, resources) -> PluginBuild` |
| `always_on` | If True, can't be disabled via CLI or config |
| `enabled_default` | If False, only loads when explicitly enabled |
| `requires_resource` | List of resource names this plugin needs |
| `provides_resource` | List of resource names this plugin's factory will return |
| `description` | Free-form description for introspection |

`register_tool()` accepts a subset: `name`, `factory`, `requires_resource`,
`description`. Tools don't provide resources or have a `kind`.

### Third-Party Plugins & Tools

External packages can contribute plugins and tools without forking Kazka. In
their `pyproject.toml`:

```toml
[project.entry-points."kazka.plugins"]
my_pkg = "my_pkg.kazka_plugins"

[project.entry-points."kazka.tools"]
my_pkg = "my_pkg.kazka_tools"
```

The loader discovers these alongside the internal manifests during startup.

## Communication Between Plugins

Kazka has four communication mechanisms, each suited to a different problem.
When you're writing a plugin and need to interact with the rest of the system,
pick from this table first:

| If you want to… | Use | Direction | Timing |
|-----------------|-----|-----------|--------|
| Feed user input into the LLM pipeline | `emit_input()` (InputPlugin) | input → engine | async |
| Receive assistant output (text, tool calls) | `output()` / `output_chunk()` (OutputPlugin) | engine → outputs (broadcast) | async, engine thread |
| Call one specific peer plugin at runtime | **Endpoints** | plugin ↔ plugin (1-1) | async or sync |
| Hold a reference to a peer plugin's API | **Resources** | wired once at build time | n/a (DI) |
| React to engine lifecycle (interaction/sleep/undo) | **Service hooks** | engine → service plugin | sync, engine thread |

Rules of thumb:

- **Endpoints vs. resources is the most common point of confusion.** Use a
  resource when plugin B's API is part of plugin A's *construction* — A needs
  to hold a reference to talk to B repeatedly. Use an endpoint when the
  interaction is occasional and discovered at runtime ("the button was just
  pressed; tell voice to start listening").
- **`endpoint_call` blocks until the engine thread services it.** Cheap from
  background threads; safe from the engine thread too (it short-circuits to
  a direct dispatch — see `core/engine.py`).
- **Service hooks run inline on the engine thread.** Keep them
  near-instantaneous. If a hook needs to do real work, spawn a timer or
  thread from inside it (see `plugins/services/sleep_watchdog.py`).

### Endpoints

1-1 messaging between plugins. A plugin registers a named endpoint during
`start()`; any other plugin (or input thread) can call into it.

```python
# In a plugin's start()
engine.register_endpoint("voice", "wake_requested", self._on_wake_requested)

# From anywhere
engine.endpoint_send("voice", "wake_requested", {'source': 'button'})  # fire-and-forget
response = engine.endpoint_call("voice", "get_state", {})              # waits for response
```

Common built-in endpoints:

| Component | Endpoint | Description |
|-----------|----------|-------------|
| voice | wake_requested | Skip wake word, start listening |
| voice | ptt_started | Start push-to-talk recording |
| voice | ptt_stopped | Stop PTT, process audio |
| voice | get_state | Get current voice state |
| led | set_state | Update LED pattern |
| text | submit | Submit text input |

### Resources

Build-time dependency injection. A plugin's factory returns a narrow facade in
`PluginBuild(plugin, resources={"name": facade})`; consumers declare
`requires_resource=["name"]` and read it from the factory's `resources` arg.
See "Resources: Sharing Between Plugins and Tools" above for the full pattern.

Prefer resources over endpoints when:
- The consumer needs the API at construction time (it stores the reference).
- The contract is stable enough to express as method calls (`scheduler.schedule(...)`)
  rather than message names.
- The same handle is used many times — endpoint dispatch overhead and
  stringly-typed names get tedious.

### Service Hooks

`ServicePlugin` subclasses can override these methods to observe the engine:

| Hook | When |
|------|------|
| `on_interaction_start(text, metadata, images)` | Interaction starting (user input or wake), before LLM dispatch |
| `on_interaction_end()` | LLM finished responding to a user turn |
| `on_sleep_complete()` | After a sleep cycle (memory flush + reset) |
| `on_undo()` | After a turn was successfully undone |
| `on_conversation_log_saved(log_path)` | A `.jsonl` conversation log was written |

All hooks run *inline on the engine thread*. That means:

- You can safely read engine state (conversation history, plugin registries)
  without locks.
- You must not block. If you need to do real work, do what `sleep_watchdog`
  does: set a `threading.Timer` from inside the hook and let it fire on its
  own thread.
- `endpoint_call` from a hook is safe — it short-circuits to a direct dispatch
  when invoked on the engine thread.

## Voice Plugin States

```
WAITING → VERIFYING → LISTENING → PROCESSING_VAD → WAITING
   │                      ↑
   └── PTT ──────────────→ PROCESSING_PTT ──→ WAITING
```

- **WAITING**: Listening for wake word
- **VERIFYING**: Phonetic verification of detected wake word
- **LISTENING**: Recording command with VAD
- **PTT**: Push-to-talk recording (no VAD end detection)
- **PROCESSING_***: Transcribing and querying LLM

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_engine

# Run integration tests
python -m unittest tests.integration.test_full_system

# Verbose output
python -m unittest discover tests -v
```

## Hardware Requirements

For full voice assistant functionality (on any Windows/Linux platform, not just Raspberry Pi!):
- Microphone (USB or I2S)
- Speaker (USB, I2S, or 3.5mm)
- Optional: GPIO button + LED (This one *is* for Raspberry Pi, but for other platforms just swap out the GPIO/LED plugins!)

For text-only mode:
- Just a terminal!

## License

Apache 2.0 Licence - see LICENSE file for details.

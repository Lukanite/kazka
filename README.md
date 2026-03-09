# Kazka - A Naturally Learning Voice Assistant with Personality

A locally-driven personal Voice Assistant with easily customizeable and modular:
 - Brains (LLM Inference provider) - Anthropic or OpenAI API
 - Ears (Speech to Text) - OpenAI API, or local Whisper
 - Mouth (Text to Speech) - OpenAI API, or local Piper
 - Personality - Prompt .txt files

## Features

1. A truly **Personalized** assistant: Bored of compliant assistants that do anything you ask? Swap out a single prompt file to add sass or any other personality you can think!
2. A **Learning** assistant that doesn't get overwhelmed: Uses your assistant's personality and LLM itself to remember facts and details across conversations, without going insane from context! Automatically creates memories and forgets the least important ones so the assistant can learn about you and stay up to date with what's happening.
2. **Local Data**: You're fully in control of the data that your assistant remembers about you! Look into its internal memorization processes, current memory, and conversation logs (and even tweak them, if you wish).
3. Can run **fully locally**: Running on a decently powerful machine? Do everything on-device (STT, LLM Inference, TTS) for complete privacy or delegate portions of it to other servers (Anthropic and OpenAI API compatible)
4. **Modular Plugins**: Want your assistant to be able to do something? Easily integrate your own tools via a single Python file
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
```

> **Prefer local?** `assistant_settings.toml` has ready-to-uncomment blocks for
> Ollama (LLM), faster-whisper-server (STT), and Kokoro (TTS).

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
│   ├── engine.py          # Core engine with request queue
│   ├── requests.py        # Request/response objects
│   ├── plugin_base.py     # Base plugin classes
│   ├── config.py          # Configuration system
│   ├── llm_interface.py   # LLM communication (OpenAI-compatible)
│   ├── anthropic_llm_interface.py  # LLM communication (Anthropic)
│   ├── memory_manager.py  # Persistent memory system
│   └── tool_manager.py    # Tool execution
├── plugins/
│   ├── inputs/
│   │   ├── voice/         # Wake word + VAD + STT
│   │   ├── button/        # Hardware button + LED controller
│   │   └── text/          # Keyboard input
│   └── outputs/
│       ├── console.py     # Console output
│       ├── tts_plugin.py  # Text-to-speech
│       └── led_plugin.py  # LED visual feedback
├── tools/                 # Tool implementations
├── prompts/               # System prompts (customize these!)
├── tests/
│   ├── integration/       # End-to-end tests
│   └── test_*.py          # Unit tests
├── main.py                # Entry point
├── assistant_settings.toml        # Your configuration (gitignored)
└── assistant_settings.example.toml  # Example configuration (commit this)
```

## Plugin Development

### Creating an Input Plugin

```python
from core.plugin_base import InputPlugin

class MyInputPlugin(InputPlugin):
    def __init__(self, engine, config=None):
        super().__init__(engine, "my_input")
        self.config = config or {}

    def start(self):
        # Initialize resources
        # Optionally register endpoints
        self.engine.register_endpoint("my_input", "trigger", self._on_trigger)

    def stop(self):
        # Cleanup resources
        pass

    def _on_trigger(self, data):
        # Handle endpoint call
        text = data.get('text', '')
        self.emit_input(text, {'source': 'MY_INPUT'})
```

### Creating an Output Plugin

For outputs that need sequential processing (like TTS):

```python
from core.plugin_base import QueuedOutputPlugin

class MyOutputPlugin(QueuedOutputPlugin):
    def __init__(self, engine, config=None):
        super().__init__(engine, "my_output")

    def start_internal(self):
        # Initialize resources
        pass

    def _process_output(self, text, metadata):
        # Handle output (runs in worker thread, can block)
        print(f"Output: {text}")

    def should_handle(self, metadata):
        # Return True if this output should handle the message
        return metadata.get('source') in ['VAD', 'PTT']

    def stop_internal(self):
        # Cleanup resources
        pass
```

## Endpoint System

Plugins communicate via endpoints for 1-1 messaging:

```python
# Register an endpoint
engine.register_endpoint("voice", "wake_requested", handler_callback)

# Send async message (fire-and-forget)
engine.endpoint_send("voice", "wake_requested", {'source': 'button'})

# Sync call (waits for response)
response = engine.endpoint_call("voice", "get_state", {})
```

### Common Endpoints

| Component | Endpoint | Description |
|-----------|----------|-------------|
| voice | wake_requested | Skip wake word, start listening |
| voice | ptt_started | Start push-to-talk recording |
| voice | ptt_stopped | Stop PTT, process audio |
| voice | get_state | Get current voice state |
| led | set_state | Update LED pattern |
| text | submit | Submit text input |

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

For full voice assistant functionality:
- Microphone (USB or I2S)
- Speaker (USB, I2S, or 3.5mm)
- Optional: GPIO button + LED (Raspberry Pi)

For text-only mode:
- Just a terminal!

## License

Apache 2.0 Licence - see LICENSE file for details.

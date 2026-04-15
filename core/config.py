"""
Configuration module for the voice assistant.
Loads settings from assistant_settings.toml file for easy configuration management.
"""

import tomllib
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


# ============================================================================
# SYSTEM-LEVEL TRIGGER CONSTANTS
# ============================================================================

# System-initiated speech is now handled via query_system_initiated() method
# No trigger string needed


@dataclass
class AssistantConfig:
    """Assistant identity and behavior configuration."""
    name: str = "Assistant"  # Configurable assistant name
    system_prompt_file: str = "prompts/system_prompt.txt"
    character_prompt_file: str = "prompts/character_prompt.txt"
    _system_prompt: str = ""  # Internal cache for the loaded prompt template
    # Thinking mode settings
    enable_thinking_mode: bool = False  # Toggle to enable thinking mode (disabled by default for faster responses)
    # Token limits for LLM responses (None = server default)
    max_chat_tokens: Optional[int] = None  # Limit for conversation queries
    max_task_tokens: Optional[int] = None  # Limit for background tasks (memory, summarization, etc.)

    @staticmethod
    def _apply_substitutions(template: str, **kwargs) -> str:
        """Apply {placeholder} substitutions to a template string."""
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", value)
        return result

    def _load_character_prompt(self) -> str:
        """Load and return the character prompt with {assistant_name} substituted."""
        if not os.path.exists(self.character_prompt_file):
            return ""
        try:
            with open(self.character_prompt_file, 'r', encoding='utf-8') as f:
                character_prompt = f.read().strip()
            character_prompt = self._apply_substitutions(character_prompt, assistant_name=self.name)
            print(f"📝 Loaded character prompt from {self.character_prompt_file}")
            return character_prompt
        except Exception as e:
            print(f"⚠️  Failed to load character prompt from {self.character_prompt_file}: {e}")
            return ""

    def get_system_prompt(self, resume_history: str = "") -> str:
        """
        Load and cache the system prompt from file, with placeholder substitution.

        The cached template retains {resume_history} as a literal placeholder.
        The resume_history parameter is substituted on each call.

        Args:
            resume_history: Formatted previous conversation text to inject

        Returns:
            The fully-substituted system prompt
        """
        if not self._system_prompt:
            try:
                if os.path.exists(self.system_prompt_file):
                    with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
                        self._system_prompt = f.read().strip()
                    print(f"📝 Loaded system prompt from {self.system_prompt_file}")
                else:
                    print(f"⚠️  System prompt file '{self.system_prompt_file}' not found. Using hardcoded default.")
                    self._system_prompt = "You are {assistant_name}, a helpful voice assistant."
            except Exception as e:
                print(f"❌ Failed to load system prompt from {self.system_prompt_file}: {e}")
                self._system_prompt = "You are {assistant_name}, a helpful voice assistant."

            # Load character prompt and substitute placeholders
            character_prompt = self._load_character_prompt()

            self._system_prompt = self._apply_substitutions(
                self._system_prompt,
                character_prompt=character_prompt,
                assistant_name=self.name,
            )

        # Apply resume_history on each call (not cached)
        return self._apply_substitutions(self._system_prompt, resume_history=resume_history)

    def get_task_system_prompt(self, **kwargs) -> str:
        """
        Load the task system prompt from file and apply placeholder substitutions.

        The task system prompt is used for background LLM tasks (memory extraction,
        ranking, summarization, etc.) and is not cached — it is loaded fresh each
        call so that callers can inject dynamic values via keyword arguments.

        {assistant_name} is always substituted automatically. Any additional
        placeholders can be supplied as keyword arguments, e.g.::

            config.assistant.get_task_system_prompt(user_name="Alice")

        Args:
            **kwargs: Extra placeholder values to substitute in the template.

        Returns:
            The fully-substituted task system prompt string.
        """
        from core.config import config as _config  # avoid circular import at module level
        task_prompt_file = _config.memory.get_prompt_file('task_system')
        try:
            with open(task_prompt_file, 'r', encoding='utf-8') as f:
                template = f.read().strip()
        except Exception as e:
            print(f"⚠️  Failed to load task system prompt from {task_prompt_file}: {e}")
            template = "You are a concise, neutral text processor."

        character_prompt = self._load_character_prompt()
        return self._apply_substitutions(
            template,
            assistant_name=self.name,
            character_prompt=character_prompt,
            **kwargs,
        )

    def set_system_prompt(self, prompt: str):
        """Set system prompt directly (for testing or programmatic use)."""
        self._system_prompt = prompt


@dataclass
class NetworkConfig:
    """Network configuration for LLM API connection."""
    api_url: str = "http://192.168.25.19:5000/v1/chat/completions"
    model: str = "llama3"
    api_key: Optional[str] = None  # API key for remote LLM providers (OpenAI-compatible). None = no auth.
    api_type: str = "openai"  # API format: "openai" (default) or "anthropic"
    # Cache warming settings
    enable_cache_warming: bool = True  # Warm up LLM prompt cache on startup
    # Streaming settings
    enable_streaming: bool = True  # Enable streaming LLM responses for real-time console output


@dataclass
class AudioDeviceConfig:
    """Audio device configuration."""
    input_device_name: str = "Yeti"
    output_device_name: str = "DigiHug"
    mic_sample_rate: int = 48000
    sample_rate: int = 16000
    chunk_size: int = 1280

    @property
    def downsample_factor(self) -> int:
        """Calculate downsample factor from mic to target sample rate."""
        return int(self.mic_sample_rate / self.sample_rate)


@dataclass
class WakeWordConfig:
    """Wake word detection configuration."""
    target: str = ""
    model_file: str = ""
    confidence_threshold: float = 0.5
    vocab_prompt: str = ""
    # Phonetic verification settings
    phonetic_aliases: list = None  # Known misheard variants of the wake word
    max_levenshtein_distance: int = 1  # Max edit distance for fuzzy phonetic matching

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.phonetic_aliases is None:
            self.phonetic_aliases = []



@dataclass
class TTSConfig:
    """Text-to-speech configuration."""
    enabled: bool = True  # Enable local TTS (can be used as fallback to remote TTS)
    binary_path: str = "./piper/piper"
    voice_model: str = "./en_US-kathleen-low.onnx"

    def get_voice_config_path(self) -> str:
        """Get the path to the voice model JSON configuration."""
        return f"{self.voice_model}.json"


@dataclass
class HardwareConfig:
    """Hardware optimization configuration."""
    compute_type: str = "int8"
    device: str = "cpu"


@dataclass
class SpeechRecognitionConfig:
    """Speech recognition configuration."""
    verifier_model: str = "tiny.en"
    scribe_model: str = "distil-small.en"

    # Vocabulary prompt to help Whisper recognize domain-specific words
    # (e.g., proper nouns, technical terms)
    vocab_prompt: str = ""

    # Remote OpenAI-compatible API configuration
    use_remote_api: bool = False
    remote_api_url: str = "http://faceless:8881/v1/audio/transcriptions"
    remote_api_model: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    remote_api_use_multipart: bool = True
    remote_api_key: str = ""


@dataclass
class RemoteTTSConfig:
    """Remote OpenAI-compatible TTS configuration."""
    use_remote_tts: bool = False
    remote_api_url: str = "http://faceless:8880/v1/audio/speech"
    remote_tts_model: str = "kokoro"
    remote_tts_voice: str = "af_heart"
    remote_tts_format: str = "wav"
    remote_api_key: str = ""


@dataclass
class MemoryConfig:
    """Memory management configuration with category-based limits."""
    file_path: str = "assistant_memories.json"
    prompts_dir: str = "prompts"

    # Category-specific memory limits for context injection
    # Personal facts like names should never be pushed out
    max_user_facts: int = 10   # Keep all personal facts (names, location, etc.)
    max_self_facts: int = 10    # Keep assistant's personal memories about itself
    max_preferences: int = 10    # Keep recent preferences
    max_events: int = 10         # Keep recent important events
    max_projects: int = 10       # Keep recent projects/tasks
    max_conversation: int = 5   # Keep minimal conversation summaries

    # Conversation log directory. Set to null/None to disable conversation logging.
    conversation_log_dir: Optional[str] = "log"

    # Memory backup directory. Set to null/None to disable memory backups.
    memory_backup_dir: Optional[str] = "log"

    # Number of previous conversation exchanges to include in system prompt on startup
    resume_history_count: int = 3

    def get_prompt_file(self, prompt_name: str) -> str:
        """Get the full path to a prompt file."""
        return f"{self.prompts_dir}/{prompt_name}.txt"


@dataclass
class ConversationSearchConfig:
    """Configuration for semantic search over conversation logs."""
    enabled: bool = False  # Disabled by default until model is set up
    model_path: str = "models/all-MiniLM-L6-v2.onnx"
    tokenizer_path: str = "models/tokenizer.json"
    index_dir: str = "search_index"
    top_k: int = 3  # Number of search results to return
    context_window: int = 1  # Turns of context around each search result
    min_score: float = 0.15  # Minimum cosine similarity to include a result


@dataclass
class WebConfig:
    """Web server plugin configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class ConsoleConfig:
    """Console output plugin configuration."""
    show_thinking: bool = False  # Show LLM thinking/reasoning in console output


@dataclass
class SleepConfig:
    """Sleep watchdog configuration for periodic memory flush and conversation reset."""
    enabled: bool = True
    inactivity_minutes: int = 10    # Minutes of inactivity before sleep can trigger
    sleeping_hours_start: int = 2   # Start of sleeping window (hour, 24h format)
    sleeping_hours_end: int = 6     # End of sleeping window (hour, 24h format)
    min_exchanges: int = 5          # Minimum exchanges since last sleep before triggering


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    model_path: str = "models/silero_vad_v4.onnx"  # Path to VAD model file
    silence_threshold: float = 0.5  # Probability threshold for silence detection (0.0 to 1.0)


@dataclass
class ToolsConfig:
    """Tool and MCP integration configuration with plugin-style settings."""
    enable_tools: bool = True  # Master toggle for tool system
    enable_native_function_calling: bool = True  # Use native OpenAI function calling
    enable_intent_parser: bool = False  # Use intent parser for tool selection
    disabled_tools: list = None  # Blacklist of disabled tool names

    # Tool-specific configurations namespace
    # Each tool has its own subsection with 'enabled' flag and tool-specific options
    tool_settings: dict = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.disabled_tools is None:
            self.disabled_tools = []
        if self.tool_settings is None:
            self.tool_settings = {}


class Config:
    """
    Main configuration class that loads and manages all settings.

    This class loads configuration from assistant_settings.json and provides
    access to all configuration sections.
    """

    def __init__(self, config_file: str = "assistant_settings.toml"):
        """
        Initialize configuration by loading from JSON file.

        Args:
            config_file: Path to the configuration JSON file
        """
        self.config_file = config_file
        self.assistant = AssistantConfig()
        self.network = NetworkConfig()
        self.audio_devices = AudioDeviceConfig()
        self.wake_word = WakeWordConfig()
        self.tts = TTSConfig()
        self.hardware = HardwareConfig()
        self.speech_recognition = SpeechRecognitionConfig()
        self.remote_tts = RemoteTTSConfig()
        self.memory = MemoryConfig()
        self.web = WebConfig()
        self.console = ConsoleConfig()
        self.sleep = SleepConfig()
        self.tools = ToolsConfig()
        self.vad = VADConfig()
        self.conversation_search = ConversationSearchConfig()

        self._load_from_file()
        self._load_env_overrides()  # Optional environment variable overrides

    def _load_from_file(self):
        """Load configuration settings from the TOML file."""
        if not os.path.exists(self.config_file):
            example_file = "assistant_settings.example.toml"
            if os.path.exists(example_file):
                import shutil
                shutil.copy(example_file, self.config_file)
                print(f"")
                print(f"╔══════════════════════════════════════════════════════════════╗")
                print(f"║              Welcome to Kazka! First-time setup.            ║")
                print(f"╠══════════════════════════════════════════════════════════════╣")
                print(f"║  A starter config has been created:                         ║")
                print(f"║    {self.config_file:<56}  ║")
                print(f"║                                                              ║")
                print(f"║  Add your API keys, then run again:                         ║")
                print(f"║    python main.py                (full assistant)           ║")
                print(f"║    python main.py --text-only    (no mic/speakers needed)   ║")
                print(f"╚══════════════════════════════════════════════════════════════╝")
                print(f"")
            else:
                print(f"⚠️  Configuration file '{self.config_file}' not found. Using defaults.")
                self._create_default_config_file()
                return
            raise SystemExit(0)

        try:
            with open(self.config_file, 'rb') as f:
                data = tomllib.load(f)

            print(f"📋 Loading configuration from {self.config_file}")

            # Load assistant configuration section
            if 'assistant' in data:
                assistant_data = data['assistant'].copy()
                # Handle system_prompt if present (direct prompt override)
                if 'system_prompt' in assistant_data:
                    self.assistant.set_system_prompt(assistant_data['system_prompt'])
                    del assistant_data['system_prompt']
                self._update_dataclass(self.assistant, assistant_data)

            # Load each configuration section
            if 'network' in data:
                self._update_dataclass(self.network, data['network'])
            if 'audio_devices' in data:
                self._update_dataclass(self.audio_devices, data['audio_devices'])
            if 'wake_word' in data:
                self._update_dataclass(self.wake_word, data['wake_word'])
            if 'tts' in data:
                self._update_dataclass(self.tts, data['tts'])
            if 'hardware' in data:
                self._update_dataclass(self.hardware, data['hardware'])
            if 'speech_recognition' in data:
                self._update_dataclass(self.speech_recognition, data['speech_recognition'])
            if 'remote_tts' in data:
                self._update_dataclass(self.remote_tts, data['remote_tts'])
            if 'memory' in data:
                self._update_dataclass(self.memory, data['memory'])
            if 'web' in data:
                self._update_dataclass(self.web, data['web'])
            if 'console' in data:
                self._update_dataclass(self.console, data['console'])
            if 'sleep' in data:
                self._update_dataclass(self.sleep, data['sleep'])
            if 'tools' in data:
                tools_data = data['tools'].copy()
                # Handle tool_settings separately to preserve nested structure
                tool_settings = tools_data.pop('tool_settings', {})
                self._update_dataclass(self.tools, tools_data)
                self.tools.tool_settings = tool_settings
            if 'vad' in data:
                self._update_dataclass(self.vad, data['vad'])
            if 'conversation_search' in data:
                self._update_dataclass(self.conversation_search, data['conversation_search'])

            print("✅ Configuration loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load configuration from {self.config_file}: {e}")
            print("Using default configuration values.")

    def _create_default_config_file(self):
        """Create a default configuration file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write(self._build_toml_string())
            print(f"📝 Created default configuration file: {self.config_file}")
        except Exception as e:
            print(f"❌ Failed to create default configuration file: {e}")

    @staticmethod
    def _toml_value(value) -> str:
        """Format a Python value as a TOML literal."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{value}"'
        if value is None:
            return '""'
        if isinstance(value, list):
            items = ", ".join(Config._toml_value(v) for v in value)
            return f"[{items}]"
        return str(value)

    @staticmethod
    def _dict_to_toml_section(d: Dict[str, Any], prefix: str = "") -> str:
        """Recursively render a flat-valued dict as TOML key = value lines.
        Nested dicts are rendered as sub-tables using dotted headers."""
        lines = []
        nested = {}
        for key, value in d.items():
            if isinstance(value, dict):
                nested[key] = value
            else:
                lines.append(f"{key} = {Config._toml_value(value)}")
        result = "\n".join(lines)
        for key, sub in nested.items():
            header = f"{prefix}.{key}" if prefix else key
            result += f"\n\n[{header}]\n"
            result += Config._dict_to_toml_section(sub, header)
        return result

    def _build_toml_string(self) -> str:
        """Build a TOML string representing the current configuration."""
        a, n, ad, ww = self.assistant, self.network, self.audio_devices, self.wake_word
        t, hw, sr, rt = self.tts, self.hardware, self.speech_recognition, self.remote_tts
        m, c, sl, to, vad, cs, wb = self.memory, self.console, self.sleep, self.tools, self.vad, self.conversation_search, self.web

        def v(val):
            return self._toml_value(val)

        lines = []

        lines.append("[assistant]")
        lines.append(f'name = {v(a.name)}')
        lines.append(f'system_prompt_file = {v(a.system_prompt_file)}')
        lines.append(f'character_prompt_file = {v(a.character_prompt_file)}')
        lines.append(f'enable_thinking_mode = {v(a.enable_thinking_mode)}')
        lines.append(f'max_chat_tokens = {v(a.max_chat_tokens)}')
        lines.append(f'max_task_tokens = {v(a.max_task_tokens)}')

        lines.append("\n[network]")
        lines.append(f'api_url = {v(n.api_url)}')
        lines.append(f'model = {v(n.model)}')
        lines.append(f'api_key = {v(n.api_key)}')
        lines.append(f'api_type = {v(n.api_type)}')
        lines.append(f'enable_cache_warming = {v(n.enable_cache_warming)}')
        lines.append(f'enable_streaming = {v(n.enable_streaming)}')

        lines.append("\n[audio_devices]")
        lines.append(f'input_device_name = {v(ad.input_device_name)}')
        lines.append(f'output_device_name = {v(ad.output_device_name)}')
        lines.append(f'mic_sample_rate = {v(ad.mic_sample_rate)}')
        lines.append(f'sample_rate = {v(ad.sample_rate)}')
        lines.append(f'chunk_size = {v(ad.chunk_size)}')

        lines.append("\n[wake_word]")
        lines.append(f'target = {v(ww.target)}')
        lines.append(f'model_file = {v(ww.model_file)}')
        lines.append(f'confidence_threshold = {v(ww.confidence_threshold)}')
        lines.append(f'vocab_prompt = {v(ww.vocab_prompt)}')
        lines.append(f'phonetic_aliases = {v(ww.phonetic_aliases)}')
        lines.append(f'max_levenshtein_distance = {v(ww.max_levenshtein_distance)}')

        lines.append("\n[tts]")
        lines.append(f'binary_path = {v(t.binary_path)}')
        lines.append(f'voice_model = {v(t.voice_model)}')

        lines.append("\n[hardware]")
        lines.append(f'compute_type = {v(hw.compute_type)}')
        lines.append(f'device = {v(hw.device)}')

        lines.append("\n[speech_recognition]")
        lines.append(f'verifier_model = {v(sr.verifier_model)}')
        lines.append(f'scribe_model = {v(sr.scribe_model)}')
        lines.append(f'vocab_prompt = {v(sr.vocab_prompt)}')
        lines.append(f'use_remote_api = {v(sr.use_remote_api)}')
        lines.append(f'remote_api_url = {v(sr.remote_api_url)}')
        lines.append(f'remote_api_key = {v(sr.remote_api_key)}')
        lines.append(f'remote_api_model = {v(sr.remote_api_model)}')
        lines.append(f'remote_api_use_multipart = {v(sr.remote_api_use_multipart)}')

        lines.append("\n[remote_tts]")
        lines.append(f'use_remote_tts = {v(rt.use_remote_tts)}')
        lines.append(f'remote_api_url = {v(rt.remote_api_url)}')
        lines.append(f'remote_api_key = {v(rt.remote_api_key)}')
        lines.append(f'remote_tts_model = {v(rt.remote_tts_model)}')
        lines.append(f'remote_tts_voice = {v(rt.remote_tts_voice)}')
        lines.append(f'remote_tts_format = {v(rt.remote_tts_format)}')

        lines.append("\n[memory]")
        lines.append(f'file_path = {v(m.file_path)}')
        lines.append(f'prompts_dir = {v(m.prompts_dir)}')
        lines.append(f'max_user_facts = {v(m.max_user_facts)}')
        lines.append(f'max_self_facts = {v(m.max_self_facts)}')
        lines.append(f'max_preferences = {v(m.max_preferences)}')
        lines.append(f'max_events = {v(m.max_events)}')
        lines.append(f'max_projects = {v(m.max_projects)}')
        lines.append(f'max_conversation = {v(m.max_conversation)}')
        lines.append(f'conversation_log_dir = {v(m.conversation_log_dir)}')
        lines.append(f'memory_backup_dir = {v(m.memory_backup_dir)}')
        lines.append(f'resume_history_count = {v(m.resume_history_count)}')

        lines.append("\n[web]")
        lines.append(f'enabled = {v(wb.enabled)}')
        lines.append(f'host = {v(wb.host)}')
        lines.append(f'port = {v(wb.port)}')

        lines.append("\n[console]")
        lines.append(f'show_thinking = {v(c.show_thinking)}')

        lines.append("\n[sleep]")
        lines.append(f'enabled = {v(sl.enabled)}')
        lines.append(f'inactivity_minutes = {v(sl.inactivity_minutes)}')
        lines.append(f'sleeping_hours_start = {v(sl.sleeping_hours_start)}')
        lines.append(f'sleeping_hours_end = {v(sl.sleeping_hours_end)}')
        lines.append(f'min_exchanges = {v(sl.min_exchanges)}')

        lines.append("\n[tools]")
        lines.append(f'enable_tools = {v(to.enable_tools)}')
        lines.append(f'disabled_tools = {v(to.disabled_tools)}')
        if to.tool_settings:
            lines.append(self._dict_to_toml_section({"tool_settings": to.tool_settings}, "tools"))

        lines.append("\n[vad]")
        lines.append(f'model_path = {v(vad.model_path)}')
        lines.append(f'silence_threshold = {v(vad.silence_threshold)}')

        lines.append("\n[conversation_search]")
        lines.append(f'enabled = {v(cs.enabled)}')
        lines.append(f'model_path = {v(cs.model_path)}')
        lines.append(f'tokenizer_path = {v(cs.tokenizer_path)}')
        lines.append(f'index_dir = {v(cs.index_dir)}')
        lines.append(f'top_k = {v(cs.top_k)}')
        lines.append(f'context_window = {v(cs.context_window)}')
        lines.append(f'min_score = {v(cs.min_score)}')

        return "\n".join(lines) + "\n"

    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass object with data from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    def _load_env_overrides(self):
        """
        Load configuration overrides from environment variables.
        These override both defaults and JSON file settings.
        """
        # Assistant overrides
        if os.getenv("ASSISTANT_NAME"):
            self.assistant.name = os.getenv("ASSISTANT_NAME")
        if os.getenv("SYSTEM_PROMPT"):
            self.assistant.set_system_prompt(os.getenv("SYSTEM_PROMPT"))
        if os.getenv("SYSTEM_PROMPT_FILE"):
            self.assistant.system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE")
            self.assistant._system_prompt = ""  # Reset cache to force reload
        if os.getenv("ENABLE_THINKING_MODE"):
            self.assistant.enable_thinking_mode = os.getenv("ENABLE_THINKING_MODE").lower() == "true"

        # Network overrides
        if os.getenv("LLM_API_URL"):
            self.network.api_url = os.getenv("LLM_API_URL")
        if os.getenv("LLM_MODEL"):
            self.network.model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_API_KEY"):
            self.network.api_key = os.getenv("LLM_API_KEY")

        # Audio device overrides
        if os.getenv("INPUT_DEVICE_NAME"):
            self.audio_devices.input_device_name = os.getenv("INPUT_DEVICE_NAME")
        if os.getenv("OUTPUT_DEVICE_NAME"):
            self.audio_devices.output_device_name = os.getenv("OUTPUT_DEVICE_NAME")

        # Wake word overrides
        if os.getenv("WAKE_WORD_TARGET"):
            self.wake_word.target = os.getenv("WAKE_WORD_TARGET")
        if os.getenv("WAKE_WORD_MODEL_FILE"):
            self.wake_word.model_file = os.getenv("WAKE_WORD_MODEL_FILE")
        if os.getenv("WAKE_WORD_CONFIDENCE"):
            self.wake_word.confidence_threshold = float(os.getenv("WAKE_WORD_CONFIDENCE"))

        # TTS overrides
        if os.getenv("PIPER_BINARY"):
            self.tts.binary_path = os.getenv("PIPER_BINARY")
        if os.getenv("PIPER_VOICE"):
            self.tts.voice_model = os.getenv("PIPER_VOICE")

        # Hardware overrides
        if os.getenv("COMPUTE_TYPE"):
            self.hardware.compute_type = os.getenv("COMPUTE_TYPE")
        if os.getenv("DEVICE"):
            self.hardware.device = os.getenv("DEVICE")

        # Remote API overrides
        if os.getenv("USE_REMOTE_API"):
            self.speech_recognition.use_remote_api = os.getenv("USE_REMOTE_API").lower() == "true"
        if os.getenv("REMOTE_API_URL"):
            self.speech_recognition.remote_api_url = os.getenv("REMOTE_API_URL")
        if os.getenv("REMOTE_API_MODEL"):
            self.speech_recognition.remote_api_model = os.getenv("REMOTE_API_MODEL")
        if os.getenv("REMOTE_API_USE_MULTIPART"):
            self.speech_recognition.remote_api_use_multipart = os.getenv("REMOTE_API_USE_MULTIPART").lower() == "true"

        # Remote TTS overrides
        if os.getenv("USE_REMOTE_TTS"):
            self.remote_tts.use_remote_tts = os.getenv("USE_REMOTE_TTS").lower() == "true"
        if os.getenv("REMOTE_TTS_URL"):
            self.remote_tts.remote_api_url = os.getenv("REMOTE_TTS_URL")
        if os.getenv("REMOTE_TTS_MODEL"):
            self.remote_tts.remote_tts_model = os.getenv("REMOTE_TTS_MODEL")
        if os.getenv("REMOTE_TTS_VOICE"):
            self.remote_tts.remote_tts_voice = os.getenv("REMOTE_TTS_VOICE")
        if os.getenv("REMOTE_TTS_FORMAT"):
            self.remote_tts.remote_tts_format = os.getenv("REMOTE_TTS_FORMAT")

    def save_to_file(self):
        """Save current configuration to the TOML file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write(self._build_toml_string())
            print(f"💾 Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all current settings as a dictionary."""
        return {
            "assistant": asdict(self.assistant),
            "network": asdict(self.network),
            "audio_devices": asdict(self.audio_devices),
            "wake_word": asdict(self.wake_word),
            "tts": asdict(self.tts),
            "hardware": asdict(self.hardware),
            "speech_recognition": asdict(self.speech_recognition),
            "remote_tts": asdict(self.remote_tts),
            "memory": asdict(self.memory),
            "web": asdict(self.web),
            "console": asdict(self.console),
            "sleep": asdict(self.sleep),
            "tools": asdict(self.tools),
            "vad": asdict(self.vad),
            "conversation_search": asdict(self.conversation_search)
        }

    def print_current_config(self):
        """Print current configuration in a readable format."""
        print("\n📋 Current Voice Assistant Configuration:")
        print("=" * 50)

        settings = self.get_all_settings()
        for section, config_dict in settings.items():
            print(f"\n{section.upper()}:")
            for key, value in config_dict.items():
                print(f"  {key}: {value}")
        print("=" * 50)


# Global configuration instance
config = Config()

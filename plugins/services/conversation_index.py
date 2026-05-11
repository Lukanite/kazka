"""
Conversation Index Service Plugin.

Owns the semantic search index over conversation logs. Builds and loads the
index on startup, then appends to it whenever a new conversation log file is
written (triggered via on_conversation_log_saved from the engine).

The index instance is exposed as `self.index` for direct read access by the
conversation search tools registered in the engine.
"""

from typing import Optional

from core.plugin_base import ServicePlugin
from core.config import config


class ConversationIndexPluginApi:
    """Narrow facade over ConversationIndexPlugin for external consumers (tools).

    Exposes only `.index` (the underlying ConversationSearchIndex or None).
    Reads through to the plugin so callers observe late initialization
    correctly — the property may be None at construction time and become
    populated after the plugin's start() runs.
    """

    def __init__(self, plugin: 'ConversationIndexPlugin'):
        self._plugin = plugin

    @property
    def index(self) -> Optional['ConversationSearchIndex']:
        return self._plugin.index


class ConversationIndexPlugin(ServicePlugin):
    """
    Service plugin that owns the ConversationSearchIndex.

    Self-disables when [conversation_search] enabled = false or when the
    index fails to initialize — in either case `self.index` remains None
    and the tools see "search not available".
    """

    def __init__(self, engine: 'AssistantEngine'):
        super().__init__(engine, "conversation_index")
        self.index: Optional['ConversationSearchIndex'] = None

    def api(self) -> 'ConversationIndexPluginApi':
        """Return a narrow facade for external consumers (tools)."""
        return ConversationIndexPluginApi(self)

    def start(self):
        search_config = config.conversation_search
        if not search_config.enabled:
            return

        try:
            from core.conversation_search import ConversationSearchIndex
            log_dir = config.memory.conversation_log_dir or "log"
            self.index = ConversationSearchIndex(
                index_dir=search_config.index_dir,
                model_path=search_config.model_path,
                tokenizer_path=search_config.tokenizer_path,
                log_dir=log_dir
            )
            self.index.load()
            print(f"   ✅ Conversation search initialized ({self.index.get_entry_count()} indexed entries)")
        except Exception as e:
            print(f"   ⚠️  Conversation search not available: {e}")
            self.index = None

    def stop(self):
        pass

    def on_conversation_log_saved(self, log_path: str):
        if self.index is None:
            return
        try:
            self.index.index_conversation_log(log_path)
            self.index.save()
        except Exception as e:
            print(f"⚠️  Error updating search index: {e}")

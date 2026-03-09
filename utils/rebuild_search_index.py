"""
Rebuild the conversation search index from .jsonl log files.

Usage:
    python -m utils.rebuild_search_index [log_dir]

If log_dir is not provided, reads from assistant_settings.json config.
Overwrites the existing search index entirely.
"""

import sys
import os
import glob


def rebuild(log_dir: str = None):
    from core.config import config
    from core.conversation_search import ConversationSearchIndex

    search_config = config.conversation_search
    if log_dir is None:
        log_dir = config.memory.conversation_log_dir or "log"

    if not os.path.isdir(log_dir):
        print(f"❌ Log directory not found: {log_dir}")
        sys.exit(1)

    # Find all .jsonl files sorted by name (chronological by timestamp)
    log_files = sorted(glob.glob(os.path.join(log_dir, "conversation_*.jsonl")))
    if not log_files:
        print(f"❌ No conversation_*.jsonl files found in {log_dir}")
        sys.exit(1)

    print(f"📂 Found {len(log_files)} conversation log files in {log_dir}")

    index = ConversationSearchIndex(
        index_dir=search_config.index_dir,
        model_path=search_config.model_path,
        tokenizer_path=search_config.tokenizer_path,
        log_dir=log_dir
    )

    for log_file in log_files:
        index.index_conversation_log(log_file)

    index.save()
    print(f"✅ Rebuilt search index: {index.get_entry_count()} entries from {len(log_files)} files")


if __name__ == "__main__":
    log_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None
    rebuild(log_dir_arg)

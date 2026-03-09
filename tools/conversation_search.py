"""
Conversation search tools for the voice assistant.
Provides semantic search over past conversation logs and
targeted retrieval of conversation context.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from core.tool_manager import Tool
from core.conversation_search import ConversationSearchIndex
from core.config import config

try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    dateparser = None


class SearchConversationLogsTool(Tool):
    """Tool to semantically search past conversation logs."""

    config_schema = {"enabled": True}

    def __init__(self, search_index: ConversationSearchIndex,
                 context_window: int = 1, top_k: int = 3,
                 min_score: float = 0.15):
        self._search_index = search_index
        self._context_window = context_window
        self._top_k = top_k
        self._min_score = min_score

    @property
    def name(self) -> str:
        return "search_conversation_logs"

    @property
    def description(self) -> str:
        return (
            "Search past conversations for specific topics, questions, or details "
            "discussed previously. Use this when the user asks about something you "
            "talked about before, or when you want to recall details from a past "
            "conversation. Returns matching exchanges with surrounding context. "
            "Optionally filter by time period using time_period_start and/or "
            "time_period_end. If the user wants to browse or discover what "
            "conversations happened in a time period without a specific topic, "
            "prefer list_conversations_in_time instead."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query describing what to find in past conversations"
                },
                "time_period_start": {
                    "type": "string",
                    "description": "Optional start of time range in natural language (e.g., 'yesterday', 'last week', '3 days ago', 'February 1st')"
                },
                "time_period_end": {
                    "type": "string",
                    "description": "Optional end of time range in natural language (e.g., 'today', 'yesterday', 'now')"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }

    def _parse_time_params(self, kwargs: Dict[str, Any]):
        """Parse optional time period parameters with dateparser.

        Returns:
            (start_iso, end_iso, error_dict) — error_dict is None on success.
        """
        time_start_str = kwargs.get('time_period_start')
        time_end_str = kwargs.get('time_period_end')

        if not time_start_str and not time_end_str:
            return None, None, None

        if not DATEPARSER_AVAILABLE:
            return None, None, {
                "error": "dateparser is not installed. "
                         "Time period filtering is unavailable."
            }

        now = datetime.now()
        settings = {'PREFER_DATES_FROM': 'past', 'RELATIVE_BASE': now}
        start_iso = None
        end_iso = None

        if time_start_str:
            parsed = dateparser.parse(time_start_str, settings=settings)
            if not parsed:
                return None, None, {
                    "error": f"Could not understand time_period_start: "
                             f"'{time_start_str}'"
                }
            start_iso = parsed.isoformat()

        if time_end_str:
            parsed = dateparser.parse(time_end_str, settings=settings)
            if not parsed:
                return None, None, {
                    "error": f"Could not understand time_period_end: "
                             f"'{time_end_str}'"
                }
            end_iso = parsed.isoformat()

        return start_iso, end_iso, None

    def execute(self, **kwargs) -> Dict[str, Any]:
        query = kwargs.get('query', '')
        if not query:
            return {"error": "No search query provided"}

        if self._search_index.get_entry_count() == 0:
            return {"message": "No conversation logs have been indexed yet."}

        # Parse optional time period parameters
        start_iso, end_iso, error = self._parse_time_params(kwargs)
        if error:
            return error

        if start_iso or end_iso:
            results = self._search_index.search_in_time_range(
                query, start_time=start_iso, end_time=end_iso,
                top_k=self._top_k, context_window=self._context_window,
                min_score=self._min_score
            )
        else:
            results = self._search_index.search(
                query, top_k=self._top_k, context_window=self._context_window,
                min_score=self._min_score
            )

        if not results:
            return {"message": "No matching conversations found."}

        matches = []
        for result in results:
            # Get surrounding context
            ctx = self._search_index.read_context_window(
                result.file, result.line, window=self._context_window
            )
            context_turns = ctx.get('context', [])
            total_turns = ctx.get('total_turns', 0)

            formatted_turns = []
            for turn in context_turns:
                marker = " >> " if turn['is_match'] else "    "
                formatted_turns.append(
                    f"{marker}User: {turn['user_input']}\n"
                    f"{marker}{config.assistant.name}: {turn['assistant_response']}"
                )

            # Line range being shown
            shown_lines = [turn['line'] for turn in context_turns]

            matches.append({
                "score": round(result.score, 3),
                "timestamp": result.timestamp,
                "file": result.file,
                "line": result.line,
                "total_turns_in_session": total_turns,
                "showing_lines": [min(shown_lines), max(shown_lines)] if shown_lines else [],
                "context": "\n".join(formatted_turns)
            })

        response = {
            "query": query,
            "total_indexed": self._search_index.get_entry_count(),
            "matches": matches
        }
        if start_iso:
            response["time_period_start"] = start_iso
        if end_iso:
            response["time_period_end"] = end_iso
        return response


class ReadConversationContextTool(Tool):
    """Tool to read a broader window of a past conversation."""

    config_schema = {"enabled": True}

    def __init__(self, search_index: ConversationSearchIndex):
        self._search_index = search_index

    @property
    def name(self) -> str:
        return "read_conversation_context"

    @property
    def description(self) -> str:
        return (
            "Read a range of turns from a past conversation log. Use this when "
            "you want to see more of a conversation."
            "Provide the filename from a search or listing result and the "
            "start/end line numbers of the range you want to read."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "The conversation log filename from a search result"
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line number to read (inclusive, 0-indexed)",
                    "minimum": 0
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line number to read (inclusive, 0-indexed)",
                    "minimum": 0
                }
            },
            "required": ["file", "start_line", "end_line"],
            "additionalProperties": False
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        filename = kwargs.get('file', '')
        start_line = kwargs.get('start_line', 0)
        end_line = kwargs.get('end_line', 0)

        if not filename:
            return {"error": "No filename provided"}

        if end_line < start_line:
            return {"error": "end_line must be >= start_line"}

        # Convert start/end to center/window for the underlying API
        center = (start_line + end_line) // 2
        window = max(center - start_line, end_line - center)

        ctx = self._search_index.read_context_window(
            filename, center, window=window
        )
        context_turns = ctx.get('context', [])
        total_turns = ctx.get('total_turns', 0)

        if not context_turns:
            return {"error": f"Could not read conversation from {filename}"}

        # Filter to the exact requested range
        context_turns = [
            t for t in context_turns
            if start_line <= t['line'] <= end_line
        ]

        formatted_turns = []
        for turn in context_turns:
            formatted_turns.append(
                f"    [{turn['timestamp'][:16]}] "
                f"User: {turn['user_input']}\n"
                f"    {config.assistant.name}: {turn['assistant_response']}"
            )

        shown_lines = [turn['line'] for turn in context_turns]

        return {
            "file": filename,
            "total_turns_in_session": total_turns,
            "showing_lines": [min(shown_lines), max(shown_lines)] if shown_lines else [],
            "context": "\n".join(formatted_turns)
        }


class ListConversationsInTimeTool(Tool):
    """Tool to list conversation sessions within a time period."""

    config_schema = {"enabled": True}

    def __init__(self, log_dir: str):
        self._log_dir = log_dir

    @property
    def name(self) -> str:
        return "list_conversations_in_time"

    @property
    def description(self) -> str:
        return (
            "List conversation sessions that occurred within a time period. "
            "Use this when the user wants to know what you talked about during "
            "a time window (e.g., 'what did we discuss last Thursday?', "
            "'show me our conversations from this week'). Returns session "
            "summaries with timestamps and turn counts. Use "
            "read_conversation_context to read the full content of any "
            "session that looks relevant."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "time_period_start": {
                    "type": "string",
                    "description": "Start of time range in natural language (e.g., 'yesterday', 'last Thursday', '3 days ago', 'February 1st')"
                },
                "time_period_end": {
                    "type": "string",
                    "description": "End of time range in natural language (e.g., 'today', 'now', 'yesterday evening')"
                }
            },
            "additionalProperties": False
        }

    def _read_session_info(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Read metadata from a conversation log file.

        Returns dict with filename, start_time, end_time, turn_count,
        and first_turn_preview, or None if the file is empty/unreadable.
        """
        try:
            entries = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))

            if not entries:
                return None

            first = entries[0]
            last = entries[-1]
            response = first.get(
                'assistant_response', first.get('kazka_response', ''))

            return {
                "filename": os.path.basename(filepath),
                "start_time": first.get('timestamp', ''),
                "end_time": last.get('timestamp', ''),
                "turn_count": len(entries),
                "first_turn_preview": {
                    "user": first.get('user_input', ''),
                    "assistant": response[:200] + ('...' if len(response) > 200 else '')
                }
            }
        except Exception:
            return None

    def execute(self, **kwargs) -> Dict[str, Any]:
        time_start_str = kwargs.get('time_period_start')
        time_end_str = kwargs.get('time_period_end')

        if not time_start_str and not time_end_str:
            return {
                "error": "At least one of time_period_start or "
                         "time_period_end is required."
            }

        if not DATEPARSER_AVAILABLE:
            return {
                "error": "dateparser is not installed. "
                         "Time period filtering is unavailable."
            }

        now = datetime.now()
        settings = {'PREFER_DATES_FROM': 'past', 'RELATIVE_BASE': now}
        start_iso = None
        end_iso = None

        if time_start_str:
            parsed = dateparser.parse(time_start_str, settings=settings)
            if not parsed:
                return {
                    "error": f"Could not understand time_period_start: "
                             f"'{time_start_str}'"
                }
            start_iso = parsed.isoformat()

        if time_end_str:
            parsed = dateparser.parse(time_end_str, settings=settings)
            if not parsed:
                return {
                    "error": f"Could not understand time_period_end: "
                             f"'{time_end_str}'"
                }
            end_iso = parsed.isoformat()

        # Scan log directory for .jsonl files
        if not os.path.isdir(self._log_dir):
            return {"error": f"Log directory not found: {self._log_dir}"}

        jsonl_files = sorted(
            f for f in os.listdir(self._log_dir)
            if f.endswith('.jsonl')
        )

        if not jsonl_files:
            return {"message": "No conversation log files found."}

        # Read session info for each file and filter by time range.
        # Also include the previous neighbor if its session extends
        # into the range.
        sessions = []
        prev_session = None

        for filename in jsonl_files:
            filepath = os.path.join(self._log_dir, filename)
            info = self._read_session_info(filepath)
            if info is None:
                continue

            session_start = info['start_time']
            session_end = info['end_time']

            # Check if session overlaps with the requested range
            in_range = True
            if start_iso and session_end and session_end < start_iso:
                # Session ended before range — remember as potential neighbor
                in_range = False
                prev_session = info
            elif end_iso and session_start and session_start > end_iso:
                # Session started after range — stop scanning
                break

            if in_range:
                # On the first in-range session, check if the previous
                # neighbor actually extends into the range
                if prev_session and not sessions:
                    if not start_iso or prev_session['end_time'] >= start_iso:
                        sessions.append(prev_session)
                    prev_session = None
                sessions.append(info)

        if not sessions:
            return {"message": "No conversation sessions found in the specified time period."}

        response = {
            "session_count": len(sessions),
            "sessions": sessions
        }
        if start_iso:
            response["time_period_start"] = start_iso
        if end_iso:
            response["time_period_end"] = end_iso
        return response

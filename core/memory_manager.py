"""
Memory Management module for persistent conversation context.
Handles memory extraction, ranking, summarization, and injection into system prompts.
"""

import json
import re
import uuid
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from core.config import config
from core.llm_interface import LLMInterface
from core.anthropic_llm_interface import AnthropicLLMInterface


@dataclass
class Memory:
    """Represents a single memory entry."""
    id: str
    content: str
    category: str  # 'user_fact', 'preference', 'event', 'project', 'conversation', 'self_fact'
    timestamp: str
    importance_score: Optional[float] = None
    access_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class MemoryManager:
    """
    Manages persistent memory for the voice assistant using LLM-based extraction and ranking.

    This class handles:
    - Extracting memories from conversations
    - Ranking memories by importance using LLM
    - Summarizing and compacting memories
    - Injecting relevant memories into system prompts
    """

    def __init__(self, memory_file: str = "assistant_memories.json"):
        """
        Initialize memory manager.

        Args:
            memory_file: Path to the memory storage file
        """
        self.memory_file = memory_file

        # Memory storage
        self.memories: List[Memory] = []  # Long-term stored memories
        self.conversation_log: List[Dict] = []  # Current session conversation pairs

        # Directories for conversation logs and memory backups (from config)
        # Set to None in settings to disable either independently
        self.log_dir = config.memory.conversation_log_dir
        self.backup_dir = config.memory.memory_backup_dir

        # Load prompts from files
        self.prompts = self._load_prompts()

        # Dedicated LLM client for background tasks, initialized with the
        # task system prompt so we never need to swap prompts at runtime.
        task_system_prompt = config.assistant.get_task_system_prompt()
        if config.network.api_type.lower() == "anthropic":
            self.llm = AnthropicLLMInterface(system_prompt=task_system_prompt)
        else:
            self.llm = LLMInterface(system_prompt=task_system_prompt)

    def _load_prompts(self) -> Dict[str, str]:
        """
        Load all prompts from files.

        Returns:
            Dictionary mapping prompt names to their content
        """
        prompts = {}


        prompt_files = {
            'memory_extraction': 'memory_extraction',
            'memory_ranking': 'memory_ranking',
            'conversation_summarization': 'conversation_summarization',
            'memory_summarization': 'memory_summarization'
        }

        for prompt_name, file_name in prompt_files.items():
            try:
                prompt_path = config.memory.get_prompt_file(file_name)
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompts[prompt_name] = f.read().strip()
                print(f"📝 Loaded prompt: {prompt_name}")
            except Exception as e:
                print(f"⚠️ Failed to load prompt {prompt_name}: {e}")
                raise

        return prompts

    def _remove_thinking_blocks(self, text: str) -> str:
        """
        Remove thinking blocks from LLM responses.

        Some models (especially in thinking mode) include <thinking>...</thinking>
        blocks in their responses. This removes them to clean up the output.

        Args:
            text: Response text that may contain thinking blocks

        Returns:
            Cleaned text with thinking blocks removed
        """
        if not text:
            return text

        # Remove <thinking>...</thinking> blocks (with or without newlines)
        pattern = r'<thinking>.*?</thinking>'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL)

        # Clean up any extra whitespace
        cleaned = cleaned.strip()

        return cleaned

    def load_memories(self):
        """Load existing memories from file."""
        try:
            if hasattr(self, 'memory_file'):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memories = [Memory(**mem_data) for mem_data in data.get('memories', [])]
                print(f"🧠 Loaded {len(self.memories)} existing memories")
            else:
                print("⚠️ No memory file path set, starting with empty memory")
                self.memories = []
        except FileNotFoundError:
            print("📝 No existing memories found. Starting fresh!")
            self.memories = []
        except Exception as e:
            print(f"❌ Error loading memories: {e}")
            self.memories = []

    def log_conversation(self, user_input: str, assistant_response: str):
        """
        Log a conversation exchange for batch processing at shutdown.

        Args:
            user_input: User's transcribed speech
            assistant_response: Assistant's response
        """
        self.conversation_log.append({
            "user_input": user_input,
            "assistant_response": assistant_response,
            "timestamp": datetime.now().isoformat()
        })

    def _extract_from_conversation_log(self) -> List[Dict]:
        """
        Extract memories from the entire conversation log at shutdown.

        Returns:
            List of extracted memory items
        """
        if not self.conversation_log:
            return []



        # Format conversation for LLM analysis
        conversation_lines = []
        for entry in self.conversation_log:
            conversation_lines.append(f"User: {entry['user_input']}")
            conversation_lines.append(f"{config.assistant.name}: {entry.get('assistant_response', entry.get('kazka_response', ''))}")
            conversation_lines.append("")  # Empty line for readability

        conversation_text = "\n".join(conversation_lines)

        print(f"🧠 Analyzing conversation with {len(self.conversation_log)} exchanges and {len(self.memories)} existing memories...")

        # Format existing memories for the prompt
        existing_memories_text = ""
        if self.memories:
            existing_memories_text = "\n".join([
                f"- {m.content} ({m.category})"
                for m in self.memories
            ])

        # Use safe string replacement to avoid format placeholder issues
        prompt = self.prompts['memory_extraction'].replace("{existing_memories}", existing_memories_text).replace("{conversation_text}", conversation_text)

        # Debug: Print the full request being sent
        print(f"\n📤 MEMORY EXTRACTION REQUEST TO LLM:")
        print("=" * 60)
        print(f"Conversation exchanges: {len(self.conversation_log)}")
        print(f"Existing memories: {len(self.memories)}")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"\n--- PROMPT CONTENT ---")
        print(prompt)
        print("=" * 60)

        # Use chat completions API to extract memories
        response = self.llm.query_chat_for_task(prompt)
        print(f"\n🔍 LLM batch extraction response: '{response[:200] if response else 'EMPTY'}...'")

        # Handle empty response
        if not response or not response.strip():
            print("⚠️ Empty response from LLM during memory extraction")
            return []

        # Parse the JSON response
        try:
            # Clean up the response - sometimes LLMs add extra text or markdown
            if "```json" in response:
                # Extract JSON from code block
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                # Extract JSON from code block without language specifier
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
            else:
                # Extract JSON from response - look for the first { and last }
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end > start:
                    response = response[start:end]

            response = response.strip()

            print(f"🔍 Complete JSON to parse: {response}")

            result = json.loads(response)
            return result.get('extracted', [])

        except json.JSONDecodeError as e:
            print(f"⚠️ Memory extraction JSON parsing failed: {e}")
            print(f"Full LLM response was: '{response[:500]}'")
            return []

    
    def process_and_save(self):
        """
        Extract memories from conversation log and save to file.
        Creates conversation summaries and manages memory limits.
        Called during shutdown.
        """
        if not self.conversation_log and not self.memories:
            print("🧠 No conversation or memories to process")
            return

        try:
            # Backup the current memory file BEFORE making any modifications
            self._backup_memories_to_log()

            # Extract memories from conversation log
            extracted_items = self._extract_from_conversation_log()

            # Convert extracted items to Memory objects
            new_memories = []
            for item in extracted_items:
                memory = Memory(
                    id="",  # Will be generated in __post_init__
                    content=item.get('content', ''),
                    category=item.get('category', 'conversation'),
                    timestamp=datetime.now().isoformat()
                )
                new_memories.append(memory)
                print(f"🧠 New memory extracted: {memory.category} - {memory.content}")

            # Create a conversation summary if there were any exchanges
            if self.conversation_log:
                print("📝 Creating conversation summary...")
                conversation_summary = self._create_conversation_summary()
                if conversation_summary:
                    summary_memory = Memory(
                        id="",
                        content=conversation_summary,
                        category="conversation",
                        timestamp=datetime.now().isoformat()
                    )
                    new_memories.append(summary_memory)
                    print(f"📝 Conversation summary created: {conversation_summary}")

            if not new_memories and not self.memories:
                print("🧠 No new memories extracted from conversation")
                return

            # Add new memories to existing ones
            self.memories.extend(new_memories)

            # Apply category-based memory limits with intelligent ranking
            self._manage_memory_limits()

            # Save to file
            self._save_to_file()

            print(f"✅ Processed and saved {len(self.memories)} memories")

            # Clear conversation log for next session
            self.conversation_log = []

        except Exception as e:
            print(f"❌ Error processing memories: {e}")

    def _create_conversation_summary(self) -> str:
        """
        Create a concise summary of the current conversation log.

        Returns:
            Concise bullet-point summary of the conversation
        """
        try:
            # Format conversation for summarization
            conversation_lines = []
            for entry in self.conversation_log:
                conversation_lines.append(f"User: {entry['user_input']}")
                conversation_lines.append(f"{config.assistant.name}: {entry.get('assistant_response', entry.get('kazka_response', ''))}")
                conversation_lines.append("")  # Empty line for readability

            conversation_text = "\n".join(conversation_lines)

            prompt = self.prompts['conversation_summarization'].replace("{conversation_text}", conversation_text)
            response = self.llm.query_chat_for_task(prompt)

            # Remove thinking blocks if present (some models include them in responses)
            response = self._remove_thinking_blocks(response)

            return response

        except Exception as e:
            print(f"❌ Conversation summarization error: {e}")
            return ""

    def _manage_memory_limits(self):
        """Manage memory limits by category, only summarizing when necessary."""


        # Group memories by category
        categorized_memories = {
            'user_fact': [],
            'self_fact': [],
            'preference': [],
            'event': [],
            'project': [],
            'conversation': []
        }

        for memory in self.memories:
            if memory.category in categorized_memories:
                categorized_memories[memory.category].append(memory)

        # Sort each category by timestamp (newest first)
        for category in categorized_memories:
            categorized_memories[category].sort(key=lambda m: m.timestamp, reverse=True)

        # Get limits for each category
        limits = {
            'user_fact': config.memory.max_user_facts,
            'self_fact': config.memory.max_self_facts,
            'preference': config.memory.max_preferences,
            'event': config.memory.max_events,
            'project': config.memory.max_projects,
            'conversation': config.memory.max_conversation
        }

        # Keep memories within limits for each category
        final_memories = []
        for category, category_memories in categorized_memories.items():
            limit = limits[category]

            if len(category_memories) <= limit:
                # Keep all memories in this category
                final_memories.extend(category_memories)
            else:
                # Use LLM ranking to determine which memories to keep for ALL categories
                print(f"🧠 Ranking {len(category_memories)} {category} memories to keep {limit} most important...")

                # Retry loop: ask LLM up to N times and validate that returned codes are
                # members of the set of available short codes (presence validation).
                available_codes = {f"m{i+1}" for i in range(len(category_memories))}
                max_retries = 2
                attempt = 0
                selected_ids = []

                while attempt <= max_retries:
                    ranking_result = self._rank_memories_by_importance(category_memories, limit)
                    candidate_codes = ranking_result.get('ranked', []) if isinstance(ranking_result, dict) else []
                    code_to_id = ranking_result.get('code_to_id', {})

                    # Presence validation: accept only codes that are strings and present in available_codes
                    valid_codes = [code for code in candidate_codes if isinstance(code, str) and code in available_codes]

                    # Require the LLM to return every code we provided (i.e., it must have considered all entries).
                    # Accept only when the set of returned valid codes exactly matches the available codes.
                    if set(valid_codes) == available_codes:
                        # The LLM returned a full ordering that includes all items; take the top `limit`.
                        selected_codes = valid_codes[:min(limit, len(category_memories))]
                        # Convert codes back to actual UUIDs
                        selected_ids = [code_to_id[code] for code in selected_codes if code in code_to_id]
                        break

                    # Otherwise, log and retry
                    attempt += 1
                    print(f"⚠️ Ranking attempt {attempt} returned {len(valid_codes)} valid codes (needed {len(available_codes)}). Retrying...")

                # If retries exhausted and we still don't have enough valid ids, do NOT modify
                # the memories for this category — keep them as-is to avoid accidental deletion.
                if not selected_ids or len(selected_ids) < min(limit, len(category_memories)):
                    print(f"⚠️ Ranking failed after {max_retries+1} attempts for category '{category}'. Keeping all {len(category_memories)} memories for this category unchanged.")
                    final_memories.extend(category_memories)
                    continue

                # Build selected memories preserving order of category_memories
                selected_set = set(selected_ids)
                selected_memories = [m for m in category_memories if m.id in selected_set]

                # Identify memories being forgotten (culled)
                forgotten_memories = [m for m in category_memories if m.id not in selected_set]

                # Print what's being forgotten
                if forgotten_memories:
                    print(f"🗑️  Forgetting {len(forgotten_memories)} {category} memories:")
                    for memory in forgotten_memories:
                        print(f"      • {memory.content}")

                final_memories.extend(selected_memories)
                print(f"✅ Kept {len(selected_memories)} most important {category} memories")

        self.memories = final_memories

    def _rank_memories_by_importance(self, memories: List[Memory], limit: int) -> Dict[str, List[str]]:
        """
        Use LLM to rank memories by importance and select which ones to keep.

        Uses short codes (m1, m2, etc.) instead of full UUIDs for easier LLM regurgitation.

        Args:
            memories: List of memories to rank
            limit: Maximum number of memories to keep

        Returns:
            Dictionary with 'ranked' key containing list of short codes (e.g., ['m1', 'm2'])
            and 'code_to_id' mapping short codes to actual UUIDs
        """
        try:
            # Create short codes for memories (m1, m2, m3, etc.)
            code_to_id = {}
            memories_with_codes = []
            for idx, memory in enumerate(memories, start=1):
                code = f"m{idx}"
                code_to_id[code] = memory.id
                memories_with_codes.append((code, memory))

            # Format memories for ranking using short codes
            memories_text = "\n".join([
                f"{code}: {m.content}"
                for code, m in memories_with_codes
            ])

            ranking_prompt = self.prompts['memory_ranking'] \
                .replace("{limit}", str(limit)) \
                .replace("{memories_text}", memories_text) \
                .replace("{memories_count}", str(len(memories)))

            response = self.llm.query_chat_for_task(ranking_prompt)

            # Parse the JSON response
            try:
                # Clean up the response - sometimes LLMs add extra text or markdown
                if "```json" in response:
                    # Extract JSON from code block
                    start = response.find("```json") + 7
                    end = response.find("```", start)
                    response = response[start:end].strip()
                elif "```" in response:
                    # Extract JSON from code block without language specifier
                    start = response.find("```") + 3
                    end = response.find("```", start)
                    response = response[start:end].strip()
                else:
                    # Extract JSON from response - look for the first { and last }
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start != -1 and end > start:
                        response = response[start:end]

                response = response.strip()

                print(f"🔍 Ranking JSON to parse: {response}")

                result = json.loads(response)
                ranked_codes = result.get('ranked', [])

                return {"ranked": ranked_codes, "code_to_id": code_to_id}

            except json.JSONDecodeError as e:
                print(f"⚠️ Memory ranking JSON parsing failed: {e}")
                print(f"⚠️ Response was: {response[:200]}")
                # Return empty selection to let caller decide (caller will keep originals)
                return {"ranked": [], "code_to_id": code_to_id}

        except Exception as e:
            print(f"❌ Memory ranking error: {e}")
            # Return empty selection to let caller decide (caller will keep originals)
            return {"ranked": [], "code_to_id": {}}

    def _summarize_memories(self, memories: List[Memory]) -> str:
        """Summarize a batch of memories using LLM."""
        try:
            memories_text = "\n".join([
                f"{m.category}: {m.content}"
                for m in memories
            ])

            prompt = self.prompts['memory_summarization'].replace("{memories_text}", memories_text)
            response = self.llm.query_chat_for_task(prompt)

            # Remove thinking blocks if present
            response = self._remove_thinking_blocks(response)

            return response

        except Exception as e:
            print(f"❌ Memory summarization error: {e}")
            return ""

    def save_conversation_log(self) -> Optional[str]:
        """
        Save the current conversation log to a .jsonl file in the log directory.
        Each line is a JSON object with user_input, assistant_response, and timestamp.
        Called during shutdown and sleep cycles to preserve raw conversation history.
        Disabled when conversation_log_dir is set to null in settings.

        Returns:
            Path to the saved log file, or None if nothing was saved.
        """
        if self.log_dir is None:
            return None

        if not self.conversation_log:
            print("📝 No conversation exchanges to save")
            return None

        try:
            os.makedirs(self.log_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(self.log_dir, f"conversation_{timestamp}.jsonl")

            with open(log_path, 'w', encoding='utf-8') as f:
                for entry in self.conversation_log:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"💬 Saved conversation log ({len(self.conversation_log)} exchanges) to {log_path}")
            return log_path

        except Exception as e:
            print(f"⚠️ Failed to save conversation log: {e}")
            return None

    def get_resume_history(self, count: int) -> str:
        """
        Load the last N exchanges from the most recent conversation log.

        Args:
            count: Number of exchanges to load

        Returns:
            Formatted string of previous conversation, or "" if unavailable
        """
        if self.log_dir is None or count <= 0:
            return ""

        try:
            import glob
            log_files = sorted(glob.glob(os.path.join(self.log_dir, "conversation_*.jsonl")))
            if not log_files:
                return ""

            # Read entries from the most recent log file
            with open(log_files[-1], 'r', encoding='utf-8') as f:
                entries = [json.loads(line) for line in f if line.strip()]

            if not entries:
                return ""

            # Take the last N entries
            recent = entries[-count:]

            from core.config import config
            name = config.assistant.name
            log_filename = os.path.basename(log_files[-1])
            total_exchanges = len(entries)

            lines = [f"=== PREVIOUS CONVERSATION (from {log_filename}, showing last {len(recent)} of {total_exchanges} exchanges) ==="]
            for entry in recent:
                user = entry.get('user_input', '')
                response = entry.get('assistant_response', entry.get('kazka_response', ''))
                lines.append(f"User: {user}")
                lines.append(f"{name}: {response}")
            lines.append("=== END PREVIOUS CONVERSATION ===")
            lines.append("")  # trailing newline

            result = "\n".join(lines) + "\n"
            print(f"📜 Loaded {len(recent)} previous exchanges from {log_filename} for resume history")
            return result

        except Exception as e:
            print(f"⚠️  Failed to load resume history: {e}")
            return ""

    def _backup_memories_to_log(self):
        """
        Backup the current memory file to the backup directory before updating.
        Creates a timestamped backup file.
        Disabled when memory_backup_dir is set to null in settings.
        """
        if self.backup_dir is None:
            return

        try:
            # Check if the memory file exists
            if not os.path.exists(self.memory_file):
                return  # No existing file to backup

            # Ensure backup directory exists
            os.makedirs(self.backup_dir, exist_ok=True)

            # Create timestamp for backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"assistant_memories_backup_{timestamp}.json"
            backup_path = os.path.join(self.backup_dir, backup_filename)

            # Copy the current memory file to backup directory
            shutil.copy2(self.memory_file, backup_path)

            print(f"💾 Backed up memories to {backup_path}")

        except Exception as e:
            print(f"⚠️ Warning: Failed to backup memories: {e}")

    def _save_to_file(self):
        """Save memories to JSON file."""
        try:
            data = {
                'memories': [asdict(memory) for memory in self.memories],
                'last_updated': datetime.now().isoformat()
            }

            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Error saving memories: {e}")

    def get_recent_context(self, limit: Optional[int] = None) -> List[Memory]:
        """Get memories for context injection using category-based limits."""


        # Group memories by category
        categorized_memories = {
            'user_fact': [],
            'self_fact': [],
            'preference': [],
            'event': [],
            'project': [],
            'conversation': []
        }

        # Sort all memories by timestamp (newest first)
        sorted_memories = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)

        # Group memories by category
        for memory in sorted_memories:
            if memory.category in categorized_memories:
                categorized_memories[memory.category].append(memory)

        # Apply category-specific limits in priority order
        selected_memories = []

        # Add all user facts (names, personal info) - these are most important
        selected_memories.extend(categorized_memories['user_fact'][:config.memory.max_user_facts])

        # Add assistant's self-facts (personal memories) - second priority for personality development
        selected_memories.extend(categorized_memories['self_fact'][:config.memory.max_self_facts])

        # Add preferences
        selected_memories.extend(categorized_memories['preference'][:config.memory.max_preferences])

        # Add events
        selected_memories.extend(categorized_memories['event'][:config.memory.max_events])

        # Add projects
        selected_memories.extend(categorized_memories['project'][:config.memory.max_projects])

        # Add conversation summaries (keep minimal)
        selected_memories.extend(categorized_memories['conversation'][:config.memory.max_conversation])

        # If a specific limit is provided (legacy support), truncate the final list
        if limit is not None:
            selected_memories = selected_memories[:limit]

        return selected_memories

    def _get_relative_time(self, timestamp: str) -> str:
        """
        Calculate a human-readable relative time string.

        Args:
            timestamp: ISO format timestamp string

        Returns:
            Human-readable relative time (e.g., "today", "2 days ago", "last week")
        """
        try:
            # Parse the timestamp
            memory_time = datetime.fromisoformat(timestamp)
            now = datetime.now(memory_time.tzinfo)  # Preserve timezone if present

            # Calculate the difference
            delta = now - memory_time

            # Handle future timestamps (shouldn't happen, but handle gracefully)
            if delta.total_seconds() < 0:
                return ""

            # Calculate days
            days = delta.days

            # Return appropriate string
            if days == 0:
                return ""  # Today, don't add label
            elif days == 1:
                return "(yesterday)"
            elif days < 7:
                return f"({days} days ago)"
            elif days < 14:
                return "(last week)"
            elif days < 30:
                weeks = days // 7
                return f"({weeks} week{'s' if weeks > 1 else ''} ago)"
            elif days < 60:
                return "(last month)"
            else:
                months = days // 30
                return f"({months} month{'s' if months > 1 else ''} ago)"

        except Exception as e:
            # If we can't parse the timestamp, return empty string
            print(f"⚠️ Failed to parse timestamp {timestamp}: {e}")
            return ""

    def inject_into_prompt(self, system_prompt: str, memories: List[Memory]) -> str:
        """
        Inject relevant memories into system prompt with organized sections.

        Args:
            system_prompt: Base system prompt
            memories: List of memories to inject

        Returns:
            Enhanced system prompt with organized memory context
        """

        if not memories:
            return system_prompt.replace("{memory_bank}", "")

        # Group memories by category
        categorized = {
            'user_fact': [],
            'self_fact': [],
            'preference': [],
            'event': [],
            'project': [],
            'conversation': []
        }

        for memory in memories:
            if memory.category in categorized:
                categorized[memory.category].append(memory)

        # Build organized memory context
        memory_context = "\n=== MEMORY BANK ===\n"
        memory_context += "Use these memories from previous conversations to provide personalized, informed responses that demonstrate continuity and awareness of shared history."

        # Section: About the user (most important for personalization)
        if categorized['user_fact']:
            memory_context += "\nABOUT THE USER:\n"
            for memory in categorized['user_fact']:
                memory_context += f"  - {memory.content}\n"

        # Section: User preferences
        if categorized['preference']:
            memory_context += "\nUSER PREFERENCES:\n"
            for memory in categorized['preference']:
                memory_context += f"  - {memory.content}\n"

        # Section: Assistant's personality and experiences
        if categorized['self_fact']:
            memory_context += f"\nABOUT {config.assistant.name.upper()} (YOUR PERSONALITY & EXPERIENCES):\n"
            for memory in categorized['self_fact']:
                memory_context += f"  - {memory.content}\n"

        # Section: Active projects and tasks
        if categorized['project']:
            memory_context += "\nACTIVE PROJECTS & TASKS:\n"
            for memory in categorized['project']:
                memory_context += f"  - {memory.content}\n"

        # Section: Important events
        if categorized['event']:
            memory_context += "\nIMPORTANT EVENTS:\n"
            for memory in categorized['event']:
                memory_context += f"  - {memory.content}\n"

        # Section: Conversation history (chronological, most recent first)
        if categorized['conversation']:
            memory_context += "\nCONVERSATION HISTORY (most recent first):\n"
            # Sort by timestamp descending (newest first)
            sorted_conversations = sorted(
                categorized['conversation'],
                key=lambda m: m.timestamp,
                reverse=True
            )
            for i, memory in enumerate(sorted_conversations):
                # Add relative time context as a header for older conversations
                relative_time = self._get_relative_time(memory.timestamp)
                if relative_time:
                    memory_context += f"{relative_time}:\n"
                else:
                    memory_context += "Conversation:\n"
                for line in memory.content.splitlines():
                    if line:
                        memory_context += f"  {line}\n"

                # Add blank line between conversations (but not after the last one)
                if i < len(sorted_conversations) - 1:
                    memory_context += "\n"

        memory_context += "\n=== END MEMORY BANK ===\n"

        if "{memory_bank}" in system_prompt:
            return system_prompt.replace("{memory_bank}", memory_context)
        return system_prompt + memory_context

    def has_memories(self) -> bool:
        """Check if there are any stored memories."""
        return len(self.memories) > 0

    def get_memory_count(self) -> int:
        """Get total number of stored memories."""
        return len(self.memories)

    def print_memories(self):
        """Print all stored memories (for debugging)."""
        if not self.memories:
            print("📝 No memories stored")
            return


        print(f"\n🧠 {config.assistant.name}'s Memory Bank ({len(self.memories)} memories):")
        print("=" * 60)

        # Group by category
        by_category = {}
        for memory in self.memories:
            if memory.category not in by_category:
                by_category[memory.category] = []
            by_category[memory.category].append(memory)

        for category, category_memories in by_category.items():
            print(f"\n{category.upper()} ({len(category_memories)}):")
            for memory in category_memories:
                print(f"  • {memory.content}")
                print(f"    (Stored: {memory.timestamp[:10]})")

    def print_memory_injection_breakdown(self):
        """Print what memories would be injected using category-based limits."""


        if not self.memories:
            print("📝 No memories to analyze")
            return

        print("\n💉 Memory Injection Analysis:")
        print("=" * 50)

        # Get the memories that would be injected
        injected_memories = self.get_recent_context()

        # Group all memories by category
        categorized_memories = {
            'user_fact': [],
            'self_fact': [],
            'preference': [],
            'event': [],
            'project': [],
            'conversation': []
        }

        for memory in self.memories:
            if memory.category in categorized_memories:
                categorized_memories[memory.category].append(memory)

        # Show limits vs actual counts
        print(f"User Facts: {len(categorized_memories['user_fact'])} available, "
              f"{min(len(categorized_memories['user_fact']), config.memory.max_user_facts)} injected "
              f"(limit: {config.memory.max_user_facts})")

        print(f"{config.assistant.name}'s Self-Facts: {len(categorized_memories['self_fact'])} available, "
              f"{min(len(categorized_memories['self_fact']), config.memory.max_self_facts)} injected "
              f"(limit: {config.memory.max_self_facts})")

        print(f"Preferences: {len(categorized_memories['preference'])} available, "
              f"{min(len(categorized_memories['preference']), config.memory.max_preferences)} injected "
              f"(limit: {config.memory.max_preferences})")

        print(f"Events: {len(categorized_memories['event'])} available, "
              f"{min(len(categorized_memories['event']), config.memory.max_events)} injected "
              f"(limit: {config.memory.max_events})")

        print(f"Projects: {len(categorized_memories['project'])} available, "
              f"{min(len(categorized_memories['project']), config.memory.max_projects)} injected "
              f"(limit: {config.memory.max_projects})")

        print(f"Conversations: {len(categorized_memories['conversation'])} available, "
              f"{min(len(categorized_memories['conversation']), config.memory.max_conversation)} injected "
              f"(limit: {config.memory.max_conversation})")

        print(f"\nTotal memories: {len(self.memories)}")
        print(f"Total injected: {len(injected_memories)}")

        print(f"\n📋 Injected memories:")
        for memory in injected_memories:
            print(f"  [{memory.category}] {memory.content}")

    def clear_all_memories(self):
        """Clear all stored memories and conversation log."""
        self.memories = []
        self.conversation_log = []
        try:
            if hasattr(self, 'memory_file'):
                if os.path.exists(self.memory_file):
                    os.remove(self.memory_file)
            print("🗑️ All memories and conversation log cleared")
        except Exception as e:
            print(f"❌ Error clearing memory file: {e}")

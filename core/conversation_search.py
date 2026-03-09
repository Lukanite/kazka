"""
Conversation search index using ONNX embeddings.
Provides semantic search over conversation logs stored as .jsonl files.
"""

import json
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SearchResult:
    """A single search result with source location and context."""
    file: str
    line: int
    score: float
    timestamp: str
    user_input: str
    assistant_response: str


class ConversationSearchIndex:
    """
    Semantic search index over conversation logs.

    Uses an ONNX sentence embedding model to embed conversation turns
    and stores vectors in a numpy archive for fast cosine similarity search.

    The index stores only vectors and metadata pointers (file, line, timestamp).
    Full conversation text is read from the .jsonl files at query time.
    """

    def __init__(self, index_dir: str, model_path: str, tokenizer_path: str,
                 log_dir: str):
        """
        Initialize the search index.

        Args:
            index_dir: Directory to store the index file
            model_path: Path to the ONNX embedding model
            tokenizer_path: Path to the tokenizer.json file
            log_dir: Directory containing conversation .jsonl files
        """
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "conversation_index.npz")
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.log_dir = log_dir

        # Lazily loaded on first use
        self._session: Optional[ort.InferenceSession] = None
        self._tokenizer: Optional[Tokenizer] = None

        # Index data (loaded from disk or built fresh)
        self.vectors: Optional[np.ndarray] = None  # (N, 384) float32
        self.files: List[str] = []
        self.lines: List[int] = []
        self.timestamps: List[str] = []

    def _load_model(self):
        """Lazily load the ONNX model and tokenizer, downloading if necessary."""
        if self._session is not None:
            return

        # Download model if it doesn't exist
        if not os.path.exists(self.model_path):
            print(f"📥 Downloading conversation search model to {self.model_path}...")
            # Ensure the directory exists
            model_dir = os.path.dirname(self.model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                print(f"📁 Created model directory: {model_dir}")

            try:
                import urllib.request
                # Download from Hugging Face (all-MiniLM-L6-v2 ONNX)
                # Using the correct ONNX model path from sentence-transformers
                model_url = "https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx/resolve/main/model.onnx"
                urllib.request.urlretrieve(model_url, self.model_path)
                print(f"✅ Conversation search model downloaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to download conversation search model: {e}")

        # Download tokenizer if it doesn't exist
        if not os.path.exists(self.tokenizer_path):
            print(f"📥 Downloading tokenizer to {self.tokenizer_path}...")
            try:
                import urllib.request
                # Download from Hugging Face
                tokenizer_url = "https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx/resolve/main/tokenizer.json"
                urllib.request.urlretrieve(tokenizer_url, self.tokenizer_path)
                print(f"✅ Tokenizer downloaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to download tokenizer: {e}")

        try:
            self._session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            self._tokenizer = Tokenizer.from_file(self.tokenizer_path)
            self._tokenizer.no_padding()
            print(f"🔍 Loaded embedding model: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model from {self.model_path}: {e}")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts using the ONNX model.

        Args:
            texts: List of strings to embed

        Returns:
            numpy array of shape (len(texts), 384)
        """
        self._load_model()

        embeddings = []
        for text in texts:
            enc = self._tokenizer.encode(text)
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            type_ids = np.array([[0] * len(enc.ids)], dtype=np.int64)

            outputs = self._session.run(None, {
                'input_ids': ids,
                'attention_mask': mask,
                'token_type_ids': type_ids
            })

            # sentence_embedding is already L2-normalized
            embeddings.append(outputs[1][0])

        return np.array(embeddings, dtype=np.float32)

    def _load_from_file(self, path: str) -> bool:
        """
        Load index data from a specific .npz file.

        Args:
            path: Path to the .npz file

        Returns:
            True if loaded successfully
        """
        data = np.load(path, allow_pickle=True)
        self.vectors = data['vectors']
        self.files = data['files'].tolist()
        self.lines = data['lines'].tolist()
        self.timestamps = data['timestamps'].tolist()
        return True

    def load(self) -> bool:
        """
        Load existing index from disk. Falls back to backup if the
        primary index is corrupted.

        Returns:
            True if index was loaded, False if no index exists
        """
        if not os.path.exists(self.index_path):
            return False

        try:
            self._load_from_file(self.index_path)
            print(f"🔍 Loaded search index: {len(self.files)} entries")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load search index: {e}")

            # Try loading from backup
            backup_path = os.path.join(
                self.index_dir, "conversation_index_backup.npz"
            )
            if os.path.exists(backup_path):
                try:
                    self._load_from_file(backup_path)
                    print(f"⚠️ Recovered search index from backup: {len(self.files)} entries")
                    return True
                except Exception as backup_e:
                    print(f"⚠️ Backup index also failed to load: {backup_e}")

            return False

    def save(self):
        """Save the current index to disk, backing up the existing index first."""
        if self.vectors is None or len(self.files) == 0:
            return

        try:
            os.makedirs(self.index_dir, exist_ok=True)

            # Backup existing index before overwriting
            if os.path.exists(self.index_path):
                backup_path = os.path.join(
                    self.index_dir, "conversation_index_backup.npz"
                )
                import shutil
                shutil.copy2(self.index_path, backup_path)

            np.savez(
                self.index_path,
                vectors=self.vectors,
                files=np.array(self.files),
                lines=np.array(self.lines),
                timestamps=np.array(self.timestamps)
            )
            print(f"🔍 Saved search index: {len(self.files)} entries")
        except Exception as e:
            print(f"⚠️ Failed to save search index: {e}")

    def index_conversation_log(self, log_path: str):
        """
        Add a conversation log file to the index.

        Reads each line from the .jsonl file, embeds the combined
        user+response text, and appends to the index.

        Args:
            log_path: Path to the .jsonl conversation log file
        """
        filename = os.path.basename(log_path)

        try:
            entries = []
            with open(log_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    entries.append((line_num, entry))

            if not entries:
                return

            # Build text for embedding: combined user + response
            texts = []
            new_files = []
            new_lines = []
            new_timestamps = []

            for line_num, entry in entries:
                response = entry.get('assistant_response', '')
                text = f"User: {entry['user_input']} | Assistant: {response}"
                texts.append(text)
                new_files.append(filename)
                new_lines.append(line_num)
                new_timestamps.append(entry.get('timestamp', ''))

            # Embed all entries
            new_vectors = self._embed(texts)

            # Append to existing index
            if self.vectors is not None and len(self.files) > 0:
                self.vectors = np.vstack([self.vectors, new_vectors])
            else:
                self.vectors = new_vectors

            self.files.extend(new_files)
            self.lines.extend(new_lines)
            self.timestamps.extend(new_timestamps)

            print(f"🔍 Indexed {len(entries)} entries from {filename}")

        except Exception as e:
            print(f"⚠️ Failed to index {log_path}: {e}")

    def search(self, query: str, top_k: int = 3,
               context_window: int = 1,
               min_score: float = 0.15) -> List[SearchResult]:
        """
        Search the index for conversation turns matching the query.

        Args:
            query: Natural language search query
            top_k: Number of distinct results to return
            context_window: Context window size (used for dedup radius)
            min_score: Minimum cosine similarity to include a result

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        if self.vectors is None or len(self.files) == 0:
            return []

        query_vector = self._embed([query])[0]
        scores = self.vectors @ query_vector
        return self._rank_and_read(scores, top_k, context_window, min_score)

    def search_in_time_range(self, query: str,
                             start_time: Optional[str] = None,
                             end_time: Optional[str] = None,
                             top_k: int = 3,
                             context_window: int = 1,
                             min_score: float = 0.15) -> List[SearchResult]:
        """
        Search with a timestamp filter. Same as search() but masks out
        entries outside [start_time, end_time] before ranking.

        Args:
            query: Natural language search query
            start_time: ISO 8601 lower bound (inclusive), or None for open start
            end_time: ISO 8601 upper bound (inclusive), or None for open end
            top_k: Number of results
            context_window: Dedup radius
            min_score: Minimum cosine similarity

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        if self.vectors is None or len(self.files) == 0:
            return []

        query_vector = self._embed([query])[0]
        scores = self.vectors @ query_vector

        # Mask entries outside the time range
        for i, ts in enumerate(self.timestamps):
            if not ts:
                scores[i] = -1
                continue
            if start_time and ts < start_time:
                scores[i] = -1
            elif end_time and ts > end_time:
                scores[i] = -1

        return self._rank_and_read(scores, top_k, context_window, min_score)

    def _rank_and_read(self, scores: np.ndarray, top_k: int,
                       context_window: int,
                       min_score: float) -> List[SearchResult]:
        """
        Rank scored entries, deduplicate nearby hits, and read full
        conversation data from the .jsonl files.

        Args:
            scores: Pre-computed similarity scores array (one per indexed entry)
            top_k: Number of distinct results to return
            context_window: Context window size (used for dedup radius)
            min_score: Minimum score to include a result

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        # Over-fetch candidates to account for dedup filtering
        candidate_count = min(top_k * 3, len(scores))
        candidate_indices = np.argsort(scores)[::-1][:candidate_count]

        # Deduplicate: skip candidates that overlap with already-accepted results
        results = []
        accepted: List[Tuple[str, int]] = []  # (file, line) of accepted results

        for idx in candidate_indices:
            if len(results) >= top_k:
                break

            if float(scores[idx]) < min_score:
                break  # Sorted by score, so all remaining are below threshold

            file = self.files[idx]
            line = self.lines[idx]

            # Check if this overlaps with any accepted result's context window
            is_duplicate = False
            for acc_file, acc_line in accepted:
                if file == acc_file and abs(line - acc_line) <= context_window * 2:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # Read the actual conversation entry from the .jsonl file
            entry = self._read_entry(file, line)
            if entry is None:
                continue

            results.append(SearchResult(
                file=file,
                line=line,
                score=float(scores[idx]),
                timestamp=self.timestamps[idx],
                user_input=entry.get('user_input', ''),
                assistant_response=entry.get('assistant_response', '')
            ))
            accepted.append((file, line))

        return results

    def read_context_window(self, filename: str, line: int,
                            window: int = 2) -> Dict:
        """
        Read a window of conversation turns around a specific line.

        Args:
            filename: Name of the .jsonl file
            line: Center line number (0-indexed)
            window: Number of turns to include on each side

        Returns:
            Dict with 'context' (list of turn dicts with 'line',
            'user_input', 'assistant_response', 'timestamp', 'is_match'),
            and 'total_turns' (int) for the entire session.
            Returns empty dict on failure.
        """
        log_path = os.path.join(self.log_dir, filename)
        if not os.path.exists(log_path):
            return {}

        try:
            all_entries = []
            with open(log_path, 'r', encoding='utf-8') as f:
                for line_num, raw_line in enumerate(f):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    all_entries.append((line_num, json.loads(raw_line)))

            # Find the range
            start = max(0, line - window)
            end = min(len(all_entries), line + window + 1)

            context = []
            for line_num, entry in all_entries[start:end]:
                context.append({
                    'line': line_num,
                    'user_input': entry.get('user_input', ''),
                    'assistant_response': entry.get('assistant_response', ''),
                    'timestamp': entry.get('timestamp', ''),
                    'is_match': line_num == line
                })

            return {
                'context': context,
                'total_turns': len(all_entries)
            }

        except Exception as e:
            print(f"⚠️ Failed to read context from {filename}: {e}")
            return {}

    def _read_entry(self, filename: str, line: int) -> Optional[Dict]:
        """Read a single entry from a .jsonl file by line number."""
        log_path = os.path.join(self.log_dir, filename)
        if not os.path.exists(log_path):
            return None

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for current_line, raw_line in enumerate(f):
                    if current_line == line:
                        return json.loads(raw_line.strip())
            return None
        except Exception as e:
            print(f"⚠️ Failed to read entry {filename}:{line}: {e}")
            return None

    def get_entry_count(self) -> int:
        """Get the number of indexed entries."""
        return len(self.files)

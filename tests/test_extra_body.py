"""
Tests for the [network.extra_body] passthrough feature.

Verifies that:
1. extra_body defaults to an empty dict when absent.
2. Values under [network.extra_body] are merged into the built request payload
   for both the OpenAI and Anthropic interfaces.
3. A config round-trip (_build_toml_string then reload) preserves extra_body.
"""

import unittest
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import Config, NetworkConfig


class TestNetworkConfigExtraBodyDefault(unittest.TestCase):
    """Test that extra_body defaults to an empty dict."""

    def test_default_is_empty_dict(self):
        """extra_body should default to {} when not specified."""
        cfg = NetworkConfig()
        self.assertEqual(cfg.extra_body, {})
        self.assertIsInstance(cfg.extra_body, dict)

    def test_instances_are_independent(self):
        """Each instance should get its own dict, not share a mutable default."""
        cfg1 = NetworkConfig()
        cfg2 = NetworkConfig()
        cfg1.extra_body["key"] = "value"
        self.assertEqual(cfg2.extra_body, {})


class TestExtraBodyLoading(unittest.TestCase):
    """Test that [network.extra_body] values are loaded from TOML."""

    def _write_config(self, toml_content: str) -> str:
        """Write a TOML string to a temp file and return its path."""
        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.toml', delete=False, encoding='utf-8'
        )
        f.write(toml_content)
        f.close()
        return f.name

    def tearDown(self):
        # Remove any leftover temp files / auto-created default configs
        if hasattr(self, '_tmp_path') and os.path.exists(self._tmp_path):
            os.unlink(self._tmp_path)

    def test_extra_body_loaded_from_toml(self):
        """Values under [network.extra_body] should populate extra_body."""
        toml = (
            "[network]\n"
            "api_url = \"http://localhost/v1/chat/completions\"\n"
            "model = \"test-model\"\n"
            "api_type = \"openai\"\n"
            "\n"
            "[network.extra_body]\n"
            'reasoning_effort = "high"\n'
        )
        self._tmp_path = self._write_config(toml)
        cfg = Config(config_file=self._tmp_path)
        self.assertEqual(cfg.network.extra_body.get("reasoning_effort"), "high")

    def test_missing_extra_body_defaults_to_empty(self):
        """When [network.extra_body] is absent, extra_body should be {}."""
        toml = (
            "[network]\n"
            "api_url = \"http://localhost/v1/chat/completions\"\n"
            "model = \"test-model\"\n"
            "api_type = \"openai\"\n"
        )
        self._tmp_path = self._write_config(toml)
        cfg = Config(config_file=self._tmp_path)
        self.assertEqual(cfg.network.extra_body, {})


class TestExtraBodyRoundTrip(unittest.TestCase):
    """Test that save/load round-trips preserve extra_body."""

    def test_round_trip_preserves_extra_body(self):
        """_build_toml_string then Config reload should preserve extra_body."""
        toml = (
            "[network]\n"
            "api_url = \"http://localhost/v1/chat/completions\"\n"
            "model = \"test-model\"\n"
            "api_type = \"openai\"\n"
            "\n"
            "[network.extra_body]\n"
            'reasoning_effort = "medium"\n'
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.toml', delete=False, encoding='utf-8'
        ) as f:
            f.write(toml)
            tmp_path = f.name
        try:
            cfg = Config(config_file=tmp_path)
            serialized = cfg._build_toml_string()

            # Write serialized form back and reload
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.toml', delete=False, encoding='utf-8'
            ) as f2:
                f2.write(serialized)
                tmp_path2 = f2.name
            try:
                cfg2 = Config(config_file=tmp_path2)
                self.assertEqual(
                    cfg2.network.extra_body.get("reasoning_effort"), "medium"
                )
            finally:
                os.unlink(tmp_path2)
        finally:
            os.unlink(tmp_path)

    def test_round_trip_empty_extra_body_no_section(self):
        """When extra_body is empty, the serialized TOML should not include [network.extra_body]."""
        toml = (
            "[network]\n"
            "api_url = \"http://localhost/v1/chat/completions\"\n"
            "model = \"test-model\"\n"
            "api_type = \"openai\"\n"
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.toml', delete=False, encoding='utf-8'
        ) as f:
            f.write(toml)
            tmp_path = f.name
        try:
            cfg = Config(config_file=tmp_path)
            serialized = cfg._build_toml_string()
            self.assertNotIn("[network.extra_body]", serialized)
        finally:
            os.unlink(tmp_path)


class TestExtraBodyPayloadMerge(unittest.TestCase):
    """Test that extra_body values appear in the built request payload."""

    def _make_minimal_config_file(self, extra_body: dict = None) -> str:
        """Write a minimal TOML config to a temp file and return its path."""
        lines = [
            "[network]",
            'api_url = "http://localhost/v1/chat/completions"',
            'model = "test-model"',
            'api_type = "openai"',
        ]
        if extra_body:
            lines.append("\n[network.extra_body]")
            for k, v in extra_body.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                else:
                    lines.append(f'{k} = {v}')
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.toml', delete=False, encoding='utf-8'
        ) as f:
            f.write("\n".join(lines) + "\n")
            return f.name

    def test_openai_interface_includes_extra_body(self):
        """extra_body keys should appear in the OpenAI interface payload."""
        from core.llm_interface import LLMInterface
        import core.llm_interface as llm_module

        tmp_path = self._make_minimal_config_file({"reasoning_effort": "high"})
        try:
            original_config = llm_module.config
            llm_module.config = Config(config_file=tmp_path)
            try:
                iface = LLMInterface(
                    api_url="http://localhost/v1/chat/completions",
                    model="test-model",
                    system_prompt="test",
                )
                payload = iface._build_query_payload("hello")
                self.assertEqual(payload.get("reasoning_effort"), "high")
            finally:
                llm_module.config = original_config
        finally:
            os.unlink(tmp_path)

    def test_anthropic_interface_includes_extra_body(self):
        """extra_body keys should appear in the Anthropic interface payload."""
        from core.anthropic_llm_interface import AnthropicLLMInterface
        import core.anthropic_llm_interface as anthropic_module
        import core.config as config_module

        tmp_path = self._make_minimal_config_file({"reasoning_effort": "low"})
        try:
            original_config = anthropic_module.config
            new_cfg = Config(config_file=tmp_path)
            # Anthropic interface uses max_chat_tokens; set a value to avoid None
            new_cfg.assistant.max_chat_tokens = 1024  # set a value since max_chat_tokens is required by the interface
            anthropic_module.config = new_cfg
            try:
                iface = AnthropicLLMInterface(
                    api_url="http://localhost/v1/chat/completions",
                    model="test-model",
                    system_prompt="test",
                )
                payload = iface._build_query_payload("hello")
                self.assertEqual(payload.get("reasoning_effort"), "low")
            finally:
                anthropic_module.config = original_config
        finally:
            os.unlink(tmp_path)

    def test_empty_extra_body_is_noop(self):
        """When extra_body is empty, the payload should not be altered."""
        from core.llm_interface import LLMInterface
        import core.llm_interface as llm_module

        tmp_path = self._make_minimal_config_file()
        try:
            original_config = llm_module.config
            llm_module.config = Config(config_file=tmp_path)
            try:
                iface = LLMInterface(
                    api_url="http://localhost/v1/chat/completions",
                    model="test-model",
                    system_prompt="test",
                )
                payload = iface._build_query_payload("hello")
                # Standard keys should be present; no extra keys from extra_body
                self.assertIn("model", payload)
                self.assertIn("messages", payload)
            finally:
                llm_module.config = original_config
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()

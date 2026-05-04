#!/usr/bin/env python3
"""
Voice Assistant - Main Entry Point.

Plugin wiring lives in plugins/kazka_plugins.py — this file just builds
the engine, asks the loader to discover and load plugins, and runs the
main loop.
"""

import time
import argparse

from core.engine import AssistantEngine
from core.config import config
from core.plugin_loader import PluginLoader


# Plugins that the per-flag CLI options turn off. Anything not in this
# map can still be disabled via the generic --disable flag below.
CLI_FLAG_MAP = {
    "no_voice":  ["voice"],
    "no_button": ["button"],
    "no_tts":    ["tts"],
    "no_led":    ["led"],
    "no_web":    ["web_input", "web_output", "web_service"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Full voice assistant
    python main.py --no-voice         # No voice input (text + button only)
    python main.py --text-only        # Text-only mode (no voice/TTS/button)
    python main.py --no-tts           # No TTS (console output only)
    python main.py --disable led      # Generic disable for any plugin by name
"""
    )
    parser.add_argument('--no-voice',  action='store_true', help='Disable voice input')
    parser.add_argument('--no-button', action='store_true', help='Disable hardware button input')
    parser.add_argument('--no-tts',    action='store_true', help='Disable TTS output')
    parser.add_argument('--no-led',    action='store_true', help='Disable LED output')
    parser.add_argument('--no-web',    action='store_true', help='Disable web server input/output')
    parser.add_argument('--text-only', action='store_true',
                        help='Text-only mode (no voice/TTS/button/LED/web)')
    parser.add_argument('--disable',   action='append', default=[], metavar='NAME',
                        help='Disable a plugin by name (repeatable)')
    return parser.parse_args()


def build_disabled_set(args) -> set:
    """Translate CLI flags + config gates into a set of disabled plugin names."""
    disabled: set = set()

    if args.text_only:
        for flag in CLI_FLAG_MAP:
            setattr(args, flag, True)

    for flag, names in CLI_FLAG_MAP.items():
        if getattr(args, flag):
            disabled.update(names)

    # Config-driven gates that today's main.py honors.
    if not config.web.enabled:
        disabled.update(["web_input", "web_output", "web_service"])
    if not config.sleep.enabled:
        disabled.add("sleep_watchdog")

    disabled.update(args.disable)
    return disabled


def main():
    args = parse_args()

    print("=" * 60)
    print(f"            {config.assistant.name.upper()} VOICE ASSISTANT")
    print("=" * 60)
    print()

    engine = AssistantEngine()

    loader = PluginLoader(engine, config, disabled=build_disabled_set(args))
    loader.discover().load_all()

    engine.startup()

    print()
    print("-" * 60)
    print("Ready! Say the wake word or press 't' to type a message.")
    print("Press 'q' to quit, Ctrl-C for emergency shutdown.")
    print("-" * 60)
    print()

    keyboard_interrupt = False
    try:
        while engine.running:
            time.sleep(1)
    except KeyboardInterrupt:
        keyboard_interrupt = True
        print("\n\n👋 Keyboard interrupt - fast shutdown (skipping memories)...")

    if engine.running:
        engine.shutdown(save_memories=not keyboard_interrupt)

    if engine.engine_thread and engine.engine_thread.is_alive():
        engine.engine_thread.join(timeout=30.0)

    print(f"\n✅ {config.assistant.name} shut down cleanly. Goodbye!")


if __name__ == "__main__":
    main()

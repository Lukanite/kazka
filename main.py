#!/usr/bin/env python3
"""
Voice Assistant - Main Entry Point

This is the main entry point for the voice assistant.
It initializes all plugins based on configuration and runs the engine.

Usage:
    python main.py                    # Start with default config
    python main.py --no-voice        # Start without voice input
    python main.py --text-only       # Text-only mode (no voice/TTS)
"""

import time
import argparse

from core.engine import AssistantEngine
from core.config import config

# Input plugins
from plugins.inputs.voice.voice_plugin import VoiceInputPlugin
from plugins.inputs.button.button_plugin import ButtonInputPlugin
from plugins.inputs.text.text_plugin import TextInputPlugin

# Web plugins
from plugins.inputs.web.web_plugin import WebInputPlugin
from plugins.outputs.web_output_plugin import WebOutputPlugin
from plugins.services.web_service_plugin import WebServicePlugin

# Output plugins
from plugins.outputs.console import ConsoleOutputPlugin
from plugins.outputs.tts_plugin import TTSOutputPlugin
from plugins.outputs.led_plugin import LEDOutputPlugin

# Service plugins
from plugins.services.sleep_watchdog import SleepWatchdogPlugin
from plugins.services.scheduler import SchedulerPlugin


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Full voice assistant
    python main.py --no-voice        # No voice input (text + button only)
    python main.py --text-only       # Text-only mode (no voice/TTS/button)
    python main.py --no-tts          # No TTS (console output only)
"""
    )

    parser.add_argument(
        '--no-voice',
        action='store_true',
        help='Disable voice input (wake word, VAD, STT)'
    )

    parser.add_argument(
        '--no-button',
        action='store_true',
        help='Disable hardware button input'
    )

    parser.add_argument(
        '--no-tts',
        action='store_true',
        help='Disable TTS output (console only)'
    )

    parser.add_argument(
        '--no-led',
        action='store_true',
        help='Disable LED output'
    )

    parser.add_argument(
        '--no-web',
        action='store_true',
        help='Disable web server input/output'
    )

    parser.add_argument(
        '--text-only',
        action='store_true',
        help='Text-only mode (no voice/TTS/button/LED)'
    )

    return parser.parse_args()


def main():
    """Main entry point for the voice assistant."""
    args = parse_args()

    # Handle text-only mode
    if args.text_only:
        args.no_voice = True
        args.no_button = True
        args.no_tts = True
        args.no_led = True
        args.no_web = True

    print("=" * 60)
    print(f"            {config.assistant.name.upper()} VOICE ASSISTANT")
    print("=" * 60)
    print()

    # Create engine
    engine = AssistantEngine()

    # Register input plugins
    if not args.no_voice:
        try:
            voice = VoiceInputPlugin(engine)
            engine.register_input("voice", voice)
        except Exception as e:
            print(f"   ⚠️  Voice plugin failed to load: {e}")
            print("   Voice input disabled.")

    if not args.no_button:
        try:
            button = ButtonInputPlugin(engine)
            engine.register_input("button", button)
        except Exception as e:
            print(f"   ⚠️  Button plugin failed to load: {e}")
            print("   Button input disabled.")

    # Always enable text input
    text = TextInputPlugin(engine)
    engine.register_input("text", text)

    # Set up shutdown callback from text plugin
    text.on_shutdown(lambda: engine.shutdown(save_memories=True))

    # Register web plugins (input + output share the same server instance)
    if not args.no_web and config.web.enabled:
        try:
            web_input = WebInputPlugin(engine)
            engine.register_input("web", web_input)
            web_output = WebOutputPlugin(engine, web_input.web_server)
            engine.register_output("web", web_output)
            web_service = WebServicePlugin(engine, web_input.web_server)
            engine.register_service("web_service", web_service)
        except Exception as e:
            print(f"   ⚠️  Web plugin failed to load: {e}")
            print("   Web server disabled.")

    # Register output plugins
    console = ConsoleOutputPlugin(engine, {})
    engine.register_output("console", console)

    if not args.no_tts:
        try:
            tts = TTSOutputPlugin(engine)
            engine.register_output("tts", tts)
        except Exception as e:
            print(f"   ⚠️  TTS plugin failed to load: {e}")
            print("   TTS output disabled.")

    if not args.no_led:
        try:
            led = LEDOutputPlugin(engine)
            engine.register_output("led", led)
        except Exception as e:
            print(f"   ⚠️  LED plugin failed to load: {e}")
            print("   LED output disabled.")

    # Register service plugins
    scheduler = SchedulerPlugin(engine)
    engine.register_service("scheduler", scheduler)

    if config.sleep.enabled:
        try:
            sleep_watchdog = SleepWatchdogPlugin(engine)
            engine.register_service("sleep_watchdog", sleep_watchdog)
        except Exception as e:
            print(f"   ⚠️  Sleep watchdog failed to load: {e}")
            print("   Sleep watchdog disabled.")

    # Start engine and all plugins
    engine.startup()

    print()
    print("-" * 60)
    print("Ready! Say the wake word or press 't' to type a message.")
    print("Press 'q' to quit, Ctrl-C for emergency shutdown.")
    print("-" * 60)
    print()

    # Main thread waits for shutdown
    keyboard_interrupt = False
    try:
        while engine.running:
            time.sleep(1)
    except KeyboardInterrupt:
        keyboard_interrupt = True
        print("\n\n👋 Keyboard interrupt - fast shutdown (skipping memories)...")

    # Handle shutdown
    if engine.running:
        # Ctrl-C = fast shutdown, 'q' = already triggered shutdown with memories
        engine.shutdown(save_memories=not keyboard_interrupt)

    # Wait for engine thread to fully stop (memory processing happens here)
    if engine.engine_thread and engine.engine_thread.is_alive():
        engine.engine_thread.join(timeout=30.0)  # Give time for memory processing

    print(f"\n✅ {config.assistant.name} shut down cleanly. Goodbye!")


if __name__ == "__main__":
    main()

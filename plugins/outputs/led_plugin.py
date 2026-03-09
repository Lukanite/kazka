"""
LED Output Plugin - Visual feedback for voice assistant states.

This plugin provides visual feedback via LED patterns:
- Controls LED based on voice plugin state changes
- Updates LED when speaking (via output)
- Fast operations - no queue needed

Note: This plugin requires gpiozero and hardware GPIO access.
Uses endpoint registration for state updates from voice plugin.
"""

from typing import Dict, Any

from core.plugin_base import OutputPlugin


class LEDOutputPlugin(OutputPlugin):
    """
    LED output plugin for visual state feedback.

    Provides visual feedback through LED patterns:
    - WAITING: LED off
    - VERIFYING: Solid 100%
    - LISTENING: Solid 100%
    - PROCESSING: Medium pulse
    - SPEAKING: Slow pulse

    Unlike TTS/Console, this is NOT queued because LED
    operations are very fast (<1ms) and don't block.

    Registers endpoint 'led.set_state' for voice plugin to update state.
    """

    def __init__(self, engine: 'AssistantEngine', plugin_config: Dict[str, Any] = None):
        """
        Initialize LED output plugin.

        Args:
            engine: Reference to the engine
            plugin_config: Optional configuration overrides
                - led_pin: GPIO pin for LED (default: 25)
        """
        super().__init__(engine, "led")
        self.plugin_config = plugin_config or {}
        self.led_pin = self.plugin_config.get('led_pin', 25)
        self.led_controller = None
        self.running = False

    def start(self):
        """Start LED output plugin."""
        print(f"💡 Starting LED output plugin (pin={self.led_pin})...")
        self.running = True

        try:
            # Import here to handle systems without GPIO
            from plugins.outputs.led_controller import LEDController

            # Initialize LED controller
            self.led_controller = LEDController(self.led_pin)
            self.led_controller.start()

            # Register endpoint for state control
            self.engine.register_endpoint("led", "set_state", self._on_set_state)

            # Set initial state
            self.led_controller.set_state("OFF")

            print("✅ LED output plugin ready")

        except ImportError as e:
            print(f"   ⚠️  LED plugin failed (no gpiozero?): {e}")
            print("   LED output disabled.")
            self.led_controller = None
            self.running = False

        except Exception as e:
            print(f"   ⚠️  LED plugin failed to start: {e}")
            print("   LED output disabled.")
            self.led_controller = None
            self.running = False

    def output(self, text: str, metadata: Dict[str, Any]):
        """
        Update LED when output is sent (e.g., speaking).

        Args:
            text: Output text (ignored for LED)
            metadata: Output metadata
        """
        if not self.led_controller:
            return

        # Set speaking pattern when output is broadcast
        # This provides visual feedback that the assistant is speaking
        self.led_controller.set_state("PULSE_SLOW")

    def should_handle(self, metadata: Dict[str, Any]) -> bool:
        """
        LED handles all output to show speaking state.

        Args:
            metadata: Output metadata

        Returns:
            True if LED controller is active
        """
        return self.led_controller is not None

    def stop(self):
        """Stop LED output plugin."""
        print("🛑 Stopping LED output plugin...")
        self.running = False

        if self.led_controller:
            self.led_controller.stop()
            self.led_controller = None

        print("   LED output stopped")

    def _on_set_state(self, data: Dict[str, Any]) -> None:
        """
        Handle state change request from voice plugin.

        Called via endpoint: led.set_state

        Args:
            data: {'state': 'WAITING'/'LISTENING'/etc.}
        """
        if not self.led_controller:
            return

        state = data.get('state', 'WAITING')

        # Map voice states to LED patterns
        state_patterns = {
            "WAITING": "OFF",
            "VERIFYING": "SOLID_FULL",
            "LISTENING": "SOLID_FULL",
            "PTT": "SOLID_FULL",
            "PROCESSING_VAD": "PULSE_MEDIUM",
            "PROCESSING_PTT": "PULSE_MEDIUM",
            "SPEAKING": "PULSE_SLOW",
        }

        pattern = state_patterns.get(state, "PULSE_SLOW")
        self.led_controller.set_state(pattern)

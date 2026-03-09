"""
Button Input Plugin - Hardware button for wake word bypass and PTT.

This plugin handles GPIO button input:
- Short press: Triggers wake on RELEASE (bypasses wake word detection)
  Note: Wake triggers on release to avoid the button "ker-thunk" sound
  confusing the VAD (voice activity detection).
- Hold: Starts push-to-talk (PTT) recording
- Release: Stops PTT and processes command

Communicates with voice plugin via endpoints.
"""

from typing import Dict, Any

from core.plugin_base import InputPlugin
from plugins.inputs.button.button_controller import ButtonController


class ButtonInputPlugin(InputPlugin):
    """
    Hardware button input plugin.

    Provides two input modes:
    1. Short press (< 0.5s): Triggers wake on RELEASE - bypasses wake word detection
       (Delayed to release to avoid button click sound confusing VAD)
    2. Hold (> 0.5s): Push-to-talk mode - hold to record, release to process

    Communicates with voice plugin via endpoints:
    - voice.wake_requested: Skip wake word and start listening
    - voice.ptt_started: Start PTT recording
    - voice.ptt_stopped: Stop PTT and process

    Note: This plugin requires gpiozero and hardware GPIO access.
    LED control is handled separately by the LED output plugin.
    """

    def __init__(self, engine: 'AssistantEngine', plugin_config: Dict[str, Any] = None):
        """
        Initialize button input plugin.

        Args:
            engine: Reference to the engine
            plugin_config: Optional configuration overrides
                - button_pin: GPIO pin for button (default: 23)
        """
        super().__init__(engine, "button")
        self.plugin_config = plugin_config or {}

        # GPIO pin (Voice HAT default)
        self.button_pin = self.plugin_config.get('button_pin', 23)

        # Controller (initialized in start())
        self.controller = None
        self.running = False

        # Track if this was a short press (for release handling)
        self._was_short_press = False

    def start(self):
        """Start button input plugin and GPIO controller."""
        print(f"🔘 Starting button input plugin (pin={self.button_pin})...")
        self.running = True

        try:
            # Initialize button controller (LED is separate)
            self.controller = ButtonController(button_pin=self.button_pin)

            # Register button callbacks
            # Note: short press callback just sets a flag, actual wake happens on release
            self.controller.on_short_press(self._on_short_press)
            self.controller.on_hold_start(self._on_hold_start)
            self.controller.on_release(self._on_release)

            # Start controller
            self.controller.start()

            print("✅ Button input plugin ready")

        except Exception as e:
            print(f"   ⚠️  Button plugin failed to start (no GPIO?): {e}")
            print("   Button input disabled.")
            self.controller = None
            self.running = False

    def stop(self):
        """Stop button input plugin."""
        print("🛑 Stopping button input plugin...")
        self.running = False

        if self.controller:
            self.controller.stop()
            self.controller = None

        print("   Button input stopped")

    def _on_short_press(self):
        """Handle short button press detection - just set flag, wake on release."""
        if not self.running:
            return

        # Mark that this was a short press - actual wake happens on release
        # to avoid the button click sound confusing VAD
        self._was_short_press = True
        print("\n[🔘 Button] Short press detected (will trigger wake on release)")

    def _on_hold_start(self):
        """Handle button hold start - begin PTT recording."""
        if not self.running:
            return

        # Clear short press flag since this is a hold
        self._was_short_press = False

        print("\n[🔘 Button] Hold detected - starting PTT!")

        # Send PTT start to voice plugin
        self.engine.endpoint_send("voice", "ptt_started", {'source': 'button'})

    def _on_release(self):
        """Handle button release - trigger wake if short press, or stop PTT."""
        if not self.running:
            return

        if self._was_short_press:
            # Short press release - NOW trigger wake (after button click sound is done)
            self._was_short_press = False
            print("[🔘 Button] Release after short press - triggering wake!")
            self.engine.endpoint_send("voice", "wake_requested", {'source': 'button'})
        else:
            # Hold release - stop PTT and process
            # The voice plugin will ignore this if not in PTT state
            self.engine.endpoint_send("voice", "ptt_stopped", {'source': 'button'})

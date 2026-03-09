"""
LED controller for Voice Assistant.
Provides visual feedback for different assistant states using PWM LED patterns.
"""

import threading
from typing import Optional

try:
    from gpiozero import PWMLED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    PWMLED = None


class LEDController:
    """
    Controls PWM LED with various patterns to indicate voice assistant states.

    Uses Event-based synchronization for efficient state changes instead of polling.

    Patterns:
    - OFF: LED completely off
    - SOLID: Constant brightness
    - PULSE: Smooth breathing effect
    - BLINK: On/off flashing
    - DOUBLE_BLINK: Two quick blinks, repeated
    """

    def __init__(self, led_pin: int = 25):
        """
        Initialize LED controller.

        Args:
            led_pin: GPIO pin number for LED (default: 25 for Voice HAT)
        """
        if not GPIO_AVAILABLE:
            raise RuntimeError("gpiozero not available - LED controller requires GPIO")

        self.led = PWMLED(led_pin)
        self._running = True
        self._current_state = "OFF"
        self._target_state = "OFF"
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._state_changed = threading.Event()

    def _run_pattern(self):
        """Main pattern runner thread - event-driven state changes."""
        while self._running:
            # Wait for state change or stop signal
            self._state_changed.wait()

            # Check if we should stop
            if not self._running:
                break

            # Get the new state
            with self._lock:
                state = self._target_state
                self._current_state = state
                self._state_changed.clear()

            # Execute the appropriate pattern
            # Each pattern runs until state changes or stopped
            if state == "OFF":
                self.led.off()
                # Wait for next state change
                self._wait_for_state_change()
            elif state == "SOLID":
                with self._lock:
                    self.led.value = 0.8
                self._wait_for_state_change()
            elif state == "SOLID_HIGH":
                with self._lock:
                    self.led.value = 0.8
                self._wait_for_state_change()
            elif state == "SOLID_FULL":
                with self._lock:
                    self.led.value = 1.0
                self._wait_for_state_change()
            elif state == "PULSE_SLOW":
                self._pulse_pattern(min_val=0.1, max_val=1.0, duration=2.0)
            elif state == "PULSE_FAST":
                self._pulse_pattern(min_val=0.1, max_val=0.9, duration=0.5)
            elif state == "PULSE_MEDIUM":
                self._pulse_pattern(min_val=0.0, max_val=0.6, duration=0.8)
            elif state == "BLINK_FAST":
                self._blink_pattern(on_duration=0.1, off_duration=0.1)
            elif state == "DOUBLE_BLINK":
                self._double_blink_pattern()

    def _wait_for_state_change(self, timeout: Optional[float] = None):
        """Wait for state change with optional timeout.

        Args:
            timeout: Maximum time to wait in seconds. None = wait indefinitely.
        """
        # We use a loop with small timeout to check _running frequently
        while self._running:
            if self._state_changed.wait(timeout=0.1):
                # State changed!
                return
            if timeout is not None:
                timeout -= 0.1
                if timeout <= 0:
                    return

    def _pulse_pattern(self, min_val: float, max_val: float, duration: float):
        """
        Breathing pulse pattern with event-based state change detection.

        Args:
            min_val: Minimum brightness (0.0-1.0)
            max_val: Maximum brightness (0.0-1.0)
            duration: Full cycle time in seconds
        """
        step_duration = duration / 40  # 40 steps for smoothness

        while self._running:
            # Check if we should still be pulsing
            with self._lock:
                if not self._current_state.startswith("PULSE"):
                    return
                current_state = self._current_state

            # Fade up
            for i in range(20):
                progress = i / 20
                value = min_val + (max_val - min_val) * progress
                self.led.value = value

                # Wait for step_duration, but wake immediately if state changes
                if self._state_changed.wait(timeout=step_duration):
                    return  # State changed, exit to process new state

                # Verify we're still in the same pulse mode
                with self._lock:
                    if not self._current_state.startswith("PULSE") or self._current_state != current_state:
                        return

            # Fade down
            for i in range(20):
                progress = i / 20
                value = max_val - (max_val - min_val) * progress
                self.led.value = value

                # Wait for step_duration, but wake immediately if state changes
                if self._state_changed.wait(timeout=step_duration):
                    return  # State changed, exit to process new state

                # Verify we're still in the same pulse mode
                with self._lock:
                    if not self._current_state.startswith("PULSE") or self._current_state != current_state:
                        return

    def _blink_pattern(self, on_duration: float, off_duration: float):
        """
        Simple blink pattern with event-based state change detection.

        Args:
            on_duration: Time to stay on (seconds)
            off_duration: Time to stay off (seconds)
        """
        while self._running:
            with self._lock:
                if not self._current_state.startswith("BLINK"):
                    return

            self.led.on()

            # Wait on_duration, but wake immediately if state changes
            if self._state_changed.wait(timeout=on_duration):
                return  # State changed

            with self._lock:
                if not self._current_state.startswith("BLINK"):
                    return

            self.led.off()

            # Wait off_duration, but wake immediately if state changes
            if self._state_changed.wait(timeout=off_duration):
                return  # State changed

    def _double_blink_pattern(self):
        """Double blink pattern (blink-blink-pause) with event detection."""
        while self._running:
            with self._lock:
                if self._current_state != "DOUBLE_BLINK":
                    return

            # First blink
            self.led.on()
            if self._state_changed.wait(timeout=0.1):
                return
            self.led.off()
            if self._state_changed.wait(timeout=0.1):
                return

            # Second blink
            self.led.on()
            if self._state_changed.wait(timeout=0.1):
                return
            self.led.off()
            if self._state_changed.wait(timeout=0.1):
                return

            # Pause
            if self._state_changed.wait(timeout=0.6):
                return

    def set_state(self, state: str):
        """
        Set LED state/pattern (thread-safe).

        Args:
            state: One of: OFF, SOLID, PULSE_SLOW, PULSE_FAST, PULSE_MEDIUM,
                   BLINK_FAST, DOUBLE_BLINK, SOLID_HIGH, SOLID_FULL
        """
        with self._lock:
            if self._target_state != state:
                self._target_state = state
                self._state_changed.set()

    def start(self):
        """Start the LED pattern runner thread."""
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self._run_pattern, daemon=True)
            self._thread.start()
            # Set initial state
            self.set_state("OFF")

    def stop(self):
        """Stop the LED controller and turn off LED."""
        self._running = False
        self._state_changed.set()  # Wake up the thread
        if self._thread:
            self._thread.join(timeout=1.0)
        self.led.off()

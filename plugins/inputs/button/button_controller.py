"""
Button controller for Voice Assistant.
Handles button input with press detection and callbacks.
"""

import threading
import time
from typing import Optional, Callable

try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    Button = None


class ButtonController:
    """
    Handles button input with press detection and callbacks.

    Uses efficient event-based timing instead of polling loops.

    Features:
    - Short press detection (< 0.5s)
    - Hold-to-speak (press and hold to record, release to process)
    - Button release detection
    """

    def __init__(self, button_pin: int = 23):
        """
        Initialize button controller.

        Args:
            button_pin: GPIO pin number for button (default: 23 for Voice HAT)
        """
        if not GPIO_AVAILABLE:
            raise RuntimeError("gpiozero not available - button controller requires GPIO")

        self.button = Button(button_pin)
        self._press_start_time = None
        self._short_press_callback: Optional[Callable[[], None]] = None
        self._hold_start_callback: Optional[Callable[[], None]] = None
        self._release_callback: Optional[Callable[[], None]] = None
        self._hold_triggered = False
        self._hold_timer: Optional[threading.Timer] = None
        self._running = True
        self._lock = threading.Lock()

    def on_short_press(self, callback: Callable[[], None]):
        """Register callback for short press (< 0.5s)."""
        with self._lock:
            self._short_press_callback = callback

    def on_hold_start(self, callback: Callable[[], None]):
        """Register callback for when button is held (> 0.5s)."""
        with self._lock:
            self._hold_start_callback = callback

    def on_release(self, callback: Callable[[], None]):
        """Register callback for button release."""
        with self._lock:
            self._release_callback = callback

    def _on_press(self):
        """Handle button press event."""
        with self._lock:
            self._press_start_time = time.time()
            self._hold_triggered = False

        # Start hold timer (more efficient than polling thread)
        if self._hold_start_callback:
            self._hold_timer = threading.Timer(0.5, self._trigger_hold)
            self._hold_timer.start()

    def _trigger_hold(self):
        """Trigger hold callback when timer expires."""
        with self._lock:
            if self.button.is_pressed and not self._hold_triggered:
                self._hold_triggered = True
                callback = self._hold_start_callback
            else:
                callback = None

        if callback:
            callback()

    def _on_release(self):
        """Handle button release event."""
        # Cancel hold timer if still running
        if self._hold_timer:
            self._hold_timer.cancel()
            self._hold_timer = None

        with self._lock:
            if self._press_start_time is None:
                return

            press_duration = time.time() - self._press_start_time
            hold_triggered = self._hold_triggered
            short_callback = self._short_press_callback
            release_callback = self._release_callback
            self._press_start_time = None

        # Trigger short press if released quickly (< 0.5s)
        if press_duration < 0.5 and not hold_triggered and short_callback:
            short_callback()

        # Always trigger release callback (for hold-to-speak)
        if release_callback:
            release_callback()

    def start(self):
        """Start button event handling."""
        self.button.when_pressed = self._on_press
        self.button.when_released = self._on_release

    def stop(self):
        """Stop button controller."""
        self._running = False
        if self._hold_timer:
            self._hold_timer.cancel()
        self.button.when_pressed = None
        self.button.when_released = None

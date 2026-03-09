"""
Sleep Watchdog Service Plugin.

Monitors user activity and triggers periodic memory flush + conversation reset
during configured quiet hours after sufficient inactivity. This prevents the
conversation context from growing unboundedly during long-running sessions.
"""

import time
import threading
from datetime import datetime

from core.plugin_base import ServicePlugin
from core.requests import SleepRequest
from core.config import config


class SleepWatchdogPlugin(ServicePlugin):
    """
    Event-driven watchdog that triggers sleep cycles based on inactivity.

    Receives on_interaction() calls from the engine thread after each user
    exchange. Resets an inactivity timer on each call. When the timer fires
    (user has been quiet long enough), checks if the current time is within
    the configured sleeping hours window and if enough exchanges have
    accumulated, then puts a SleepRequest on the engine queue.
    """

    def __init__(self, engine: 'AssistantEngine'):
        super().__init__(engine, "sleep_watchdog")

        self.inactivity_seconds = config.sleep.inactivity_minutes * 60
        self.sleeping_hours_start = config.sleep.sleeping_hours_start
        self.sleeping_hours_end = config.sleep.sleeping_hours_end
        self.min_exchanges = config.sleep.min_exchanges

        self._exchange_count = 0
        self._last_interaction_time = 0.0
        self._timer: threading.Timer = None
        self._lock = threading.Lock()
        self._stopped = False
        self._sleep_pending = False

    def start(self):
        """Initialize watchdog state."""
        self._exchange_count = 0
        self._stopped = False
        print(f"   😴 Sleep watchdog active: "
              f"{config.sleep.inactivity_minutes}min inactivity, "
              f"{self.sleeping_hours_start}:00-{self.sleeping_hours_end}:00 window, "
              f"min {self.min_exchanges} exchanges")

    def stop(self):
        """Cancel any pending timer."""
        with self._lock:
            self._stopped = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def on_interaction_start(self):
        """
        Called on the engine thread when user input arrives, before LLM processing.

        Cancels any running inactivity timer — the user is actively engaged,
        so we shouldn't be counting idle time during LLM response latency.
        """
        with self._lock:
            if self._stopped:
                return

            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def on_interaction_end(self):
        """
        Called on the engine thread after a user interaction is fully processed.

        Increments exchange counter and starts the inactivity timer.
        The idle clock only starts ticking after the LLM has finished responding.
        """
        with self._lock:
            if self._stopped:
                return

            self._exchange_count += 1
            self._last_interaction_time = time.monotonic()

            # Don't start timers while a sleep cycle is pending or in progress
            if self._sleep_pending:
                return

            self._timer = threading.Timer(
                self.inactivity_seconds,
                self._on_idle
            )
            self._timer.daemon = True
            self._timer.start()

    def on_sleep_complete(self):
        """Reset state after a sleep cycle completes."""
        with self._lock:
            self._sleep_pending = False
            self._exchange_count = 0
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _on_idle(self):
        """
        Timer callback — runs on the timer's background thread.

        Checks if conditions are met for a sleep cycle:
        1. Enough exchanges since last sleep
        2. User has actually been idle for the full inactivity period
        3. Current time is within the sleeping hours window

        If the user interacted recently (e.g., this was a rescheduled-for-window
        timer), reschedules for the remaining inactivity period.
        """
        with self._lock:
            if self._stopped or self._sleep_pending:
                return

            self._timer = None

            # Check minimum exchanges threshold
            if self._exchange_count < self.min_exchanges:
                return

            # Check actual inactivity — the user may have interacted since
            # this timer was scheduled (e.g., rescheduled for window boundary)
            elapsed = time.monotonic() - self._last_interaction_time
            remaining_inactivity = self.inactivity_seconds - elapsed
            if remaining_inactivity > 1:
                # User was recently active — reschedule for when inactivity is met
                self._timer = threading.Timer(
                    remaining_inactivity,
                    self._on_idle
                )
                self._timer.daemon = True
                self._timer.start()
                return

            # Check sleeping hours window
            if self._in_sleeping_window():
                self._sleep_pending = True
                print(f"😴 Sleep watchdog: {self._exchange_count} exchanges, "
                      f"idle for {config.sleep.inactivity_minutes}+ min, "
                      f"within sleeping hours — triggering sleep cycle")
                self.engine.request_queue.put(SleepRequest())
            else:
                # Not in the window yet — schedule for when the window opens
                seconds_until_window = self._seconds_until_window()
                if seconds_until_window > 0:
                    print(f"😴 Sleep watchdog: idle with {self._exchange_count} exchanges, "
                          f"but outside sleeping hours. "
                          f"Will check again in {seconds_until_window // 60:.0f}min")
                    self._timer = threading.Timer(
                        seconds_until_window,
                        self._on_idle
                    )
                    self._timer.daemon = True
                    self._timer.start()

    def _in_sleeping_window(self) -> bool:
        """Check if current time is within the configured sleeping hours."""
        current_hour = datetime.now().hour

        if self.sleeping_hours_start <= self.sleeping_hours_end:
            # Simple range (e.g., 2-6)
            return self.sleeping_hours_start <= current_hour < self.sleeping_hours_end
        else:
            # Wraps midnight (e.g., 22-6)
            return current_hour >= self.sleeping_hours_start or current_hour < self.sleeping_hours_end

    def _seconds_until_window(self) -> int:
        """Calculate seconds until the sleeping hours window opens."""
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute

        start = self.sleeping_hours_start
        current_minutes_total = current_hour * 60 + current_minute
        start_minutes_total = start * 60

        if current_minutes_total < start_minutes_total:
            # Window is later today
            return (start_minutes_total - current_minutes_total) * 60
        else:
            # Window is tomorrow
            return ((24 * 60 - current_minutes_total) + start_minutes_total) * 60

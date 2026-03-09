"""
Scheduler Service Plugin.

Manages timers for self-wake functionality. Timers live only in memory
and are cancelled on shutdown — no persistence across restarts.
"""

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Callable, List

from core.plugin_base import ServicePlugin


@dataclass
class ScheduledTimer:
    """Represents a scheduled timer."""
    id: str
    delay_description: str  # Human-readable description of the delay
    trigger_time: str  # ISO 8601 timestamp
    created_at: str  # ISO 8601 timestamp
    is_active: bool = True
    timer_thread: Optional[threading.Timer] = None


class SchedulerPlugin(ServicePlugin):
    """
    Service plugin that manages timers for self-wake events.

    Timers fire on background threads but route their effects through
    the engine's request queue to stay thread-safe. All timer state is
    in-memory only — timers are lost on shutdown/restart.
    """

    def __init__(self, engine: 'AssistantEngine', max_timers: int = 20):
        super().__init__(engine, "scheduler")
        self.max_timers = max_timers

        self.timers: Dict[str, ScheduledTimer] = {}
        self._lock = threading.Lock()

    def start(self):
        """Initialize scheduler state."""
        print(f"   ⏰ Scheduler active (max {self.max_timers} timers)")

    def stop(self):
        """Cancel all active timers."""
        with self._lock:
            for timer in self.timers.values():
                if timer.is_active and timer.timer_thread:
                    timer.timer_thread.cancel()
                    timer.is_active = False
            self.timers.clear()
        print("   ⏰ Scheduler shutdown: all timers cancelled")

    def schedule_timer(self, delay_description: str, delay_seconds: float,
                       callback: Optional[Callable] = None) -> ScheduledTimer:
        """
        Schedule a new timer.

        Args:
            delay_description: Human-readable description (e.g., "30 minutes")
            delay_seconds: Delay in seconds
            callback: Optional callback when timer fires (timer_id, delay_description)

        Returns:
            ScheduledTimer object

        Raises:
            ValueError: If max timers limit reached
        """
        with self._lock:
            active_count = sum(1 for t in self.timers.values() if t.is_active)
            if active_count >= self.max_timers:
                raise ValueError(f"Maximum number of timers ({self.max_timers}) reached")

            timer_id = str(uuid.uuid4())[:8]
            now = datetime.now()
            trigger_time = datetime.fromtimestamp(now.timestamp() + delay_seconds)

            timer = ScheduledTimer(
                id=timer_id,
                delay_description=delay_description,
                trigger_time=trigger_time.isoformat(),
                created_at=now.isoformat(),
                is_active=True
            )

            timer.timer_thread = threading.Timer(
                delay_seconds,
                self._timer_fired,
                args=(timer_id, callback)
            )
            timer.timer_thread.daemon = True
            timer.timer_thread.start()

            self.timers[timer_id] = timer
            return timer

    def cancel_timer(self, timer_id: str) -> bool:
        """
        Cancel a timer by ID.

        Returns:
            True if cancelled, False if not found
        """
        with self._lock:
            timer = self.timers.get(timer_id)
            if not timer:
                return False

            if timer.timer_thread:
                timer.timer_thread.cancel()
            timer.is_active = False
            return True

    def list_timers(self, active_only: bool = True) -> List[ScheduledTimer]:
        """List all timers, optionally filtering to active only."""
        with self._lock:
            timers = list(self.timers.values())
            if active_only:
                timers = [t for t in timers if t.is_active]
            return timers

    def _timer_fired(self, timer_id: str, callback: Optional[Callable] = None):
        """
        Internal callback when a timer fires (runs on timer's background thread).

        Routes the wake event through the engine's request queue so that
        conversation mutations happen on the engine thread.
        """
        with self._lock:
            timer = self.timers.get(timer_id)
            if not timer:
                return
            timer.is_active = False
            delay_description = timer.delay_description

        # Route through the engine's request queue
        from core.requests import WakeRequest
        self.engine.request_queue.put(WakeRequest(
            timer_id=timer_id,
            delay_description=delay_description
        ))

        # Execute optional direct callback (for custom use)
        if callback:
            try:
                callback(timer_id, delay_description)
            except Exception as e:
                print(f"   ⏰ Timer callback error: {e}")

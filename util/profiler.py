"""Profiling utilities for timing and stats tracking in easyprobe."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class ProbeProfiler:
    """Tracks timing and stats for probing operations.

    Usage:
        profiler = ProbeProfiler(verbose=True)

        with profiler.time("model_loading"):
            model = load_model()

        profiler.record("n_layers", 12)
        profiler.record("hidden_dim", 768)

        print(profiler.summary())
    """

    verbose: bool = True
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def time(self, name: str, log_start: str | None = None, log_end: str | None = None) -> Generator[None, None, None]:
        """Context manager for timing a block of code.

        Args:
            name: Key to store the timing under
            log_start: Optional message to log when starting (if verbose)
            log_end: Optional message template to log when done (if verbose).
                     Can include {elapsed:.2f} placeholder for the elapsed time.

        Example:
            with profiler.time("extraction", log_start="Extracting...", log_end="Done in {elapsed:.2f}s"):
                extract_activations()
        """
        if log_start and self.verbose:
            print(log_start)

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[name] = elapsed

            if log_end and self.verbose:
                print(log_end.format(elapsed=elapsed, **self.stats))

    def record(self, name: str, value: Any) -> None:
        """Record a stat value.

        Args:
            name: Key to store the stat under
            value: The value to record
        """
        self.stats[name] = value

    def log(self, message: str) -> None:
        """Print a message if verbose mode is enabled.

        Args:
            message: Message to print. Can include placeholders for stats/timings.
        """
        if self.verbose:
            print(message)

    def get_timing(self, name: str) -> float | None:
        """Get a recorded timing value.

        Args:
            name: The timing key

        Returns:
            The elapsed time in seconds, or None if not recorded
        """
        return self.timings.get(name)

    def get_stat(self, name: str) -> Any | None:
        """Get a recorded stat value.

        Args:
            name: The stat key

        Returns:
            The stat value, or None if not recorded
        """
        return self.stats.get(name)

    def total_time(self) -> float:
        """Get the total of all recorded timings.

        Returns:
            Sum of all timing values in seconds
        """
        return sum(self.timings.values())

    def summary(self, title: str = "Profiler Summary") -> str:
        """Generate a formatted summary of all timings and stats.

        Args:
            title: Title for the summary section

        Returns:
            Formatted string with all timings and stats
        """
        lines = [f"\n{'='*60}", title, '='*60]

        if self.timings:
            lines.append("\nTimings:")
            max_key_len = max(len(k) for k in self.timings)
            for name, elapsed in self.timings.items():
                lines.append(f"  {name:<{max_key_len}}: {elapsed:>8.2f}s")
            lines.append(f"  {'-'*(max_key_len + 12)}")
            lines.append(f"  {'Total':<{max_key_len}}: {self.total_time():>8.2f}s")

        if self.stats:
            lines.append("\nStats:")
            max_key_len = max(len(k) for k in self.stats)
            for name, value in self.stats.items():
                lines.append(f"  {name:<{max_key_len}}: {value}")

        lines.append('='*60)
        return '\n'.join(lines)

    def reset(self) -> None:
        """Clear all recorded timings and stats."""
        self.timings.clear()
        self.stats.clear()

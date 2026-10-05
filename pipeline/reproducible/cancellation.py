"""Cooperative cancellation for SQLite work and stage workers."""

from __future__ import annotations

import signal
import sqlite3
import threading
from typing import Callable


class CancellationRequested(RuntimeError):
    """A worker received a stop request."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._previous: dict[int, object] = {}

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self, _signum: int | None = None, _frame: object | None = None) -> None:
        self._event.set()

    def progress(self) -> int:
        return 1 if self.requested else 0

    def check(self) -> None:
        if self.requested:
            raise CancellationRequested("stage cancellation requested")

    def install(self) -> None:
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self.request)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()


def attach_sqlite_progress(connection: sqlite3.Connection, callback: Callable[[], int], *, every: int = 1000) -> None:
    if every <= 0:
        raise ValueError("SQLite progress interval must be positive")
    connection.set_progress_handler(callback, every)


__all__ = ["CancellationRequested", "CancellationToken", "attach_sqlite_progress"]


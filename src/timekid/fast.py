"""timekid.fast

A low-overhead timing backend-oriented API.

Design goals:
- Minimize Python overhead in the hot path.
- Store raw integer nanoseconds internally; convert/round only at reporting time.
- Provide a stable Python API with an optional native (Rust) accelerator.

The initial implementation is pure Python and intentionally minimal.
A future Rust backend can implement the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Union
import time

Key = Union[str, int]


@dataclass(frozen=True, slots=True)
class Token:
    """Opaque handle returned by start()."""

    key: Key
    t0_ns: int


class FastTimer:
    """Low-overhead timer intended for *lots* of measurements.

    Typical usage:

        ft = FastTimer()
        key = ft.key_id("db")
        tok = ft.start(key)
        ...
        ft.stop(tok)

    Notes:
    - For lowest overhead, prefer integer key ids via key_id().
    - Results are stored as raw nanoseconds in-memory.
    """

    __slots__ = ("_key_map", "_next_key", "_times_ns")

    def __init__(self) -> None:
        self._key_map: Dict[str, int] = {}
        self._next_key: int = 0
        self._times_ns: Dict[Key, List[int]] = {}

    def key_id(self, name: str) -> int:
        """Return a stable integer id for a key name.

        Use this once up-front, then pass the id (int) to start().
        """

        try:
            return self._key_map[name]
        except KeyError:
            kid = self._next_key
            self._next_key += 1
            self._key_map[name] = kid
            return kid

    def start(self, key: Key) -> Token:
        """Start timing and return a Token."""

        return Token(key=key, t0_ns=time.perf_counter_ns())

    def stop(self, token: Token) -> int:
        """Stop timing for a given token and record the elapsed time.

        Returns:
            elapsed_ns: int
        """

        dt = time.perf_counter_ns() - token.t0_ns
        self._times_ns.setdefault(token.key, []).append(dt)
        return dt

    def clear(self, key: Optional[Key] = None) -> None:
        """Clear recorded measurements."""

        if key is None:
            self._times_ns.clear()
        else:
            self._times_ns.pop(key, None)

    @property
    def times_ns(self) -> Dict[Key, List[int]]:
        """Raw nanosecond timings (no rounding)."""

        # Return a shallow copy to avoid accidental external mutation.
        return {k: list(v) for k, v in self._times_ns.items()}

    def times_s(self, *, precision: Optional[int] = None) -> Dict[Key, List[float]]:
        """Return timings in seconds.

        Args:
            precision: if set, round to this many decimal places.
        """

        out: Dict[Key, List[float]] = {}
        for k, arr in self._times_ns.items():
            vals = [ns / 1e9 for ns in arr]
            if precision is not None:
                vals = [round(x, precision) for x in vals]
            out[k] = vals
        return out

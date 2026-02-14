"""timekid.fast

A low-overhead timing API.

This module provides a stable Python-facing API (`FastTimer`, `Token`) with an
optional native accelerator implemented in Rust.

Design goals:
- Minimize Python overhead in the hot path.
- Store raw integer nanoseconds internally; convert/round only at reporting time.
- Prefer a fast token-based start/stop API.

Backend selection:
- If `timekid._fast` (native extension) is importable, `FastTimer` and `Token`
  are provided by that extension.
- Otherwise, a pure-Python fallback is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Union
import time

Key = Union[str, int]


# -------------------------
# Pure-Python fallback
# -------------------------

@dataclass(frozen=True, slots=True)
class _PyToken:
    """Opaque handle returned by start() (pure Python fallback)."""

    key: Key
    t0_ns: int


class _PyFastTimer:
    """Low-overhead timer intended for *lots* of measurements (pure Python).

    For lowest overhead, prefer integer key ids via key_id().
    """

    __slots__ = ("_key_map", "_next_key", "_times_ns")

    def __init__(self) -> None:
        self._key_map: Dict[str, int] = {}
        self._next_key: int = 0
        self._times_ns: Dict[Key, List[int]] = {}

    def key_id(self, name: str) -> int:
        try:
            return self._key_map[name]
        except KeyError:
            kid = self._next_key
            self._next_key += 1
            self._key_map[name] = kid
            return kid

    def start(self, key: Key) -> _PyToken:
        return _PyToken(key=key, t0_ns=time.perf_counter_ns())

    def stop(self, token: _PyToken) -> int:
        dt = time.perf_counter_ns() - token.t0_ns
        self._times_ns.setdefault(token.key, []).append(dt)
        return dt

    def clear(self, key: Optional[Key] = None) -> None:
        if key is None:
            self._times_ns.clear()
        else:
            self._times_ns.pop(key, None)

    @property
    def times_ns(self) -> Dict[Key, List[int]]:
        return {k: list(v) for k, v in self._times_ns.items()}

    def times_s(self, *, precision: Optional[int] = None) -> Dict[Key, List[float]]:
        out: Dict[Key, List[float]] = {}
        for k, arr in self._times_ns.items():
            vals = [ns / 1e9 for ns in arr]
            if precision is not None:
                vals = [round(x, precision) for x in vals]
            out[k] = vals
        return out


# -------------------------
# Backend selection
# -------------------------

try:
    # Native backend (Rust via PyO3)
    from timekid._fast import FastTimer as FastTimer  # type: ignore
    from timekid._fast import Token as Token  # type: ignore
except Exception:  # pragma: no cover
    FastTimer = _PyFastTimer  # type: ignore
    Token = _PyToken  # type: ignore

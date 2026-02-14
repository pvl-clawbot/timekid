"""timekid - High-precision timing and profiling library for Python."""

from .timer import Timer, StopWatch, TimerContext, Status
from .fast import FastTimer, Token

__all__ = [
    'Timer', 'StopWatch', 'TimerContext', 'Status',
    'FastTimer', 'Token',
]
__version__ = '0.1.0'
__author__ = 'Peter Vestereng Larsen'

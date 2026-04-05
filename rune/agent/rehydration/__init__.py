"""Adaptive Context Rehydration — automatically surface compacted context.

Public API for the rehydration subsystem.  The agent loop imports from
here; internal modules should not be imported directly by external callers.
"""

from rune.agent.rehydration.protocols import (
    CompactedRecord,
    LoopStateView,
    RehydrationDecision,
    SignalReading,
)
from rune.agent.rehydration.recorder import CompactionRecorder
from rune.agent.rehydration.retrieval import (
    RehydrationResult,
    format_injection,
    rehydrate,
)
from rune.agent.rehydration.signals import Signal
from rune.agent.rehydration.trigger import RehydrationTrigger

__all__ = [
    "CompactedRecord",
    "CompactionRecorder",
    "LoopStateView",
    "RehydrationDecision",
    "RehydrationResult",
    "RehydrationTrigger",
    "Signal",
    "SignalReading",
    "format_injection",
    "rehydrate",
]

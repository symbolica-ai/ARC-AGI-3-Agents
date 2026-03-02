from dataclasses import dataclass
from typing import Literal

from .frame import DiffRegion, Frame
from .memories import Memories, Memory, MemoryQueryError

__all__ = [
    "FinishStatus",
    "Frame",
    "DiffRegion",
    "Memories",
    "Memory",
    "MemoryQueryError",
]


@dataclass(slots=True)
class FinishStatus:
    """
    Status of the game when you are done playing.

    Properties:
        status: Did you win or lose?
        reason: Reason for the status.
        levels_completed: Tuple of levels completed and total levels.
    """

    status: Literal["win", "lose"]
    reason: str
    levels_completed: tuple[int, int]

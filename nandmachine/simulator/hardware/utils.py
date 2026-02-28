from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Generic

from Desim import Event

from nandmachine.commands.macro import MacroOp

T = TypeVar('T')

@dataclass
class DepSlot(Generic[T]):

    # Used by hardware components to maintain dependencies between instructions
    payload: T

    is_finished: bool = False
    finish_event: Event = field(default_factory=Event)

    input_slots: list[DepSlot[T]] = field(default_factory=list,init=False)



    
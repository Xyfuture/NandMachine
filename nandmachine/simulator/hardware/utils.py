from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from Desim import Event

from nandmachine.commands.macro import MacroOp


@dataclass 
class MarcoOpSlot:
    macro_op:MacroOp

    is_finished:bool = False
    finish_event:Event = Event()

    input_slots:Optional[list[MarcoOpSlot]] = None 



    
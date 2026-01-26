from __future__ import annotations
from dataclasses import dataclass
from Desim.Core import Event
from transformers import Optional
from nandmachine.commands.macro import MacroOp


@dataclass
class HwOp:
    macro_op:MacroOp
    is_finished:bool = False
    finish_event:Event = Event()

    input_ops:Optional[list[HwOp]] = None 



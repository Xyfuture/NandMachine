from Desim.Core import SimModule
from nandmachine.commands.macro import MacroOp, SramPrefetch
from nandmachine.commands.micro import HwOp



class Accelerator(SimModule):
    def __init__(self):
        super().__init__()

        self.prologue_commands:list[MacroOp] = None 

        self.prefetch_command_queue:list[HwOp] = []
        self.normal_command_queue:list[HwOp] = []

        pass 


    def load_commands(self,prologue_commands:list[MacroOp],commands:list[MacroOp]):
        # Build command queues with dependency chain

        self.prologue_commands = prologue_commands
        self.prefetch_command_queue = []
        self.normal_command_queue = []

        prev_hw_op: HwOp = None

        for macro_op in commands:
            hw_op = HwOp(macro_op=macro_op)

            # Link to previous op (head-to-tail chain)
            if prev_hw_op is not None:
                hw_op.input_ops = [prev_hw_op]

            # Add to appropriate queue
            if isinstance(macro_op, SramPrefetch):
                self.prefetch_command_queue.append(hw_op)
            else:
                self.normal_command_queue.append(hw_op)

            prev_hw_op = hw_op



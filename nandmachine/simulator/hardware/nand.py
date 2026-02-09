from collections import deque
from Desim.Core import SimModule,Event,SimTime
from Desim import EventQueue

from nandmachine.simulator.hardware.utils import MarcoOpSlot



class NandController(SimModule):
    def __init__(self):
        super().__init__()

        self.waiting_request_queue:deque[MarcoOpSlot] = deque()
        # self.runing_request_queue:

        self.core_event_queue:EventQueue = EventQueue()
        

    def request_handler(self,command:MarcoOpSlot):
        self.waiting_request_queue.append(command)

        self.core_event_queue.next_notify(SimTime(1))

    
    def process(self):
        while True:
            SimModule.wait(self.core_event_queue.event)

            # 处理新请求
            while self.waiting_request_queue:
                cur_request = self.waiting_request_queue.popleft()

                # 执行时间



                cur_request.finish_event.notify()


            pass 
            




class NandSimCore:
    def __init__(self) -> None:
        
        pass 


    def process_request(self):
        pass 
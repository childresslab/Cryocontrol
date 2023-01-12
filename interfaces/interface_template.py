from logging import getLogger
log = getLogger(__name__)

from apis import rdpg
from typing import Union

dpg = rdpg.dpg

class Interface():
    def __init__(self):
        self.controls = []
        self.params = []
        self.parent = None
        self.gui_exists = False

    def makeGUI(self,parent:Union[str,int]) -> None:
        self.parent = parent
        raise NotImplementedError("This should be overloaded.")

    def initialize(self) -> None:
        raise NotImplementedError("This should be overloaded.")

    def set_params(self,state:bool) -> None:
        if state:
            for param in self.params:
                log.debug(f"Enabling {param}")
                dpg.enable_item(param)
        else:
            for param in self.params:
                log.debug(f"Disabling {param}")
                dpg.disable_item(param)

    def set_controls(self,state:bool,ignore:Union[list[str],str]=None) -> None:
        if isinstance(ignore,str):
            ignore = [ignore]
        if state:
            for control in self.controls:
                if control not in ignore:
                    log.debug(f"Enabling {control}")
                    dpg.enable_item(control)
        else:
            for control in self.controls:
                if control not in ignore:
                    log.debug(f"Disabling {control}")
                    dpg.disable_item(control)
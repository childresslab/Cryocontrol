from . import NKTP_DLL as nktpdll
import logging
log = logging.getLogger(__name__)

class SuperK():

    def __init__(self, port='COM4'):
        self.port = port
        self.mod_addr = 1
    
    def writeU8(self,reg,value,index=-1):
        result = nktpdll.registerWriteU8(self.port,self.mod_addr,reg,value,index)
        log.debug(f"Writing Register: {nktpdll.RegisterResultTypes(result)}")
    
    def readU8(self,reg,index=-1):
        result, value = nktpdll.registerReadU8(self.port,self.mod_addr,reg,index)
        log.debug(f"Reading Register: {nktpdll.RegisterResultTypes(result)}")
        return value

    def writeU32(self,reg,value,index=-1):
        result = nktpdll.registerWriteU32(self.port,self.mod_addr,reg,value,index)
        log.debug(f"Writing Register: {nktpdll.RegisterResultTypes(result)}")
    
    def readU32(self,reg,index=-1):
        result, value = nktpdll.registerReadU32(self.port,self.mod_addr,reg,index)
        log.debug(f"Reading Register: {nktpdll.RegisterResultTypes(result)}")
        return value

    def max_pulse_rate(self):
        return self.readU32(0x36)

    def initialize(self):
        result = nktpdll.openPorts(self.port,0,0)
        log.debug(f'Opening Port:{nktpdll.PortResultTypes(result)}')

    def deinitialize(self):
        result = nktpdll.closePorts(self.port)
        log.debug(f'Closing Port:{nktpdll.PortResultTypes(result)}')

    def turn_on(self):
        self.writeU8(0x30,1,-1)

    def turn_off(self):
        self.writeU8(0x30,0,-1)

    def read_power(self):
        return self.readU8(0x30,-1)

    def set_power(self, value):
        if value < 0 or value > 100:
            raise ValueError(f'Power: {value} outside range [0,100].')
        self.writeU8(0x3E, value, -1)

    def get_power(self):
        return self.readU8(0x7A)

    def set_rep_rate(self, value):
        max_rep = self.max_pulse_rate()
        if value < 1 or value > max_rep:
            raise ValueError(f'Rep Rate: {value} outside range [1,max_rep].')

    def get_rep_rate(self):
        return self.readU32(0x71)

    # TODO
    # def get_interlock(self):
    #     return 

    # def reset_interlock(self):
    
    # def get_pulse_over_run(self):

from this import d
import numpy as np
from logging import getLogger
from pathlib import Path
from time import sleep
from warnings import warn
_log = getLogger(__name__)

# Serial commands with clear names
# There are more, but these should be all we need
commands = { "acceleration" : "ACC",
             "deceleration" : "DEC",
             "read_errors" : "ERR",
             "dead_band" : "DBD",
             "estop" : "EST",
             "set_feedback" : "FBK",
             "set_motor" : "MOT",
             "status"   : "STA",
             "move_abs" : "MVA",
             "move_rel" : "MVR",
             "position" : "POS",
             "set_pid" : "PID",
             "set_resolution" : "REZ",
             "reset" : "RST",
             "stop" : "STP",
             "soft_lower" : "TLN",
             "soft_upper" : "TLP",
             "velocity" : "VEL",
             "zero" : "ZRO"
}

default_config = { "_port"           : 3,       #Serial Port COM#
                   "_velocity"       : 0.1,     #mm/s
                   "_acceleration"   : 25.0,    #mm/s^2
                   "_deceleration"   : 1.0,     #mm/s^2
                   "_soft_lower"     : -4000.0, #um
                   "_soft_upper"     : 4000.0,  #um
                   "_resolution_stp" : 8000.0,  #steps/um
                   "_resolution_enc" : 0.01,    #um/cnt
                   "_max_move"       : 100,     #um
                   "_axis"           : 1        #Axis to use, nominally 1
}

class DummyInterface():
    def __init__(self):
        self.dict = {}
        self.buffer = ""

    def write(self,msg):
        components = msg.split(' ')
        key = components[0][:-1]
        if components[0][-1] == "?":
            self.buffer = str(self.dict.get(key,"0"))
        else:
            self.dict[key] = components[1]
    
    def read(self):
        msg = self.buffer
        self.buffer = ""
        return msg

    def query(self,msg):
        key = msg[:-1]
        msg = str(self.dict.get(key,"0"))
        return msg

class ObjValueError(Exception):
    pass

class DummyObjective():
    def __init__(self,config_dic:dict[str,any] = default_config,**kwargs) -> None:
        # Modify config with parameters
        for key, value in config_dic.items():
            if key not in default_config.keys():
                print("Warning, unmatched config option: '%s' in config dictionary." % key)
                default_config[key] = value
        for key, value in kwargs.items():
            if key not in default_config.keys():
                print("Warning, unmatched config option: '%s' from kwargs." % key)
                default_config[key] = value
        for key, value in default_config.items():
            setattr(self,key,value)
        self.instr = None
        self.commands = commands
        self.set_point = None
        self.initialized=False
        
    def __repr__(self):
        if self.instr is None:
            return f"Stage Uninitialized"
        else:
            position = self.position
            set_point = self.set_point
            status = self.status
            model = self.instr.model
            return f"Stage Initialized\n{model}\nSet Position: {set_point:.2f}um\nRead Position: {position:.2f}um\nStatus: {status}"

    def query(self,msg:str) -> str:
        if self.instr is None:
            raise RuntimeError("Objective not Initialized")
        else:
            #Strip leading # and any whitespace
            return self.instr.query(f"{self._axis}{msg}")[1:].strip()

    def write(self,msg:str) -> None:
        if self.instr is None:
            raise RuntimeError("Objective not Initialized")
        else:
            self.instr.write(f"{self._axis}{msg}")

    def read(self) -> str:
        if self.instr is None:
            raise RuntimeError("Objective not Initialized")
        else:
            #Strip leading # and any whitespace
            return self.instr.read()[1:].strip()
        
    @property
    def max_move(self) -> float:
        return self._max_move
    @max_move.setter
    def max_move(self, value:float):
        if value < 0:
            raise ObjValueError("Max move must be a positive number")
        self._max_move = value

    @property
    def accel(self) -> float:
        """The accel property."""
        return float(self.query(commands['acceleration']))
    @accel.setter
    def accel(self, value:float) -> float:
        self._accel = value
        self.write(f"{commands['acceleration']} {value}")

    @property
    def decel(self) -> float:
        """The decel property."""
        return float(self.query(commands['deceleration']))

    @decel.setter
    def decel(self, value:float):
        self._decel = value
        self.write(f"{commands['deceleration']} {value}")

    @property
    def velocity(self) -> float:
        """The velocity property."""
        return float(self.query(commands['velocity']))
    @velocity.setter
    def velocity(self, value:float):
        self._velocity = value
        self.write(f"{commands['velocity']} {value}")

    @property
    def dead_band(self) -> int:
        """The dead_band property."""
        res = self.query(commands['dead_band']).split(',')
        return int(res[0]),float(res[1])

    @dead_band.setter
    def dead_band(self, steps:int, time:float):
        self._dead_band_steps = steps
        self._dead_band_time = time
        self.write(f"{commands['dead_band']} {steps},{time}")

    @property
    def feedback(self) -> bool:
        """The feedback property."""
        value = int(self.query(commands['set_feedback']))
        if value == 0:
            return False
        elif value == 3:
            return True
        else:
            print("Unknown feedback value.")
            return None

    @feedback.setter
    def feedback(self, value:bool):
        self._feedback = value
        if value:
            value = 3
        else:
            value = 0
        self.write(f"{commands['set_feedback']} {value}")

    @property
    def motor_power(self) -> bool:
        """The motor_power property."""
        return bool(self.query(commands['set_motor']))
    @motor_power.setter
    def motor_power(self, value:bool):
        self._motor_power = value
        self.write(f"{commands['set_motor']} {int(value)}")

    @property
    def enc_position(self) -> float:
        """The position property."""
        return float(self.query(commands['position']).split(',')[0]) * 1000 # convert to um from mm
    @property
    def position(self) -> float:
        """The position property."""
        return float(self.query(commands['position']).split(',')[1]) * 1000 # convert to um from mm

    @property
    def soft_lower(self) -> float:
        """The lower_limit property."""
        return float(self.query(commands['soft_lower'])) * 1000
    @soft_lower.setter
    def soft_lower(self, value:float):
        if self._soft_upper < value:
            raise ObjValueError(f"Lower limit must be less than upper limit = {self.upper_limit}.")
        self._soft_lower = value
        self.write(f"{commands['soft_lower']} {value/1000}")

    @property
    def soft_upper(self) -> float:
        """The upper_limit property."""
        return float(self.query(commands['soft_upper'])) * 1000
    @soft_upper.setter
    def soft_upper(self, value:float):
        if value < self._soft_lower:
            raise ObjValueError(f"Upper limit must be greater than lower limit = {self.lower_limit}.")
        self._soft_upper = value
        self.write(f"{commands['soft_upper']} {value/1000}")
        
    @property
    def status(self) -> dict[str,bool]:
        """The status property"""
        byte =  int(self.query(commands['status']))
        status_dict = {'error' : bool(byte&128),
                       'accel'  : bool(byte&64),
                       'const'  : bool(byte&32),
                       'decel'  : bool(byte&16),
                       'idle'   : bool(byte&8),
                       'prgrm'  : bool(byte&4),
                       '+lim'   : bool(byte&2),
                       '-lim'   : bool(byte&1)
                       }
        return status_dict

    @property
    def errors(self) -> list[str]:
        return self.query(commands['read_errors']).split(',')

    def stop(self):
        self.write(commands['stop'])

    def estop(self):
        self.write(commands['estop'])

    def initialize(self):
        self.instr = DummyInterface()
        self.feedback = 1
        self.set_point = self.position
        self.initialized=True

    def deinitialize(self, maintain_feedback = False):
        #Disable closed loop feedback
        if not maintain_feedback:
            self.feedback = 0
        self.instr = None
        self.set_point = None
        self.initialized = False

    def move_abs(self,position,monitor=True,monitor_callback=None):
        delta = self.position - position
        if np.abs(delta) > self._max_move:
            raise ObjValueError(f"Change in position {delta} greater than max = {self.max_move}")
        elif not (self._soft_lower < position < self._soft_upper):
            raise ObjValueError(f"New position {position} outside soft limits [{self._soft_lower},{self._soft_upper}]")
        self.write(f"{commands['move_abs']} {position/1000}")
        self.set_point = position
        if monitor:
            self.monitor_move(monitor_callback)

    def move_rel(self,distance,monitor=True,monitor_callback=None):
        position = self.position + distance
        if np.abs(distance) > self._max_move:
            raise ObjValueError(f"Change in position {distance} greater than max = {self.max_move}")
        elif not (self._soft_lower < position < self._soft_upper):
            raise ObjValueError(f"New position {position} outside soft limits [{self._soft_lower},{self._soft_upper}]")
        self.write(f"{commands['move_rel']} {distance/1000}")
        self.set_point = position
        if monitor:
            self.monitor_move(monitor_callback)

    def move_up(self,distance,monitor=True,monitor_callback=None):
        if distance < 0:
            raise ObjValueError("Move up distance must be positive")
        self.move_rel(-distance,monitor,monitor_callback) #Negative values is going up, absolutely certain, do not change.
    
    def move_down(self,distance,monitor=True,monitor_callback=None):
        if distance < 0:
            raise ObjValueError("Move up distance must be positive")
        self.move_rel(distance,monitor,monitor_callback) #Negative values is going up, absolutely certain, do not change.

    def move_abs(self,position,monitor=True,monitor_callback=None):
        distance = position - self.position
        if np.abs(distance) > self._max_move:
            raise ObjValueError(f"Change in position {distance} greater than max = {self.max_move}")
        elif not (self._soft_lower < self.position < self._soft_upper):
            raise ObjValueError(f"New position {position} outside soft limits [{self._soft_lower},{self._soft_upper}]")
        self.write(f"{commands['move_abs']} {position/1000}")
        self.set_point = position
        if monitor:
            self.monitor_move(monitor_callback)

    def monitor_move(self, callback=None):
        # Setup a default callback, also servers as example
        def default_callback(status,position,setpoint):
            if status['idle']:
                msg = "at position."
            elif status['accel']:
                msg = "accelerating."
            elif status['decel']:
                msg = "decelerating."
            elif status['const']:
                msg = 'at constant speed.'
            else:
                msg = 'slipping.'
            if status['error']:
                msg += "\n\tError detected, aborting."
            print(f"At {position:.3f}um, target is {setpoint:.3f}um. Stage is {msg}")
            return status['error']
        # Set default callback if none given
        if callback is None:
            callback = default_callback

        # Setup first iteration of loop
        status = self.status
        set_point = self.set_point
        while not status['idle']:
            # Throw in try catch to allow for keyboard interrupts of motion
            try:
                # Run the callback and check the response.
                abort = callback(status,self.position,self.set_point)
                if abort:
                    break
                # Update status for next iteration
                status = self.status
            # If keyboard intterupt (CTRL+C) sent, act as if aborting
            except KeyboardInterrupt:
                abort = True
                break

        if abort:
            # Loop through different levels of stopping.
            self.stop()
            if not self.status['idle']:
                warn("Normal stop didn't work, trying emergency stop!")
                self.estop()
                if not self.status['idle']:
                    warn("~~~~Emergency stop didn't work, pull the plug!!!!!!~~~~")
        # run callback one more time to ensure we get the final position printed.
        else:
            callback(self.status,self.position,self.set_point)
        pass

    def zero(self):
        self.write(f"{commands['zero']}")


def parse_command(command:str) -> list[str]:
    return list(map(lambda s: s.strip(),command.split(";")))

def get_id(inst) -> str:
    inst.write("1VER?")
    sleep(0.5)
    identity = inst.read()
    return identity


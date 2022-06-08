import pyvisa
from enum import Enum
import signal_generator

from time import sleep

class Waveform(Enum):
    SIN = 0
    SQU = 1
    RAMP = 2
    PULSE = 3
    ARB = 4

_wfm_props = {Waveform.SIN.name   : [],
              Waveform.SQU.name    : ["duty"],
              Waveform.RAMP.name   : ["symmetry"],
              Waveform.PULSE.name  : [],
              Waveform.ARB.name    : []
}

_prop_commands = {"frequency" : "frequency",
                  "amplitude" : "voltage",
                  "offset" : "voltage:offset",
                  "phase" : "phase",
                  "duty" : "function:square:dcycle",
                  "symmetry" : "function:ramp:symmetry"}

_debug = True

class RigolDG1022U(signal_generator.SignalGeneratorVisaInterface):
    def __init__(self, visa_id):
        rm = pyvisa.ResourceManager()
        self.inst = rm.open_resource(visa_id)
        self.inst.query_delay = 0.1
    
    def log_write(self, msg):
        if _debug:
            print(msg)
        self.inst.write(msg)
        sleep(self.inst.query_delay)

    def log_query(self, msg):
        if _debug:
            print(msg)
        res = self.inst.query(msg)
        print("\t" + res)
        return res

    def off(self, chn=1):
        try:
            if chn==1:
                self.log_write("output off")
            elif chn==2:
                self.log_write("output:ch2 off")
        except:
            return 1
        return 0

    def on(self, chn=1):
        try:
            if chn==1:
                self.log_write("output on")
            elif chn==2:
                self.log_write("output:ch2 on")
        except:
            return 1
        return 0
            
    def get_status(self):
        ch1 = _process_status(self.log_query("apply?"))
        ch1on = self.log_query("output?").strip("\n\r")
        ch1["output"] = ch1on

        ch2 = _process_status(self.log_query("apply:ch2?"))
        ch2on = self.log_query("output:ch2?").strip("\n\r")
        ch2["output"] = ch2on

        return [ch1, ch2]

    def set_waveform(self,wfm,chn=1):
        wfmname = wfm.name
        suffix = "" if chn==1 else ":CH2"
        try:
            self.log_write("function" + suffix + " " + wfmname)
        except:
            return 1
        return 0
    
    def set_property(self,prop,val,chn=1):
        if prop not in self.get_properties(chn):
            raise ValueError("Property does not apply for current waveform")
        suffix = "" if chn==1 else ":CH2"
        cmd = _prop_commands[prop]
        self.log_write(cmd + suffix + " " + str(val))
        return 0

    def get_properties(self,chn=1):
        wfm = self.get_status()[chn-1]["waveform"]
        default = ["frequency", "amplitude", "offset", "phase"]
        return _wfm_props[wfm] + default

    def get_waveforms(self):
        return [wfm.name for wfm in Waveform]

def _process_status(msg):
    fields = msg.split(':')[1].strip('"\n').split(",")
    return {"waveform":fields[0],
            "frequency":fields[1],
            "amplitude":fields[2],
            "offset":fields[3]}

if __name__=="__main__":
    name = "USB0::0x0400::0x09C4::DG1G150300137::INSTR"
    rigol = RigolDG1022U(name)

    rigol.set_waveform(Waveform(0),1)
    rigol.set_property("frequency",100)
    rigol.set_property("amplitude",1)
    rigol.set_property("offset",0)
    rigol.on()

    rigol.set_waveform(Waveform(1),2)
    rigol.set_property("frequency",100,2)
    rigol.set_property("amplitude",1,2)
    rigol.set_property("offset",0,2)
    rigol.on(2)

    sleep(5)
    rigol.set_waveform(Waveform(1),1)
    rigol.set_property("frequency",100,1)
    rigol.set_property("amplitude",1,1)
    rigol.set_property("offset",0,1)

    rigol.set_waveform(Waveform(0),2)
    rigol.set_property("frequency",100,2)
    rigol.set_property("amplitude",1,2)
    rigol.set_property("offset",0,2)
    sleep(5)

    for f in range(100,1000,10):
            rigol.set_property("frequency",f,1)
            sleep(0.4)

    rigol.off(2)
    rigol.off(1)
    rigol.inst.close()
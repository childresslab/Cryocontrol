from enum import Enum
from abc import abstractmethod

class SignalGeneratorVisaInterface():

    @abstractmethod
    def off(self):
        pass

    @abstractmethod
    def on(self):
        pass

    @abstractmethod
    def get_status(self):
        pass

    @abstractmethod
    def set_waveform(self,wfm,chn):
        pass

    @abstractmethod
    def set_property(self,prop,val,chn):
       pass

    @abstractmethod
    def get_properties(self,chn):
        pass

    @abstractmethod
    def get_waveforms(self):
        pass
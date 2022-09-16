# -*- coding: utf-8 -*-
"""
@author: Rigel Zifkin <rydgel.code@gmail.com>
"""

import numpy as np
from time import time
from time import sleep

class FPGAValueError(Exception):
    pass

# Functions for converting between fpga bits and volts
def _volts_to_bits(voltage, vmax, bit_depth):
    if np.abs(voltage) > vmax:
        raise FPGAValueError("Given voltage outside of given max voltage.")
    if voltage == vmax:
        return 2**(bit_depth - 1) - 1
    return int(np.round(voltage/vmax * (2**(bit_depth-1)-1) + (0.5 if voltage > 0 else -0.5)))

def _bits_to_volts(bits, vmax, bit_depth):
    if np.abs(bits) > 2**(bit_depth-1):
        raise FPGAValueError("Given int outside binary depth range.")
    if bits == -2**(bit_depth-1):
        return -vmax
    return (bits - (0.5 * np.sign(bits))) / (2**(bit_depth-1) - 1) * vmax

def _within(value,vmin,vmax):
    """Check that a value is within a certain range.     
       If you have a range array `bounds = [vmin,vmax]` you can simply call `_within(value, *bounds)`

    Parameters
    ----------
    value : number
        The value which is being checked
    vmin : number
        The minimum acceptable value
    vmax : number
        The maximum acceptable value

    Returns
    -------
    boolean
        weather the value is within the given bounds.
    """
    if vmin > vmax:
        raise FPGAValueError("Range minimum must be less than range maximum.")
    return (value >= vmin and value <= vmax)

def _multi_gaussian(xs,mus,sigmas):
    return np.exp(-0.5 * (xs-mus)**2 / sigmas**2)

class DummyData():
    def __init__(self,values):
        self.data = values
        self.elements_remaining = 1

class DummyFIFO():
    def __init__(self,fpga):
        self.fpga = fpga
        self._num_points = 500
        self._position_range = [[-5,5], [-5,5], [-2,0],[-0.0005,0.0005]]
        # Taken from qudi
        # put randomly distributed NVs in the scanner, first the x,y scan
        self._x_positions = np.random.uniform(self._position_range[0][0],self._position_range[0][1],self._num_points)
        self._y_positions = np.random.uniform(self._position_range[1][0],self._position_range[1][1],self._num_points)
        self._z_positions = np.random.uniform(self._position_range[2][0],self._position_range[2][1],self._num_points)
        self._cav_positions = np.random.uniform(self._position_range[3][0],self._position_range[3][1],self._num_points)
        self._amplitudes = np.random.uniform(1E5,5E5,self._num_points)
        self._x_sigmas = np.random.uniform(0.2,0.4,self._num_points)
        self._y_sigmas = np.random.uniform(0.2,0.4,self._num_points)
        self._z_sigmas = np.random.uniform(0.5,2,self._num_points)
        self._cav_sigmas = np.random.uniform(0.00001,0.00004,self._num_points)

    def stop(self):
        pass

    def reset(self):
        pass

    def clear(self):
        pass

    def write(self,data,timeout):
        pass

    def read(self,n):
        galvo_pos = self.fpga.get_galvo()
        jpe_pos = self.fpga.get_jpe_pzs()
        xpos = galvo_pos[0]*117 - jpe_pos[1]*20*14/1000
        ypos = galvo_pos[1]*117 - jpe_pos[0]*20*14/1000
        zpos = jpe_pos[2]*20*14/1000
        cav_pos = self.fpga.get_cavity()[0]
        zcav = cav_pos * 20 * 0.025 / 1000
        value = np.random.poisson(80 * self.fpga.count_time)
        values = (_multi_gaussian(xpos,self._x_positions,self._x_sigmas) * 
                 _multi_gaussian(ypos,self._y_positions,self._y_sigmas) *
                 _multi_gaussian(zpos,self._z_positions,self._z_sigmas) *
                 _multi_gaussian(zcav,self._cav_positions,self._cav_sigmas))
        value += np.random.poisson(np.sum(self._amplitudes*values) * self.fpga.count_time)
        data = DummyData(np.tile(value,n))
        dtime = (self.fpga.wait_after_ao + self.fpga.count_time)
        if dtime > 10:
            sleep(dtime/1000)
        else:
            sleep(0)
        return data

    def configure(self,size):
        return size

class DummyRegister():
    def __init__(self,val=0):
        self.val = val
    def read(self):
        return self.val
    def write(self,val):
        self.val = val
    
class DummySession():
    def __init__(self,fpga,vmax,bit_depth):
        self.registers = {'AO0' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AO1' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AO2' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AO3' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AO4' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AO5' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AO6' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AO7' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AO8' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AO9' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AI0' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AI1' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AI2' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AI3' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AI4' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AI5' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AI6' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AI7' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AI8' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'AI9' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'Counting Mode' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'H toT Size' : DummyRegister(_volts_to_bits(0,vmax,bit_depth)),
                            'Wait after AO set (us)' : DummyRegister(_volts_to_bits(10,vmax,bit_depth)),
                            'Start FPGA 1' : DummyRegister(_volts_to_bits(0,vmax,bit_depth))}

        self.fifos = {'Host to Target DMA' : DummyFIFO(fpga),
                      'Target to Host DMA' : DummyFIFO(fpga)}

class DummyNiFPGA():
    """ Class that handles the NI FPGA
    """
    _n_AI  = 8
    _n_AO = 8
    _n_DIO = 8
    _vmax = 10
    _bit_depth = 16

    def __init__(self, **kwargs):
        self._max_voltage_range = np.array([-10.0,10.0],dtype=np.double)
        self._clock_frequency = 120E6
        self._voltage_ranges = np.tile(self._max_voltage_range, [self._n_AO,1])

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        # Open the session with the FPGA card
        print("Opening fpga session")
        try:
            self._fpga = DummySession(self,self._vmax,self._bit_depth)
        except:
            print("Couldn't create fpga session")
            raise

    def reset_hardware(self):
        """ Resets the hardware, so the connection is lost and other programs
            can access it.

        @return int: error code (0:OK, -1:error)
        """
        try:
            for fifo in self._fpga.fifos.keys():
                self._fpga.fifos[fifo].stop()
            self._fpga.reset()
            self._fpga.close()
        except:
            print("Could not close fpga device")
            raise
        return 0

    # Register Methods
    def read_register(self, name):
        try:
            return self._fpga.registers[name].read()
        except:
            raise

    def write_register(self, name, value):
        try:
            return self._fpga.registers[name].write(value)
        except:
            raise
    
    # Fifo Methods
    def read_fifo(self, name, n_elem=None):
        if n_elem is None:
            n_elem = self._fpga.fifos[name].read(0).elements_remaining
        data = np.fromiter(self._fpga.fifos[name].read(n_elem).data, dtype=np.uint32, count=n_elem)
        return data

    def write_fifo(self, name, data,timeout=5000):
        try:
            self._fpga.fifos[name].write(data,timeout)
        except:
            raise

    def set_size_fifo(self, name, size):
        return self._fpga.fifos[name].configure(size)

    def stop_fifo(self, name):
        self._fpga.fifos[name].stop()

                        #################
                        #               #
                        #   AO Methods  #
                        #               #
################################################################################
    def set_AO_range(self, index=None, vrange=None):
        """ Gets the voltage range(s) of the AO(s),
            if index is given, only returns that range, else returns a list
            of all ranges.
        """
        if vrange is None:
            vrange = self._max_voltage_range
        vrange = np.asarray(vrange)
        if not np.isscalar(vrange[0]):
            print('Found non numerical value in range.')
            return -1

        if len(vrange) != 2:
            print('Given range should have dimension 2, but has '
                    '{0:d} instead.'.format(len(vrange)))
            return -1

        if vrange[0]>vrange[1]:
            print('Given range limit {0:d} has the wrong '
                    'order.'.format(vrange))
            return -1

        if vrange[0] < self._max_voltage_range[0]:
           print('Lower limit is below -10V')
        if vrange[1] > self._max_voltage_range[1]:
           print('Higher limit is above -10V')

        self._voltage_ranges[index] = vrange

        return 0

    def get_AO_range(self, index=None):
        if index is None:
            return self._voltage_ranges
        return self._voltage_ranges[index]

    def set_AO_volts(self, chns: float, vs: float):
        """Move galvo to x, y.

        @param float x: voltage in x-direction
        @param float y: voltage in y-direction

        @return int: error code (0:OK, -1:error)
        """
        vranges = self.get_AO_range(chns)
        for i,(v,vrange) in enumerate(zip(vs,vranges)):
            if not _within(v,*vrange):
                raise FPGAValueError(f"Given voltage {v} outside range {vrange} on chn {chns[i]}")
        for chn,v in zip(chns,vs):
            try:
                self._fpga.registers[f"AO{chn}"].write(_volts_to_bits(v,self._vmax,self._bit_depth))
            except:
                raise

    def get_AO_volts(self,chns=None):
        """ Get the current position of the scanner hardware.

        @return float[n]: current position in (z1,z2,z3).
        """
        if chns is None:
            chns = list(range(self._n_AO))
        try:
            volts = [_bits_to_volts(self._fpga.registers[f"AO{chn}"].read(),
                                    self._vmax, self._bit_depth) 
                     for chn in chns]
        except:
            raise

        return volts

                        ##################
                        #                #
                        #    AI Methods  #
                        #                #
################################################################################
    def get_AI_volts(self,chns=None):
        """ Get the current position of the scanner hardware.

        @return float[n]: current position in (z1,z3,z3).
        """
        if chns is None:
            chns = list(range(self._n_AI))
        try:
            volts = [_bits_to_volts(self._fpga.registers[f"AI{chn}"].read(),
                                    self._vmax, self._bit_depth) 
                        for chn in chns]
        except:
            raise

        return volts

                        ##################
                        #                #
                        #   DIO Methods  #
                        #                #
################################################################################
    def get_DIO_state(self, chns=None):
        if chns is None:
            chns = list(range(self._n_DIO))
        try:
            states = [self._fpga.registers[f"DIO{chn}"].read() for chn in chns]
        except:
            raise

        return states

    def set_DIO_state(self,channels,state):
        pass

    def toggle_DIO_state(self, channels, state):
        states = self.get_DIO_state(channels)
        on_channels = channels[np.where(states)]
        off_channels = channels[np.where(np.invert(states))]

        retvalon  = self.enable_DIO(on_channels)
        retvaloff = self.disable_DIO(off_channels)

        if retvalon != 0:
            return retvalon
        elif retvaloff != 0:
            return retvaloff
        else:
            return 0

    def enable_DIO(self, channels):
        return self.set_DIO_state(self,channels,True)

    def disable_DIO(self,channels):
        return self.set_DIO_state(self,channels,False)

    def close_fpga(self):
        """ Closes the fpga and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        try:
            for fifo in self._fpga.fifos.keys():
                self._fpga.fifos[fifo].stop()
        except:
            print("Couldn't Stop FIFOs")
            raise
        return 0
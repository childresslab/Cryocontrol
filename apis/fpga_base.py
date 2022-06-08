# -*- coding: utf-8 -*-
"""
@author: Rigel Zifkin <rydgel.code@gmail.com>
"""

import numpy as np
from nifpga.session import Session

# Functions for converting between fpga bits and volts
def _volts_to_bits(voltage, vmax, bit_depth):
    if np.abs(voltage) > vmax:
        raise ValueError("Given voltage outside of given max voltage.")
    if voltage == vmax:
        return 2**(bit_depth - 1) - 1
    return int(np.round(voltage/vmax * (2**(bit_depth-1)-1) + (0.5 if voltage > 0 else -0.5)))

def _bits_to_volts(bits, vmax, bit_depth):
    if np.abs(bits) > 2**(bit_depth-1):
        raise ValueError("Given int outside binary depth range.")
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
        raise ValueError("Range minimum must be less than range maximum.")
    return (value >= vmin and value <= vmax)

class NiFPGA():
    """ Class that handles the NI FPGA
    """
    _bitfile_path = 'X:\\DiamondCloud\\Fiber_proj_ctrl_softwares\\Cryo Control\\FPGA Bitfiles\\Pulsepattern(bet_FPGATarget_FPGAFULLV2_DX29Iv2+L+Y.lvbitx'
    _resource_num = "RIO0"
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
            self._fpga = Session(bitfile=self._bitfile_path, resource=self._resource_num)
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
        return np.fromiter(self._fpga.fifos[name].read(n_elem).data, dtype=np.uint32, count=n_elem)

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
                raise ValueError(f"Given voltage {v} outside range {vrange} on chn {chns[i]}")
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
        """try:
        self._fpga.registers['Start FPGA 1'].write(0)
        self._fpga.registers[self._x_channel].write(_volts_to_bits(x))
        self._fpga.registers[self._y_channel].write(_volts_to_bits(y))
        try:
            #self._fpga.fifos[self._counts_fifo].stop()
            self._fpga.fifos[self._pulse_pattern_fifo].stop()
            self._fpga.fifos[self._pulse_pattern_fifo].write(self._move_pattern,5000)
        except:
            print("Timed out while writing pulse pattern to FPGA")
            raise

        # Query fpga status until it's done pulsing and counting.
        self._fpga.registers['Start FPGA 1'].write(1)
        for i in range(20): # Limit to 20 loops
            if not self._fpga.registers['Start FPGA 1'].read():
                break
            else:
                time.sleep(0.5/self._clock_frequency)
        else: # This only runs if for loop finishes without break
            raise TimeoutError

        except:
        print("Could not set position on fpga device")
        raise
        return -1

        return 0"""
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
            volts = [_bits_to_volts(self._fpga.registers[f"AI{chn}"],
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
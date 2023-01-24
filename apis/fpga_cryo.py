from .import fpga_base as fb
from .pulse_generator import PulseGen
from .jpe_coord_convert import JPECoord
import numpy as np
from time import sleep, time

pg_config = {"dio_chns" : 16,
             "user_dio_chns" : 14,
             "counter_dio_chn" : 0,
             "gate_dio_chn" : 1,
             "pause_time" : 1E-6,
             "clock_rate" : 120E6,
             "max_ticks" : np.iinfo(np.uint16).max}
pulser = PulseGen(pg_config)

pz_config = {"vmax" : 0,
             "vmin" : -6.5,
             "vgain" : -20,
             "R": 6.75,
             "h": 45.1}
pz_conv = JPECoord(pz_config['R'], pz_config['h'],
                   pz_config['vmin'], pz_config['vmax'])

FPGAValueError = fb.FPGAValueError

class CryoFPGA(fb.NiFPGA):
    _pulse_pattern_fifo = 'Host to Target DMA'
    _counts_fifo = 'Target to Host DMA'
    _galvo_x = 0
    _galvo_y = 1
    _cavity_z = 2
    _green_aom = 3
    _red_aom = 4
    _jpe_uno = 5 
    _jpe_due = 6 
    _jpe_tre = 7
    _photodiode_in = 1
    _dio_array = [0 for n in range(pg_config['user_dio_chns'])]
    _max_waits = 2
    _max_wait_multiple = 60

    def __init__(self) -> None:
        super().__init__()
        self.pulse_pattern = {}
        self.pulse_fifo_data = []
        self._duration = 0
        self.wait_after_ao = 5
        self.count_time = 0

        self.set_AO_range(self._galvo_x,   [-10.0,10.0])
        self.set_AO_range(self._galvo_y,   [-10.0,10.0])
        self.set_AO_range(self._green_aom, [0.0,10.0])
        self.set_AO_range(self._red_aom,   [0.0,10.0])
        self.set_AO_range(self._jpe_uno,   [-6.5,0])
        self.set_AO_range(self._jpe_due,   [-6.5,0])
        self.set_AO_range(self._jpe_tre,   [-6.5,0])

        self.open_fpga()

    def set_jpe_pzs(self, x:float = None, y:float = None, z:float = None, write:bool=True) -> None:
        volts = [x,y,z]

        if any([v is None for v in volts]):
            current = self.get_jpe_pzs()
            for i, v in enumerate(volts):
                if v is None:
                    volts[i] = current[i]
                    
        #Fixed the mistake had a None issue because it was still taking the x,y,z instead of updated valued
        if not pz_conv.check_bounds(volts[0],volts[1],volts[2]): 
            raise FPGAValueError(f"New JPE Position is outside bounds.")

        z_volts = pz_conv.zs_from_cart(volts)
        self.set_AO_volts([self._jpe_uno, self._jpe_due, self._jpe_tre], z_volts)
        if write:
            self.write_values_to_fpga()

    def get_jpe_pzs(self) -> list[float]:
        z_volts = self.get_AO_volts([self._jpe_uno,self._jpe_due,self._jpe_tre])
        return pz_conv.cart_from_zs(z_volts)

    def set_galvo(self, x:float = None, y:float = None, write:bool=True) -> None:
        volts = [x,y]
        if any([v is None for v in volts]):
            current = self.get_galvo()
            for i, v in enumerate(volts):
                if v is None:
                    volts[i] = current[i]
        self.set_AO_volts([self._galvo_x, self._galvo_y], volts)
        if write:
            self.write_values_to_fpga()
    def get_galvo(self) -> list[float]:
        return self.get_AO_volts([self._galvo_x, self._galvo_y])

    def set_cavity(self, z:float, write:bool=True) -> None:
        self.set_AO_volts([self._cavity_z], [z])
        if write:
            self.write_values_to_fpga()

    def get_cavity(self) -> float:
        return self.get_AO_volts([self._cavity_z])

    def set_aoms(self, red:float = None, green:float = None, write:bool=True) -> None:
        volts = [red, green]
        if any([v is None for v in volts]):
            current = self.get_aoms()
            for i, v in enumerate(volts):
                if v is None:
                    volts[i] = current[i]
        self.set_AO_volts([self._red_aom, self._green_aom], volts)
        if write:
            self.write_values_to_fpga()
            
    def get_aoms(self) -> list[float]:
        return self.get_AO_volts([self._red_aom, self._green_aom])

    def get_photodiode(self) -> float:
        return self.get_AI_volts(self._photodiode_in)

    def set_dio_array(self, dio_array:list[int], write:bool=True) -> None:
        if len(dio_array) != len(self._dio_array):
            raise FPGAValueError("Invalid length of dio_array")
        self._dio_array = dio_array
        if write:
            self.write_values_to_fpga()

    def get_dio_array(self) -> list[int]:
        return self._dio_array

    def set_ao_wait(self, value : float, write:bool = True) -> None:
        self.wait_after_ao = value
        if write:
            self.write_values_to_fpga()

    def get_ao_wait(self) -> float:
        return self.wait_after_ao

    def write_values_to_fpga(self) -> None:
        self.just_count(1/pg_config['clock_rate'] * 1E3)

    def just_count(self, ms : float) -> list[float]:
        dio_array = [0,1] + self.get_dio_array()
        pulse_pattern = [{'duration' : ms, 'dio_array' : dio_array}]
        self.pulse_pattern = pulse_pattern
        self.count_time = ms * 1E-3

        return self.write_pulse_count(pulse_pattern)[0]

    def count_n_times(self, ms : float, n:int = 1000) -> list[float]:
        dio_array = [0,1] + self.get_dio_array()
        pulse_pattern = [{'duration' : ms, 'dio_array' : dio_array}]
        self.pulse_pattern = pulse_pattern
        self.count_time = ms * 1E-3
        counts = np.empty(n)
        self.prep_pulse_pattern(pulse_pattern)

        for i in range(n):
            self.write_pulse_pattern()
            counts[i] = self.pulse_and_count()[0]

        return counts

    def write_pulse_count(self, pulse_pattern : dict[str, object]) -> None:
        self.prep_pulse_pattern(pulse_pattern)
        self.write_pulse_pattern()
        return self.pulse_and_count()

    def prep_pulse_pattern(self, pulse_pattern : dict[str, object]) -> None:
        self._duration = 2/pg_config['clock_rate'] * 1E3
        for step in pulse_pattern:
            self._duration += step['duration'] * 1E-3

        fifo_data = pulser.parse_sequence(pulse_pattern,default_dio=self.get_dio_array())
        self.pulse_fifo_data = fifo_data

    def write_pulse_pattern(self) -> None:
        fifo_data = self.pulse_fifo_data

        self.write_register('Start FPGA 1', 0)

        # Attempt to set host->target size
        ht_size = self.set_size_fifo(self._pulse_pattern_fifo, len(fifo_data))
        # Let FPGA know the real size
        self.write_register('H toT Size', ht_size)
        # Set target->host size with true size
        self.set_size_fifo(self._counts_fifo, ht_size)
        self.write_register('Wait after AO set (us)', int(round(self.wait_after_ao * 1000)))
        # Set count mode to false => no averaging
        self.write_register('Counting Mode', 0)

        # Stop the fifo, flushing them
        self.stop_fifo(self._pulse_pattern_fifo)
        self.stop_fifo(self._counts_fifo)
        
        # Send the pulse pattern data to the fpga, with 5s timeout
        self.write_fifo(self._pulse_pattern_fifo, fifo_data, 5000)
        
    def pulse_and_count(self) -> list[float]:
        self.write_register('Start FPGA 1', 1)
        if self._duration > 1E-3:
            duration = self._duration
        else:
            # Computers are too imprecise in timing
            # so always wait at least 1ms
            duration = 1E-3
        # For 'slow' pulses, we can wait a reasonable time.
        if duration > 15E-3:
            sleep(duration)
            for i in range(self._max_waits):
                if not self.read_register('Start FPGA 1'):
                    break
                else:
                    sleep(duration)
            else:
                raise TimeoutError("Pulse pattern timed out.")
        # For 'fast' pulse, python can't wait a small enough amount of time (>15ms)
        # so instead we constantly poll until the fpga is done, with a timeout.
        # sleep(0) is much faster, a little under a millisecond, so adds a nice small delay
        else:
            start = time()
            while (time() - start) < (duration * self._max_wait_multiple):
                if not self.read_register('Start FPGA 1'):
                    break
                else:
                    sleep(0)
            else:
                raise TimeoutError("Pulse pattern timed out.")

        return self.get_counts()

    def get_counts(self, per_second:bool = True) -> list[float]:
        if per_second:
            return self.read_fifo(self._counts_fifo)/self.count_time
        else:
            return self.read_fifo(self._counts_fifo)/1.0


    
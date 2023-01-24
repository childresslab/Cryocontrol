import numpy as np
import itertools
config = {"dio_chns" : 16,
          "user_dio_chns" : 14,
          "counter_dio_chn" : 0,
          "gate_dio_chn" : 1,
          "pause_time" : 1E-6,
          "clock_rate" : 120E6,
          "max_ticks" : np.iinfo(np.uint16).max}

class PulseGen():
    def __init__(self, config):
        self.cr = config['clock_rate']
        self.mt = config['max_ticks']
        self.pt = config['pause_time']

        self.dchns = config['dio_chns']
        self.gate = config['gate_dio_chn']
        self.counter = config['counter_dio_chn']
        self.gate = config['gate_dio_chn']

    def bin_to_int(self, array):
        total = 0
        for idx, val in enumerate(array):
            if val:
                total += 2**idx
        return total

    def int_to_bin(self, val):
        return [int(x) for x in bin(val)[:1:-1]]
    
    def split_time(self, time):
        total_ticks = int(round(time * self.cr))
        q = int(round(total_ticks // self.mt))
        r = int(round(total_ticks % self.mt))
        if q > 0 and r != 0:
            ticks = np.zeros(q+1)
            ticks[:-1] = np.repeat(self.mt,q)
            ticks[-1] = r
        elif q > 0:
            ticks = np.repeat(self.mt,q)
        elif q == 0:
            ticks = np.array([r])
        return ticks.astype(np.uint16)

    def parse_step(self, step : dict[str, object]):
        time = step['duration'] * 1E-3
        ticks = self.split_time(time)
        values = []
        for tick in ticks:
            binary_time = self.int_to_bin(tick)
            # Pad binary tick count to a length of 16 
            binary_time += [0] * (16 - len(binary_time))
            full_binary = binary_time + step['dio_array']
            values.append(self.bin_to_int(full_binary))
        return values
    
    def parse_sequence(self, 
                       steps : list[dict[str, object]], 
                       default_dio:list[bool] = None):
        if default_dio is None:
            default_dio = [0] * self.dchns
        pause_step = {'duration':self.pt*1E3,
                      'dio_array': default_dio}
        paused_steps = [pause_step] + steps + [pause_step]
        values = list(map(self.parse_step,paused_steps))
        flat = list(itertools.chain.from_iterable(values))
        return flat

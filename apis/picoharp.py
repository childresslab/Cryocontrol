import ctypes as ct
from multiprocessing.sharedctypes import Value
import numpy as np
import logging as log
log.basicConfig(format='%(levelname)s:%(message)s ', level=log.DEBUG)

MAXDEVNUM = 8
HISTCHAN = 65536
SYNCDIVMIN = 1
SYNCDIVMAX = 8
ZCMIN = 0            #mV
ZCMAX = 20           #mV
DISCRMIN = 0         #mV
DISCRMAX = 800       #mV
SYNCOFFSMIN = -99999 #ps
SYNCOFFSMAX =  99999 #ps
CHANOFFSMIN = -8000  #ps
CHANOFFSMAX =  8000  #ps
ACQTMIN = 1          #ms
ACQTMAX = 360000000  #ms (=100h)
MAXBINSTEPS = 8

ERROR_DEVICE_OPEN_FAIL = -1

class PicoDeviceNotFoundError(Exception):
    pass

class PicoDeviceError(Exception):
    pass
 
default_config = {"binning" : 0,
                  "sync_offset" : 23000,
                  "acq_time" : 1000,
                  "sync_divider" : 8,
                  "cfd_zero_cross" : [10,10],
                  "cfd_level" : [190,120],
                  "stop" : False,
                  "stop_count" : 1000,
                  }
class PicoHarp():
    def __init__(self,config = {}):
        self._dll = ct.dll.LoadLibrary('phlib64') 
        self._init_dll_functions()

        self.times     = np.zeros(HISTCHAN,dtype = np.float)
        self.histogram = np.zeros(HISTCHAN,dtype = np.uint)
        self.last_hist = np.zeros(HISTCHAN,dtype = np.uint)

        config = default_config.update(config)
        self.binning = config['binning']
        self.sync_offset = config['sync_offset']
        self.sync_div = config['sync_divider']
        self.zero_crossing = config['cfd_zero_cross']
        self.cfd_level = config['cfd_level']
        self.stop = config['stop']
        self.stop_count = config['stop_count']
        self.acq_time = config['acq_time']

    def _init_dll_functions(self):
        cint = ct.c_int
        cintp = ct.pointer(ct.c_int)
        cuintp = np.ctypeslib.ndpointer(dtype=np.uint,ndim=1,flags="C_CONTIGUOUS")
        cdoublep = ct.pinter(ct.c_double)
        charp = ct.c_char_p

        # Unless specified return value is int, 
        # 0 indicates success
        # <0 indicates error

        #param1: errstring >=40 char buffer
        #param2: errcode
        self._dll.PH_GetErrorString.argtypes = [charp, cint]
        self._dll.PH_GetErrorString.restype = cint

        #param1: devidx, device index 0..7
        #param2: serial >=8 char buffer
        self._dll.PH_OpenDevice.argtypes = [cint, charp]
        self._dll.PH_OpenDevice.restype = cint

        #param1: devidx
        self._dll.PH_CloseDevice.argtypes = [cint]
        self._dll.PH_CloseDevice.restype = cint

        #param1: devidx
        #param2: mode 0=histogramming, 2=T2, 3=T3
        self._dll.PH_Initialize.argtypes = [cint, cint]
        self._dll.PH_Initialize.restype = cint

        #param1: devidx
        self._dll.PH_Calibrate.argtypes = [cint]
        self._dll.PH_Calibrate.restype = cint

        #param1: devidx
        #param2: channel, 0/1
        #param3: level
        #param4: zerocross
        self._dll.PH_SetInputCFD.argtypes = [cint, cint, cint, cint]
        self._dll.PH_SetInputCFD.restype = cint

        #param1: devidx
        #param2: div, 1/2/4/8
        self._dll.PH_SetSyncDiv.argtypes = [cint,cint]
        self._dll.PH_SetSyncDiv.restype = cint

        #param1: devidx
        #param2: offset shift in ps
        self._dll.PH_SetSyncOffset.argtypes = [cint, cint]
        self._dll.PH_SetSyncOffset.restype = cint

        #param1: devidx
        #param2: stop_ofl 0/1 (don't stop/stop)
        #param3: stopcount level at which to stop (max 65535)
        self._dll.PH_SetStopOverflow.argtypes = [cint,cint,cint]
        self._dll.PH_SetStopOverflow.restype = cint

        #param1: devidx
        #param2: binning, power of two: 0=1x, 1=2x, 2=4x
        self._dll.PH_SetBinning.argtypes = [cint,cint]
        self._dll.PH_SetBinning.restype = cint

        #param1: devidx
        #param2: block (always 0 for hist mode)
        self._dll.PH_ClearHistMem.argtypes = [cint,cint]
        self._dll.PH_ClearHistMem.restype = cint

        #param1: devidx
        #param2: tacq, time to acquire in ms
        self._dll.PH_StartMeas.argtypes = [cint,cint]
        self._dll.PH_StartMeas.restype = cint

        #param1: devidx
        # MUST BE CALLED EVERYTIME MEASUREMENTS ARE DONE
        self._dll.PH_StopMeas.argtypes = [cint]
        self._dll.PH_StopMeas.restype = cint

        #param1: devidx
        #param2: ctcstatus 0/>0 (running/done)
        self._dll.PH_CTCStatus.argtypes = [cint,cintp]
        self._dll.PH_CTCStatus.restype = cint

        #param1: devidx
        #param2: chcount output counts at least HISTCHAN 
        #param3: block (always 0 for hist mode)
        self._dll.PH_GetHistogram.argtypes = [cint,cuintp,cint]
        self._dll.PH_GetHistogram.restype = cint

        #param1: devidx
        #param2: resolution
        self._dll.PH_GetResolution.argtypes = [cint, cdoublep]
        self._dll.PH_GetResolution.restype = cint

        #param1: devidx
        #param2: channel
        #param3: rate
        self._dll.PH_GetCountRate.argtypes = [cint, cint, cintp]
        self._dll.PH_GetCountRate.restype = cint

        #param1: devidx
        #param2: elapsed time in ms
        self._dll.PH_GetElapsedMeasTime.argtypes = [cint,cdoublep]
        self._dll.PH_GetElapsedMeasTime.restype = cint

    def open_device(self,devidx=0):
        serial = ct.c_char_p(8)
        rc = self._dll.PH_OpenDevice(devidx,serial)
        if rc == 0:
            return str(serial.value)
        if rc == ERROR_DEVICE_OPEN_FAIL:
            raise PicoDeviceError(self.get_error_string(rc))
        else:
            raise PicoDeviceNotFoundError()

    def close_device(self):
        rc = self._dll.PH_CloseDevice(self.devidx)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
    
    def initialize(self):
        rc = self._dll.PH_Initialize(self.devidx,self.mode)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def calibrate(self):
        rc = self._dll.PH_Calibrate(self.devidx)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def set_input_CFD(self,channel:int,level:int,zero_cross:int):
        if channel not in [0,1]:
            raise ValueError(f"Channel {channel} must be either 0 or 1.")
        if (level < DISCRMIN or level > DISCRMAX):
            raise ValueError(f"Discriminator Level {level} outside range [{DISCRMIN},{DISCRMAX}].")
        if (zero_cross < ZCMIN or zero_cross > ZCMAX):
            raise ValueError(f"Zero Crossing Level {zero_cross} outside range [{ZCMIN},{ZCMAX}].")
        rc = self._dll.PH_SetInputCFD(self.devidx,channel,level,zero_cross)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def set_sync_div(self,div:int):
        if div not in [1,2,4,8]:
            raise ValueError(f"Rate divider must be one of [1,2,4,8].")
        rc = self._dll.PH_SetSyncDiv(self.devidx, div)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def set_sync_offset(self,offset:int):
        if (offset < SYNCOFFSMIN or offset > SYNCOFFSMAX):
            raise ValueError(f"Sync offset {offset} outside range [{SYNCOFFSMIN}, {SYNCOFFSMAX}].")
        rc = self._dll.PH_SetSyncDiv(self.devidx, offset)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def set_stop_overflow(self,stop:bool,stop_count:int):
        stop = int(stop)
        if (stop_count < 0 or stop_count > 65535):
            raise ValueError(f"Stop count {stop_count} outside range [0,65535].")
        rc = self._dll.PH_SetStopOverflow(self.devidx, stop, stop_count)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def set_binning(self,binning:int):
        """
            The binning code corresponds to a power of 2, i.e.:
            0 =   1x base resolution,     => 4*2^0 =    4ps
            1 =   2x base resolution,     => 4*2^1 =    8ps
            2 =   4x base resolution,     => 4*2^2 =   16ps
            3 =   8x base resolution,     => 4*2^3 =   32ps
            4 =  16x base resolution,     => 4*2^4 =   64ps
            5 =  32x base resolution,     => 4*2^5 =  128ps
            6 =  64x base resolution,     => 4*2^6 =  256ps
            7 = 128x base resolution,     => 4*2^7 =  512ps
        """
        if (binning < 0 or binning > MAXBINSTEPS-1):
            raise ValueError(f"Binning {binning} outside range [0,{MAXBINSTEPS-1}].")
        rc = self._dll.PH_SetBinning(self.devidx, binning)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def clear_hist_mem(self,block:int=0):
        rc = self._dll.PH_ClearHistMem(self.devidx, block)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def start_meas(self,tacq:int):
        if (tacq < ACQTMIN or tacq > ACQTMAX):
            raise ValueError(f"Count time {tacq} ms outside range [{ACQTMIN}, {ACQTMAX}].")
        rc = self._dll.PH_StartMeas(self.devidx, tacq)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def stop_meas(self):
        rc = self._dll.PH_StopMeas(self.devidx)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def ctc_status(self) -> bool:
        status = ct.c_int()
        rc = self._dll.PH_CTCStatus(self.devidx, ct.pointer(status))
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        # convert int to a bool
        # False means running, True means done
        return bool(status)

    def get_histogram(self,block:int=0) -> np.NDArray[np.uint]:
        hist = np.zeros(HISTCHAN,dtype=np.uint)
        rc = self._dll.PH_GetHistogram(self.devidx,hist,block)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        self.last_hist = self.histogram
        self.last_times = self.times
        self.histogram = hist
        self.times = self.get_resolution * np.array(range(HISTCHAN))
        return hist

    def get_resolution(self) -> float:
        res = ct.c_double()
        rc = self._dll.PH_GetResolution(self.devidx, ct.pointer(res))
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        return float(res)

    def get_countrate(self,channel:int=1) -> int:
        if channel not in [0,1]:
            raise ValueError(f"Channel {channel} not 0 or 1.")
        rate = ct.c_int()
        rc = self._dll.PH_GetCountRate(self.devidx, channel, ct.pointer(rate))
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        return int(rate)

    def get_elapsed_meas_time(self) -> float:
        time = ct.c_double()
        rc = self._dll.PH_GetElapsedMeasTime(self.devidx, ct.pointer(time))
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        return float(time)

    def get_error_string(self, error_num:int) -> str:
        error_msg = ct.c_char_p(80)
        rc = self._dll.PH_GetErrorString(error_msg,error_num)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        return str(error_msg.value)

    # Okay there's the slog of library functions handled.
    # Now we can get around to some higher level stuff.
    def initialize(self):
        for i in range(MAXDEVNUM):
            try:
                serial = self.open_device(i)
            except PicoDeviceNotFoundError:
                continue
            except PicoDeviceError as e:
                print(self.get_error_string(i))
                raise e
            self.devidx = i
            self.serial = serial
        else:
            log.error("No picoharp device found!")

    def deinitialize(self):
        self.stop_meas()
        self.clear_hist_mem()
        self.close_device()

    @property
    def binning(self) -> int:
        return self._binning
    
    @binning.setter
    def binning(self, val:int) -> None:
        self.set_binning(val)
        self._binning = val

    @property
    def zero_crossing(self) -> list[int]:
        return self._zero_crossing
    
    @zero_crossing.setter
    def zero_crossing(self, val:list[int]) -> None:
        for i in [0,1]:
            self.set_input_CFD(i,self.cfd_level[i],val[i])
        self._zero_crossing = val[:2]
    
    @property
    def cfd_level(self) -> list[int]:
        return self._cfd_level
    
    @cfd_level.setter
    def cfd_level(self, val:list[int]) -> None:
        for i in [0,1]:
            self.set_input_CFD(i,val[i],self.zero_crossing[i])
        self._cfd_level = val[:2]
    
    @property
    def sync_div(self) -> int:
        return self._sync_div
    
    @sync_div.setter
    def sync_div(self, val:int) -> None:
        self.set_sync_div(val)
        self._sync_div = val
    
    @property
    def sync_offset(self) -> int:
        return self._sync_offset
    
    @sync_offset.setter
    def sync_offset(self, val:int) -> None:
        self.set_sync_offset(val)
        self._sync_offset = val

    @property
    def stop(self) -> bool:
        return self._stop
    
    @stop.setter
    def stop(self, val:bool) -> None:
        self.set_stop_overflow(val,self.stop_count)
        self._stop = val

    @property
    def stop_count(self) -> int:
        return self._stop_count        

    @stop_count.setter
    def stop_count(self, val:int) -> None:
        self.set_stop_overflow(self.stop,val)
        self._stop_count = val
    
    @property
    def resolution(self) -> int:
        self.get_resolution()

    @property
    def acq_time(self) -> int:
        return self._acq_time
    
    @acq_time.setter
    def acq_time(self, val:int) -> None:
        if (val < ACQTMIN or val > ACQTMAX):
            raise ValueError(f"Count time {val} ms outside range [{ACQTMIN}, {ACQTMAX}].")
        self._acq_time = val

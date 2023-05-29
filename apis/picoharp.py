import ctypes as ct
from multiprocessing.sharedctypes import Value
import numpy as np
import logging
log = logging.getLogger(__name__)
from pathlib import Path

from numpy.typing import NDArray
from typing import Union

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
                  "acq_time" : 3600,
                  "sync_divider" : 4,
                  "cfd_zero_cross" : [10,10],
                  "cfd_level" : [190,120],
                  "stop" : False,
                  "stop_count" : 1000,
                  "mode" : 0
                  }

class PicoHarp():
    def __init__(self,config = {}) -> None:
        """Initialize the picoharp class. This prepares everything we need
        but doesn't actually connect to an instrument yet.
        Loads up the phlib dll, initializes some storage variables and then
        loads up the config we provide. Config should follow the following
        format:
        default_config = {"binning" : 0,
                          "sync_offset" : 23000,
                          "acq_time" : 3600,
                          "sync_divider" : 4,
                          "cfd_zero_cross" : [10,10],
                          "cfd_level" : [190,120],
                          "stop" : False,
                          "stop_count" : 1000,
                          "mode" : 0
                         }


        Parameters
        ----------
        config : dict, optional
            Configuration dictionary for the picoharp, see above for defaults.
            Addiitonal context can be found in the picoharp manual.
        """
        self._dll = ct.cdll.LoadLibrary('phlib64.dll') 
        self._init_dll_functions()

        self.times     = np.zeros(HISTCHAN+1,dtype = float)
        self.histogram = np.zeros(HISTCHAN,dtype = np.uint)
        self.last_hist = np.zeros(HISTCHAN,dtype = np.uint)
        self.last_times = np.zeros(HISTCHAN,dtype = float)
        self.elaps = 0
        self.last_elaps = 0
        self.devidx = None
        # Override defaults with given configuration.
        default_config.update(config)
        self.default_config = default_config
        self.mode = default_config['mode']

        self._binning = self.default_config['binning']
        self._sync_offset = self.default_config['sync_offset']
        self._sync_div = self.default_config['sync_divider']
        self._zero_crossing = self.default_config['cfd_zero_cross']
        self._cfd_level = self.default_config['cfd_level']
        self._stop = self.default_config['stop']
        self._stop_count = self.default_config['stop_count']
        self._acq_time = self.default_config['acq_time']

    def _init_dll_functions(self):
        """
        In a sense this function doesn't do anything. But it tells
        ctypes what we expect all the types to be, which can makes things work a bit nicer.
        This function essentially acts like a header file.
        """
        cint = ct.c_int
        cintp = ct.POINTER(ct.c_int)
        cuintp = np.ctypeslib.ndpointer(dtype=np.uint,ndim=1,flags="C_CONTIGUOUS")
        cdoublep = ct.POINTER(ct.c_double)
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

    def open_device(self,devidx:int=0) -> str:
        """Attemps to open a device with id number given by `devidx`

        Parameters
        ----------
        devidx : int, optional
            The device ID to attempt to open, by default 0

        Returns
        -------
        str
            The serial number associated with the opened device.

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if there's an error opening the given device.
        PicoDeviceNotFoundError
            Raised if no device is found associated with the given id.
        """
        serial = ct.create_string_buffer(b"",8)
        rc = self._dll.PH_OpenDevice(ct.c_int(devidx),serial)
        if rc == 0:
            return str(serial.value.decode("utf-8"))
        if rc == ERROR_DEVICE_OPEN_FAIL:
            raise PicoDeviceError(self.get_error_string(rc))
        else:
            raise PicoDeviceNotFoundError(self.get_error_string(rc))

    def close_device(self):
        """Closes a previously opened device.

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if the device can't be closed.
        """
        rc = self._dll.PH_CloseDevice(self.devidx)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
    
    def initialize(self):
        """Initializes the picoharp device. This will put it in the desired mode.

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if device can't be initialized. 
            See picoharp manual for additional details.
            If your picoharp does not have the correct firmware license, an error
            will be raised here and this software will not work. 
        """
        rc = self._dll.PH_Initialize(self.devidx,self.mode)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def calibrate(self):
        """Calls the picoharp internal calibration function. Should be run
        whenever the device hasn't been used for a while or if it underwent a
        significant change in temperature.

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        rc = self._dll.PH_Calibrate(self.devidx)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def set_input_CFD(self,channel:int,level:int,zero_cross:int):
        """Set the level and zero-crossing of one of the discriminator channels.
        See picoharp manual for additional information.

        Parameters
        ----------
        channel : int
            CFD channel to set
        level : int
            The voltage crossing level in mV. Value must be positive but represents
            a negative voltage.
        zero_cross : int
            The zero crossing level in mV.

        Raises
        ------
        ValueError
            If any given value is out of bounds for the device. See constants
            in code.
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
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
        """Set the sync divider for channel one. Input rate must always be below
        10 MHz when divided by this value. This value is then taken into
        account when reading out the countrate.

        Parameters
        ----------
        div : int
            Divider amount, must be one of [1,2,4,8]

        Raises
        ------
        ValueError
            Raised if the given value is not an acceptable value.
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        if div not in [1,2,4,8]:
            raise ValueError(f"Rate divider must be one of [1,2,4,8].")
        rc = self._dll.PH_SetSyncDiv(self.devidx, div)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def set_sync_offset(self,offset:int):
        """Set the offset time for the sync channel, allowing for tuning
        of the zero point, in ps. Easier than using a cable of the right length.

        Parameters
        ----------
        offset : int
            Offset in ps, anyhere between +/- 10 ns.
        Raises
        ------
        ValueError
            Raised if given offset is outside range.
        PicoDeviceError
            Raised with an associated message if any error occurs.

        """
        if (offset < SYNCOFFSMIN or offset > SYNCOFFSMAX):
            raise ValueError(f"Sync offset {offset} outside range [{SYNCOFFSMIN}, {SYNCOFFSMAX}].")
        rc = self._dll.PH_SetSyncOffset(self.devidx, offset)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def set_stop_overflow(self,stop:bool,stop_count:int):
        """Setup overflow value and whether to stop when this is encountered.
        For example, if set to True and 100, the device will stop acquiring once
        any bin goes over a count of 100.

        Parameters
        ----------
        stop : bool
            Wether to stop when an overflow is encountered.
        stop_count : int
            The value at which to overflow.

        Raises
        ------
        ValueError
            Raised if the stop_count is outside the acceptable range.
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        stop = int(stop)
        if (stop_count < 0 or stop_count > 65535):
            raise ValueError(f"Stop count {stop_count} outside range [0,65535].")
        rc = self._dll.PH_SetStopOverflow(self.devidx, stop, stop_count)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def set_binning(self,binning:int):
        """Set the binning of the histogram data according to the following chart:

            The binning code corresponds to a power of 2, i.e.:
            0 =   1x base resolution,     => 4*2^0 =    4ps
            1 =   2x base resolution,     => 4*2^1 =    8ps
            2 =   4x base resolution,     => 4*2^2 =   16ps
            3 =   8x base resolution,     => 4*2^3 =   32ps
            4 =  16x base resolution,     => 4*2^4 =   64ps
            5 =  32x base resolution,     => 4*2^5 =  128ps
            6 =  64x base resolution,     => 4*2^6 =  256ps
            7 = 128x base resolution,     => 4*2^7 =  512ps

        Parameters
        ----------
        binning : int
            The power of two by which the time axis will be binned.
            i.e. resolution = 4ps * 2^<binning>

        Raises
        ------
        ValueError
            Raised if the binning exponent is outside the acceptable range.
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        if (binning < 0 or binning > MAXBINSTEPS-1):
            raise ValueError(f"Binning {binning} outside range [0,{MAXBINSTEPS-1}].")
        rc = self._dll.PH_SetBinning(self.devidx, binning)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def clear_hist_mem(self,block:int=0):
        """Clear the histogram memory of the device. 
        Will not touch `self.histogram`, 'self.last_hist` or their associated time data.

        Parameters
        ----------
        block : int, optional
            Which memory block to clear, by default 0

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        rc = self._dll.PH_ClearHistMem(self.devidx, block)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def start_meas(self,tacq:int=0):
        """Start the measurement.

        Parameters
        ----------
        tacq : int, optional
            The amount of time to acquire for in ms.
            If the overflow stop is enabled, may not count for the full time.
            By default, will acquire for the time given by `self.acq_time`.

        Raises
        ------
        ValueError
            Raised if the acquisition time is outside the correct range.
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        if tacq == 0:
            tacq = self.acq_time * 1000
        if (tacq < ACQTMIN or tacq > ACQTMAX):
            raise ValueError(f"Count time {tacq} ms outside range [{ACQTMIN}, {ACQTMAX}].")
        rc = self._dll.PH_StartMeas(self.devidx, tacq)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def stop_meas(self):
        """Stop the measurment regardless of status.
        Note that this should also be called at the end of acquisition even when the device
        stops by itself, for internal tracking or something. See picoharp manual...

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        rc = self._dll.PH_StopMeas(self.devidx)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))

    def ctc_status(self) -> bool:
        """Returns True if the device is done counting, else False.

        Returns
        -------
        bool
            The counting status of the device. False if done counting.

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        status = ct.c_int()
        rc = self._dll.PH_CTCStatus(self.devidx, ct.pointer(status))
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        # convert int to a bool
        # False means running, True means done
        return bool(status.value)

    def get_histogram(self,block:int=0) -> NDArray[np.uint]:
        """Gets the current histogram data from the device,
        this can be performed while the measurement is being taken.
        Additionally, calling this function saves the counting data internally.
        Parameters
        ----------
        block : int, optional
            The memory block from which the data should be read, by default 0

        Returns
        -------
        npt.NDArray[np.uint]
            Array of count data.

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        hist = np.zeros(HISTCHAN,dtype=np.uint)
        rc = self._dll.PH_GetHistogram(self.devidx,hist,block)
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        hist = hist.astype(int)
        self.last_hist = self.histogram
        self.last_times = self.times
        self.last_elaps = self.elaps

        self.histogram = hist
        self.elaps = self.get_elapsed_meas_time() / 1000 # Convert to seconds
        self.times = self.get_resolution() * (np.arange(HISTCHAN) + 0.5)
        return hist

    def get_resolution(self) -> float:
        """Computes and returns the timing resolution of the device.
        See `self.set_binning` docs for details.

        Returns
        -------
        float
            The device resolution in ps

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        res = ct.c_double()
        rc = self._dll.PH_GetResolution(self.devidx, ct.pointer(res))
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        return float(res.value)

    def get_countrate(self,channel:int=1) -> int:
        """Get the current countrate on the specified channel. Only updates on
        the device every 100ms.

        Parameters
        ----------
        channel : int, optional
            Device channel to interrogate, either 0 or 1, by default 1

        Returns
        -------
        int
            The count rate in counts/s on the given channel.

        Raises
        ------
        ValueError
            Raised if an invalid channel number is given.
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        if channel not in [0,1]:
            raise ValueError(f"Channel {channel} not 0 or 1.")
        rate = ct.c_int()
        rc = self._dll.PH_GetCountRate(self.devidx, channel, ct.pointer(rate))
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        return int(rate.value)

    def get_elapsed_meas_time(self) -> float:
        """Returns how long the current measurement has run for.

        Returns
        -------
        float
            Elapsed time in seconds.

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if any error occurs.
        """
        time = ct.c_double()
        rc = self._dll.PH_GetElapsedMeasTime(self.devidx, ct.pointer(time))
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        return float(time.value)

    def get_error_string(self, error_num:int) -> str:
        """Converts a given integer return code to an associated error message.

        Parameters
        ----------
        error_num : int
            The returned error code to be translated.

        Returns
        -------
        str
            The terse error message.

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if any error occurs.
            In this case, hopefully this won't happen...
        """
        error_msg = ct.create_string_buffer(b"",40)
        rc = self._dll.PH_GetErrorString(error_msg,ct.c_int(error_num))
        if rc != 0:
            raise PicoDeviceError(self.get_error_string(rc))
        return str(error_msg.value.decode("utf-8"))

    # Okay there's the slog of library functions handled.
    # Now we can get around to some higher level stuff.
    def init_device(self) -> None:
        """Loos through all possible device numbers and tries to open the device.
        Once it finds one, it saves it's number and serial, and initializes it with the given
        configuration.

        Raises
        ------
        PicoDeviceError
            Raised with an associated message if any error with a device occurs.
        """
        num = 0
        for i in range(MAXDEVNUM):
            try:
                serial = self.open_device(i)
                num = i
                break
            except PicoDeviceNotFoundError:
                continue
            except PicoDeviceError as e:
                print(self.get_error_string(i))
                raise e
        else:
            log.error("No picoharp device found!")

        self.devidx = num
        self.serial = serial
        self.initialize()

        self.binning = self.default_config['binning']
        self.sync_offset = self.default_config['sync_offset']
        self.sync_div = self.default_config['sync_divider']
        self.zero_crossing = self.default_config['cfd_zero_cross']
        self.cfd_level = self.default_config['cfd_level']
        self.stop = self.default_config['stop']
        self.stop_count = self.default_config['stop_count']
        self.acq_time = self.default_config['acq_time']

    def deinitialize(self) -> None:
        """Gracefully stops and closes down the connected device.
        This should always be called before quitting.
        """
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
    
    def save(self,filename:Union[str,Path]="picoharp.npz") -> None:
        x = self.times
        y = self.histogram
        e = self.elaps

        path = Path(filename)
        path = _safe_file_name(path)
        ext = str(path.suffix).lower()
        if ext == ".dat":
            _save_dat(path,x,y,e)
        elif ext == ".npz":
            _save_npz(path,x,y,e)
        elif ext == ".csv":
            _save_csv(path,x,y,e)

def _safe_file_name(filename : Path):
    """Check the file path for a file with the same name, if one is found
       append a number to the given filename to avoid conflicts.


    Parameters
    ----------
    filename : Path
        A path object pointing to a file.

    Returns
    -------
    Path
        The same filename with a number added if needed.
    """
    suffix = ""
    if filename.is_file():
        folder = filename.parent
        inc = len(list(folder.glob(f"{filename.stem}*{filename.suffix}")))
        return folder / f"{filename.stem}{inc}{filename.suffix}"
    else:
        return filename

def _save_dat(path:Path,x,y,e):
    with path.open('w') as f:
        f.write(f"#PicoHarp 300  Histogram Data - Elapsed Time: {e}\n")
        f.write("#channels per curve\n")
        f.write(f"{HISTCHAN}\n")
        f.write("#display curve no.\n")
        f.write(f"0\n")
        f.write("#memory block no.\n")
        f.write(f"0\n")
        f.write("#ns/channel\n")
        f.write(f"{np.mean(np.diff(x))/1000}\n")
        f.write("#counts\n")
        for c in y:
            f.write(f"{c}\n")

def _save_npz(path:Path,x,y,e):
    np.savez(path,times=x,counts=y,elapsed=[e])

def _save_csv(path:Path,x,y,e):
    np.savetxt(path,np.vstack([x,y]).T,fmt='%d',delimiter=',',header=f"Elapsed Time {e}\ntime,count")
from time import sleep #time delay
import numpy as np
import ctypes as ct #C compatible data types
from datetime import datetime
from threading import Thread
from warnings import warn
import sys

# Annoying Int Defs from the andor dll header
DRV_SUCCESS = 20002
DRV_IDLE = 20073
DRV_NOT_INITIALIZED = 20075
DRV_NO_NEW_DATA = 20024
DRV_TEMP_NOT_REACHED = 20037
DRV_TEMP_STABILIZED = 20036
DRV_TEMP_NOT_STABILIZED = 20035
SHAMROCK_SUCCESS = 20202

class Spectrometer():
    def __init__(self, config_dic={}, **kwargs):
        # Default configuration and values
        config = {"_read_mode" : 4,
                  "_acq_mode"  : 1,
                  "_exp_time"  : 1.0,

                  "_h_bin"   : 1,
                  "_h_start" : 1,
                  "_h_width" : 1024,

                  "_v_bin"   : 1,
                  "_v_start" : 1,
                  "_v_width" : 256,

                  "_warm_temp" : -20,
                  "_cold_temp" : -70,

                  #wl = 715.44 with grating #2
                  #"_coeffs" : [585.47071,0.260088882,3.24016036e-05,-2.11424503e-08]
                  #wl = 817.57 with grating #2
                  #"_coeffs" : [686.959837,0.276316309,-4.86947064e-06,1.01879426e-09]
                  #wl = 600.38 with grating #3 (600l)
                  "_coeffs" : [536.867892,0.133935535,-2.73442347e-07,-9.95078791e-10]
                  #Dodgy calibration at 823.07 on grating #2
                  #"_coeffs" : [683.378255,0.273975792,-1.70519679e-06,-1.05695461e-09]
                 }

        # Modify config with parameters
        for key, value in config_dic.items():
            if key not in config.keys():
                print("Warning, unmatched config option: '%s' in config dictionary." % key)
                config[key] = value
        for key, value in kwargs.items():
            if key not in config.keys():
                print("Warning, unmatched config option: '%s' from kwargs." % key)
                config[key] = value
        
        for key, value in config.items():
            setattr(self,key,value)

        # Various Flags
        self.cooling = False
        self.acquiring = False
        
        # API for accessing device, from dll in andor sdk
        self.api = ct.cdll.LoadLibrary("C:\\Program Files\\Andor SDK\\Shamrock64\\atmcd64d.dll")
        self.sapi = ct.cdll.LoadLibrary("C:\\Program Files\\Andor SDK\\Shamrock64\\ShamrockCIF.dll")
        self.api.SetReadMode.argtypes = [ct.c_int]
        self.api.SetAcquisitionMode.argtypes = [ct.c_int]
        self.api.SetExposureTime.argtypes = [ct.c_float]
        self.api.SetSingleTrack.argtypes = [ct.c_int,ct.c_int]

        # Path to directory containing detector.ini
        # Not actually needed
        # Should not be changed unless you know what you're doing.
        # Needs to be a bytes string due to conversion to C char *.
        path = b""

        print("Initializing Device, this takes a few seconds...")
        stat1 = self.api.Initialize(ct.c_char_p(path))
        stat2 = self.sapi.ShamrockInitialize(ct.c_char_p(path))
        if (stat1 == DRV_SUCCESS and stat2 == SHAMROCK_SUCCESS):
            print("Device Initialization Was Succesful!")
        else:
            raise RuntimeError("Could not Initialize a device.")
        try:
            print("Initializing Spectromemter Parameters")
            self.get_temp_range()
            self.get_sensor_size()
            self.api.SetReadMode(self._read_mode)
            self.api.SetAcquisitionMode(self._acq_mode)
            self.api.SetExposureTime(self._exp_time)
            self.set_image(self._h_bin,self._v_bin,
                           self._h_start,self._h_width,
                           self._v_start,self._v_width)
        except Exception as e:
            print("Error occured during Setup, shutting down:")
            self.api.ShutDown()
            self.sapi.ShamrockClose()
            raise(e)
        
        def __del__(self):
            self.close(force=True)

    ##########################
    # Acquisition Management #
    ##########################
    @property
    def exp_time(self):
        return self._exp_time
    
    @exp_time.setter
    def exp_time(self, time):
        self._exp_time = time
        self.api.SetExposureTime(self._exp_time)
        times = self.get_timings()
        print(f"Exp Time actually set to {times[0]:.2e}")
        print(f"Cycle Time is {times[2]:.2e}")
        self._cycle_time = times[2]

    def get_timings(self):
        exp = ct.c_float()
        acc = ct.c_float()
        kin = ct.c_float()
        self.api.GetAcquisitionTimings(ct.byref(exp),
                                       ct.byref(acc),
                                       ct.byref(kin))
        return exp.value, acc.value, kin.value

    def get_sensor_size(self):
        h_pixels = ct.c_int()
        v_pixels = ct.c_int()
        self.api.GetDetector(ct.byref(h_pixels), ct.byref(v_pixels))
        self._h_pixels = h_pixels.value
        self._v_pixels = v_pixels.value
        return (self._h_pixels, self._v_pixels)
        
    def set_image(self,h_bin=1,v_bin=1,h_start=1,h_width=1024,v_start=1,v_width=256):
        self.api.SetReadMode(4)
        if h_bin > 1:
            print("Warning, recommended to keep horizontal binning at 1.")
        if (h_bin < 1 or h_bin > self._h_pixels):
            raise ValueError("Invalid Horizontal Binning: %d." % h_bin)
        self._h_bin = h_bin
        if (h_start < 1 or h_start > self._h_pixels):
            raise ValueError("Invalid Horizontal Start: %d." % h_start)
        self._h_start = h_start
        if (h_width < 1 or h_width > self._h_pixels-h_start+1):
            raise ValueError("Invalid Horizontal Width: %d." % h_width)
        if ((h_width % h_bin) != 0):
            raise ValueError("Horizontal Width must be multiple of horizontal binning.")
        self._h_width = h_width

        if (v_bin < 1 or v_bin > self._v_pixels):
            raise ValueError("Invalid Vertical Binning: %d." % v_bin)
        self._v_bin = v_bin
        if (v_start < 1 or v_start > self._v_pixels):
            raise ValueError("Invalid Vertical Start: %d." % v_start)
        self._v_start = v_start
        if (v_width < 1 or v_width > self._v_pixels-v_start+1):
            raise ValueError("Invalid Vertical Width: %d." % v_width)
        if ((v_width % v_bin) != 0):
            raise ValueError("Vertical Width must be multiple of vertical binning.")
        self._v_width = v_width

        h_end = self._h_start - 1 + self._h_width * self._h_bin
        v_end = self._v_start - 1 + self._v_width * self._v_bin
        retval =  self.api.SetImage(self._h_bin, self._v_bin, 
                                    self._h_start, h_end,
                                    self._v_start, v_end)
        return(retval)
        
    def vertical_bin(self, vbin):
        if (vbin%2):
            print("Full image vertical binning only works with a power of 2")
            print("For finer control, use set_image()")
            vbin = 1<<(vbin-1).bit_length()
            print("Setting binning to next power of 2: %d" % vbin)
        return self.set_image(1,vbin,1,1024,1,16)
        
    def prep_acq(self):
        return self.api.PrepareAcquisition()

    def get_acq(self):
        if self.get_status() != DRV_IDLE:
            raise RuntimeError("Spectrometer is not Idle")
        size = [self._v_width,self._h_width]
        data = (ct.c_long * (size[0] * size[1]))()

        # Start acquirinng and wait for the acquisition to be done, 
        # signaled by the driver
        # timeout set to 1.5 times the acquisition cycle time.
        self.api.StartAcquisition()
        resp = self.api.WaitForAcquisitionTimeOut(int(self._cycle_time * 5 * 1000))
        if resp == DRV_NO_NEW_DATA:
            if self.get_status() != DRV_IDLE:
                raise RuntimeError("Waiting on spectrometer timed out")
        elif resp != DRV_SUCCESS:
            raise RuntimeError("An unknown error occured")
        self.api.GetMostRecentImage(ct.byref(data),ct.c_ulong(size[0]*size[1]))
        return np.array(data).reshape(size)

    def run_video(self,max_runs=-1,process_callback=None,cycle_delay=None):
        if self.get_status() != DRV_IDLE:
            raise RuntimeError("Spectrometer is not Idle")
        self.api.SetAcquisitionMode(5) # Set mode to run till abort
        if process_callback is None:
            process_callback = lambda *args: None

        size = [self._v_width,self._h_width]
        data = (ct.c_long * (size[0] * size[1]))()
        error = False
        interrupted = False
        self.api.StartAcquisition()
        i = 0
        while i != max_runs:
            try:
                ret = self.api.WaitForAcquisitionTimeOut(int(self._exp_time * 1.5 * 1000))
                if ret == DRV_NO_NEW_DATA:
                    continue
                if ret != DRV_SUCCESS:
                    error = True
                    break
                self.api.GetMostRecentImage(ct.byref(data),ct.c_ulong(size[0]*size[1]))
                res = process_callback(i,datetime.now().timestamp(), np.array(data).reshape(size))
                if res is not None and res is False:
                    interrupted = True
                    break
                if cycle_delay is not None:
                    sleep(cycle_delay)
            except KeyboardInterrupt:
                interrupted = True
                break    
            i += 1

        ret = self.api.AbortAcquisition()
        if ret != DRV_SUCCESS:
            warn("Could not abort spectrometer acquisition!")
        if error:
            warn("Video mode exited due to error")
        if interrupted:
            warn("Video mode interrupted by Keyboard or Abort")
        self.api.SetAcquisitionMode(1) # Reset to single shot

    def async_run_video(self,max_runs=-1,process_callback=None,cycle_delay=None):
        t = Thread(target=self.run_video, 
                   args = (max_runs,process_callback,cycle_delay))
        t.start()
        return t


    def get_status(self):
        status = ct.c_int()
        self.api.GetStatus(ct.byref(status))
        return status.value

    def set_fvb(self):
        self._v_width = 1
        self._h_width = 1024
        return(self.api.SetReadMode(0))
    
    def set_single_track(self,center,width):
        retval = self.api.SetReadMode(3)
        self.api.SetSingleTrack(center,width)
        self._h_width = 1024
        self._v_width = width
        return retval

    ##########################
    # Temperature Management #
    ##########################
    def get_temp_range(self):
        min_temp = ct.c_int()
        max_temp = ct.c_int()
        self.api.GetTemperatureRange(ct.byref(min_temp), ct.byref(max_temp))
        self.max_temp = max_temp.value
        self.min_temp = min_temp.value

    def cap_temp(self, temp):
        if temp > self.max_temp:
            print("Warning, temperature %d > Max Temp, setting to %d." % temp, self.max_temp)
            return self.max_temp
        if temp < self.min_temp:
            print("Warning, temperature %d < Min Temp, setting to %d." % temp, self.min_temp)
            return self.min_temp
        return temp
    
    @property
    def cold_temp(self):
        return self._cold_temp

    @property
    def warm_temp(self):
        return self._warm_temp

    @cold_temp.setter
    def cold_temp(self, temp):
        temp = self.cap_temp(self,temp)
        self._cold_temp = temp
        if self.cooling:
            self.start_cooling()
    
    @warm_temp.setter
    def warm_temp(self, temp):
        temp = self.cap_temp(self,temp)
        self._warm_temp = temp
        if not self.cooling:
            self.stop_cooling()

    def get_temp(self):
        temp = ct.c_int()
        ret = self.api.GetTemperature(ct.byref(temp))
        return temp.value, ret

    def start_cooling(self):
        self.api.SetTemperature(self.cold_temp)
        if not self.cooling:
            self.api.CoolerON()
            self.cooling = True

    def stop_cooling(self):
        self.api.SetTemperature(self.warm_temp)
        if self.cooling:
            self.api.CoolerOFF()
            self.cooling = False

    def waitfor_temp(self, stable=False, disp_callback = None):
        if self.cooling:
            target = self.cold_temp
            print("Cooling Down, Please Wait.")
        else:
            print("Cooler not on, try calling self.start_cooling() first.")
            return
        print("",end='\r')
        def default_callback(t_status,i):
            spin = ['-','\\','|','/']
            msg_lookup = {DRV_TEMP_NOT_REACHED:"Temperature not yet reached",
                          DRV_TEMP_NOT_STABILIZED:"Temperature stabilizing"}
            sys.stdout.write("\033[K")
            print(f"{spin[i%4]} {msg_lookup[t_status[1]]}. Current: {t_status[0]}, Target: {target}",end='\r')
            sleep(0.5)

        if disp_callback is None:
            disp_callback = default_callback 
        waitfor = DRV_TEMP_STABILIZED if stable else DRV_TEMP_NOT_STABILIZED 
        try:
            i = 0
            while (t_status:= self.get_temp())[1] != waitfor:
                disp_callback(t_status,i)
                i += 1
            print("")
            print("Target temperature reached!")
        except KeyboardInterrupt:
            print("")
            print("Aborting wait for temp.")
            return

    #####################
    # System Management #
    #####################
    def close(self, force=False):
        self.api.ShutDown()
        self.sapi.ShamrockClose()
        print("Spectrometer Closed")

    ##########################
    # Wavelength Calibration #
    ##########################
    def get_wavelengths(self):
        coeffs = np.flip(np.array(self._coeffs))
        pixels = np.arange(1,1025)
        wavelengths = np.polyval(coeffs,pixels)
        wavelengths = wavelengths[self._h_start-1:self._h_start+self._h_width-1]
        if self._h_bin != 1:
            print("Warning, horionztal binning not recommended.")
            nbins = wavelengths.size//self._h_bin
            wavelengths = wavelengths.reshape(nbins,self._h_bin)
            wavelengths = np.mean(wavelengths,axis=1)
        return wavelengths

    # This is hella sketch and will likely result in a not properly calibrated
    # wavelength axis. Ideally use Andor to setup the range and 
    # get the proper calibration coeffs from there via file -> Configuration Files
    # The current coeffs are correct for wl = 715.44 with grating #2
    def set_central_wavelength(self, wavelength):
        current_wl = ct.c_float()
        self.sapi.ShamrockGetWavelength(0, ct.byref(current_wl))
        current_wl = current_wl.value

        diff = current_wl - self._coeffs[0]
        self._coeffs[0] = wavelength - diff

        new_wl = ct.c_float(wavelength)
        ret = self.sapi.ShamrockSetWavelength(0, new_wl)
        if ret == SHAMROCK_SUCCESS:
            return
        else:
            raise RuntimeError("Something went wrong, maybe quit and double check in Andor...")

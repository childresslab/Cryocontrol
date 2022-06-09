import numpy as np
from time import sleep
from pathlib import Path
from datetime import datetime
from scipy.signal import find_peaks
import cavspy as cs
from scipy.constants import c

import apis.rdpg as rdpg
from apis.wl_refl_fitter import WLFitter
from apis.spect import Spectrometer
dpg = rdpg.dpg

wlfitter = WLFitter()
devices = {}

data = {'times'    : [],
        'lengths'  : [],
        'errors' : [],
        'spectrum' : [],
        'wavelength' : np.array([]),
        'first_peak' : 0,
        }

def choose_save_dir(*args):
    dpg.add_file_dialog(label="Chose Save Directory", 
                        default_path=dpg.get_value('save_dir'), 
                        directory_selector=True, modal=True,callback=set_save_dir)

def set_save_dir(sender,chosen_dir,user_data):
    dpg.set_value('save_dir',chosen_dir['file_path_name'])

def toggle_spectrometer(sender,value,user_data):
    if value:
        # Setup Spectrometer
        win_size = dpg.get_item_rect_size("main_window")
        with dpg.window(modal=True, tag='sp_warning',autosize=True):
            popup_size = dpg.get_item_rect_size("sp_warning")
            dpg.set_item_pos('sp_warning',  [int((win_size[0]/2 - popup_size[0]/2)), int((win_size[1]/2 - popup_size[1]/2))])
            dpg.add_text("Please wait for spectrometer to connect.",tag="sp_warn")
            dpg.add_loading_indicator(tag='sp_load')
        try:
            devices['spect'] = Spectrometer()
        except Exception as err:
            print("Failed to open connection to spectrometer.")
        dpg.set_value("sp_warn", "Please wait for spectrometer to cool down.")
        dpg.set_value("Spectrometer/Status", "Cooling")
        with dpg.group(horizontal=True,parent="sp_warning"):
            dpg.add_input_float(label="Temperature",tag="spec_temp",step=0,readonly=True)

        def update_temp(temp,i):
            dpg.set_value("spec_temp",temp[0])
            sleep(0.1)

        devices['spect'].start_cooling()
        devices['spect'].waitfor_temp(disp_callback=update_temp)
        dpg.set_value("sp_warn", "All done!")
        dpg.hide_item('sp_load')
        dpg.delete_item('sp_warning')
        set_spectrometer_exp()
        dpg.set_value("Spectrometer/Status", "Cold")
        dpg.set_exit_callback(lambda _: devices['spect'].close if devices['spect'] is not None else None)

    else:
        dpg.set_value("Spectrometer/Status", "Warming")
        devices['spect'].stop_cooling()
        devices['spect'].close()
        devices['spect'] = None
        dpg.set_value("Spectrometer/Status", "Unitialized")

def set_spectrometer_exp(*args):
    wl_tree.save()
    if wl_tree["Spectrometer/Connect"]:
        devices['spect'].exp_time = wl_tree["Spectrometer/Exposure Time (s)"]
    if wl_tree['Spectrometer/Data Row (-1 for FVB)'] < 0:
        devices['spect'].set_fvb()
    else:
        devices['spect'].vertical_bin(16)

def save_scan(sender,child_data,user_data):
    save_dir = dpg.get_value("save_dir")
    save_path = Path(save_dir)
    n = len(list(save_path.glob("normalized_spectrum*.csv")))
    save_file = save_path / f"normalized_spectrum{n}.csv"
    wl = wlfitter.wavelength
    pairs = zip(wl,data['spectrum'])
    with save_file.open("w") as f:
        f.write("Wavelength,Intensity\n")
        for w,i in pairs:
            f.write(f"{w:.2f},{i:.5e}\n")

def load_spectrum(sender,chosen_dir,user_data):
    directory = list(chosen_dir['selections'].values())[0]
    loaded_data = np.genfromtxt(directory,names=True,delimiter=',')
    data['wavelength'] = loaded_data['Wavelength']
    data['spectrum'] = loaded_data['Intensity']
    fit_scan()


def save_lengths(sender,child_data,user_data):
    save_dir = dpg.get_value("save_dir")
    save_path = Path(save_dir)
    n = len(list(save_path.glob("length_log*.csv")))
    save_file = save_path / f"length_log{n}.csv"
    pairs = zip(data['times'],data['lengths'],data['errors'])
    with save_file.open("w") as f:
        f.write("Time,Length,Error\n")
        for t,l,e in pairs:
            f.write(f"{t},{l},{e}\n")

def _get_acq():
    spectrum = devices['spect'].get_acq()
    trim = wl_tree['Spectrometer/Data Row (-1 for FVB)']
    if trim > 0:
        spectrum = spectrum[trim]
    else:
        spectrum = spectrum[0]
    signal = np.copy(spectrum)
    signal = (signal-min(signal))
    signal = signal/np.max(signal)
    data['wavelength'] = devices['spect'].get_wavelengths()
    data['spectrum'] = signal

def cont_scan_callback(i,time,spectrum):
    trim = wl_tree['Spectrometer/Data Row (-1 for FVB)']
    if trim > 0:
        spectrum = spectrum[trim]
    else:
        spectrum = spectrum[0]
    signal = np.copy(spectrum)
    signal = (signal-min(signal))
    signal = signal/np.max(signal)
    data['spectrum'] = signal
    fit_scan()
    return dpg.get_value("continuous")

def cont_scan(sender,value,user_data):
    if value:
        # Disable button
        for item in ["single_scan","load_scan"]:
            dpg.disable_item(item)
        data['wavelength'] = devices['spect'].get_wavelengths()
        devices['spect'].prep_acq()
        thread = devices["spect"].async_run_video(-1,cont_scan_callback,wl_tree["Spectrometer/Pause Time (ms)"] / 1000)
    else:
        # Enable buttons
        for item in ["single_scan","load_scan"]:
            dpg.enable_item(item)

def single_scan(*args):
    # Disable button
    for item in ["single_scan","load_scan","continuous"]:
        dpg.disable_item(item)
    devices['spect'].prep_acq()
    _get_acq()
    # fit
    fit_scan()
    # Enable button
    for item in ["single_scan","load_scan","continuous"]:
        dpg.enable_item(item)

def fit_scan(write=True):
    # Do the fit
    minwl = wl_tree["Fitting/Min Wavelength"]
    maxwl = wl_tree["Fitting/Max Wavelength"]
    min_index = np.argmax(data['wavelength'] >= minwl)
    peaks = find_peaks(data["spectrum"][np.logical_and(data['wavelength']<=maxwl,
                                                       data['wavelength']>=minwl)],
                       prominence=wl_tree["Fitting/Prominence"],
                       distance=wl_tree["Fitting/Distance (px)"],
                       wlen=wl_tree["Fitting/Window Length (px)"])
    peaks_idx = peaks[0] + min_index
    if list(peaks_idx) == []:
        peaks_idx = [0]
    data['first_peak'] = peaks_idx[0]
    peak_wl = data["wavelength"][peaks_idx]
    peak_freq = c/(peak_wl * 1E-9)
    fsrs = np.diff(peak_freq[::-1])
    lengths = c/(2*fsrs)

    length = cs.uncert.from_floats(lengths * 1E6)  # um
    if write:
        data["errors"].append(length.u)
        data["lengths"].append(length.x)
        data['times'].append(datetime.now().timestamp())
    # Update Plot
    lower_d = data["wavelength"][int(data['first_peak']-wl_tree["Fitting/Distance (px)"])]
    upper_d = data["wavelength"][int(data['first_peak']+wl_tree["Fitting/Distance (px)"])]
    dpg.set_value("distance_shade",[[upper_d,upper_d,lower_d,lower_d],[0,1,1,0]])
    lower_w = data["wavelength"][int(data['first_peak']-wl_tree["Fitting/Window Length (px)"]//2)]
    upper_w = data["wavelength"][int(data['first_peak']+wl_tree["Fitting/Window Length (px)"]//2)]
    dpg.set_value("window_shade",[[upper_w,upper_w,lower_w,lower_w],[0,1,1,0]])
    dpg.show_item("distance_shade")
    dpg.show_item("window_shade")
    dpg.show_item("spect_peaks")
    dpg.set_value("spect_peaks", [list(peak_wl),list(data["spectrum"][peaks_idx])])
    dpg.set_value("spect_sig", [list(data["wavelength"]), list(data['spectrum'])])
    if write:
        dpg.set_value("length_e", [data['times'],data['lengths'],data['errors'],data['errors']])
        dpg.set_value("length", [data['times'],data['lengths']])

def set_fitter(*args):
    wl_tree.save()
    if dpg.is_item_shown("distance_shade"):
        # Update Plot
        lower_d = data["wavelength"][int(data['first_peak']-wl_tree["Fitting/Distance (px)"])]
        upper_d = data["wavelength"][int(data['first_peak']+wl_tree["Fitting/Distance (px)"])]
        dpg.set_value("distance_shade",[[upper_d,upper_d,lower_d,lower_d],[0,1,1,0]])
        lower_w = data["wavelength"][int(data['first_peak']-wl_tree["Fitting/Window Length (px)"]//2)]
        upper_w = data["wavelength"][int(data['first_peak']+wl_tree["Fitting/Window Length (px)"]//2)]
        dpg.set_value("window_shade",[[upper_w,upper_w,lower_w,lower_w],[0,1,1,0]])

def refit(*args):
    set_fitter()
    if len(data['spectrum']) != 0:
        fit_scan(write=False)

def clear_data(*args):
    for key in data.keys():
        if key == 'wavelength':
            data[key] = np.array([])
        else:
            data[key] = []
    #Clear Plots
    dpg.set_value("spect_sig",[[0],[0]])
    dpg.set_value("length",[[0],[0]])
    dpg.set_value("length_e",[[0],[0],[0],[0]])
    dpg.hide_item("window_shade")
    dpg.hide_item("distance_shade")
    dpg.hide_item("spect_peaks")

def set_prominence(sender,value,data):
    if sender == "prominence_line":
        value = dpg.get_value(sender)
        if value > 1.0:
            dpg.set_value("prominence_line",1.0)
            value = 1.0
        if value < 0:
            dpg.set_value("prominence_line",0.0)
            value = 0.0
        dpg.set_value("Fitting/Prominence", value)
    elif sender == "Fitting/Prominence":
        dpg.set_value("prominence_line",dpg.get_value(sender))
    set_fitter()

def set_minmax(sender,value,data):
    if sender == "max_line":
        value = dpg.get_value(sender)
        maxval = dpg.get_item_configuration("Fitting/Max Wavelength")['max_value']
        minval = dpg.get_value("min_line")+1
        if value > maxval:
            dpg.set_value("max_line",maxval)
            value = maxval
        if value < minval:
            dpg.set_value("max_line",minval)
            value = minval
        dpg.set_value("Fitting/Max Wavelength", value)
    if sender == "min_line":
        value = dpg.get_value(sender)
        minval = dpg.get_item_configuration("Fitting/Min Wavelength")['min_value']
        maxval = dpg.get_value("max_line")-1
        if value > maxval:
            dpg.set_value("min_line",maxval)
            value = maxval
        if value < minval:
            dpg.set_value("min_line",minval)
            value = minval
        dpg.set_value("Fitting/Min Wavelength", value)
    elif sender == "Fitting/Min Wavelength":
        maxval = dpg.get_value("Fitting/Max Wavelength")-1
        if value > maxval:
            value = maxval
        dpg.set_value("Fitting/Min Wavelength",value)
        dpg.set_value("min_line",value)
    elif sender == "Fitting/Max Wavelength":
        minval = dpg.get_value("Fitting/Min Wavelength")+1
        if value < minval:
            value = minval
        dpg.set_value("Fitting/Max Wavelength",value)
        dpg.set_value("max_line",value)
    refit()

rdpg.initialize_dpg("Transmitted Whitelight Interferometer")

with dpg.window(label='T Whitelight Length', tag='main_window'):
    with dpg.group(horizontal=True):
        dpg.add_text("Data Directory:")
        dpg.add_input_text(default_value="X:\\DiamondCloud\\", tag="save_dir")
        dpg.add_button(label="Pick Directory", callback=choose_save_dir)

    # Begin Tabs
    with dpg.tab_bar() as main_tabs:
        # Begin Saving Tab
        with dpg.tab(label="Scanner"):
            with dpg.child_window(autosize_x=True,autosize_y=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(tag="single_scan",label="Take Scan", callback=single_scan)
                    dpg.add_checkbox(tag="continuous",label="Continuous Scan", callback=cont_scan)
                    dpg.add_button(tag="clear",label="Clear",callback=clear_data)
                    dpg.add_button(tag="load_scan",label="Load Scan",callback=lambda:dpg.show_item("scan_picker"))
                with dpg.group(horizontal=True):
                    dpg.add_text("Filename:")
                    dpg.add_input_text(tag="save_file", default_value="datafile.npz", width=200)
                    dpg.add_button(tag="save_scan_button", label="Save Scan",callback=save_scan)
                    dpg.add_checkbox(tag="auto_save", label="Auto")
                    dpg.add_button(tag="save_length_button", label="Save Lengths",callback=save_lengths)
                    dpg.add_button(tag="refit",label="Refit",callback=refit)
                with dpg.group(horizontal=True, width=0):
                    with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="wl_tree"):
                        wl_tree = rdpg.TreeDict('wl_tree','t_wl_params_save.csv')
                        wl_tree.add("Spectrometer/Connect",False,callback=toggle_spectrometer, save=False)
                        wl_tree.add("Spectrometer/Status", "Uninitialized", callback=None, save=False)
                        wl_tree.add("Spectrometer/Exposure Time (s)",0.00001,item_kwargs={'step':0,'format':"%.2e"},
                                     callback = set_spectrometer_exp)
                        wl_tree.add("Spectrometer/Data Row (-1 for FVB)", -1,
                                     callback = set_spectrometer_exp)
                        wl_tree.add("Spectrometer/Pause Time (ms)", 10,item_kwargs={'step':0})

                        wl_tree.add("Fitting/Prominence", 0.1,drag=True,item_kwargs={'min_value':0.0,'speed':0.01,'max_value':1.0,'clamped':True,'format':"%.2f"},
                                    callback=set_prominence)
                        wl_tree.add("Fitting/Min Wavelength", 600,drag=True,item_kwargs={'min_value':0.0,'speed':1,'max_value':1600,'clamped':True},
                                    callback=set_minmax)
                        wl_tree.add("Fitting/Max Wavelength", 650,drag=True,item_kwargs={'min_value':0.0,'speed':1,'max_value':1600,'clamped':True},
                                    callback=set_minmax)
                        wl_tree.add("Fitting/Distance (px)", 100,item_kwargs={'step':1},
                                    callback=set_fitter)
                        wl_tree.add("Fitting/Window Length (px)", 50, item_kwargs={'step':1},
                                    callback=set_fitter)
                        
                    with dpg.child_window(width=-1,autosize_x=True,autosize_y=True):
                        with dpg.subplots(2,1,row_ratios=[1/2,1/2],width=-1,height=-1,link_all_x=False): 
                            with dpg.plot(label="Spectra", tag="spectra_plot",
                                        width=-0,height=-0, anti_aliased=True,
                                        fit_button=True):
                                dpg.bind_font("plot_font")
                                dpg.add_plot_axis(dpg.mvXAxis, label="Wavelength (nm)", tag="spect_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="Intensity (A.U.)", tag="spect_y")
                                dpg.add_area_series([0,0,0,0],[0,0,0,0],show=False,
                                                    parent='spect_y', tag="window_shade",label="Peak Window")
                                dpg.add_area_series([0,0,0,0],[0,0,0,0],show=False,
                                                    parent='spect_y', tag="distance_shade",label="Peak Distance")
                                dpg.add_line_series([0],[0],parent="spect_y",tag="spect_sig", label="Signal Spectrum")
                                dpg.add_drag_line(default_value=wl_tree["Fitting/Prominence"],show=True,callback=set_prominence,vertical=False,
                                                     parent='spectra_plot', tag="prominence_line",label="Prominence")
                                dpg.add_drag_line(default_value=wl_tree["Fitting/Min Wavelength"],show=True,callback=set_minmax,vertical=True,
                                                     parent='spectra_plot', tag="min_line",label="Wavelength Min")
                                dpg.add_drag_line(default_value=wl_tree["Fitting/Max Wavelength"],show=True,callback=set_minmax,vertical=True,
                                                     parent='spectra_plot', tag="max_line",label="Wavelength Max")
                                dpg.add_stem_series([0],[0],parent='spect_y',tag="spect_peaks",label="Peaks",show=False)
                            with dpg.plot(label="Length", tag="length_plot",
                                        width=-0,height=-0, anti_aliased=True,
                                        fit_button=True):
                                dpg.bind_font("plot_font")
                                dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="length_x", time=True)
                                dpg.add_plot_axis(dpg.mvYAxis, label="Length (um)", tag="length_y")
                                dpg.add_line_series([0],[0],parent='length_y',tag='length',label="Length")
                                dpg.add_error_series([0],[0],[0],[0],parent="length_y",tag="length_e", label="Error (n>2)")

# Make file picker window
with dpg.file_dialog(label="Load Spectrum", 
                    default_path=dpg.get_value('save_dir'),
                    default_filename="new_ref.csv",
                    modal=True,callback=load_spectrum,
                    tag="scan_picker",
                    show = False):
    dpg.add_file_extension(".*")
    dpg.add_file_extension("", color=(150, 255, 150, 255))
    dpg.add_file_extension(".csv", color=(0, 255, 0, 255), custom_text="[CSV]")
dpg.set_primary_window('main_window',True)
rdpg.start_dpg()
        

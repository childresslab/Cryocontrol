from asyncio.windows_events import proactor_events
from calendar import c
import rdpg as rdpg
import numpy as np
from wl_refl_fitter import WLFitter
from spect import Spectrometer
from time import sleep
from pathlib import Path
from datetime import datetime
dpg = rdpg.dpg

wlfitter = WLFitter()
devices = {}

data = {'times'    : [],
        'lengths1'  : [],
        'errors1' : [],
        'lengths2'  : [],
        'errors2' : [],
        'lengths3'  : [],
        'errors3' : [],
        'lengths4'  : [],
        'errors4' : [],
        'errors'   : [],
        'spectrum' : [],
        'fft'      : []}

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
        dpg.set_value("sp_warn", "Please wait for spectrometer to cooldown")
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

def save_lengths(sender,child_data,user_data):
    save_dir = dpg.get_value("save_dir")
    save_path = Path(save_dir)
    n = len(list(save_path.glob("length_log*.csv")))
    save_file = save_path / f"length_log{n}.csv"
    points = zip(data['times'],*[data[f"{s}{i+1}"] 
                                 for i in range(wl_tree["Fitting/N Cavity Peaks"]) 
                                 for s in ["lengths","errors"]])
    with save_file.open("w") as f:
        f.write("Time")
        for i in range(wl_tree["Fitting/N Cavity Peaks"]):
            f.write(f",Length{i+1},Error{i+1}")
        f.write('\n')
        for point in points:
            f.write(','.join([f"{d}" for d in point]))
            f.write('\n')

def _get_acq():
    try:
        spectrum = devices['spect'].get_acq()
    except RuntimeError as e:
        print(str(e))
        spectrum = [list(np.ones_like(wlfitter.wavelength)*0)]
        data['spectrum'] = [list(np.ones_like(wlfitter.wavelength)*0)]
        return spectrum
    trim = wl_tree['Spectrometer/Data Row (-1 for FVB)']
    print(spectrum)
    if trim > 0:
        spectrum = spectrum[trim]
    else:
        spectrum = spectrum[0]
    signal = np.copy(spectrum)
    signal = (signal-min(signal))/np.max(signal)
    signal /= np.max(signal)
    data['spectrum'] = signal
    return spectrum

def cont_scan_callback(i,time,spectrum):
    trim = wl_tree['Spectrometer/Data Row (-1 for FVB)']
    if trim > 0:
        spectrum = spectrum[trim]
    else:
        spectrum = spectrum[0]
    signal = np.copy(spectrum)
    signal = (signal-min(signal))/np.max(signal)
    signal /= np.max(signal)
    fit_scan(signal)
    data['spectrum'] = signal
    return dpg.get_value("continuous")

def cont_scan(sender,value,user_data):
    if value:
        # Disable button
        for item in ["single_scan","take_reference","load_reference"]:
            dpg.disable_item(item)
        devices['spect'].prep_acq()
        thread = devices["spect"].async_run_video(-1,cont_scan_callback,wl_tree["Spectrometer/Pause Time (ms)"] / 1000)
    else:
        # Enable buttons
        for item in ["single_scan","take_reference","load_reference"]:
            dpg.enable_item(item)

def single_scan(*args):
    # Disable button
    for item in ["single_scan","take_reference","load_reference","continuous"]:
        dpg.disable_item(item)
    try:
        devices['spect'].prep_acq()
        spectrum = _get_acq()
        # fit
        fit_scan(spectrum)
        # Enable button
    except Exception as e:
        # Enable buttons
        for item in ["single_scan","take_reference","load_reference"]:
            dpg.enable_item(item)
        raise e
    for item in ["single_scan","take_reference","load_reference","continuous"]:
        dpg.enable_item(item)

def fit_scan(spectrum):
    # Do the fit
    wlfitter.fit_spectra(spectrum)
    lengths = [wlfitter.fit.params[f'g{i+2}_center'].value for i in range(wl_tree["Fitting/N Cavity Peaks"])] 
    errors = [wlfitter.fit.params[f'g{i+2}_center'].stderr for i in range(wl_tree["Fitting/N Cavity Peaks"])] 
    lengths.sort()
    for i in range(wlfitter.settings['n_max'] - 1):
        data[f'lengths{i+1}'].append(lengths[i])
        data[f'errors{i+1}'].append(errors[i])
    for i in range(wlfitter.settings['n_max'] - 1,4):
        data[f'lengths{i+1}'].append(-1.0)
        data[f'errors{i+1}'].append(-1.0)
    data['times'].append(datetime.now().timestamp())
    # Update Plot
    dpg.set_value("spect_sig", [list(wlfitter.wavelength), list(data['spectrum'])])
    dpg.set_value("spect_wind",[list(wlfitter.wavelength), list(wlfitter.wind)])
    dpg.set_value("fft_init", [list(wlfitter.fit_lengths), list(wlfitter.fit.init_fit)])
    dpg.set_value("fft_data", [list(wlfitter.lengthu), list(wlfitter.fft)])
    dpg.set_value("fft_fit", [list(wlfitter.fit_lengths), list(wlfitter.fit.best_fit)])
    dpg.set_value("fft_cut", [[np.max(wlfitter.fit_lengths)],[1.0]])
    dpg.set_value("length1", [rdpg.offset_timezone(data['times']),data['lengths1']])
    dpg.set_value("length2", [rdpg.offset_timezone(data['times']),data['lengths2']])
    dpg.set_value("length3", [rdpg.offset_timezone(data['times']),data['lengths3']])
    dpg.set_value("length4", [rdpg.offset_timezone(data['times']),data['lengths4']])

def get_reference(*args):
    # Acquire a spectrum
    spectrum = _get_acq()
    # Get the wavelengths
    wl = devices['spect'].get_wavelengths()
    # Pass off to process_reference
    process_reference(wl,spectrum)

def load_ref_callback(sender,chosen_dir,user_data):
    file_directory = list(chosen_dir['selections'].values())[0]
    # Load in the data
    data = np.genfromtxt(file_directory,delimiter=',',names=True)
    # Pass off to process_reference
    process_reference(data['Wavelength'],data['Intensity'])

def process_reference(wl,spectrum):
    # Set the reference data
    wlfitter.set_reference(wl,spectrum)
    set_fitter()
    # Plot it on the spectrum plot
    dpg.set_value("spect_ref",[list(wlfitter.wavelength),list(wlfitter.reference_spectrum)])

def set_fitter(*args):
    wl_tree.save()
    wlfitter.settings['auto_gaussian'] = wl_tree["Fitting/Auto Gaussian"]
    wlfitter.settings['scale'] = wl_tree["Fitting/Scale"]
    wlfitter.settings['shift'] = wl_tree["Fitting/Shift"]
    wlfitter.settings['n_max'] = wl_tree["Fitting/N Cavity Peaks"] + 1
    wlfitter.settings['n_min'] = wl_tree["Fitting/N Cavity Peaks"] + 1
    wlfitter.settings['chi2_tol'] = wl_tree["Fitting/Chi2 Tol."]
    wlfitter.settings['offset_cut'] = float(wl_tree["Fitting/Offset Cut"])
    wlfitter.calc_reference_gaussian()

    for i in range(4):
        if i in range(wl_tree["Fitting/N Cavity Peaks"]):
            dpg.show_item(f"length{i+1}")
        else:
            dpg.hide_item(f"length{i+1}")
    

    wlfitter.settings['init_centers'] = [0] + wl_tree["Fitting/Init. Centers"]
    wlfitter.settings['init_sigmas'] = [4] + wl_tree["Fitting/Init. Sigmas"]
    wlfitter.settings['init_amps'] = [1] + wl_tree["Fitting/Init. Amps"]
    
    dpg.set_value("spect_win",[list(wlfitter.wavelength),list(wlfitter.reference_gaussian)])
    if len(data['spectrum']) != 0:
        fit_scan(data['spectrum'])
    else:
        print(wl_tree["Fitting/Offset Cut"])
        cut = list([float(wl_tree["Fitting/Offset Cut"])])
        print(type(cut),type(cut[0]))
        dpg.set_value("fft_cut", [cut,[1.0]])

def clear_data(*args):
    for key in data.keys():
        data[key] = []
    #Clear Plots
    dpg.set_value("spect_sig",[[0],[0]])
    dpg.set_value("spect_wind",[[0],[0]])
    dpg.set_value("fft_data",[[0],[0]])
    dpg.set_value("fft_fit",[[0],[0]])
    dpg.set_value("length1",[[0],[0]])
    dpg.set_value("length2",[[0],[0]])
    dpg.set_value("length3",[[0],[0]])
    dpg.set_value("length4",[[0],[0]])

rdpg.initialize_dpg("Reflected Whitelight Interferometer")

with dpg.window(label='Whitelight Length', tag='main_window'):
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
                    dpg.add_button(tag="take_reference",label="Take Reference", callback=get_reference)
                    dpg.add_button(tag="load_reference",label="Load Reference", callback=lambda:dpg.show_item("ref_picker"))
                    dpg.add_button(tag="clear",label="Clear",callback=clear_data)
                with dpg.group(horizontal=True):
                    dpg.add_text("Filename:")
                    dpg.add_input_text(tag="save_file", default_value="datafile.npz", width=200)
                    dpg.add_button(tag="save_scan_button", label="Save Scan",callback=save_scan)
                    dpg.add_checkbox(tag="auto_save", label="Auto")
                    dpg.add_button(tag="save_length_button", label="Save Lengths",callback=save_lengths)

                with dpg.group(horizontal=True, width=0):
                    with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="wl_tree"):
                        wl_tree = rdpg.TreeDict('wl_tree','wl2_params_save.csv')
                        wl_tree.add("Spectrometer/Connect",False,callback=toggle_spectrometer, save=False)
                        wl_tree.add("Spectrometer/Status", "Uninitialized", callback=None, save=False)
                        wl_tree.add("Spectrometer/Exposure Time (s)",0.00001,item_kwargs={'step':0,'format':"%.2e","on_enter":True},
                                     callback = set_spectrometer_exp)
                        wl_tree.add("Spectrometer/Data Row (-1 for FVB)", -1,
                                     callback = set_spectrometer_exp,item_kwargs={"on_enter":True})
                        wl_tree.add("Spectrometer/Pause Time (ms)", 10,item_kwargs={'step':0})

                        wl_tree.add("Fitting/Auto Gaussian",True)
                        wl_tree.add("Fitting/Scale", 1.0,item_kwargs={'step':0.1,'format':"%.2f", "on_enter":True},
                                    callback=set_fitter)
                        wl_tree.add("Fitting/Shift", 0.0,item_kwargs={'step':1,'format':"%.2f", "on_enter":True},
                                    callback=set_fitter)
                        wl_tree.add("Fitting/Chi2 Tol.", 0.01, item_kwargs={'step':0,'format':"%.2e", "on_enter":True},
                                    callback=set_fitter)
                        wl_tree.add("Fitting/N Cavity Peaks", 3, item_kwargs={'min_value' : 1,'max_value':4, "on_enter":True, "min_clamped":True,"max_clamped":True},
                                    callback=set_fitter)
                        wl_tree.add("Fitting/Offset Cut",20.0,item_kwargs={'step':0.1,'format':"%.2f",'min_value' : 0.0, "on_enter":True, "min_clamped":True},
                                    callback=set_fitter)

                        wl_tree.add("Fitting/Init. Centers", [1.0,2.0,3.0,4.0],
                                    item_kwargs={'format':"%.2f"}, callback=set_fitter)
                        wl_tree.add("Fitting/Init. Sigmas", [4.0,4.0,4.0,4.0],
                                    item_kwargs={'format':"%.2f"}, callback=set_fitter)
                        wl_tree.add("Fitting/Init. Amps", [1.0,0.5,0.25,0.123],
                                    item_kwargs={'format':"%.2f"}, callback=set_fitter)

                        
                    with dpg.child_window(width=-1,autosize_x=True,autosize_y=True):
                        with dpg.subplots(3,1,row_ratios=[1/3,1/3,1/3],width=-1,height=-1,link_all_x=False): 
                            with dpg.plot(label="Spectra", tag="spectra_plot",
                                        width=-0,height=-0, anti_aliased=True,
                                        fit_button=True):
                                dpg.bind_font("plot_font")
                                dpg.add_plot_axis(dpg.mvXAxis, label="Wavelength (nm)", tag="spect_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="Intensity (A.U.)", tag="spect_y")
                                dpg.add_line_series([0],[0],parent="spect_y",tag="spect_ref", label="Reference Spectrum")
                                dpg.add_line_series([0],[0],parent="spect_y",tag="spect_sig", label="Signal Spectrum")
                                dpg.add_line_series([0],[0],parent="spect_y",tag="spect_win", label="Window Gaussian")
                                dpg.add_line_series([0],[0],parent="spect_y",tag="spect_wind", label="Combined")
                            with dpg.plot(label="FFT", tag="fft_plot",
                                        width=-0,height=-0, anti_aliased=True,
                                        fit_button=True):
                                dpg.bind_font("plot_font")
                                dpg.add_plot_axis(dpg.mvXAxis, label="Cavity Length (um)", tag="fft_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="Intensity (A.U.)", tag="fft_y")
                                dpg.add_line_series([0],[0],parent="fft_y",tag="fft_init", label="Fit Guess")
                                dpg.add_line_series([0],[0],parent="fft_y",tag="fft_data", label="Time Data")
                                dpg.add_line_series([0],[0],parent="fft_y",tag="fft_fit", label="Time Fit")
                                dpg.add_stem_series([float(wl_tree["Fitting/Offset Cut"])],[1.0],parent="fft_y",tag="fft_cut", label="Fit Limit")
                            with dpg.plot(label="Length", tag="length_plot",
                                        width=-0,height=-0, anti_aliased=True,
                                        fit_button=True):
                                dpg.bind_font("plot_font")
                                dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="length_x", time=True)
                                dpg.add_plot_axis(dpg.mvYAxis, label="Length (um)", tag="length_y")
                                dpg.add_line_series([0],[0],parent="length_y",tag="length1", label="Length 1")
                                dpg.add_line_series([0],[0],parent="length_y",tag="length2", label="Length 2",show=False)
                                dpg.add_line_series([0],[0],parent="length_y",tag="length3", label="Length 3",show=False)
                                dpg.add_line_series([0],[0],parent="length_y",tag="length4", label="Length 4",show=False)

# Make file picker window
with dpg.file_dialog(label="Chose Reference File", 
                    default_path=dpg.get_value('save_dir'),
                    default_filename="new_ref.csv",
                    modal=True,callback=load_ref_callback,
                    tag="ref_picker",
                    show = False,
                    width = dpg.get_item_rect_size("main_window")[0],
                    height = dpg.get_item_rect_size("main_window")[1]):
    dpg.add_file_extension(".*")
    dpg.add_file_extension("", color=(150, 255, 150, 255))
    dpg.add_file_extension(".csv", color=(0, 255, 0, 255), custom_text="[CSV]")
dpg.set_primary_window('main_window',True)
rdpg.start_dpg()
        

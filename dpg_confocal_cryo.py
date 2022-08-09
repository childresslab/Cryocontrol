from typing import Callable
import dearpygui.dearpygui as dpg
import numpy as np
from time import sleep,time
from numpy.lib.histograms import histogram
from numpy.lib.stride_tricks import sliding_window_view
from threading import Thread
from datetime import datetime
import datetime as dt
from pathlib import Path
import lmfit as lm
import logging as log
from apis.dummy import fpga_base_dummy, fpga_cryo_dummy, objective_dummy
from numpy.typing import NDArray

from apis.scanner import Scanner
from apis.fpga_cryo import CryoFPGA, FPGAValueError
from apis.objective_control import Objective
from interfaces.hist_plot import mvHistPlot

import apis.rdpg as rdpg
from apis.jpe_coord_convert import JPECoord
pz_config = {"vmax" : 0,
             "vmin" : -6.5,
             "vgain" : -20,
             "R": 6.75,
             "h": 45.1}
pz_conv = JPECoord(pz_config['R'], pz_config['h'],
                   pz_config['vmin'], pz_config['vmax'])
dpg = rdpg.dpg

#TODO TODO TODO TODO TODO
# Add tooltips to everything!
# Remove uneeded plus/minus boxes
# Tune step value on plus/minus boxes
# Documentation
# Encapsulation
# Disabling/Enabling in a more programmic way.
# XY Scans, make histograms ignore zeros
# Histogram autoscale area to only non-zero
#TODO TODO TODO TODO TODO

# Slowly turning into a mess of a file
# Ideally this should be better encapsulated into individual modules
# that then combine together into one file. However that requires
# figuring out how to properly share the instrument/data between files
# which will be a bit tricky...
log.basicConfig(format='%(levelname)s:%(message)s ', level=log.INFO)

# Setup real control
log.warning("Using Real Controls")
fpga = CryoFPGA()
obj = Objective()

# Setup counts data
counts_data = {'counts':[0],
               'AI1' :[0],
               'time':[datetime.now().timestamp()]}
position_register = {"temp_galvo_position":fpga.get_galvo()}

#################
# Galvo Control #
#################
galvo_plot = mvHistPlot("Galvo Plot",True,None,True,True,1000,0,300,50,'viridis',True,1,1E9,50,50)
def set_galvo(x:float,y:float,write:bool=True) -> None:
    """
    Sets the galvo position, keeping track of updating the cursor position
    and position read out

    Parameters
    ----------
    x : float
        new x position of the galvo
    y : float
        new y position of the galvo
    write : bool, optional
        wether to immediately update update the position from
        the fpga by quickly pulsing, slightly slower so set to False if 
        your next step is to pulse or count, by default True
    """
    # If we only changed one axis, get the other's value
    if (x is None or y is None):
        galvo_pos = fpga.get_galvo()
        if x is None:
            x = galvo_pos[0]
        if y is None:
            y = galvo_pos[1]
    # Update the actual position
    fpga.set_galvo(x,y,write=write)
    # Update the cursor position
    galvo_plot.set_cursor([x,y])
    # Update the readout position
    galvo_tree["Galvo/Position"] = [x,y]

def man_set_galvo(*args):
    """
    Callback for when the galvo position is manually updated in the GUI.
    Simply calls `set_galvo` with the new position.
    """
    pos = galvo_tree["Galvo/Position"]
    set_galvo(pos[0],pos[1])

def galvo(y:float,x:float) -> float:
    """
    Scanning function for the galvo, sets the position along y and x 
    (note flipped order) and then counts for the amount of time set in the GUI.

    Parameters
    ----------
    y : float
        y position to get counts at.
    x : float
        x position to get counts at.

    Returns
    -------
    float
        the count rate acquired
    """
    log.debug(f"Set galvo to ({x},{y}) V.")
    set_galvo(x,y,write=False)
    count = get_count(galvo_tree["Scan/Count Time (ms)"])
    log.debug(f"Got count rate of {count}.")
    return count

def get_count(time:float) -> float:
    """
    Gets the count rate at the current position, appending it to the stored
    counts data if the current GUI state calls for it. Also updates
    the AI1 position in that case.

    Parameters
    ----------
    time : float
        Time to count for in ms

    Returns
    -------
    float
        The acquired count rate
    """
    count = fpga.just_count(time)
    draw_count(count)
    log.debug(f"Got count rate of {count}.")
    if count_tree["Counts/Plot Scan Counts"] or dpg.get_value("count"):
        counts_data['counts'].append(count)
        counts_data['AI1'].append(fpga.get_AI_volts([1])[0])
        counts_data['time'].append(datetime.now().timestamp())
    return count

def toggle_AI(sender,user,app):
    if user:
        dpg.show_item("AI1_series")
        dpg.show_item("AI1_series2")
        dpg.show_item("AI1_series3")
    else:
        dpg.hide_item("AI1_series")
        dpg.hide_item("AI1_series2")
        dpg.hide_item("AI1_series3")


# Base initialization of the galvo scanner object. Using the above `galvo` function
# for scanning.
galvo_scan = Scanner(galvo,[0,0],[1,1],[50,50],[1],[],float,['y','x'],default_result=-1)

def start_scan(sender,app_data,user_data):
    """
    Callback function that starts the galvo scan. Taking care of updating
    all the scan parameters.

    Parameters
    ----------
    Called through DPG callback, no need...
    """
    if not dpg.get_value("galvo_scan"):
        return -1
    abort_counts()
    steps = galvo_tree["Scan/Points"]
    galvo_scan.steps = steps[1::-1]
    galvo_scan.centers = galvo_tree["Scan/Centers (V)"][1::-1]
    galvo_scan.spans = galvo_tree["Scan/Spans (V)"][1::-1]
    
    def init():
        """
        Initialization callback for the scanner.
        """
        # Set FPGA wait time
        fpga.set_ao_wait(galvo_tree["Scan/Wait Time (ms)"],write=False)
        # Get spans and steps of scan to update plot range
        pos = galvo_scan._get_positions()
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        galvo_plot.set_size(int(galvo_scan.steps[0]),int(galvo_scan.steps[1]))
        galvo_plot.set_bounds(xmin,xmax,ymin,ymax)
        # Store the current position of the galvo for reseting later
        position_register["temp_galvo_position"] = fpga.get_galvo()
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                if control != "galvo_scan":
                    dpg.disable_item(control)
        for param in galvo_params:
            dpg.disable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.disable_cursor()
    
    def abort(i,imax,idx,pos,res):
        """
        Abort callback for the scanner. Just checks if the scan button is still
        checked, allowing for cancelling of the scan.
        """
        return not dpg.get_value("galvo_scan")

    def prog(i,imax,idx,pos,res):
        """
        Progress callback for the scanner. Run after every acquisition point
        """
        # update the progress bar of the GUI
        log.debug("Setting Progress Bar")
        dpg.set_value("pb",(i+1)/imax)
        dpg.configure_item("pb",overlay=f"Galvo Scan {i+1}/{imax}")
        # Check if we want to update the plot after every point
        # Otherwise only every line
        if galvo_tree["Plot/Update Every Point"]:
            check = True
        else:
            check = (not (i+1) % galvo_tree["Scan/Points"][0]) or (i+1)==imax
        if check:
            log.debug("Updating Galvo Scan Plot")
            # Format data for the plot
            plot_data = np.copy(np.flipud(galvo_scan.results))
            galvo_plot.autoscale = galvo_tree["Plot/Autoscale"]
            galvo_plot.nbin = galvo_tree["Plot/N Bins"]
            galvo_plot.update_plot(plot_data)
            if count_tree["Counts/Plot Scan Counts"]:
                plot_counts()

    def finish(results,completed):
        """
        Finish callback for the scanner.
        """
        # Uncheck scan button, indicating that we're done
        dpg.set_value("galvo_scan",False)
        # Reset the galvo to it's position at the start of the scan
        set_galvo(*position_register["temp_galvo_position"],write=True)
        # If autosave set, save the scan data.
        if dpg.get_value("auto_save"):
            save_galvo_scan()
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                    dpg.enable_item(control)
        for param in galvo_params:
            dpg.enable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.enable_cursor()

    # Setup the scanner
    galvo_scan._init_func = init
    galvo_scan._abort_func = abort
    galvo_scan._prog_func = prog
    galvo_scan._finish_func = finish
    # Run the scan in a new thread
    galvo_scan.run_async()

def get_galvo_range(*args):
    """
    Callback for querying the scan range from the galvo plot. Or just 
    copying the galvo position as the center of the scan.
    """
    # If plot is queried, copy full scan area
    xmin,xmax,ymin,ymax = galvo_plot.query_plot()
    if xmin is not None:
        new_centers = [(xmin+xmax)/2, (ymin+ymax)/2]
        new_spans = [xmax-xmin, ymax-ymin]
        galvo_tree["Scan/Centers (V)"] = new_centers
        galvo_tree["Scan/Spans (V)"] = new_spans
    # Otherwise, make the center of the scan the current position
    else:
        galvo_tree["Scan/Centers (V)"] = galvo_tree["Galvo/Position"]

def guess_galvo_time(*args):
    """
    Update the estimated scan time from the number of points and count times.
    """
    pts = galvo_tree["Scan/Points"]
    ctime = galvo_tree["Scan/Count Time (ms)"] + galvo_tree["Scan/Wait Time (ms)"]
    scan_time = pts[0] * pts[1] * ctime / 1000
    time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
    galvo_tree["Scan/Estimated Time"] = time_string

def galvo_cursor_callback(sender,point):
    """Keep the galvo position within range on the cursor.

    Parameters
    ----------
    point : list[float] 
        The (x,y) position of the cursor, to be limited.

    Returns
    -------
    list[float]
        The updated position, with all values between [-10,10]
    """
    if point[0] < -10:
        point[0] = -10
    if point[0] > 10:
        point[0] = 10
    if point[1] < -10:
        point[1] = -10
    if point[1] > 10:
        point[1] = 10
    set_galvo(point[0],point[1])

galvo_plot.cursor_callback = galvo_cursor_callback

# Saving Scans
def choose_save_dir(*args):
    """ 
    Select the save directory using the DPG file selector.
    """
    chosen_dir = dpg.add_file_dialog(label="Chose Save Directory", 
                        default_path=dpg.get_value("save_dir"), 
                        directory_selector=True, modal=True,callback=set_save_dir)

def set_save_dir(sender,chosen_dir,user_data):
    """
    Callback to actually set the value from the chosen file.
    """
    dpg.set_value("save_dir",chosen_dir['file_path_name'])

def save_galvo_scan(*args):
    """
    Save the scan using the scanner save function. Saves to the filename
    set by the directory and filename set in the GUI.
    """
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_galvo_file")
    path /= filename
    as_npz = not (".csv" in filename)
    header = galvo_scan.gen_header("Galvo Scan")
    galvo_scan.save_results(str(path),as_npz=as_npz,header=header)

####################
# Counting Control #
####################
def clear_counts(*args):
    """
    Clear the stored counts data. Doesn't immediately update the plot. So
    data will persist there.
    """
    counts_data['counts'] = []
    counts_data['AI1'] = []
    counts_data['time'] = []

def moving_average(values:NDArray[np.float64],window:int) -> NDArray[np.float64]:
    """
    Simple sliding window moving average. Could potentially be improved with
    weighting or better edge handling.

    Parameters
    ----------
    values : NDArray[np.float64]
        The array of values to average over.
    
    window : int
        The size of the window to average over.
        

    Returns
    -------
    NDArray[np.float64]
        The averaged data.
    """

    return np.average(sliding_window_view(values, window_shape = window), axis=1)

def average_counts(times : NDArray[np.float64],
                   counts : NDArray[np.float64],
                   window : int) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """_summary_

    Parameters
    ----------
    times : NDArray[np.float64]
        Time data to be average
    counts : NDArray[np.float64]
        Counts data to be averaged
    window : int
        Averaging window size

    Returns
    -------
    tuple[NDArray[np.float64],NDArray[np.float64]]
        Averaged time and counts data
    """
    avg_times = moving_average(times,window)
    avg_counts = moving_average(counts,window)
    return avg_times,avg_counts

def plot_counts(*args):
    """
    Update the count plots on various pages
    """
    # Truncate the count data down to the desired number of points
    delta = len(counts_data['counts']) - count_tree["Counts/Max Points"]
    while delta >= 0:
        try:
            counts_data['counts'].pop(0)
            counts_data['AI1'].pop(0)
            counts_data['time'].pop(0)
            delta -= 1
        except IndexError:
            break
    # Average the time and counts data
    avg_time, avg_counts= average_counts(counts_data['time'],
                                         counts_data['counts'],
                                         min(len(counts_data['time']),
                                             count_tree["Counts/Average Points"]))
    # Update all the copies of the count plots.
    dpg.set_value('counts_series',[rdpg.offset_timezone(counts_data['time']),counts_data['counts']])
    dpg.set_value('avg_counts_series',[rdpg.offset_timezone(avg_time),avg_counts])
    dpg.set_value('AI1_series',[rdpg.offset_timezone(counts_data['time']),counts_data['AI1']])

def draw_count(val):
    """
    Prints the count rate in numbers in the special count rate window.
    Sets the color according to some limits for our equipment.
    Yellow warning for above 1 million
    Red critical warning for above 10 million
    """
    if val > 1E6:
        dpg.set_value("count_rate",f"{val:0.2G}")
    else:
        dpg.set_value("count_rate",f"{val:0.0f}")
    if val > 1E7:
        dpg.configure_item("count_rate",color=[255,0,0])
    elif val > 1E6:
        dpg.configure_item("count_rate",color=[250,189,47])
    else:
        dpg.configure_item("count_rate",color=[255,255,255])

def abort_counts():
    """
    Stop the couting with a small sleep time to ensure that the last
    count opperation exists.
    This is to be run before any scan starts, making sure to stop the ongoing
    counting. 
    """
    if dpg.get_value("count"):
        dpg.set_value("count",False)
        sleep(count_tree["Counts/Count Time (ms)"]/1000)

def start_counts():
    """
    Function triggered when enabling counts.
    Starts a new thread that takes care of getting the count data and
    updating the plot.
    """
    # If counts are disabled, don't do anything, thread will handle 
    # stopping itself
    if not dpg.get_value('count'):
        return
    
    # Function for the new thread to run
    def count_func():
        # Make a second thread for updating the plot
        plot_thread = Thread(target=plot_counts)
        # As long as we still want to count
        while dpg.get_value("count"):
            # Get the new count data
            count = get_count(count_tree["Counts/Count Time (ms)"])
            # If the plot isn't currently being updated update it.
            if not plot_thread.is_alive():
                plot_thread = Thread(target=plot_counts)
                plot_thread.start()

    # Start the thread
    count_thread = Thread(target=count_func)
    count_thread.start()

def save_counts(*args):
    """
    Save the count data to a file.
    Effectively only saves the plotted data.
    """
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_counts_file")
    path /= filename
    with path.open('w') as f:
        f.write("Timestamp,Counts,AI1\n")
        for d in zip(counts_data['time'],counts_data['counts'],counts_data['AI1']):
                f.write(f"{d[0]},{d[1]},{d[2]}\n")

###################
# Galvo Optimizer #
###################
def optim_scanner_func(axis='x'):
    """This function generates the functio be called by the optimization scanner
    Basically just makes a function that scans over a single axis, getting the counts.

    Parameters
    ----------
    axis : str, optional
        which function to be scanned over, by default 'x'

    Returns
    -------
    Callable
        The function that sets the single axis scan position.

    Raises
    ------
    ValueError
        If an incorrect axis is called for, this should never happen.
    """
    if axis == 'x':
        def optim_func(x):
            y = position_register['temp_galvo_position'][1]
            set_galvo(x,y)
            count = get_count(optim_tree["Optimizer/Count Time (ms)"])
            log.debug(f"Got count rate of {count}.")
            return count
    elif axis == 'y':
        def optim_func(y):
            x = position_register['temp_galvo_position'][0]
            set_galvo(x,y)
            count = get_count(optim_tree["Optimizer/Count Time (ms)"])
            log.debug(f"Got count rate of {count}.")
            return count
    else:
        raise ValueError(f"Invalid Axis {axis}, must be either 'x' or 'y'.")
    return optim_func

def fit_galvo_optim(position:NDArray[np.float64],counts:NDArray[np.float64]) -> lm.model.ModelResult:
    """
    Fit a quadratic to the given data, for optimizing the galvo position.

    Parameters
    ----------
    position : NDArray[np.float64]
        The array containing the position data
    counts : NDArray[np.float64]
        The array containing the counts data

    Returns
    -------
    lm.model.ModelResult
        The fit result object from lmfit.
    """
    model = lm.models.QuadraticModel()
    params = model.guess(counts,x=position)
    params['a'].set(max=0)
    # Probably more annoying to do it right.
    weights = 1/np.sqrt(np.array([count if count > 0 else 1 for count in counts]))
    return model.fit(counts,params,x=position,weights=weights)

def optimize_galvo(*args):
    """
    Run multiple iterations of the optimization routine. To narrow in on the
    optimal position. Number of iterations set from GUI
    """
    def loop_optim():
        # Do the loop the desired number of times.
        for i in range(optim_tree["Optimizer/Iterations"]):
            single_optimize_run().join()
    # Run this all in an additional thread to avoid freezing the UI
    optim_thread = Thread(target=loop_optim)
    optim_thread.start()

def single_optimize_run():
    """
    Function for running the optimization scan.

    Returns
    -------
    Thread
        The thread on which the scan is being run.
    """
    # Save the initial galvo position, in case of aborted scan.
    position_register["temp_galvo_position"] = fpga.get_galvo()
    init_galvo_pos = position_register["temp_galvo_position"]
    # The scanner which will do the x scan
    galvo_scanner_x = Scanner(optim_scanner_func('x'),
                              [init_galvo_pos[0]],
                              [optim_tree["Optimizer/Scan Range (XY)"]],
                              [optim_tree["Optimizer/Scan Points"]],
                              output_dtype=float,
                              labels=["Galvo X"])
    # The scanner which will do the y scan
    galvo_scanner_y = Scanner(optim_scanner_func('y'),
                              [init_galvo_pos[1]],
                              [optim_tree["Optimizer/Scan Range (XY)"]],
                              [optim_tree["Optimizer/Scan Points"]],
                              output_dtype=float,
                              labels=["Galvo Y"])
    # The data for the optimizer to consider.
    optim_data = {}
    # Stop counting if that's currently happening.
    abort_counts()
    # Setup the functions for the scanners.
    def init_x():
        """
        Prepares the fpga and plot for the scan.
        """
        fpga.set_ao_wait(optim_tree["Optimizer/Wait Time (ms)"],write=False)
        position_register['temp_galvo_position'] = fpga.get_galvo()
        optim_data['counts'] = []
        optim_data['pos'] = []
        dpg.set_value('optim_x_counts',[[0],[0]])
        dpg.set_value('optim_y_counts',[[0],[0]])
        dpg.set_value('optim_x_fit',[[],[]])
        dpg.set_value('optim_y_fit',[[],[]])
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                dpg.disable_item(control)
        for param in optim_params:
            dpg.disable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.disable_cursor()

    def prog_x(i,imax,idx,pos,res):
        """
        Update the x plot of the scan. Since we're doing few points, we can
        afford to plot them all as they come in.
        """
        # Set Progress Bar
        dpg.set_value("pb",(i+1)/(2*imax))
        dpg.configure_item("pb",overlay=f"Opt. Galvo (X) {i+1}/{2*imax}")
        # Update the optimization data
        optim_data['counts'].append(res)
        optim_data['pos'].append(pos[0])
        # Update the plots
        dpg.set_value('optim_x_counts',[optim_data['pos'],optim_data['counts']])
        if count_tree["Counts/Plot Scan Counts"]:
                    plot_counts()

    def finish_x(results,completed):
        """
        Once the scan is completed, we fit the data, and move the x position
        to the optimal position, then we trigger the y-scan to be run.
        """
        positions = galvo_scanner_x.positions[0]
        fit_x = fit_galvo_optim(positions,results)
        vals = fit_x.best_values
        # Get the peak x based on the quadratic fit results.
        optim = -vals['b']/(2*vals['a'])
        # Fix the ideal position to be within the bounds of the scan.
        optim = min(optim,np.max(positions))
        optim = max(optim,np.min(positions))
        # If the desired position is outside range, we just use the starting
        # position.
        try:
            set_galvo(optim,fpga.get_galvo()[1])
        except ValueError:
            set_galvo(*position_register['temp_galvo_position'])
        # Plot the fit on the optimization plot
        new_axis = np.linspace(np.min(positions),np.max(positions),1000)
        fit_data = fit_x.eval(fit_x.params,x=new_axis)
        dpg.set_value('optim_x_fit',[new_axis,fit_data])
        # Start the y-scan.
        galvo_scanner_y.run_async().join()

    # Setup the scanner object.
    galvo_scanner_x._init_func = init_x
    galvo_scanner_x._prog_func = prog_x
    galvo_scanner_x._finish_func = finish_x

    def init_y():
        """
        Prepares for the second scan be clearing the data and updating
        the starting position.
        """
        optim_data['counts'] = []
        optim_data['pos'] = []
        position_register['temp_galvo_position'] = fpga.get_galvo()
    def prog_y(i,imax,idx,pos,res):
        """
        Update the plot while we take the scan.
        """
        dpg.set_value("pb",(i+1+imax)/(2*imax))
        dpg.configure_item("pb",overlay=f"Opt. Galvo (Y) {i+1+imax}/{2*imax}")
        optim_data['counts'].append(res)
        optim_data['pos'].append(pos[0])
        dpg.set_value('optim_y_counts',[optim_data['pos'],optim_data['counts']])
        if count_tree["Counts/Plot Scan Counts"]:
            plot_counts()

    def finish_y(results,completed):
        """
        Do the fit of the data and set the new position. Limiting to scan
        and galvo ranges.
        """
        positions = galvo_scanner_y.positions[0]
        fit_y = fit_galvo_optim(positions,results)
        vals = fit_y.best_values
        optim = -vals['b']/(2*vals['a'])
        optim = min(optim,np.max(positions))
        optim = max(optim,np.min(positions))
        try:
            set_galvo(fpga.get_galvo()[0],optim)
        except ValueError:
            set_galvo(*position_register['temp_galvo_position'])
        # Plot the fit.
        new_axis = np.linspace(np.min(positions),np.max(positions),1000)
        fit_data = fit_y.eval(fit_y.params,x=new_axis)
        dpg.set_value('optim_y_fit',[new_axis,fit_data])
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                dpg.enable_item(control)
        for param in optim_params:
            dpg.enable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.enable_cursor()

    # Setup the y-scanner object.
    galvo_scanner_y._init_func = init_y
    galvo_scanner_y._prog_func = prog_y
    galvo_scanner_y._finish_func = finish_y

    # Run the x-scan asynchronously, which will then trigger the y-scan.
    return galvo_scanner_x.run_async()

######################
# Objective Scanning #
######################
# Initialize the objective hist plot, which contains the heatmap and histogram of the data.
obj_plot = mvHistPlot("Obj. Plot",False,None,True,False,1000,0,300,50,'viridis',True,1,1E9,50,50)

# NOTE:
# The bare api of the objective control is setup so that more negative values
# are physically upwards in the cryostat.
# Here, we have opted to invert that, such that a more positive value
# is upwards in the cryo, such that plotting and moving makes more sense.
def obj_scan_func(galvo_axis:str='x') -> Callable:
    """Generates the functino which we use to scan over the objective and glavo

    Parameters
    ----------
    galvo_axis : str, optional
        Which axis of the galvo we should scan, keeping the other fixed, by default 'x'

    Returns
    -------
    Callable
        Function which we use to scan the objective and galvo.

    Raises
    ------
    ValueError
        If an incorrect axis is requested, this should never happen.
    """
    if galvo_axis not in ['x','y']:
        raise ValueError("Axis must be 'x' or 'y'")
    if galvo_axis=='y':
        # Make the function which scans the z of the objective and the y of the galvo.
        def func(z,y):
            log.debug(f"Set galvo y to {y} V.")
            obj_move = not (abs(z - obj_tree["Objective/Set Position (um)"]) < 0.01)
            if obj_move:
                obj_thread = set_obj_abs(z)
                log.debug(f"Set obj. position to {z} um.")
            set_galvo(None,y,write=False)
            if obj_move:
                obj_thread.join()
            count = get_count(obj_tree["Scan/Count Time (ms)"])
            log.debug(f"Got count rate of {count}.")
            return count
        return func
    if galvo_axis=='x':
        # Make the function which scans the z of the objective and the x of the galvo.
        def func(z,x):
            log.debug(f"Set galvo x to {x} V.")
            log.debug(f"Set obj. position to {z} um.")
            obj_move = not (abs(z - obj_tree["Objective/Set Position (um)"]) < 0.01)
            if obj_move:
                obj_thread = set_obj_abs(z)
                log.debug(f"Set obj. position to {z} um.")
            set_galvo(x,None,write=False)
            if obj_move:
                obj_thread.join()
            count = get_count(obj_tree["Scan/Count Time (ms)"])
            log.debug(f"Got count rate of {count}.")
            return count
        return func

# Initialize the objective scanner object.
obj_scan = Scanner(obj_scan_func('x'),[0,0],[1,1],[50,50],[1],[],float,['y','x'],default_result=-1)

def start_obj_scan(sender,app_data,user_data):
    """
    Take care of setting up and running an objective scan.
    Starts by cancelling any counting.
    Then setups up the scanner from the GUI values
    Following that, it defines the functions that the scanner will run.
    Then it asynchronously runs the scan.
    """
    # Check if we are starting a scan or aborting one.
    if not dpg.get_value("obj_scan"):
        return -1
    # Cancel the counting if that's ongoing.
    abort_counts()

    # Get paramters from the GUI and set the scanner object up.
    obj_steps = obj_tree["Scan/Obj./Steps"]
    obj_center = obj_tree["Scan/Obj./Center (um)"]
    obj_span = obj_tree["Scan/Obj./Span (um)"]
    galv_steps = obj_tree["Scan/Galvo/Steps"]
    galv_center = obj_tree["Scan/Galvo/Center (V)"]
    galv_span = obj_tree["Scan/Galvo/Span (V)"]
    obj_scan.steps = [obj_steps,galv_steps]
    obj_scan.centers = [obj_center,galv_center]
    obj_scan.spans = [obj_span,galv_span]
    
    def init():
        """
        The scan initialization function.
        Setsup the galvo and plots.
        """
        
        fpga.set_ao_wait(obj_tree["Scan/Wait Time (ms)"],write=False)
        pos = obj_scan._get_positions()
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        position_register["temp_obj_position"] = -obj.position
        position_register["temp_galvo_position"] = fpga.get_galvo()
        obj_plot.set_size(int(obj_scan.steps[0]),int(obj_scan.steps[1]))
        obj_plot.set_bounds(xmin,xmax,ymin,ymax)
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                if control != "obj_scan":
                    dpg.disable_item(control)
        for param in objective_params:
            dpg.disable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.disable_cursor()
    
    def abort(i,imax,idx,pos,res):
        return not dpg.get_value("obj_scan")

    def prog(i,imax,idx,pos,res):
            log.debug("Setting Progress Bar")
            dpg.set_value("pb",(i+1)/imax)
            dpg.configure_item("pb",overlay=f"Obj. Scan {i+1}/{imax}")
            if obj_tree["Plot/Update Every Point"]:
                check = True
            else:
                check = (not (i+1) % obj_tree["Scan/Galvo/Steps"]) or (i+1)==imax
            if check:
                log.debug("Updating Galvo Scan Plot")
                plot_data = np.copy(np.flip(obj_scan.results,0))
                obj_plot.autoscale = obj_tree["Plot/Autoscale"]
                obj_plot.nbin = obj_tree["Plot/N Bins"]
                obj_plot.update_plot(plot_data)
                if count_tree["Counts/Plot Scan Counts"]:
                    plot_counts()

    def finish(results,completed):
        dpg.set_value("obj_scan",False)
        set_obj_abs(position_register["temp_obj_position"])
        set_galvo(*position_register["temp_galvo_position"])
        if dpg.get_value("auto_save"):
            save_obj_scan()
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                dpg.enable_item(control)
        for param in objective_params:
            dpg.enable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.enable_cursor()
            
    obj_scan._func = obj_scan_func(obj_tree['Scan/Galvo/Axis'])
    obj_scan._init_func = init
    obj_scan._abort_func = abort
    obj_scan._prog_func = prog
    obj_scan._finish_func = finish
    return obj_scan.run_async()

def save_obj_scan(*args):
    """
    Saves the objective scan data, using the Scanners built in saving method.
    """
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_obj_file")
    path /= filename
    as_npz = not (".csv" in filename)
    header = obj_scan.gen_header("Objective Scan")
    obj_scan.save_results(str(path),as_npz=as_npz,header=header)

def toggle_objective(sender,app_data,user_data):
    """
    Callback for initializing the objective.
    Uses the app_data, i.e. new checkmar value to check if we initialize
    or deinitialize.

    Parameters
    ----------
    See standard DPG callback parameters.
    """
    if app_data:
        obj.initialize()
        obj_tree["Objective/Status"] = "Initialized"
        pos = -obj.position
        obj_tree["Objective/Current Position (um)"] = pos
        obj_tree["Objective/Set Position (um)"] = pos
        obj_tree["Objective/Limits (um)"] = [obj.soft_lower,obj.soft_upper]
        set_objective_params()
    else:
        obj.deinitialize()
        obj_tree["Objective/Status"] = "Deinitialized"

def set_objective_params(*args):
    if obj.initialized:
        limits = obj_tree["Objective/Limits (um)"]
        obj.soft_lower = limits[0]
        obj.soft_upper = limits[1]
        obj.max_move = obj_tree["Objective/Max Move (um)"]
        dpg.configure_item('obj_pos_set',min_value=limits[0],max_value=limits[1])
        dpg.configure_item('obj_pos_get',min_value=limits[0],max_value=limits[1])
        dpg.set_value("obj_pos_set",-obj.position)
        dpg.set_value("obj_pos_get",-obj.position)

def set_obj_callback(sender,app_data,user_data):
    return set_obj_abs(app_data)

def set_obj_abs(position):
    position = -position
    log.debug(f"Set objective to {-position}")
    def func():
        obj.move_abs(position,monitor=True,monitor_callback=obj_move_callback)
    t = Thread(target=func)
    t.start()
    return t

def obj_step_up(*args):
    def func():
        step = obj_tree["Objective/Rel. Step (um)"]
        obj.move_up(step,monitor=True,monitor_callback=obj_move_callback)
    t = Thread(target=func)
    t.start()
    return t

def obj_step_down(*args):
    def func():
        step = obj_tree["Objective/Rel. Step (um)"]
        obj.move_down(step,monitor=True,monitor_callback=obj_move_callback)
    t = Thread(target=func)
    t.start()
    return t

def obj_move_callback(status,position,setpoint):
    if status['idle']:
        msg = "Initialized"
    elif status['accel']:
        msg = "Accelerating"
    elif status['decel']:
        msg = "Deccelerating"
    elif status['const']:
        msg = 'At Speed'
    else:
        msg = 'Slpping'
    if status['error']:
        msg += " - Error"
    obj_tree["Objective/Status"] = msg
    obj_tree["Objective/Current Position (um)"] = -position
    obj_tree["Objective/Set Position (um)"] = -setpoint
    dpg.set_value("obj_pos_set",-setpoint)
    dpg.set_value("obj_pos_get",-position)
    return status['error']

def guess_obj_time():
    obj_pts = obj_tree["Scan/Obj./Steps"]
    galvo_pts = obj_tree["Scan/Galvo/Steps"]
    ctime = obj_tree["Scan/Count Time (ms)"] + obj_tree["Scan/Wait Time (ms)"]
    scan_time = obj_pts * galvo_pts * ctime / 1000
    time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
    obj_tree["Scan/Estimated Time"] = time_string

def get_obj_range():
    xmin,xmax,ymin,ymax = obj_plot.query_plot()
    if xmin is None:
        idx = 0 if obj_tree["Scan/Galvo/Axis"] == 'x' else 1
        obj_tree["Scan/Galvo/Center (V)"] = galvo_tree["Galvo/Position"][idx]
        obj_tree["Scan/Obj./Center (um)"] = obj_tree["Objective/Current Position (um)"]
    else:
        new_centers = [(xmin+xmax)/2, (ymin+ymax)/2]
        new_spans = [xmax-xmin, ymax-ymin]
        obj_tree["Scan/Galvo/Center (V)"] = new_centers[0]
        obj_tree["Scan/Galvo/Span (V)"] = new_spans[0]
        obj_tree["Scan/Obj./Center (um)"] = new_centers[1]
        obj_tree["Scan/Obj./Span (um)"] = new_spans[1]

###################
# Cavity Scanning #
###################
pzt_plot = mvHistPlot("Piezo Scan",True,None,True,True,1000,0,300,1000,'viridis',True,1,1E12,50,50)
def set_cav_and_count(z):
    try:
        set_cav_pos(z,write=False)
    except FPGAValueError:
        return 0
    count = get_count(pzt_tree["Scan/Count Time (ms)"])
    return count

def set_xy_and_count(y,x):
    try:
        set_jpe_pos(x,y,None,write=False)
    except FPGAValueError:
        return 0
    count = get_count(pzt_tree["Scan/Count Time (ms)"])
    return count

def do_cav_scan_step():
    if not dpg.get_value("pzt_3d_scan"):
        return [-1]

    steps = pzt_tree["Scan/Cavity/Steps"]
    centers = pzt_tree["Scan/Cavity/Center"]
    spans = pzt_tree["Scan/Cavity/Span"]
    jpe_cav_scan.steps = [steps]
    jpe_cav_scan.centers = [centers]
    jpe_cav_scan.spans = [spans]
    cav_data = {}

    def init():
        log.debug("Starting cav scan sub.")
        pos = jpe_cav_scan._get_positions()
        cav_data['counts'] = []
        cav_data['pos'] = []
        xmin = np.min(pos[0])
        xmax = np.max(pos[0])
        position_register["temp_cav_position"] = fpga.get_cavity()
        if pzt_tree['Plot/Autoscale']:
            dpg.set_axis_limits("cav_count_x",xmin,xmax)
        else:
            dpg.set_axis_limits_auto("cav_count_x")
    
    def abort(i,imax,idx,pos,res):
        return not dpg.get_value("pzt_3d_scan")

    def prog(i,imax,idx,pos,res):
        log.debug("Updating Cav Scan Plot")
        cav_data['counts'].append(res)
        cav_data['pos'].append(pos[0])
        check = pzt_tree["Plot/Update Every Point"] or i + 1 >= imax
        if check:
            dpg.set_value("cav_counts",[cav_data['pos'],cav_data['counts']])
        if count_tree["Counts/Plot Scan Counts"]:
            plot_counts()

    def finish(results,completed):
        if pzt_tree['Plot/Autoscale']:
            dpg.set_axis_limits("cav_count_y",0,np.max(results))
        else:
            dpg.set_axis_limits_auto("cav_count_y")
        set_cav_pos(*position_register["temp_cav_position"])

    jpe_cav_scan._init_func = init
    jpe_cav_scan._abort_func = abort
    jpe_cav_scan._prog_func = prog
    jpe_cav_scan._finish_func = finish
    return jpe_cav_scan.run_async()

def set_xy_get_cav(y,x):
    try:
        set_jpe_pos(x,y,None,write=False)
        do_cav_scan_step().join()
    except FPGAValueError:
        return np.array([-1])
    return jpe_cav_scan.results

jpe_xy_scan = Scanner(set_xy_and_count,[0,0],[1,1],[50,50],[1],[],float,['y','x'],default_result=-1)
jpe_cav_scan = Scanner(set_cav_and_count,[0],[1],[50],[],[],float,['z'],default_result=-1)
jpe_3D_scan = Scanner(set_xy_get_cav,[0,0],[1,1],[50,50],[1],[],object,['y','x'],default_result=np.array([-1]))

def start_xy_scan():
    if not dpg.get_value("pzt_xy_scan"):
        return -1
    abort_counts()

    fpga.set_ao_wait(pzt_tree["Scan/Wait Time (ms)"],write=False)
    steps = pzt_tree["Scan/JPE/Steps"][:2][::-1]
    centers = pzt_tree["Scan/JPE/Center"][:2][::-1]
    spans = pzt_tree["Scan/JPE/Span"][:2][::-1]
    jpe_xy_scan.steps = steps
    jpe_xy_scan.centers = centers
    jpe_xy_scan.spans = spans
    
    def init():
        pos = jpe_xy_scan._get_positions()
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        position_register["temp_jpe_position"] = fpga.get_jpe_pzs()
        pzt_plot.set_size(int(jpe_xy_scan.steps[0]),int(jpe_xy_scan.steps[1]))
        pzt_plot.set_bounds(xmin,xmax,ymin,ymax)
        dpg.configure_item("Piezo Scan_heat_series",label="2D Scan")
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                if control != "pzt_xy_scan":
                    dpg.disable_item(control)
        for param in piezo_params:
            dpg.disable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.disable_cursor()
    
    def abort(i,imax,idx,pos,res):
        return not dpg.get_value("pzt_xy_scan")

    def prog(i,imax,idx,pos,res):
            log.debug("Setting Progress Bar")
            dpg.set_value("pb",(i+1)/imax)
            dpg.configure_item("pb",overlay=f"JPE XY Scan {i+1}/{imax}")
            if pzt_tree["Plot/Update Every Point"]:
                check = True
            else:
                check = (not (i+1) % pzt_tree["Scan/JPE/Steps"][1]) or (i+1)==imax
            if check:
                log.debug("Updating XY Scan Plot")
                update_pzt_plot("manual",None,None)
                if count_tree["Counts/Plot Scan Counts"]:
                    plot_counts()

    def finish(results,completed):
        dpg.set_value("pzt_xy_scan",False)
        set_jpe_pos(*position_register["temp_jpe_position"])
        if dpg.get_value("pzt_auto_save"):
            save_xy_scan()
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                dpg.enable_item(control)
        for param in piezo_params:
            dpg.enable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.enable_cursor()
    jpe_xy_scan._init_func = init
    jpe_xy_scan._abort_func = abort
    jpe_xy_scan._prog_func = prog
    jpe_xy_scan._finish_func = finish
    return jpe_xy_scan.run_async()

def get_xy_range():
    xmin,xmax,ymin,ymax = pzt_plot.query_plot()
    if xmin is not None:
        new_centers = [(xmin+xmax)/2, (ymin+ymax)/2]
        new_spans = [xmax-xmin, ymax-ymin]
        pzt_tree["Scan/JPE/Center"] = new_centers
        pzt_tree["Scan/JPE/Span"] = new_spans
    else:
        pzt_tree["Scan/JPE/Center"] = pzt_tree["JPE/XY Position"]

def start_cav_scan():
    if not dpg.get_value("pzt_cav_scan"):
        return -1
    abort_counts()

    steps = pzt_tree["Scan/Cavity/Steps"]
    centers = pzt_tree["Scan/Cavity/Center"]
    spans = pzt_tree["Scan/Cavity/Span"]
    jpe_cav_scan.steps = [steps]
    jpe_cav_scan.centers = [centers]
    jpe_cav_scan.spans = [spans]
    cav_data = {}

    def init():
        fpga.set_ao_wait(pzt_tree["Scan/Wait Time (ms)"],write=False)
        pos = jpe_cav_scan._get_positions()
        cav_data['counts'] = []
        cav_data['pos'] = []
        xmin = np.min(pos[0])
        xmax = np.max(pos[0])
        position_register["temp_cav_position"] = fpga.get_cavity()
        if pzt_tree['Plot/Autoscale']:
            dpg.set_axis_limits("cav_count_x",xmin,xmax)
        else:
            dpg.set_axis_limits_auto("cav_count_x")
        dpg.configure_item("pzt_tree_Plot/3D/Slice Index",max_value=pzt_tree["Scan/Cavity/Steps"]-1)
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                if control != "pzt_cav_scan":
                    dpg.disable_item(control)
        for param in piezo_params:
            dpg.disable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.disable_cursor()
    
    def abort(i,imax,idx,pos,res):
        return not dpg.get_value("pzt_cav_scan")

    def prog(i,imax,idx,pos,res):
        log.debug("Setting Progress Bar")
        dpg.set_value("pb",(i+1)/imax)
        dpg.configure_item("pb",overlay=f"JPE XY Scan {i+1}/{imax}")
        log.debug("Updating XY Scan Plot")
        cav_data['counts'].append(res)
        cav_data['pos'].append(pos[0])
        dpg.set_value("cav_counts",[cav_data['pos'],cav_data['counts']])
        if count_tree["Counts/Plot Scan Counts"]:
            plot_counts()

    def finish(results,completed):
        dpg.set_value("pzt_cav_scan",False)
        if pzt_tree['Plot/Autoscale']:
            dpg.set_axis_limits("cav_count_y",0,np.max(results))
        else:
            dpg.set_axis_limits_auto("cav_count_y")
        set_cav_pos(*position_register["temp_cav_position"],write=True)
        if dpg.get_value("pzt_auto_save"):
            save_cav_scan()
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                dpg.enable_item(control)
        for param in piezo_params:
            dpg.enable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.enable_cursor()

    jpe_cav_scan._init_func = init
    jpe_cav_scan._abort_func = abort
    jpe_cav_scan._prog_func = prog
    jpe_cav_scan._finish_func = finish
    return jpe_cav_scan.run_async()

def get_cav_range():
    if dpg.is_plot_queried("cav_plot"):
        xmin,xmax,ymin,ymax = dpg.get_plot_query_area("cav_plot")
        new_center = (xmin+xmax)/2
        new_span = xmax-xmin
        pzt_tree["Scan/Cavity/Center"] = new_center
        pzt_tree["Scan/Cavity/Span"] = new_span
    else:
        pzt_tree["Scan/Cavity/Center"] = pzt_tree["Cavity/Position"]

def get_reducing_func():
    func_str = pzt_tree["Plot/3D/Reducing Func."]
    if func_str == "Delta":
        f = lambda d:np.max(d) - np.min(d) if np.atleast_1d(d)[0] >= 0 else -1.0
        do_func = np.vectorize(f)
        return do_func
    if func_str == "Max":
        f = lambda d:np.max(d) if np.atleast_1d(d)[0] >= 0 else -1.0
        do_func = np.vectorize(f)
        return do_func
    if func_str == "Average":
        f = lambda d:np.mean(d) if np.atleast_1d(d)[0] >= 0 else -1.0
        do_func = np.vectorize(f)
        return do_func
    if func_str == "Slice":
        f = lambda d:d[pzt_tree["Plot/3D/Slice Index"]] if np.atleast_1d(d)[0] >= 0 else -1.0
        do_func = np.vectorize(f)
        return do_func

def start_3d_scan():
    if not dpg.get_value("pzt_3d_scan"):
        return -1
    abort_counts()

    jpe_steps = pzt_tree["Scan/JPE/Steps"][:2][::-1]
    jpe_centers = pzt_tree["Scan/JPE/Center"][:2][::-1]
    jpe_spans = pzt_tree["Scan/JPE/Span"][:2][::-1]
    jpe_3D_scan.steps = jpe_steps
    jpe_3D_scan.centers = jpe_centers
    jpe_3D_scan.spans = jpe_spans

    def init():
        fpga.set_ao_wait(pzt_tree["Scan/Wait Time (ms)"],write=False)
        pos = jpe_3D_scan._get_positions()
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        position_register["temp_jpe_position"] = fpga.get_jpe_pzs()
        position_register["temp_cav_position"] = fpga.get_cavity()
        pzt_plot.set_size(int(jpe_3D_scan.steps[0]),int(jpe_3D_scan.steps[1]))
        pzt_plot.set_bounds(xmin,xmax,ymin,ymax)
        dpg.configure_item("Piezo Scan_heat_series",label="3D Scan")
        dpg.configure_item("pzt_tree_Plot/3D/Slice Index",max_value=pzt_tree["Scan/Cavity/Steps"]-1)
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                if control != "pzt_3d_scan":
                    dpg.disable_item(control)
        for param in piezo_params:
            dpg.disable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.disable_cursor()
    
    def abort(i,imax,idx,pos,res):
        return not dpg.get_value("pzt_3d_scan")

    def prog(i,imax,idx,pos,res):
            log.debug("Setting Progress Bar")
            dpg.set_value("pb",(i+1)/imax)
            dpg.configure_item("pb",overlay=f"JPE 3D Scan {i+1}/{imax}")
            if pzt_tree["Plot/Update Every Point"]:
                check = True
            else:
                check = (not (i+1) % jpe_3D_scan.steps[1]) or (i+1)==imax
            if check:
                log.debug("Updating 3D Scan Plot")
                update_pzt_plot("manual",None,None)
                if count_tree["Counts/Plot Scan Counts"]:
                    plot_counts()

    def finish(results,completed):
        dpg.set_value("pzt_3d_scan",False)
        set_jpe_pos(*position_register["temp_jpe_position"])
        set_cav_pos(*position_register["temp_cav_position"])
        if dpg.get_value("pzt_auto_save"):
            save_xy_scan()
        for controls in [galvo_controls,optim_controls,objective_controls,piezo_controls]:
            for control in controls:
                dpg.enable_item(control)
        for param in piezo_params:
            dpg.enable_item(param)
        for plot in [galvo_plot,pzt_plot,obj_plot]:
            plot.enable_cursor()

    jpe_3D_scan._init_func = init
    jpe_3D_scan._abort_func = abort
    jpe_3D_scan._prog_func = prog
    jpe_3D_scan._finish_func = finish
    return jpe_3D_scan.run_async()

def update_pzt_plot(sender,app_data,user_data):
    if "3D" in dpg.get_item_configuration("Piezo Scan_heat_series")['label']:
        log.debug("Updating 3D Scan Plot")
        func = get_reducing_func()
        plot_data = func(np.copy(np.flip(jpe_3D_scan.results,0)))
    elif "2D" in dpg.get_item_configuration("Piezo Scan_heat_series")['label']:
        plot_data = np.copy(np.flip(jpe_xy_scan.results,0))
    if pzt_tree["Plot/Deinterleave"]:
        if not pzt_tree["Plot/Reverse"]:
            plot_data[::2,:] = plot_data[1::2,:]
        else:
            plot_data[1::2,:] = plot_data[::2,:]
    pzt_plot.autoscale = pzt_tree["Plot/Autoscale"]
    pzt_plot.nbin = pzt_tree["Plot/N Bins"]
    pzt_plot.update_plot(plot_data)
    if sender == "cav_count_cut":
        volts = dpg.get_value("cav_count_cut")
        index = np.argmin(np.abs(jpe_cav_scan.positions[0]-volts))
        pzt_tree['Plot/3D/Slice Index'] = int(index)
    if sender == "pzt_tree_Plot/3D/Slice Index":
        index = app_data
        volts = jpe_cav_scan.positions[0][index]
        dpg.set_value("cav_count_cut",volts)

def guess_piezo_time():
    pts = pzt_tree["Scan/JPE/Steps"]
    ctime = pzt_tree["Scan/Count Time (ms)"] + pzt_tree["Scan/Wait Time (ms)"]
    scan_time = pts[0] * pts[1] * ctime / 1000
    time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
    pzt_tree["Scan/JPE/Estimated Time"] = time_string

def guess_3d_time():
    jpe_pts = pzt_tree["Scan/JPE/Steps"]
    cav_pts = pzt_tree["Scan/Cavity/Steps"]
    total_pts = jpe_pts[0]*jpe_pts[1]*cav_pts
    ctime = pzt_tree["Scan/Count Time (ms)"] + pzt_tree["Scan/Wait Time (ms)"]
    scan_time = total_pts * ctime / 1000
    time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
    pzt_tree["Scan/Estimated Time"] = time_string

def guess_pzt_times(*args):
    guess_piezo_time()
    guess_3d_time()

def xy_pos_callback(sender,app_data,user_data):
    try:
        set_jpe_pos(app_data[0],app_data[1],None)
    except FPGAValueError:
        pass

def z_pos_callback(sender,app_data,user_data):
    try:
        set_jpe_pos(None,None,app_data)
    except FPGAValueError:
        pass

def set_jpe_pos(x=None,y=None,z=None,write=True):
    current_pos = fpga.get_jpe_pzs()
    if x is None:
        x = current_pos[0]
    if y is None:
        y = current_pos[1]
    if z is None:
        z = current_pos[2]
    volts = pz_conv.zs_from_cart([x,y,z])
    in_bounds = pz_conv.check_bounds(x,y,z)
    if not in_bounds:
        pzt_tree["JPE/XY Position"] = current_pos[:2]
        pzt_tree["JPE/Z Position"] = current_pos[2]
        raise FPGAValueError("Out of bounds")

    pzt_tree["JPE/Z Volts"] = volts
    pzt_tree["JPE/Z Position"] = z
    pzt_tree["JPE/XY Position"] = [x,y]
    fpga.set_jpe_pzs(x,y,z,write=write)
    pzt_plot.set_cursor([x,y])
    log.debug(f"Set JPE Position to ({x},{y},{z})")
    draw_bounds()

def set_cav_pos(z,write=False):
    fpga.set_cavity(z,write=write)
    pzt_tree["Cavity/Position"] = z
    log.debug(f"Set cavity to {z} V.")

def man_set_cavity(sender,app_data,user_data):
    set_cav_pos(app_data,True)

def draw_bounds():
    zpos = pzt_tree['JPE/Z Position']
    bound_points = list(pz_conv.bounds('z',zpos))
    bound_points.append(bound_points[0])
    if dpg.does_item_exist('pzt_bounds'):
        dpg.delete_item('pzt_bounds')
    dpg.draw_polygon(bound_points,tag='pzt_bounds',parent="Piezo Scan_plot")

def xy_cursor_callback(sender,position):
    zpos = pzt_tree['JPE/Z Position']
    cur_xy = pzt_tree['JPE/XY Position'][:2]
    if not pz_conv.check_bounds(position[0],position[1],zpos):
        pzt_plot.set_cursor(cur_xy)
    else:
        set_jpe_pos(position[0],position[1],zpos)
pzt_plot.cursor_callback=xy_cursor_callback

def save_xy_scan(*args):
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_pzt_file")
    path /= filename
    as_npz = not (".csv" in filename)
    header = jpe_xy_scan.gen_header("XY Piezo Scan")
    jpe_xy_scan.save_results(str(path),as_npz=as_npz,header=header)
def save_3d_scan(*args):
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_pzt_file")
    path /= filename
    as_npz = not (".csv" in filename)
    header = jpe_3D_scan.gen_header("3D Piezo Scan")
    header += f"cavity center, {repr(jpe_cav_scan.centers)}\n"
    header += f"cavity spans, {repr(jpe_cav_scan.spans)}\n"
    header += f"cavity steps, {repr(jpe_cav_scan.steps)}\n"

    jpe_3D_scan.save_results(str(path),as_npz=as_npz,header=header)
def save_cav_scan(*args):
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_pzt_file")
    path /= filename
    as_npz = not (".csv" in filename)
    header = jpe_cav_scan.gen_header("Cavity Piezo Scan")
    jpe_cav_scan.save_results(str(path),as_npz=as_npz,header=header)

################################################################################
############################### UI Building ####################################
################################################################################
rdpg.initialize_dpg("Cryocontrol",docking=False)
with dpg.theme(tag="avg_count_theme"):
    with dpg.theme_component(dpg.mvLineSeries):
        dpg.add_theme_color(dpg.mvPlotCol_Line, (252, 167, 130), category=dpg.mvThemeCat_Plots)
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 6, category=dpg.mvThemeCat_Plots)
###############
# Main Window #
###############
#with dpg.window(label="Count Rate", tag="count_window",pos=[0,750],width=425):
#    dpg.add_text('0',tag="count_rate",parent="count_window")
#    dpg.bind_item_font("count_rate","massive_font")

with dpg.window(label="Cryocontrol", tag='main_window'):
    ##################
    # Persistant Bar #
    ##################
    with dpg.group(horizontal=True):
        with dpg.child_window(width=-425,height=125):
            # Data Directory
            with dpg.group(horizontal=True):
                dpg.add_text("Data Directory:")
                dpg.add_input_text(default_value="X:\\DiamondCloud\\Cryostat Setup", tag="save_dir",width=-1)
                dpg.add_button(label="Pick Directory", callback=choose_save_dir)
            # Counts and Optimization
            with dpg.group(horizontal=True):
                dpg.add_checkbox(tag="count", label="Count", callback=start_counts)
                dpg.add_button(tag="clear_counts", label="Clear Counts",callback=clear_counts)
                dpg.add_button(tag="optimize", label="Optimize Galvo", callback=optimize_galvo)
            # Confocal Scan Control
            with dpg.group(horizontal=True):
                dpg.add_progress_bar(label="Scan Progress",tag='pb',width=-1)

        with dpg.child_window(width=425,height=125,no_scrollbar=True):
            dpg.add_text('0',tag="count_rate",parent="count_window")
            dpg.bind_item_font("count_rate","massive_font")
    ##############
    # START TABS #
    ##############
    with dpg.tab_bar():
        ###############
        # COUNTER TAB #
        ###############
        with dpg.tab(label="Counter"):
            with dpg.group(horizontal=True):
                    dpg.add_text("Filename:")
                    dpg.add_input_text(tag="save_counts_file", default_value="counts.npz", width=200)
                    dpg.add_button(tag="save_counts", label="Save Counts",callback=save_counts)
            with dpg.group(horizontal=True,width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="count_tree"):
                    count_tree = rdpg.TreeDict('count_tree','cryo_gui_settings/count_tree_save.csv')
                    count_tree.add("Counts/Count Time (ms)", 10,
                                   item_kwargs={'min_value':1,'min_clamped':True},
                                   tooltip="How long the fpga acquires counts for.")
                    count_tree.add("Counts/Max Points", 100000,
                                   item_kwargs={'on_enter':True,'min_value':1,'min_clamped':True,'step':100},
                                   tooltip="How many plot points to display before cutting old ones.")
                    count_tree.add("Counts/Average Points", 5, callback=plot_counts,
                                   item_kwargs={'min_value':1,'min_clamped':True},
                                   tooltip="Size of moving average window.")
                    count_tree.add("Counts/Plot Scan Counts", True, 
                                   callback=plot_counts,
                                   tooltip="Wether to plot counts acquired during other scanning procedures.")
                    count_tree.add("Counts/Show AI1", True, callback=toggle_AI)
                with dpg.child_window(width=-1,autosize_y=True):
                    with dpg.plot(label="Count Rate",width=-1,height=-1,tag="count_plot"):
                        dpg.bind_font("plot_font") 
                        # REQUIRED: create x and y axes
                        dpg.add_plot_axis(dpg.mvXAxis, label="Time", time=True, tag="count_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Counts",tag="count_y")
                        dpg.add_plot_axis(dpg.mvYAxis,label="Sync", tag="count_AI1",no_gridlines=True)
                        dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                            counts_data['counts'],
                                            parent='count_y',label='counts', tag='counts_series')
                        dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                            counts_data['counts'],
                                            parent='count_y',label='avg. counts', tag='avg_counts_series')
                        dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                            counts_data['AI1'],
                                            parent='count_AI1',label='AI1', tag='AI1_series')
                        dpg.add_plot_legend()
                        dpg.bind_item_theme("counts_series","plot_theme_blue")
                        dpg.bind_item_theme("avg_counts_series","avg_count_theme")
                        dpg.bind_item_theme("AI1_series","plot_theme_purple")
                        
        #############
        # GALVO TAB #
        #############
        galvo_controls = ["galvo_tree_Galvo/Position",
                          "galvo_scan"]
        galvo_params = ["galvo_tree_Scan/Centers (V)",
                        "galvo_tree_Scan/Spans (V)",
                        "galvo_tree_Scan/Points",
                        "galvo_tree_Scan/Count Time (ms)",
                        "galvo_tree_Scan/Wait Time (ms)"]
        with dpg.tab(label="Galvo"):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(tag="galvo_scan",label="Scan Galvo", callback=start_scan)
                dpg.add_button(tag="query_plot",label="Copy Scan Params",callback=get_galvo_range)
                dpg.add_text("Filename:")
                dpg.add_input_text(tag="save_galvo_file", default_value="scan.npz", width=200)
                dpg.add_button(tag="save_galvo_button", label="Save Scan",callback=save_galvo_scan)
                dpg.add_checkbox(tag="auto_save_galvo", label="Auto")
            with dpg.group(horizontal=True, width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="galvo_tree"):
                    galvo_tree = rdpg.TreeDict('galvo_tree','cryo_gui_settings/scan_params_save.csv')
                    galvo_tree.add("Galvo/Position",[float(f) for f in fpga.get_galvo()],
                                    item_kwargs={'min_value':-10,
                                                'max_value':10,
                                                "min_clamped":True,
                                                "max_clamped":True,
                                                "on_enter":True},
                                    callback=man_set_galvo)
                    galvo_tree.add("Plot/Autoscale",False,tooltip="Autoscale the plot at each update")
                    galvo_tree.add("Plot/N Bins",50,item_kwargs={'min_value':1,
                                                                 'max_value':1000},
                                   tooltip="Number of bins in plot histogram")
                    galvo_tree.add("Plot/Update Every Point",False,
                                   tooltip="Update the plot at every position vs each line, slows down scans.")
                    galvo_tree.add("Scan/Centers (V)",[0.0,0.0],item_kwargs={'min_value':-10,
                                                                             'max_value':10,
                                                                             "min_clamped":True,
                                                                             "max_clamped":True},
                                    tooltip="Central position of scan in volts.")
                    galvo_tree.add("Scan/Spans (V)", [1.0,1.0],item_kwargs={'min_value':-20,
                                                                            'max_value':20,
                                                                            "min_clamped":True,
                                                                            "max_clamped":True},
                                    tooltip="Width of scan in volts.")
                    galvo_tree.add("Scan/Points", [100,100],item_kwargs={'min_value':0},
                                callback=guess_galvo_time,
                                tooltip="Number of points along the scan axes.")
                    galvo_tree.add("Scan/Count Time (ms)", 10.0, item_kwargs={'min_value':0,
                                                                              'max_value':1000,
                                                                              'step':1},
                                callback=guess_galvo_time,
                                tooltip="How long the fpga counts at each position.")
                    galvo_tree.add("Scan/Wait Time (ms)", 1.0,item_kwargs={'min_value':0,
                                                                        'max_value':1000,
                                                                        "step":1},
                                callback=guess_galvo_time,
                                tooltip="How long the fpga waits before counting, after moving.")
                    galvo_tree.add("Scan/Estimated Time", "00:00:00", item_kwargs={'readonly':True},
                                   save=False,
                                   tooltip="Rough guess of how long scan will take = (count_time + wait_time)*Npts")
                with dpg.group():
                    galvo_plot.parent = dpg.last_item()
                    galvo_plot.height = -330
                    galvo_plot.scale_width = 335
                    galvo_plot.make_gui()
                    with dpg.child_window(width=-0,height=320):
                        with dpg.plot(label="Count Rate",width=-1,height=300,tag="count_plot2"):
                            dpg.bind_font("plot_font") 
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="x", time=True, tag="count_x2")
                            dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="count_y2")
                            dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="count_AI12",no_gridlines=True)
                            dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                                counts_data['counts'],
                                                parent='count_y2',label='counts', tag='counts_series2')
                            dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                                counts_data['counts'],
                                                parent='count_y2',label='avg. counts', tag='avg_counts_series2')
                            dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                                counts_data['AI1'],
                                                parent='count_AI12',label='AI1', tag='AI1_series2')
                            dpg.set_item_source('counts_series2','counts_series')
                            dpg.set_item_source('avg_counts_series2','avg_counts_series')
                            dpg.set_item_source('AI1_series2','AI1_series')
                            dpg.bind_item_theme("counts_series2","plot_theme_blue")
                            dpg.bind_item_theme("avg_counts_series2","avg_count_theme")
                            dpg.bind_item_theme("AI1_series2","plot_theme_purple")
                            dpg.add_plot_legend()
        #################
        # Optimizer Tab #
        #################
        optim_controls = ["optimize"]
        optim_params = ["optim_tree_Optimizer/Count Time (ms)",
                        "optim_tree_Optimizer/Wait Time (ms)",
                        "optim_tree_Optimizer/Scan Points",
                        "optim_tree_Optimizer/Scan Range (XY)",
                        "optim_tree_Optimizer/Iterations"]
        with dpg.tab(label="Galvo Optimizer"):
            with dpg.group(horizontal=True,width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="optim_tree"):
                    optim_tree = rdpg.TreeDict('optim_tree','cryo_gui_settings/optim_tree_save.csv')
                    optim_tree.add("Optimizer/Count Time (ms)", 10.0,item_kwargs={'min_value':1.0,'min_clamped':True,'step':1},
                                   tooltip="How long the fpga counts at each position.")
                    optim_tree.add("Optimizer/Wait Time (ms)", 10.0,item_kwargs={'min_value':1.0,'min_clamped':True,'step':1},
                                   tooltip="How long the fpga waits before counting after moving.")
                    optim_tree.add("Optimizer/Scan Points", 50,item_kwargs={'min_value':2,'min_clamped':True},
                                   tooltip="Number of points to scan along each axis.")
                    optim_tree.add("Optimizer/Scan Range (XY)", 0.1,item_kwargs={'min_value':0.0,'min_clamped':True,'step':0},
                                   tooltip="Size of scan along each axis in volts.")
                    optim_tree.add("Optimizer/Iterations", 1,item_kwargs={'min_value':1,'min_clamped':True},
                                   tooltip="How many times to rerun the optimization around each new point.")
                with dpg.child_window(width=-1,autosize_y=True): 
                    with dpg.subplots(1,2,label="Optimizer",width=-1,height=-1,tag="optim_plot"):
                        with dpg.plot(label="X Scan", tag="optim_x"):
                            dpg.bind_font("plot_font") 
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="optim_x_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="optim_x_y")
                            dpg.add_line_series([0],[0],
                                                parent='optim_x_y',label='counts', tag='optim_x_counts')
                            dpg.add_line_series([0],[0],
                                                parent='optim_x_y',label='fit', tag='optim_x_fit')
                            dpg.add_plot_legend()
                        with dpg.plot(label="Y Scan", tag="optim_y"):
                            dpg.bind_font("plot_font") 
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="optim_y_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="optim_y_y")
                            dpg.add_line_series([0],[0],
                                                parent='optim_y_y',label='counts', tag='optim_y_counts')
                            dpg.add_line_series([0],[0],
                                                parent='optim_y_y',label='fit', tag='optim_y_fit')
                            dpg.add_plot_legend()
        #################
        # Objective Tab #
        #################
        objective_controls = ["obj_tree_Objective/Initialize",
                              "obj_tree_Objective/Set Position (um)",
                              "obj_tree_Objective/Limits (um)",
                              "obj_tree_Objective/Max Move (um)",
                              "obj_scan"]
        objective_params = ["obj_tree_Scan/Count Time (ms)",
                            "obj_tree_Scan/Wait Time (ms)",
                            "obj_tree_Scan/Obj./Center (um)",
                            "obj_tree_Scan/Obj./Span (um)",
                            "obj_tree_Scan/Obj./Steps",
                            "obj_tree_Scan/Galvo/Axis",
                            "obj_tree_Scan/Galvo/Center (V)",
                            "obj_tree_Scan/Galvo/Span (V)",
                            "obj_tree_Scan/Galvo/Steps"]
        with dpg.tab(label="Objective Control"):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Scan Objective",tag="obj_scan", default_value=False, callback=start_obj_scan)
                dpg.add_button(tag="query_obj_plot",label="Copy Scan Params",callback=get_obj_range)
                dpg.add_text("Filename:")
                dpg.add_input_text(tag="save_obj_file", default_value="scan.npz", width=200)
                dpg.add_button(tag="save_obj_button", label="Save Scan",callback=save_obj_scan)
                dpg.add_checkbox(tag="auto_save_obj", label="Auto")
            with dpg.group(horizontal=True,width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="obj_tree"):
                    obj_tree = rdpg.TreeDict('obj_tree','cryo_gui_settings/obj_tree_save.csv')
                    obj_tree.add("Objective/Initialize", False,save=False,callback=toggle_objective)
                    obj_tree.add("Objective/Status","Uninitialized",save=False,item_kwargs={"readonly":True})
                    obj_tree.add("Objective/Set Position (um)", 100.0,callback=set_obj_callback,item_kwargs={"on_enter":True,'step':0})
                    obj_tree.add("Objective/Current Position (um)", 100.0,item_kwargs={"readonly":True,'step':0})
                    obj_tree.add("Objective/Limits (um)",[-8000.0,8000.0],callback=set_objective_params,item_kwargs={"on_enter":True})
                    obj_tree.add("Objective/Max Move (um)", 100.0,callback=set_objective_params,item_kwargs={"on_enter":True,'step':1})
                    obj_tree.add("Objective/Rel. Step (um)", 5.0,item_kwargs={"min_value":0,"min_clamped":True,"step":1})
                    obj_tree.add("Scan/Count Time (ms)", 10.0,callback=guess_obj_time,item_kwargs={'min_value':0,'min_clamped':True,"step":1})
                    obj_tree.add("Scan/Wait Time (ms)", 5.0,callback=guess_obj_time,item_kwargs={'min_value':0,'min_clamped':True,"step":1})
                    obj_tree.add("Scan/Obj./Center (um)",0.0, item_kwargs={"step":0})
                    obj_tree.add("Scan/Obj./Span (um)",50.0, item_kwargs={"step":0})
                    obj_tree.add("Scan/Obj./Steps",50,callback=guess_obj_time, item_kwargs={"step":0})
                    obj_tree.add_radio("Scan/Galvo/Axis",['x','y'],'x')
                    obj_tree.add("Scan/Galvo/Center (V)",0.0, item_kwargs={"step":0})
                    obj_tree.add("Scan/Galvo/Span (V)",0.05, item_kwargs={"step":0})
                    obj_tree.add("Scan/Galvo/Steps",50,callback=guess_obj_time, item_kwargs={"step":0})
                    obj_tree.add("Scan/Estimated Time", "00:00:00", item_kwargs={"readonly":True})
                    obj_tree.add("Plot/Autoscale",False)
                    obj_tree.add("Plot/N Bins",50,item_kwargs={'min_value':1,
                                                                    'max_value':1000})
                    obj_tree.add("Plot/Update Every Point",False)

                with dpg.child_window(width=0,height=0,autosize_y=True):
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=100):
                            dpg.add_button(width=0,indent=25,arrow=True,direction=dpg.mvDir_Up,callback=obj_step_up)
                            dpg.add_button(width=0,indent=25,arrow=True,direction=dpg.mvDir_Down,callback=obj_step_down)
                            dpg.add_text('Obj. Pos.')
                            dpg.bind_item_font(dpg.last_item(),"small_font")
                            with dpg.group(horizontal=True):
                                dpg.bind_item_font(dpg.last_item(),"small_font")
                                dpg.add_slider_float(height=600,width=35,default_value=0,
                                                     min_value=obj._soft_lower, vertical=True,
                                                     max_value=obj._soft_upper,tag='obj_pos_set',
                                                     enabled=False,no_input=True)
                                dpg.add_slider_float(height=600,width=35,default_value=0,
                                                     min_value=obj._soft_lower, vertical=True,
                                                     max_value=obj._soft_upper,tag='obj_pos_get',
                                                     enabled=False,no_input=True)
                                
                        with dpg.group():
                            obj_plot.parent = dpg.last_item()
                            obj_plot.height = -330
                            obj_plot.scale_width = 335
                            obj_plot.make_gui()
                            with dpg.child_window(width=-0,height=320):
                                with dpg.plot(label="Count Rate",width=-1,height=300,tag="count_plot3"):
                                    dpg.bind_font("plot_font") 
                                    # REQUIRED: create x and y axes
                                    dpg.add_plot_axis(dpg.mvXAxis, label="x", time=True, tag="count_x3")
                                    dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="count_y3")
                                    dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="count_AI13",no_gridlines=True)
                                    dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                                        counts_data['counts'],
                                                        parent='count_y3',label='counts', tag='counts_series3')
                                    dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                                        counts_data['counts'],
                                                        parent='count_y3',label='avg. counts', tag='avg_counts_series3')
                                    dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                                        counts_data['AI1'],
                                                        parent='count_AI13',label='AI1', tag='AI1_series3')
                                    dpg.set_item_source('counts_series3','counts_series')
                                    dpg.set_item_source('avg_counts_series3','avg_counts_series')
                                    dpg.set_item_source('AI1_series3','AI1_series')
                                    dpg.bind_item_theme("counts_series3","plot_theme_blue")
                                    dpg.bind_item_theme("avg_counts_series3","avg_count_theme")
                                    dpg.bind_item_theme("AI1_series3","plot_theme_purple")
                                    dpg.add_plot_legend()
        #############
        # Piezo Tab #
        #############
        piezo_controls = ["pzt_tree_JPE/Z Position",
                          "pzt_tree_JPE/XY Position",
                          "pzt_tree_Cavity/Position",
                          "pzt_xy_scan",
                          "pzt_cav_scan",
                          "pzt_3d_scan"]
        piezo_params = ["pzt_tree_Scan/Wait Time (ms)",
                        "pzt_tree_Scan/Count Time (ms)",
                        "pzt_tree_Scan/Cavity/Center",
                        "pzt_tree_Scan/Cavity/Span",
                        "pzt_tree_Scan/Cavity/Steps",
                        "pzt_tree_Scan/JPE/Center",
                        "pzt_tree_Scan/JPE/Span",
                        "pzt_tree_Scan/JPE/Steps"]

        with dpg.tab(label="Piezo Control"):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(tag="pzt_xy_scan",label="Scan XY", callback=start_xy_scan)
                dpg.add_checkbox(tag="pzt_cav_scan",label="Scan Cav.", callback=start_cav_scan)
                dpg.add_checkbox(tag="pzt_3d_scan",label="Scan 3D", callback=start_3d_scan)
                dpg.add_text("Filename:")
                dpg.add_input_text(tag="save_pzt_file", default_value="scan.npz", width=200)
                dpg.add_text("Save:")
                dpg.add_button(label="XY", tag="save_xy_button",callback=save_xy_scan)
                dpg.add_button(label="3D", tag="save_3d_button",callback=save_3d_scan)
                dpg.add_button(label="Cav", tag="save_cav_button",callback=save_cav_scan)
                dpg.add_checkbox(tag="pzt_auto_save", label="Auto",default_value=False)
                dpg.add_button(tag="query_xy_plot",label="XY Copy Scan Params",callback=get_xy_range)
                dpg.add_button(tag="query_cav_plot",label="Cav. Copy Scan Params",callback=get_cav_range)
                
            with dpg.group(horizontal=True,width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="pzt_tree"):
                    pzt_tree = rdpg.TreeDict('pzt_tree','cryo_gui_settings/pzt_tree_save.csv')
                    pzt_tree.add("JPE/Z Position",0.0,
                                 item_kwargs={'min_value':-6.5,'max_value':0,
                                              'min_clamped':True,'max_clamped':True,
                                              'on_enter':True,'step':0.05},
                                              callback=z_pos_callback,save=False)
                    pzt_tree.add("JPE/XY Position",[0.0,0.0],
                                 item_kwargs={'on_enter':True},
                                 callback=xy_pos_callback,save=False)
                    pzt_tree.add("JPE/Z Volts", [0.0,0.0,0.0], save=False,
                                 item_kwargs={"readonly" : True})

                    pzt_tree.add("Cavity/Position",0.0,
                                 item_kwargs={'min_value':-8,'max_value':8,
                                              'min_clamped':True,'max_clamped':True,
                                              'on_enter':True,'step':0.5},
                                 callback=man_set_cavity, save=False)
                    pzt_tree.add("Scan/Wait Time (ms)",10.0,callback=guess_pzt_times,item_kwargs={'step':1})
                    pzt_tree.add("Scan/Count Time (ms)",5.0,callback=guess_pzt_times,item_kwargs={'step':1})
                    pzt_tree.add("Scan/Cavity/Center",0.0, item_kwargs={"step":0})
                    pzt_tree.add("Scan/Cavity/Span",16.0, item_kwargs={"step":0})
                    pzt_tree.add("Scan/Cavity/Steps",300,callback=guess_pzt_times, item_kwargs={"step":0})
                    pzt_tree.add("Scan/JPE/Center",[0.0,0.0])
                    pzt_tree.add("Scan/JPE/Span",[5.0,5.0])
                    pzt_tree.add("Scan/JPE/Steps",[15,15],callback=guess_pzt_times)
                    pzt_tree.add("Scan/JPE/Estimated Time", "00:00:00", save=False,item_kwargs={'readonly':True})
                    pzt_tree.add("Scan/Estimated Time", "00:00:00", save=False,item_kwargs={'readonly':True})
                    pzt_tree.add("Plot/Autoscale",False)
                    pzt_tree.add("Plot/N Bins",50,item_kwargs={'min_value':1,
                                                                'max_value':1000},
                                 callback=update_pzt_plot)
                    pzt_tree.add("Plot/Update Every Point",False)
                    pzt_tree.add("Plot/Deinterleave",False,callback=update_pzt_plot)
                    pzt_tree.add("Plot/Reverse",False,callback=update_pzt_plot)
            
                    pzt_tree.add("Plot/3D/Slice Index",0,drag=True,callback=update_pzt_plot,
                                 item_kwargs={"min_value":0,"max_value":100,"clamped":True})
                    # with dpg.group(horizontal=True,parent="pzt_tree_Plot/3D"):
                    #     dpg.add_text("Reducing Func.")
                    #     dpg.add_combo(["Delta","Max","Average","Slice"],default_value="Delta",
                    #                   tag="reducing_func",callback=update_pzt_plot)
                    pzt_tree.add_combo("Plot/3D/Reducing Func.",
                                       values=["Delta","Max","Average","Slice"],
                                       default="Delta",
                                       callback=update_pzt_plot)
                with dpg.child_window(width=0,height=0,autosize_y=True):
                        with dpg.group():
                            pzt_plot.parent = dpg.last_item()
                            pzt_plot.height = -330
                            pzt_plot.scale_width = 335
                            pzt_plot.make_gui()
                            with dpg.child_window(width=-0,height=320):
                                with dpg.plot(label="Cavity Scan",width=-1,height=300,tag="cav_plot"):
                                    dpg.bind_font("plot_font") 
                                    # REQUIRED: create x and y axes
                                    dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="cav_count_x")
                                    dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="cav_count_y")
                                    dpg.add_line_series([0],
                                                        [0],
                                                        parent='cav_count_y',label='counts', tag='cav_counts')
                                    dpg.add_drag_line(tag="cav_count_cut",parent='cav_plot',default_value=0,
                                                      callback=update_pzt_plot)
                                    dpg.add_plot_legend()

##################
# Initialization #
##################

# Load in tree values
count_tree.load()
galvo_tree.load()
optim_tree.load()
obj_tree.load()
pzt_tree.load()
# Initialize Values
galvo_position = fpga.get_galvo()
cavity_position = fpga.get_cavity()
jpe_position = fpga.get_jpe_pzs()
galvo_tree["Galvo/Position"] = galvo_position
galvo_plot.set_cursor(galvo_position)
pzt_tree["Cavity/Position"] = cavity_position[0]
pzt_tree["JPE/Z Position"] = jpe_position[2]
pzt_tree["JPE/XY Position"] = jpe_position[:2]
pzt_tree["JPE/Z Volts"] = pz_conv.zs_from_cart(jpe_position)
toggle_AI(None,count_tree["Counts/Show AI1"], None)


guess_pzt_times()
guess_galvo_time()
guess_obj_time()
draw_bounds()
pzt_plot.set_cursor(jpe_position[:2])
dpg.set_primary_window('main_window',True)
dpg.show_item_registry()
rdpg.start_dpg()
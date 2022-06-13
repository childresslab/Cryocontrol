from pydoc import doc
from re import U
from sys import ps1
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
from apis.dummy import fpga_base_dummy, fpga_cryo_dummy

from apis.scanner import Scanner
from apis.dummy.fpga_cryo_dummy import DummyCryoFPGA,FPGAValueError
from apis.dummy.objective_dummy import DummyObjective
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

# Slowly turning into a mess of a file
# Ideally this should be better encapsulated into individual modules
# that then combine together into one file. However that requires
# figuring out how to properly share the instrument/data between files
# which will be a bit tricky...
log.basicConfig(format='%(levelname)s:%(message)s ', level=log.DEBUG)

log.warning("Using Dummy Controls")
fpga = DummyCryoFPGA()
obj = DummyObjective()

# Setup counts data
counts_data = {'counts':[0],
               'AI1' :[0],
               'time':[datetime.now().timestamp()]}
position_register = {"temp_galvo_position":fpga.get_galvo()}

#################
# Galvo Control #
#################
def set_galvo(x,y,write=True):
    fpga.set_galvo(x,y,write=write)
    if (x is None or y is None):
        galvo_pos = fpga.get_galvo()
        if x is None:
            x = galvo_pos[0]
        if y is None:
            y = galvo_pos[1]
    set_cursor(x,y)

def man_set_galvo(*args):
    pos = galvo_tree["Galvo/Position"]
    set_galvo(pos[0],pos[1])

def galvo(y,x):
    log.debug(f"Set galvo to ({x},{y}) V.")
    fpga.set_galvo(x,y,write=False)
    set_cursor(x,y)
    galvo_tree["Galvo/Position"] = [x,y]
    count = get_count(galvo_tree["Scan/Count Time (ms)"])
    log.debug(f"Got count rate of {count}.")
    return count

def get_count(time):
    count = fpga.just_count(time)
    log.debug(f"Got count rate of {count}.")
    if count_tree["Counts/Plot Scan Counts"] or dpg.get_value("count"):
        counts_data['counts'].append(count)
        counts_data['AI1'].append(fpga.get_AI_volts([1])[0])
        counts_data['time'].append(datetime.now().timestamp())
    return count
    

galvo_scan = Scanner(galvo,[0,0],[1,1],[50,50],[1],[],float,['y','x'],default_result=-1)

def start_scan(sender,app_data,user_data):
    if not dpg.get_value('scan'):
        return -1
    if dpg.get_value("count"):
        dpg.set_value("count",False)
        sleep(count_tree["Counts/Count Time (ms)"]/1000)
    steps = galvo_tree["Scan/Points"]
    galvo_scan.steps = steps[1::-1]
    galvo_scan.centers = galvo_tree["Scan/Centers (V)"][1::-1]
    galvo_scan.spans = galvo_tree["Scan/Spans (V)"][1::-1]
    
    def init():
        pos = galvo_scan._get_positions()
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        position_register["temp_galvo_position"] = fpga.get_galvo()
        dpg.configure_item("heat_series",
                           rows=int(galvo_scan.steps[0]),
                           cols=int(galvo_scan.steps[1]),
                           bounds_min=(xmin,ymin),bounds_max=(xmax,ymax))
    
    def abort(i,imax,idx,pos,res):
        return not dpg.get_value('scan')

    def prog(i,imax,idx,pos,res):
            log.debug("Setting Progress Bar")
            dpg.set_value("pb",(i+1)/imax)
            dpg.configure_item("pb",overlay=f"Galvo Scan {i+1}/{imax}")
            if galvo_tree["Plot/Update Every Point"]:
                check = True
            else:
                check = (not (i+1) % galvo_tree["Scan/Points"][0]) or (i+1)==imax
            if check:
                log.debug("Updating Galvo Scan Plot")
                plot_data = np.copy(np.flip(galvo_scan.results,0))
                dpg.set_value("heat_series", [plot_data,[0.0,1.0],[],[],[]])
                if galvo_tree["Plot/Autoscale"]:
                    lower = np.min(plot_data[np.where(plot_data>=0)])
                    upper = np.max(plot_data)
                    dpg.configure_item("colormap",min_scale=lower,max_scale=upper)
                    dpg.configure_item("heat_series",scale_min=lower,scale_max=upper)
                    dpg.set_value("line1",lower)
                    dpg.set_value("line2",upper) 
                    for ax in ["heat_x","heat_y","hist_x","hist_y"]:
                        dpg.fit_axis_data(ax)
                update_histogram(plot_data)
                if count_tree["Counts/Plot Scan Counts"]:
                    plot_counts()

    def finish(results,completed):
        dpg.set_value('scan',False)
        set_galvo(*position_register["temp_galvo_position"])
        if dpg.get_value("auto_save"):
            save_galvo_scan()

    galvo_scan._init_func = init
    galvo_scan._abort_func = abort
    galvo_scan._prog_func = prog
    galvo_scan._finish_func = finish
    galvo_scan.run_async()

# Plot Updating
def update_histogram(data):
    data = data[np.where(data>=0)]
    nbins = galvo_tree["Plot/N Bins"]
    occ,edges = np.histogram(data,bins=nbins)
    xs = [0] + list(np.repeat(occ,2)) + [0,0] 
    ys = list(np.repeat(edges,2)) + [0]
    dpg.set_value("histogram",[xs,ys,[],[],[]])

def set_scale(sender,app_data,user_data):
    val1 = dpg.get_value("line1")
    val2 = dpg.get_value("line2")
    lower = min([val1,val2])
    upper = max([val1,val2])
    galvo_tree["Plot/Autoscale"] = False
    dpg.configure_item("colormap",min_scale=lower,max_scale=upper)
    dpg.configure_item("heat_series",scale_min=lower,scale_max=upper)

def get_scan_range(*args):
    if dpg.is_plot_queried("plot"):
        xmin,xmax,ymin,ymax = dpg.get_plot_query_area("plot")
        new_centers = [(xmin+xmax)/2, (ymin+ymax)/2]
        new_spans = [xmax-xmin, ymax-ymin]
        galvo_tree["Scan/Centers (V)"] = new_centers
        galvo_tree["Scan/Spans (V)"] = new_spans

def guess_galvo_time(*args):
    pts = galvo_tree["Scan/Points"]
    ctime = galvo_tree["Scan/Count Time (ms)"] + galvo_tree["Scan/Wait Time (ms)"]
    scan_time = pts[0] * pts[1] * ctime / 1000
    time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
    galvo_tree["Scan/Estimated Time"] = time_string

def bound_galvo(point):
    if point[0] < -10:
        point[0] = -10
    if point[0] > 10:
        point[0] = 10
    if point[1] < -10:
        point[1] = -10
    if point[1] > 10:
        point[1] = 10
    return point

def cursor_drag(sender,value,user_data):
    point = dpg.get_value("cc")[:2]
    if sender == "cx":
        point[0] = dpg.get_value("cx")
    elif sender == "cy":
        point[1] = dpg.get_value("cy")
    point = bound_galvo(point)
    dpg.set_value("cx",point[0])
    dpg.set_value("cy",point[1])
    dpg.set_value("cc",point)
    fpga.set_galvo(point[0],point[1],write=False)
    galvo_tree["Galvo/Position"] = point
    return

def set_cursor(x,y):
    point = [x,y]
    dpg.set_value("cc",point)
    dpg.set_value("cx",point[0])
    dpg.set_value("cy",point[1])

# Getting and Setting Values
def set_wait_time(*args):
    fpga.set_ao_wait(float(galvo_tree["Scan/Wait Time (ms)"]),write=True)
    guess_galvo_time()

# Saving Scans
def choose_save_dir(*args):
    chosen_dir = dpg.add_file_dialog(label="Chose Save Directory", 
                        default_path=dpg.get_value("save_dir"), 
                        directory_selector=True, modal=True,callback=set_save_dir)

def set_save_dir(sender,chosen_dir,user_data):
    dpg.set_value("save_dir",chosen_dir['file_path_name'])

def save_galvo_scan(*args):
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_scan_file")
    path /= filename
    as_npz = not (".csv" in filename)
    galvo_scan.save_results(str(path),as_npz=as_npz)

####################
# Counting Control #
####################
def clear_counts(*args):
    counts_data['counts'] = []
    counts_data['AI1'] = []
    counts_data['time'] = []

def moving_average(values,window):
    return np.average(sliding_window_view(values, window_shape = window), axis=1)

def average_counts(times,counts,window):
    avg_times = moving_average(times,window)
    avg_counts = moving_average(counts,window)
    return avg_times,avg_counts

def plot_counts(*args):
    delta = len(counts_data['counts']) - count_tree["Counts/Max Points"]
    while delta >= 0:
        try:
            counts_data['counts'].pop(0)
            counts_data['AI1'].pop(0)
            counts_data['time'].pop(0)
            delta -= 1
        except IndexError:
            break
    avg_time, avg_counts= average_counts(counts_data['time'],
                                         counts_data['counts'],
                                         min(len(counts_data['time']),
                                             count_tree["Counts/Average Points"]))
    dpg.set_value('counts_series',[rdpg.offset_timezone(counts_data['time']),counts_data['counts']])
    dpg.set_value('avg_counts_series',[rdpg.offset_timezone(avg_time),avg_counts])
    dpg.set_value('AI1_series',[rdpg.offset_timezone(counts_data['time']),counts_data['AI1']])
    dpg.set_value('counts_series2',[rdpg.offset_timezone(counts_data['time']),counts_data['counts']])
    dpg.set_value('avg_counts_series2',[rdpg.offset_timezone(avg_time),avg_counts])
    dpg.set_value('counts_series3',[rdpg.offset_timezone(counts_data['time']),counts_data['counts']])
    dpg.set_value('avg_counts_series3',[rdpg.offset_timezone(avg_time),avg_counts])

def start_counts():
    if not dpg.get_value('count'):
        return
    
    def count_func():
        plot_thread = Thread(target=plot_counts)
        while dpg.get_value("count"):
            count = get_count(count_tree["Counts/Count Time (ms)"])
            if not plot_thread.is_alive():
                plot_thread = Thread(target=plot_counts)
                plot_thread.start()

    count_thread = Thread(target=count_func)
    count_thread.start()

def save_counts(*args):
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_counts_file")
    path /= filename
    with path.open('w') as f:
        f.write("Timestamp,Counts,AI1\n")
        for d in enumerate(zip(counts_data['time'],counts_data['counts'],counts_data['AI1'])):
                f.write(f"{d[0]},{d[1]},{d[2]}\n")

###################
# Galvo Optimizer #
###################
def optim_scanner_func(axis='x'):
    if axis == 'x':
        def optim_func(x):
            y = position_register['temp_galvo_position'][1]
            log.debug(f"Set galvo to ({x},{y}) V.")
            fpga.set_galvo(x,y,write=False)
            set_cursor(x,y)
            galvo_tree["Galvo/Position"] = [x,y]
            count = get_count(optim_tree["Optimizer/Count Time (ms)"])
            log.debug(f"Got count rate of {count}.")
            if count_tree["Counts/Plot Scan Counts"]:
                counts_data['counts'].append(count)
                counts_data['time'].append(datetime.now().timestamp())
            return count
    elif axis == 'y':
        def optim_func(y):
            x = position_register['temp_galvo_position'][0]
            log.debug(f"Set galvo to ({x},{y}) V.")
            fpga.set_galvo(x,y,write=False)
            set_cursor(x,y)
            galvo_tree["Galvo/Position"] = [x,y]
            count = get_count(optim_tree["Optimizer/Count Time (ms)"])
            log.debug(f"Got count rate of {count}.")
            if count_tree["Counts/Plot Scan Counts"]:
                counts_data['counts'].append(count)
                counts_data['time'].append(datetime.now().timestamp())
            return count
    else:
        raise ValueError(f"Invalid Axis {axis}, must be either 'x' or 'y'.")
    return optim_func

def fit_galvo_optim(position,counts):
    model = lm.models.QuadraticModel()
    params = model.guess(counts,x=position)
    params['a'].set(max=0)
    # Probably more annoying to do it right.
    weights = 1/np.sqrt(np.array([count if count > 0 else 1 for count in counts]))
    return model.fit(counts,params,x=position,weights=weights)

def optimize_galvo(*args):
    def loop_optim():
        for i in range(optim_tree["Optimizer/Iterations"]):
            single_optimize_run().join()
    optim_thread = Thread(target=loop_optim)
    optim_thread.run()

def single_optimize_run():
    position_register["temp_galvo_position"] = fpga.get_galvo()
    init_galvo_pos = position_register["temp_galvo_position"]
    galvo_scanner_x = Scanner(optim_scanner_func('x'),
                              [init_galvo_pos[0]],
                              [optim_tree["Optimizer/Scan Range (XY)"]],
                              [optim_tree["Optimizer/Scan Points"]],
                              output_dtype=float,
                              labels=["Galvo X"])
    galvo_scanner_y = Scanner(optim_scanner_func('y'),
                              [init_galvo_pos[1]],
                              [optim_tree["Optimizer/Scan Range (XY)"]],
                              [optim_tree["Optimizer/Scan Points"]],
                              output_dtype=float,
                              labels=["Galvo Y"])

    optim_data = {}

    def init_x():
        position_register['temp_galvo_position'] = fpga.get_galvo()
        optim_data['counts'] = []
        optim_data['pos'] = []
        dpg.set_value('optim_x_counts',[[0],[0]])
        dpg.set_value('optim_y_counts',[[0],[0]])
        dpg.set_value('optim_x_fit',[[],[]])
        dpg.set_value('optim_y_fit',[[],[]])
    def prog_x(i,imax,idx,pos,res):
        dpg.set_value("pb",(i+1)/(2*imax))
        dpg.configure_item("pb",overlay=f"Opt. Galvo (X) {i+1}/{2*imax}")
        optim_data['counts'].append(res)
        optim_data['pos'].append(pos[0])
        dpg.set_value('optim_x_counts',[optim_data['pos'],optim_data['counts']])
        if count_tree["Counts/Plot Scan Counts"]:
                    plot_counts()
    def finish_x(results,completed):
        positions = galvo_scanner_x.positions[0]
        fit_x = fit_galvo_optim(positions,results)
        vals = fit_x.best_values
        optim = -vals['b']/(2*vals['a'])
        optim = min(optim,np.max(positions))
        optim = max(optim,np.min(positions))
        try:
            set_galvo(optim,fpga.get_galvo()[1])
        except ValueError:
            set_galvo(*position_register['temp_galvo_position'])
        new_axis = np.linspace(np.min(positions),np.max(positions),1000)
        fit_data = fit_x.eval(fit_x.params,x=new_axis)
        dpg.set_value('optim_x_fit',[new_axis,fit_data])
        galvo_scanner_y.run_async().join()

    galvo_scanner_x._init_func = init_x
    galvo_scanner_x._prog_func = prog_x
    galvo_scanner_x._finish_func = finish_x

    def init_y():
        optim_data['counts'] = []
        optim_data['pos'] = []
        position_register['temp_galvo_position'] = fpga.get_galvo()
    def prog_y(i,imax,idx,pos,res):
        dpg.set_value("pb",(i+1+imax)/(2*imax))
        dpg.configure_item("pb",overlay=f"Opt. Galvo (Y) {i+1+imax}/{2*imax}")
        optim_data['counts'].append(res)
        optim_data['pos'].append(pos[0])
        dpg.set_value('optim_y_counts',[optim_data['pos'],optim_data['counts']])
        if count_tree["Counts/Plot Scan Counts"]:
                    plot_counts()

    def finish_y(results,completed):
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
        new_axis = np.linspace(np.min(positions),np.max(positions),1000)
        fit_data = fit_y.eval(fit_y.params,x=new_axis)
        dpg.set_value('optim_y_fit',[new_axis,fit_data])

    galvo_scanner_y._init_func = init_y
    galvo_scanner_y._prog_func = prog_y
    galvo_scanner_y._finish_func = finish_y

    return galvo_scanner_x.run_async()

######################
# Objective Scanning #
######################
obj_plot = mvHistPlot("Obj. Plot",False,None,True,False,1000,0,300,50,'viridis',True,0,1E9,50,50)

# NOTE:
# The bare api of the objective control is setup so that more negative values
# are physically upwards in the cryostat.
# Here, we have opted to invert that, such that a more positive value
# is upwards in the cryo, such that plotting and moving makes more sense.
def obj_scan_func(fixed_galvo_axis='y'):
    if fixed_galvo_axis not in ['x','y']:
        raise ValueError("Axis must be 'x' or 'y'")
    if fixed_galvo_axis=='x':
        def func(z,y):
            log.debug(f"Set galvo y to {y} V.")
            log.debug(f"Set obj. position to {z} um.")
            set_galvo(None,y,write=False)
            set_obj_abs(z)
            count = get_count(obj_tree["Scan/Count Time (ms)"])
            log.debug(f"Got count rate of {count}.")
            return count
        return func
    if fixed_galvo_axis=='y':
        def func(z,x):
            log.debug(f"Set galvo x to {x} V.")
            log.debug(f"Set obj. position to {z} um.")
            set_galvo(x,None,write=False)
            set_obj_abs(z)
            count = get_count(obj_tree["Scan/Count Time (ms)"])
            log.debug(f"Got count rate of {count}.")
            return count
        return func
obj_scan = Scanner(obj_scan_func('x'),[0,0],[1,1],[50,50],[1],[],float,['y','x'],default_result=-1)


def start_obj_scan(sender,app_data,user_data):
    if not dpg.get_value("obj_scan"):
        return -1
    if dpg.get_value("count"):
        dpg.set_value("count",False)
        sleep(count_tree["Counts/Count Time (ms)"]/1000)
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
        pos = obj_scan._get_positions()
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        position_register["temp_obj_position"] = obj.position
        obj_plot.set_size(int(obj_scan.steps[0]),int(obj_scan.steps[1]))
        obj_plot.set_bounds(xmin,xmax,ymin,ymax)
    
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
        if dpg.get_value("auto_save"):
            save_obj_scan()

    obj_scan._init_func = init
    obj_scan._abort_func = abort
    obj_scan._prog_func = prog
    obj_scan._finish_func = finish
    return obj_scan.run_async()

def save_obj_scan(*args):
    pass

def toggle_objective(sender,app_data,user_data):
    if app_data:
        obj.initialize()
        obj_tree["Objective/Status"] = "Initialized"
        pos = -obj.position
        obj_tree["Objective/Current Position (um)"] = pos
        obj_tree["Objective/Set Position (um)"] = pos
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

def set_obj_callback(sender,app_data,user_data):
    return set_obj_abs(app_data)

def set_obj_abs(position):
    position = -position
    def func():
        obj.move_abs(position,monitor=True,monitor_callback=obj_move_callback)
    t = Thread(target=func)
    t.run()
    return t

def obj_step_up(*args):
    def func():
        step = obj_tree["Objective/Rel. Step (um)"]
        obj.move_up(step,monitor=True,monitor_callback=obj_move_callback)
    t = Thread(target=func)
    t.run()
    return t

def obj_step_down(*args):
    def func():
        step = obj_tree["Objective/Rel. Step (um)"]
        obj.move_down(step,monitor=True,monitor_callback=obj_move_callback)
    t = Thread(target=func)
    t.run()
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

###################
# Cavity Scanning #
###################
pzt_plot = mvHistPlot("Piezo Scan",True,None,True,True,1000,0,300,1000,'viridis',True,0,1E12,50,50)
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
        dpg.set_value("cav_counts",[cav_data['pos'],cav_data['counts']])
        if count_tree["Counts/Plot Scan Counts"]:
            plot_counts()

    def finish(results,completed):
        if pzt_tree['Plot/Autoscale']:
            dpg.set_axis_limits("cav_count_y",np.min(results),np.max(results))
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
    if dpg.get_value("count"):
        dpg.set_value("count",False)
        sleep(count_tree["Counts/Count Time (ms)"]/1000)

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
        if dpg.get_value("auto_save"):
            save_xy_scan()

    jpe_xy_scan._init_func = init
    jpe_xy_scan._abort_func = abort
    jpe_xy_scan._prog_func = prog
    jpe_xy_scan._finish_func = finish
    return jpe_xy_scan.run_async()

def get_xy_range():
    pass
def start_cav_scan():
    if not dpg.get_value("pzt_cav_scan"):
        return -1
    if dpg.get_value("count"):
        dpg.set_value("count",False)
        sleep(count_tree["Counts/Count Time (ms)"]/1000)

    steps = pzt_tree["Scan/Cavity/Steps"]
    centers = pzt_tree["Scan/Cavity/Center"]
    spans = pzt_tree["Scan/Cavity/Span"]
    jpe_cav_scan.steps = [steps]
    jpe_cav_scan.centers = [centers]
    jpe_cav_scan.spans = [spans]
    cav_data = {}

    def init():
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
        dpg.configure_item("pzt_tree_Plot/Slice Index",max_value=pzt_tree["Scan/Cavity/Steps"]-1)
    
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
            dpg.set_axis_limits("cav_count_y",np.min(results),np.max(results))
        else:
            dpg.set_axis_limits_auto("cav_count_y")
        set_cav_pos(*position_register["temp_cav_position"])
        if dpg.get_value("auto_save"):
            save_cav_scan()

    jpe_cav_scan._init_func = init
    jpe_cav_scan._abort_func = abort
    jpe_cav_scan._prog_func = prog
    jpe_cav_scan._finish_func = finish
    return jpe_cav_scan.run_async()

def get_cav_range():
    pass

def get_reducing_func():
    func_str = dpg.get_value("reducing_func")
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
    if dpg.get_value("count"):
        dpg.set_value("count",False)
        sleep(count_tree["Counts/Count Time (ms)"]/1000)

    jpe_steps = pzt_tree["Scan/JPE/Steps"][:2][::-1]
    jpe_centers = pzt_tree["Scan/JPE/Center"][:2][::-1]
    jpe_spans = pzt_tree["Scan/JPE/Span"][:2][::-1]
    jpe_3D_scan.steps = jpe_steps
    jpe_3D_scan.centers = jpe_centers
    jpe_3D_scan.spans = jpe_spans

    def init():
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
        dpg.configure_item("pzt_tree_Plot/Slice Index",max_value=pzt_tree["Scan/Cavity/Steps"]-1)
    
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
        if dpg.get_value("auto_save"):
            save_xy_scan()

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

def guess_pzt_times():
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

def save_xy_scan():
    pass

def save_cav_scan():
    pass

def save_3d_scan():
    pass

################################################################################
############################### UI Building ####################################
################################################################################
rdpg.initialize_dpg("Cryocontrol",docking=False)
###############
# Main Window #
###############
with dpg.window(label="Cryocontrol", tag='main_window'):
    ##################
    # Persistant Bar #
    ##################
    # Data Directory
    with dpg.group(horizontal=True):
        dpg.add_text("Data Directory:")
        dpg.add_input_text(default_value="X:\\DiamondCloud\\Cryostat Setup", tag="save_dir")
        dpg.add_button(label="Pick Directory", callback=choose_save_dir)
    # Counts and Optimization
    with dpg.group(horizontal=True):
        dpg.add_checkbox(tag="count", label="Count", callback=start_counts)
        dpg.add_button(tag="clear_counts", label="Clear Counts",callback=clear_counts)
        dpg.add_button(tag="optimize", label="Optimize Galvo", callback=optimize_galvo)
    # Confocal Scan Control
    with dpg.group(horizontal=True):
        dpg.add_progress_bar(label="Scan Progress",tag='pb',width=-1)

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
                    dpg.add_button(tag="save_counts", label="Save Counts TODO",callback=save_counts)
            with dpg.group(horizontal=True,width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="count_tree"):
                    count_tree = rdpg.TreeDict('count_tree','cryo_gui_settings/count_tree_save.csv')
                    count_tree.add("Counts/Count Time (ms)", 10,item_kwargs={'min_value':1,'min_clamped':True})
                    count_tree.add("Counts/Max Points", 100000,item_kwargs={'on_enter':True,'min_value':1,'min_clamped':True})
                    count_tree.add("Counts/Plot Scan Counts", True, callback=plot_counts)
                    count_tree.add("Counts/Average Points", 5, callback=plot_counts,item_kwargs={'min_value':1,'min_clamped':True})
                with dpg.child_window(width=-1,autosize_y=True): 
                    with dpg.theme() as count_series_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 100, category=dpg.mvThemeCat_Core)

                    with dpg.plot(label="Count Rate",width=-1,height=-1,tag="count_plot"):
                        dpg.bind_font("plot_font") 
                        # REQUIRED: create x and y axes
                        dpg.add_plot_axis(dpg.mvXAxis, label="Time", time=True, tag="count_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Counts",tag="count_y")
                        dpg.add_plot_axis(dpg.mvYAxis,label="Sync", tag="count_AI1")
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
                        dpg.bind_item_theme("counts_series",count_series_theme)
                        dpg.bind_item_theme("avg_counts_series",count_series_theme)
                        
        #############
        # GALVO TAB #
        #############
        with dpg.tab(label="Galvo"):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(tag="scan",label="Scan Galvo", callback=start_scan)
                dpg.add_button(tag="query_plot",label="Query Plot Range",callback=get_scan_range)
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
                    galvo_tree.add("Plot/Autoscale",False)
                    galvo_tree.add("Plot/N Bins",50,item_kwargs={'min_value':1,
                                                                    'max_value':1000})
                    galvo_tree.add("Plot/Update Every Point",False)
                    galvo_tree.add("Scan/Centers (V)",[0.0,0.0],item_kwargs={'min_value':-10,
                                                                            'max_value':10,
                                                                            "min_clamped":True,
                                                                            "max_clamped":True})
                    galvo_tree.add("Scan/Spans (V)", [1.0,1.0],item_kwargs={'min_value':-20,
                                                                        'max_value':20,
                                                                        "min_clamped":True,
                                                                        "max_clamped":True})
                    galvo_tree.add("Scan/Points", [100,100],item_kwargs={'min_value':0},
                                callback=guess_galvo_time)
                    galvo_tree.add("Scan/Count Time (ms)", 10.0, item_kwargs={'min_value':0,
                                                                            'max_value':1000},
                                callback=guess_galvo_time)
                    galvo_tree.add("Scan/Wait Time (ms)", 1.0,item_kwargs={'min_value':0,
                                                                        'max_value':1000},
                                callback=guess_galvo_time)
                    galvo_tree.add("Scan/Estimated Time", "00:00:00", item_kwargs={'readonly':True},
                    save=False)
                with dpg.group():
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=-400,height=-300): 
                            with dpg.plot(label="Heat Series",width=-1,height=-1,
                                            equal_aspects=True,tag="plot",query=True):
                                dpg.bind_font("plot_font")
                                # REQUIRED: create x and y axes
                                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="heat_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="heat_y")
                                dpg.add_heat_series(np.zeros((100,100)),100,100,
                                                    scale_min=0,scale_max=1000,
                                                    parent="heat_y",label="heatmap",
                                                    tag="heat_series",format='')
                                dpg.add_drag_point(color=(204,36,29,122),parent="plot",
                                                        callback=cursor_drag,
                                                        default_value=(0.5,0.5),
                                                        tag="cc")
                                dpg.add_drag_line(color=(204,36,29,122),parent="plot",
                                                        callback=cursor_drag,
                                                        default_value=0.5,vertical=True,
                                                        tag="cx")
                                dpg.add_drag_line(color=(204,36,29,122),parent="plot",
                                                        callback=cursor_drag,
                                                        default_value=0.5,vertical=False,
                                                        tag="cy")
                                dpg.bind_colormap("plot",dpg.mvPlotColormap_Viridis)

                        with dpg.child_window(width=-0,height=-300):
                            with dpg.group(horizontal=True):
                                dpg.add_colormap_scale(min_scale=0,max_scale=1000,
                                                        width=100,height=-1,tag="colormap",
                                                        colormap=dpg.mvPlotColormap_Viridis)
                                with dpg.plot(label="Histogram", width=-1,height=-1) as histogram:
                                    dpg.bind_font("plot_font")
                                    dpg.add_plot_axis(dpg.mvXAxis,label="Occurance",tag="hist_x")
                                    dpg.add_plot_axis(dpg.mvYAxis,label="Counts",tag="hist_y")
                                    dpg.add_area_series([0],[0],parent="hist_x",
                                                        fill=[120,120,120,120],tag="histogram")
                                    dpg.add_drag_line(callback=set_scale,default_value=0,
                                                        parent=histogram,tag="line1",vertical=False)
                                    dpg.add_drag_line(callback=set_scale,default_value=0,
                                                        parent=histogram,tag="line2",vertical=False)
                    with dpg.child_window(width=-0,height=-0):
                        with dpg.plot(label="Count Rate",width=-1,height=300,tag="count_plot2"):
                            dpg.bind_font("plot_font") 
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="x", time=True, tag="count_x2")
                            dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="count_y2")
                            dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                                counts_data['counts'],
                                                parent='count_y2',label='counts', tag='counts_series2')
                            dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                                counts_data['counts'],
                                                parent='count_y2',label='avg. counts', tag='avg_counts_series2')
                            dpg.add_plot_legend()
        #################
        # Optimizer Tab #
        #################
        with dpg.tab(label="Galvo Optimizer"):
            with dpg.group(horizontal=True,width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="optim_tree"):
                    optim_tree = rdpg.TreeDict('optim_tree','cryo_gui_settings/optim_tree_save.csv')
                    optim_tree.add("Optimizer/Count Time (ms)", 10.0,item_kwargs={'min_value':1.0,'min_clamped':True})
                    optim_tree.add("Optimizer/Scan Points", 50,item_kwargs={'min_value':2,'min_clamped':True})
                    optim_tree.add("Optimizer/Scan Range (XY)", 0.1,item_kwargs={'min_value':0.0,'min_clamped':True})
                    optim_tree.add("Optimizer/Iterations", 1,item_kwargs={'min_value':1,'min_clamped':True})
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
        with dpg.tab(label="Objective Control"):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Scan Objective",tag="obj_scan", default_value=False, callback=start_obj_scan)
                dpg.add_text("Filename:")
                dpg.add_input_text(tag="save_obj_file", default_value="scan.npz", width=200)
                dpg.add_button(tag="save_obj_button", label="Save Scan",callback=save_galvo_scan)
                dpg.add_checkbox(tag="auto_save_obj", label="Auto")
            with dpg.group(horizontal=True,width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="obj_tree"):
                    obj_tree = rdpg.TreeDict('obj_tree','cryo_gui_settings/obj_tree_save.csv')
                    obj_tree.add("Objective/Initialize", False,save=False,callback=toggle_objective)
                    obj_tree.add("Objective/Status","Uninitialized",save=False,item_kwargs={"readonly":True})
                    obj_tree.add("Objective/Set Position (um)", 100.0,callback=set_obj_callback,item_kwargs={"on_enter":True})
                    obj_tree.add("Objective/Current Position (um)", 100.0,item_kwargs={"readonly":True,'step':0})
                    obj_tree.add("Objective/Limits (um)",[-8000.0,8000.0],callback=set_objective_params,item_kwargs={"on_enter":True})
                    obj_tree.add("Objective/Max Move (um)", 100.0,callback=set_objective_params,item_kwargs={"on_enter":True})
                    obj_tree.add("Objective/Rel. Step (um)", 5)
                    obj_tree.add("Scan/Count Time (ms)", 10,callback=guess_obj_time,item_kwargs={'min_value':0,'min_clamped':True})
                    obj_tree.add("Scan/Wait Time (ms)", 5,callback=guess_obj_time,item_kwargs={'min_value':0,'min_clamped':True})
                    obj_tree.add("Scan/Obj./Center (um)",0)
                    obj_tree.add("Scan/Obj./Span (um)",50)
                    obj_tree.add("Scan/Obj./Steps",50,callback=guess_obj_time)
                    obj_tree.add("Scan/Galvo/Fixed Axis",'x')
                    obj_tree.add("Scan/Galvo/Center (V)",0)
                    obj_tree.add("Scan/Galvo/Span (V)",0.05)
                    obj_tree.add("Scan/Galvo/Steps",50,callback=guess_obj_time)
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
                                    dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                                        counts_data['counts'],
                                                        parent='count_y3',label='counts', tag='counts_series3')
                                    dpg.add_line_series(rdpg.offset_timezone(counts_data['time']),
                                                        counts_data['counts'],
                                                        parent='count_y3',label='avg. counts', tag='avg_counts_series3')
                                    dpg.add_plot_legend()
        #############
        # Piezo Tab #
        #############
        with dpg.tab(label="Piezo Control"):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(tag="pzt_xy_scan",label="Scan XY", callback=start_xy_scan)
                dpg.add_checkbox(tag="pzt_cav_scan",label="Scan Cav.", callback=start_cav_scan)
                dpg.add_checkbox(tag="pzt_3d_scan",label="Scan 3D", callback=start_3d_scan)
                dpg.add_button(tag="query_xy_plot",label="XY Query Plot Range",callback=get_xy_range)
                dpg.add_button(tag="query_cav_plot",label="Cav. Query Plot Range",callback=get_cav_range)
                
            with dpg.group(horizontal=True,width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="pzt_tree"):
                    pzt_tree = rdpg.TreeDict('pzt_tree','cryo_gui_settings/pzt_tree_save.csv')
                    pzt_tree.add("JPE/Z Position",0.0,
                                 item_kwargs={'min_value':-6.5,'max_value':0,
                                              'min_clamped':True,'max_clamped':True,
                                              'on_enter':True},
                                              callback=z_pos_callback,save=False)
                    pzt_tree.add("JPE/XY Position",[0.0,0.0],
                                 item_kwargs={'on_enter':True},
                                 callback=xy_pos_callback,save=False)
                    pzt_tree.add("JPE/Z Volts", [0.0,0.0,0.0], save=False,
                                 item_kwargs={"readonly" : True})

                    pzt_tree.add("Cavity/Position",0.0,
                                 item_kwargs={'min_value':-8,'max_value':8,
                                              'min_clamped':True,'max_clamped':True,
                                              'on_enter':True})
                    pzt_tree.add("Scan/Wait Time (ms)",10,callback=guess_pzt_times)
                    pzt_tree.add("Scan/Count Time (ms)",5,callback=guess_pzt_times)
                    pzt_tree.add("Scan/Cavity/Center",0)
                    pzt_tree.add("Scan/Cavity/Span",16)
                    pzt_tree.add("Scan/Cavity/Steps",300,callback=guess_pzt_times)
                    pzt_tree.add("Scan/JPE/Center",[0,0])
                    pzt_tree.add("Scan/JPE/Span",[5,5])
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
                    with dpg.group(horizontal=True,parent="pzt_tree_Plot/3D"):
                        dpg.add_text("Reducing Func.")
                        dpg.add_combo(["Delta","Max","Average","Slice"],default_value="Delta",
                                      tag="reducing_func",callback=update_pzt_plot)
                    pzt_tree.add("Plot/3D/Slice Index",0,drag=True,callback=update_pzt_plot,
                                 item_kwargs={"min_value":0,"max_value":100,"clamped":True})


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
        
# Initialize Values
galvo_position = fpga.get_galvo()
cavity_position = fpga.get_cavity()
jpe_position = fpga.get_jpe_pzs()
galvo_tree["Scan/Centers (V)"] = [galvo_position[0],galvo_position[1]]
pzt_tree["Cavity/Position"] = cavity_position[0]
pzt_tree["JPE/Z Position"] = jpe_position[2]
pzt_tree["JPE/XY Position"] = jpe_position[:2]
pzt_tree["JPE/Z Volts"] = pz_conv.zs_from_cart(jpe_position)
guess_pzt_times()
guess_galvo_time()
guess_obj_time()
draw_bounds()
dpg.set_primary_window('main_window',True)
dpg.show_item_registry()
rdpg.start_dpg()
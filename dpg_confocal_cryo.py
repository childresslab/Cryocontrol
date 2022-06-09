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

from apis.scanner import Scanner
from apis.dummy.fpga_cryo_dummy import DummyCryoFPGA
from apis.dummy.objective_dummy import DummyObjective
import apis.rdpg as rdpg
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
def set_galvo(x,y):
    fpga.set_galvo(x,y,write=True)
    set_cursor(x,y)

def man_set_galvo(*args):
    pos = dpg.get_value("Galvo/Position")
    set_galvo(pos[0],pos[1])

def galvo(y,x):
    log.debug(f"Set galvo to ({x},{y}) V.")
    fpga.set_galvo(x,y,write=False)
    set_cursor(x,y)
    dpg.set_value("Galvo/Position",[x,y])
    count = fpga.just_count(dpg.get_value("Scan/Count Time (ms)"))
    log.debug(f"Got count rate of {count}.")
    if dpg.get_value("Counts/Plot Scan Counts"):
        counts_data['counts'].append(count)
        counts_data['time'].append(datetime.now().timestamp())
    return count

galvo_scan = Scanner(galvo,[0,0],[1,1],[50,50],[1],[],float,['y','x'],default_result=-1)

def start_scan(sender,app_data,user_data):
    if not dpg.get_value('scan'):
        return -1
    if dpg.get_value("count"):
        dpg.set_value("count",False)
        sleep(dpg.get_value("Counts/Count Time (ms)")/1000)
    steps = dpg.get_value("Scan/Points")
    galvo_scan.steps = steps[1::-1]
    galvo_scan.centers = dpg.get_value("Scan/Centers (V)")[1::-1]
    galvo_scan.spans = dpg.get_value("Scan/Spans (V)")[1::-1]
    
    def init():
        pos = galvo_scan._get_positions()
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        position_register["temp_galvo_position"] = fpga.get_galvo()
        dpg.configure_item("heat_series",rows=steps[0],cols=steps[1],
                           bounds_min=(xmin,ymin),bounds_max=(xmax,ymax))
    
    def abort(i,imax,idx,pos,res):
        return not dpg.get_value('scan')

    def prog(i,imax,idx,pos,res):
            log.debug("Setting Progress Bar")
            dpg.set_value("pb",(i+1)/imax)
            dpg.configure_item("pb",overlay=f"{i+1}/{imax}")
            if dpg.get_value("Plot/Update Every Point"):
                check = True
            else:
                check = (not (i+1) % dpg.get_value("Scan/Points")[0]) or (i+1)==imax
            if check:
                log.debug("Updating Galvo Scan Plot")
                plot_data = np.copy(np.flip(galvo_scan.results,0))
                dpg.set_value("heat_series", [plot_data,[0.0,1.0],[],[],[]])
                if dpg.get_value("Plot/Autoscale"):
                    lower = np.min(plot_data[np.where(plot_data>=0)])
                    upper = np.max(plot_data)
                    dpg.configure_item("colormap",min_scale=lower,max_scale=upper)
                    dpg.configure_item("heat_series",scale_min=lower,scale_max=upper)
                    dpg.set_value("line1",lower)
                    dpg.set_value("line2",upper) 
                    for ax in ["heat_x","heat_y","hist_x","hist_y"]:
                        dpg.fit_axis_data(ax)
                update_histogram(plot_data)
                if dpg.get_value("Counts/Plot Scan Counts"):
                    plot_counts()

    def finish(results,completed):
        dpg.set_value('scan',False)
        set_galvo(*position_register["temp_galvo_position"])
        if dpg.get_value("auto_save"):
            save_scan()

    galvo_scan._init_func = init
    galvo_scan._abort_func = abort
    galvo_scan._prog_func = prog
    galvo_scan._finish_func = finish
    galvo_scan.run_async()

# Plot Updating
def update_histogram(data):
    bin_width = dpg.get_value("Plot/Bin Width")
    data = data[np.where(data>=0)]
    nbins = max([10,int(round((np.nanmax(data)-np.nanmin(data))/bin_width))])
    occ,edges = np.histogram(data,bins=nbins)
    xs = [0] + list(np.repeat(occ,2)) + [0,0] 
    ys = list(np.repeat(edges,2)) + [0]
    dpg.set_value("histogram",[xs,ys,[],[],[]])

def set_scale(sender,app_data,user_data):
    val1 = dpg.get_value("line1")
    val2 = dpg.get_value("line2")
    lower = min([val1,val2])
    upper = max([val1,val2])
    dpg.set_value("Plot/Autoscale", False)
    dpg.configure_item("colormap",min_scale=lower,max_scale=upper)
    dpg.configure_item("heat_series",scale_min=lower,scale_max=upper)

def get_scan_range(*args):
    if dpg.is_plot_queried("plot"):
        xmin,xmax,ymin,ymax = dpg.get_plot_query_area("plot")
        new_centers = [(xmin+xmax)/2, (ymin+ymax)/2]
        new_spans = [xmax-xmin, ymax-ymin]
        dpg.set_value("Scan/Centers (V)",new_centers)
        dpg.set_value("Scan/Spans (V)",new_spans)

def guess_time(*args):
    pts = dpg.get_value("Scan/Points")
    ctime = dpg.get_value("Scan/Count Time (ms)") + dpg.get_value("Scan/Wait Time (ms)")
    scan_time = pts[0] * pts[1] * ctime / 1000
    time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
    dpg.set_value("Scan/Estimated Time",time_string)

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
    dpg.set_value("Galvo/Position",point)
    return

def set_cursor(x,y):
    point = [x,y]
    dpg.set_value("cc",point)
    dpg.set_value("cx",point[0])
    dpg.set_value("cy",point[1])

# Getting and Setting Values
def set_wait_time(*args):
    fpga.set_ao_wait(float(dpg.get_value("Scan/Wait Time (ms)")),write=True)
    guess_time()

# Saving Scans
def choose_save_dir(*args):
    chosen_dir = dpg.add_file_dialog(label="Chose Save Directory", 
                        default_path=dpg.get_value("save_dir"), 
                        directory_selector=True, modal=True,callback=set_save_dir)

def set_save_dir(sender,chosen_dir,user_data):
    dpg.set_value("save_dir",chosen_dir['file_path_name'])

def save_scan(*args):
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_scan_file")
    path /= filename
    as_npz = not (".csv" in filename)
    print(path)
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
    delta = len(counts_data['counts']) - dpg.get_value("Counts/Max Points")
    while delta >= 0:
        counts_data['counts'].pop(0)
        counts_data['AI1'].pop(0)
        counts_data['time'].pop(0)
        delta -= 1
    avg_time, avg_counts= average_counts(counts_data['time'],
                                         counts_data['counts'],
                                         min(len(counts_data['time']),
                                             dpg.get_value("Counts/Average Points")))
    dpg.set_value('counts_series',[rdpg.offset_timezone(counts_data['time']),counts_data['counts']])
    dpg.set_value('avg_counts_series',[rdpg.offset_timezone(avg_time),avg_counts])
    dpg.set_value('AI1_series',[rdpg.offset_timezone(counts_data['time']),counts_data['AI1']])
    dpg.set_value('counts_series2',[rdpg.offset_timezone(counts_data['time']),counts_data['counts']])
    dpg.set_value('avg_counts_series2',[rdpg.offset_timezone(avg_time),avg_counts])

def start_counts():
    if not dpg.get_value('count'):
        return
    
    def count_func():
        plot_thread = Thread(target=plot_counts)
        while dpg.get_value("count"):
            count = fpga.just_count(dpg.get_value("Counts/Count Time (ms)"))
            counts_data['counts'].append(count)
            counts_data['AI1'].append(fpga.get_AI_volts([1])[0])
            counts_data['time'].append(datetime.now().timestamp())
            if not plot_thread.is_alive():
                plot_thread = Thread(target=plot_counts)
                plot_thread.start()

    count_thread = Thread(target=count_func)
    count_thread.start()

def save_counts(*args):
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_counts_file")
    path /= filename
    print(path)
    with path.open('w') as f:
        f.write("Timestamp,Counts,AI1\n")
        for point in enumerate(zip(counts_data['time'],counts_data['counts'],counts_data['AI1'])):
            for d in point:
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
            dpg.set_value("Galvo/Position",[x,y])
            count = fpga.just_count(dpg.get_value("Optimizer/Count Time (ms)"))
            log.debug(f"Got count rate of {count}.")
            if dpg.get_value("Counts/Plot Scan Counts"):
                counts_data['counts'].append(count)
                counts_data['time'].append(datetime.now().timestamp())
            return count
    elif axis == 'y':
        def optim_func(y):
            x = position_register['temp_galvo_position'][0]
            log.debug(f"Set galvo to ({x},{y}) V.")
            fpga.set_galvo(x,y,write=False)
            set_cursor(x,y)
            dpg.set_value("Galvo/Position",[x,y])
            count = fpga.just_count(dpg.get_value("Optimizer/Count Time (ms)"))
            log.debug(f"Got count rate of {count}.")
            if dpg.get_value("Counts/Plot Scan Counts"):
                counts_data['counts'].append(count)
                counts_data['time'].append(datetime.now().timestamp())
            return count
    else:
        raise ValueError(f"Invalid Axis {axis}, must be either 'x' or 'y'.")
    return optim_func

def fit_galvo_optim(position,counts):
    print(position)
    print(counts)
    model = lm.models.QuadraticModel()
    params = model.guess(counts,x=position)
    params['a'].set(max=0)
    # Probably more annoying to do it right.
    weights = 1/np.sqrt(np.array([count if count > 0 else 1 for count in counts]))
    return model.fit(counts,params,x=position,weights=weights)

def optimize_galvo(*args):
    def loop_optim():
        for i in range(dpg.get_value("Optimizer/Iterations")):
            print(f"Optim {i}")
            single_optimize_run().join()
    optim_thread = Thread(target=loop_optim)
    optim_thread.run()

def single_optimize_run():
    position_register["temp_galvo_position"] = fpga.get_galvo()
    init_galvo_pos = position_register["temp_galvo_position"]
    galvo_scanner_x = Scanner(optim_scanner_func('x'),
                              [init_galvo_pos[0]],
                              [dpg.get_value("Optimizer/Scan Range (XY)")],
                              [dpg.get_value("Optimizer/Scan Points")],
                              output_dtype=float,
                              labels=["Galvo X"])
    galvo_scanner_y = Scanner(optim_scanner_func('y'),
                              [init_galvo_pos[1]],
                              [dpg.get_value("Optimizer/Scan Range (XY)")],
                              [dpg.get_value("Optimizer/Scan Points")],
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
        dpg.configure_item("pb",overlay=f"{i+1}/({2*imax})")
        optim_data['counts'].append(res)
        optim_data['pos'].append(pos[0])
        dpg.set_value('optim_x_counts',[optim_data['pos'],optim_data['counts']])
        if dpg.get_value("Counts/Plot Scan Counts"):
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
        dpg.configure_item("pb",overlay=f"{i+1+imax}/({2*imax})")
        optim_data['counts'].append(res)
        optim_data['pos'].append(pos[0])
        dpg.set_value('optim_y_counts',[optim_data['pos'],optim_data['counts']])
        if dpg.get_value("Counts/Plot Scan Counts"):
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

###################
# Cavity Scanning #
###################
def set_cav_and_count(z):
    log.debug(f"Set cavity to {z} V.")
    fpga.set_cavity(z,write=False)
    dpg.set_value("Cavity/Position",z)
    count = fpga.just_count(dpg.get_value("Scan/Count Time (ms)"))
    log.debug(f"Got count rate of {count}.")
    if dpg.get_value("Counts/Plot Scan Counts"):
        counts_data['counts'].append(count)
        counts_data['time'].append(datetime.now().timestamp())
    return count

######################
# Objective Scanning #
######################
def obj_scan_func(fixed_galvo_axis='y'):
    if fixed_galvo_axis not in ['x','y']:
        raise ValueError("Axis must be 'x' or 'y'")
    if fixed_galvo_axis=='y':
        def func(x,z):
                log.debug(f"Set galvo x to {x} V.")
                log.debug(f"Set obj. position to {z} um.")
                fpga.set_galvo(x,None,write=False)
                obj.set_position(z)
                dpg.set_value("Galvo/Position",x)
                dpg.set_value("Objective/Position",z)
                count = fpga.just_count(dpg.get_value("Scan/Count Time (ms)"))
                log.debug(f"Got count rate of {count}.")
                if dpg.get_value("Counts/Plot Scan Counts"):
                    counts_data['counts'].append(count)
                    counts_data['time'].append(datetime.now().timestamp())
                return count

def toggle_objective(sender,app_data,user_data):
    if app_data:
        obj.initialize()
        dpg.set("Objective/Status", "Initialized")
        pos = obj.position
        dpg.set("Objective/Current Position (um)",pos)
        dpg.set("Objective/Set Position (um)",pos)
        set_objective_params()
    else:
        obj.deinitialize()
        dpg.set("Objective/Status", "Deinitialized")

def set_objective_params(*args):
    if obj.initialized:
        limits = dpg.get("Objective/Limits (um)")
        obj.soft_lower = limits[0]
        obj.soft_upper = limits[1]
        obj.max_move = dpg.get("Objective/Max Move (um)")
        dpg.configure_item('obj_pos_set',min_scale=limits[0],max_scale=limits[1])
        dpg.configure_item('obj_pos_get',min_scale=limits[0],max_scale=limits[1])

def set_obj_abs_pos(position):
    def func():
        obj.move_abs(position,monitor=True,monitor_callback=obj_move_callback)
    t = Thread(target=func)
    t.run()
    return t

def obj_step_up(*args):
    def func():
        step = dpg.get_value("Objective/Rel. Step (um)")
        obj.move_up(step,monitor=True,monitor_callback=obj_move_callback)
    t = Thread(target=func)
    t.run()
    return t

def obj_step_down(*args):
    def func():
        step = dpg.get_value("Objective/Rel. Step (um)")
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
    dpg.set_value("Objective/Status", msg)
    dpg.set_value("Objective/Current Position (um)",position)
    dpg.set_value("Objective/Set Position (um)",setpoint)
    dpg.set_value("obj_pos_set",[setpoint,setpoint])
    dpg.set_value("obj_pos_get",[position,position])
    dpg.set("")
    return status['error']


################################################################################
############################### UI Building ####################################
################################################################################

###############
# Main Window #
###############
rdpg.initialize_dpg("Confocal")
with dpg.window(label="Confocal Scanner Window", tag='main_window'):
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
        dpg.add_checkbox(tag="scan",label="Take Scan", callback=start_scan)
        dpg.add_progress_bar(label="Scan Progress",tag='pb')
        dpg.add_button(tag="load_scan",label="Load Scan", callback=lambda:dpg.show_item("scan_picker"))

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
                dpg.add_text("Filename:")
                dpg.add_input_text(tag="save_scan_file", default_value="scan.npz", width=200)
                dpg.add_button(tag="save_scan_button", label="Save Scan",callback=save_scan)
                dpg.add_checkbox(tag="auto_save", label="Auto")
                dpg.add_button(tag="query_plot",label="Query Plot Range",callback=get_scan_range)

            with dpg.group(horizontal=True, width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="scan_tree"):
                    scan_tree = rdpg.TreeDict('scan_tree','cryo_gui_settings/scan_params_save.csv')
                    scan_tree.add("Galvo/Position",[float(f) for f in fpga.get_galvo()],
                                    item_kwargs={'min_value':-10,
                                                'max_value':10,
                                                "min_clamped":True,
                                                "max_clamped":True,
                                                "on_enter":True},
                                    callback=man_set_galvo)
                    scan_tree.add("Plot/Autoscale",False)
                    scan_tree.add("Plot/Bin Width",10,item_kwargs={'min_value':-10,
                                                                    'max_value':1})
                    scan_tree.add("Plot/Update Every Point",False)
                    scan_tree.add("Scan/Centers (V)",[0.0,0.0],item_kwargs={'min_value':-10,
                                                                            'max_value':10,
                                                                            "min_clamped":True,
                                                                            "max_clamped":True})
                    scan_tree.add("Scan/Spans (V)", [1.0,1.0],item_kwargs={'min_value':-20,
                                                                        'max_value':20,
                                                                        "min_clamped":True,
                                                                        "max_clamped":True})
                    scan_tree.add("Scan/Points", [100,100],item_kwargs={'min_value':0},
                                callback=guess_time)
                    scan_tree.add("Scan/Count Time (ms)", 10.0, item_kwargs={'min_value':0,
                                                                            'max_value':1000},
                                callback=guess_time)
                    scan_tree.add("Scan/Wait Time (ms)", 1.0,item_kwargs={'min_value':0,
                                                                        'max_value':1000},
                                callback=set_wait_time)
                    scan_tree.add("Scan/Estimated Time", "00:00:00", item_kwargs={'readonly':True},
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
                                dpg.add_heat_series(np.zeros((50,50)),50,50,
                                                    scale_min=0,scale_max=1000,
                                                    parent="heat_y",label="heatmap",
                                                    tag="heat_series",format='',)
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
            with dpg.group(horizontal=True,width=0):
                with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="obj_tree"):
                    obj_tree = rdpg.TreeDict('obj_tree','cryo_gui_settings/obj_tree_save.csv')
                    obj_tree.add("Objective/Initialize", False,save=False,callback=toggle_objective)
                    obj_tree.add("Objective/Status","Uninitialized",save=False,item_kwargs={"readonly":True})
                    obj_tree.add("Objective/Set Position (um)", 100.0,callback=set_objective_params,item_kwargs={"on_enter":True})
                    obj_tree.add("Objective/Current Position (um)", 100.0,callback=set_objective_params,item_kwargs={"readonly":True})
                    obj_tree.add("Objective/Limits (um)",[-8000.0,8000.0],callback=set_objective_params,item_kwargs={"on_enter":True})
                    obj_tree.add("Objective/Max Move (um)", 100.0,callback=set_objective_params,item_kwargs={"on_enter":True})
                    obj_tree.add("Objective/Rel. Step (um)", 5)
                    obj_tree.add("Joint Scan/Obj./Center (um)",0)
                    obj_tree.add("Joint Scan/Obj./Span (um)",50)
                    obj_tree.add("Joint Scan/Obj./Steps",50)
                    obj_tree.add("Joint Scan/Galvo/Fixed Axis",'x')
                    obj_tree.add("Joint Scan/Galvo/Center (V)",0)
                    obj_tree.add("Joint Scan/Galvo/Span (V)",0.05)
                    obj_tree.add("Joint Scan/Galvo/Steps",50)
                with dpg.child_window(width=0,height=0,autosize_y=True): 
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=100):
                            dpg.add_button(width=0,indent=25,arrow=True,direction=dpg.mvDir_Up,callback=obj_step_up)
                            dpg.add_button(width=0,indent=25,arrow=True,direction=dpg.mvDir_Down,callback=obj_step_down)
                            dpg.add_text('Obj. Pos.')
                            dpg.add_simple_plot(height=650,default_value=[0,0],min_scale=obj._soft_lower,max_scale=obj._soft_upper,tag='obj_pos_set')
                            dpg.add_simple_plot(height=650,default_value=[0,0],min_scale=obj._soft_lower,max_scale=obj._soft_upper,tag='obj_pos_get')
                        with dpg.child_window(width=-400,height=-200): 
                            with dpg.plot(label="Heat Series",width=-1,height=-1,
                                            equal_aspects=True,tag="obj_plot",query=True):
                                dpg.bind_font("plot_font")
                                # REQUIRED: create x and y axes
                                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="obj_heat_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="obj_heat_y")
                                dpg.add_heat_series(np.zeros((50,50)),50,50,
                                                    scale_min=0,scale_max=1000,
                                                    parent="obj_heat_y",label="heatmap",
                                                    tag="obj_heat_series",format='',)
                                dpg.bind_colormap("obj_plot",dpg.mvPlotColormap_Viridis)

                        with dpg.child_window(width=-0,height=-200):
                            with dpg.group(horizontal=True):
                                dpg.add_colormap_scale(min_scale=0,max_scale=1000,
                                                        width=100,height=-1,tag="obj_colormap",
                                                        colormap=dpg.mvPlotColormap_Viridis)
                                with dpg.plot(label="Histogram", width=-1,height=-1,tag='obj_histogram'):
                                    dpg.bind_font("plot_font")
                                    dpg.add_plot_axis(dpg.mvXAxis,label="Occurance",tag="obj_hist_x")
                                    dpg.add_plot_axis(dpg.mvYAxis,label="Counts",tag="obj_hist_y")
                                    dpg.add_area_series([0],[0],parent="obj_hist_x",
                                                        fill=[120,120,120,120],tag="obj_hist")
                                    dpg.add_drag_line(callback=set_scale,default_value=0,
                                                        parent="obj_histogram",tag="obj_line1",vertical=False)
                                    dpg.add_drag_line(callback=set_scale,default_value=0,
                                                        parent="obj_histogram",tag="obj_line2",vertical=False)
# Initialize Values
galvo_position = fpga.get_galvo()
print(galvo_position)
dpg.set_value("Scan/Centers (V)", [galvo_position[0],galvo_position[1]])
guess_time()

dpg.set_primary_window('main_window',True)
rdpg.start_dpg()
from .interface_template import Interface
from .hist_plot import mvHistPlot
from apis.scanner import Scanner
from apis import rdpg

from pathlib import Path
from threading import Thread
from typing import Callable

import datetime as dt
import logging
log = logging.getLogger(__name__)
import numpy as np

from logging import getLogger
log = getLogger(__name__)

dpg = rdpg.dpg

class ObjectiveInterface(Interface):
    def __init__(self, set_interfaces,obj,fpga,galvo,counter,treefix="obj_tree"):
        super().__init__()
        self.set_interfaces = set_interfaces
        self.obj = obj
        self.fpga = fpga
        self.galvo=galvo
        self.counter = counter
        self.treefix = treefix
        # Intialize Scanner object
        self.obj_scan = Scanner(self.obj_scan_func('x'),[0,0],[1,1],[50,50],[],[],float,['z','x'],default_result=-1)
        # Initialize the objective hist plot, which contains the heatmap and histogram of the data.
        self.obj_plot = mvHistPlot("Obj. Plot",False,None,True,False,1000,0,300,50,'viridis',True,1,1E9,50,50)

        objective_controls = [f"{self.treefix}_Objective/Initialize",
                              f"{self.treefix}_Objective/Set Position (um)",
                              f"{self.treefix}_Objective/Limits (um)",
                              f"{self.treefix}_Objective/Max Move (um)",
                              "obj_scan",
                              "obj_up",
                              "obj_dn",
                              "obj_get_errors"]
        objective_params = [f"{self.treefix}_Scan/Count Time (ms)",
                            f"{self.treefix}_Scan/Wait Time (ms)",
                            f"{self.treefix}_Scan/Obj./Center (um)",
                            f"{self.treefix}_Scan/Obj./Span (um)",
                            f"{self.treefix}_Scan/Obj./Steps",
                            f"{self.treefix}_Scan/Galvo/Axis",
                            f"{self.treefix}_Scan/Galvo/Center (V)",
                            f"{self.treefix}_Scan/Galvo/Span (V)",
                            f"{self.treefix}_Scan/Galvo/Steps"]
    def initialize(self):
        if not self.gui_exists:
            raise RuntimeError("GUI must be made before initialization.")
        self.tree.load()

        self.guess_obj_time()

    def makeGUI(self, parent):
        with dpg.group(horizontal=True,parent=parent):
            dpg.add_checkbox(label="Scan Objective",tag="obj_scan", default_value=False, callback=self.start_obj_scan)
            dpg.add_button(tag="query_obj_plot",label="Copy Scan Params",callback=self.get_obj_range)
            dpg.add_text("Filename:")
            dpg.add_input_text(tag="save_obj_file", default_value="scan.npz", width=200)
            dpg.add_button(tag="save_obj_button", label="Save Scan",callback=self.save_obj_scan)
            dpg.add_checkbox(tag="auto_save_obj", label="Auto")
            dpg.add_button(tag="obj_get_errors",label="Get Obj. Errors",callback=self.get_obj_errors)
        with dpg.group(horizontal=True,width=0,parent=parent):
            with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag=f"{self.treefix}"):
                self.tree = rdpg.TreeDict(self.treefix,f'cryo_gui_settings/{self.treefix}_save.csv')
                self.tree.add("Objective/Initialize", False,save=False,callback=self.toggle_objective)
                self.tree.add("Objective/Status","Uninitialized",save=False,item_kwargs={"readonly":True})
                self.tree.add("Objective/Set Position (um)", 100.0,callback=self.set_obj_callback,item_kwargs={"on_enter":True,'step':0})
                self.tree.add("Objective/Current Position (um)", 100.0,item_kwargs={"readonly":True,'step':0})
                self.tree.add("Objective/Limits (um)",[-8000.0,8000.0],callback=self.set_objective_params,item_kwargs={"on_enter":True})
                self.tree.add("Objective/Max Move (um)", 100.0,callback=self.set_objective_params,item_kwargs={"on_enter":True,'step':1})
                self.tree.add("Objective/Rel. Step (um)", 5.0,item_kwargs={"min_value":0,"min_clamped":True,"step":1})
                self.tree.add("Scan/Count Time (ms)", 10.0,callback=self.guess_obj_time,item_kwargs={'min_value':0,'min_clamped':True,"step":1})
                self.tree.add("Scan/Wait Time (ms)", 5.0,callback=self.guess_obj_time,item_kwargs={'min_value':0,'min_clamped':True,"step":1})
                self.tree.add("Scan/Obj./Center (um)",0.0, item_kwargs={"step":0})
                self.tree.add("Scan/Obj./Span (um)",50.0, item_kwargs={"step":0})
                self.tree.add("Scan/Obj./Steps",50,callback=self.guess_obj_time, item_kwargs={"step":0})
                self.tree.add_radio("Scan/Galvo/Axis",['x','y'],'x')
                self.tree.add("Scan/Galvo/Center (V)",0.0, item_kwargs={"step":0})
                self.tree.add("Scan/Galvo/Span (V)",0.05, item_kwargs={"step":0})
                self.tree.add("Scan/Galvo/Steps",50,callback=self.guess_obj_time, item_kwargs={"step":0})
                self.tree.add("Scan/Estimated Time", "00:00:00", item_kwargs={"readonly":True})
                self.tree.add("Plot/Autoscale",False)
                self.tree.add("Plot/N Bins",50,item_kwargs={'min_value':1,
                                                                'max_value':1000})
                self.tree.add("Plot/Update Every Point",False)

            with dpg.child_window(width=0,height=0,autosize_y=True):
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=100):
                        dpg.add_button(width=0,indent=25,arrow=True,direction=dpg.mvDir_Up,callback=self.obj_step_up,tag='obj_up')
                        dpg.add_button(width=0,indent=25,arrow=True,direction=dpg.mvDir_Down,callback=self.obj_step_down,tag='obj_dn')
                        dpg.add_text('Obj. Pos.')
                        dpg.bind_item_font(dpg.last_item(),"small_font")
                        with dpg.group(horizontal=True):
                            dpg.bind_item_font(dpg.last_item(),"small_font")
                            dpg.add_slider_float(height=600,width=35,default_value=0,
                                                    min_value=self.obj._soft_lower, vertical=True,
                                                    max_value=self.obj._soft_upper,tag='obj_pos_set',
                                                    enabled=False,no_input=True)
                            dpg.add_slider_float(height=600,width=35,default_value=0,
                                                    min_value=self.obj._soft_lower, vertical=True,
                                                    max_value=self.obj._soft_upper,tag='obj_pos_get',
                                                    enabled=False,no_input=True)
                            
                    with dpg.group():
                        self.obj_plot.parent = dpg.last_item()
                        self.obj_plot.height = -330
                        self.obj_plot.scale_width = 335
                        self.obj_plot.make_gui()
                        with dpg.child_window(width=-0,height=320):
                            with dpg.plot(label="Count Rate",width=-1,height=300,tag="count_plot3"):
                                dpg.bind_font("plot_font") 
                                # REQUIRED: create x and y axes
                                dpg.add_plot_axis(dpg.mvXAxis, label="x", time=True, tag="count_x3")
                                dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="count_y3")
                                dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="count_AI13",no_gridlines=True)
                                dpg.add_line_series(rdpg.offset_timezone(self.counter.data['time']),
                                                    self.counter.data['counts'],
                                                    parent='count_y3',label='counts', tag='counts_series3')
                                dpg.add_line_series(rdpg.offset_timezone(self.counter.data['time']),
                                                    self.counter.data['counts'],
                                                    parent='count_y3',label='avg. counts', tag='avg_counts_series3')
                                dpg.add_line_series(rdpg.offset_timezone(self.counter.data['time']),
                                                    self.counter.data['AI1'],
                                                    parent='count_AI13',label='AI1', tag='AI1_series3')
                                dpg.set_item_source('counts_series3','counts_series')
                                dpg.set_item_source('avg_counts_series3','avg_counts_series')
                                dpg.set_item_source('AI1_series3','AI1_series')
                                dpg.bind_item_theme("counts_series3","plot_theme_blue")
                                dpg.bind_item_theme("avg_counts_series3","avg_count_theme")
                                dpg.bind_item_theme("AI1_series3","plot_theme_purple")
                                dpg.add_plot_legend()
        self.gui_exists = True


    # NOTE:
    # The bare api of the objective control is setup so that more negative values
    # are physically upwards in the cryostat.
    # Here, we have opted to invert that, such that a more positive value
    # is upwards in the cryo, such that plotting and moving makes more sense.
    def obj_scan_func(self, galvo_axis:str='x') -> Callable:
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
                obj_move = not (abs(z - self.tree["Objective/Set Position (um)"]) < 0.01)
                if obj_move:
                    obj_thread = self.set_obj_abs(z)
                    log.debug(f"Set obj. position to {z} um.")
                self.galvo.set_galvo(None,y,write=False)
                if obj_move:
                    obj_thread.join()
                count = self.counter.get_count(self.tree["Scan/Count Time (ms)"])
                log.debug(f"Got count rate of {count}.")
                return count
            return func
        if galvo_axis=='x':
            # Make the function which scans the z of the objective and the x of the galvo.
            def func(z,x):
                log.debug(f"Set galvo x to {x} V.")
                log.debug(f"Set obj. position to {z} um.")
                obj_move = not (abs(z - self.tree["Objective/Set Position (um)"]) < 0.01)
                if obj_move:
                    obj_thread = self.set_obj_abs(z)
                    log.debug(f"Set obj. position to {z} um.")
                self.galvo.set_galvo(x,None,write=False)
                if obj_move:
                    obj_thread.join()
                count = self.counter.get_count(self.tree["Scan/Count Time (ms)"])
                log.debug(f"Got count rate of {count}.")
                return count
            return func

    def start_obj_scan(self, sender,app_data,user_data):
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
        self.counter.abort_counts()

        # Get paramters from the GUI and set the scanner object up.
        obj_steps = self.tree["Scan/Obj./Steps"]
        obj_center = self.tree["Scan/Obj./Center (um)"]
        obj_span = self.tree["Scan/Obj./Span (um)"]
        galv_steps = self.tree["Scan/Galvo/Steps"]
        galv_center = self.tree["Scan/Galvo/Center (V)"]
        galv_span = self.tree["Scan/Galvo/Span (V)"]
        self.obj_scan.steps = [obj_steps,galv_steps]
        self.obj_scan.centers = [obj_center,galv_center]
        self.obj_scan.spans = [obj_span,galv_span]
        
        def init():
            """
            The scan initialization function.
            Setsup the galvo and plots.
            """
            
            self.fpga.set_ao_wait(self.tree["Scan/Wait Time (ms)"],write=False)
            pos = self.obj_scan._get_positions()
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            self.position_register["temp_obj_position"] = -self.obj.position
            self.position_register["temp_galvo_position"] = self.fpga.get_galvo()
            self.obj_plot.set_size(int(self.obj_scan.steps[0]),int(self.obj_scan.steps[1]))
            self.obj_plot.set_bounds(xmin,xmax,ymin,ymax)
            self.set_interfaces("obj",False)
        
        def abort(i,imax,idx,pos,res):
            return not dpg.get_value("obj_scan")

        def prog(i,imax,idx,pos,res):
                log.debug("Setting Progress Bar")
                dpg.set_value("pb",(i+1)/imax)
                dpg.configure_item("pb",overlay=f"Obj. Scan {i+1}/{imax}")
                if self.tree["Plot/Update Every Point"]:
                    check = True
                else:
                    check = (not (i+1) % self.tree["Scan/Galvo/Steps"]) or (i+1)==imax
                if check:
                    log.debug("Updating Galvo Scan Plot")
                    plot_data = np.copy(np.flip(self.obj_scan.results,0))
                    self.obj_plot.autoscale = self.tree["Plot/Autoscale"]
                    self.obj_plot.nbin = self.tree["Plot/N Bins"]
                    self.obj_plot.update_plot(plot_data)
                    if self.counter.counter.tree["Counts/Plot Scan Counts"]:
                        self.counter.plot_counts()

        def finish(results,completed):
            #Reenable controls first to avoid blocking
            self.set_interfaces("obj",True)
            dpg.set_value("obj_scan",False)
            self.set_obj_abs(self.position_register["temp_obj_position"])
            self.galvo.set_galvo(*self.position_register["temp_galvo_position"])
            if dpg.get_value("auto_save"):
                self.save_obj_scan()

                
        self.obj_scan._func = self.obj_scan_func(self.tree['Scan/Galvo/Axis'])
        self.obj_scan._init_func = init
        self.obj_scan._abort_func = abort
        self.obj_scan._prog_func = prog
        self.obj_scan._finish_func = finish
        return self.obj_scan.run_async()

    def save_obj_scan(self,*args):
        """
        Saves the objective scan data, using the Scanners built in saving method.
        """
        path = Path(dpg.get_value("save_dir"))
        filename = dpg.get_value("save_obj_file")
        path /= filename
        as_npz = not (".csv" in filename)
        header = self.obj_scan.gen_header("Objective Scan")
        self.obj_scan.save_results(str(path),as_npz=as_npz,header=header)

    def toggle_objective(self,sender,app_data,user_data):
        """
        Callback for initializing the objective.
        Uses the app_data, i.e. new checkmar value to check if we initialize
        or deinitialize.

        Parameters
        ----------
        See standard DPG callback parameters.
        """
        if app_data:
            self.obj.initialize()
            self.tree["Objective/Status"] = "Initialized"
            pos = -self.obj.position
            self.tree["Objective/Current Position (um)"] = pos
            self.tree["Objective/Set Position (um)"] = pos
            self.tree["Objective/Limits (um)"] = [self.obj.soft_lower,self.obj.soft_upper]
            self.set_objective_params()
        else:
            self.obj.deinitialize()
            self.tree["Objective/Status"] = "Deinitialized"

    def set_objective_params(self,*args):
        if self.obj.initialized:
            limits = self.tree["Objective/Limits (um)"]
            self.obj.soft_lower = limits[0]
            self.obj.soft_upper = limits[1]
            self.obj.max_move = self.tree["Objective/Max Move (um)"]
            dpg.configure_item('obj_pos_set',min_value=limits[0],max_value=limits[1])
            dpg.configure_item('obj_pos_get',min_value=limits[0],max_value=limits[1])
            dpg.set_value("obj_pos_set",-self.obj.position)
            dpg.set_value("obj_pos_get",-self.obj.position)

    def set_obj_callback(self,sender,app_data,user_data):
        return self.set_obj_abs(app_data)

    def set_obj_abs(self,position):
        position = -position
        log.debug(f"Set objective to {-position}")
        def func():
            self.obj.move_abs(position,monitor=True,monitor_callback=self.obj_move_callback)
        t = Thread(target=func)
        t.start()
        return t

    def obj_step_up(self,*args):
        def func():
            step = self.tree["Objective/Rel. Step (um)"]
            self.obj.move_up(step,monitor=True,monitor_callback=self.obj_move_callback)
        t = Thread(target=func)
        t.start()
        return t

    def obj_step_down(self,*args):
        def func():
            step = self.tree["Objective/Rel. Step (um)"]
            self.obj.move_down(step,monitor=True,monitor_callback=self.obj_move_callback)
        t = Thread(target=func)
        t.start()
        return t

    def obj_move_callback(self,status,position,setpoint):
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
        self.tree["Objective/Status"] = msg
        self.tree["Objective/Current Position (um)"] = -position
        self.tree["Objective/Set Position (um)"] = -setpoint
        dpg.set_value("obj_pos_set",-setpoint)
        dpg.set_value("obj_pos_get",-position)
        return status['error']

    def guess_obj_time(self,*args):
        obj_pts = self.tree["Scan/Obj./Steps"]
        galvo_pts = self.tree["Scan/Galvo/Steps"]
        ctime = self.tree["Scan/Count Time (ms)"] + self.tree["Scan/Wait Time (ms)"]
        scan_time = obj_pts * galvo_pts * ctime / 1000
        time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
        self.tree["Scan/Estimated Time"] = time_string

    def get_obj_range(self):
        xmin,xmax,ymin,ymax = self.obj_plot.query_plot()
        if xmin is None:
            idx = 0 if self.tree["Scan/Galvo/Axis"] == 'x' else 1
            self.tree["Scan/Galvo/Center (V)"] = self.galvo.tree["Galvo/Position"][idx]
            self.tree["Scan/Obj./Center (um)"] = self.tree["Objective/Current Position (um)"]
        else:
            new_centers = [(xmin+xmax)/2, (ymin+ymax)/2]
            new_spans = [xmax-xmin, ymax-ymin]
            self.tree["Scan/Galvo/Center (V)"] = new_centers[0]
            self.tree["Scan/Galvo/Span (V)"] = new_spans[0]
            self.tree["Scan/Obj./Center (um)"] = new_centers[1]
            self.tree["Scan/Obj./Span (um)"] = new_spans[1]

    def get_obj_errors(self):
        errors = self.obj.errors
        if errors == []:
            return
        with dpg.window(label="Objective Errors", modal=True, show=True, tag="obj_errors", 
                        pos=[int((dpg.get_viewport_width() // 2 - 500 // 2)),
                            int((dpg.get_viewport_height() // 2 - 500 // 2))],
                        width=500,
                        height=500):
            dpg.add_text('\n'.join(errors))
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="OK", width=75, callback=lambda: dpg.delete_item("obj_errors"))
from .interface_template import Interface
from .hist_plot import mvHistPlot
from apis.scanner import Scanner
from apis.fpga_base import FPGAValueError
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

from apis.jpe_coord_convert import JPECoord
pz_config = {"vmax" : 0,
             "vmin" : -6.5,
             "vgain" : -20,
             "R": 6.75,
             "h": 45.1}

class PiezoInterface(Interface):

    def __init__(self,set_interfaces,fpga,counter,treefix="pzt_tree"):
        super().__init__()
        
        self.fpga = fpga
        self.counter = counter
        self.set_interfaces = set_interfaces
        self.treefix = treefix
        self.position_register = {}
        self.pz_conv = JPECoord(pz_config['R'], pz_config['h'],
                                pz_config['vmin'], pz_config['vmax'])

        self.jpe_xy_scan = Scanner(self.set_xy_and_count,[0,0],[1,1],[50,50],[1],[],float,['y','x'],default_result=-1)
        self.jpe_cav_scan = Scanner(self.set_cav_and_count,[0],[1],[50],[],[],float,['z'],default_result=-1)
        self.jpe_3D_scan = Scanner(self.set_xy_get_cav,[0,0],[1,1],[50,50],[1],[],object,['y','x'],default_result=np.array([-1]))
        self.plot = mvHistPlot("Piezo Scan",True,None,True,True,1000,0,300,1000,'viridis',True,1,1E12,50,50)
        self.plot.cursor_callback=self.xy_cursor_callback

        self.controls = [ f"{self.treefix}_JPE/Z0 Position",
                          f"{self.treefix}_JPE/Set Z Position",
                          f"{self.treefix}_JPE/XY Position",
                          f"{self.treefix}_Cavity/Position",
                          "pzt_xy_scan",
                          "pzt_cav_scan",
                          "pzt_3d_scan"]
        self.params =  [f"{self.treefix}_Scan/Count Time (ms)",
                        f"{self.treefix}_Scan/Cavity/Center",
                        f"{self.treefix}_Scan/Cavity/Wait Time (ms)",
                        f"{self.treefix}_Scan/Cavity/Span",
                        f"{self.treefix}_Scan/Cavity/Steps",
                        f"{self.treefix}_Scan/JPE/Wait Time (ms)",
                        f"{self.treefix}_Scan/JPE/Center",
                        f"{self.treefix}_Scan/JPE/Span",
                        f"{self.treefix}_Scan/JPE/Steps",
                        f"{self.treefix}_JPE/Tilt/Enable",
                        f"{self.treefix}_JPE/Tilt/XY Slope (V\V)",
                        f"{self.treefix}_JPE/Tilt/Abs. Limit"]
    def initialize(self):
        if not self.gui_exists:
            raise RuntimeError("GUI must be made before initialization.")
        self.tree.load()

        cavity_position = self.fpga.get_cavity()
        jpe_position = self.fpga.get_jpe_pzs()
        self.tree["Cavity/Position"] = cavity_position[0]
        # Enabling tilt will ensure that no sudden z0 position change occurs
        # In doing so, since we recall the tilt after the z position
        # It will properly set 
        self.tree["JPE/Z0 Position"] = jpe_position[2]
        self.tree["JPE/XY Position"] = jpe_position[:2]
        self.tree["JPE/Z Volts"] = self.pz_conv.zs_from_cart(jpe_position)
        self.toggle_tilt(None,self.tree["JPE/Tilt/Enable"], None)

        self.guess_pzt_times()
        self.draw_bounds()
        self.plot.set_cursor(jpe_position[:2])


    def set_controls(self,state:bool,ignore:str=None) -> None:
        if state:
            for control in self.controls:
                if control != ignore:
                    log.debug(f"Enabling {control}")
                    dpg.enable_item(control)
            self.plot.enable_cursor()
        else:
            for control in self.controls:
                if control != ignore:
                    log.debug(f"Disabling {control}")
                    dpg.disable_item(control)
            self.plot.disable_cursor()

    def makeGUI(self, parent):
        with dpg.group(horizontal=True,parent=parent):
                dpg.add_checkbox(tag="pzt_xy_scan",label="Scan XY", callback=self.start_xy_scan)
                dpg.add_checkbox(tag="pzt_cav_scan",label="Scan Cav.", callback=self.start_cav_scan)
                dpg.add_checkbox(tag="pzt_3d_scan",label="Scan 3D", callback=self.start_3d_scan)
                dpg.add_text("Filename:")
                dpg.add_input_text(tag="save_pzt_file", default_value="scan.npz", width=200)
                dpg.add_text("Save:")
                dpg.add_button(label="XY", tag="save_xy_button",callback=self.save_xy_scan)
                dpg.add_button(label="3D", tag="save_3d_button",callback=self.save_3d_scan)
                dpg.add_button(label="Cav", tag="save_cav_button",callback=self.save_cav_scan)
                dpg.add_checkbox(tag="pzt_auto_save", label="Auto",default_value=False)
                dpg.add_button(tag="query_xy_plot",label="XY Copy Scan Params",callback=self.get_xy_range)
                dpg.add_button(tag="query_cav_plot",label="Cav. Copy Scan Params",callback=self.get_cav_range)
                
        with dpg.group(horizontal=True,width=0,parent=parent):
            with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag=f"{self.treefix}"):
                self.tree = rdpg.TreeDict(self.treefix,f'cryo_gui_settings/{self.treefix}_save.csv')
                self.tree.add("JPE/Z0 Position",0.0,
                                item_kwargs={'min_value':-6.5,'max_value':0,
                                            'min_clamped':True,'max_clamped':True,
                                            'on_enter':True,'step':0.05},
                                            callback=self.z_pos_callback,save=False)
                self.tree.add("JPE/Set Z Position",0.0,
                                item_kwargs={'min_value':-6.5,'max_value':0,
                                'min_clamped':True,'max_clamped':True,
                                'on_enter':True,'step':0.05},
                                callback=self.tilt_z_pos_callback,save=False)
                self.tree.add("JPE/XY Position",[0.0,0.0],
                                item_kwargs={'on_enter':True},
                                callback=self.xy_pos_callback,save=False)
                self.tree.add("JPE/Z Volts", [0.0,0.0,0.0], save=False,
                                item_kwargs={"readonly" : True})
                self.tree.add("JPE/Tilt/Enable", False, callback=self.toggle_tilt)
                self.tree.add("JPE/Tilt/XY Slope (V\V)", [0.0,0.0],callback=self.tilt_callback,item_kwargs={'on_enter':True})
                self.tree.add("JPE/Tilt/Abs. Limit", 1.0, callback=self.tilt_callback,item_kwargs={"min_value":0,"min_clamped":True})
                self.tree.add("JPE/Tilt/Z Offset",0.0,save=False, item_kwargs={'readonly':True,'step':0})

                self.tree.add("Cavity/Position",0.0,
                                item_kwargs={'min_value':-8,'max_value':8,
                                            'min_clamped':True,'max_clamped':True,
                                            'on_enter':True,'step':0.5},
                                callback=self.man_set_cavity, save=False)
                self.tree.add("Scan/Count Time (ms)",5.0,callback=self.guess_pzt_times,item_kwargs={'step':1})
                self.tree.add("Scan/Cavity/Wait Time (ms)",1.0,callback=self.guess_pzt_times,item_kwargs={'step':1})
                self.tree.add("Scan/Cavity/Center",0.0, item_kwargs={"step":0})
                self.tree.add("Scan/Cavity/Span",16.0, item_kwargs={"step":0})
                self.tree.add("Scan/Cavity/Steps",300,callback=self.guess_pzt_times, item_kwargs={"step":0})
                self.tree.add("Scan/Cavity/Estimated Time", "00:00:00", save=False,item_kwargs={'readonly':True})
                self.tree.add("Scan/JPE/Wait Time (ms)",25.0,callback=self.guess_pzt_times,item_kwargs={'step':1})
                self.tree.add("Scan/JPE/Center",[0.0,0.0])
                self.tree.add("Scan/JPE/Span",[5.0,5.0])
                self.tree.add("Scan/JPE/Steps",[15,15],callback=self.guess_pzt_times)
                self.tree.add("Scan/JPE/Estimated Time", "00:00:00", save=False,item_kwargs={'readonly':True})
                self.tree.add("Scan/Estimated Time", "00:00:00", save=False,item_kwargs={'readonly':True})
                self.tree.add("Plot/Autoscale",False)
                self.tree.add("Plot/N Bins",50,item_kwargs={'min_value':1,
                                                            'max_value':1000},
                                callback=self.update_pzt_plot)
                self.tree.add("Plot/Update Every Point",False)
                self.tree.add("Plot/Deinterleave",False,callback=self.update_pzt_plot)
                self.tree.add("Plot/Reverse",False,callback=self.update_pzt_plot)
        
                self.tree.add("Plot/3D/Slice Index",0,drag=True,callback=self.update_pzt_plot,
                                item_kwargs={"min_value":0,"max_value":100,"clamped":True})
                self.tree.add_combo("Plot/3D/Reducing Func.",
                                    values=["Delta","Max","Average","Slice"],
                                    default="Delta",
                                    callback=self.update_pzt_plot)
            with dpg.child_window(width=0,height=0,autosize_y=True):
                    with dpg.group():
                        self.plot.parent = dpg.last_item()
                        self.plot.height = -330
                        self.plot.scale_width = 335
                        self.plot.make_gui()
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
                                                    callback=self.update_pzt_plot)
                                dpg.add_plot_legend()
        self.gui_exists = True

    def set_cav_and_count(self,z):
        try:
            self.set_cav_pos(z,write=False)
        except FPGAValueError:
            return 0
        count = self.counter.get_count(self.tree["Scan/Count Time (ms)"])
        return count

    def set_xy_and_count(self,y,x):
        try:
            self.set_jpe_xy(x,y,write=False)
        except FPGAValueError:
            return 0
        count = self.counter.get_count(self.tree["Scan/Count Time (ms)"])
        return count

    def do_cav_scan_step(self):
        if not dpg.get_value("pzt_3d_scan"):
            return [-1]

        cav_data = {'counts': [], 'pos' : []}

        def init():
            self.fpga.set_ao_wait(self.tree["Scan/Cavity/Wait Time (ms)"],write=False)
            log.debug("Starting cav scan sub.")
            pos = self.jpe_cav_scan._get_positions()
            xmin = np.min(pos[0])
            xmax = np.max(pos[0])
            self.position_register["temp_cav_position"] = self.fpga.get_cavity()
            if self.tree['Plot/Autoscale']:
                dpg.set_axis_limits("cav_count_x",xmin,xmax)
            else:
                dpg.set_axis_limits_auto("cav_count_x")
        
        def abort(i,imax,idx,pos,res):
            return not dpg.get_value("pzt_3d_scan")

        def prog(i,imax,idx,pos,res):
            log.debug("Updating Cav Scan Plot")
            cav_data['counts'].append(res)
            cav_data['pos'].append(pos[0])
            check = self.tree["Plot/Update Every Point"] or i + 1 >= imax
            if check:
                dpg.set_value("cav_counts",[cav_data['pos'],cav_data['counts']])
                if self.tree['Plot/Autoscale']:
                    dpg.set_axis_limits_auto("cav_count_y")
                    dpg.fit_axis_data("cav_count_y")
                if self.counter.tree["Counts/Plot Scan Counts"]:
                    self.counter.plot_counts()

        def finish(results,completed):
            if self.tree['Plot/Autoscale']:
                dpg.set_axis_limits_auto("cav_count_y")
                dpg.fit_axis_data("cav_count_y")
            else:
                dpg.set_axis_limits_auto("cav_count_y")
            self.set_cav_pos(*self.position_register["temp_cav_position"])
            self.fpga.set_ao_wait(self.tree["Scan/JPE/Wait Time (ms)"],write=False)

        self.jpe_cav_scan._init_func = init
        self.jpe_cav_scan._abort_func = abort
        self.jpe_cav_scan._prog_func = prog
        self.jpe_cav_scan._finish_func = finish
        return self.jpe_cav_scan.run_async()

    def set_xy_get_cav(self,y,x):
        try:
            self.set_jpe_xy(x,y,write=True)
            self.do_cav_scan_step().join()
        except FPGAValueError:
            return np.array([-1])
        return self.jpe_cav_scan.results

    def start_xy_scan(self):
        if not dpg.get_value("pzt_xy_scan"):
            return -1
        self.counter.abort_counts()

        self.fpga.set_ao_wait(self.tree["Scan/JPE/Wait Time (ms)"],write=False)
        steps = self.tree["Scan/JPE/Steps"][:2][::-1]
        centers = self.tree["Scan/JPE/Center"][:2][::-1]
        spans = self.tree["Scan/JPE/Span"][:2][::-1]
        self.jpe_xy_scan.steps = steps
        self.jpe_xy_scan.centers = centers
        self.jpe_xy_scan.spans = spans
        
        def init():
            pos = self.jpe_xy_scan._get_positions()
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            self.position_register["temp_jpe_position"] = self.fpga.get_jpe_pzs()
            self.plot.set_size(int(self.jpe_xy_scan.steps[0]),int(self.jpe_xy_scan.steps[1]))
            self.plot.set_bounds(xmin,xmax,ymin,ymax)
            dpg.configure_item("Piezo Scan_heat_series",label="2D Scan")
            self.set_interfaces("pzt",False, "pzt_xy_scan")
        
        def abort(i,imax,idx,pos,res):
            return not dpg.get_value("pzt_xy_scan")

        def prog(i,imax,idx,pos,res):
                log.debug("Setting Progress Bar")
                dpg.set_value("pb",(i+1)/imax)
                dpg.configure_item("pb",overlay=f"JPE XY Scan {i+1}/{imax}")
                if self.tree["Plot/Update Every Point"]:
                    check = True
                else:
                    check = (not (i+1) % self.tree["Scan/JPE/Steps"][0]) or (i+1)==imax
                if check:
                    log.debug("Updating XY Scan Plot")
                    self.update_pzt_plot("manual",None,None)
                    if self.counter.tree["Counts/Plot Scan Counts"]:
                        self.counter.plot_counts()

        def finish(results,completed):
            #Reenable controls first to avoid blocking
            self.set_interfaces("pzt",True, "pzt_xy_scan")
            dpg.set_value("pzt_xy_scan",False)
            try:
                self.set_jpe_xy(*self.position_register["temp_jpe_position"][:2])
            except FPGAValueError:
                log.warn("Couldn't Reset PZT Position.")
            if dpg.get_value("pzt_auto_save"):
                self.save_xy_scan()
            self.fpga.set_ao_wait(self.counter.tree["Counts/Wait Time (ms)"],write=False)
        self.jpe_xy_scan._init_func = init
        self.jpe_xy_scan._abort_func = abort
        self.jpe_xy_scan._prog_func = prog
        self.jpe_xy_scan._finish_func = finish
        return self.jpe_xy_scan.run_async()

    def get_xy_range(self):
        xmin,xmax,ymin,ymax = self.plot.query_plot()
        if xmin is not None:
            new_centers = [(xmin+xmax)/2, (ymin+ymax)/2]
            new_spans = [xmax-xmin, ymax-ymin]
            self.tree["Scan/JPE/Center"] = new_centers
            self.tree["Scan/JPE/Span"] = new_spans
        else:
            self.tree["Scan/JPE/Center"] = self.tree["JPE/XY Position"]

    def start_cav_scan(self):
        if not dpg.get_value("pzt_cav_scan"):
            return -1
        self.counter.abort_counts()

        steps = self.tree["Scan/Cavity/Steps"]
        centers = self.tree["Scan/Cavity/Center"]
        spans = self.tree["Scan/Cavity/Span"]
        self.jpe_cav_scan.steps = [steps]
        self.jpe_cav_scan.centers = [centers]
        self.jpe_cav_scan.spans = [spans]
        cav_data = {}

        def init():
            self.fpga.set_ao_wait(self.tree["Scan/Cavity/Wait Time (ms)"],write=False)
            pos = self.jpe_cav_scan._get_positions()
            cav_data['counts'] = []
            cav_data['pos'] = []
            xmin = np.min(pos[0])
            xmax = np.max(pos[0])
            self.position_register["temp_cav_position"] = self.fpga.get_cavity()
            if self.tree['Plot/Autoscale']:
                dpg.set_axis_limits("cav_count_x",xmin,xmax)
            else:
                dpg.set_axis_limits_auto("cav_count_x")
            dpg.configure_item(f"{self.treefix}_Plot/3D/Slice Index",max_value=self.tree["Scan/Cavity/Steps"]-1)
            self.set_interfaces("pzt",False, "pzt_cav_scan")
        
        def abort(i,imax,idx,pos,res):
            return not dpg.get_value("pzt_cav_scan")

        def prog(i,imax,idx,pos,res):
            log.debug("Setting Progress Bar")
            dpg.set_value("pb",(i+1)/imax)
            dpg.configure_item("pb",overlay=f"Cavity Scan {i+1}/{imax}")
            log.debug("Updating XY Scan Plot")
            cav_data['counts'].append(res)
            cav_data['pos'].append(pos[0])
            dpg.set_value("cav_counts",[cav_data['pos'],cav_data['counts']])
            if self.tree['Plot/Autoscale']:
                dpg.set_axis_limits_auto("cav_count_y")
                dpg.fit_axis_data("cav_count_y")
            if self.counter.tree["Counts/Plot Scan Counts"]:
                self.counter.plot_counts()

        def finish(results,completed):
            #Reenable controls first to avoid blocking.
            self.set_interfaces("pzt",True, "pzt_cav_scan")

            dpg.set_value("pzt_cav_scan",False)
            if self.tree['Plot/Autoscale']:
                dpg.set_axis_limits_auto("cav_count_y")
                dpg.fit_axis_data("cav_count_y")
            else:
                dpg.set_axis_limits_auto("cav_count_y")
            self.set_cav_pos(*self.position_register["temp_cav_position"],write=True)
            if dpg.get_value("pzt_auto_save"):
                self.save_cav_scan()
            self.fpga.set_ao_wait(self.counter.tree["Counts/Wait Time (ms)"],write=False)

        self.jpe_cav_scan._init_func = init
        self.jpe_cav_scan._abort_func = abort
        self.jpe_cav_scan._prog_func = prog
        self.jpe_cav_scan._finish_func = finish
        return self.jpe_cav_scan.run_async()

    def get_cav_range(self):
        if dpg.is_plot_queried("cav_plot"):
            xmin,xmax,ymin,ymax = dpg.get_plot_query_area("cav_plot")
            new_center = (xmin+xmax)/2
            new_span = xmax-xmin
            self.tree["Scan/Cavity/Center"] = new_center
            self.tree["Scan/Cavity/Span"] = new_span
        else:
            self.tree["Scan/Cavity/Center"] = self.tree["Cavity/Position"]

    def get_reducing_func(self):
        func_str = self.tree["Plot/3D/Reducing Func."]
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
            f = lambda d:d[self.tree["Plot/3D/Slice Index"]] if np.atleast_1d(d)[0] >= 0 else -1.0
            do_func = np.vectorize(f)
            return do_func

    def start_3d_scan(self):
        if not dpg.get_value("pzt_3d_scan"):
            return -1
        self.counter.abort_counts()
        # To maximize performance, disable plotting of scan counts.
        self.counter.tree['Counts/Plot Scan Counts'] = False

        jpe_steps = self.tree["Scan/JPE/Steps"][:2][::-1]
        jpe_centers = self.tree["Scan/JPE/Center"][:2][::-1]
        jpe_spans = self.tree["Scan/JPE/Span"][:2][::-1]
        self.jpe_3D_scan.steps = jpe_steps
        self.jpe_3D_scan.centers = jpe_centers
        self.jpe_3D_scan.spans = jpe_spans

        steps = self.tree["Scan/Cavity/Steps"]
        centers = self.tree["Scan/Cavity/Center"]
        spans = self.tree["Scan/Cavity/Span"]
        self.jpe_cav_scan.steps = [steps]
        self.jpe_cav_scan.centers = [centers]
        self.jpe_cav_scan.spans = [spans]

        def init():
            self.fpga.set_ao_wait(self.tree["Scan/JPE/Wait Time (ms)"],write=False)
            pos = self.jpe_3D_scan._get_positions()
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            self.position_register["temp_jpe_position"] = self.fpga.get_jpe_pzs()
            self.position_register["temp_cav_position"] = self.fpga.get_cavity()
            self.plot.set_size(int(self.jpe_3D_scan.steps[0]),int(self.jpe_3D_scan.steps[1]))
            self.plot.set_bounds(xmin,xmax,ymin,ymax)
            dpg.configure_item("Piezo Scan_heat_series",label="3D Scan")
            dpg.configure_item(f"{self.treefix}_Plot/3D/Slice Index",max_value=self.tree["Scan/Cavity/Steps"]-1)
            self.set_interfaces("pzt",False, "pzt_3d_scan")
        
        def abort(i,imax,idx,pos,res):
            return not dpg.get_value("pzt_3d_scan")

        def prog(i,imax,idx,pos,res):
            log.debug("Setting Progress Bar")
            dpg.set_value("pb",(i+1)/imax)
            dpg.configure_item("pb",overlay=f"JPE 3D Scan {i+1}/{imax}")
            if self.tree["Plot/Update Every Point"]:
                check = True
            else:
                check = (not (i+1) % self.jpe_3D_scan.steps[1]) or (i+1)==imax
            if check:
                log.debug("Updating 3D Scan Plot")
                self.update_pzt_plot("manual",None,None)
                if self.counter.tree["Counts/Plot Scan Counts"]:
                    self.counter.plot_counts()

        def finish(results,completed):
            #Reenable controls first to avoid blocking
            self.set_interfaces("pzt",True, "pzt_3d_scan")

            dpg.set_value("pzt_3d_scan",False)
            self.set_jpe_xy(*self.position_register["temp_jpe_position"][:2])
            self.set_cav_pos(*self.position_register["temp_cav_position"])
            self.fpga.set_ao_wait(self.counter.tree["Counts/Wait Time (ms)"],write=False)
            if dpg.get_value("pzt_auto_save"):
                self.save_xy_scan()

        self.jpe_3D_scan._init_func = init
        self.jpe_3D_scan._abort_func = abort
        self.jpe_3D_scan._prog_func = prog
        self.jpe_3D_scan._finish_func = finish
        return self.jpe_3D_scan.run_async()

    def update_pzt_plot(self,sender,app_data,user_data):
        if "3D" in dpg.get_item_configuration("Piezo Scan_heat_series")['label']:
            log.debug("Updating 3D Scan Plot")
            func = self.get_reducing_func()
            plot_data = func(np.copy(np.flip(self.jpe_3D_scan.results,0)))
        elif "2D" in dpg.get_item_configuration("Piezo Scan_heat_series")['label']:
            plot_data = np.copy(np.flip(self.jpe_xy_scan.results,0))
        if self.tree["Plot/Deinterleave"]:
            if not self.tree["Plot/Reverse"]:
                plot_data[::2,:] = plot_data[1::2,:]
            else:
                plot_data[1::2,:] = plot_data[::2,:]
        self.plot.autoscale = self.tree["Plot/Autoscale"]
        self.plot.nbin = self.tree["Plot/N Bins"]
        self.plot.update_plot(plot_data)
        if sender == "cav_count_cut":
            volts = dpg.get_value("cav_count_cut")
            index = np.argmin(np.abs(self.jpe_cav_scan.positions[0]-volts))
            self.tree['Plot/3D/Slice Index'] = int(index)
        if sender == f"{self.treefix}_Plot/3D/Slice Index":
            index = app_data
            volts = self.jpe_cav_scan.positions[0][index]
            dpg.set_value("cav_count_cut",volts)

    def guess_cav_time(self):
        pts = self.tree["Scan/Cavity/Steps"]
        ctime = self.tree["Scan/Count Time (ms)"] + self.tree["Scan/Cavity/Wait Time (ms)"]
        scan_time = pts * ctime / 1000
        time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
        self.tree["Scan/Cavity/Estimated Time"] = time_string

    def guess_piezo_time(self):
        pts = self.tree["Scan/JPE/Steps"]
        ctime = self.tree["Scan/Count Time (ms)"] + self.tree["Scan/JPE/Wait Time (ms)"]
        scan_time = pts[0] * pts[1] * ctime / 1000
        time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
        self.tree["Scan/JPE/Estimated Time"] = time_string

    def guess_3d_time(self):
        jpe_pts = self.tree["Scan/JPE/Steps"]
        cav_pts = self.tree["Scan/Cavity/Steps"]
        total_jpe_pts = jpe_pts[0]*jpe_pts[1]
        total_cav_pts = (cav_pts - 1) * total_jpe_pts

        cav_time = self.tree["Scan/Count Time (ms)"] + self.tree["Scan/Cavity/Wait Time (ms)"]
        jpe_time = self.tree["Scan/Count Time (ms)"] + self.tree["Scan/JPE/Wait Time (ms)"]
        scan_time = (total_jpe_pts * jpe_time + total_cav_pts * cav_time) / 1000
        time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
        self.tree["Scan/Estimated Time"] = time_string

    def guess_pzt_times(self,*args):
        self.guess_piezo_time()
        self.guess_cav_time()
        self.guess_3d_time()

    def xy_pos_callback(self,sender,app_data,user_data):
        try:
            write = not dpg.get_value("count")
            self.set_jpe_xy(app_data[0],app_data[1],write=write)
        except FPGAValueError:
            pass

    def z_pos_callback(self,sender,app_data,user_data):
        try:
            write = not dpg.get_value("count")
            self.set_jpe_z(app_data,write=write)
        except FPGAValueError:
            pass

    def tilt_z_pos_callback(self,sender,app_data,user_data):
        try:
            write = not dpg.get_value("count")
            self.set_jpe_z_tilt(app_data,write=write)
        except FPGAValueError:
            pass

    def tilt_callback(self,sender, app_data, user_data):
        try:
            write = not dpg.get_value("count")
            self.set_jpe_xy(None,None,write=write)
        except FPGAValueError:
            pass

    def get_tilt(self,x,y):
        if self.tree["JPE/Tilt/Enable"]:
            x_tilt, y_tilt = self.tree["JPE/Tilt/XY Slope (V\V)"][:2]
            z_delta = x_tilt * x + y_tilt * y
            z_abs_lim = self.tree["JPE/Tilt/Abs. Limit"]
            if z_delta < -z_abs_lim:
                z_delta = -z_abs_lim
            if z_delta > z_abs_lim:
                z_delta = z_abs_lim
        else:
            z_delta = 0
        return z_delta

    def toggle_tilt(self,sender, app_data,user_data):
        if app_data:
            try:
                self.set_jpe_z()
                dpg.enable_item(f"{self.treefix}_JPE/Tilt/Z Offset")
                dpg.enable_item(f"{self.treefix}_JPE/Set Z Position")
            except FPGAValueError:
                # Changing tilt would put us out of bounds
                self.tree["JPE/Tilt/Enable"] = False
        else:
            try:
                self.set_jpe_z()
                dpg.disable_item(f"{self.treefix}_JPE/Tilt/Z Offset")
                dpg.disable_item(f"{self.treefix}_JPE/Set Z Position")
            except FPGAValueError:
                # Changing tilt would put us out of bounds
                self.tree["JPE/Tilt/Enable"] = True

    def set_jpe_xy(self,x=None,y=None,write=True):
        current_pos = self.fpga.get_jpe_pzs()
        if x is None:
            x = current_pos[0]
        if y is None:
            y = current_pos[1]
        z0 = self.tree["JPE/Z0 Position"]
        z_delta = self.get_tilt(x,y)
        
        if self.tree["JPE/Tilt/Enable"]:
            compensated_z = z0 + z_delta
        else:
            compensated_z = z0

        volts = self.pz_conv.zs_from_cart([x,y,compensated_z])
        in_bounds =self.pz_conv.check_bounds(x,y,compensated_z)
        if not in_bounds:
            self.tree["JPE/XY Position"] = current_pos[:2]
            raise FPGAValueError("Out of bounds")

        self.tree["JPE/Set Z Position"] = compensated_z
        self.tree["JPE/Tilt/Z Offset"] = z_delta
        self.tree["JPE/Z Volts"] = volts
        self.tree["JPE/XY Position"] = [x,y]
        self.fpga.set_jpe_pzs(x,y,compensated_z,write=write)
        self.plot.set_cursor([x,y])
        log.debug(f"Set JPE Position to ({x},{y},{compensated_z})")
        self.draw_bounds()
        
    def set_jpe_z(self,z=None,write=True):
        current_pos = self.fpga.get_jpe_pzs()
        x = current_pos[0]
        y = current_pos[1]
        z_delta = self.get_tilt(x,y)
        if z is None:
            z = current_pos[2] - z_delta
        if self.tree["JPE/Tilt/Enable"]:
            compensated_z = z + z_delta
        else:
            compensated_z = z
        volts = self.pz_conv.zs_from_cart([x,y,compensated_z])
        in_bounds = self.pz_conv.check_bounds(x,y,compensated_z)
        if not in_bounds:
            self.tree["JPE/Z0 Position"] = current_pos[2] - z_delta
            raise FPGAValueError("Out of bounds")

        self.tree["JPE/Z Volts"] = volts
        self.tree["JPE/Z0 Position"] = z
        self.tree["JPE/Set Z Position"] = compensated_z
        self.fpga.set_jpe_pzs(None,None,compensated_z,write=write)
        log.debug(f"Set JPE Z to {z}")
        self.draw_bounds()

    def set_jpe_z_tilt(self,z=None,write=True):
        current_pos = self.fpga.get_jpe_pzs()
        x = current_pos[0]
        y = current_pos[1]
        if z is None:
            z = self.tree["JPE/Set Z Position"]
        z_delta = self.get_tilt(x,y)
        if self.tree["JPE/Tilt/Enable"]:
            z0 = z - z_delta
        else:
            z0 = z
        volts = self.pz_conv.zs_from_cart([x,y,z])
        in_bounds = self.pz_conv.check_bounds(x,y,z)
        if not in_bounds:
            self.tree["JPE/Set Z Position"] = current_pos[2]
            raise FPGAValueError("Out of bounds")

        self.tree["JPE/Z Volts"] = volts
        self.tree["JPE/Z0 Position"] = z0
        self.tree["JPE/Set Z Position"] = z
        self.fpga.set_jpe_pzs(None,None,z,write=write)
        log.debug(f"Set JPE Tilted Z to {z}")
        self.draw_bounds()

    def set_cav_pos(self,z,write=True):
        self.fpga.set_cavity(z,write=write)
        self.tree["Cavity/Position"] = z
        log.debug(f"Set cavity to {z} V.")

    def man_set_cavity(self,sender,app_data,user_data):
        write = not dpg.get_value("count")
        self.set_cav_pos(app_data,write=write)

    def draw_bounds(self):
        zpos = self.fpga.get_jpe_pzs()[2]
        bound_points = list(self.pz_conv.bounds('z',zpos))
        bound_points.append(bound_points[0])
        if dpg.does_item_exist('pzt_bounds'):
            dpg.delete_item('pzt_bounds')
        dpg.draw_polygon(bound_points,tag='pzt_bounds',parent="Piezo Scan_plot")

    def xy_cursor_callback(self,sender,position):
        cur_xy = self.tree['JPE/XY Position'][:2]
        write = not dpg.get_value("count")
        try:
            self.set_jpe_xy(position[0],position[1],write=write)
        except FPGAValueError:
            self.plot.set_cursor(cur_xy)

    def save_xy_scan(self,*args):
        path = Path(dpg.get_value("save_dir"))
        filename = dpg.get_value("save_pzt_file")
        path /= filename
        as_npz = not (".csv" in filename)
        header = self.jpe_xy_scan.gen_header("XY Piezo Scan")
        self.jpe_xy_scan.save_results(str(path),as_npz=as_npz,header=header)

    def save_3d_scan(self,*args):
        path = Path(dpg.get_value("save_dir"))
        filename = dpg.get_value("save_pzt_file")
        path /= filename
        as_npz = not (".csv" in filename)
        header = self.jpe_3D_scan.gen_header("3D Piezo Scan")
        header += f"cavity center, {repr(self.jpe_cav_scan.centers)}\n"
        header += f"cavity spans, {repr(self.jpe_cav_scan.spans)}\n"
        header += f"cavity steps, {repr(self.jpe_cav_scan.steps)}\n"

        self.jpe_3D_scan.save_results(str(path),as_npz=as_npz,header=header)

    def save_cav_scan(self,*args):
        path = Path(dpg.get_value("save_dir"))
        filename = dpg.get_value("save_pzt_file")
        path /= filename
        as_npz = not (".csv" in filename)
        header = self.jpe_cav_scan.gen_header("Cavity Piezo Scan")
        self.jpe_cav_scan.save_results(str(path),as_npz=as_npz,header=header)

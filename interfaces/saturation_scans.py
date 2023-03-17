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

#TODO TODO TODO TODO
# Mostly working for now!
# Ideally we'd use the laser interface for more seemless control of AOM voltage.
#TODO TODO TODO TODO

dpg = rdpg.dpg
class SaturationInterface(Interface):
    
    def __init__(self,set_interfaces,fpga,counter,piezos,treefix="sat_tree"):
        self.fpga = fpga
        self.counter = counter
        self.set_interfaces = set_interfaces
        self.piezos = piezos
        self.treefix = treefix

        self.position_register = {}
        # TODO: Fill in when scanning function is defined
        self.sat_scanner = Scanner(self.step_aom_count_pd,
                                   [0],[10],[50],
                                   [],[],object,['Vaom'],default_result=np.array([-1,-1.0]))
        self.cav_scan = Scanner(self.set_cav_and_count,[0],[1],[50],[],[],float,['z'],default_result=-1)
        self.cav_sat_scanner = Scanner(self.step_aom_scan_cav,
                                       [0],[1],[50],
                                       [],[],object,['Vaom'],
                                       default_result=np.array([np.array([-1.0]),
                                                                np.array([-1.0,-1.0])],
                                                               dtype=object))

        self.controls = ['sat_scan','cav_sat_scan']
        self.params = []

        self.counts_data = np.array([0])
        self.pd_data = np.array([0])
        self.ao_data = np.array([0])

        self.gui_exists = False

    def initialize(self):
        if not self.gui_exists:
            raise RuntimeError("GUI must be made before initializaion")
        self.tree.load()
        self.guess_scan_times()

    def makeGUI(self, parent):
        with dpg.group(horizontal = True, parent=parent):
            dpg.add_checkbox(tag="sat_scan", label="Saturation Scan", callback=self.start_sat_scan)
            dpg.add_checkbox(tag="cav_sat_scan", label="Cavity Sat. Scan", callback=self.start_cav_sat_scan)
            dpg.add_text("Filename:")
            dpg.add_input_text(tag="save_sat_file", default_value="saturation.npz", width=200)
            dpg.add_text("Save:")
            dpg.add_button(label="Saturation", tag="save_sat_button",callback=self.save_sat_scan)
            dpg.add_button(label="Cavity Sat.", tag="save_cav_sat_button",callback=self.save_cav_sat_scan)
            dpg.add_checkbox(tag="sat_auto_save", label="Auto",default_value=False)

        with dpg.group(horizontal=True, width=0, parent=parent):
            with dpg.child_window(width=400, autosize_x=False, autosize_y=True,tag=f"{self.treefix}"):
                self.tree = rdpg.TreeDict(self.treefix,f"cryo_gui_settings/{self.treefix}_save.csv")
                self.tree.add("Scan/Count Time (ms)",5.0,callback=self.guess_scan_times,item_kwargs={'step':1})
                self.tree.add("Scan/Cavity/Center",0.0, item_kwargs={"step":0})
                self.tree.add("Scan/Cavity/Span",16.0, item_kwargs={"step":0})
                self.tree.add("Scan/Cavity/Steps",300,callback=self.guess_scan_times, item_kwargs={"step":0})
                self.tree.add("Scan/Cavity/Wait Time (ms)",1.0,callback=self.guess_scan_times,item_kwargs={'step':1})
                self.tree.add("Scan/Cavity/Estimated Time", "00:00:00", save=False,item_kwargs={'readonly':True})
                #TODO: Fix Parameter Ranges
                self.tree.add("Scan/AOM/Center",0.0, item_kwargs={"step":0})
                self.tree.add("Scan/AOM/Span",16.0, item_kwargs={"step":0})
                self.tree.add("Scan/AOM/Steps",300,callback=self.guess_scan_times, item_kwargs={"step":0})
                self.tree.add("Scan/AOM/Count Time (ms)",1.0,callback=self.guess_scan_times,item_kwargs={'step':1})
                self.tree.add("Scan/AOM/Wait Time (ms)",1.0,callback=self.guess_scan_times,item_kwargs={'step':1})
                self.tree.add("Scan/AOM/Estimated Time", "00:00:00", save=False,item_kwargs={'readonly':True})

                self.tree.add("Scan/Estimated Time", "00:00:00", save=False,item_kwargs={'readonly':True})

                self.tree.add("Plot/Autoscale",False,callback=self.update_plot)
                self.tree.add("Plot/Update Every Point",False)
                self.tree.add("Plot/Convert",False, callback=self.update_plot)
                self.tree.add("Plot/Conversion (mW\\V)",1.0, callback=self.update_plot, item_kwargs={"step":0},
                              tooltip="Set to Zero to keep Volts")

            with dpg.child_window(width=0,height=0,autosize_y=True):
                with dpg.group(width=0):
                    with dpg.child_window(width=-0,height=320):
                        with dpg.plot(label="Sat Scan",width=-1,height=300,tag="sat_plot"):
                            dpg.bind_font("plot_font") 
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="AOM (V)", tag="sat_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="Counts",tag="sat_y")
                            dpg.add_plot_axis(dpg.mvYAxis, label="PD (V)",tag="sat_pd_y")
                            dpg.add_line_series([0],
                                                [0],
                                                parent='sat_y',label='Sat Counts', tag='sat_curve')
                            dpg.add_line_series([0],
                                                [0],
                                                parent='sat_pd_y',label='PD Voltage', tag='sat_pd_curve')
                            dpg.add_plot_legend()

                    with dpg.child_window(width=-0,height=320):
                        with dpg.plot(label="Cavity Scan",width=-1,height=300,tag="cav_sat_plot"):
                            dpg.bind_font("plot_font") 
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="Fb. Piezo (V)", tag="cav_sat_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="Count Rate",tag="cav_sat_y")
                            dpg.add_line_series([0],
                                                [0],
                                                parent='cav_sat_y',label='counts', tag='cav_sat_counts')
                            dpg.add_plot_legend()
            
        self.gui_exists = True

    def step_aom_count_pd(self,aom):
        # Set AOM Voltage
        self.fpga.set_aoms(green=aom,write=True)
        # Get Counts from Counter
        counts = self.counter.get_count(self.tree["Scan/Count Time (ms)"])
        # Get AO Voltage
        pd = self.fpga.get_photodiode()
        return np.array([counts,pd])

    def start_sat_scan(self):
        if not dpg.get_value("sat_scan"):
            return -1
        self.counter.abort_counts()

        steps = self.tree["Scan/AOM/Steps"]
        centers = self.tree["Scan/AOM/Center"]
        spans = self.tree["Scan/AOM/Span"]
        self.sat_scanner.steps = [steps]
        self.sat_scanner.centers = [centers]
        self.sat_scanner.spans = [spans]

        def init():
            self.fpga.set_ao_wait(self.tree["Scan/Cavity/Wait Time (ms)"], write=False)
            pos = self.sat_scanner._get_positions()
            xmin = np.min(pos[0])
            xmax = np.max(pos[0])
            self.position_register["temp_aom_value"] = self.fpga.get_aoms()
            self.position_register["temp_dio_array"] = self.fpga.get_dio_array()
            temp_dio = self.fpga.get_dio_array()
            temp_dio[0] = 1
            self.fpga.set_dio_array(temp_dio,write=True)
            if self.tree['Plot/Autoscale']:
                dpg.set_axis_limits("sat_x", xmin, xmax)
            else:
                dpg.set_axis_limits_auto("sat_x")
            self.set_interfaces("saturation", False, "sat_scan")
        
        def abort(i,imax,idx,pos,res):
            return not dpg.get_value("sat_scan")

        def prog(i,imax,idx,pos,res):
            log.debug("Setting Progress Bar")
            dpg.set_value("pb",(i+1)/imax)
            dpg.configure_item("pb",overlay=f"Saturation Scan {i+1}/{imax}")
            log.debug("Updating Saturation Plot")
            if i > 1:
                res = np.stack(self.sat_scanner.results[:i])
                pos = self.sat_scanner.positions[0][:i]
                self.ao_data = np.array(pos)
                self.counts_data = np.array(res[:,0])
                self.pd_data = np.array(res[:,1])
                self.update_plot()

            if self.counter.tree["Counts/Plot Scan Counts"]:
                self.counter.plot_counts()

        def finish(results,completed):
            #Reenable controls first to avoid blocking.
            self.set_interfaces("saturation", True, "sat_scan")
            dpg.set_value("sat_scan",False)

            res = np.stack(self.sat_scanner.results)
            pos = self.sat_scanner.positions[0]
            self.ao_data = np.array(pos)
            self.counts_data = np.array(res[:,0])
            self.pd_data = np.array(res[:,1])
            self.update_plot()
            
            #Reset AOM to start
            self.fpga.set_aoms(*self.position_register["temp_aom_value"],write=False)
            self.fpga.set_dio_array(self.position_register["temp_dio_array"],write=True)
            # Autosave if desired
            if dpg.get_value("sat_auto_save"):
                self.save_sat_scan()
            # Return count time to default set by counter.
            self.fpga.set_ao_wait(self.counter.tree["Counts/Wait Time (ms)"],write=False)

        self.sat_scanner._init_func = init
        self.sat_scanner._abort_func = abort
        self.sat_scanner._prog_func = prog
        self.sat_scanner._finish_func = finish
        return self.sat_scanner.run_async()

    def set_cav_and_count(self,z):
        try:
            self.piezos.set_cav_pos(z,write=False)
        except FPGAValueError:
            return 0
        count = self.counter.get_count(self.tree["Scan/Count Time (ms)"])
        return count

    def step_aom_scan_cav(self,aom):
        # Set AOM Voltage
        self.fpga.set_aoms(green=aom,write=True)
        # Get AO Voltage At Start
        pd1 = self.fpga.get_photodiode()
        # Run Cavity Scan -  Wait for thread to finish with join
        # start = dt.datetime.now() # DEBUG TIMING
        self.do_cav_scan_step().join()
        # DEBUG TIMING
        # end = dt.datetime.now()
        # self.tock = end
        # log.debug(f"Cav Scan Took {(end-start).microseconds} us")
        # log.debug(f"Point Took {(self.tock-self.tick).microseconds} us" )
        # log.debug(f"Duty Cycle = {(end-start).microseconds/(self.tock-self.tick).microseconds * 100}")
        # self.tick = end
        # Get AO Voltage At End
        pd2 = self.fpga.get_photodiode()
        return np.array([self.cav_scan.results,np.array([pd1,pd2])],dtype=object)

    def prep_cav_scan_step(self):
        if not dpg.get_value("cav_sat_scan"):
            return [-1]

        def init():
            self.fpga.set_ao_wait(self.tree["Scan/Cavity/Wait Time (ms)"],write=False)
            log.debug("Starting cav scan sub.")
            pos = self.cav_scan._get_positions()
            xmin = np.min(pos[0])
            xmax = np.max(pos[0])
            self.position_register["temp_cav_position"] = self.fpga.get_cavity()
            if self.tree['Plot/Autoscale']:
                dpg.set_axis_limits("cav_sat_x",xmin,xmax)
            else:
                dpg.set_axis_limits_auto("cav_sat_x")
        
        def abort(i,imax,idx,pos,res):
            return not dpg.get_value("cav_sat_scan")

        def prog(i,imax,idx,pos,res):
            check = self.tree["Plot/Update Every Point"] or i + 1 >= imax
            if check:
                dpg.set_value("cav_sat_counts",[self.cav_scan.positions[0][:i],self.cav_scan.results[:i]])
                if self.tree['Plot/Autoscale']:
                    dpg.set_axis_limits_auto("cav_sat_y")
                    dpg.fit_axis_data("cav_sat_y")
                if self.counter.tree["Counts/Plot Scan Counts"]:
                    self.counter.plot_counts()

        def finish(results,completed):
            if self.tree['Plot/Autoscale']:
                dpg.set_axis_limits_auto("sat_y")
                dpg.fit_axis_data("sat_y")
            else:
                dpg.set_axis_limits_auto("sat_y")
            self.piezos.set_cav_pos(*self.position_register["temp_cav_position"])
            self.fpga.set_ao_wait(self.tree["Scan/AOM/Wait Time (ms)"],write=False)

        self.cav_scan._init_func = init
        self.cav_scan._abort_func = abort
        self.cav_scan._prog_func = prog
        self.cav_scan._finish_func = finish

    def do_cav_scan_step(self):
        return self.cav_scan.run_async()

    def start_cav_sat_scan(self):
        if not dpg.get_value("cav_sat_scan"):
            return -1
        self.counter.abort_counts()

        steps = self.tree["Scan/AOM/Steps"]
        centers = self.tree["Scan/AOM/Center"]
        spans = self.tree["Scan/AOM/Span"]
        self.cav_sat_scanner.steps = [steps]
        self.cav_sat_scanner.centers = [centers]
        self.cav_sat_scanner.spans = [spans]

        steps = self.tree["Scan/Cavity/Steps"]
        centers = self.tree["Scan/Cavity/Center"]
        spans = self.tree["Scan/Cavity/Span"]
        self.cav_scan.steps = [steps]
        self.cav_scan.centers = [centers]
        self.cav_scan.spans = [spans]

        self.prep_cav_scan_step()

        # #DEBUG TIMING
        # self.tick = dt.datetime.now()
        # self.tock = dt.datetime.now()

        def init():
            # Debug Timing
            # self.scan_start = dt.datetime.now()
            self.fpga.set_ao_wait(self.tree["Scan/Cavity/Wait Time (ms)"], write=False)
            pos = self.cav_sat_scanner._get_positions()
            self.position_register["temp_aom_value"] = self.fpga.get_aoms()
            self.position_register["temp_dio_array"] = self.fpga.get_dio_array()
            temp_dio = self.fpga.get_dio_array()
            temp_dio[0] = 1
            self.fpga.set_dio_array(temp_dio,write=True)
            self.set_interfaces("saturation", False, "cav_sat_scan")
        
        def abort(i,imax,idx,pos,res):
            return not dpg.get_value("cav_sat_scan")

        def prog(i,imax,idx,pos,res):
            log.debug("Setting Progress Bar")
            dpg.set_value("pb",(i+1)/imax)
            dpg.configure_item("pb",overlay=f"Cavity Saturation Scan {i+1}/{imax}")
            log.debug("Updating Saturation Plot")
            if i > 1:
                res = np.stack(self.cav_sat_scanner.results[:i])
                maxs = [np.max(res[j][0]) for j in range(i)]
                avg_pd = [np.mean(res[j][1]) for j in range(i)]
                pos = self.cav_sat_scanner.positions[0][:i]
                self.ao_data = np.atleast_1d(pos)
                self.pd_data = np.atleast_1d(avg_pd)
                self.counts_data = np.atleast_1d(maxs)

                self.update_plot()
            if self.counter.tree["Counts/Plot Scan Counts"]:
                self.counter.plot_counts()

        def finish(results,completed):
            res = np.stack(self.cav_sat_scanner.results)
            maxs = [np.max(res[j][0]) for j in range(len(res))]
            avg_pd = [np.mean(res[j][1]) for j in range(len(res))]
            pos = self.cav_sat_scanner.positions[0]
            self.ao_data = np.atleast_1d(pos)
            self.pd_data = np.atleast_1d(avg_pd)
            self.counts_data = np.atleast_1d(maxs)
            self.update_plot()
            #Reenable controls first to avoid blocking.
            self.set_interfaces("saturation", True, "cav_sat_scan")
            dpg.set_value("cav_sat_scan",False)
            if self.tree['Plot/Autoscale']:
                dpg.set_axis_limits_auto("sat_y")
                dpg.fit_axis_data("sat_y")
                dpg.set_axis_limits_auto("sat_pd_y")
                dpg.fit_axis_data("sat_pd_y")
            else:
                dpg.set_axis_limits_auto("sat_y")
                dpg.set_axis_limits_auto("sat_pd_y")
            #Reset AOM to start
            self.fpga.set_aoms(*self.position_register["temp_aom_value"],write=False)
            self.fpga.set_dio_array(self.position_register["temp_dio_array"],write=True)
            # Autosave if desired
            if dpg.get_value("sat_auto_save"):
                self.save_sat_scan()
            # Return count time to default set by counter.
            self.fpga.set_ao_wait(self.counter.tree["Counts/Wait Time (ms)"],write=False)

            # DEBUG TIMING
            # self.scan_finish = dt.datetime.now()
            # log.debug(f"Total Scan Time {(self.scan_finish-self.scan_start).seconds}")

        self.cav_sat_scanner._init_func = init
        self.cav_sat_scanner._abort_func = abort
        self.cav_sat_scanner._prog_func = prog
        self.cav_sat_scanner._finish_func = finish
        return self.cav_sat_scanner.run_async()

    def save_sat_scan(self):
        path = Path(dpg.get_value("save_dir"))
        filename = dpg.get_value("save_sat_file")
        path /= filename
        header = self.sat_scanner.gen_header("3D Piezo Scan")

        self.sat_scanner.save_results(str(path),as_npz=True,header=header)

    def save_cav_sat_scan(self):
        path = Path(dpg.get_value("save_dir"))
        filename = dpg.get_value("save_sat_file")
        path /= filename
        header = self.cav_sat_scanner.gen_header("3D Piezo Scan")
        header += f"cavity center, {repr(self.cav_scan.centers)}\n"
        header += f"cavity spans, {repr(self.cav_scan.spans)}\n"
        header += f"cavity steps, {repr(self.cav_scan.steps)}\n"

        self.cav_sat_scanner.save_results(str(path),as_npz=True,header=header)

    def update_plot(self,*args):
        if self.tree['Plot/Convert']:
            dpg.hide_item('sat_pd_y')
            conversion = self.tree['Plot/Conversion (mW\V)']
            if np.isclose(conversion,0.0):
                dpg.configure_item('sat_x',label="PD (V)")
                xdata = self.pd_data
            else:
                dpg.configure_item('sat_x',label="PD (~mW)")
                xdata = self.pd_data * conversion
        else:
            dpg.configure_item('sat_x',label='AOM (V)')
            xdata = self.ao_data
            dpg.show_item('sat_pd_y')
            dpg.set_value('sat_pd_curve',[xdata,self.pd_data])
        
        dpg.set_value('sat_curve', [xdata,self.counts_data])

        if self.tree['Plot/Autoscale']:
            dpg.set_axis_limits_auto("sat_y")
            dpg.fit_axis_data("sat_y")
            dpg.set_axis_limits_auto("sat_pd_y")
            dpg.fit_axis_data("sat_pd_y")
            dpg.set_axis_limits_auto("cav_sat_y")
            dpg.fit_axis_data("cav_sat_y")
            dpg.set_axis_limits("sat_x", min(xdata), max(xdata))
            dpg.fit_axis_data("sat_x")
        else:
            dpg.set_axis_limits_auto("sat_x")
            dpg.set_axis_limits_auto("sat_y")
            dpg.set_axis_limits_auto("sat_pd_y")
            dpg.set_axis_limits_auto("cav_sat_y")

    def guess_scan_times(self, *args):
        # Get Times
        ct = self.tree['Scan/Count Time (ms)']
        cwt = self.tree['Scan/Cavity/Wait Time (ms)']
        awt = self.tree['Scan/AOM/Wait Time (ms)']

        # Calculate Total Scan Times
        ctt = (ct + cwt) * self.tree['Scan/Cavity/Steps']/1000
        att = (ct + awt) * self.tree['Scan/AOM/Steps']/1000
        catt = (ctt + (ct + awt)) * self.tree['Scan/AOM/Steps']

        # Format string and set in interface
        time_string = str(dt.timedelta(seconds=ctt)).split(".")[0]
        self.tree["Scan/Cavity/Estimated Time"] = time_string
        time_string = str(dt.timedelta(seconds=att)).split(".")[0]
        self.tree["Scan/AOM/Estimated Time"] = time_string
        time_string = str(dt.timedelta(seconds=catt)).split(".")[0]
        self.tree["Scan/Estimated Time"] = time_string
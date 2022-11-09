from .interface_template import Interface
from apis.scanner import Scanner
from threading import Thread
from apis.picoharp import PicoHarp
from ana_scripts import lifetime_fitter
from apis import rdpg
from time import sleep
from datetime import datetime
from metrolopy import gummy
gummy.style = 'pm'
gummy.nsig = 1

from pathlib import Path

import datetime as dt
import logging
log = logging.getLogger(__name__)
import numpy as np

from logging import getLogger
log = getLogger(__name__)

dpg = rdpg.dpg

class PicoHarpInterface(Interface):
    def __init__(self,set_interfaces,harp,treefix="pico_tree"):
        super().__init__()
        self.set_interfaces = set_interfaces
        self.harp = harp
        self.treefix = treefix

        self.controls = []
        self.params = [f"{self.treefix}_Counting/Time",
                       f"{self.treefix}_Counting/Stop",
                       f"{self.treefix}_Counting/Stop At",
                       f"{self.treefix}_Counting/Divider",
                       f"{self.treefix}_Counting/Binning",
                       f"{self.treefix}_Advanced/CFD Zero",
                       f"{self.treefix}_Advanced/CFD Crossing",
                       f"{self.treefix}_Advanced/Sync Offset"]

        # Hardcoded, probably dumb
        irf_edges,irf_counts = lifetime_fitter.import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2022-04-07_sample_fluro_scans\lifetime\IRF_offset.dat")
        # After changing offset
        irf_times = lifetime_fitter.edges_to_centers(irf_edges) - 22
        irf_times,irf_counts = lifetime_fitter.strip_zeros(irf_times,irf_counts)
        self._irf_times = irf_times
        self._irf_counts = irf_counts

        self.rate_data = {'time1' : [], 
                          'chn1_rate' : [], 
                          'time2' : [], 
                          'chn2_rate' : []}
        
        self.update_thread = None
    
    def initialize(self):
        if not self.gui_exists:
            raise RuntimeError("GUI must be made before initialization.")
        self.tree.load()

    def makeGUI(self,parent):
        self.parent = parent
        with dpg.group(parent=parent,horizontal=True):
                dpg.add_checkbox(tag="pico_count",label="Count", callback=self.start_count)
                dpg.add_text("Filename:")
                dpg.add_input_text(tag="save_pico_file", default_value="lifetime.npz", width=200)
                dpg.add_button(tag="save_pico_button", label="Save Histogram",callback=self.save_pico)
                dpg.add_checkbox(tag="auto_save_pico", label="Auto")
                dpg.add_button(tag="clear_pico_rate", label="Clear Rate", callback=self.clear_rates)

        with dpg.group(horizontal=True, width=0):
            with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag=self.treefix):
                self.tree = rdpg.TreeDict(f"{self.treefix}",f'cryo_gui_settings/{self.treefix}_save.csv')
                self.tree.add("Picoharp/Initialized",False,save=False,callback=self.toggle_init)
                self.tree.add("Picoharp/Update Rate", 1)

                self.tree.add("Counting/Time", 3600, callback=self.setup_pico,
                              item_kwargs={'min_value' : 1, 'max_value':360000000,
                              'min_clamped':True, 'max_clamped':True, 'step':0, 'on_enter':True})
                self.tree.add("Counting/Stop", True, callback=self.setup_pico)
                self.tree.add("Counting/Stop At", 100, callback=self.setup_pico,
                              item_kwargs={'min_value' : 1, 'max_value':65536,
                              'min_clamped':True, 'max_clamped':True, 'step':0, 'on_enter':True})
                self.tree.add("Counting/Divider", 4, callback=self.setup_pico,
                              item_kwargs={'min_value' : 1, 'max_value':8,
                              'min_clamped':True, 'max_clamped':True, 'step':0, 'on_enter':True})
                self.tree.add("Counting/Binning", 0, callback=self.setup_pico,
                              item_kwargs={'min_value' : 0, 'max_value':8,
                              'min_clamped':True, 'max_clamped':True, 'step':0, 'on_enter':True})

                self.tree.add("Fit/Enable",True, callback=self.set_fit)
                self.tree.add("Fit/Max Fun Eval", 20, callback=self.set_fit,
                              item_kwargs = {'min_value' : 1, 'min_clamped' : True, 'step':0})
                self.tree.add("Fit/Min Bin", 10, callback=self.set_fit,
                              item_kwargs = {'min_value' : 1, 'min_clamped' : True, 'step':0})

                self.tree.add("Fit/Tau", "None", save=False, item_kwargs={'readonly' : True})
                self.tree.add("Fit/Amp", "None", save=False, item_kwargs={'readonly' : True})
                self.tree.add("Fit/Offset", "None", save=False, item_kwargs={'readonly' : True})
                self.tree.add("Fit/Chi2", "None", save=False, item_kwargs={'readonly' : True})


                self.tree.add("Advanced/CFD Zero", [0,0], callback=self.setup_pico,
                              item_kwargs={'min_value' : 0, 'max_value':20,
                              'min_clamped':True, 'max_clamped':True,
                              'on_enter':True})
                self.tree.add("Advanced/CFD Crossing", [160,120], callback=self.setup_pico,
                              item_kwargs={'min_value' : 0, 'max_value':800,
                              'min_clamped':True, 'max_clamped':True,
                              'on_enter':True})
                self.tree.add("Advanced/Sync Offset", 23000, callback=self.setup_pico,
                              item_kwargs={'min_value' : -99999, 'max_value':99999,
                              'min_clamped':True, 'max_clamped':True, 'step':0,
                              'on_enter':True})

                self.tree.add("Plot/Channel 1", False, callback=self.set_rate_plot)
                self.tree.add("Plot/Channel 2", True, callback=self.set_rate_plot)
                self.tree.add("Plot/Max Points", 1000, callback=self.set_rate_plot)

            with dpg.group():
                with dpg.child_window(width=-0,height=470):
                    with dpg.plot(label="Count Histogram",width=-1,height=450,tag="pico_hist_plot"):
                        dpg.bind_font("plot_font")
                        # REQUIRED: create x and y axes
                        dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="hist_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="hist_y", log_scale=True)
                        dpg.add_scatter_series([],
                                            [],
                                            parent='hist_y',label='Update', tag='hist_series')
                        dpg.add_scatter_series([],
                                            [],
                                            parent='hist_y',label='Counts', tag='old_hist_series')
                        dpg.add_line_series([],
                                            [],
                                            parent='hist_y',label='fit', tag='fit_series')
                        dpg.bind_item_theme("hist_series","plot_theme_purple")
                        dpg.bind_item_theme("old_hist_series","plot_theme_blue")
                        dpg.bind_item_theme("fit_series","plot_theme_green")
                        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)

                with dpg.child_window(width=-0,height=300):
                    with dpg.plot(label="Count Rate",width=-1,height=280,tag="pico_count_plot"):
                        dpg.bind_font("plot_font") 
                        # REQUIRED: create x and y axes
                        dpg.add_plot_axis(dpg.mvXAxis, label="x", time=True, tag="rate_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="rate_y")
                        dpg.add_line_series(rdpg.offset_timezone(self.rate_data['time1']),
                                            self.rate_data['chn1_rate'],
                                            parent='rate_y',label='chn1', tag='chn1_rate_plot')
                        dpg.add_line_series(rdpg.offset_timezone(self.rate_data['time2']),
                                            self.rate_data['chn2_rate'],
                                            parent='rate_y',label='chn2', tag='chn2_rate_plot')
                        dpg.bind_item_theme("chn1_rate_plot","plot_theme_blue")
                        dpg.bind_item_theme("chn2_rate_plot","plot_theme_blue")
                        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
        self.gui_exists = True

    def toggle_init(self, sender, app_data, user_data):
        if app_data:
            self.harp.init_device()
            self.setup_pico(None,None,None)
            self.thread = Thread(target=self.count_update_callback)
            self.thread.start()
        else:
            self.harp.deinitialize()
            self.thread.join()

    def start_count(self, sender, app_data, user_data):
        if app_data:
            self.harp.stop_meas()
            self.harp.clear_hist_mem()
            self.harp.start_meas(tacq = self.tree["Counting/Time"] * 1000)
        else:
            self.harp.stop_meas()

    def set_fit(self, sender, app_data, user_data):
        self.fit_hist()

    def fit_hist(self):
        if self.tree['Fit/Enable'] and (np.max(self.harp.histogram) >= self.tree['Fit/Min Bin']):
            max_fev = self.tree['Fit/Max Fun Eval']
            time = self.harp.times/1000
            time,counts = lifetime_fitter.strip_zeros(time,self.harp.histogram)
            ct, cc, irft,ifrc,decay,irf = lifetime_fitter.fit_data(time,
                                                                   counts,
                                                                   self._irf_times,
                                                                   self._irf_counts,
                                                                   max_fev=max_fev,
                                                                   double=False)
            pparams = decay.params.valuesdict()
            new_cut_times = np.linspace(ct[0],ct[-1],4*len(ct))
            func_eval = lifetime_fitter.single_gexp(new_cut_times,
                                                    pparams['sigma'],
                                                    pparams['center'],
                                                    pparams['amp'],
                                                    pparams['lifetime'],
                                                    pparams['offset'])
            dpg.set_value('fit_series',[new_cut_times,func_eval])
            
            tau = gummy(pparams['lifetime'],decay.params['lifetime'].stderr)
            amp = gummy(pparams['amp'],decay.params['amp'].stderr)
            offset = gummy(pparams['offset'],decay.params['offset'].stderr)
            chi2 = gummy(decay.redchi,np.sqrt(2/decay.nfree))
            self.tau = tau
            self.amp = amp
            self.offset = offset
            self.chi2 = chi2
            self.tree['Fit/Tau'] = str(tau)
            self.tree['Fit/Amp'] = str(amp)
            self.tree['Fit/Offset'] = str(offset)
            self.tree['Fit/Chi2'] = str(chi2)
            self.fit_params = decay.params

            dpg.show_item('fit_series')
        else:
            dpg.hide_item('fit_series')
            self.tree['Fit/Tau'] = "None"
            self.tree['Fit/Amp'] = "None"
            self.tree['Fit/Offset'] = "None"
            self.tree['Fit/Chi2'] = "None"

    def setup_pico(self, sender, app_data, user_data):
        if not(self.harp.devidx is None):
            self.harp.binning = self.tree['Counting/Binning']
            self.harp.sync_offset = self.tree['Advanced/Sync Offset']
            self.harp.sync_div = self.tree['Counting/Divider']
            self.harp.zero_crossing = self.tree['Advanced/CFD Zero'][:2]
            self.harp.cfd_level = self.tree['Advanced/CFD Crossing'][:2]
            self.harp.stop = self.tree['Counting/Stop']
            self.harp.stop_count = self.tree['Counting/Stop At']
            self.harp.acq_time = self.tree['Counting/Time']

    def save_pico(self, sender, app_data, user_data):
        self.harp.save(Path(dpg.get_value('save_pico_file')))

    def set_rate_plot(self, sender, app_data, user_data):
        pass

    def count_update_callback(self):
        prev_count = False
        while self.tree['Picoharp/Initialized']:
            sleep(1/self.tree['Picoharp/Update Rate'])
            # Update Rate Plots
            if self.tree['Plot/Channel 1']:
                self.rate_data['time1'].append(datetime.now().timestamp())
                self.rate_data['chn1_rate'].append(self.harp.get_countrate(0))
                delta = len(self.rate_data['chn1_rate']) - self.tree["Plot/Max Points"]
                for _ in range(delta):
                    try:
                        self.rate_data['chn1_rate'].pop(0)
                        self.rate_data['time1'].pop(0)
                    except IndexError:
                        break
                dpg.set_value('chn1_rate_plot',
                              [rdpg.offset_timezone(self.rate_data['time1']),
                              self.rate_data['chn1_rate']])
                dpg.show_item('chn1_rate_plot')

            else:
                dpg.hide_item('chn1_rate_plot')
            if self.tree['Plot/Channel 2']:
                self.rate_data['time2'].append(datetime.now().timestamp())
                self.rate_data['chn2_rate'].append(self.harp.get_countrate(1))
                delta = len(self.rate_data['chn2_rate']) - self.tree["Plot/Max Points"]
                for _ in range(delta):
                    try:
                        self.rate_data['chn2_rate'].pop(0)
                        self.rate_data['time2'].pop(0)
                    except IndexError:
                        break
                dpg.set_value('chn2_rate_plot',
                              [rdpg.offset_timezone(self.rate_data['time2']),
                              self.rate_data['chn2_rate']])
                dpg.show_item('chn2_rate_plot')

            else:
                dpg.hide_item('chn2_rate_plot')
            # If taking data, show histogram updates
            if dpg.get_value('pico_count'):
                dpg.set_value('pico_count', not self.harp.ctc_status())
                self.harp.get_histogram()
                old_times, old_hist = self.harp.last_times/1000, self.harp.last_hist
                times, hist = self.harp.times/1000, self.harp.histogram
                updates = np.where(hist != old_hist)
                
                updated_times, updated_counts = _strip_trailing_zeros(times[updates],hist[updates])
                old_times, old_counts = _strip_trailing_zeros(np.copy(np.delete(times,updates)), np.copy(np.delete(hist,updates)))

                if len(updated_counts) > 0:
                    dpg.set_value('hist_series' ,[updated_times,updated_counts])
                else:
                    dpg.set_value('hist_series' ,[[],[]])

                if len(old_counts) > 0:
                    dpg.set_value('old_hist_series', [list(old_times),list(old_counts)])
                else:
                    dpg.set_value('old_hist_series' ,[[],[]])

                try:
                    self.fit_hist()
                except Exception as e:
                    print(e)

            if not dpg.get_value('pico_count') and prev_count:
                self.harp.get_histogram()
                times, hist = self.harp.times/1000, self.harp.histogram                
                times, counts = _strip_trailing_zeros(times,hist)

                if len(counts) > 0:
                    dpg.set_value('old_hist_series' ,[times,counts])
                else:
                    dpg.set_value('old_hist_series' ,[[],[]])

                dpg.set_value('hist_series' ,[[],[]])

                try:
                    self.fit_hist()
                except Exception as e:
                    print(e)

            prev_count = dpg.get_value('pico_count')
            
    def clear_rates(self):
        self.rate_data = {'time1' : [], 
                          'chn1_rate' : [], 
                          'time2' : [], 
                          'chn2_rate' : []}
        dpg.set_value('chn1_rate_plot',
                      [rdpg.offset_timezone(self.rate_data['time1']),
                      self.rate_data['chn1_rate']])
        dpg.set_value('chn2_rate_plot',
                      [rdpg.offset_timezone(self.rate_data['time2']),
                      self.rate_data['chn2_rate']])

def _strip_trailing_zeros(times,data):
    if len(data) == 0:
        return times, data
    if data[-1] > 0:
        return times,data
    index = np.argmax(np.flip(data) > 0)
    return times[:-index], data[:-index]
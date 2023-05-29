import numpy as np

from .interface_template import Interface

from datetime import datetime
from apis import rdpg
from typing import Union
from threading import Thread
from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from time import sleep
from pathlib import Path

from logging import getLogger
log = getLogger(__name__)

dpg = rdpg.dpg

class CounterInterface(Interface):

    def __init__(self,set_interfaces,fpga,treefix='count_tree'):
        super().__init__()
        self.set_interfaces = set_interfaces
        self.fpga = fpga
        self.treefix = treefix
        self.data = {'counts':[0],
                     'AI1' :[0],
                     'time':[datetime.now().timestamp()]}
        self.controls = ['count']

        params = ["Counts/Count Time (ms)", 
                  "Counts/Wait Time (ms)", 
                  "Counts/Max Points",
                  "Counts/Average Points",
                  "Counts/Plot Scan Counts",
                  "Counts/Show AI1"]
        self.plot_thread = None
        self.draw_thread = None
        self.params = [f"{self.treefix}_{param}" for param in params]

    def initialize(self):
        if not self.gui_exists:
            raise RuntimeError("GUI must be made before initialization.")
        self.tree.load()
        self.toggle_AI(None, self.tree["Counts/Show AI1"], None)

    def makeGUI(self, parent: Union[str, int]) -> None:
        self.parent = parent
        with dpg.group(parent=parent,horizontal=True):
                    dpg.add_text("Filename:")
                    dpg.add_input_text(tag="save_counts_file", default_value="counts.npz", width=200)
                    dpg.add_button(tag="save_counts", label="Save Counts",callback=self.save_counts)
        with dpg.group(horizontal=True,width=0):
            with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag=f"{self.treefix}"):
                self.tree = rdpg.TreeDict(f'{self.treefix}',f'cryo_gui_settings/{self.treefix}_save.csv')
                self.tree.add("Counts/Count Time (ms)", 10,
                               item_kwargs={'min_value':1,'min_clamped':True},
                               tooltip="How long the fpga acquires counts for.")
                self.tree.add("Counts/Wait Time (ms)", 1,
                              item_kwargs={'min_value':1,'min_clamped':True},
                              tooltip="How long the fpga waits before counting.")
                self.tree.add("Counts/Max Points", 100000,
                              item_kwargs={'on_enter':True,'min_value':1,'min_clamped':True,'step':100},
                              tooltip="How many plot points to display before cutting old ones.")
                self.tree.add("Counts/Average Points", 5, callback=self.plot_counts,
                              item_kwargs={'min_value':1,'min_clamped':True},
                              tooltip="Size of moving average window.")
                self.tree.add("Counts/Plot Scan Counts", True, 
                              callback=self.plot_counts,
                              tooltip="Wether to plot counts acquired during other scanning procedures.")
                self.tree.add("Counts/Show AI1", True, callback=self.toggle_AI)
            with dpg.child_window(width=-1,autosize_y=True):
                with dpg.plot(label="Count Rate",width=-1,height=-1,tag="count_plot",
                              use_local_time=True,use_ISO8601=True):
                    dpg.bind_font("plot_font") 
                    # REQUIRED: create x and y axes
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time", time=True, tag="count_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Counts",tag="count_y")
                    dpg.add_plot_axis(dpg.mvYAxis,label="Sync", tag="count_AI1",no_gridlines=True)
                    dpg.add_line_series(self.data['time'],
                                        self.data['counts'],
                                        parent='count_y',label='counts', tag='counts_series')
                    dpg.add_line_series(self.data['time'],
                                        self.data['counts'],
                                        parent='count_y',label='avg. counts', tag='avg_counts_series')
                    dpg.add_line_series(self.data['time'],
                                        self.data['AI1'],
                                        parent='count_AI1',label='AI1', tag='AI1_series')
                    dpg.add_plot_legend()
                    dpg.bind_item_theme("counts_series","plot_theme_blue")
                    dpg.bind_item_theme("avg_counts_series","avg_count_theme")
                    dpg.bind_item_theme("AI1_series","plot_theme_purple")
        self.gui_exists = True

    def get_count(self,time:float) -> float:
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
        try:
            count = self.fpga.just_count(time)
        except TimeoutError as e:
            self.fpga.write_register('Start FPGA 1', 0)
            raise e
        self.draw_count(count)
        log.debug(f"Got count rate of {count}.")
        if self.tree["Counts/Plot Scan Counts"] or dpg.get_value("count"):
            self.data['counts'].append(count)
            self.data['AI1'].append(self.fpga.get_photodiode())
            self.data['time'].append(datetime.now().timestamp())
        return count

    def toggle_AI(self,sender,user,app):
        if user:
            dpg.show_item("AI1_series")
            dpg.show_item("AI1_series2")
            dpg.show_item("AI1_series3")
        else:
            dpg.hide_item("AI1_series")
            dpg.hide_item("AI1_series2")
            dpg.hide_item("AI1_series3")

    # Encapsulated
    def clear_counts(self,*args):
        """
        Clear the stored counts data. Doesn't immediately update the plot. So
        data will persist there.
        """
        self.data['counts'] = []
        self.data['AI1'] = []
        self.data['time'] = []

    # Encapsulated
    def plot_counts(self,*args):
        if self.plot_thread is None or not self.plot_thread.is_alive():
            self.plot_thread = Thread(target=self.plot_counts_thread)
            self.plot_thread.start()

    def plot_counts_thread(self):
        """
        Update the count plots on various pages
        """
        # Truncate the count data down to the desired number of points
        if len(self.data['counts']) > self.tree['Counts/Max Points']:
            try:
                self.data['counts'] = self.data['counts'][-self.tree['Counts/Max Points']:]
                self.data['AI1'] = self.data['AI1'][-self.tree['Counts/Max Points']:]
                self.data['time'] = self.data['time'][-self.tree['Counts/Max Points']:]
            except IndexError:
                 pass

        # Average the time and counts data
        avg_time, avg_counts= average_counts(self.data['time'],
                                             self.data['counts'],
                                             min(len(self.data['time']),
                                             self.tree["Counts/Average Points"]))
        # Update all the copies of the count plots.
        dpg.set_value('counts_series',[self.data['time'],self.data['counts']])
        dpg.set_value('avg_counts_series',[list(avg_time),list(avg_counts)])
        dpg.set_value('AI1_series',[self.data['time'],self.data['AI1']])

    # Encapsulated
    def draw_count(self,val):
        if self.draw_thread is None or not self.draw_thread.is_alive():
            self.draw_thread = Thread(target=self.draw_count_thread, args=(val,))
            self.draw_thread.start()

    def draw_count_thread(self,val):
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

    # Encapsulated
    def abort_counts(self):
        """
        Stop the couting with a small sleep time to ensure that the last
        count opperation exits.
        This is to be run before any scan starts, making sure to stop the ongoing
        counting. 
        """
        if dpg.get_value("count"):
            dpg.set_value("count",False)
            sleep(self.tree["Counts/Count Time (ms)"]/1000)

    # Encapsulated
    def start_counts(self):
        """
        Function triggered when enabling counts.
        Starts a new thread that takes care of getting the count data and
        updating the plot.
        """
        # If counts are disabled, don't do anything, thread will handle 
        # stopping itself
        if not dpg.get_value('count'):
            return
            
        self.fpga.set_ao_wait(self.tree["Counts/Wait Time (ms)"],write=False)
        # Function for the new thread to run
        def count_func():
            # As long as we still want to count
            while dpg.get_value("count"):
                # Get the new count data
                count = self.get_count(self.tree["Counts/Count Time (ms)"])
                # Threaded plot updater
                self.plot_counts()

        # Start the thread
        count_thread = Thread(target=count_func)
        count_thread.start()

    # Encapsulated
    def save_counts(self,*args):
        """
        Save the count data to a file.
        Effectively only saves the plotted data.
        """
        path = Path(dpg.get_value("save_dir"))
        filename = dpg.get_value("save_counts_file")
        path /= filename
        if '.npz' in filename:
            np.savez(path,time=self.data['time'],
                        counts=self.data['counts'],
                        AI1=self.data['AI1'])
        else:
            with path.open('w') as f:
                f.write("Timestamp,Counts,AI1\n")
                for d in zip(self.data['time'],self.data['counts'],self.data['AI1']):
                        f.write(f"{d[0]},{d[1]},{d[2]}\n")

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
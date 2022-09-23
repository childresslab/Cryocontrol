from .interface_template import Interface
from .hist_plot import mvHistPlot
from apis.scanner import Scanner
from apis import rdpg

from pathlib import Path

import datetime as dt
import logging as log
import numpy as np

dpg = rdpg.dpg

class GalvoInterface(Interface):
    
    def __init__(self,set_interfaces,fpga,counter,treefix="galvo_tree"):
        super().__init__()
        self.set_interfaces = set_interfaces
        self.fpga=fpga
        self.treefix = treefix
        self.counter = counter
        self.position_register = {}

        self.controls = ["galvo_tree_Galvo/Position",
                         "galvo_scan"]

        self.params = ["galvo_tree_Scan/Centers (V)",
                       "galvo_tree_Scan/Spans (V)",
                       "galvo_tree_Scan/Points",
                       "galvo_tree_Scan/Count Time (ms)",
                       "galvo_tree_Scan/Wait Time (ms)"]

        self.plot = mvHistPlot("Galvo Plot",True,None,True,True,1000,0,300,50,'viridis',True,1,1E9,50,50)
        self.plot.cursor_callback = self.galvo_cursor_callback

        self.scanner = Scanner(self.galvo_set_count,[0,0],[1,1],[50,50],[],[],float,['y','x'],default_result=-1)

    def set_controls(self,state:bool) -> None:
        if state:
            for control in self.controls:
                log.debug(f"Enabling {control}")
                dpg.enable_item(control)
            self.plot.enable_cursor()
        else:
            for control in self.controls:
                log.debug(f"Disabling {control}")
                dpg.disable_item(control)
            self.plot.disable_cursor()

    def makeGUI(self,parent):
        self.parent = parent
        with dpg.group(parent=parent,horizontal=True):
                dpg.add_checkbox(tag="galvo_scan",label="Scan Galvo", callback=self.start_scan)
                dpg.add_button(tag="query_plot",label="Copy Scan Params",callback=self.get_galvo_range)
                dpg.add_text("Filename:")
                dpg.add_input_text(tag="save_galvo_file", default_value="scan.npz", width=200)
                dpg.add_button(tag="save_galvo_button", label="Save Scan",callback=self.save_galvo_scanner)
                dpg.add_checkbox(tag="auto_save_galvo", label="Auto")
        with dpg.group(horizontal=True, width=0):
            with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="galvo_tree"):
                self.tree = rdpg.TreeDict(f"{self.treefix}",f'cryo_gui_settings/{self.treefix}_save.csv')
                self.tree.add("Galvo/Position",[float(f) for f in self.fpga.get_galvo()],
                                item_kwargs={'min_value':-10,
                                            'max_value':10,
                                            "min_clamped":True,
                                            "max_clamped":True,
                                            "on_enter":True},
                                callback=self.man_set_galvo)
                self.tree.add("Plot/Autoscale",False,tooltip="Autoscale the plot at each update")
                self.tree.add("Plot/N Bins",50,item_kwargs={'min_value':1,
                                                                'max_value':1000},
                                tooltip="Number of bins in plot histogram")
                self.tree.add("Plot/Update Every Point",False,
                                tooltip="Update the plot at every position vs each line, slows down scans.")
                self.tree.add("Scan/Centers (V)",[0.0,0.0],item_kwargs={'min_value':-10,
                                                                            'max_value':10,
                                                                            "min_clamped":True,
                                                                            "max_clamped":True},
                                tooltip="Central position of scan in volts.")
                self.tree.add("Scan/Spans (V)", [1.0,1.0],item_kwargs={'min_value':-20,
                                                                        'max_value':20,
                                                                        "min_clamped":True,
                                                                        "max_clamped":True},
                                tooltip="Width of scan in volts.")
                self.tree.add("Scan/Points", [100,100],item_kwargs={'min_value':0},
                            callback=self.guess_galvo_time,
                            tooltip="Number of points along the scan axes.")
                self.tree.add("Scan/Count Time (ms)", 10.0, item_kwargs={'min_value':0,
                                                                            'max_value':1000,
                                                                            'step':1},
                            callback=self.guess_galvo_time,
                            tooltip="How long the fpga counts at each position.")
                self.tree.add("Scan/Wait Time (ms)", 1.0,item_kwargs={'min_value':0,
                                                                    'max_value':1000,
                                                                    "step":1},
                            callback=self.guess_galvo_time,
                            tooltip="How long the fpga waits before counting, after moving.")
                self.tree.add("Scan/Estimated Time", "00:00:00", item_kwargs={'readonly':True},
                                save=False,
                                tooltip="Rough guess of how long scan will take = (count_time + wait_time)*Npts")
            with dpg.group():
                self.plot.parent = dpg.last_item()
                self.plot.height = -330
                self.plot.scale_width = 335
                self.plot.make_gui()
                with dpg.child_window(width=-0,height=320):
                    with dpg.plot(label="Count Rate",width=-1,height=300,tag="count_plot2"):
                        dpg.bind_font("plot_font") 
                        # REQUIRED: create x and y axes
                        dpg.add_plot_axis(dpg.mvXAxis, label="x", time=True, tag="count_x2")
                        dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="count_y2")
                        dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="count_AI12",no_gridlines=True)
                        dpg.add_line_series(rdpg.offset_timezone(self.counter.data['time']),
                                            self.counter.data['counts'],
                                            parent='count_y2',label='counts', tag='counts_series2')
                        dpg.add_line_series(rdpg.offset_timezone(self.counter.data['time']),
                                            self.counter.data['counts'],
                                            parent='count_y2',label='avg. counts', tag='avg_counts_series2')
                        dpg.add_line_series(rdpg.offset_timezone(self.counter.data['time']),
                                            self.counter.data['AI1'],
                                            parent='count_AI12',label='AI1', tag='AI1_series2')
                        dpg.set_item_source('counts_series2','counts_series')
                        dpg.set_item_source('avg_counts_series2','avg_counts_series')
                        dpg.set_item_source('AI1_series2','AI1_series')
                        dpg.bind_item_theme("counts_series2","plot_theme_blue")
                        dpg.bind_item_theme("avg_counts_series2","avg_count_theme")
                        dpg.bind_item_theme("AI1_series2","plot_theme_purple")
                        dpg.add_plot_legend()

    def set_galvo(self,x:float,y:float,write:bool=True) -> None:
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
            galvo_pos = self.fpga.get_galvo()
            if x is None:
                x = galvo_pos[0]
            if y is None:
                y = galvo_pos[1]
        # Update the actual position
        self.fpga.set_galvo(x,y,write=write)
        # Update the cursor position
        self.plot.set_cursor([x,y])
        # Update the readout position
        self.tree["Galvo/Position"] = [x,y]

    def man_set_galvo(self,*args):
        """
        Callback for when the galvo position is manually updated in the GUI.
        Simply calls `set_galvo` with the new position.
        """
        pos = self.tree["Galvo/Position"]
        write = not dpg.get_value("count")
        self.set_galvo(pos[0],pos[1],write=write)

    def galvo_set_count(self,y:float,x:float) -> float:
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
        self.set_galvo(x,y,write=False)
        count = self.counter.get_count(self.tree["Scan/Count Time (ms)"])
        log.debug(f"Got count rate of {count}.")
        return count

    def start_scan(self,sender,app_data,user_data):
        """
        Callback function that starts the galvo scan. Taking care of updating
        all the scan parameters.

        Parameters
        ----------
        Called through DPG callback, no need...
        """
        if not dpg.get_value("galvo_scan"):
            return -1
        self.counter.abort_counts()
        steps = self.tree["Scan/Points"]
        self.scanner.steps = steps[1::-1]
        self.scanner.centers = self.tree["Scan/Centers (V)"][1::-1]
        self.scanner.spans = self.tree["Scan/Spans (V)"][1::-1]
        
        def init():
            """
            Initialization callback for the scanner.
            """
            # Set FPGA wait time
            self.fpga.set_ao_wait(self.tree["Scan/Wait Time (ms)"],write=False)
            # Get spans and steps of scan to update plot range
            pos = self.scanner._get_positions()
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            self.plot.set_size(int(self.scanner.steps[0]),int(self.scanner.steps[1]))
            self.plot.set_bounds(xmin,xmax,ymin,ymax)
            # Store the current position of the galvo for reseting later
            self.position_register["temp_galvo_position"] = self.fpga.get_galvo()
            self.set_interfaces("galvo",False)
        
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
            if self.tree["Plot/Update Every Point"]:
                check = True
            else:
                check = (not (i+1) % self.tree["Scan/Points"][0]) or (i+1)==imax
            if check:
                log.debug("Updating Galvo Scan Plot")
                # Format data for the plot
                plot_data = np.copy(np.flipud(self.scanner.results))
                self.plot.autoscale = self.tree["Plot/Autoscale"]
                self.plot.nbin = self.tree["Plot/N Bins"]
                self.plot.update_plot(plot_data)
                if self.counter.tree["Counts/Plot Scan Counts"]:
                    self.counter.plot_counts()

        def finish(results,completed):
            """
            Finish callback for the scanner.
            """
            self.set_interfaces("galvo",True)
            # Uncheck scan button, indicating that we're done
            dpg.set_value("galvo_scan",False)
            # Reset the galvo to it's position at the start of the scan
            self.set_galvo(*self.position_register["temp_galvo_position"],write=True)
            # If autosave set, save the scan data.
            if dpg.get_value("auto_save"):
                self.save_galvo_scanner()
            self.fpga.set_ao_wait(self.counter.tree["Counts/Wait Time (ms)"],write=False)


        # Setup the scanner
        self.scanner._init_func = init
        self.scanner._abort_func = abort
        self.scanner._prog_func = prog
        self.scanner._finish_func = finish
        # Run the scan in a new thread
        log.debug("Starting Galvo Scanner")
        self.scanner.run_async()

    def get_galvo_range(self,*args):
        """
        Callback for querying the scan range from the galvo plot. Or just 
        copying the galvo position as the center of the scan.
        """
        # If plot is queried, copy full scan area
        xmin,xmax,ymin,ymax = self.plot.query_plot()
        if xmin is not None:
            new_centers = [(xmin+xmax)/2, (ymin+ymax)/2]
            new_spans = [xmax-xmin, ymax-ymin]
            self.tree["Scan/Centers (V)"] = new_centers
            self.tree["Scan/Spans (V)"] = new_spans
        # Otherwise, make the center of the scan the current position
        else:
            self.tree["Scan/Centers (V)"] = self.tree["Galvo/Position"]

    def guess_galvo_time(self,*args):
        """
        Update the estimated scan time from the number of points and count times.
        """
        pts = self.tree["Scan/Points"]
        ctime = self.tree["Scan/Count Time (ms)"] + self.tree["Scan/Wait Time (ms)"]
        scan_time = pts[0] * pts[1] * ctime / 1000
        time_string = str(dt.timedelta(seconds=scan_time)).split(".")[0]
        self.tree["Scan/Estimated Time"] = time_string

    def galvo_cursor_callback(self,sender,point):
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
        write = not dpg.get_value("count")
        self.set_galvo(point[0],point[1],write=write)

    def save_galvo_scanner(self,*args):
        """
        Save the scan using the scanner save function. Saves to the filename
        set by the directory and filename set in the GUI.
        """
        path = Path(dpg.get_value("save_dir"))
        filename = dpg.get_value("save_galvo_file")
        path /= filename
        as_npz = not (".csv" in filename)
        header = self.scanner.gen_header("Galvo Scan")
        self.scanner.save_results(str(path),as_npz=as_npz,header=header)
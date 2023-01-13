from .interface_template import Interface
from apis.scanner import Scanner
from apis.fpga_base import FPGAValueError
from apis import rdpg

from logging import getLogger
log = getLogger(__name__)

import numpy as np
import lmfit as lm

from numpy.typing import NDArray
from threading import Thread

dpg = rdpg.dpg

###################
# Galvo Optimizer #
###################
class GalvoOptInterface(Interface):
    def __init__(self,set_interfaces,fpga,galvo,counter,treefix="galvo_opt_tree"):
        super().__init__()
        self.set_interfaces = set_interfaces
        self.fpga = fpga
        self.galvo=galvo
        self.treefix = treefix
        self.counter = counter
        self.position_register = {}

        self.controls = ["optimize"]

        self.params = [f"{self.treefix}_Optimizer/Count Time (ms)",
                       f"{self.treefix}_Optimizer/Wait Time (ms)",
                       f"{self.treefix}_Optimizer/Scan Points",
                       f"{self.treefix}_Optimizer/Scan Range (XY)",
                       f"{self.treefix}_Optimizer/Iterations"]

    def initialize(self):
        if not self.gui_exists:
            raise RuntimeError("GUI must be made before initialization.")
        self.tree.load()

    def makeGUI(self,parent):
        with dpg.group(horizontal=True,width=0, parent=parent):
            with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag=f"{self.treefix}"):
                self.tree = rdpg.TreeDict(self.treefix,f'cryo_gui_settings/{self.treefix}_save.csv')
                self.tree.add("Optimizer/Count Time (ms)", 10.0,item_kwargs={'min_value':1.0,'min_clamped':True,'step':1},
                            tooltip="How long the fpga counts at each position.")
                self.tree.add("Optimizer/Wait Time (ms)", 10.0,item_kwargs={'min_value':1.0,'min_clamped':True,'step':1},
                            tooltip="How long the fpga waits before counting after moving.")
                self.tree.add("Optimizer/Scan Points", 50,item_kwargs={'min_value':2,'min_clamped':True},
                            tooltip="Number of points to scan along each axis.")
                self.tree.add("Optimizer/Scan Range (XY)", 0.1,item_kwargs={'min_value':0.0,'min_clamped':True,'step':0},
                            tooltip="Size of scan along each axis in volts.")
                self.tree.add("Optimizer/Iterations", 1,item_kwargs={'min_value':1,'min_clamped':True},
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

        self.gui_exists = True

    def optim_scanner_func(self,axis='x'):
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
                y = self.position_register['temp_galvo_position'][1]
                self.galvo.set_galvo(x,y)
                count = self.counter.get_count(self.tree["Optimizer/Count Time (ms)"])
                log.debug(f"Got count rate of {count}.")
                return count
        elif axis == 'y':
            def optim_func(y):
                x = self.position_register['temp_galvo_position'][0]
                self.galvo.set_galvo(x,y)
                count = self.counter.get_count(self.tree["Optimizer/Count Time (ms)"])
                log.debug(f"Got count rate of {count}.")
                return count
        else:
            raise ValueError(f"Invalid Axis {axis}, must be either 'x' or 'y'.")
        return optim_func

    def fit_galvo_optim(self, position:NDArray[np.float64],counts:NDArray[np.float64]) -> lm.model.ModelResult:
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

    def optimize_galvo(self,*args):
        """
        Run multiple iterations of the optimization routine. To narrow in on the
        optimal position. Number of iterations set from GUI
        """
        def loop_optim():
            # Do the loop the desired number of times.
            for i in range(self.tree["Optimizer/Iterations"]):
                self.single_optimize_run().join()
        # Run this all in an additional thread to avoid freezing the UI
        optim_thread = Thread(target=loop_optim)
        optim_thread.start()

    def single_optimize_run(self):
        """
        Function for running the optimization scan.

        Returns
        -------
        Thread
            The thread on which the scan is being run.
        """
        # Save the initial galvo position, in case of aborted scan.
        self.position_register["temp_galvo_position"] = self.fpga.get_galvo()
        init_galvo_pos = self.position_register["temp_galvo_position"]
        # The scanner which will do the x scan
        galvo_scanner_x = Scanner(self.optim_scanner_func('x'),
                                [init_galvo_pos[0]],
                                [self.tree["Optimizer/Scan Range (XY)"]],
                                [self.tree["Optimizer/Scan Points"]],
                                output_dtype=float,
                                labels=["Galvo X"])
        # The scanner which will do the y scan
        galvo_scanner_y = Scanner(self.optim_scanner_func('y'),
                                [init_galvo_pos[1]],
                                [self.tree["Optimizer/Scan Range (XY)"]],
                                [self.tree["Optimizer/Scan Points"]],
                                output_dtype=float,
                                labels=["Galvo Y"])
        # The data for the optimizer to consider.
        optim_data = {}
        # Stop counting if that's currently happening.
        self.counter.abort_counts()
        # Setup the functions for the scanners.
        def init_x():
            """
            Prepares the fpga and plot for the scan.
            """
            self.fpga.set_ao_wait(self.tree["Optimizer/Wait Time (ms)"],write=False)
            self.position_register['temp_galvo_position'] = self.fpga.get_galvo()
            optim_data['counts'] = []
            optim_data['pos'] = []
            dpg.set_value('optim_x_counts',[[0],[0]])
            dpg.set_value('optim_y_counts',[[0],[0]])
            dpg.set_value('optim_x_fit',[[],[]])
            dpg.set_value('optim_y_fit',[[],[]])
            self.set_interfaces("galvo_opt",False)

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
            if self.counter.tree["Counts/Plot Scan Counts"]:
                        self.counter.plot_counts()

        def finish_x(results,completed):
            """
            Once the scan is completed, we fit the data, and move the x position
            to the optimal position, then we trigger the y-scan to be run.
            """
            positions = galvo_scanner_x.positions[0]
            fit_x = self.fit_galvo_optim(positions,results)
            vals = fit_x.best_values
            # Get the peak x based on the quadratic fit results.
            optim = -vals['b']/(2*vals['a'])
            # Fix the ideal position to be within the bounds of the scan.
            optim = min(optim,np.max(positions))
            optim = max(optim,np.min(positions))
            # If the desired position is outside range, we just use the starting
            # position.
            try:
                self.galvo.set_galvo(optim,self.fpga.get_galvo()[1],write=True)
            except FPGAValueError:
                self.galvo.set_galvo(*self.position_register['temp_galvo_position'],write=True)
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
            self.position_register['temp_galvo_position'] = self.fpga.get_galvo()

        def prog_y(i,imax,idx,pos,res):
            """
            Update the plot while we take the scan.
            """
            dpg.set_value("pb",(i+1+imax)/(2*imax))
            dpg.configure_item("pb",overlay=f"Opt. Galvo (Y) {i+1+imax}/{2*imax}")
            optim_data['counts'].append(res)
            optim_data['pos'].append(pos[0])
            dpg.set_value('optim_y_counts',[optim_data['pos'],optim_data['counts']])
            if self.counter.tree["Counts/Plot Scan Counts"]:
                self.counter.plot_counts()

        def finish_y(results,completed):
            """
            Do the fit of the data and set the new position. Limiting to scan
            and galvo ranges.
            """
            #Re-enable Controls, make sure this is at the start to avoid errors
            #Blocking the controls
            self.set_interfaces("galvo_opt",True)

            positions = galvo_scanner_y.positions[0]
            fit_y = self.fit_galvo_optim(positions,results)
            vals = fit_y.best_values
            optim = -vals['b']/(2*vals['a'])
            optim = min(optim,np.max(positions))
            optim = max(optim,np.min(positions))
            try:
                self.galvo.set_galvo(self.fpga.get_galvo()[0],optim,write=True)
            except FPGAValueError:
                self.galvo.set_galvo(*self.position_register['temp_galvo_position'],write=True)
            # Plot the fit.
            new_axis = np.linspace(np.min(positions),np.max(positions),1000)
            fit_data = fit_y.eval(fit_y.params,x=new_axis)
            dpg.set_value('optim_y_fit',[new_axis,fit_data])
            self.fpga.set_ao_wait(self.counter.tree["Counts/Wait Time (ms)"],write=False)

        # Setup the y-scanner object.
        galvo_scanner_y._init_func = init_y
        galvo_scanner_y._prog_func = prog_y
        galvo_scanner_y._finish_func = finish_y

        # Run the x-scan asynchronously, which will then trigger the y-scan.
        return galvo_scanner_x.run_async()
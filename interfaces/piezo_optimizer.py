from .interface_template import Interface
from apis.scanner import Scanner
from apis import rdpg

import logging as log
import numpy as np
import lmfit as lm

from numpy.typing import NDArray
from threading import Thread

dpg = rdpg.dpg

class PiezoOptInterface(Interface):
    def __init__(self, set_interfaces,fpga,piezo,counter,treefix="self.tree"):
        super().__init__()
        self.fpga = fpga
        self.piezo = piezo
        self.counter = counter
        self.treefix=treefix
        self.set_interfaces = set_interfaces

        self.position_register = {}

        self.controls = ["pzt_optimize"]
        self.params = [f"{self.treefix}_XY/Count Time (ms)",
                       f"{self.treefix}_XY/Wait Time (ms)",
                       f"{self.treefix}_XY/Scan Points",
                       f"{self.treefix}_XY/Scan Range (XY)",
                       f"{self.treefix}_XY/Iterations",
                       f"{self.treefix}_Cav/Count Time (ms)",
                       f"{self.treefix}_Cav/Wait Time (ms)",
                       f"{self.treefix}_Cav/Scan Points",
                       f"{self.treefix}_Cav/Narrow Scan Range"]

    def initialize(self):
        if not self.gui_exists:
            raise RuntimeError("GUI must be made before initialization.")
        self.tree.load()

    def makeGUI(self,parent):
        self.gui_exists = True
        with dpg.group(horizontal=True,parent=parent):
                dpg.add_button(tag="optim_pzt_xy",label="Optimize JPE XY",callback=self.optimize_jpe)
                dpg.add_button(tag="optim_pzt_cav",label="Optimize Fiber",callback=self.optimize_cav_coarse)
        with dpg.group(horizontal=True,width=0,parent=parent):
            with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag=self.treefix):
                self.tree = rdpg.TreeDict(self.treefix,f'cryo_gui_settings/{self.treefix}_save.csv')
                self.tree.add("XY/Count Time (ms)", 50.0,item_kwargs={'min_value':1.0,'min_clamped':True,'step':1},
                                tooltip="How long the fpga counts at each position.")
                self.tree.add("XY/Wait Time (ms)", 20.0,item_kwargs={'min_value':1.0,'min_clamped':True,'step':1},
                                tooltip="How long the fpga waits before counting after moving.")
                self.tree.add("XY/Scan Points", 50,item_kwargs={'min_value':2,'min_clamped':True},
                                tooltip="Number of points to scan along each axis.")
                self.tree.add("XY/Scan Range (XY)", 2,item_kwargs={'min_value':0.0,'min_clamped':True,'step':0},
                                tooltip="Size of scan along each axis in volts.")
                self.tree.add("XY/Iterations", 2,item_kwargs={'min_value':1,'min_clamped':True},
                                tooltip="How many times to rerun the optimization around each new point.")

                self.tree.add("Cav/Count Time (ms)", 50.0,item_kwargs={'min_value':1.0,'min_clamped':True,'step':1},
                                tooltip="How long the fpga counts at each position.")
                self.tree.add("Cav/Wait Time (ms)", 5.0,item_kwargs={'min_value':1.0,'min_clamped':True,'step':1},
                                tooltip="How long the fpga waits before counting after moving.")
                self.tree.add("Cav/Scan Points", 300,item_kwargs={'min_value':2,'min_clamped':True},
                                tooltip="Number of points to scan along each axis.")
                self.tree.add("Cav/Wide Scan Range", 16,item_kwargs={'min_value':0.0,'min_clamped':True,'step':0},
                                tooltip="Size of scan along each axis in volts.")
                self.tree.add("Cav/Narrow Scan Range", 0.1,item_kwargs={'min_value':0.0,'min_clamped':True,'step':0},
                                tooltip="Size of scan along each axis in volts.")
                self.tree.add("Cav/Narrow Iterations", 1,item_kwargs={'min_value':1,'min_clamped':True,'step':0},
                                tooltip="Size of scan along each axis in volts.")

            with dpg.child_window(width=-1,autosize_y=True): 
                with dpg.group(width=-1):
                    with dpg.child_window(width=0,height=333):
                        with dpg.subplots(1,2,label="Optimizer",width=-1,height=-1,tag="pzt_optim_plot"):
                            with dpg.plot(label="X Scan", tag="pzt_optim_x"):
                                dpg.bind_font("plot_font") 
                                # REQUIRED: create x and y axes
                                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="pzt_optim_x_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="pzt_optim_x_y")
                                dpg.add_line_series([0],[0],
                                                    parent='pzt_optim_x_y',label='counts', tag='pzt_optim_x_counts')
                                dpg.add_line_series([0],[0],
                                                    parent='pzt_optim_x_y',label='fit', tag='pzt_optim_x_fit')
                                dpg.add_plot_legend()
                            with dpg.plot(label="Y Scan", tag="pzt_optim_y"):
                                dpg.bind_font("plot_font") 
                                # REQUIRED: create x and y axes
                                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="pzt_optim_y_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="pzt_optim_y_y")
                                dpg.add_line_series([0],[0],
                                                    parent='pzt_optim_y_y',label='counts', tag='pzt_optim_y_counts')
                                dpg.add_line_series([0],[0],
                                                    parent='pzt_optim_y_y',label='fit', tag='pzt_optim_y_fit')
                                dpg.add_plot_legend()

                    with dpg.child_window(width=0,height=333):
                        with dpg.plot(label="Cav Scan", tag="pzt_optim_cav", width=-1,height=-1):
                            dpg.bind_font("plot_font") 
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="pzt_optim_cav_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="pzt_optim_cav_y")
                            dpg.add_line_series([0],[0],
                                                parent='pzt_optim_cav_x',label='counts', tag='pzt_optim_cav_counts')
                            dpg.add_line_series([0],[0],
                                                parent='pzt_optim_cav_x',label='fit', tag='pzt_optim_cav_fit')
                            dpg.add_plot_legend()
        self.gui_exists = True

    def jpe_optim_scanner_func(self,axis='x'):
        """This function generates the function be called by the optimization scanner
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
                y = self.position_register['temp_jpe_position'][1]
                self.piezo.set_jpe_xy(x,y)
                count = self.counter.get_count(self.tree["XY/Count Time (ms)"])
                return count
        elif axis == 'y':
            def optim_func(y):
                x = self.position_register['temp_jpe_position'][0]
                self.piezo.set_jpe_xy(x,y)
                count = self.counter.get_count(self.tree["XY/Count Time (ms)"])
                return count
        else:
            raise ValueError(f"Invalid Axis {axis}, must be either 'x' or 'y'.")
        return optim_func

    def fit_jpe_optim(self,position:NDArray[np.float64],counts:NDArray[np.float64]) -> lm.model.ModelResult:
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
        model = lm.models.GaussianModel()
        params = model.guess(counts,x=position)
        model = lm.models.GaussianModel() + lm.models.ConstantModel()
        params['center'].set(min=np.min(position),max=np.max(position))
        params.add('c',value=np.min(counts))
        # Probably more annoying to do it right.
        weights = 1/np.sqrt(np.array([count if count > 0 else 1 for count in counts]))
        return model.fit(counts,params,x=position,weights=weights)

    def fit_cav_optim(self,position:NDArray[np.float64],counts:NDArray[np.float64]) -> lm.model.ModelResult:
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
        model = lm.models.LorentzianModel()
        params = model.guess(counts,x=position)
        model = lm.models.LorentzianModel() + lm.models.ConstantModel()
        params['center'].set(min=np.min(position),max=np.max(position))
        params.add('c',value=np.min(counts))
        # Probably more annoying to do it right.
        weights = 1/np.sqrt(np.array([count if count > 0 else 1 for count in counts]))
        return model.fit(counts,params,x=position,weights=weights)

    def optimize_jpe(self,*args):
        """
        Run multiple iterations of the optimization routine. To narrow in on the
        optimal position. Number of iterations set from GUI
        """
        def loop_optim():
            # Do the loop the desired number of times.
            for i in range(self.tree["XY/Iterations"]):
                self.single_jpe_run().join()
        # Run this all in an additional thread to avoid freezing the UI
        optim_thread = Thread(target=loop_optim)
        optim_thread.start()

    def single_jpe_run(self):
        """
        Function for running the optimization scan.

        Returns
        -------
        Thread
            The thread on which the scan is being run.
        """
        # Save the initial galvo position, in case of aborted scan.
        self.position_register["temp_jpe_position"] = self.fpga.get_jpe_pzs()
        init_jpe_pos = self.position_register["temp_jpe_position"]
        # The scanner which will do the x scan
        jpe_scanner_x = Scanner(self.jpe_optim_scanner_func('x'),
                                [init_jpe_pos[0]],
                                [self.tree["XY/Scan Range (XY)"]],
                                [self.tree["XY/Scan Points"]],
                                output_dtype=float,
                                labels=["JPE X"])
        # The scanner which will do the y scan
        jpe_scanner_y = Scanner(self.jpe_optim_scanner_func('y'),
                                [init_jpe_pos[1]],
                                [self.tree["XY/Scan Range (XY)"]],
                                [self.tree["XY/Scan Points"]],
                                output_dtype=float,
                                labels=["JPE Y"])
        # The data for the optimizer to consider.
        optim_data = {}
        # Stop counting if that's currently happening.
        self.counter.abort_counts()
        # Setup the functions for the scanners.
        def init_x():
            """
            Prepares the fpga and plot for the scan.
            """
            self.fpga.set_ao_wait(self.tree["XY/Wait Time (ms)"],write=False)
            self.position_register["temp_jpe_position"] = self.fpga.get_jpe_pzs()
            optim_data['counts'] = []
            optim_data['pos'] = []
            dpg.set_value('pzt_optim_x_counts',[[],[]])
            dpg.set_value('pzt_optim_y_counts',[[],[]])
            dpg.set_value('pzt_optim_x_fit',[[],[]])
            dpg.set_value('pzt_optim_y_fit',[[],[]])
            self.set_interfaces("pzt_opt",False)

        def prog_x(i,imax,idx,pos,res):
            """
            Update the x plot of the scan. Since we're doing few points, we can
            afford to plot them all as they come in.
            """
            # Set Progress Bar
            dpg.set_value("pb",(i+1)/(2*imax))
            dpg.configure_item("pb",overlay=f"Opt. JPE (X) {i+1}/{2*imax}")
            # Update the optimization data
            optim_data['counts'].append(res)
            optim_data['pos'].append(pos[0])
            # Update the plots
            dpg.set_value('pzt_optim_x_counts',[optim_data['pos'],optim_data['counts']])
            if self.counter.tree["Counts/Plot Scan Counts"]:
                        self.counter.plot_counts()

        def finish_x(results,completed):
            """
            Once the scan is completed, we fit the data, and move the x position
            to the optimal position, then we trigger the y-scan to be run.
            """
            positions = jpe_scanner_x.positions[0]
            fit_x = self.fit_jpe_optim(positions,results)
            vals = fit_x.best_values
            optim = vals['center']

            # If the desired position is outside range, we just use the starting
            # position.
            try:
                self.piezo.set_jpe_xy(optim,self.fpga.get_jpe_pzs()[1])
            except self.fpga.FPGAValueError:
                self.piezo.set_jpe_xy(*self.position_register['temp_jpe_position'][:2])
            # Plot the fit on the optimization plot
            new_axis = np.linspace(np.min(positions),np.max(positions),1000)
            fit_data = fit_x.eval(fit_x.params,x=new_axis)
            dpg.set_value('pzt_optim_x_fit',[new_axis,fit_data])
            # Start the y-scan.
            jpe_scanner_y.run_async().join()
            

        # Setup the scanner object.
        jpe_scanner_x._init_func = init_x
        jpe_scanner_x._prog_func = prog_x
        jpe_scanner_x._finish_func = finish_x

        def init_y():
            """
            Prepares for the second scan be clearing the data and updating
            the starting position.
            """
            optim_data['counts'] = []
            optim_data['pos'] = []
            self.position_register["temp_jpe_position"] = self.fpga.get_jpe_pzs()

        def prog_y(i,imax,idx,pos,res):
            """
            Update the plot while we take the scan.
            """
            dpg.set_value("pb",(i+1+imax)/(2*imax))
            dpg.configure_item("pb",overlay=f"Opt. JPE (Y) {i+1+imax}/{2*imax}")
            optim_data['counts'].append(res)
            optim_data['pos'].append(pos[0])
            dpg.set_value('pzt_optim_y_counts',[optim_data['pos'],optim_data['counts']])
            if self.counter.tree["Counts/Plot Scan Counts"]:
                self.counter.plot_counts()

        def finish_y(results,completed):
            """
            Do the fit of the data and set the new position. Limiting to scan
            and galvo ranges.
            """
            #Reenable controls first to avoid blocking.
            self.set_interfaces("pzt_opt",True)

            positions = jpe_scanner_y.positions[0]
            fit_y = self.fit_jpe_optim(positions,results)
            vals = fit_y.best_values
            optim = vals['center']
            try:
                self.piezo.set_jpe_xy(self.fpga.get_jpe_pzs()[0],optim)
            except self.fpga.FPGAValueError:
                self.piezo.set_jpe_xy(*self.position_register['temp_jpe_position'][:2])
            # Plot the fit.
            new_axis = np.linspace(np.min(positions),np.max(positions),1000)
            fit_data = fit_y.eval(fit_y.params,x=new_axis)
            dpg.set_value('pzt_optim_y_fit',[new_axis,fit_data])
            self.fpga.set_ao_wait(self.counter.tree["Counts/Wait Time (ms)"],write=False)

        # Setup the y-scanner object.
        jpe_scanner_y._init_func = init_y
        jpe_scanner_y._prog_func = prog_y
        jpe_scanner_y._finish_func = finish_y

        # Run the x-scan asynchronously, which will then trigger the y-scan.
        return jpe_scanner_x.run_async()

    def cav_optim_func(self,z):
        try:
            self.piezo.set_cav_pos(z,write=False)
        except self.fpga.FPGAValueError:
            return 0
        count = self.counter.get_count(self.tree["Cav/Count Time (ms)"])
        return count

    def optimize_cav_fine_loop(self):
        def loop_optim():
            # Do the loop the desired number of times.
            for i in range(self.tree["Cav/Narrow Iterations"]):
                self.optimize_cav_fine(i).join()
            #Reenable controls first to avoid blocking
            self.set_interfaces("pzt_opt",True)
        # Run this all in an additional thread to avoid freezing the UI
        optim_thread = Thread(target=loop_optim())
        optim_thread.start()

    def optimize_cav_fine(self,count=0):
        cav_scanner_fine = Scanner(self.cav_optim_func,
                                   self.fpga.get_cavity(),
                                   [self.tree["Cav/Narrow Scan Range"]],
                                   [self.tree["Cav/Scan Points"]],
                                   output_dtype=float,
                                   labels=["Cav"])
        optim_data = {}
        def init_cav_fine():
            """
            Prepares the fpga and plot for the scan.
            """
            self.position_register["temp_cav_position"] = self.fpga.get_cavity()
            optim_data['counts'] = []
            optim_data['pos'] = []
            dpg.set_value('pzt_optim_cav_counts',[[],[]])
            dpg.set_value('pzt_optim_cav_fit',[[],[]])

        def prog_cav_fine(i,imax,idx,pos,res):
            """
            Update the cav plot of the scan. Since we're doing few points, we can
            afford to plot them all as they come in.
            """
            # Set Progress Bar
            n = (1+self.tree['Cav/Narrow Iterations'])
            dpg.set_value("pb",(i+1 + (count+1)*imax)/(n*imax))
            dpg.configure_item("pb",overlay=f"Opt. Cav Fine {i+1 + (count+1)*imax}/{n*imax}")
            # Update the optimization data
            optim_data['counts'].append(res)
            optim_data['pos'].append(pos[0])
            # Update the plots
            dpg.set_value('pzt_optim_cav_counts',[optim_data['pos'],optim_data['counts']])
            if self.counter.tree["Counts/Plot Scan Counts"]:
                self.counter.plot_counts()
        
        def finish_cav_fine(results,completed):
            """
            Once the scan is completed, we fit the data, and move the cav position
            to the optimal position. Then we start a fine scan.
            """
            positions = cav_scanner_fine.positions[0]
            fit_results = self.fit_cav_optim(positions,results)
            vals = fit_results.best_values
            optim=vals['center']
            # If the desired position is outside range, we just use the starting
            # position.
            try:
                self.piezo.set_cav_pos(optim)
            except self.fpga.FPGAValueError:
                self.piezo.set_cav_pos(self.position_register['temp_cav_position'])
            except TimeoutError:
                self.piezo.set_cav_pos(optim,write=False)
            # Plot the fit.
            new_axis = np.linspace(np.min(positions),np.max(positions),1000)
            fit_data = fit_results.eval(fit_results.params,x=new_axis)
            dpg.set_value('pzt_optim_cav_fit',[new_axis,fit_data])
            self.fpga.set_ao_wait(self.counter.tree["Counts/Wait Time (ms)"],write=False)

        cav_scanner_fine._init_func = init_cav_fine
        cav_scanner_fine._prog_func = prog_cav_fine
        cav_scanner_fine._finish_func = finish_cav_fine

        return cav_scanner_fine.run_async()

    def optimize_cav_coarse(self):
        cav_scanner = Scanner(self.cav_optim_func,
                              self.fpga.get_cavity(),
                              [self.tree["Cav/Wide Scan Range"]],
                              [self.tree["Cav/Scan Points"]],
                              output_dtype=float,
                              labels=["Cav"])

        optim_data = {}
        # Setup the functions for the scanners.
        def init_cav():
            """
            Prepares the fpga and plot for the scan.
            """
            self.fpga.set_ao_wait(self.tree["Cav/Wait Time (ms)"],write=False)
            self.position_register["temp_cav_position"] = self.fpga.get_cavity()
            optim_data['counts'] = []
            optim_data['pos'] = []
            dpg.set_value('pzt_optim_cav_counts',[[],[]])
            dpg.set_value('pzt_optim_cav_fit',[[],[]])
            self.set_interfaces("pzt_opt",False)

        def prog_cav(i,imax,idx,pos,res):
            """
            Update the cav plot of the scan. Since we're doing few points, we can
            afford to plot them all as they come in.
            """
            # Set Progress Bar
            n = (1+self.tree['Cav/Narrow Iterations'])
            dpg.set_value("pb",(i+1)/(n*imax))
            dpg.configure_item("pb",overlay=f"Opt. Cav Wide {i+1}/{n*imax}")
            # Update the optimization data
            optim_data['counts'].append(res)
            optim_data['pos'].append(pos[0])
            # Update the plots
            dpg.set_value('pzt_optim_cav_counts',[optim_data['pos'],optim_data['counts']])
            if self.counter.tree["Counts/Plot Scan Counts"]:
                self.counter.plot_counts()

        def finish_cav(results,completed):
            """
            Once the scan is completed, we fit the data, and move the cav position
            to the optimal position. Then we start a fine scan.
            """
            positions = cav_scanner.positions[0]
            max_idx = np.argmax(results)
            optim = positions[max_idx]
            # Fix the ideal position to be within the bounds of the scan.
            optim = min(optim,np.max(positions))
            optim = max(optim,np.min(positions))
            # If the desired position is outside range, we just use the starting
            # position.
            try:
                self.piezo.set_cav_pos(optim)
            except self.fpga.FPGAValueError:
                self.piezo.set_cav_pos(self.position_register['temp_cav_position'])
            except TimeoutError:
                self.piezo.set_cav_pos(optim,write=False)
            self.optimize_cav_fine_loop()

                
        # Setup the scanner object.
        cav_scanner._init_func = init_cav
        cav_scanner._prog_func = prog_cav
        cav_scanner._finish_func = finish_cav

        cav_scanner.run_async()
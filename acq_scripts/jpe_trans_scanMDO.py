from ..apis import fpga_cryo as fc # Code for controlling cryo fpga stuff
import vipyr as vp # Library for easy oscilloscope interfacing github.com/ry-dgel/vipyr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import sleep

from functools import partial
from ..apis.scanner import Scanner
from pathlib import Path

# Set the save directory, and ensure it exists
SAVE_DIR = Path(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\trans_scan")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Setup FPGA
cryo = fc.CryoFPGA()
starting_pos = {} # For holding onto the data of where we start

# Setup scope
# Some manual setup is also required on the scope.
# Ensure that the signal you want is on channel 2, and that it's setup to trigger
# on the external sync signal provided by the signal generator.
# Also, make sure that there's a single period of the signal generator's sweep.
# With the beginning of the sweep being at the left edge of the screen.
#scope_name = "USB0::0x0699::0x03AF::C011023::INSTR" # This is the visa interface string name, shouldn't change for the same device
#scope_name = "USB0::0x0699::0x03B5::C010585::INSTR"
scope_name = 'USB0::0x0699::0x0456::C012257::INSTR' #MDO
scope = vp.VisaScope(vp.VisaInterface(scope_name)) # Open up a vipyr Visa Scope using the above name
scope.trigger.single = True # Setup the scope to single trigger mode, i.e. stop after a single acquisition

# Setup Sig Gen
# Some manual setup is also requred on the signal generator
# Namely, make sure that it's outputing an external sync for the scope,
# and set the amplitude, scan type and frequency appropriately 
#sig_name = 'USB0::0x0400::0x09C4::DG1G150300137::INSTR'
sig_name = 'USB0::0x1AB1::0x0642::DG1ZA232603172::INSTR' # New Sig Gen
# Bypass all of viyprs fancy stuff, and just open a normal pyvisa interface
sig = vp.resource_manager.open_resource(sig_name)
sig.query_delay = 0.1 # Seems like the RIGOL signal generator needs this delay

# Setup the scan, centers and spans are in volts
centers = [0,0]
spans = [15,30]
steps = [48,96]
labels = ["JPEY","JPEX"] # We're scanning in y slowly, and x quickly
output_type = float # At each point we get a number

# Setting up the function to be run at each point
# This first function tells the scope to start acquiring
# Since it's in single trigger mode, it'll stop once it's done
# Then, we return the value of the immediate measurement that we setup above.
# In this case, the maximum in the signal on channel 2.
def scope_get_trans_max():
    # Use the bare interface for quicker running.
    scope._interface.write(scope._commands['acq_start'])
    #Normalize to reflected power
    return np.max(scope.channels[1].get_waveform(get_conv=True))/np.mean(scope.channels[0].get_waveform(get_conv=True))
def scope_get_refl_min():
    # Use the bare interface for quicker running.
    scope._interface.write(scope._commands['acq_start'])
    return np.min(scope.channels[0].get_waveform(get_conv=True))

# This is the function we run at each setting
def jpe_xy_trans_scope(jpe_y,jpe_x):
    try:
        # If the x and y position are invalid, this function
        # raises a Value Error.
        # setting the z value to None makes it keep it's current setting
        # saying write=True makes the fpga immediately update the value.
        cryo.set_jpe_pzs(jpe_x,jpe_y,None, write=True)
    except fc.FPGAValueError:
        # In this case, we know it's outside the piezo scan range
        # so just return 0 instead of crashing
        return 0
    # Otherwise call the above function to get the max at this point.
    #return scope_get_refl_min()
    return scope_get_trans_max()

# Setup the scanner object to run the above function at every point,
# the [1] sets the second axis, in this case x to be snaked, alternating
# the scan direction for every y value.
cavity_trans_scan = Scanner(jpe_xy_trans_scope,
                         centers, spans, steps, [1], [], output_type,
                         labels=labels)

# To live plot, we need a buffer to hold all the data separate from the scan
# object
buffer = np.zeros(steps[::-1])
# We then setup the plot axes and object
Y,X = np.meshgrid(*cavity_trans_scan.positions)
fig,axes = plt.subplots(1,2)
# By naming this object, we can directly update the data later.
imobjf = axes[0].pcolormesh(X,Y,np.repeat(buffer[::2,:],2,axis=0),shading='auto')
imobjr = axes[1].pcolormesh(X,Y,np.repeat(buffer[1::2,:],2,axis=0),shading='auto')
plt.show(block=False)
# Function to be run once at the start of the scan
def init():
    # Show the plot
    # Get the FPGAs current position for multiple objects.
    starting_pos['jpe_pos'] = cryo.get_jpe_pzs()
    starting_pos['cavity_pos'] = cryo.get_cavity()
    starting_pos['galvo_pos'] = cryo.get_galvo()
    # Setup scope acquisition
    scope._interface.write(f"{scope._commands['data_start']} 1")
    scope._interface.write(f"{scope._commands['data_stop']} {scope.acq_points}")
    # Print some useful info
    print("Initial FPGA Positions:")
    print(f"\tJPE: {starting_pos['jpe_pos']}")
    print(f"\tGalvo: {starting_pos['galvo_pos']}")
    # Turn on the signal generator
    sig.write("output on")
    # Pause for a bit, both for the signal generator to turn on
    # and also for the plot to have time to render.
    plt.pause(0.5)

# Function to be run after acquiring every point
def progress(i,imax,index,pos,results):
    # Print out some useful info at every point to track progress
    print(f"{i+1}/{imax}, {pos} -> {results}")
    print(f"\tCryo pos: {cryo.get_jpe_pzs()}")
    # Save new results to buffer array for plotting
    buffer[index[::-1]] = results
    # Lets only plot every line by only running this every (# of steps in X) points.
    if not (i+1)%steps[1]:
        # Update the plot object directly, it takes a 1D array, which is what ravel gives.
        imobjf.set_array(np.repeat(buffer[::2,:],2,axis=0).ravel())
        imobjr.set_array(np.repeat(buffer[1::2,:],2,axis=0).ravel())
        # Update colorscale if needed
        imobjf.autoscale()
        imobjr.autoscale()
        # Force rendering of the plot
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

# Function to be run after the entire scan
def finish(results, completed):
    # Reset the jpe position to what it was at the start
    cryo.set_jpe_pzs(*starting_pos['jpe_pos'], write=True)
    # Turn off the signal generator
    sig.write("output off")
    plt.colorbar(imobjr,ax=axes[1])
    # Check if the scan was completed, if so just close objects
    # However I haven't implemented closing visa objects.
    if not completed:
        print("Something went wrong, I won't close devices")
    else:
        print("Scan succesful, I'll close devices")
        cryo.close_fpga()

# Tell the scanner object to use these functions.
cavity_trans_scan._init_func = init
cavity_trans_scan._prog_func = progress
cavity_trans_scan._finish_func = finish
# Run the scan
thread = cavity_trans_scan.run()
# Once done, save the results as a csv, with a header.
cavity_trans_scan.save_results(SAVE_DIR/'trans_scan_30x15.csv', as_npz=False, header=f"type: trans\ncenters: {centers}\nspans: {spans}\nsteps: {steps}")
cavity_trans_scan.save_results(SAVE_DIR/'trans_scan_30x15.npz', as_npz=True, header=f"type: trans\ncenters: {centers}\nspans: {spans}\nsteps: {steps}")

"""
cavity_trans_scan = Scanner(jpe_xy_trans_scope,
                         centers, spans, steps, [1], [], output_type,
                         labels=labels,
                         init = init, progress = progress, finish=finish)
results = cavity_trans_scan.run()
cavity_trans_scan.save_results(SAVE_DIR/'trans_scan.csv', as_npz=False, header=f"type: trans\ncenters: {centers}\nspans: {spans}\nsteps: {steps}")
"""
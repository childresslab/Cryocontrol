import fpga_cryo as fc # Code for controlling cryo fpga stuff
import vipyr as vp # Library for easy oscilloscope interfacing github.com/ry-dgel/vipyr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import sleep

from functools import partial
from scanner import Scanner
from pathlib import Path

# Set the save directory, and ensure it exists
SAVE_DIR = Path(r"X:\DiamondCloud\Cryostat setup\Data\2021_09_03_pos2_short")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Setup the scan, centers and spans are in volts
centers = [0,0]
spans = [45,50]
steps = [140,140]
labels = ["JPEY","JPEX"] # We're scanning in y slowly, and x quickly
output_type = float # At each point we get an array of two numbers
jpe_z = -3.25790175
x_bound = 1

# This is the function we run at each setting
def jpe_xy_trans_scope(jpe_y,jpe_x):
    pos = np.array([jpe_x,jpe_y,jpe_z])
    zs = fc.pz_conv.zs_from_cart(pos)
    bits = [fc.fb._volts_to_bits(z,10,16) for z in zs]
    rezs = [fc.fb._bits_to_volts(b,10,16) for b in bits]
    recarts = fc.pz_conv.cart_from_zs(rezs)
    inbounds = fc.pz_conv.check_bounds(jpe_x,jpe_y,jpe_z)
    #if np.abs(jpe_x) < x_bound:
        #print(f"Testing ({jpe_x},{jpe_y},{jpe_z})")
        #print(f"Zs ({rezs[0]},{rezs[1]},{rezs[2]})")
        #print(f"Result ({recarts[0]},{recarts[1]},{recarts[2]}) -> {inbounds}")
    if inbounds:
        return 1
    else:
        return -1

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
imobjf = axes[0].pcolormesh(X,Y,np.repeat(buffer[::2,:],2,axis=0).T,shading='auto')
imobjr = axes[1].pcolormesh(X,Y,np.repeat(buffer[1::2,:],2,axis=0).T,shading='auto')
plt.show(block=False)
# Function to be run once at the start of the scan
def init():
    plt.pause(0.5)

# Function to be run after acquiring every point
def progress(i,imax,index,pos,results):
    # Print out some useful info at every point to track progress
    print(f"{i+1}/{imax}, {pos} -> {results}")
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
    if not completed:
        print("Something went wrong, I won't close devices")
    else:
        print("Scan succesful, I'll close devices")

# Tell the scanner object to use these functions.
cavity_trans_scan._init_func = init
cavity_trans_scan._prog_func = progress
cavity_trans_scan._finish_func = finish
# Run the scan
thread = cavity_trans_scan.run()
# Once done, save the results as a csv, with a header.
#cavity_trans_scan.save_results(SAVE_DIR/'trans_scan.csv', as_npz=False, header=f"type: trans\ncenters: {centers}\nspans: {spans}\nsteps: {steps}")
#cavity_trans_scan.save_results(SAVE_DIR/'trans_scan.npz', as_npz=True, header=f"type: trans\ncenters: {centers}\nspans: {spans}\nsteps: {steps}")

"""
cavity_trans_scan = Scanner(jpe_xy_trans_scope,
                         centers, spans, steps, [1], [], output_type,
                         labels=labels,
                         init = init, progress = progress, finish=finish)
results = cavity_trans_scan.run()
cavity_trans_scan.save_results(SAVE_DIR/'trans_scan.csv', as_npz=False, header=f"type: trans\ncenters: {centers}\nspans: {spans}\nsteps: {steps}")
"""
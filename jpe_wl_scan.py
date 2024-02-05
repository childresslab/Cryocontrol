

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from pathlib import Path

from apis.scanner import Scanner
from apis import fpga_cryo as fc # Code for controlling cryo fpga stuff
from apis import spect as spect # Code for controlling andor spectrometer

# Set the save directory, and ensure it exists
SAVE_DIR = Path(r"X:\DiamondCloud\Cryostat setup\Data\2023-08-23_B11")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Initial creation of live update plot
def plot_data(wl, data):
    fig, axes = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios' : [0.3,0.7]})
    
    lobj = axes[0].plot(wl,np.average(data,axis=0))[0]
    imobj = plt.imshow(data,
                       extent=[min(wl),max(wl),min(rows),max(rows)], 
                       origin="lower",
                       aspect='auto')
    cbobj = plt.colorbar()
    plt.ion()
    plt.show(block=False)
    # Return plot objects for direct updating
    return [fig, lobj, imobj]
    
# Update the plot with most recent data
def update_data(fig, lobj, imobj, data):
    # Needs plot objects to be updated.
    lobj.set_ydata(np.average(data,axis=0))
    plt.gcf().axes[0].relim()
    plt.gcf().axes[0].autoscale_view(scalex=False)
    imobj.set_data(data)
    imobj.autoscale()
    # Force rendering of the plot.
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    
# Crop out the data that we don't care about.
# Since we only care about a certain wavelength range
# and a few rows of the spectrometer camera image.
def crop_data(wl,data,wlmin,wlmax,rows):
    idxs = np.where(np.logical_and(wlmin <= wl, wl <= wlmax))[0]
    return wl[idxs], data[np.ix_(rows,idxs)]

# Setup FPGA
# A connection to the fpga can be opened even while others exist
# However, once the fpga tries to move you'll get some weird
# error about a FIFO that can't be reserved/accessed
cryo = fc.CryoFPGA()
starting_pos = {}

# Setup Spectrometer
# 
andor = spect.Spectrometer()
# Set the exposure time
# TODO: Make sure this is set to what you want it to be!!
# It's useful to check in andor first
andor.exp_time = 7
#andor.api.SetExposureTime(andor._exp_time)

# Get the calibrated wavelength axis
wl = andor.get_wavelengths()
# Set the min and max wavelength bounds for cropping
wlmin = 500
wlmax = 800
# Set the binning of the spectrometer, play with this in
# andor to get an idea for what's best. I find 16 is ideal
# doesn't have to be a power of 2, but ideally should be.
andor.vertical_bin(16)
# Which rows of the image to keep
# If you change the binning, this definitely needs updating.
rows = [12,13,14]
# Cooldown the spectrometer and wait for it to reach
# The target temperature
# If it gets stuck at the target without continuing
# you can hit ctrl+C and it will stop waiting and continue the script.
# Make sure you don't hit ctrl+C more times than necessary as it will quit
# the entire script.
andor.start_cooling()
andor.waitfor_temp()
print("Setting Up Spectrometer Succeeded!")
# Making the initial plot with an initial acquisition.
print(f"Acquiring test spectrum should take {andor._exp_time}s")
data = np.array(andor.get_acq()) # Get Data
wlc,data = crop_data(wl,data,wlmin,wlmax,rows) # Crop Data
objs = plot_data(wlc,data) # Plot Data
plt.pause(2) # Allow time for plot to render.
print(f"Test Spectrum Succeeded, starting scan")

# To be run before the scan starts
def init():
    # Save initial positions and print for debug
    starting_pos['jpe_pos'] = cryo.get_jpe_pzs()
    starting_pos['cavity_pos'] = cryo.get_cavity()
    starting_pos['galvo_pos'] = cryo.get_galvo()
    print(f"{starting_pos['galvo_pos'] = }")
    print(f"{starting_pos['jpe_pos'] = }")
# Function to run at every point in scan
def jpe_z_spectra(jpe_z):
    # If we go to an invalid z, just return 0 and skip the point
    try:
        # Stay at current x,y position and move z value, then immediately
        # write the new position to fpga outputs
        cryo.set_jpe_pzs(None, None, jpe_z, write=True)
    except fc.FPGAValueError:
        return [[0]]
    # Start spectrum acquisition
    print("Acquiring...")
    data = np.array(andor.get_acq())
    # Crop the new data
    wlc, data = crop_data(wl, data, wlmin, wlmax, rows)
    return data

# Function to run at every point
def progress(i,imax,index,pos,results):
    # Print Useful info
    print(f"{i+1}/{imax}, {pos[0]}")
    # Update the plot objects with new data
    update_data(*objs, results)

# Function to run when scan is done
def finish(results, completed):
    # Reset jpe z position
    if not completed:
        print("Something went wrong, I'll keep the spectrometer cold and not close devices")
    else:
        print("Scan succesful, I'll turn off the spectrometer and close devices")
        cryo.set_jpe_pzs(*starting_pos['jpe_pos'], write=True)
        cryo.close_fpga()
        andor.stop_cooling()
        andor.close()
    
# Scan Parameters
centers = [-3.25]
spans = [-5.88] #negative span means start high and go low, in this case 0 -> -6
steps = [320]
labels = ["JPEZ"]
output_type = object # 2D images from spectrometer are objects

# Setup the scanner object
cavity_scan_wl = Scanner(jpe_z_spectra,
                         centers, spans, steps, [], [],output_type,
                         labels=labels,
                         #init = init, progress = progress, finish=finish)
                         init = init, progress = progress) # For when you want to also run reverse
# Run the scan
results = cavity_scan_wl.run()
# Save the scan, object type requires npz so set that
# save cropped wavelengths by putting it in header.
cavity_scan_wl.save_results(SAVE_DIR/'wl_fwd_scan_B11', as_npz=True, header=str(wlc.tolist()))

# To be run before the scan starts
def init():
    # Save initial positions and print for debug
    starting_pos['cavity_pos'] = cryo.get_cavity()
    starting_pos['galvo_pos'] = cryo.get_galvo()
    print(f"{starting_pos['galvo_pos'] = }")
# Can uncomment the following to run the scan in reverse as well
# Remove the finish function from the above cavity_scan_3D definition
# to make sure the devices don't get closed.
spans = [5.88]
cavity_scan_wl_rev = Scanner(jpe_z_spectra,
                             centers, spans, steps,[],[], output_type,
                             labels=labels,
                             init = init, progress = progress, finish = finish)
results = cavity_scan_wl_rev.run()
cavity_scan_wl_rev.save_results(SAVE_DIR/'wl_rev_scan_B11', as_npz=True, header=str(wlc.tolist()))
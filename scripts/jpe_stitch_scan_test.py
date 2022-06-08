import numpy as np
import jpe_steppers as jse
import fpga_cryo as fc
import vipyr as vp
from scanner import Scanner
import matplotlib.pyplot as plt
from time import sleep

from pathlib import Path

# Set the save directory, and ensure it exists
SAVE_DIR = Path(r"X:\DiamondCloud\Cryostat setup\Data\2022_01_13_stiched_test")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use(r"X:\DiamondCloud\Personal\Rigel\style_pub.mplstyle")


# # Setup scope
# # Some manual setup is also required on the scope.
# # Ensure that the signal you want is on channel 2, and that it's setup to trigger
# # on the external sync signal provided by the signal generator.
# # Also, make sure that there's a single period of the signal generator's sweep.
# # With the beginning of the sweep being at the left edge of the screen.
# scope_name = "USB0::0x0699::0x03B5::C010585::INSTR" # This is the visa interface string name, shouldn't change for the same device
# scope = vp.VisaScope(vp.VisaInterface(scope_name)) # Open up a vipyr Visa Scope using the above name
# scope.trigger.single = True # Setup the scope to single trigger mode, i.e. stop after a single acquisition
# scope.set_property("MEASU:IMMED:TYP", "MAXI") # Setup the type and source of oscilloscope measurement
# scope.set_property("MEASU:IMMED:SOURCE", "CH2")

# # Setup Sig Gen
# # Some manual setup is also requred on the signal generator
# # Namely, make sure that it's outputing an external sync for the scope,
# # and set the amplitude, scan type and frequency appropriately 
# sig_name = 'USB0::0x0400::0x09C4::DG1G150300137::INSTR'
# # Bypass all of viyprs fancy stuff, and just open a normal pyvisa interface
# sig = vp.resource_manager.open_resource(sig_name)
# sig.query_delay = 0.1 # Seems like the RIGOL signal generator needs this delay

cryo = fc.CryoFPGA()
stepper = jse.JPEStepper()
stepper.initialize()
# Setting very strict limits based on the current position,
# This should be adjusted for future runs.

initial_pz = cryo.get_jpe_pzs()
initial_st = stepper.position
stepper.lims = [[initial_st[0]-51,initial_st[0]+51],
               [initial_st[1]-51,initial_st[1]+51],
               [initial_st[2]-50,initial_st[2]]]

centers = [initial_st[1],initial_st[0],0,0]
spans = [100,100,7.6,7.6]
steps = [10,10,10,10]
# centers = [initial_st[1],initial_st[0]]
# spans = [100,100]
# steps = [10,10]


stepper.rel_lims = [max(abs(spans[0])/2,stepper.res[0]),
                    max(abs(spans[1])/2,stepper.res[1]),
                    0.0]

labels = ["JPESTY","JPESTX","JPEPZY","JPEPZX"] # We're scanning in y slowly, and x quickly
#labels = ["JPESTY","JPESTX"] # We're scanning in y slowly, and x quickly
output_type = float # At each point we get a number

starting_pos = {"piezo" : initial_pz, "stepper"  :initial_st}

# # Setting up the function to be run at each point
# # This first function tells the scope to start acquiring
# # Since it's in single trigger mode, it'll stop once it's done
# # Then, we return the value of the immediate measurement that we setup above.
# # In this case, the maximum in the signal on channel 2.
# def scope_get_max():
#     # Use the bare interface for quicker running.
#     scope._interface.write(scope._commands['acq_start'])
#     return scope.get_property("MEASU:IMMED:VAL")

# # This is the function we run at each setting
def jpe_st_pz_xy_move(jpe_sty,jpe_stx,jpe_pzy,jpe_pzx):
    try:
        # Setting the stepper position, here monitor makes it wait for the
        # move to finish before continuing, and also prints out the status of
        # the motion,then we save the position to the file.
        stepper.set_position(jpe_stx,jpe_sty,monitor=True,write_pos=True)
        # If the x and y position are invalid, this function
        # raises a Value Error.
        # setting the z value to None makes it keep it's current setting
        # saying write=True makes the fpga immediately update the value.
        cryo.set_jpe_pzs(jpe_pzx,jpe_pzy,None, write=True)
    except jse.JPEPositionError as e:
        # In this case, we know it's outside the piezo scan range
        # so just return 0 instead of crashing
        print(e)
        return 0
    except fc.FPGAValueError as e:
        # In this case, we know it's outside the piezo scan range
        # so just return 0 instead of crashing
        print(e)
        return 0
    # Otherwise call the above function to get the max at this point.
    return cryo.just_count(1)
# This is the function we run at each setting
# def jpe_st_xy_move(jpe_sty,jpe_stx):
#     try:
#         # Setting the stepper position, here monitor makes it wait for the
#         # move to finish before continuing, and also prints out the status of
#         # the motion,then we save the position to the file.
#         stepper.set_position(jpe_stx,jpe_sty,monitor=True,write_pos=True)
#         # If the x and y position are invalid, this function
#         # raises a Value Error.
#     except ValueError as e:
#         # In this case, we know it's outside the range
#         # so just return 0 instead of crashing
#         print(e)
#         return 0
#     # Otherwise call the above function to get the max at this point.
#     sleep(1)
#     return cryo.just_count(1)


stitch_scan = Scanner(jpe_st_pz_xy_move,
                     centers, spans, steps, [1,3],[],output_type,
                     labels=labels)
buffer = np.zeros((steps[1]*steps[3],steps[0]*steps[2]))
xs = np.linspace(initial_st[1]-spans[1]/2,initial_st[1]+spans[1]/2,steps[1]*steps[3])
ys = np.linspace(initial_st[0]-spans[0]/2,initial_st[0]+spans[0]/2,steps[0]*steps[2])
# stitch_scan = Scanner(jpe_st_xy_move,
#                      centers, spans, steps, [1],[],output_type,
#                      labels=labels)
# buffer = np.zeros((steps[1],steps[0]))
# xs = np.linspace(initial_st[1]-spans[1]/2,initial_st[1]+spans[1]/2,steps[1])
# ys = np.linspace(initial_st[0]-spans[0]/2,initial_st[0]+spans[0]/2,steps[0])
X,Y = np.meshgrid(xs,ys)
fig,ax = plt.subplots()
imobj = ax.pcolormesh(Y,X,buffer,shading='auto')
plt.show(block=False)

# Function to be run once at the start of the scan
def init():
    # Show the plot
    # Get the FPGAs current position for multiple objects.
    starting_pos['galvo'] = cryo.get_galvo()
    # Print some useful info
    print("Initial FPGA Positions:")
    print(f"\tJPE Piezos: {starting_pos['piezo']}")
    print(f"\tJPE Stepper: {starting_pos['stepper']}")
    print(f"\tGalvo: {starting_pos['galvo']}")
    # Turn on the signal generator
    # sig.write("output on")
    # Pause for a bit, both for the signal generator to turn on
    # and also for the plot to have time to render.
    plt.pause(0.5)

# Function to be run after acquiring every point
def progress(i,imax,index,pos,results):
    # Print out some useful info at every point to track progress
    print(f"{i+1}/{imax}, {pos} -> {results}")
    print(f"\tPiezo pos: {cryo.get_jpe_pzs()}")
    print(f"\tStepper pos: {stepper.position}")
    # Save new results to buffer array for plotting
    print(index)
    # buffer[index[1]*steps[3] + index[3], 
    #        index[0]*steps[2] + index[2]] = results
    buffer[index[1], index[0]] = results
    # Lets only plot every pz scan by only running this every (# of steps in pz) points.
    if not (i+1)%(1):
        # Update the plot object directly, it takes a 1D array, which is what ravel gives.
        imobj.set_array(buffer.ravel())
        # Update colorscale if needed
        imobj.autoscale()
        # Force rendering of the plot
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

# Function to be run after the entire scan
def finish(results, completed):
    # Reset the jpe position to what it was at the start
    cryo.set_jpe_pzs(*starting_pos['piezo'], write=True)
    stepper.set_position(*starting_pos['stepper'],monitor=True,write_pos=True)
    # Turn off the signal generator
    #sig.write("output off")
    plt.pause(0.5)
    #sig.write("output off")
    # Check if the scan was completed, if so just close objects
    # However I haven't implemented closing visa objects.
    if not completed:
        print("Something went wrong, I won't close devices")
    else:
        print("Scan succesful, I'll close devices")
        cryo.close_fpga()
        stepper.deinitialize()

def abort(I,Imax,index,position,result):
    return stepper.error_flag

# Tell the scanner object to use these functions.
stitch_scan._init_func = init
stitch_scan._prog_func = progress
stitch_scan._finish_func = finish
stitch_scan._abort_func = abort
# Run the scan
stitch_scan.run()
# Once done, save the results as a csv, with a header.
stitch_scan.save_results(SAVE_DIR/'trans_scan_big_no_bugs.csv', as_npz=False, header=f"type: trans\ncenters: {centers}\nspans: {spans}\nsteps: {steps}")
stitch_scan.save_results(SAVE_DIR/'trans_scan_big_no_bugs.npz', as_npz=True, header=f"type: trans\ncenters: {centers}\nspans: {spans}\nsteps: {steps}")



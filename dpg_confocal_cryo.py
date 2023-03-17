import dearpygui.dearpygui as dpg

from interfaces.counter import CounterInterface
from interfaces.galvo import GalvoInterface
from interfaces.galvo_optimizer import GalvoOptInterface
from interfaces.objective import ObjectiveInterface
from interfaces.piezos import PiezoInterface
from interfaces.piezo_optimizer import PiezoOptInterface
from interfaces.picoharp import PicoHarpInterface
from interfaces.lasers import LaserInterface
from interfaces.saturation_scans import SaturationInterface

from threading import Thread

from typing import Union

import apis.rdpg as rdpg
dpg = rdpg.dpg

import logging
import json

log = logging.getLogger(__name__)
with open("./app_config.json",'r') as f:
    config = json.load(f)

# Setup control, devices should be defined outside interfaces to allow interuse
if bool(config['dummy']):
    logging.basicConfig(level=logging.DEBUG)
    from apis.dummy.fpga_cryo_dummy import DummyCryoFPGA
    from apis.dummy.objective_dummy import DummyObjective
    from apis.picoharp import PicoHarp
    from apis.superk import SuperK
    from toptica.lasersdk.dlcpro.v2_5_2 import DLCpro, NetworkConnection

    # Setup Dummy Control for Simulations/Debugging
    log.warning("Using Dummy Controls")
    fpga = DummyCryoFPGA()
    objective = DummyObjective()
    harp = PicoHarp()
    superk = SuperK('COM4')
    # Toptica connections must absolutely be opened in the main thread
    # otherwise weird asynchronous stuff goes wrong.
    _conn = NetworkConnection('192.168.1.106')
    laser602 = DLCpro(_conn)
else:
    logging.basicConfig(level=logging.WARNING)
    from apis.fpga_cryo import CryoFPGA
    from apis.objective_control import Objective
    from apis.picoharp import PicoHarp
    from apis.superk import SuperK
    from toptica.lasersdk.dlcpro.v2_5_2 import DLCpro, NetworkConnection

    # Setup real control
    log.warning("Using Real Controls")
    fpga = CryoFPGA()
    objective = Objective()
    harp = PicoHarp()
    superk = SuperK('COM4')
    # Toptica connections must absolutely be opened in the main thread
    # otherwise weird asynchronous stuff goes wrong.
    _conn = NetworkConnection('192.168.1.106')
    laser602 = DLCpro(_conn)

devices = {'fpga':fpga, 'obj':objective, 'harp':harp, 'superk':superk, 'laser602':laser602}

# Dictionary to hold all loaded interfaces.
interfaces = {}

# Shared function for enabling and disabling interfaces.
def set_interfaces(caller:str,state:bool,control:Union[list[str],str]=None,ignore:Union[list[str],str]=None) -> None:
    """Loops through the loaded interfaces enabling or disabling their listed
    controls and paramters according to <state>.

    Parameters
    ----------
    caller : str
        The string name of the interface making the call. This sets the interface
        that will have its parameters disabled, as well as its controls.
    state : bool
        True to enable the controls/parameters false to disable them.
    control : Union[list[str],str], optional
        The string tag of a control whose state shouldn't be altered useful
        for skipping the control that is currently running so it can be disabled, 
        by default None
    ignore : Union[list[str],str], optinal
        An interface name or list of interface names to skip

    Raises
    ------
    ValueError
        If the calling interface is not a valid name.
    """
    if isinstance(ignore,str):
        ignore = [ignore]
    elif ignore is None:
        ignore = []
    if isinstance(control,str):
        control = [control]
    elif control is None:
        control = []
        
    log.debug(f"{caller} is setting interfaces {state}.")
    if caller not in interfaces.keys() and caller is not None:
        raise ValueError(f"{caller} not in dict of interfaces: {list(interfaces.keys())}")
    for name,interface in interfaces.items():
        if name not in ignore:
            if state:
                log.debug(f"Enabling {name} controls.")
            else:
                log.debug(f"Disabling {name} controls.")
            interface.set_controls(state,control)
    if caller is not None:
        interfaces[caller].set_params(state)

def toggle_fpga(sender,value,user_data):
    if value:
        fpga.open_fpga()
        set_interfaces(None,True,[],['obj'])
        dpg.enable_item('obj_scan')
    else:
        fpga.close_fpga()
        set_interfaces(None,False,[],['obj'])
        dpg.disable_item('obj_scan')


# Setup Counter Interface
counter = CounterInterface(set_interfaces,fpga)
interfaces['counter'] = counter

# Setup Galvo Interface
galvo = GalvoInterface(set_interfaces,fpga,counter)
interfaces['galvo'] = galvo

# Setup Galvo Optimizer Interface
galvo_opt = GalvoOptInterface(set_interfaces,fpga,galvo,counter)
interfaces["galvo_opt"] = galvo_opt

# Setup Objective Interface
obj = ObjectiveInterface(set_interfaces,objective,fpga,galvo,counter)
interfaces['obj'] = obj

# Setup Piezo Interface
pzt = PiezoInterface(set_interfaces,fpga,counter)
interfaces["pzt"] = pzt

# Setup Piezo Optimizer Interface
pzt_opt = PiezoOptInterface(set_interfaces,fpga,pzt,counter)
interfaces["pzt_opt"] = pzt_opt

# Setup Saturation Scans
saturation = SaturationInterface(set_interfaces,fpga,counter,pzt)
interfaces['saturation'] = saturation

# Setup Picoharp
pico = PicoHarpInterface(set_interfaces, harp)
interfaces['pico'] = pico

# Setup Lasers
lasers = LaserInterface(set_interfaces,fpga,superk,laser602)
interfaces['lasers'] = lasers

# Saving Scans
def choose_save_dir(*args):
    """ 
    Select the save directory using the DPG file selector.
    """
    chosen_dir = dpg.add_file_dialog(label="Chose Save Directory", 
                        default_path=dpg.get_value("save_dir"), 
                        directory_selector=True, modal=True,callback=set_save_dir)

def set_save_dir(sender,chosen_dir,user_data):
    """
    Callback to actually set the value from the chosen file in choose_save_dir.
    """
    dpg.set_value("save_dir",chosen_dir['file_path_name'])

################################################################################
############################### UI Building ####################################
################################################################################
# Initializiation of dearpygui and adding a theme for some averaged counts.add()
# New themes and fonts can be defined here.
rdpg.initialize_dpg("Cryocontrol",docking=False)
with dpg.theme(tag="avg_count_theme"):
    with dpg.theme_component(dpg.mvLineSeries):
        dpg.add_theme_color(dpg.mvPlotCol_Line, (252, 167, 130), category=dpg.mvThemeCat_Plots)
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 6, category=dpg.mvThemeCat_Plots)
###############
# Main Window #
###############
# Setup the main window
with dpg.window(label="Cryocontrol", tag='main_window'):
    ##################
    # Persistant Bar #
    ##################
    with dpg.group(horizontal=True):
        with dpg.child_window(width=-425,height=125):
            # Data Directory
            with dpg.group(horizontal=True):
                dpg.add_text("Data Directory:")
                dpg.add_input_text(default_value=config["data_folder"], tag="save_dir",width=-1)
                dpg.add_button(label="Pick Directory", callback=choose_save_dir)
            # Counts and Optimization
            with dpg.group(horizontal=True):
                dpg.add_checkbox(tag="count", label="Count", callback=counter.start_counts)
                dpg.add_button(tag="clear_counts", label="Clear Counts",callback=counter.clear_counts)
                dpg.add_checkbox(tag="FPGA_connected", default_value=True, label="FPGA", callback=toggle_fpga)
                dpg.add_button(tag="optimize", label="Optimize Galvo", callback=galvo_opt.optimize_galvo)
            # Progress Bar
            with dpg.group(horizontal=True):
                dpg.add_progress_bar(label="Scan Progress",tag='pb',width=-1)

        # Persistent Counts
        with dpg.child_window(width=425,height=125,no_scrollbar=True):
            dpg.add_text('0',tag="count_rate",parent="count_window")
            dpg.bind_item_font("count_rate","massive_font")

    ##############
    # START TABS #
    ##############
    # Each tab holds an interface which takes care of creating its own GUI.
    with dpg.tab_bar():
        # ###############
        # # COUNTER TAB #
        # ###############
        with dpg.tab(label="Counter"):
            counter.makeGUI(dpg.last_item())
                        
        #############
        # GALVO TAB #
        #############
        with dpg.tab(label="Galvo"):
            galvo.makeGUI(parent=dpg.last_item())

        #################
        # Optimizer Tab #
        #################
        with dpg.tab(label="Galvo Optimizer"):
            galvo_opt.makeGUI(parent=dpg.last_item())

        #################
        # Objective Tab #
        #################
        with dpg.tab(label="Objective Control"):
            obj.makeGUI(parent=dpg.last_item())
            
        #############
        # Piezo Tab #
        #############
        with dpg.tab(label="Piezo Control"):
            pzt.makeGUI(parent=dpg.last_item())
        #################
        # Optimizer Tab #
        #################
        with dpg.tab(label="Piezo Optimizer"):
            pzt_opt.makeGUI(dpg.last_item())
        
        with dpg.tab(label="Saturation"):
            saturation.makeGUI(dpg.last_item())

        with dpg.tab(label="Picoharp"):
            pico.makeGUI(dpg.last_item())
        
        with dpg.tab(label="Lasers"):
            lasers.makeGUI(dpg.last_item())

##################
# Initialization #
##################
# Initialize all interfaces, which requires that their GUI exists first.
for interface in interfaces.values():
    interface.initialize()

# Make the main window take up the whole page.
dpg.set_primary_window('main_window',True)

# Start the application.
# Running this in a seperate thread should be okay and allow for terminal control
# However, this may cause issues on macOS if you're trying to develope from there.
dpg_thread = Thread(target=rdpg.start_dpg)
dpg_thread.start()
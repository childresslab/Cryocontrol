import dearpygui.dearpygui as dpg
import logging as log

from apis.dummy.fpga_cryo_dummy import DummyCryoFPGA
from apis.dummy.objective_dummy import DummyObjective
from apis.picoharp import PicoHarp

from interfaces.counter import CounterInterface
from interfaces.galvo import GalvoInterface
from interfaces.galvo_optimizer import GalvoOptInterface
from interfaces.objective import ObjectiveInterface
from interfaces.piezos import PiezoInterface
from interfaces.piezo_optimizer import PiezoOptInterface
from interfaces.picoharp import PicoHarpInterface

import apis.rdpg as rdpg
dpg = rdpg.dpg

#TODO TODO TODO TODO TODO
# Add tooltips to everything!
# Remove uneeded plus/minus boxes
# Tune step value on plus/minus boxes
# Documentation
#TODO TODO TODO TODO TODO

log.basicConfig(format='%(levelname)s:%(message)s ', level=log.DEBUG)

# Setup control, devices should be defined outside interfaces to allow interuse
log.warning("Using Dummy Controls")
fpga = DummyCryoFPGA()
objective = DummyObjective()
harp = PicoHarp()
devices = {'fpga':fpga, 'obj':objective, 'harp':harp}

interfaces = {}

def set_interfaces(caller:str,state:bool) -> None:
    if caller not in interfaces.keys():
        raise ValueError(f"{caller} not in dict of interfaces: {list(interfaces.keys())}")
    for name,interface in interfaces.items():
        if state:
            log.debug(f"Enabling {name} controls.")
        else:
            log.debug(f"Disabling {name} controls.")
        interface.set_controls(state)
    interfaces[caller].set_params(state)

counter = CounterInterface(set_interfaces,fpga)
get_count = counter.get_count
abort_counts = counter.abort_counts
plot_counts = counter.plot_counts
start_counts = counter.start_counts
interfaces['counter'] = counter

galvo = GalvoInterface(set_interfaces,fpga,counter)
set_galvo = galvo.set_galvo
galvo_plot = galvo.plot
galvo_controls = galvo.controls
galvo_params = galvo.params
interfaces['galvo'] = galvo

# Setup Galvo Optimizer
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

# Setup Picoharp
pico = PicoHarpInterface(set_interfaces, harp)
interfaces['pico'] = pico

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
    Callback to actually set the value from the chosen file.
    """
    dpg.set_value("save_dir",chosen_dir['file_path_name'])

################################################################################
############################### UI Building ####################################
################################################################################
rdpg.initialize_dpg("Cryocontrol",docking=False)
with dpg.theme(tag="avg_count_theme"):
    with dpg.theme_component(dpg.mvLineSeries):
        dpg.add_theme_color(dpg.mvPlotCol_Line, (252, 167, 130), category=dpg.mvThemeCat_Plots)
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 6, category=dpg.mvThemeCat_Plots)
###############
# Main Window #
###############
with dpg.window(label="Cryocontrol", tag='main_window'):
    ##################
    # Persistant Bar #
    ##################
    with dpg.group(horizontal=True):
        with dpg.child_window(width=-425,height=125):
            # Data Directory
            with dpg.group(horizontal=True):
                dpg.add_text("Data Directory:")
                dpg.add_input_text(default_value="X:\\DiamondCloud\\Cryostat Setup", tag="save_dir",width=-1)
                dpg.add_button(label="Pick Directory", callback=choose_save_dir)
            # Counts and Optimization
            with dpg.group(horizontal=True):
                dpg.add_checkbox(tag="count", label="Count", callback=start_counts)
                dpg.add_button(tag="clear_counts", label="Clear Counts",callback=counter.clear_counts)
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

        with dpg.tab(label="Picoharp"):
            pico.makeGUI(dpg.last_item())

##################
# Initialization #
##################
# Initialize all interfaces
for interface in interfaces.items():
    interface.initialize()

dpg.set_primary_window('main_window',True)
rdpg.start_dpg()
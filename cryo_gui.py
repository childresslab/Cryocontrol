from typing import DefaultDict
import numpy as np
from time import sleep,time
from numpy.lib.histograms import histogram
from datetime import datetime
from pathlib import Path
from threading import Thread

import apis.cryo_remote as cr
import apis.rdpg as rdpg

dpg = rdpg.dpg


t_names = ["Platform", "Sample", "User", "Stage1", "Stage2"]
devices = {'cryo' : None}

c_thread = Thread()
wl_thread = Thread()

logs = {"pressures" : [],
        "temps"     : [],
        "times"   : []}

def choose_save_dir(*args):
    chosen_dir = dpg.add_file_dialog(label="Chose Save Directory", 
                        default_path=dpg.get_value('save_dir'), 
                        directory_selector=True, modal=True,callback=set_save_dir)

def set_save_dir(sender,chosen_dir,user_data):
    dpg.set_value('save_dir',chosen_dir['file_path_name'])

def start_logging(sender,app_data,user_data):
    if not dpg.get_value('logging'):
        return -1
    # Get cryo values
    def loop_cryo():
        while(dpg.get_value('logging')):
            sleep(wl_tree["Log/Cycle Time"])
            if not dpg.get_value('logging'):
                break
            update_cryo_props()
    
    # If we want cryo, start that
    if wl_tree['Cryostat/Connect']:
        t1 = Thread(target=loop_cryo)
        t1.start()

def update_cryo_props():
    pressure = devices['cryo'].get_pressure()/1000
    temp = np.array(devices['cryo'].get_temps())
    if len(logs['times']) == wl_tree["Log/Max Points"]:
        logs["pressures"].pop(0)
        logs["temps"].pop(0)
        logs["times"].pop(0)
    logs["pressures"].append(pressure)
    logs["temps"].append(temp)
    logs["times"].append(datetime.now().timestamp())
    dpg.set_value('P',[rdpg.offset_timezone(logs["times"]),logs["pressures"]])
    for i,name in enumerate(t_names):
        n = name.lower()
        log = [temp[i] for temp in logs["temps"]]
        dpg.set_value(n,[rdpg.offset_timezone(logs["times"]),log])

def save_log(*args):
    path = Path(dpg.get_value('save_dir'))
    filename = dpg.get_value('save_file')
    stem = filename.split('.')[0]
    path_c = path / (stem+"_cryo.csv")
    path_l = path / (stem+"_length.csv")

    if len(logs["times"]) > 0:
        path_c.touch()
        with path_c.open('r+') as f:
            f.write("Timestamp, Pressure (mbar)" + 
                    ''.join([", " + name + " (K)" for name in t_names]) + "\n")
            for i,time in enumerate(logs["times"]):
                f.write(f"{time}")
                f.write(f", {logs['pressures'][i]:.2e}")
                for temp in logs["temps"][i]:
                    f.write(f", {temp:.2f}")
                f.write("\n")
    if len(logs["l_times"]) > 0:
        path_l.touch()
        with path_l.open('r+') as f:
            f.write("Timestamp, Length (um)\n")
            for i,time in enumerate(logs["l_times"]):
                f.write(f"{time}")
                f.write(f", {logs['lengths'][i]:.2f}")
                f.write("\n")

def clear_log(*args):
    logs.update({"pressures" : [],
                 "temps"     : [],
                 "times"   : []})

def toggle_cryo(sender,value,user):
    if value:
        try:
            ip = wl_tree['Cryostat/IP']
            port = int(wl_tree['Cryostat/Port'])
            devices['cryo'] = cr.CryoComm(ip,port)
        except Exception as err:
                print("Failed to open connection to cryostation!")
                raise err
    else:
        del devices['cryo']
        devices['cryo'] = None

# START DPG STUFF HERE
rdpg.initialize_dpg("Cryostat")

# Begin Menu
with dpg.window(label="Cryo Log") as main_window:
    with dpg.group(horizontal=True):
        dpg.add_text("Data Directory:")
        dpg.add_input_text(default_value="X:\\DiamondCloud\\", tag="save_dir")
        dpg.add_button(label="Pick Directory", callback=choose_save_dir)
    # Begin Tabs
    with dpg.tab_bar() as main_tabs:
        # Begin Scanner Tab
        with dpg.tab(label="Cryo Log"):
            # Begin  
            with dpg.child_window(autosize_x=True,autosize_y=True):

                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Log",callback=start_logging, tag='logging')
                    dpg.add_button(label="Clear Log", callback=clear_log, tag='clear_log')

                with dpg.group(horizontal=True):
                    dpg.add_text("Filename:")
                    save_file = dpg.add_input_text(default_value="log.csv", width=200, tag='save_file')
                    save_button = dpg.add_button(label="Save",callback=save_log)
                    
                with dpg.group(horizontal=True, width=-0):
                    with dpg.child_window(width=400,autosize_y=True,autosize_x=False,tag='wl_tree'):
                        wl_tree = rdpg.TreeDict('wl_tree','cryolog_params.csv')
                        wl_tree.add("Cryostat/Connect",True,callback=toggle_cryo,save=False)
                        wl_tree.add("Cryostat/IP","192.168.1.103")
                        wl_tree.add("Cryostat/Port",7773)
                        toggle_cryo(None,True,None)

                        wl_tree.add("Log/Cycle Time",5.0)
                        wl_tree.add("Log/Max Points",1000000)

                    # create plot
                    with dpg.child_window(width=-0,autosize_y=True,autosize_x=True):
                        # Main Log Plots
                        with dpg.subplots(2,1,row_ratios=[2/3,1/3],width=-1,height=-1,link_all_x=True) as pt_subplot_id:
                            # Temperature Plot
                            with dpg.plot(no_title=True,width=-0,height=-0,tag="T_plot",
                                          anti_aliased=True,fit_button=0):
                                dpg.bind_font("plot_font")
                                dpg.add_plot_axis(dpg.mvXAxis, label=None,time=True, tag="T_t_axis")
                                dpg.add_plot_axis(dpg.mvYAxis, label="Tempearture (K)", tag="T_y_axis")
                                for name in t_names:
                                    dpg.add_line_series([datetime.now().timestamp()], [0], label=name, 
                                                            parent="T_y_axis",tag=name.lower())
                                dpg.set_axis_limits_auto('T_t_axis')
                                dpg.set_axis_limits_auto('T_y_axis')
                                dpg.add_plot_legend()
                            # Pressure Plot
                            with dpg.plot(no_title=True,width=-0,height=-0,tag="P_plot",
                                          anti_aliased=True,fit_button=0):
                                dpg.bind_font("plot_font")
                                dpg.add_plot_axis(dpg.mvXAxis, label=None,time=True, tag="P_t_axis")
                                dpg.add_plot_axis(dpg.mvYAxis, label="Pressure (mbar)",log_scale=True,
                                                  tag="P_y_axis")
                                dpg.set_axis_limits_auto('P_y_axis')
                                dpg.add_line_series([datetime.now().timestamp()], [0], label="P", 
                                                        parent=dpg.last_item(),tag='P')

dpg.set_primary_window(main_window, True)
rdpg.start_dpg()
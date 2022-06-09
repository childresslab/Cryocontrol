from typing import DefaultDict
import dearpygui.dearpygui as dpg
import numpy as np
from time import sleep,time
from numpy.lib.histograms import histogram
import datetime
from pathlib import Path

import apis.rdpg as rdpg
from apis.scanner import Scanner
from apis.fpga_cryo import CryoFPGA
from apis.jpe_coord_convert import JPECoord

dpg = rdpg.dpg

pz_config = {"vmax" : 0,
             "vmin" : -6.5,
             "vgain" : -20,
             "R": 6.75,
             "h": 45.1}
pz_conv = JPECoord(pz_config['R'], pz_config['h'],
                   pz_config['vmin'], pz_config['vmax'])
# Setup fpga control
#fpga = CryoFPGA()

def gen_point(delay = 1E-3):
    if delay >= 15E-3:
        sleep(delay)
    else:
        now = time()
        while (time() - now) < delay:
            sleep(0)
    return np.random.randn() * 100

def gen_line(n,delay=1E-3):
    return np.array([gen_point(delay) for _ in range(n)])

def gen_grid(n,m,delay=1E-3):
    return np.array([gen_line(m, delay) for _ in range(n)])

#def galvo(y,x):
#    fpga.set_galvo(x,y,write=False)
#    return fpga.just_count(dpg.get_value(ct))

def dummy_galvo(y,x):
    sleep(dpg.get_value("Scan/Count Time (ms)")*1E-3)
    return np.abs(x**2 + y + np.random.randn())

galvo_scan = Scanner(dummy_galvo,[0,0],[1,1],[50,50],[1],[],float,['y','x'])

def start_scan(sender,app_data,user_data):
    if not dpg.get_value('scan'):
        return -1
    steps = dpg.get_value("Scan/Points")
    galvo_scan.steps = steps[1::-1]
    galvo_scan.centers = dpg.get_value("Scan/Centers (V)")[1::-1]
    galvo_scan.spans = dpg.get_value("Scan/Spans (V)")[1::-1]
    
    def init():
        pos = galvo_scan._get_positions()
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        dpg.configure_item("heat_series",rows=steps[0],cols=steps[1],
                           bounds_min=(xmin,ymin),bounds_max=(xmax,ymax))
    
    def abort(i,imax,idx,pos,res):
        return not dpg.get_value('scan')

    def prog(i,imax,idx,pos,res):
            dpg.set_value("pb",(i+1)/imax)
            dpg.configure_item("pb",overlay=f"{i+1}/{imax}")
            if (not (i+1) % dpg.get_value("Scan/Points")[0]) or (i+1)==imax: 
                plot_data = np.copy(np.flip(galvo_scan.results,0))
                dpg.set_value("heat_series", [plot_data,[0.0,1.0],[],[],[]])
                if dpg.get_value("Plot/Autoscale"):
                    lower = np.min(plot_data)
                    upper = np.max(plot_data)
                    dpg.configure_item("colormap",min_scale=lower,max_scale=upper)
                    dpg.configure_item("heat_series",scale_min=lower,scale_max=upper)
                    dpg.set_value("line1",lower)
                    dpg.set_value("line2",upper) 
                    for ax in ["heat_x","heat_y","hist_x","hist_y"]:
                        dpg.fit_axis_data(ax)
                update_histogram(plot_data)

    def finish(results,completed):
        dpg.set_value('scan',False)
        if dpg.get_value("auto_save"):
            save_scan()

    galvo_scan._init_func = init
    galvo_scan._abort_func = abort
    galvo_scan._prog_func = prog
    galvo_scan._finish_func = finish
    galvo_scan.run_async()

def update_histogram(data,bin_width = 10):
    nbins = max([10,int(round((np.max(data)-np.min(data))/bin_width))])
    occ,edges = np.histogram(data,bins=nbins)
    xs = [0] + list(np.repeat(occ,2)) + [0,0] 
    ys = list(np.repeat(edges,2)) + [0]
    dpg.set_value("histogram",[xs,ys,[],[],[]])

def set_scale(sender,app_data,user_data):
    val1 = dpg.get_value("line1")
    val2 = dpg.get_value("line2")
    lower = min([val1,val2])
    upper = max([val1,val2])
    dpg.set_value("Plot/Autoscale", False)
    dpg.configure_item("colormap",min_scale=lower,max_scale=upper)
    dpg.configure_item("heat_series",scale_min=lower,scale_max=upper)

def get_scan_range(*args):
    if dpg.is_plot_queried("plot"):
        xmin,xmax,ymin,ymax = dpg.get_plot_query_area("plot")
        new_centers = [(xmin+xmax)/2, (ymin+ymax)/2]
        new_spans = [xmax-xmin, ymax-ymin]
        dpg.set_value("Scan/Centers (V)",new_centers)
        dpg.set_value("Scan/Spans (V)",new_spans)

def guess_time(*args):
    pts = dpg.get_value("Scan/Points")
    ctime = dpg.get_value("Scan/Count Time (ms)") + dpg.get_value("Scan/Wait Time (ms)")
    scan_time = pts[0] * pts[1] * ctime / 1000
    time_string = str(datetime.timedelta(seconds=scan_time)).split(".")[0]
    dpg.set_value("Scan/Estimated Time",time_string)

def choose_save_dir(*args):
    chosen_dir = dpg.add_file_dialog(label="Chose Save Directory", 
                        default_path=dpg.get_value("save_dir"), 
                        directory_selector=True, modal=True,callback=set_save_dir)

def set_save_dir(sender,chosen_dir,user_data):
    dpg.set_value("save_dir",chosen_dir['file_path_name'])

def save_scan(*args):
    path = Path(dpg.get_value("save_dir"))
    filename = dpg.get_value("save_file")
    path /= filename
    as_npz = not (".csv" in filename)
    print(path)
    galvo_scan.save_results(str(path),as_npz=as_npz)

def cursor_drag(sender,value,user_data):
    if sender == "cc":
        point = dpg.get_value("cc")[:2]
        dpg.set_value("cx",point[0])
        dpg.set_value("cy",point[1])
    if sender == "cx":
        point = dpg.get_value("cc")[:2]
        point[0] = dpg.get_value("cx")
        dpg.set_value("cc",point)
    if sender == "cy":
        point = dpg.get_value("cc")[:2]
        point[1] = dpg.get_value("cy")
        dpg.set_value("cc",point)
    return

def draw_bounds():
    pos = dpg.get_value("JPE/Position")
    zpos = pos[2]
    bound_points = list(pz_conv.bounds('z',zpos))
    bound_points.append(bound_points[0])
    if dpg.does_item_exist('bounds'):
        dpg.delete_item('bounds')
    dpg.draw_polygon(bound_points,tag='bounds',parent="plot")

def set_jpe_pos(sender,value,user_data):
    vx = value[0]
    vy = value[1]
    vz = value[2]
    volts = [vx,vy,vz]
    in_bounds = pz_conv.check_bounds(vx,vy,vz)

    if not in_bounds or np.any(np.array(volts) > pz_config['vmax']) or np.any(np.array(volts) < pz_config['vmin']):
        zs = dpg.get_value("JPE/ZVolts")[:3]
        print(zs)
        carts = pz_conv.cart_from_zs(np.array(zs))
        print(carts)
        carts = [float(cart) for cart in carts]
        carts.append(0.0)
        print(carts)
        print(type(carts))
        print(type(carts[0]))
        dpg.set_value("JPE/Position",carts)
        return

    zs = pz_conv.zs_from_cart(volts)
    dpg.set_value("JPE/ZVolts",zs)
    draw_bounds()
# Begin Menu
rdpg.initialize_dpg("Confocal")
with dpg.window(label="Confocal Scanner Window", tag='main_window'):
    with dpg.group(horizontal=True):
        dpg.add_text("Data Directory:")
        dpg.add_input_text(default_value="X:\\DiamondCloud\\Cryostat Setup", tag="save_dir")
        dpg.add_button(label="Pick Directory", callback=choose_save_dir)
    with dpg.tab_bar():
        with dpg.tab(label="Scanner"):
            with dpg.child_window(autosize_x=True,autosize_y=True):
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(tag="scan",label="Take Scan", callback=start_scan)
                    dpg.add_progress_bar(label="Scan Progress",tag='pb')
                    dpg.add_button(tag="load_scan",label="Load Scan", callback=lambda:dpg.show_item("scan_picker"))
                with dpg.group(horizontal=True):
                    dpg.add_text("Filename:")
                    dpg.add_input_text(tag="save_file", default_value="scan.npz", width=200)
                    dpg.add_button(tag="save_scan_button", label="Save Scan",callback=save_scan)
                    dpg.add_checkbox(tag="auto_save", label="Auto")
                with dpg.group(horizontal=True):
                    dpg.add_button(tag="query_plot",label="Query Plot Range",callback=get_scan_range)

                with dpg.group(horizontal=True, width=0):
                    with dpg.child_window(width=400,autosize_x=False,autosize_y=True,tag="wl_tree"):
                        wl_tree = rdpg.TreeDict('jpe_scan_tree','jpe_scan_params_save.csv')
                        wl_tree.add("Plot/Autoscale",False)
                        wl_tree.add("Scan/Centers (V)",[0.0,0.0],item_kwargs={'min_value':-10,
                                                                                'max_value':10})
                        wl_tree.add("Scan/Spans (V)", [1.0,1.0],item_kwargs={'min_value':-20,
                                                                            'max_value':20})
                        wl_tree.add("Scan/Points", [100,100],item_kwargs={'min_value':0},
                                    callback=guess_time)
                        wl_tree.add("Scan/Count Time (ms)", 10.0, item_kwargs={'min_value':0,
                                                                                'max_value':1000},
                                    callback=guess_time)
                        wl_tree.add("Scan/Wait Time (ms)", 1.0,item_kwargs={'min_value':0,
                                                                            'max_value':1000},
                                    callback=guess_time)
                        wl_tree.add("Scan/Estimated Time", "00:00:00", item_kwargs={'readonly':True},
                        save=False)
                        wl_tree.add("JPE/Position",[0.0,0.0,-3.25],item_kwargs={'min_value':-6.5,
                                                                            "max_value":0,
                                                                            'on_enter':True},
                                    callback=set_jpe_pos)
                        wl_tree.add("JPE/ZVolts",[0.0,0.0,0.0],item_kwargs={'readonly':True},
                                    save=False)
                        guess_time()

                    with dpg.child_window(width=-400,autosize_y=True): 
                        with dpg.plot(label="Heat Series",width=-1,height=-1,
                                        equal_aspects=True,tag="plot",query=True):
                            dpg.bind_font("plot_font")
                            dpg.add_drag_point(color=(204,36,29,122),parent="plot",
                                                    callback=cursor_drag,
                                                    default_value=(0.5,0.5),
                                                    tag="cc")
                            dpg.add_drag_line(color=(204,36,29,122),parent="plot",
                                                    callback=cursor_drag,
                                                    default_value=0.5,vertical=True,
                                                    tag="cx")
                            dpg.add_drag_line(color=(204,36,29,122),parent="plot",
                                                    callback=cursor_drag,
                                                    default_value=0.5,vertical=False,
                                                    tag="cy")
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="heat_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="heat_y")
                            dpg.add_heat_series(np.zeros((50,50)),50,50,
                                                scale_min=0,scale_max=1000,
                                                parent="heat_y",label="heatmap",
                                                tag="heat_series",format='')
                            dpg.bind_colormap("plot",dpg.mvPlotColormap_Viridis)

                    with dpg.child_window(width=-0,autosize_y=True):
                        with dpg.group(horizontal=True):
                            dpg.add_colormap_scale(min_scale=0,max_scale=1000,
                                                    width=100,height=-1,tag="colormap",
                                                    colormap=dpg.mvPlotColormap_Viridis)
                            with dpg.plot(label="Histogram", width=-1,height=-1) as histogram:
                                dpg.bind_font("plot_font")
                                dpg.add_plot_axis(dpg.mvXAxis,label="Occurance",tag="hist_x")
                                dpg.add_plot_axis(dpg.mvYAxis,label="Counts",tag="hist_y")
                                dpg.add_area_series([0],[0],parent="hist_x",
                                                    fill=[120,120,120,120],tag="histogram")
                                dpg.add_drag_line(callback=set_scale,default_value=0,
                                                    parent=histogram,tag="line1",vertical=False)
                                dpg.add_drag_line(callback=set_scale,default_value=0,
                                                    parent=histogram,tag="line2",vertical=False)
draw_bounds()
"""
rdpg.initialize_dpg("Confocal")
with dpg.window(tag='main_window', label="Test Window") as main_window:
    with dpg.group(horizontal=True):
        dpg.add_text("Data Directory:")
        dpg.add_input_text(default_value="X:\\DiamondCloud\\",tag="save_dir")
        dpg.add_button(label="Pick Directory", callback=choose_save_dir)
    # Begin Tabs
    with dpg.tab_bar() as main_tabs:
        # Begin Scanner Tab
        with dpg.tab(label="Scanner"):
            # Begin  
            with dpg.child_window(autosize_x=True,autosize_y=True):
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Scan",callback=start_scan, tag='scan')
                    dpg.add_progress_bar(label="Scan Progress",tag='pb')

                with dpg.group(horizontal=True):
                    dpg.add_text("Filename:")
                    dpg.add_input_text(default_value="datafile.npz", width=200, tag="save_file")
                    dpg.add_button(label="Save",callback=save_scan,tag="save_button")
                    dpg.add_checkbox(label="Auto",tag="auto_save")
                    
                with dpg.group(horizontal=True, width=0):
                    with dpg.child_window(width=200,autosize_y=True):
                        with dpg.group(horizontal=True):
                            dpg.add_text("Autoscale")
                            dpg.add_checkbox(default_value=True)
                        dpg.add_button(label="Query Scan Range",width=-1,callback=get_scan_range)
                        dpg.add_text("Scan Settings")
                        dpg.add_text("Center (V)", indent=1)
                        dpg.add_input_floatx(default_value=[-0.312965,-0.0164046],
                                             min_value=-10.0, max_value=10.0, 
                                             width=-1, indent=1,size=2,
                                             tag="centers")
                        dpg.add_text("Span (V)", indent=1)
                        dpg.add_input_floatx(default_value=[1.0,1.0],min_value=-20.0, tag="spans", 
                                             max_value=20.0, width=-1, indent=1,size=2)
                        dpg.add_text("Points", indent=1)
                        dpg.add_input_intx(default_value=[100,100],min_value=0, tag="points",
                                                    max_value=10000.0, width=-1, 
                                                    indent=1,callback=guess_time,size=2)
                        dpg.add_text("Count Time (ms)")
                        dpg.add_input_float(default_value=10.0,min_value=0.0,max_value=10000.0, tag="ct",
                                                    width=-1.0,step=0,callback=guess_time)
                        dpg.add_text("Wait Time (ms)")
                        dpg.add_input_float(default_value=1.0,min_value=0.0,max_value=10000.0, tag="wt",
                                                    width=-1,step=0,callback=guess_time)
                        dpg.add_text("Estimate Scan Time")
                        dpg.add_input_text(default_value="00:00:00",width=-1, readonly=True, tag="st")
                        guess_time()

                    # create plot
                    with dpg.child_window(width=-400,autosize_y=True): 
                        with dpg.plot(label="Heat Series",width=-1,height=-1,
                                        equal_aspects=True,tag="plot",query=True):
                            dpg.bind_font("plot_font")
                            dpg.add_drag_point(color=(204,36,29,122),parent="plot",
                                                    callback=cursor_drag,
                                                    default_value=(0.5,0.5))
                            dpg.add_drag_line(color=(204,36,29,122),parent="plot",
                                                    callback=cursor_drag,
                                                    default_value=0.5,vertical=True)
                            dpg.add_drag_line(color=(204,36,29,122),parent="plot",
                                                    callback=cursor_drag,
                                                    default_value=0.5,vertical=False)
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="heat_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="y",tag="heat_y")
                            dpg.add_heat_series(np.zeros((50,50)),50,50,
                                                scale_min=0,scale_max=1000,
                                                parent="heat_y",label="heatmap",
                                                tag="heat_series",format='',)

                    with dpg.child_window(width=-0,autosize_y=True):
                        with dpg.group(horizontal=True):
                            dpg.add_colormap_scale(min_scale=0,max_scale=1000,
                                                    width=100,height=-1,tag="colormap")
                            with dpg.plot(label="Histogram", width=-1,height=-1) as histogram:
                                dpg.bind_font("plot_font")
                                dpg.add_plot_axis(dpg.mvXAxis,label="Occurance",tag="hist_x")
                                dpg.add_plot_axis(dpg.mvYAxis,label="Counts",tag="hist_y")
                                dpg.add_area_series([0],[0],parent="hist_x",
                                                    fill=[120,120,120,120],tag="histogram")
                                dpg.add_drag_line(callback=set_scale,default_value=0,
                                                    parent=histogram,tag="line1",vertical=False)
                                dpg.add_drag_line(callback=set_scale,default_value=0,
                                                    parent=histogram,tag="line2",vertical=False)
                    for wid in ["colormap", "plot"]:
                        dpg.set_colormap(wid,dpg.mvPlotColormap_Viridis)
"""
dpg.set_primary_window('main_window',True)
rdpg.start_dpg()
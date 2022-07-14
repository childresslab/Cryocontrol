from multiprocessing.sharedctypes import Value
from turtle import color, update, window_width
import numpy as np
from apis import rdpg
from threading import Thread
import logging as log
dpg = rdpg.dpg
log.basicConfig(format='%(levelname)s:%(message)s ', level=log.WARNING)


colormaps = {'viridis' : dpg.mvPlotColormap_Viridis,
             'plasma' : dpg.mvPlotColormap_Plasma,
             'hot' : dpg.mvPlotColormap_Hot,
             'cool' : dpg.mvPlotColormap_Cool,
             'pink' : dpg.mvPlotColormap_Pink,
             'twilight' : dpg.mvPlotColormap_Twilight,
             'rdbu' : dpg.mvPlotColormap_RdBu,
             'brbg' : dpg.mvPlotColormap_BrBG,
             'piyg' : dpg.mvPlotColormap_PiYG,
             'spectral' : dpg.mvPlotColormap_Spectral,
             'greys' : dpg.mvPlotColormap_Greys}

def get_colormap(colormap:str):
    colormap.lower()
    if colormap not in colormaps.keys():
        raise ValueError(f"{colormap} not found in {list(colormaps.keys())}")
    return colormaps[colormap]

class mvHistPlot():
    def __init__(self,label:str,
                 cursor:bool=False,
                 cursor_callback = None,
                 query=True,
                 equal=True,
                 width=500,
                 height=500,
                 scale_width=300,
                 nbin=50,
                 colormap='viridis',
                 autoscale=True,
                 min_hist=0,
                 max_hist=1E12,
                 rows = 50,
                 cols = 50,
                 parent=None):
        self.parent = parent
        self.label = label
        self.queryable = query
        self.equal = equal
        self.cmap = colormap
        self.autoscale = autoscale
        self.minhist = min_hist
        self.maxhist = max_hist
        self.width = width
        self.height = height
        self.scale_width = scale_width
        self.nbin = nbin
        self.data = np.zeros((rows,cols))
        self.rows = rows
        self.cols = cols
        self.cmap = get_colormap(colormap)
        self.cursor = cursor
        self.cursor_callback = cursor_callback
        self.update_thread = Thread(target=self.update_func)

    def make_gui(self):
        if self.parent is None:
            self.parent = dpg.add_window(label="Hist Plot", width=self.width, height=self.height)
        else:
            self.parent = self.parent
        scale_width = self.scale_width
        height = self.height
        label = self.label
        equal = self.equal
        query = self.queryable
        rows = self.rows
        cols = self.cols
        colormap = self.cmap
        cursor = self.cursor
        cursor_callback = self.cursor_callback

        with dpg.group(horizontal=True,parent=self.parent):
            with dpg.child_window(width=-scale_width,height=height): 
                with dpg.plot(label=label,width=-1,height=-1,
                                equal_aspects=equal,tag=f"{label}_plot",query=query):
                    dpg.add_plot_legend()
                    dpg.bind_font("plot_font")
                    # REQUIRED: create x and y axes
                    dpg.add_plot_axis(dpg.mvXAxis, label="x", tag=f"{label}_heat_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="y",tag=f"{label}_heat_y")
                    dpg.add_heat_series(self.data,rows,cols,
                                        scale_min=0,scale_max=1000,
                                        parent=f"{label}_heat_y",label=f"{label} Heatmap",
                                        tag=f"{label}_heat_series",format='',)
                    dpg.add_button(tag=f"{label}_autofit",parent=f"{label}_heat_series",callback=self.autoscale_plots,label="Autoscale")
                    dpg.add_combo(list(colormaps.keys()),parent=f"{label}_heat_series",callback=self.context_select_colormap,
                                  default_value = colormap,label="Select Colormap")
                    dpg.bind_colormap(f"{label}_plot",self.cmap)

            with dpg.child_window(width=scale_width,height=height):
                with dpg.group(horizontal=True):
                    dpg.add_colormap_scale(min_scale=0,max_scale=1000,
                                            width=100,height=-1,tag=f"{label}_colormap",
                                            colormap=self.cmap)
                    with dpg.plot(label="Histogram", width=-1,height=-1) as histogram:
                        dpg.bind_font("plot_font")
                        dpg.add_plot_axis(dpg.mvXAxis,label="Occurance",tag=f"{label}_hist_x")
                        dpg.add_plot_axis(dpg.mvYAxis,tag=f"{label}_hist_y")
                        dpg.add_area_series([0],[0],parent=f"{label}_hist_x",
                                            fill=[120,120,120,120],tag=f"{label}_histogram")
                        dpg.add_drag_line(callback=self.set_scale,default_value=0,
                                            parent=histogram,tag=f"{label}_line1",vertical=False)
                        dpg.add_drag_line(callback=self.set_scale,default_value=0,
                                            parent=histogram,tag=f"{label}_line2",vertical=False)
        if cursor:
            self.add_cursor(callback=cursor_callback)

    def update_plot(self,data):
        if len(data) != self.rows*self.cols and data.shape[0]*data.shape[1] != self.rows*self.cols:
            raise ValueError(f"Length of data array {len(data)},{data.shape} is not equal to num_rows * num_cols = {self.rows*self.cols}. Use set_size() first.")
        self.data = data
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = Thread(target=self.update_func)
            self.update_thread.run()

    def update_func(self):
        dpg.set_value(f"{self.label}_heat_series", [self.data,[0.0,1.0],[],[],[]])
        self.update_histogram()
        if self.autoscale:
            self.autoscale_plots()

    def update_histogram(self):
        data = self.data
        hist_data = self.data[np.where(np.logical_and(data>=self.minhist,
                                                      data<=self.maxhist))]
        nbins = self.nbin
        occ,edges = np.histogram(hist_data,bins=nbins)
        xs = [0] + list(np.repeat(occ,2)) + [0,0] 
        ys = list(np.repeat(edges,2)) + [0]
        self.hist_bins = edges 
        self.hist_counts = occ
        dpg.set_value(f"{self.label}_histogram",[xs,ys,[],[],[]])

    def set_scale(self,sender,app_data,user_data):
        val1 = dpg.get_value(f"{self.label}_line1")
        val2 = dpg.get_value(f"{self.label}_line2")
        lower = min([val1,val2])
        upper = max([val1,val2])
        dpg.configure_item(f"{self.label}_colormap",min_scale=lower,max_scale=upper)
        dpg.configure_item(f"{self.label}_heat_series",scale_min=lower,scale_max=upper)

    def autoscale_plots(self):
        lower = np.min(self.hist_bins)
        upper = np.max(self.hist_bins)
        dpg.configure_item(f"{self.label}_colormap",min_scale=lower,max_scale=upper)
        dpg.configure_item(f"{self.label}_heat_series",scale_min=lower,scale_max=upper)
        dpg.set_value(f"{self.label}_line1",lower)
        dpg.set_value(f"{self.label}_line2",upper) 
        for ax in [f"{self.label}_heat_x",f"{self.label}_heat_y",
                   f"{self.label}_hist_x",f"{self.label}_hist_y"]:
            dpg.fit_axis_data(ax)

    def add_cursor(self,callback=None):
        if callback is None:
            cursor_callback = self.bind_cursor
        else:
            def cursor_callback(sender,app_data, user_data):
                self.bind_cursor(sender,app_data,user_data)
                callback(sender,dpg.get_value(f"{self.label}_cc")[:2])
                return
        log.debug("Adding Cursor")
        dpg.add_drag_point(color=(204,36,29,122),parent=f"{self.label}_plot",
                                callback=cursor_callback,
                                default_value=(0.5,0.5),
                                tag=f"{self.label}_cc")
        dpg.add_drag_line(color=(204,36,29,122),parent=f"{self.label}_plot",
                                callback=cursor_callback,
                                default_value=0.5,vertical=True,
                                tag=f"{self.label}_cx")
        dpg.add_drag_line(color=(204,36,29,122),parent=f"{self.label}_plot",
                                callback=cursor_callback,
                                default_value=0.5,vertical=False,
                                tag=f"{self.label}_cy")

    def bind_cursor(self,sender,app_data,user_data):
        point = dpg.get_value(f"{self.label}_cc")[:2]
        if sender == f"{self.label}_cx":
            point[0] = dpg.get_value(f"{self.label}_cx")
        elif sender == f"{self.label}_cy":
            point[1] = dpg.get_value(f"{self.label}_cy")
        dpg.set_value(f"{self.label}_cx",point[0])
        dpg.set_value(f"{self.label}_cy",point[1])
        dpg.set_value(f"{self.label}_cc",point)

    def set_cursor(self,point):
        dpg.set_value(f"{self.label}_cx",point[0])
        dpg.set_value(f"{self.label}_cy",point[1])
        dpg.set_value(f"{self.label}_cc",point)

    def query_plot(self):
        if dpg.is_plot_queried(f"{self.label}_plot"):
            xmin,xmax,ymin,ymax = dpg.get_plot_query_area(f"{self.label}_plot")
            return xmin,xmax,ymin,ymax
        else:
            return None,None,None,None

    def set_size(self,rows,cols):
        log.warning(f"Setting {self.label}_heat_series size")
        self.rows = rows
        self.cols = cols
        dpg.delete_item(f"{self.label}_heat_series")
        data = np.ones((rows,cols)) * -1.0
        dpg.add_heat_series(data,self.rows,self.cols,
                            scale_min=0,scale_max=1000,
                            parent=f"{self.label}_heat_y",label=f"{self.label} Heatmap",
                            tag=f"{self.label}_heat_series",format='',)
        dpg.add_button(tag=f"{self.label}_autofit",parent=f"{self.label}_heat_series",callback=self.autoscale_plots,label="Autoscale")
        dpg.add_combo(list(colormaps.keys()),parent=f"{self.label}_heat_series",callback=self.context_select_colormap,
                                  default_value = self.cmap,label="Select Colormap")

        config = dpg.get_item_configuration(f"{self.label}_heat_series")
        if rows != config['rows'] or cols != config['cols']:
            raise RuntimeError(f"Error occured while reconfiguring {self.label}_heat_series")

    def set_bounds(self,xmin,xmax,ymin,ymax):
        log.warning(f"Setting {self.label}_heat_series bounds")
        dpg.configure_item(f"{self.label}_heat_series",bounds_min=(xmin,ymin),bounds_max=(xmax,ymax))

    def set_colormap(self,colormap:str):
        self.cmap = get_colormap(colormap)
        dpg.bind_colormap(f"{self.label}_plot",self.cmap)
        dpg.bind_colormap(f"{self.label}_colormap",self.cmap)

    def context_select_colormap(self,sender,app_data,user_data):
        self.set_colormap(app_data)
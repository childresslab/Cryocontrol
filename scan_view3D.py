from numpy.core.arrayprint import array2string
from spinmob import egg
import cavspy as cs
import traceback
import numpy as np
import spinmob as sm
import matplotlib.pyplot as plt
_p = traceback.print_last #Very usefull command to use for getting the last-not-printed error

import time

# Debug stuff.
_debug_enabled     = True

def _debug(*a):
    if _debug_enabled: 
        s = []
        for x in a: s.append(str(x))
        print(', '.join(s))

class ScanView(egg.gui.Window):
    """
    GUI for plotting cavity scans.
    """
    def __init__(self, name="Scan Viewer"): 
        """
        Initialize 
        """    
        _debug('GUIMap:__init__')
        
        # Run the basic stuff for the initialization
        egg.gui.Window.__init__(self, title=name)

        # Fill up the GUI 
        self.initialize_GUI()
        
    def initialize_GUI(self):
        """
        Fill up the GUI
        """        
        _debug('GUIMap: initialize_GUI')
        # Place a buttong for loading the scans
        self.button_load_scans = self.place_object(egg.gui.Button(), alignment=1)
        self.button_load_scans.set_text('Load scan')
        self.connect(self.button_load_scans.signal_clicked, self.button_load_scans_clicked)

        self.new_autorow()
        # Place the dictionay tree for all the parameters
        self.treeDic_settings  = egg.gui.TreeDictionary(autosettings_path='setting_map')
        self.place_object(self.treeDic_settings, column=0,row=1, row_span=1,column_span=1, alignment=1)
        list_colormap = PersonalColorMap().get_list_colormaps()
        self.treeDic_settings.add_parameter('Colormap', 0, 
                                           type='list', values=list_colormap)
        self.treeDic_settings.add_parameter('Set_aspect', False, 
                                           type='bool',
                                           tip='Wether or not to set the axis to scale. ')  
        self.treeDic_settings.add_parameter('Forward', True,
                                           type='bool',
                                           tip='Wether to plot the forward or reverse scan')
        self.treeDic_settings.add_parameter('Reducing', 0,
                                           type='list',values=['slice','mean','max','delta'])
        self.treeDic_settings.add_parameter('Slice Index',0,
                                            'int')
        # Some connections
        self.treeDic_settings.connect_signal_changed('Colormap', self.update_colormap)
        self.treeDic_settings.connect_signal_changed('Set_aspect', self.update_image)
        self.treeDic_settings.connect_signal_changed('Forward',self.update_direction)
        self.treeDic_settings.connect_signal_changed('Reducing', self.update_image)
        self.treeDic_settings.connect_signal_changed('Slice Index', self.update_slice)

    def create_plots(self):
        _debug("GUIMap: Creating Plots")
        # Create the plot and image view objects.
        # Also place them and link their axes if needed
        self.plot_item1 = egg.pyqtgraph.PlotItem()
        self.plot_image1 = egg.pyqtgraph.ImageView(view=self.plot_item1)
        self.plot_image1.view.invertY(False)
        self.place_object(self.plot_image1, row=1,column=1,column_span=10,alignment=0)


        self.plot2 = egg.pyqtgraph.PlotWidget()
        self.plot_item2 = self.plot2.getPlotItem()
        self.place_object(self.plot2, row=1,column=13,column_span=1,alignment=2)
    
    def reduce_scan(self):
        func = self.treeDic_settings['Reducing']
        if func == 'slice':
            if self.treeDic_settings['Slice Index'] < 0:
                self.treeDic_settings['Slice Index'] = 0
            if self.treeDic_settings['Slice Index'] >= self.Nz:
                self.treeDic_settings['Slice Index'] = self.Nz - 1
            self.img = self.scan['data'][:,:,self.treeDic_settings['Slice Index']]
        elif func == 'mean':
            self.img = np.mean(self.scan['data'],axis=2)
        elif func == 'max':
            self.img = np.max(self.scan['data'],axis=2)
        elif func == 'delta':
            self.img = np.max(self.scan['data'],axis=2) - np.min(self.scan['data'],axis=2)
        
        img1, img2 = cs.scans.de_interleave(self.img)
        values1 = img1.astype(np.float32)
        self.Z1 = values1

        values2 = img2.astype(np.float32)
        self.Z2 = values2

    def button_load_scans_clicked(self):
        """
        Load in scan data using Cavspy, which handles processing the data
        and converting the units. 
        """                       
        _debug('GUIMap: button_load_scans clicked')
        
        # Get the data
        filepath = sm.dialogs.load()
        scan = cs.data.read(filepath)
        cs.scans.convert_units(scan)

        # Compute all needed parameters for plotting
        self.Ny,self.Nx,self.Nz = scan['data'].shape
        self.vmin = np.min(scan['data'])
        self.vmax = np.max(scan['data'])

        xs = scan['xs']
        ys = scan['ys']
        zs = scan['zs']
        self.xmax = np.max(xs)
        self.xmin = np.min(xs)
        self.ymax = np.max(ys)
        self.ymin = np.min(ys)
        self.zmin = np.min(zs)
        self.zmax = np.max(zs)

        self.scan = scan
        
        # Update the image
        self.create_plots()
        self.initialize_image()
        self.update_image()
        # Update the color map
        self.update_colormap()
        self.plot_single_scan()

    def initialize_image(self):
        """
        Set up the axes
        """
        _debug('GUIMap: initialize_image')
        
        # Set the axis
        self.plot_item1.setLabel('bottom', text='X Pos (um)')
        self.plot_item1.setLabel('left', text='Y Pos (um)')
        # Set the scaling
        self.scale_x = (self.xmax-self.xmin)/self.Nx
        self.scale_y = (self.ymax-self.ymin)/self.Ny
        
        roi_pos = ((self.xmax+self.xmin)/2, (self.ymax+self.ymin)/2)
        roi_size = (self.scale_x, self.scale_y)
        try:
            self.plot_iamge1.removeItem(self.roi_point)
        except AttributeError:
            pass
        bounds = egg.pyqtgraph.QtCore.QRectF(self.xmin,self.ymin,self.xmax-self.xmin,self.ymax-self.ymin)
        self.roi_point = egg.pyqtgraph.ROI(pos=roi_pos, size=roi_size, 
                                           snapSize=np.min(roi_size), 
                                           translateSnap = True, resizable=False,
                                           pen='g',maxBounds=bounds)
        self.roi_point.sigRegionChanged.connect(self.plot_single_scan)
        self.plot_image1.addItem(self.roi_point)

    def update_image(self,keep_hist=False):
        """
        Update the map with the actual Z data
        """
        if keep_hist:
             levels = self.plot_image1.ui.histogram.getLevels()
        _debug('GUIMap: update_image')
        self.reduce_scan()
        # Set the ratio according to the settings
        value = self.treeDic_settings['Set_aspect']
        # True gives a 1:1 aspect ratio. False allows for arbitrary axis scaling.
        self.plot_image1.view.setAspectLocked(value)
        if self.treeDic_settings['Forward']:
            image = self.Z1.T
        else:
            image = self.Z2.T
        self.plot_image1.setImage(image,
                                  pos=(self.xmin, self.ymin),
                                  scale = (self.scale_x, self.scale_y))
        
        # scale/pan the view to fit the image.
        self.plot_image1.autoRange()
        self.plot_item2.setXRange(self.vmin,self.vmax)
        self.plot_item2.setYRange(self.zmin*1000,self.zmax*1000)
        if keep_hist:
             self.plot_image1.setLevels(*levels)

    def move_slice(self):
        zpos = self.slice_line.pos()[1]
        zs = np.linspace(self.zmin,self.zmax,self.Nz)*1000
        index = np.argmin(np.abs(zs-zpos))
        self.treeDic_settings['Slice Index'] = index

    def update_slice(self):
        if self.treeDic_settings['Reducing'] != "slice":
            pass
        else:
            self.update_image(True)
    
    def update_direction(self):
        self.update_image()
        self.plot_single_scan()

    def update_colormap(self):
        """
        Update the color of the image to fit the settings
        """        
        _debug('GUIMap: update_colormap ')
        
        name = self.treeDic_settings['Colormap']
        mycmap = PersonalColorMap().get_colormap(name)
        self.plot_image1.setColorMap(mycmap)

    def plot_single_scan(self):
        position = self.roi_point.pos()
        indices = [(position[0]-self.xmin)/self.scale_x, (position[1]-self.ymin)/self.scale_y]
        indices = [int(round(index)) for index in indices]

        zs = np.linspace(self.zmin,self.zmax,self.Nz)

        if self.treeDic_settings['Forward']:
            indices[1] = indices[1] // 2 * 2
        else:
            indices[1] = indices[1] // 2 * 2 + 1
        try:
            self.plot_item2.removeItem(self.slice_line)
        except AttributeError:
            pass
        self.slice_line = egg.pyqtgraph.InfiniteLine(zs[self.treeDic_settings['Slice Index']]*1000,0,'g',True,
                                                     bounds=[np.min(zs)*1000,np.max(zs)*1000])
        self.slice_line.sigPositionChanged.connect(self.move_slice)
        data = self.scan['data'].transpose(1,0,2)
        self.plot_item2.plot(data[indices[0],indices[1],:],zs*1000,clear=True)
        self.plot_item2.addItem(self.slice_line)


def disable_button(button):
    button._widget.setEnabled(False)

def enable_button(button):
    button._widget.setEnabled(True)

class PersonalColorMap():
    """
    This class is aimed to store various colormap that we would like to use. 
    """
    def __init__(self):
        """
        """
        
        # Contains all the name of the color map that we have. 
        self.list_colormaps = ['viridis','magma','plasma','inferno']
        
    def get_list_colormaps(self):
        return self.list_colormaps
    
    def get_colormap(self, cm_name):
        pltMap = plt.get_cmap(cm_name)
        colors = pltMap.colors
        colors = [list(map(lambda x: int(round(x*255)),color)) for color in colors] 
        colors = [c + [255] for c in colors]
        positions = np.linspace(0, 1, len(colors))
        pgMap = egg.pyqtgraph.ColorMap(positions, colors)
        return pgMap

if __name__ == "__main__":
    scan = ScanView()
    scan.create_plots()
    scan.show()
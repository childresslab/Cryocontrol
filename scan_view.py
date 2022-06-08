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
        
        # Some attibutes
        self.plots_exist = False
        self.doubled = False

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
        #self.button_over_scans = self.place_object(egg.gui.Button(), alignment=1)
        #self.button_over_scans.set_text("Overlay Scan")
        #self.connect(self.button_over_scans.signal_clicked, self.button_overlay_scans_clicked)

        self.new_autorow()
        # Place the dictionay tree for all the parameters
        self.treeDic_settings  = egg.gui.TreeDictionary(autosettings_path='setting_map')
        self.place_object(self.treeDic_settings, column=0,row=1, row_span=1,column_span=1, alignment=1)
        list_colormap = PersonalColorMap().get_list_colormaps()
        self.treeDic_settings.add_parameter('Colormap', 0, 
                                           type='list', values=list_colormap)
        self.treeDic_settings.add_parameter('Set_aspect', False, 
                                           type='bool',
                                           tip='Wether or not to fix pixels to 1x1. ')
        self.treeDic_settings.add_parameter('Auto Convert', True,
                                            type='bool',
                                            tip='If true uses cavspy to convert, else use following values.')
        self.treeDic_settings.add_parameter('X Conversion', 117,
                                            type='float',
                                            tip='um/V conversion factor for x-axis')
        self.treeDic_settings.add_parameter('Y Conversion', 117,
                                            type='float',
                                            tip='um/V conversion factor for y-axis')                                    
      
        # Some connections
        self.treeDic_settings.connect_signal_changed('Colormap', self.update_colormap)
        self.treeDic_settings.connect_signal_changed('Set_aspect', self.update_image)
        self.treeDic_settings.connect_signal_changed('Auto Convert', self.update_image)
        self.treeDic_settings.connect_signal_changed('X Conversion', self.update_image)
        self.treeDic_settings.connect_signal_changed('Y Conversion', self.update_image)


        self.button_copy_right = self.place_object(egg.gui.Button(), column=4, row=0, alignment=0)
        self.button_copy_right.set_text("Copy Values ->")
        self.connect(self.button_copy_right.signal_clicked, self.copy_right)
        self.button_copy_left = self.place_object(egg.gui.Button(), column=9, row=0, alignment=0)
        self.button_copy_left.set_text("Copy Values <-")
        self.connect(self.button_copy_left.signal_clicked, self.copy_left)
        disable_button(self.button_copy_left)
        disable_button(self.button_copy_right)
        #disable_button(self.button_over_scans)

    def create_plots(self):
        if self.plots_exist:
            try:
                self.plot_item1.clear()
                self.plot_image1.clear()
                self.remove_object(self.plot_image1)
                self.plot_item2.clear()
                self.plot_image2.clear()
                self.remove_object(self.plot_image2)
            except ValueError:
                print("Value Error when Plotting")
                pass
            except AttributeError:
                print("Attribute Error when Plotting")
                pass
        
        # Create the plot and image view objects.
        # Also place them and link their axes if needed
        self.plot_item1 = egg.pyqtgraph.PlotItem()
        self.plot_image1 = egg.pyqtgraph.ImageView(view=self.plot_item1)
        self.plot_image1.view.invertY(False)
        if not self.doubled:
            self.place_object(self.plot_image1, row=1,column=1,column_span=10,alignment=0)
            disable_button(self.button_copy_left)
            disable_button(self.button_copy_right)
        else:
            self.place_object(self.plot_image1, row=1,column=1, row_span=5,column_span=4,alignment=0)
            self.original_width = self.plot_image1.geometry().bottom() # Record the original width for the rescaling
            self.plot_item2 = egg.pyqtgraph.PlotItem()
            self.plot_image2 = egg.pyqtgraph.ImageView(view=self.plot_item2)
            self.plot_image2.view.invertY(False)

            self.place_object(self.plot_image2, row=1,column=6,row_span=5, column_span=4,alignment=0)  

            self.plot_item2.setXLink(self.plot_item1)
            self.plot_item2.setYLink(self.plot_item1)
           
            enable_button(self.button_copy_left)
            enable_button(self.button_copy_right)
            #enable_button(self.button_over_scans)
            

        self.plots_exist = True
        
    def button_load_scans_clicked(self):
        """
        Load in scan data using Cavspy, whwich haindles processing the data
        and converting the units. 
        """                       
        _debug('GUIMap: button_load_scans clicked')
        
        # Get the data
        filepath = sm.dialogs.load()
        scan = cs.data.read(filepath)
        cs.scans.convert_units(scan)

        # Deinterleave piezo scans and massage data
        self.doubled = scan['scan_type'] == 0
        if self.doubled:
            img1, img2 = cs.scans.de_interleave(scan['data'])
            values1 = img1.astype(np.float32)
            self.Z1 = values1

            values2 = img2.astype(np.float32)
            self.Z2 = values2

        else:
             img1 = scan['data']
             values1 = img1.astype(np.float32)
             self.Z1 = values1

        # Compute all needed parameters for plotting
        self.Ny,self.Nx = scan['data'].shape
        self.vmin = np.min(scan['data'])
        self.vmax = np.max(scan['data'])
        
        xs = scan['xs']
        ys = scan['ys']
        self.xmax = np.max(xs)
        self.xmin = np.min(xs)
        self.ymax = np.max(ys)
        self.ymin = np.min(ys)
        
        Vxs = scan['Vxs']
        Vys = scan['Vys']
        self.Vxmax = np.max(Vxs)
        self.Vxmin = np.min(Vxs)
        self.Vymax = np.max(Vys)
        self.Vymin = np.min(Vys)
        
        # Update the image
        self.create_plots()
        self.initialize_image()
        self.update_image()

    def button_overlay_scans_clicked(self):
        #TODO: Figure out how to overlay the images,
        # might requiring manually creating a larger square grid that encompasses both scans, and
        # stitching them together.
        """
        Load scans and overlay 
        """                       
        _debug('GUIMap: button_overlay_scans_clicked')
        QPainter = egg.pyqtgraph.Qt.QtGui.QPainter
        # Get the data
        filepath = sm.dialogs.load()
        scan = cs.data.read(filepath)
        cs.scans.convert_units(scan)

        # Deinterleave piezo scans and massage data
        self.doubled = True if scan['scan_type'] == 0 else False
        if self.doubled:
            self.plot_image1.getImageItem().setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            self.plot_image2.getImageItem().setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            img1, img2 = cs.scans.de_interleave(scan['data'])
            values1 = img1.astype(np.float32)
            self.Z1 = values1

            values2 = img2.astype(np.float32)
            self.Z2 = values2

        else:
            img1 = scan['data']
            self.plot_image1.getImageItem().setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

            if scan['scan_type'] == 3:
                img1 = np.flip(img1,axis=0)
            values1 = img1.astype(np.float32)
            self.Z1 = values1


        # Compute all needed parameters for plotting
        self.Ny,self.Nx = scan['data'].shape
        self.vmin = np.min(scan['data'])
        self.vmax = np.max(scan['data'])

        xs = scan['xs']
        ys = scan['ys']
        self.xmax = np.max(xs)
        self.xmin = np.min(xs)
        self.ymax = np.max(ys)
        self.ymin = np.min(ys)
        
    # Following two methods copy the histogram levels of one scan onto it's 
    # accompanying scan. Only useful for deintereleaved piezo scans.
    def copy_left(self):
        self.plot_image1.setLevels(*self.plot_image2.ui.histogram.getLevels())

    def copy_right(self):
        self.plot_image2.setLevels(*self.plot_image1.ui.histogram.getLevels())

    def initialize_image(self):
        """
        Set up the axes
        """
        _debug('GUIMap: initialize_image')
        
        # Set the axis
        self.plot_item1.setLabel('bottom', text='X Pos (um)')
        self.plot_item1.setLabel('left', text='Y Pos (um)')
        if self.doubled:
            self.plot_item2.setLabel('bottom', text='X Pos (um)')
            self.plot_item2.setLabel('left', text='Y Pos (um)')

        
    def update_image(self):
        """
        Update the map with the actual Z data
        """
        _debug('GUIMap: update_image')
        if self.treeDic_settings['Auto Convert']:
            xmin = self.xmin
            xmax = self.xmax
            ymin = self.ymin
            ymax = self.ymax
            # Set the scaling
            scale_x = (xmax-xmin)/self.Nx
            scale_y = (ymax-ymin)/self.Ny
        else:
            xmin = self.Vxmin * self.treeDic_settings['X Conversion']
            xmax = self.Vxmax * self.treeDic_settings['X Conversion']
            ymin = self.Vymin * self.treeDic_settings['Y Conversion']
            ymax = self.Vymax * self.treeDic_settings['Y Conversion']
            # Set the scaling
            scale_x = (xmax-xmin)/self.Nx
            scale_y = (ymax-ymin)/self.Ny
            
        # Set the ratio according to the settings
        value = self.treeDic_settings['Set_aspect']
        # True gives a 1:1 aspect ratio. False allows for arbitrary axis scaling.
        self.plot_image1.view.setAspectLocked(value)
       
        self.plot_image1.setImage(self.Z1.T,
                                  pos=(xmin, ymin),
                                  scale = (scale_x, scale_y))
        
        # scale/pan the view to fit the image.
        self.plot_image1.autoRange()

        if self.doubled:
            # True gives a 1:1 aspect ratio. False allows for arbitrary axis scaling.
            self.plot_image2.view.setAspectLocked(value)
            self.plot_image2.setImage(self.Z2.T,
                                      pos=(xmin, ymin),
                                      scale =(scale_x, scale_y))
            self.plot_image2.autoRange()

        # Update the color map
        self.update_colormap()

    def update_colormap(self):
        """
        Update the color of the image to fit the settings
        """        
        _debug('GUIMap: update_colormap ')
        
        name = self.treeDic_settings['Colormap']
        mycmap = PersonalColorMap().get_colormap(name)
        self.plot_image1.setColorMap(mycmap)
        if self.doubled:
            self.plot_image2.setColorMap(mycmap)

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
    scan.show()
import dearpygui.dearpygui as dpg
from datetime import datetime as dt
from typing import Callable, Union, Any
from pathlib import Path
from ast import literal_eval

import logging as log
log.basicConfig(format='%(levelname)s:%(message)s ', level=log.DEBUG)


class TreeDict():
    def __init__(self, parent:Union[str,int], savename:str) -> None:
        self.parent = parent
        self.dict = {}
        self.f_lookup = {'input'     : {float : dpg.add_input_float,
                                        int   : dpg.add_input_int,
                                        str   : dpg.add_input_text,
                                        bool  : dpg.add_checkbox},
                         'drag'      : {float : dpg.add_drag_float,
                                        int   : dpg.add_drag_int},
                         'list'      : {float : dpg.add_input_floatx,
                                        int   : dpg.add_input_intx},
                         'list_drag' : {float : dpg.add_drag_floatx,
                                        int   : dpg.add_drag_intx}
                        }
        self.prefix = f"{parent}_"
        self.savefile = savename
        self.skip_save = []
        dpg.set_frame_callback(1,lambda _: self.load())


    def __getitem__(self, key):
        value = dpg.get_value(f"{self.prefix}{key}")
        if value is None:
            raise ValueError(f"No value found in tree with {self.prefix}{key}")
        return value

    def __setitem__(self,key,value):
        dpg.set_value(f"{self.prefix}{key}",value)
    
    def _make_nodes(self,name,node_kwargs):
        hierarchy = name.split('/')
        layer_dict = self.dict
        # Traverse the hierarchy above the new item, creating
        # layers that don't already exist
        for i,layer in enumerate(hierarchy[:-1]):
            log.debug(f"{layer},{list(layer_dict.keys())}")
            if layer not in layer_dict.keys():
                layer_dict[layer] = {}
                if i == 0:
                    parent = self.parent
                else:
                    parent = f"{self.prefix}{'/'.join(hierarchy[:i])}"
                tag = '/'.join(hierarchy[:i+1])
                node_dict = {'label' : layer,
                             'tag' : f"{self.prefix}{tag}",
                             'parent' : parent,
                             'default_open' : True}
                node_dict.update(node_kwargs)
                log.debug(f"Creating Tree Node {self.prefix}{tag}")
                dpg.add_tree_node(**node_dict)
            layer_dict = layer_dict[layer]
        return layer_dict

    def add(self, name:str, value:Any, val_type:type = None, 
            order:int=1, drag:bool = False, save=True, callback = None,
            node_kwargs:dict = {}, tooltip:str = "", item_kwargs:dict = {}):
        # Generate a callback that also saves the tree
        callback = self.get_save_callback(callback)

        # Keep track of which items we don't save
        if not save:
            self.skip_save.append(name)
        # Make sure all the nodes needed exist
        layer_dict = self._make_nodes(name,node_kwargs)
        hierarchy = name.split('/')
        #log.debug(f"Item Hierarchy: {hierarchy}")
        
        # Block duplicate entries
        if hierarchy[-1] in layer_dict.keys():
            raise RuntimeError(f"{name} already exists in tree.")

        # Get the base items parent object and name
        parent = f"{self.prefix}{'/'.join(hierarchy[:-1])}"
        layer_dict[hierarchy[-1]] = name

        # Autodetect type and order of object.
        if val_type is None:
            val_type = type(value)
            if val_type is list:
                val_type = type(value[0])
                if order == 1:
                    order = len(value)

        # What type of object are we making?
        lookup = 'input'
        if drag:
            if order > 1:
                lookup = 'list_drag'
            else:
                lookup = 'drag'
        else:
            if order > 1:
                lookup = 'list'


        if order > 4:
            raise ValueError(f"Number of inputs can't exceed 4. {order = }.")
        try:
            creation_func = self.f_lookup[lookup][val_type]
        except KeyError:
            raise TypeError(f"Type {val_type} not valid for widget style {lookup}.")
        print(f"{self.prefix}{name}")
        item_dict = {'tag' : f"{self.prefix}{name}",
                     'default_value' : value,
                     'callback' : callback,
                     }
        if order > 1:
            item_dict['size'] = order
        if val_type is not bool:
            item_dict['width'] = -1
        item_dict.update(item_kwargs)
        with dpg.group(horizontal=True,parent=parent):
            dpg.add_text(f"{hierarchy[-1]}:",tag=f"{self.prefix}{name}_label")
            #log.debug(f"Creating item {name}")
            creation_func(**item_dict)
            if tooltip != "":
                with dpg.tooltip(name):
                    dpg.add_text(tooltip)

    def add_combo(self,name:str,values:list[str],default:str,
                  save=True,callback=None,
                  node_kwargs:dict = {}, tooltip:str = "", item_kwargs:dict = {}):
        callback = self.get_save_callback(callback)

        if not save:
            self.skip_save.append(name)
        layer_dict = self._make_nodes(name,node_kwargs)
        hierarchy = name.split('/')
        #log.debug(f"Item Hierarchy: {hierarchy}")

        if hierarchy[-1] in layer_dict.keys():
            raise RuntimeError(f"{name} already exists in tree.")

        parent = f"{self.prefix}{'/'.join(hierarchy[:-1])}"
        layer_dict[hierarchy[-1]] = name

        item_dict = {'values':values,
                     'tag' : f"{self.prefix}{name}",
                     'default_value' : default,
                     'callback' : callback,
                    }
        item_dict.update(item_kwargs)
        with dpg.group(horizontal=True,parent=parent):
            dpg.add_text(f"{hierarchy[-1]}:",tag=f"{self.prefix}{name}_label")
            #log.debug(f"Creating item {name}")
            dpg.add_combo(**item_dict)
            if tooltip != "":
                with dpg.tooltip(name):
                    dpg.add_text(tooltip)

    def save(self,filemode='w'):
        with dpg.mutex():
            path = Path(self.savefile)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            values_dict = self.collapse_item_dict(self.dict)
            with path.open(filemode) as f:
                for key,value in values_dict.items():
                    if key not in self.skip_save:
                        if isinstance(value,str):
                            f.write(f"{key},'{value}'\n")
                        else:
                            f.write(f"{key},{value}\n")
    
    def collapse_item_dict(self, d:dict):
        output = {}
        for key, item in d.items():
            if isinstance(item,dict):
                output.update(self.collapse_item_dict(item))
            elif isinstance(item,str):
                output.update({item : self[item]})
            else:
                ValueError(f"Invalid entry found in dict {item}. This shouldn't be possible.")
        return output

    def load(self):
        path = Path(self.savefile)
        if not path.exists():
            return
        with path.open('r') as f:
            for line in f.readlines():
                entries = line.split(',',maxsplit=1)
                entries = [entry.strip() for entry in entries]
                if entries[0] in self.skip_save:
                    continue
                try:
                    self.add(entries[0],literal_eval(entries[1]))
                except RuntimeError:
                    self[entries[0]] = literal_eval(entries[1])
                # If we can't evaluate it, just add it as a string.
                except ValueError:
                    try:
                        self.add(entries[0],entries[1])
                    except RuntimeError:
                        self[entries[0]] = entries[1]

    def get_save_callback(self, user_callback:Callable = None):
        # If no user callback is provided, set the item to simply save the tree
        if user_callback is None:
            return lambda _: self.save()
        # Otherwise, we first call the callback and then save.
        else:
            def save_cb(sender,app_data,user_data):
                user_callback(sender,app_data,user_data)
                self.save()
            return save_cb

# Plot time axes are always in UTC with no way to adjust.
# So we can subtract off the timezone differece before plotting.
# This should only be called when displaying the data in a plot, not for
# saving the actual time stamps.
def offset_timezone(timestamps:list[float]) -> list[float]:
    now = dt.now().timestamp()
    offset = (dt.utcfromtimestamp(now) - dt.fromtimestamp(now)).seconds
    return [timestamp - offset for timestamp in timestamps]

def initialize_dpg(title:str = "Unamed DPG App",docking=False):
    dpg.create_context()
    dpg.configure_app(
        wait_for_input=False, # Can set to true but items may not live update. Lowers CPU usage
        docking=docking
        )
    dpg.create_viewport(title=title, width=1920//2, height=1080//2, x_pos=1920//4, y_pos=1080//4)
    with dpg.font_registry():
        dpg.add_font("X:\DiamondCloud\Personal\Rigel\Scripts\FiraCode-Bold.ttf", 12, default_font=False,tag='small_font')
        dpg.add_font("X:\DiamondCloud\Personal\Rigel\Scripts\FiraCode-Bold.ttf", 18, default_font=True)
        dpg.add_font("X:\DiamondCloud\Personal\Rigel\Scripts\FiraCode-Medium.ttf", 18, default_font=False)
        dpg.add_font("X:\DiamondCloud\Personal\Rigel\Scripts\FiraCode-Regular.ttf", 18, default_font=False)
        dpg.add_font("X:\DiamondCloud\Personal\Rigel\Scripts\FiraCode-Bold.ttf", 22, default_font=False, tag="plot_font")
        dpg.add_font("X:\DiamondCloud\Personal\Rigel\Scripts\FiraCode-Bold.ttf", 48, default_font=False,tag="big_font")
        dpg.add_font("X:\DiamondCloud\Personal\Rigel\Scripts\FiraCode-Bold.ttf", 96, default_font=False,tag="huge_font")
        dpg.add_font("X:\DiamondCloud\Personal\Rigel\Scripts\FiraCode-Bold.ttf", 128, default_font=False,tag="massive_font")

def start_dpg():
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.maximize_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
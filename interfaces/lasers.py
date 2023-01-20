from .interface_template import Interface
from apis import rdpg

from pathlib import Path

import datetime as dt
import logging
import asyncio
log = logging.getLogger(__name__)
import numpy as np
from logging import getLogger
log = getLogger(__name__)

from typing import Union

dpg = rdpg.dpg

laser_config = {'1205nm_laser_ip' : '192.169.1.106'}

class LaserInterface(Interface):

    def __init__(self,set_interfaces,fpga,superk,laser602,treefix="lasers"):
        super().__init__()
        self.fpga = fpga
        self.set_interfaces = set_interfaces
        self.treefix = treefix        
        self.superk = superk
        self.laser602 = laser602
        
        self.toptica_ip = laser_config['1205nm_laser_ip']

        self.toptica_conn = None
        self.toptica_laser = None

        self.superk_laser = None

        self.controls = [f"{self.treefix}_532/Enabled",
                         f"{self.treefix}_532/AOM V",
                         f"{self.treefix}_602/Connected",
                         f"{self.treefix}_602/Firing",
                         f"{self.treefix}_Superk/Connected",
                         f"{self.treefix}_Superk/Firing",
                         f"{self.treefix}_Superk/Power"]
        self.ao_range = self.fpga.get_AO_range(self.fpga._green_aom)
        self.params = []

    def makeGUI(self, parent: Union[str, int]) -> None:
        self.tree = rdpg.TreeDict(self.treefix,f'cryo_gui_settings/{self.treefix}_save.csv')
        # Green AOM Controls
        self.tree.add("532/Enabled",False,save=False,callback=self.toggle_532)
        self.tree.add("532/AOM V",0.0,drag=True, callback = self.set_532,save=False, 
                      item_kwargs={'min_value':self.ao_range[0], 
                                   'max_value':self.ao_range[1],
                                   'clamped':True,
                                   'speed':0.1})

        # Toptica Controls
        self.tree.add("602/Connected",False,save=False,callback=self.init_toptica,
                      tooltip="Open Connection to 1205nm Toptica Laaser")
        self.tree.add("602/Emission Enabled",False,save=False,
                      tooltip="Wether the laser is set to emit, must be physically set on controller",
                      item_kwargs={"enabled":False})
        self.tree.add("602/Firing",False,save=False,callback=self.toggle_toptica,
                      tooltip="Control laser emission by toggling current control")
        
        # Coherent SuperK Controls
        self.tree.add("Superk/Connected",False, save=False,callback=self.init_superk)
        self.tree.add("Superk/Firing", False, save=False, callback=self.toggle_superk)
        self.tree.add("Superk/Power", 0, drag=True, callback=self.set_superk_power,
                      item_kwargs={'min_value':0,'max_value':100,'clamped':True})
        self.tree.add("Superk/Rep. Rate",0,save=False,item_kwargs={'readonly':True,'step':0})

        self.gui_exists = True

    def initialize(self):
        if not self.gui_exists:
            raise RuntimeError("GUI must be made before initialization.")
        self.tree.load()

    def update_callback(self):
        pass

    # 532 Functions
    def toggle_532(self, *args):
        if args[1]:
            self.fpga.set_dio_array([1]+[0]*13,write=True)
            self.tree['532/AOM V'] = self.fpga.get_aoms[1]
        else:
            self.fpga.set_dio_array([0]*14,write=True)

    def set_532(self,*args):
        value = args[1]
        self.fpga.set_aom(green=value,write=True)

    # 602 Functions
    def init_toptica(self,*args):
        if args[1]:
            self.laser602.open()
            self.tree['602/Connected'] = True
            self.tree['602/Emission Enabled'] = self.laser602.emission_button_enabled.get()
        else:
            self.laser602.close()
            self.tree['602/Connected'] = False


    def toggle_toptica(self,*args):
        if args[1]:
            self.laser602.laser1.dl.cc.enabled.set(True)
            self.tree['602/Firing'] = True
        else:
            self.laser602.laser1.dl.cc.enabled.set(False)
            self.tree['602/Firing'] = False

    # SuperK Functions
    def init_superk(self,*args):
        if args[1]:
            self.superk.initialize()
            self.tree['Superk/Connected'] = True
            self.tree['Superk/Power'] = int(self.superk.get_power())
            self.tree['Superk/Firing'] = bool(self.superk.read_power())
            self.tree['Superk/Rep. Rate'] = int(self.superk.get_rep_rate())
        else:
            self.superk.deinitialize()
            self.tree['Superk/Connected'] = False

    def toggle_superk(self,*args):
        if args[1]:
            self.superk.turn_on()
            self.tree['Superk/Firing']
        else:
            self.superk.turn_off()
            self.tree['Superk/Firing']

    def set_superk_power(self,*args):
        self.superk.set_power(args[1])
        self.tree['Superk/Rep. Rate'] = self.superk.get_rep_rate()
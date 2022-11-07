from apis.picoharp import PicoHarp
from interfaces.picoharp import PicoHarpInterface
from apis import rdpg

dpg = rdpg.dpg

rdpg.initialize_dpg("Picoharp",docking=False)
harp = PicoHarp()
pico = PicoHarpInterface(lambda x: None, harp)
with dpg.window(label="Cryocontrol", tag='main_window'):
    pico.makeGUI('main_window')
dpg.set_primary_window('main_window',True)
rdpg.start_dpg()
    
# Cryocontrol
Consolidated code for controlling our microcavity cryostat experiment.

# Organization
```
cryocontrol/
|-- *.py
|-- ana_scripts/
|   |-- *.py
|-- apis/
|   |-- *.py
|   |-- dummy/
|   |   |-- *.py
|-- interfaces/
|   |-- *.py
|-- tests/
    |-- *.py
```

Python scripts in the root folder produce GUIs and scripts for controlling various aspects of the system, 
as well as ones that are useful for viewing saved data.
e.g. the main gui for controlling the whole experiment or one for viewing plots nicely.

`ana_scripts/` contains python scripts that analyze previously saved data.
e.g. plotting some data nicely, or running fits

`apis/` contains libraries of code for controlling hardware or running complex operations.
Basically if you want to import some useful functions for doing something, this is where the script
you're importing from should go. The subdirectory `dummy/` contains simulated versions of real apis
for testing out interface code during the development cycle. 

`tests/` contains files used for testing any other code in the repo. This should produce no
real physical side effects, and only confirm that the logic of the code remains correct between
changes.

`interfaces/` contains script that produce building blocks of guis in the main folder.
This can be simple building blocks, like a wrapper around some gui to make joint plots.
Or more complex parts of the interface, like the confocal code for the main gui.

# Citation
While we have no articles directly related to this software, if you use our code in relation to a publication we request a citation of:

`Y. Fontana, R. Zifkin, E. Janitz, C. D. Rodríguez Rosenblueth, and L. Childress, "A mechanically stable and tunable cryogenic Fabry–Pérot microcavity", Review of Scientific Instruments 92, 053906 (2021) https://doi.org/10.1063/5.0049520`

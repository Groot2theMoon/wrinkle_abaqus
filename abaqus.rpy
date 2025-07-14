# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2025 replay file
# Internal Version: 2024_09_20-22.00.46 RELr427 198590
# Run by user on Thu Jul 10 20:02:23 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.55729, 1.55556), width=229.233, 
    height=154.311)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile(
    'C:/Users/user/Desktop/Seungwon/wrinkle_abaqus/abaqus_workspace/aba_run_parabolic.py', 
    __main__.__dict__)
#: The model "Model" has been created.

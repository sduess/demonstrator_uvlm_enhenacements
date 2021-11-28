import xfoil_interface
import numpy as np

xfi = xfoil_interface.XFoilInterface('Reference', save=False, output_directory='./data_xfoil/')
xfi.add_airfoil('naca0018', '0018')
xfi.reynolds = 500000
xfi.run_polar(-10, 10, 1, n_node=300, save=True)

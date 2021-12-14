import xfoil_interface
import numpy as np
import aircraft
import os

DIRECTORY = os.path.abspath(aircraft.FLEXOP_DIRECTORY)
output_directory = '../src/flex_op/src/airfoil_polars/'

def run_tip_polar():
    xfi = xfoil_interface.XFoilInterface('Reference', save=True, output_directory=output_directory)
    xfi.load_airfoils_from_excel('tip')
    xfi.max_iter = 300
    xfi.reynolds = 1.3e6
    xfi.run_polar(-6, 6, 0.25, n_node=250, xtr=(0.7, 0.02))

def run_root_polar():
    xfi = xfoil_interface.XFoilInterface('Reference', save=True, output_directory=output_directory)
    xfi.load_airfoils_from_excel('root')
    xfi.max_iter = 300
    xfi.reynolds = 1.3e6
    xfi.run_polar(-6, 6, 0.25, n_node=250, xtr=(0.7, 0.02))

def run_tail_polar():
    xfi = xfoil_interface.XFoilInterface('Reference', save=True, output_directory=output_directory)
    xfi.add_airfoil('naca0012', '0012')
    xfi.max_iter = 300
    xfi.reynolds = 1.3e6
    xfi.run_polar(-6, 6, 0.25, n_node=250)


if __name__ == '__main__':
    run_root_polar()
    run_tip_polar()
    run_tail_polar()
    

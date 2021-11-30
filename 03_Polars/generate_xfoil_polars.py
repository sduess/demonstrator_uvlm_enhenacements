import xfoil_interface
import numpy as np

output_directory = '../src/flex_op/src/airfoil_polars/'

# xfi = xfoil_interface.XFoilInterface('Reference', save=True, output_directory=output_directory)
# # xfi.add_airfoil('naca0018', '0018')
# xfi.load_airfoils_from_excel()
# xfi.reynolds = 1.3e6
# xfi.xfoil.xtr = (0.85, 0.02)
# xfi.run_angle_of_attack(-5.75, n_node=250)
# xfi.run_polar(-6, 6, 0.25, n_node=250, save=True)

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


if __name__ == '__main__':
    run_root_polar()
    # run_tip_polar()
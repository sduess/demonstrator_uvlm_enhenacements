import numpy as np
import os
import sys
sys.path.append('../src/xfoil/')
from xfoil import XFoil
from xfoil.model import Airfoil
import unittest
import matplotlib.pyplot as plt
import pandas as pd

flex_op_data_path = '../01_case_files/flexOp_data/'


def read_excel(filepath):
    """

    Args:
        filepath (str): Path to Reference.xls or Tailored.xls file

    Returns:
        dict: Containing root: x, y and tip:x, y data
    """

    out = {}
    outname = ['root', 'tip']
    for isheet, sn in enumerate([6, 7]):
        df = pd.read_excel(filepath, sheet_name=sn, header=2, usecols=[1, 2, 3, 4])

        x_upper, y_upper = interpolate_airfoil_surface(df['XU'], df['YU'], n_points=200)
        x_lower, y_lower = interpolate_airfoil_surface(df['XL'], df['YL'], n_points=200)

        x = np.concatenate((x_lower[::-1], x_upper))
        y = np.concatenate((y_lower[::-1], y_upper))

        # x = np.concatenate((np.array(df['XL'][::-1]), np.array(df['XU'])))
        # y = np.concatenate((np.array(df['YL'][::-1]), np.array(df['YU'])))
        out[outname[isheet]] = (x, y)
    return out


def interpolate_airfoil_surface(x_airfoil, y_airfoil, n_points=100):
    """Upper or lower surface only"""
    x_chord = np.linspace(np.min(x_airfoil), np.max(x_airfoil), n_points)
    x_chord = - np.cos(x_chord ** 2 * np.pi) * 0.5 + 0.5
    order = np.argsort(x_airfoil)
    x_airfoil = x_airfoil[order]
    y_airfoil = y_airfoil[order]
    y_surface = np.interp(x_chord, x_airfoil, y_airfoil)

    return x_chord, y_surface


class XFoilInterface:

    def __init__(self, configuration='Reference', **kwargs):
        self.xfoil = XFoil()
        self.configuration = configuration

        self.airfoil_data = dict() # dict: containing coords as {airfoil_name: (x, y)}

        self._reynolds = None

        self.output_directory = kwargs.get('output_directory', './')
        self.save = kwargs.get('save', False)

    @property
    def reynolds(self):
        """Simulation Reynolds Number"""
        return self._reynolds

    @reynolds.setter
    def reynolds(self, reynolds_number):
        self._reynolds = reynolds_number
        self.xfoil.Re = reynolds_number

    def load_airfoils_from_excel(self):

        if self.configuration.lower() == 'reference':
            filepath = flex_op_data_path + '/Reference.xlsx'
        elif self.configuration.lower() == 'tailored':
            filepath = flex_op_data_path + 'Tailored.xlsx'
        else:
            raise NameError('Unknown configuration {:s}'.format(self.configuration))

        for airfoil_name, airfoil_coords in read_excel(filepath).items():
            self.add_airfoil(airfoil_name, airfoil_coords)

    def add_airfoil(self, name, xy_tuple_or_str):
        """

        Args:
            name (str): Airfoil name
            x_coords (np.array): x-coordinates starting from TE
            y_coords (np.array): y-coordinates starting from TE

        """
        self.airfoil_data[name] = xy_tuple_or_str

    def reload_xfoil_airfoil(self, airfoil_entry, n_nodes=160):
        if type(airfoil_entry) is str:
            self.xfoil.naca(airfoil_entry)
        else:
            self.xfoil.airfoil = Airfoil(*airfoil_entry)
        self.xfoil.repanel(n_nodes)
        self.xfoil.reset_bls()

    def run_angle_of_attack(self, alpha, **kwargs):
        """

        Args:
            alpha (float): Angle of attack in degrees

        Returns:

        """

        for airfoil_name, airfoil_coords in self.airfoil_data.items():
            self.reload_xfoil_airfoil(airfoil_coords, kwargs.get('n_node', 160))

            cl, cd, cm, cp = self.xfoil.a(alpha)
            print('Airfoil', airfoil_name)
            print('alpha, cl, cd, cm, cpmin')
            print(alpha, cl, cd, cm, cp)
            if self.save:
                filename = self.output_directory + f'/xfoil_re{self.reynolds:g}_{airfoil_name}_alpha{alpha:04g}.txt'
                np.savetxt(filename,
                           np.column_stack((cl, cd, cm, cp)),
                           header='cl, cd, cm, cpmin')
                print(f'Saved airfoil {airfoil_name} data to:\n\t{filename}')

    def run_sequence_angle_of_attack(self, alpha_init, alpha_end, alpha_step, **kwargs):
        out_data = {}
        for airfoil_name, airfoil_coords in self.airfoil_data.items():
            self.reload_xfoil_airfoil(airfoil_coords, kwargs.get('n_node', 160))

            alpha_deg, cl, cd, cm, cp = self.xfoil.aseq(alpha_init, alpha_end, alpha_step)
            alpha_rad = alpha_deg * np.pi / 180
            out_data[airfoil_name] = np.column_stack((alpha_rad, cl, cd, cm, cp))

            if self.save or kwargs.get('save', False):
                filename = self.output_directory + f'/xfoil_seq_re{self.reynolds:g}_{airfoil_name}.txt'
                np.savetxt(filename,
                           out_data[airfoil_name],
                           header='alpha_rad, cl, cd, cm, cpmin')
                print(f'Saved airfoil {airfoil_name} data to:\n\t{filename}')

        return out_data

    def run_polar(self, alpha_min, alpha_max, alpha_increment, **kwargs):
        n = int(np.ceil((alpha_max - alpha_min) / alpha_increment) + 1)
        alpha_dom = np.linspace(alpha_min, alpha_max, n)

        out_data = {}
        for airfoil_name, airfoil_coords in self.airfoil_data.items():
            self.reload_xfoil_airfoil(airfoil_coords, kwargs.get('n_node', 160))

            polar_data = np.zeros((n, 4))
            for i_alpha in range(n):
                cl, cd, cm, _ = self.xfoil.a(alpha_dom[i_alpha])
                polar_data[i_alpha] = np.array([alpha_dom[i_alpha], cl, cd, cm])
                self.xfoil.reset_bls()

            out_data[airfoil_name] = polar_data

            if self.save or kwargs.get('save', False):
                filename = self.output_directory + f'/xfoil_seq_re{self.reynolds:g}_{airfoil_name}.txt'
                np.savetxt(filename,
                           out_data[airfoil_name],
                           header='alpha_deg, cl, cd, cm')
                print(f'Saved airfoil {airfoil_name} data to:\n\t{filename}')

        return out_data


def plot_polar(alpha_rad, cl, cd, cm, **kwargs):
    fig, ax = plt.subplots(ncols=3, **kwargs)

    alpha_deg = alpha_rad * 180 / np.pi

    ax[0].plot(alpha_deg, cl)
    ax[1].plot(alpha_deg, cd)
    ax[2].plot(alpha_deg, cm)

    for a in ax:
        a.set_xlabel('Angle of Attack, deg')
    ax[0].set_ylabel('Lift Coefficient, Cl')
    ax[1].set_ylabel('Drag Coefficient, Cd')
    ax[2].set_ylabel('Pitching Moment Coefficient, Cm')
    plt.tight_layout()

    return fig, ax


class TestXFoil(unittest.TestCase):

    xf = XFoil()

    def test_load_airfoil(self):

        airfoil_coordinates = read_excel(flex_op_data_path + 'Reference.xlsx')
        for name, coords in airfoil_coordinates.items():
            print(f'Airfoil {name}')
            self.xf.airfoil = Airfoil(coords[0], coords[1])
            self.xf.repanel(n_nodes=300)
            self.xf.reset_bls()

            plt.plot(coords[0], coords[1], label=f'source {name}', lw=0.5, color='k')
            plt.plot(self.xf.airfoil.x, self.xf.airfoil.y, label=f'xfoil {name}', lw=0.5, ls='--', ms=1, marker='o')
            plt.axis('equal')
            plt.legend()
            plt.savefig(f'./data_xfoil/airfoils_{name}.pdf')
            np.savetxt(f'./data_xfoil/{name}.dat', np.column_stack((coords[0], coords[1])))

    def test_run(self):
        xfi = XFoilInterface('Reference', save=False, output_directory='./data_xfoil/')
        xfi.load_airfoils_from_excel()
        xfi.reynolds = 500000

        # out_data = xfi.run_sequence_angle_of_attack(-5, 5, 1, save=True)
        out_data = xfi.run_polar(-3, 5, 1, save=True, n_nodes=300)
        for airfoil_name, data in out_data.items():
            fig, ax = plot_polar(data[:, 0] * np.pi / 180, data[:, 1], data[:, 2], data[:, 3])
            fig.savefig(f'{xfi.output_directory}/polar{airfoil_name}.pdf')

        # root airfoil data
        real_xfoil_file = './data_xfoil/xf_re500000_mac.dat'
        data = np.loadtxt(real_xfoil_file, skiprows=12)
        np.testing.assert_allclose(data[:, 0], out_data['root'][:, 0], err_msg='AoA not equal in both arrays. Check order')

        np.testing.assert_allclose(data[:, 1], out_data['root'][:, 1], err_msg='Cl not equal to real xfoil',
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(data[:, 2], out_data['root'][:, 2], err_msg='Cd not equal to real xfoil',
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(data[:, 4], out_data['root'][:, 3], err_msg='Cm not equal to real xfoil',
                                   atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()

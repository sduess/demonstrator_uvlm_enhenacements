import numpy as np
import glob
import configobj
from generate_flexop_polars import generate_polars_case_name, aseq
from aero import area_ref, chord_main_root
import os

# CONSTANTS
S = area_ref
c_ref = chord_main_root

def process_case(path_to_case):
    case_name = path_to_case.split('/')[-1]
    path_to_sharpy_pmor = path_to_case + f'/{case_name}.pmor.sharpy'
    if not os.path.exists(path_to_sharpy_pmor):
        raise FileNotFoundError
    pmor = configobj.ConfigObj(path_to_sharpy_pmor)
    alpha = pmor['parameters']['alpha']
    inertial_forces = np.loadtxt(f'{path_to_case}/forces/forces_aeroforces.txt',
                                 skiprows=1, delimiter=',', dtype=float)[1:4]
    inertial_moments = np.loadtxt(f'{path_to_case}/forces/moments_aeroforces.txt',
                                  skiprows=1, delimiter=',', dtype=float)[1:4]

    return alpha, inertial_forces[2], inertial_forces[0], inertial_moments[1]


def postprocess(output_folder):
    cases = glob.glob(output_folder + '/*')

    n_cases = 0
    for case in cases:
        print(f'Processing case {n_cases + 1} of {len(cases)}')
        try:
            alpha, lift, drag, moment = process_case(case)
        except FileNotFoundError:
            continue
        if n_cases == 0:
            results = np.array([alpha, lift, drag, moment], dtype=float)
        else:
            results = np.vstack((results, np.array([alpha, lift, drag, moment])))
        n_cases += 1
    try:
        results
    except NameError:
        raise FileNotFoundError(f'Unable to find result cases at {output_folder}')
    results = results.astype(float)
    results = results[results[:, 0].argsort()]
    return results


def save_results(results, fname):
    np.savetxt(fname + '.dat', results, header='AoA deg, Cl, Cd, Cm')
    print(f'Saved Results to {os.path.abspath(fname)}.dat')


def run(results_output_directory, toplevel_output_dir, case_base_name, u_inf, use_polars, use_fuselage):

    cases_folder, _ = generate_polars_case_name(case_base_name, u_inf, 0, use_polars, use_fuselage)

    output_directory = toplevel_output_dir + '/' + cases_folder

    results = postprocess(output_directory)

    results = apply_coefficients(results, 0.5 * 1.1336 * u_inf ** 2, S, c_ref)

    if not os.path.isdir(results_output_directory):
        os.makedirs(results_output_directory)

    save_results(results, results_output_directory + '/' + cases_folder)


def apply_coefficients(results, q, S, c_ref):
    qS = q * S
    results[:, 1:] /= qS  # lift drag and moment
    results[:, 3] /= c_ref  # moment only

    return results


if __name__ == '__main__':
    results_output_directory = './polar_output_data/'
    sharpy_output_directory = './output/'

    case_base_name = 'flexop_w10n1'
    u_inf = 45
    use_polars = True
    use_fuselage = False

    run(results_output_directory, sharpy_output_directory, case_base_name, u_inf, use_polars, use_fuselage)

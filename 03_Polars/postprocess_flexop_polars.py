import numpy as np
import glob
import configobj
from generate_flexop_polars import generate_polars_case_name, aseq
from aero import area_ref, chord_main_root
import os
import aircraft
import sharpy.utils.algebra as algebra

# CONSTANTS
S = area_ref
c_ref = chord_main_root


def get_pmor_data(path_to_case):
    case_name = path_to_case.split('/')[-1]
    path_to_sharpy_pmor = path_to_case + f'/{case_name}.pmor.sharpy'
    if not os.path.exists(path_to_sharpy_pmor):
        raise FileNotFoundError
    pmor = configobj.ConfigObj(path_to_sharpy_pmor)

    return case_name, path_to_sharpy_pmor, pmor


def process_case(path_to_case):
    case_name, path_to_sharpy_pmor, pmor = get_pmor_data(path_to_case)
    alpha = pmor['parameters']['alpha']
    inertial_forces = np.loadtxt(f'{path_to_case}/forces/forces_aeroforces.txt',
                                 skiprows=1, delimiter=',', dtype=float)[1:4]
    inertial_moments = np.loadtxt(f'{path_to_case}/forces/moments_aeroforces.txt',
                                  skiprows=1, delimiter=',', dtype=float)[1:4]

    return alpha, inertial_forces[2], inertial_forces[0], inertial_moments[1]


def process_wingtip_deflections(path_to_case):
    case_name, path_to_sharpy_pmor, pmor = get_pmor_data(path_to_case)
    alpha = float(pmor['parameters']['alpha'])
    wintip_node = len(glob.glob(f'{path_to_case}/WriteVariablesTime/*_pos*.dat')) - 2
    with open(f'{path_to_case}/WriteVariablesTime/struct_pos_node{wintip_node}.dat', 'r') as fid:
        out_str = fid.readline()
        deflection = np.array(out_str.split(',')[1:-1], dtype=float)

    # change deflection to g
    cga = algebra.quat2rotation(algebra.euler2quat(np.array([0, alpha * np.pi / 180, 0])))
    deflection = cga.dot(deflection)

    return alpha, deflection


def postprocess(output_folder):
    cases = glob.glob(output_folder + '/*')

    n_cases = 0
    for case in cases:
        print(f'Processing case {n_cases + 1} of {len(cases)}')
        try:
            alpha, lift, drag, moment = process_case(case)
            _, deflection = process_wingtip_deflections(case)
        except FileNotFoundError:
            continue
        if n_cases == 0:
            results = np.array([alpha, lift, drag, moment], dtype=float)
            deflection_results = np.concatenate((np.array([alpha]), deflection))
        else:
            results = np.vstack((results, np.array([alpha, lift, drag, moment])))
            deflection_results = np.vstack((deflection_results, np.concatenate((np.array([alpha]), deflection))))
        n_cases += 1
    try:
        results
    except NameError:
        raise FileNotFoundError(f'Unable to find result cases at {output_folder}')
    results = results.astype(float)
    results = results[results[:, 0].argsort()]
    deflection_results = deflection_results.astype(float)
    deflection_results = deflection_results[deflection_results[:, 0].argsort()]
    return results, deflection_results


def save_results(results, fname):
    np.savetxt(fname + '.dat', results, header='AoA deg, Cl, Cd, Cm')
    print(f'Saved Results to {os.path.abspath(fname)}.dat')


def save_deflection_results(results, fname):
    np.savetxt(fname + '.dat', results, header='AoA deg, xG, yG, zG')
    print(f'Saved Results to {os.path.abspath(fname)}.dat')


def run(results_output_directory, toplevel_output_dir, case_base_name, u_inf, use_polars, use_fuselage):

    cases_folder, _ = generate_polars_case_name(case_base_name, u_inf, 0, use_polars, use_fuselage)

    output_directory = toplevel_output_dir + '/' + cases_folder

    results, deflection_results = postprocess(output_directory)

    results = apply_coefficients(results, 0.5 * 1.1336 * u_inf ** 2, S, c_ref)

    if not os.path.isdir(results_output_directory):
        os.makedirs(results_output_directory)

    save_results(results, results_output_directory + '/' + cases_folder)
    save_deflection_results(deflection_results, results_output_directory + '/' + 'wingtip_deflection_' + cases_folder)

    create_git_info_file(results_output_directory + '/' + 'model_info_' + cases_folder + '.txt')


def create_git_info_file(filename):
    msg = aircraft.print_git_status()
    with open(filename, 'w') as fid:
        fid.write(msg)


def apply_coefficients(results, q, S, c_ref):
    qS = q * S
    results[:, 1:] /= qS  # lift drag and moment
    results[:, 3] /= c_ref  # moment only

    return results


if __name__ == '__main__':
    results_output_directory = './polar_output_data/'
    sharpy_output_directory = './output/'

    case_base_name = 'flexop_w20n1'
    u_inf = 45
    use_polars = False
    use_fuselage = False

    run(results_output_directory, sharpy_output_directory, case_base_name, u_inf, use_polars, use_fuselage)

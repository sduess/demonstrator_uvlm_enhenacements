import numpy as np
import os
import aircraft
import sharpy.utils.algebra as algebra
import sharpy.sharpy_main as smain

airfoil_polars_directory = os.path.abspath('../src/flex_op/src/airfoil_polars/')


def create_git_info_file(filename):
    msg = aircraft.print_git_status()
    with open(filename, 'w') as fid:
        fid.write(msg)


def generate_aircraft(alpha, u_inf, m, flow, case_name, case_route, output_directory, **kwargs):
    """

    Args:
        alpha: radians
        u_inf:
        m: panels
        flow:
        case_name:
        case_route:
        output_directory:
        **kwargs:

    Returns:

    """
    # List of all implemented kwargs
    use_fuselage = kwargs.get('use_fuselage', False)
    use_polars = kwargs.get('use_polars', False)
    elevator = kwargs.get('elevator', 0.)
    rudder = kwargs.get('rudder', 0.)
    thrust = kwargs.get('thrust', 0.)
    roll = kwargs.get('roll', 0)
    yaw = kwargs.get('yaw', 0)
    rho = kwargs.get('rho', 1.225)
    tolerance = kwargs.get('tolerance', 1e-4)
    wake_length = kwargs.get('wake_length', 10)
    gravity = kwargs.get('gravity', True)
    n_load_step = kwargs.get('n_load_step', 1)  # for static coupled
    fsi_tolerance = kwargs.get('fsi_tolerance', 1e-6)
    structural_relaxation_factor = kwargs.get('structural_relaxation_factor', 0.6)
    overwrite = kwargs.get('overwrite', False)  # overwrite over existing case files (if same name)
    # end of kwargs

    case_route = case_route + '/' + case_name
    print(f'Generating case {case_name} with cases route location at\n\t{case_route}')

    if not os.path.isdir(case_route):
        os.makedirs(case_route)
    else:
        if not overwrite:
            msg = input(f'Case {case_name} files already exist at\n\t{case_route}, wish to overwrite? [y/n]')
            if msg.lower() == 'y':
                pass
            else:
                raise FileExistsError('Case files exist - exiting simulation')

    acf = aircraft.FLEXOP(case_name, case_route, output_directory)

    acf.clean()

    acf.init_structure(**kwargs)
    acf.init_aero(m, **kwargs)  # polar array goes here as polars=data kwarg

    if use_fuselage:
        acf.init_fuselage(m, **kwargs)

    acf.set_flight_controls(thrust, elevator, rudder)

    acf.generate()

    # Parameter variable shorthand
    chord_root = acf.aero.chord_main_root
    chord_tip = acf.aero.chord_main_tip

    # Derived parameters
    dt = chord_root / m / u_inf

    settings = dict()
    settings['SHARPy'] = {'case': case_name,
                          'route': acf.case_route,
                          'flow': flow,
                          'write_screen': 'on',
                          'write_log': 'on',
                          'log_folder': output_directory,
                          'log_file': acf.case_name + '.log'}

    settings['BeamLoader'] = {'unsteady': 'on',
                              'orientation': algebra.euler2quat(np.array([roll,
                                                                          alpha,
                                                                          yaw]))}

    settings['AerogridLoader'] = {'unsteady': 'on',
                                'aligned_grid': 'on',
                                'mstar': int(wake_length * m),
                                'wake_shape_generator': 'StraightWake',
                                'wake_shape_generator_input': {
                                    'u_inf': u_inf,
                                    'u_inf_direction': [1., 0., 0.],
                                    'dt': dt,
                                },
                              }

    # settings['AerogridLoader'] = {'unsteady': 'on',
    #                               'aligned_grid': 'on',
    #                               'mstar': int((1 * chord_root) / (chord_tip / m)) + 9,
    #                               'wake_shape_generator': 'StraightWake',
    #                               'wake_shape_generator_input': {
    #                                   'u_inf': u_inf,
    #                                   'u_inf_direction': [1., 0., 0.],
    #                                   'dt': dt,
    #                                   'dx1': chord_root / m,  # size of first wake panel. Here equal to bound panel
    #                                   'ndx1': int((1 * chord_root) / (chord_tip / m)),
    #                                   'r': 1.5,
    #                                   'dxmax': 5 * chord_tip}
    #                               }
    # print("Number of panels with size ``dx1`` = ", int((1 * chord_root) / (chord_tip / m)) + 9)

    settings['NonliftingbodygridLoader'] = {'unsteady': 'on',
                                            'aligned_grid': 'on',
                                            'freestream_dir': ['1', '0', '0']}

    settings['NonLinearStatic'] = {'print_info': 'off',
                                   'max_iterations': 150,
                                   'num_load_steps': 1,
                                   'delta_curved': 1e-1,
                                   'min_delta': tolerance,
                                   'gravity_on': gravity,
                                   'gravity': 9.81}

    settings['StaticUvlm'] = {'print_info': 'on',
                              'horseshoe': 'off',
                              'num_cores': 4,
                              'velocity_field_generator': 'SteadyVelocityField',
                              'velocity_field_input': {'u_inf': u_inf,
                                                       'u_inf_direction': [1., 0, 0]},
                              'rho': rho,
                              'nonlifting_body_interaction': use_fuselage}

    settings['StaticCoupled'] = {'print_info': 'off',
                                 'structural_solver': 'NonLinearStatic',
                                 'structural_solver_settings': settings['NonLinearStatic'],
                                 'aero_solver': 'StaticUvlm',
                                 'aero_solver_settings': settings['StaticUvlm'],
                                 'max_iter': 100,
                                 'n_load_steps': n_load_step,
                                 'tolerance': fsi_tolerance,
                                 'relaxation_factor': structural_relaxation_factor}

    if use_polars:
        settings['StaticCoupled']['correct_forces_method'] = 'PolarCorrection'
        settings['StaticCoupled']['correct_forces_settings'] = {'cd_from_cl': 'off',
                                                                'correct_lift': 'on',
                                                                'moment_from_polar': 'on'}

    settings['StaticTrim'] = {'solver': 'StaticCoupled',
                              'solver_settings': settings['StaticCoupled'],
                              'initial_alpha': alpha,
                              'initial_deflection': elevator,
                              'initial_thrust': thrust,
                              'tail_cs_index': 4,
                              'fz_tolerance': 0.01,
                              'fy_tolerance': 0.01,
                              'm_tolerance': 0.01,
                              'initial_angle_eps': 0.05,
                              'initial_thrust_eps': 2.,
                              'save_info': True}

    settings['LiftDistribution'] = {}

    settings['BeamLoads'] = {'csv_output': True}

    settings['BeamPlot'] = {'include_rbm': 'on',
                            'include_applied_forces': 'on'}

    settings['AerogridPlot'] = {'include_rbm': 'on',
                                'include_forward_motion': 'off',
                                'include_applied_forces': 'on',
                                'plot_nonlifting_surfaces': 'off',
                                'minus_m_star': 0,
                                'u_inf': u_inf,
                                'dt': dt}

    settings['Modal'] = {'print_info': True,
                         'use_undamped_modes': True,
                         'NumLambda': 30,
                         'rigid_body_modes': True,
                         'write_modes_vtk': 'on',
                         'print_matrices': 'on',
                         'write_data': 'on',
                         'continuous_eigenvalues': 'off',
                         'dt': dt,
                         'plot_eigenvalues': False}

    settings['AeroForcesCalculator'] = {
        'write_text_file': 'on',
        'coefficients': 'off',
        'S_ref': acf.reference_area,
        'q_ref': 0.5 * rho * u_inf ** 2}

    settings['WriteVariablesTime'] = {'structure_variables': ['pos', 'psi'],
                                      'structure_nodes': list(range(acf.structure.n_node_main + 1)),
                                      'cleanup_old_solution': 'on',
                                      'delimiter': ','}

    settings['SaveParametricCase'] = {'parameters': {'alpha': alpha * 180 / np.pi,
                                                     'u_inf': u_inf}}

    acf.create_settings(settings)

    create_git_info_file(acf.case_route + '/' + 'flexop_model_info_' + acf.case_name + '.txt')

    smain.main(['', acf.case_route + '/' + acf.case_name + '.sharpy'])
    create_git_info_file(acf.output_route + '/' + acf.case_name + '/' + 'flexop_model_info_' + acf.case_name + '.txt')


def generate_case_name(case_name_base, u_inf, alpha_rad, use_polars, use_fuselage):
    """Alpha in radians - to be used for an individual case"""
    alpha_str = alpha_str_from_rad(alpha_rad)
    case_name = case_name_base + f'_uinf{u_inf * 10:04g}_alpha{alpha_str}_p{use_polars:g}_f{use_fuselage:g}'

    return case_name


def alpha_str_from_rad(alpha_rad):
    alpha_deg = alpha_rad * 180 / np.pi
    if alpha_deg < 0:
        alpha_str = f'M{np.abs(alpha_deg) * 100:03g}'
    else:
        alpha_str = f'{alpha_deg * 100:04g}'
    return alpha_str


def generate_polars_case_name(case_name_base, u_inf, alpha_rad, use_polars, use_fuselage):
    """
    Generates a cases_folder (where all cases can be grouped by top-level settings) and
    a case_name (which just contains the angle of attack info)

    Args:
        case_name_base:
        u_inf:
        alpha_rad:
        use_polars:
        use_fuselage:

    Returns:

    """
    alpha_str = alpha_str_from_rad(alpha_rad)
    cases_folder = f'seq_{case_name_base}_uinf{u_inf * 10:04g}_p{use_polars:g}_f{use_fuselage:g}'
    case_name = f'{case_name_base}_alpha{alpha_str}'

    return cases_folder, case_name


def aseq(alpha_start, alpha_end, alpha_step):
    """Input in DEGREES
    Returns:
        np.array: alpha domain in RADIANS
    """
    n = int(np.ceil((alpha_end - alpha_start) / alpha_step) + 1)
    alpha_dom = np.linspace(alpha_start, alpha_end, n) * np.pi / 180

    return alpha_dom


def generate_polar_arrays(airfoils):
    # airfoils = {name: filename}
    # Return a aoa (rad), cl, cd, cm for each airfoil
    out_data = [None] * len(airfoils)
    for airfoil_index, airfoil_filename in airfoils.items():
        out_data[airfoil_index] = np.loadtxt(airfoil_filename, skiprows=12)[:, :4]
        if any(out_data[airfoil_index][:, 0] > 1):
            # provided polar is in degrees so changing it to radians
            out_data[airfoil_index][:, 0] *= np.pi / 180
    return out_data


def create_polars_sequence(alpha_start, alpha_end, alpha_step,
                           u_inf, m, flow, case_base_name,
                           cases_route, output_route, simulation_settings):

    for i_alpha, alpha in enumerate(aseq(alpha_start, alpha_end, alpha_step)):
        cases_folder_name, case_name = generate_polars_case_name(case_base_name, u_inf, alpha,
                                                                 simulation_settings['use_polars'],
                                                                 simulation_settings['use_fuselage'])

        generate_aircraft(alpha,
                          u_inf,
                          m,
                          flow,
                          case_name,
                          case_route=cases_route + '/' + cases_folder_name,
                          output_directory=output_route + '/' + cases_folder_name,
                          **simulation_settings
                          )


def main():
    u_inf = 45
    alpha_deg = -0.2
    rho = 1.1336
    run_single_case = True
    alpha_start = -5
    alpha_end = 10
    alpha_step = 1

    case_base_name = 'flexop_'
    cases_route = './cases/'
    output_route = './output/'

    use_polars = True
    use_fuselage = False

    # numerics
    m = 8
    wake_length = 10
    sigma = 1
    n_elem_multiplier = 1
    n_load_step = 5
    case_base_name += f'w{wake_length:02g}n{n_elem_multiplier}'

    airfoil_polars = {
        0: airfoil_polars_directory + '/xfoil_seq_re1300000_root.txt',
        1: airfoil_polars_directory + '/xfoil_seq_re1300000_naca0012.txt',  # unused
        2: airfoil_polars_directory + '/xfoil_seq_re1300000_naca0012.txt',
                      }

    flow = ['BeamLoader',
            'AerogridLoader',
            'StaticCoupled',
            # 'StaticUvlm',
            'AeroForcesCalculator',
            'WriteVariablesTime',
            'BeamPlot',
            'AerogridPlot',
            'SaveParametricCase'
            ]

    simulation_settings = {
                           'use_fuselage': use_fuselage,
                           'use_polars': use_polars,
                           'overwrite': True,
                           'sigma': sigma,
                           'wake_length': wake_length,
                           'rho': rho,
                           'n_load_step': n_load_step,
                           'n_elem_multiplier': n_elem_multiplier,
                           'polars': generate_polar_arrays(airfoil_polars)}

    if run_single_case:
        alpha_rad = alpha_deg * np.pi / 180
        case_name = generate_case_name(case_base_name, u_inf, alpha_rad, use_polars, use_fuselage)
        generate_aircraft(alpha_rad,
                          u_inf,
                          m,
                          flow,
                          case_name,
                          case_route=cases_route,
                          output_directory=output_route,
                          **simulation_settings
                          )
    else:
        create_polars_sequence(alpha_start, alpha_end, alpha_step,
                               u_inf, m, flow, case_base_name,
                               cases_route,
                               output_route,
                               simulation_settings)


if __name__ == '__main__':
    main()

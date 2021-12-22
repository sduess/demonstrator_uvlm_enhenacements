import os
import aircraft
import numpy as np
import sharpy.utils.algebra as algebra


cases_route = '../01_case_files/'
output_route = './output/'

# case_name = 'test_script_1g_only_diagnos_no_first_moment'
# case_name = 'test_dynamic_wake_20_l5_i5_corrected_material'

# case_name = 'final_1g_nonlifting_test' #_nonlifting' #5g_nonlifting'
# case_name = 'final_dynamic_v30_first_moment_-+'
case_name = 'full_aircraft_stiff_hale_u20_l120_i40_w10_nolumped'
lifting_only = False #ignore nonlifting bodies
wing_only = lifting_only
dynamic = True #False #True


flexop_model = aircraft.FLEXOP(case_name, cases_route, output_route)
flexop_model.clean()
flexop_model.init_structure(sigma=1, n_elem_multiplier=2, n_elem_multiplier_fuselage = 1, lifting_only=lifting_only, wing_only = wing_only)  # 4
flexop_model.init_aero(m=8) #16

flow = ['BeamLoader', 
        'AerogridLoader',
        'NonliftingbodygridLoader',
        # 'StaticUvlm',
        # 'AeroForcesCalculator',
        'StaticCoupled',
        'LiftDistribution',
        'BeamLoads',
        'BeamPlot',
        'AerogridPlot',
        # 'AeroForcesCalculator',
        # 'SaveData',
        # # 'Modal',
        # # 'LinearAssembler',
        # # 'AsymptoticStability',
        ]
if not lifting_only:
    flexop_model.init_fuselage(m=12)
    nonlifting_body_interactions = True
else:
    flow.remove("NonliftingbodygridLoader")
    nonlifting_body_interactions = False
print(flow)
# flow = ['BeamLoader',
#         'AerogridLoader',
#         'Modal',
#         'BeamPlot'
#         ]
flexop_model.generate()

# Set cruise parameter
alpha = np.deg2rad(9.5)# np.deg2rad(-0.4) #309412669256394) #
alpha = np.deg2rad(-0.47) #1gnp.deg2rad(-0.1383) #
u_inf = 20 #30 #45 #45 
rho = 1.1336 # altitude = 800  
gravity = True
horseshoe =  not dynamic # True #False # False #True
wake_length = 10
cfl1 = True 
free_flight = False
# m_takeoff = 65

# Other parameters
CFL = 1
dt = CFL * flexop_model.aero.chord_main_root / flexop_model.aero.m / u_inf
# numerics
n_step = 5
structural_relaxation_factor = 0.6
relaxation_factor = 0.35
tolerance = 1e-6
fsi_tolerance = 1e-4
num_cores = 6
n_tstep = 5000 # Later 

# Gust velocitz field
relative_motion = True
chord_root = 0.471
gust_length = 120 #*u_inf
gust_intensity = 0.4
gust_offset = 0.*chord_root


settings = {}
settings['SHARPy'] = {'case': flexop_model.case_name,
                      'route': flexop_model.case_route,
                      'flow': flow,
                      'write_screen': 'on',
                      'write_log': 'on',
                      'log_folder': flexop_model.output_route,
                      'log_file': flexop_model.case_name + '.log'}


settings['BeamLoader'] = {'unsteady': 'on',
                              'orientation': algebra.euler2quat(np.array([0.,
                                                                          alpha,
                                                                          0.]))}


settings['LiftDistribution'] = {'q_ref': 0.5*rho*u_inf**2,
                                'coefficients': True}
settings['NonLinearStatic'] = {'print_info': 'off',
                               'max_iterations': 150,
                               'num_load_steps': 1,
                               'delta_curved': 1e-1,
                               'min_delta': tolerance,
                               'gravity_on': gravity,
                               'gravity': 9.81}

settings['StaticUvlm'] = {'print_info': 'on',
                          'horseshoe': horseshoe,
                          'num_cores': num_cores,
                          'velocity_field_generator': 'SteadyVelocityField',
                          'velocity_field_input': {'u_inf': u_inf,
                                                   'u_inf_direction': [1., 0, 0]},
                          'rho': rho,
                          'nonlifting_body_interaction': nonlifting_body_interactions}

settings['StaticCoupled'] = {'print_info': 'off',
                             'structural_solver': 'NonLinearStatic',
                             'structural_solver_settings': settings['NonLinearStatic'],
                             'aero_solver': 'StaticUvlm',
                             'aero_solver_settings': settings['StaticUvlm'],
                             'max_iter': 100,
                             'n_load_steps': n_step,
                             'tolerance': fsi_tolerance,
                             'relaxation_factor': structural_relaxation_factor}

settings['AerogridLoader'] = {'unsteady': 'on',
                            'aligned_grid': 'on',
                            'mstar': wake_length*flexop_model.aero.m, #int(20/tstep_factor),
                            'wake_shape_generator': 'StraightWake',
                            'wake_shape_generator_input': {
                                'u_inf': u_inf,
                                'u_inf_direction': [1., 0., 0.],
                                'dt': dt,
                            },
                        }
if horseshoe:
    settings['AerogridLoader']['mstar'] = 1
settings['NonliftingbodygridLoader'] = {'freestream_dir': ['1', '0', '0']}

settings['Modal'] = {'NumLambda': 20,
                    'rigid_body_modes': False,
                    'print_matrices': True,
                    'write_dat': True,
                    'continuous_eigenvalues': False,
                    'dt': 0,
                    # 'max_rotation_deg': 15.,
                    # 'max_displacement': 0.15,
                    'write_modes_vtk': True,
                    'use_undamped_modes': True}
settings['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                'linear_system_settings': {
                                    'beam_settings': {'modal_projection': True,
                                                    'inout_coords': 'nodes',
                                                    'discrete_time': True,
                                                    'newmark_damp': 0.5e-4,
                                                    'discr_method': 'newmark',
                                                    'dt': dt,
                                                    'proj_modes': 'undamped',
                                                    'use_euler': 'off',
                                                    'num_modes': 20,
                                                    'print_info': 'on',
                                                    'gravity': 'on',
                                                    'remove_dofs': []},
                                    'aero_settings': {'dt': dt,
                                                    'integr_order': 2,
                                                    'density': rho,
                                                    'integr_order': 2,
                                                    'ScalingDict': {'length': 0.5*chord_root,
                                                                    'speed': u_inf,
                                                                    'density': rho}},
                                                    }}

settings['AsymptoticStability'] = {'print_info': 'on',
                                   'velocity_analysis': [45,70, 15]}
settings['BeamPlot'] = {}

settings['AerogridPlot'] = {'include_rbm': 'off',
                            'include_applied_forces': 'on',
                            'minus_m_star': 0,
                            'u_inf': u_inf,
                            'plot_nonlifting_surfaces':  nonlifting_body_interactions
                            }

settings['AeroForcesCalculator'] = {'coefficients': False}

settings['BeamLoads'] = {'csv_output': True}
settings['SaveData'] = {'save_aero': True,
                        'save_struct': True}   

if dynamic:
    flow = ['BeamLoader', 
            'AerogridLoader',
            'NonliftingbodygridLoader',
            'StaticCoupled',
            'BeamPlot',
            'AerogridPlot',
            'AeroForcesCalculator',
            'DynamicCoupled'
            ]
    if lifting_only:
        flow.remove('NonliftingbodygridLoader')
                  
    settings['SHARPy']['flow'] = flow      
    settings['StepUvlm'] = {'num_cores': num_cores,                     
                            'convection_scheme': 2,
                            'gamma_dot_filtering': 0,
                            'cfl1': cfl1,
                            'velocity_field_generator': 'GustVelocityField',
                            'velocity_field_input': {'u_inf': u_inf,
                                                    'u_inf_direction': [1., 0, 0],
                                                    'gust_shape': '1-cos',
                                                    'relative_motion': relative_motion,
                                                    'gust_parameters': {'gust_length': gust_length,
                                                                        'gust_intensity': gust_intensity*u_inf,
                                                                        }
                                                    },
                            'rho': rho,
                            'n_time_steps': n_tstep,
                            'dt': dt,
                            'nonlifting_body_interaction': nonlifting_body_interactions,}
    settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'off',
                                        'max_iterations': 950,
                                        'delta_curved': 1e-1,
                                        'min_delta': tolerance,
                                        'newmark_damp': 1e-4,
                                        'gravity_on': gravity,
                                        'gravity': 9.81,
                                        'num_steps': n_tstep,
                                        'dt': dt}
    if free_flight:
        solver = 'NonLinearDynamicCoupledStep'
    else:
        solver = 'NonLinearDynamicPrescribedStep'
    settings['DynamicCoupled'] = {'structural_solver': solver,
                                'structural_solver_settings': settings[solver],
                                'aero_solver': 'StepUvlm',
                                'aero_solver_settings': settings['StepUvlm'],
                                'fsi_substeps': 200,
                                'fsi_tolerance': fsi_tolerance,
                                'relaxation_factor': relaxation_factor,
                                'minimum_steps': 1,
                                'relaxation_steps': 150,
                                'final_relaxation_factor': 0.05,
                                'n_time_steps': n_tstep,
                                'dt': dt,
                                'nonlifting_body_interaction': nonlifting_body_interactions,
                                'include_unsteady_force_contribution': 'on',
                                'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot', 'SaveData'],
                                'postprocessors_settings': {'BeamLoads': {'csv_output': 'off'},
                                                            'BeamPlot': {'include_rbm': 'on',
                                                                        'include_applied_forces': 'on'},
                                                            'AerogridPlot': {
                                                                'include_rbm': 'on',
                                                                'include_applied_forces': 'on',
                                                                'minus_m_star': 0,
                                                                'plot_nonlifting_surfaces': nonlifting_body_interactions,},
                                                            'SaveData': settings['SaveData']
                                                            },
                                                            }

    settings['LiftDistribution'] = {'q_ref': 0.5*1.225*u_inf**2,
                                    'coefficients': True}
    settings['BeamLoads'] = {'csv_output': True}

    settings['BeamPlot'] = {'include_rbm': 'on',
                            'include_applied_forces': 'on'}


    settings['AerogridPlot'] = {'include_rbm': 'on',
                                'include_forward_motion': 'off',
                                'include_applied_forces': 'on',
                                'minus_m_star': 0,
                                'u_inf': u_inf,
                                'dt': dt,
                                'plot_nonlifting_surfaces': nonlifting_body_interactions,}

flexop_model.create_settings(settings)
flexop_model.run()
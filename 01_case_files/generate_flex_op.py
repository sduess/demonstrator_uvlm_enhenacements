#! /usr/bin/env python3
import h5py as h5
import numpy as np
import os
import sharpy.utils.algebra as algebra
import pandas as pd
import matplotlib.pyplot as plt

case_name = 'flex_op_static_trim'
print(case_name)
route = os.path.dirname(os.path.realpath(__file__)) + '/'


# EXECUTION
flow = ['BeamLoader',
        'AerogridLoader',
        'NonliftingbodygridLoader',
        # 'StaticUvlm',
        # 'StaticCoupled',
        'StaticTrim',
        'BeamLoads',
        'AeroForcesCalculator',
        'LiftDistribution',
        'AerogridPlot',
        # 'LiftDistribution',
        # 'BeamPlot',
        # 'AeroForcesCalculator',
        # 'DynamicCoupled',
        # 'Modal',
        # 'LinearAssember',
        # 'AsymptoticStability',
        ]

# if free_flight is False, the motion of the centre of the wing is prescribed.
free_flight = True
if not free_flight:
    # case_name += '_prescribed'
    amplitude = 0*np.pi/180
    period = 3
    # case_name += '_amp_' + str(amplitude).replace('.', '') + '_period_' + str(period)

# FLIGHT CONDITIONS
# the simulation is set such that the aircraft flies at a u_inf velocity while
# the air is calm.
u_inf = 40
rho = 1.225

wake_length =  3

### Wing
# known inputs
sweep_LE = np.deg2rad(20.) 
max_engine_thrust = 300 # N

wing_span = 7.07 # m
half_wing_span = wing_span/2 # m
aspect_ratio = 19.71
S_wing = 2.54 # m2
length_fuselage = 3.44 # m
aero_center = 0.76297 # m
mass_take_off = 65 # kg

# estimated inputs
chord_root = 0.471  #0.5048 (Graph)
chord_tip = 0.236 #0.2544#

# calculated inputs
x_tip = half_wing_span*np.tan(sweep_LE)
sweep_quarter_chord = np.arctan((x_tip+chord_tip/4-chord_root/4)/(half_wing_span))
sweep_TE= np.arctan((x_tip + chord_tip - chord_root)/(half_wing_span))




### Tail
# inputs from webplotdigitizer
v_tail_angle = np.deg2rad(35.)
tail_chord_tip = 0.180325
tail_sweep_LE = np.deg2rad(19.51951)
tail_sweep_TE = np.deg2rad(18.0846)
half_tail_span = 1.318355
tail_span = 2*half_tail_span
# TODO: Get correct values
offset_tail_nose = 2.86236881559 # more precise values from plot!
vertical_offset_tail = 0.0

# calculated inputs
tail_x_tip = half_tail_span*np.tan(tail_sweep_LE)
tail_chord_root = tail_x_tip + tail_chord_tip - half_tail_span*np.tan(tail_sweep_TE)
tail_sweep_quarter_chord = np.arctan((tail_x_tip+tail_chord_tip/4-tail_chord_root/4)/(half_tail_span))
### Fuselage
# inputs from webplotdigitizer
offset_wing_nose = 0.8842 + 0.09#0.8822765386648912


alpha = 2.8*np.pi/180
beta = 0
roll = 0
gravity = 'on'
cs_deflection =  -2.08*np.pi/180
rudder_static_deflection = np.deg2rad(0.0)
rudder_step = 0.0*np.pi/180
thrust = 6.1 #max thrust 300
sigma = 1.
lambda_dihedral = 0*np.pi/180

# gust settings
gust_intensity = 0.80
gust_length = 1*u_inf
gust_offset = 0.0*u_inf


# numerics
n_step = 5
structural_relaxation_factor = 0.6
relaxation_factor = 0.35
tolerance = 1e-6
fsi_tolerance = 1e-4

num_cores = 20

# MODEL GEOMETRY
# beam
span_main = 16.0
lambda_main = 0.25
ea_main = 0.3

structural_properties = dict()
structural_properties['Goland'] = {
    'ea': 1e9,
    'ga': 1e9,
    'gj': 0.987581e6,
    'eiy': 9.77221e6,
    'eiz': 1e2 * 9.77221e6,
    'sigma': 1.,
    'm_unit': 35.71,
    'j_tors': 8.64,
}

ea = 1e4 #structural_properties['Goland']['ea'] # 1e7
ga = 1e4 #structural_properties['Goland']['ga'] # 1e5
gj = 1e4 #structural_properties['Goland']['gj'] # 1e4
eiy = 1e4 #structural_properties['Goland']['eiy'] #  2e4
eiz = 1e4 #tructural_properties['Goland']['eiz'] #  4e6
m_bar_main = structural_properties['Goland']['m_unit'] #  0.75
j_bar_main = structural_properties['Goland']['j_tors'] #  0.075

# length_fuselage = 10
offset_fuselage = 0
sigma_fuselage = 100
m_bar_fuselage = 0.2
j_bar_fuselage = 0.08

span_tail = 2.5
ea_tail = 0.5
fin_height = 2.5
ea_fin = 0.5
sigma_tail = 100
m_bar_tail = 0.3
j_bar_tail = 0.08


# lumped masses
n_lumped_mass = 1
lumped_mass_nodes = np.zeros((n_lumped_mass, ), dtype=int)
lumped_mass = np.zeros((n_lumped_mass, ))
lumped_mass[0] = 65
lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
lumped_mass_position = np.zeros((n_lumped_mass, 3))
x_lumped_mass = 0.606 - offset_wing_nose
# lumped_mass_position[0,0] = -0.2
# lumped_mass_position[0,1] = -0.01
# lumped_mass_position[0,2] = -0.025

# aero


# DISCRETISATION
# spatial discretisation
# chordiwse panels
n_elem_multiplier = 1
m = int(4*n_elem_multiplier)
m_radial_fuselage = 12 #36
# spanwise elements
n_ailerons_per_wing = 4
n_elev_per_tail_surf = 2
n_elem_junction_main = int(1*n_elem_multiplier)
n_elem_root_main = int(1*n_elem_multiplier)
n_elem_tip_main = int(1*n_elem_multiplier)
n_elem_per_aileron = int(4*n_elem_multiplier)
n_elem_per_elevator =  int(3*n_elem_multiplier)
n_elem_junction_tail = int(2*n_elem_multiplier)
n_elem_main = int(n_elem_junction_main + n_elem_root_main + n_ailerons_per_wing * n_elem_per_aileron + n_elem_tip_main)
n_elem_tail = int(n_elem_junction_tail + n_elev_per_tail_surf * n_elem_per_elevator)
n_elem_fuselage = int(8*n_elem_multiplier)
n_surfaces = 4
n_nonlifting_bodies = 1


# temporal discretisation
physical_time = 30
tstep_factor = 1.
dt = 1.0/m/u_inf*tstep_factor
n_tstep = round(physical_time/dt)


# control_surface_input
numb_ailerons = 4
y_coord_ailerons= np.array([0.862823, 2.820273, 4.301239, 5.653424, 6.928342])/2.
# for i_aileron in range(numb_ailerons):

# y_coord_ailerons = np.array
numb_elevators = 2
y_coord_elevators = np.array([0.258501, 0.788428, 1.318355])/2.
y_coord_junction = 0.144    

chord_ratio_aileron = 25./100. #25% of chord length
chord_ratio_elevator = 36./100.
control_surface_parameters = dict()



# END OF INPUT-----------------------------------------------------------------

# beam processing
n_node_elem = 3
span_main1 = (1.0 - lambda_main)*span_main
span_main2 = lambda_main*span_main

n_elem_main1 = n_elem_main
n_elem_main2 = n_elem_main - n_elem_main1

# total number of elements
n_elem = 0
n_elem += n_elem_main1 + n_elem_main1
n_elem += n_elem_fuselage
n_elem += n_elem_tail + n_elem_tail

# number of nodes per part
n_node_main1 = n_elem_main1*(n_node_elem - 1) + 1
# n_node_main2 = n_elem_main2*(n_node_elem - 1) + 1
n_node_main = n_node_main1 
n_node_fuselage = n_elem_fuselage*(n_node_elem - 1) + 1
n_node_tail = n_elem_tail*(n_node_elem - 1) + 1
n_node_fuselage = (n_elem_fuselage+1)*(n_node_elem - 1) -1
# total number of nodes
n_node = 0
n_node += n_node_main1
n_node += n_node_main1 - 1
n_node += n_node_fuselage - 1
n_node += n_node_tail - 1
n_node += n_node_tail - 1



# stiffness and mass matrices
n_stiffness = 3
base_stiffness_main = sigma*np.diag([ea, ga, ga, gj, eiy, eiz])
base_stiffness_fuselage = base_stiffness_main.copy()*sigma_fuselage
base_stiffness_fuselage[4, 4] = base_stiffness_fuselage[5, 5]
base_stiffness_tail = base_stiffness_main.copy()*sigma_tail
base_stiffness_tail[4, 4] = base_stiffness_tail[5, 5]

n_mass = 3
base_mass_main = np.diag([m_bar_main, m_bar_main, m_bar_main, j_bar_main, 0.5*j_bar_main, 0.5*j_bar_main])
base_mass_fuselage = np.diag([m_bar_fuselage,
                              m_bar_fuselage,
                              m_bar_fuselage,
                              j_bar_fuselage,
                              j_bar_fuselage*0.5,
                              j_bar_fuselage*0.5])
base_mass_tail = np.diag([m_bar_tail,
                          m_bar_tail,
                          m_bar_tail,
                          j_bar_tail,
                          j_bar_tail*0.5,
                          j_bar_tail*0.5])


# PLACEHOLDERS
# beam
x = np.zeros((n_node, ))
y = np.zeros((n_node, ))
z = np.zeros((n_node, ))
beam_number = np.zeros((n_elem, ), dtype=int)
frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
structural_twist = np.zeros((n_elem, 3))
conn = np.zeros((n_elem, n_node_elem), dtype=int)
stiffness = np.zeros((n_stiffness, 6, 6))
elem_stiffness = np.zeros((n_elem, ), dtype=int)
mass = np.zeros((n_mass, 6, 6))
elem_mass = np.zeros((n_elem, ), dtype=int)
boundary_conditions = np.zeros((n_node, ), dtype=int)
app_forces = np.zeros((n_node, 6))


# aero
airfoil_distribution = np.zeros((n_elem, n_node_elem), dtype=int)
surface_distribution = np.zeros((n_elem,), dtype=int) - 1
surface_m = np.zeros((n_surfaces, ), dtype=int)
m_distribution = 'uniform'
aero_node = np.zeros((n_node,), dtype=bool)
twist = np.zeros((n_elem, n_node_elem))
sweep = np.zeros((n_elem, n_node_elem))
chord = np.zeros((n_elem, n_node_elem,))
elastic_axis = np.zeros((n_elem, n_node_elem,))

# control surfaces
control_surface = np.zeros((n_elem, n_node_elem), dtype=int) - 1

# nonlifting body
nonlifting_body_node = np.zeros((n_node,), dtype=bool)
nonlifting_body_distribution = np.zeros((n_elem,), dtype=int) - 1
nonlifting_body_m = np.zeros((n_nonlifting_bodies, ), dtype=int)
radius = np.zeros((n_node,))
a_ellipse = np.zeros((n_node,))
b_ellipse = np.zeros((n_node,))
z_0_ellipse = np.zeros((n_node,))
radius = np.zeros((n_node,))
junction_boundary_condition_aero = np.zeros((1, n_surfaces), dtype=int) - 1


# FUNCTIONS-------------------------------------------------------------
def clean_test_files():
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    dyn_file_name = route + '/' + case_name + '.dyn.h5'
    if os.path.isfile(dyn_file_name):
        os.remove(dyn_file_name)

    aero_file_name = route + '/' + case_name + '.aero.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    aero_file_name = route + '/' + case_name + '.nonlifting_body.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    solver_file_name = route + '/' + case_name + '.sharpy'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)


def find_index_of_closest_entry(array_values, target_value):
    return (np.abs(array_values - target_value)).argmin()

def generate_dyn_file():
    global dt
    global n_tstep
    global route
    global case_name
    global num_elem
    global num_node_elem
    global num_node
    global amplitude
    global period
    global free_flight

    dynamic_forces_time = None
    with_dynamic_forces = False
    with_forced_vel = False
    if not free_flight:
        with_forced_vel = True

    if with_dynamic_forces:
        f1 = 100
        dynamic_forces = np.zeros((num_node, 6))
        app_node = [int(num_node_main - 1), int(num_node_main)]
        dynamic_forces[app_node, 2] = f1
        force_time = np.zeros((n_tstep, ))
        limit = round(0.05/dt)
        force_time[50:61] = 1

        dynamic_forces_time = np.zeros((n_tstep, num_node, 6))
        for it in range(n_tstep):
            dynamic_forces_time[it, :, :] = force_time[it]*dynamic_forces

    forced_for_vel = None
    if with_forced_vel:
        forced_for_vel = np.zeros((n_tstep, 6))
        forced_for_acc = np.zeros((n_tstep, 6))
        for it in range(n_tstep):
            # if dt*it < period:
            # forced_for_vel[it, 2] = 2*np.pi/period*amplitude*np.sin(2*np.pi*dt*it/period)
            # forced_for_acc[it, 2] = (2*np.pi/period)**2*amplitude*np.cos(2*np.pi*dt*it/period)

            forced_for_vel[it, 3] = 2*np.pi/period*amplitude*np.sin(2*np.pi*dt*it/period)
            forced_for_acc[it, 3] = (2*np.pi/period)**2*amplitude*np.cos(2*np.pi*dt*it/period)

    if with_dynamic_forces or with_forced_vel:
        with h5.File(route + '/' + case_name + '.dyn.h5', 'a') as h5file:
            if with_dynamic_forces:
                h5file.create_dataset(
                    'dynamic_forces', data=dynamic_forces_time)
            if with_forced_vel:
                h5file.create_dataset(
                    'for_vel', data=forced_for_vel)
                h5file.create_dataset(
                    'for_acc', data=forced_for_acc)
            h5file.create_dataset(
                'num_steps', data=n_tstep)

def generate_fem():
    stiffness[0, ...] = base_stiffness_main
    stiffness[1, ...] = base_stiffness_fuselage
    stiffness[2, ...] = base_stiffness_tail

    mass[0, ...] = base_mass_main
    mass[1, ...] = base_mass_fuselage
    mass[2, ...] = base_mass_tail

    we = 0
    wn = 0
    # inner right wing
    beam_number[we:we + n_elem_main1] = 0
    # junction (part without ailerons)
    n_node_junctions = int(3 + 2*(n_elem_junction_main-1))
    y[wn:wn + n_node_junctions] = np.linspace(0.0, y_coord_junction, n_node_junctions)
    n_node_root = int(3 + 2*(n_elem_root_main-1))
    y[wn + n_node_junctions:wn + n_node_junctions+n_node_root-1] = np.linspace(y_coord_junction, y_coord_ailerons[0], n_node_root)[1:]
    
    
    # Approach 1: Direct transition from one aileron to another aileron
    n_nodes_per_cs = (n_elem_per_aileron)*2+1
    wn_end = 0
    n_node_tip = int(3 + 2*(n_elem_tip_main-1))
    for i_control_surface in range(n_ailerons_per_wing):
        wn_start = wn +  n_node_junctions -1 + n_node_root- 1 + i_control_surface*(n_nodes_per_cs-1)
        wn_end= wn_start + n_nodes_per_cs
        y[wn_start:wn_end] = np.linspace(y_coord_ailerons[i_control_surface], 
                                         y_coord_ailerons[i_control_surface+1], 
                                         n_nodes_per_cs)
        # if i_control_surface == n_ailerons_per_wing - 1:
        #     wn_start = wn +  n_node_junctions  + n_node_root- 1 + i_control_surface*(n_nodes_per_cs-1)
        #     wn_end= wn_start + n_nodes_per_cs
        
    y[wn_end:wn_end + n_node_tip-1] = np.linspace(y_coord_ailerons[-1], half_wing_span, n_node_tip)[1:]
    
    # y[wn:wn + n_node_main1] = np.linspace(0.0, half_wing_span, n_node_main1)
    # TODO: Check if wingspan (tip coordinate) is still correct?
    x[wn+n_node_junctions:wn + n_node_main] += (abs(y[wn+n_node_junctions:wn + n_node_main])-y_coord_junction) * np.tan(sweep_quarter_chord)
    for ielem in range(n_elem_main1):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]

    app_forces[wn] = [0, thrust, 0, 0, 0, 0]
    elem_stiffness[we:we + n_elem_main1] = 0
    elem_mass[we:we + n_elem_main1] = 0
    boundary_conditions[0] = 1
    # remember this is in B FoR
    we += n_elem_main1
    wn += n_node_main1

    # inner left wing
    beam_number[we:we + n_elem_main1 - 1] = 1
    y[wn:wn + n_node_main1 - 1] = -y[1:n_node_main1]
    x[wn:wn + n_node_main1 - 1] = x[1:n_node_main1]
    z[wn:wn + n_node_main1 - 1] = z[1:n_node_main1]
    # y[wn:wn + n_node_main1 - 1] = np.linspace(0.0, -half_wing_span, n_node_main1)[1:]
    # x[wn:wn + n_node_main1 - 1] += abs(y[wn:wn + n_node_main1 - 1]) * np.tan(sweep_quarter_chord)
    for ielem in range(n_elem_main1):
        conn[we + ielem, :] = ((np.ones((3, ))*(we+ielem)*(n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = 0
    elem_stiffness[we:we + n_elem_main1] = 0
    elem_mass[we:we + n_elem_main1] = 0

    print("wing left -= ", y[wn:wn + n_node_main1 - 1])
    we += n_elem_main1
    wn += n_node_main1 - 1



    # fuselage
    beam_number[we:we + n_elem_fuselage] = 2
    x_fuselage = np.linspace(0.0, length_fuselage, n_node_fuselage) - offset_wing_nose
    z_fuselage = np.linspace(0.0, offset_fuselage, n_node_fuselage)
    idx_junction = find_index_of_closest_entry(x_fuselage, x[0])
    x_fuselage = np.delete(x_fuselage, idx_junction)
    z_fuselage = np.delete(z_fuselage, idx_junction)
    x[wn:wn + n_node_fuselage-1] = x_fuselage 
    z[wn:wn + n_node_fuselage-1] = z_fuselage
    adjust = False

    node_fuselage_conn = False
    for ielem in range(n_elem_fuselage):
        conn[we + ielem, :] = ((np.ones((3,))*(we + ielem)*(n_node_elem - 1)) +
                               2 + [0, 2, 1]) - 1
        if adjust:
            conn[we + ielem, :] -= 1
        else:
            if node_fuselage_conn:
                conn[we + ielem, 0] = 0
            elif (conn[we + ielem, :] ==  wn+idx_junction).any():
                adjust_elem = False
                for idx_node in [0, 2, 1]:               
                    if adjust_elem:
                        conn[we + ielem, idx_node] -= 1  

                    elif conn[we + ielem, idx_node] ==  wn+idx_junction:
                        adjust = True
                        adjust_elem = True
                        conn[we + ielem, idx_node] = 0
                        if idx_node == 1:
                            node_fuselage_conn = True
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    
    # setup lumped mass position
    wn_lumped_mass = wn + find_index_of_closest_entry(x[wn:wn + n_node_fuselage-1], x_lumped_mass)
    lumped_mass_nodes[0] = wn_lumped_mass
    lumped_mass_position[0, 0] = x[wn_lumped_mass]
    lumped_mass_position[0, 1] = y[wn_lumped_mass]
    lumped_mass_position[0, 2] = z[wn_lumped_mass]

    # x[wn:wn + n_node_fuselage - 1] = np.linspace(0.0, length_fuselage, n_node_fuselage)[1:] 
    # z[wn:wn + n_node_fuselage - 1] = np.linspace(0.0, offset_fuselage, n_node_fuselage)[1:]
    # for ielem in range(n_elem_fuselage):
    #     conn[we + ielem, :] = ((np.ones((3,))*(we + ielem)*(n_node_elem - 1)) +
    #                            [0, 2, 1])
    #     for inode in range(n_node_elem):
    #         frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    # # TODO: Update Connectivities
    # conn[we, 0] = 0
    elem_stiffness[we:we + n_elem_fuselage] = 1
    elem_mass[we:we + n_elem_fuselage] = 1
    index_tail_start = wn + find_index_of_closest_entry(x[wn:wn + n_node_fuselage-1], offset_tail_nose-offset_wing_nose)
    we += n_elem_fuselage
    wn += n_node_fuselage - 1
    global end_of_fuselage_node
    end_of_fuselage_node = wn - 1

    



    # right tail
    beam_number[we:we + n_elem_tail] = 3
    x[wn:wn + n_node_tail - 1] = x[index_tail_start]
    wn_right_tail_start = wn
    n_node_junctions = int(3 + 2*(n_elem_junction_tail-1))
    y[wn:wn + n_node_junctions - 1] = np.linspace(0.0, y_coord_elevators[0], n_node_junctions)[:-1]
    # Approach 1: Direct transition from one aileron to another aileron
    n_nodes_per_cs = (n_elem_per_elevator)*2+1

    for i_control_surface in range(n_elev_per_tail_surf):
        wn_start = wn +  n_node_junctions - 1 + i_control_surface*(n_nodes_per_cs-1)
        wn_end= wn_start + n_nodes_per_cs
        y[wn_start:wn_end] = np.linspace(y_coord_elevators[i_control_surface], 
                                         y_coord_elevators[i_control_surface+1], 
                                         n_nodes_per_cs)
    print("y tail right = ", y[wn:wn + n_node_tail - 1] )
    
    # y[wn:wn + n_node_tail - 1] = np.linspace(0.0, half_tail_span, n_node_tail)[1:]
    x[wn:wn + n_node_tail - 1]  += abs(y[wn:wn + n_node_tail - 1])* np.tan(tail_sweep_quarter_chord)
    z[wn:wn + n_node_tail - 1] = z[index_tail_start]
    z[wn:wn + n_node_tail - 1] += y[wn:wn + n_node_tail - 1] * np.tan(v_tail_angle)
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    conn[we, 0] =  index_tail_start 
    elem_stiffness[we:we + n_elem_tail] = 2
    elem_mass[we:we + n_elem_tail] = 2
    boundary_conditions[wn + n_node_tail - 2] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    # left tail
    beam_number[we:we + n_elem_tail] = 4
    x[wn:wn + n_node_tail - 1] = x[index_tail_start]
    y[wn:wn + n_node_tail - 1] = -y[wn_right_tail_start:wn_right_tail_start + n_node_tail - 1]
    print("y tail left + ", y[wn:wn + n_node_tail- 1])
    # y[wn:wn + n_node_tail - 1] = np.linspace(0.0, -half_tail_span, n_node_tail)[1:]
    x[wn:wn + n_node_tail - 1]  += abs(y[wn:wn + n_node_tail - 1])* np.tan(tail_sweep_quarter_chord)
    z[wn:wn + n_node_tail - 1] = z[index_tail_start]
    z[wn:wn + n_node_tail - 1] += abs(y[wn:wn + n_node_tail - 1]) * np.tan(v_tail_angle)
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) +
                               [0, 2, 1])
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] =  index_tail_start 
    elem_stiffness[we:we + n_elem_tail] = 2
    elem_mass[we:we + n_elem_tail] = 2
    boundary_conditions[wn + n_node_tail - 2] = -1
    we += n_elem_tail
    wn += n_node_tail - 1



    with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
        coordinates = h5file.create_dataset('coordinates', data=np.column_stack((x, y, z)))
        conectivities = h5file.create_dataset('connectivities', data=conn)
        num_nodes_elem_handle = h5file.create_dataset(
            'num_node_elem', data=n_node_elem)
        num_nodes_handle = h5file.create_dataset(
            'num_node', data=n_node)
        num_elem_handle = h5file.create_dataset(
            'num_elem', data=n_elem)
        stiffness_db_handle = h5file.create_dataset(
            'stiffness_db', data=stiffness)
        stiffness_handle = h5file.create_dataset(
            'elem_stiffness', data=elem_stiffness)
        mass_db_handle = h5file.create_dataset(
            'mass_db', data=mass)
        mass_handle = h5file.create_dataset(
            'elem_mass', data=elem_mass)
        frame_of_reference_delta_handle = h5file.create_dataset(
            'frame_of_reference_delta', data=frame_of_reference_delta)
        structural_twist_handle = h5file.create_dataset(
            'structural_twist', data=structural_twist)
        bocos_handle = h5file.create_dataset(
            'boundary_conditions', data=boundary_conditions)
        beam_handle = h5file.create_dataset(
            'beam_number', data=beam_number)
        app_forces_handle = h5file.create_dataset(
            'app_forces', data=app_forces)
        lumped_mass_nodes_handle = h5file.create_dataset(
            'lumped_mass_nodes', data=lumped_mass_nodes)
        lumped_mass_handle = h5file.create_dataset(
            'lumped_mass', data=lumped_mass)
        lumped_mass_inertia_handle = h5file.create_dataset(
            'lumped_mass_inertia', data=lumped_mass_inertia)
        lumped_mass_position_handle = h5file.create_dataset(
            'lumped_mass_position', data=lumped_mass_position)

def generate_aero_file():
    global x, y, z
    #
    # control surfaces
    n_control_surfaces = numb_ailerons* 2 + numb_elevators * 2 # on each side
    control_surface = np.zeros((n_elem, n_node_elem), dtype=int) - 1
    control_surface_type = np.zeros((n_control_surfaces, ), dtype=int)
    control_surface_deflection = np.zeros((n_control_surfaces, ))
    control_surface_chord = np.zeros((n_control_surfaces, ), dtype=int)
    control_surface_hinge_coord = np.zeros((n_control_surfaces, ), dtype=float)

    # control surface type 0 = static
    # control surface type 1 = dynamic

    # aileron 1
    control_surface_type[0] = 0
    control_surface_deflection[0] = 0
    control_surface_chord[0] = m/4 # 0.25
    control_surface_hinge_coord[0] = -0. # nondimensional wrt elastic axis (+ towards the trailing edge)

    # aileron 2
    control_surface_type[1] = 0
    control_surface_deflection[1] = 0
    control_surface_chord[1] = m/4 # 0.25
    control_surface_hinge_coord[1] = -0. # nondimensional wrt elastic axis (+ towards the trailing edge)

    # aileron 3
    control_surface_type[2] = 0
    control_surface_deflection[2] =  0
    control_surface_chord[2] = m/4 # 0.25
    control_surface_hinge_coord[2] = -0. # nondimensional wrt elastic axis (+ towards the trailing edge)
    
    # aileron 4
    control_surface_type[3] = 0
    control_surface_deflection[3] =  0
    control_surface_chord[3] = m/4 # 0.25
    control_surface_hinge_coord[3] = -0. # nondimensional wrt elastic axis (+ towards the trailing edge)
    # TODO: Setup right elevator chord length
    # rudder 1 - used for trim
    control_surface_type[4]  = 0
    control_surface_deflection[4]  = np.deg2rad(cs_deflection)
    control_surface_chord[4]  =  m/4 # Flexop@s elevator cs have a ,chord of 36%. problems with aerogrid discretization
    control_surface_hinge_coord[4]  = -0. # nondimensional wrt elastic axis (+ towards the trailing edge)
    # rudder 2
    control_surface_type[5]  = 0
    control_surface_deflection[5]  = 0
    control_surface_chord[5]  = m/4  # Flexop@s elevator cs have a ,chord of 36%. problems with aerogrid discretization
    control_surface_hinge_coord[5]  = -0. # nondimensional wrt elastic axis (+ towards the trailing edge)


    # # For Testing purposes: visualization of control surfaces in Paraview
    # list_cs_deflections = [-30, 30, -60, 60, -90, 90]
    # for i in range(len(list_cs_deflections)):
    #     control_surface_deflection[i] = np.deg2rad(list_cs_deflections[i])
    

    we = 0
    wn = 0
    # right wing (surface 0, beam 0)
    i_surf = 0
    airfoil_distribution[we:we + n_elem_main, :] = 0
    surface_distribution[we:we + n_elem_main] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_main] = y[wn:wn + n_node_main] >= y_coord_junction 
    # idx_junction = wn + min(np.where(aero_node[wn:wn + n_node_main] == 1)[0]) #find_index_of_closest_entry(y[wn:wn + n_node_main], 0.66)
    
    n_node_junctions = int(3 + 2*(n_elem_junction_main-1))
    junction_boundary_condition_aero[0, i_surf] = 1 # BC at fuselage junction
    temp_chord = np.zeros((n_node_main)) + chord_root
    temp_chord[n_node_junctions:n_node_main] = abs(y[wn+n_node_junctions:wn + n_node_main]*np.tan(sweep_LE)-(chord_root + y[wn+n_node_junctions:wn + n_node_main]*np.tan(sweep_TE)))
    temp_sweep = np.linspace(0.0, 0*np.pi/180, n_node_main)

    node_counter = 0
    for i_elem in range(we, we + n_elem_main):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            if i_local_node == 1:
                chord[i_elem, i_local_node] = temp_chord[node_counter + 1]
            elif i_local_node == 2:
                chord[i_elem, i_local_node] = temp_chord[node_counter - 1]
            else:
                chord[i_elem, i_local_node] = temp_chord[node_counter] 
            elastic_axis[i_elem, i_local_node] = ea_main
            sweep[i_elem, i_local_node] = temp_sweep[node_counter]


    # For control surfaces setup
    idx_ailerons_start_y_coordinate = find_index_of_closest_entry(y[wn:wn + n_node_main1], y_coord_ailerons[0])
    idx_ailerons_end_y_coordinate = find_index_of_closest_entry(y[wn:wn + n_node_main1], y_coord_ailerons[-1])
    cs_surface = False

    node_counter = 0
    cs_counter = -1
    cs_surface = False
    for i_elem in range(we, we + n_elem_main):
        for i_local_node in [0,2,1]:
            if not i_local_node == 0:
                node_counter += 1
            if abs(y[node_counter]) == y_coord_ailerons[0] and i_local_node == 0:
                cs_surface = True 
            if cs_surface:
                if abs(y[node_counter]) in y_coord_ailerons:
                    if i_local_node == 0:
                        cs_counter += 1
                control_surface[i_elem, i_local_node] = cs_counter
            if abs(y[node_counter]) >= y_coord_ailerons[-1]:
                cs_surface = False
   
    
    
    we += n_elem_main
    wn += n_node_main

    # left wing (surface 1, beam 1)
    i_surf = 1
    airfoil_distribution[we:we + n_elem_main, :] = 0
    # airfoil_distribution[wn:wn + n_node_main - 1] = 0
    surface_distribution[we:we + n_elem_main] = i_surf
    surface_m[i_surf] = m
    aero_node[wn:wn + n_node_main] = y[wn:wn + n_node_main] <= -y_coord_junction  #-0.11
    junction_boundary_condition_aero[0, i_surf] = 0 # BC at fuselage junction
    # aero_node[wn:wn + n_node_main - 1] = True
    # chord[wn:wn + num_node_main - 1] = np.linspace(main_chord, main_tip_chord, num_node_main)[1:]
    # chord[wn:wn + num_node_main - 1] = main_chord
    # elastic_axis[wn:wn + num_node_main - 1] = main_ea
    temp_chord = temp_chord
    node_counter = 0
    for i_elem in range(we, we + n_elem_main):
        for i_local_node in range(n_node_elem): 
            if not i_local_node == 0:
                node_counter += 1
            if i_local_node == 1:
                chord[i_elem, i_local_node] = temp_chord[node_counter + 1]
                sweep[i_elem, i_local_node] = temp_sweep[node_counter + 1]
            elif i_local_node == 2:
                chord[i_elem, i_local_node] = temp_chord[node_counter - 1]
                sweep[i_elem, i_local_node] = temp_sweep[node_counter - 1]
            else:
                chord[i_elem, i_local_node] = temp_chord[node_counter] 
                sweep[i_elem, i_local_node] = temp_sweep[node_counter]
            elastic_axis[i_elem, i_local_node] = ea_main


    # For control surfaces setup
    idx_ailerons_start_y_coordinate = find_index_of_closest_entry(y[wn:wn + n_node_main1-1], -y_coord_ailerons[0])
    idx_ailerons_end_y_coordinate = find_index_of_closest_entry(y[wn:wn + n_node_main1-1], -y_coord_ailerons[-1])

    node_counter = 0
    cs_counter = -1
    cs_surface = False
    for i_elem in range(we, we + n_elem_main):
        for i_local_node in [0,2,1]:
            if not i_local_node == 0:
                node_counter += 1
            if abs(y[node_counter]) == y_coord_ailerons[0] and i_local_node == 0:
                cs_surface = True 
            if cs_surface:
                if abs(y[node_counter]) in y_coord_ailerons:
                    if i_local_node == 0:
                        cs_counter += 1
                control_surface[i_elem, i_local_node] = cs_counter
            if abs(y[node_counter]) >= y_coord_ailerons[-1]:
                cs_surface = False
        
        


    we += n_elem_main
    wn += n_node_main - 1

    we += n_elem_fuselage
    wn += n_node_fuselage - 1 - 1
    #
    #
    # # # right tail (surface 3, beam 4)
    i_surf = 2
    airfoil_distribution[we:we + n_elem_tail, :] = 2
    # airfoil_distribution[wn:wn + n_node_tail] = 0
    surface_distribution[we:we + n_elem_tail] = i_surf
    surface_m[i_surf] = m
    # XXX not very elegant
    # aero_node[wn:] = True

    aero_node[wn:wn + n_node_tail] = y[wn:wn + n_node_tail] >= 0.04
    # idx_junction_tail = wn + min(np.where(aero_node[wn:wn + n_node_main] == 1)[0]) #find_index_of_closest_entry(y[wn:wn + n_node_main], 0.66)
    junction_boundary_condition_aero[0, i_surf] = 3 # BC at fuselage junction
    
    temp_chord = tail_chord_root - abs(y[wn:wn + n_node_tail]*np.tan(tail_sweep_LE)) + abs(y[wn:wn + n_node_tail]*np.tan(tail_sweep_TE))
    # chord[wn:wn + num_node_tail] = tail_chord
    # elastic_axis[wn:wn + num_node_main] = tail_ea
    node_counter = 0
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            twist[i_elem, i_local_node] = -0
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
                if i_local_node == 1:
                    chord[i_elem, i_local_node] = temp_chord[node_counter + 1]
                elif i_local_node == 2:
                    chord[i_elem, i_local_node] = temp_chord[node_counter - 1]
            else:
                chord[i_elem, i_local_node] = temp_chord[node_counter]            
            elastic_axis[i_elem, i_local_node] = ea_main
            # sweep[i_elem, i_local_node] = temp_sweep[node_counter]    # For control surfaces setup
    idx_elevator_start_y_coordinate = find_index_of_closest_entry(y[wn:wn + n_node_tail], y_coord_elevators[0])
    idx_elevator_end_y_coordinate = find_index_of_closest_entry(y[wn:wn + n_node_tail], y_coord_elevators[-1])
    
    node_counter = wn - 2
    cs_counter = -1
    
    cs_surface = False
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(3):
            if not i_local_node == 0:
                node_counter += 1
            if abs(y[node_counter]) == y_coord_elevators[0] and i_local_node == 0:
                cs_surface = True 
            if cs_surface:
                if abs(y[node_counter]) in y_coord_elevators:
                    if i_local_node == 0:
                        if cs_counter == -1:
                            cs_counter = 4
                        else:
                            cs_counter += 1
                control_surface[i_elem, i_local_node] = cs_counter
            if abs(y[node_counter]) >= y_coord_elevators[-1]:
                cs_surface = False
        # print("y node = ", y[node_counter])
        # print("i_elem = ", i_elem, " and control_surface ID = ",  control_surface[i_elem, i_local_node])
   

    we += n_elem_tail
    wn += n_node_tail
    #



    # # left tail (surface 4, beam 5)
    i_surf = 3
    airfoil_distribution[we:we + n_elem_tail, :] = 2
    # airfoil_distribution[wn:wn + n_node_tail - 1] = 0
    surface_distribution[we:we + n_elem_tail] = i_surf
    surface_m[i_surf] = m
    # aero_node[wn:wn + n_node_tail - 1] = True
    aero_node[wn:wn + n_node_tail] = y[wn:wn + n_node_tail] <= -0.04
    # idx_junction_tail = wn + min(np.where(aero_node[wn:wn + n_node_tail] == 1)[0]) #find_index_of_closest_entry(y[wn:wn + n_node_main], 0.66)
    junction_boundary_condition_aero[0, i_surf] = 2 # BC at fuselage junction
    # chord[wn:wn + num_node_tail] = tail_chord
    # elastic_axis[wn:wn + num_node_main] = tail_ea
    # twist[we:we + num_elem_tail] = -tail_twist
    node_counter = 0
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            twist[i_elem, i_local_node] = -0
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            if i_local_node == 1:
                chord[i_elem, i_local_node] = temp_chord[node_counter + 1]
            elif i_local_node == 2:
                chord[i_elem, i_local_node] = temp_chord[node_counter - 1]
            else:
                chord[i_elem, i_local_node] = temp_chord[node_counter]            
            elastic_axis[i_elem, i_local_node] = ea_main
            # sweep[i_elem, i_local_node] = temp_sweep[node_counter]

    # For control surfaces setup
    node_counter = wn - 2
    cs_counter = -1
    
    cs_surface = False
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(3):
            if not i_local_node == 0:
                node_counter += 1
            if abs(y[node_counter]) == y_coord_elevators[0] and i_local_node == 0:
                cs_surface = True 
            if cs_surface:
                if abs(y[node_counter]) in y_coord_elevators:
                    if i_local_node == 0:
                        if cs_counter == -1:
                            cs_counter = 4
                        else:
                            cs_counter += 1
                control_surface[i_elem, i_local_node] = cs_counter
            if abs(y[node_counter]) >= y_coord_elevators[-1]:

                cs_surface = False
    we -= n_elem_tail
    wn -= n_node_tail
    i_elem_counter = 0
    for i_elem in range(we, we + n_elem_tail):
        for i_local_node in range(3):
            control_surface[i_elem, i_local_node] = control_surface[i_elem + n_elem_tail, i_local_node]
    # node_counter = 0
    # for i_elem in range(we, we + n_elem_tail):
    #     for i_local_node in [0,2,1]: #range(n_node_elem):
    #         if not i_local_node == 0:
    #             node_counter += 1
    #         if node_counter == idx_elevator_start_y_coordinate:
    #             cs_surface = True            
    #         if cs_surface:
    #             control_surface[i_elem, i_local_node] = cs_surface_id # elevator right
    #             print("elevator at i_elem = " + str(i_elem) + " and i_local_node = " + str(i_local_node))
    #         if node_counter == idx_elevator_end_y_coordinate:
    #             cs_surface = False
    #             if i_elem  + 1== we + n_elem_tail:
    #                 control_surface[i_elem, :] = cs_surface_id 

    control_surface[control_surface == 5] = 4
    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add one airfoil
        naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))
        naca_airfoil_tail = airfoils_group.create_dataset('1', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))
        naca_airfoil_fin = airfoils_group.create_dataset('2', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))

        # chord
        chord_input = h5file.create_dataset('chord', data=chord)
        dim_attr = chord_input .attrs['units'] = 'm'

        # twist
        twist_input = h5file.create_dataset('twist', data=twist)
        dim_attr = twist_input.attrs['units'] = 'rad'

        # sweep
        sweep_input = h5file.create_dataset('sweep', data=sweep)
        dim_attr = sweep_input.attrs['units'] = 'rad'

        # airfoil distribution
        airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

        surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
        surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
        m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

        aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
        elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)
        h5file.create_dataset(
            'junction_boundary_condition', data=junction_boundary_condition_aero)
        control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
        control_surface_deflection_input = h5file.create_dataset('control_surface_deflection', data=control_surface_deflection)
        control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
        control_surface_hinge_coord_input = h5file.create_dataset('control_surface_hinge_coord', data=control_surface_hinge_coord)
        control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)


def generate_naca_camber(M=0, P=0):
    mm = M*1e-2
    p = P*1e-1

    def naca(x, mm, p):
        if x < 1e-6:
            return 0.0
        elif x < p:
            return mm/(p*p)*(2*p*x - x*x)
        elif x > p and x < 1+1e-6:
            return mm/((1-p)*(1-p))*(1 - 2*p + 2*p*x - x*x)

    x_vec = np.linspace(0, 1, 1000)
    y_vec = np.array([naca(x, mm, p) for x in x_vec])
    return x_vec, y_vec

def interpolate_fuselage_geometry(x_coord_beam, df_fuselage, coord, upper_surface=True):
    if coord == 'y':
        
        df_fuselage = df_fuselage.iloc[:,:2].dropna()
        
    else: 
        df_fuselage = df_fuselage.iloc[:,2:].dropna()
        first_and_last_row_df = df_fuselage.iloc[[0, -1]]
        if upper_surface:
            df_fuselage = df_fuselage[df_fuselage.iloc[:,1]>0.0]
        else:    
            df_fuselage= df_fuselage[df_fuselage.iloc[:,1]<0.0]
        df_fuselage = pd.concat([first_and_last_row_df, df_fuselage]).drop_duplicates()
        df_fuselage = df_fuselage.sort_values(df_fuselage.columns[0])
    y = []
    for x in  x_coord_beam:
        if x in df_fuselage.iloc[:,0].tolist():
            y.append(df_fuselage[df_fuselage.iloc[:,0] == x].iloc[0,1])
        else:
            # TODO make sure values are sorted
            values_adjacent_right = df_fuselage[df_fuselage.iloc[:,0] >= x].iloc[0, :]
            values_adjacent_left = df_fuselage[df_fuselage.iloc[:,0] <= x].iloc[-1, :]
            x_known = [values_adjacent_right.iloc[0], values_adjacent_left.iloc[0]]
            y_known = [values_adjacent_right.iloc[1], values_adjacent_left.iloc[1]]

            y.append(y_known[0]+ (x-x_known[0])/(x_known[1]- x_known[0])*(y_known[1]-y_known[0]))

    # plt.plot(x_coord_beam, y, 'x-')
    # plt.xlabel('x')
    # plt.xlabel(coord)
    # plt.show()

    return y



def generate_fuselage_geometry(x_coord_fuselage):
    df_fuselage = pd.read_csv('../01_case_files/flexOp_data/fuselage_geometry.csv', sep=";")
    y_coord_fuselage = interpolate_fuselage_geometry(x_coord_fuselage, df_fuselage, 'y', True)
    z_coord_fuselage_upper = interpolate_fuselage_geometry(x_coord_fuselage, df_fuselage, 'z', True)
    z_coord_fuselage_lower = interpolate_fuselage_geometry(x_coord_fuselage, df_fuselage, 'z', False)
    b_ellipse_tmp = (np.array(z_coord_fuselage_upper) - np.array(z_coord_fuselage_lower))/2.
    z_0_ellipse_tmp = b_ellipse_tmp - abs(np.array(z_coord_fuselage_lower))
    # z_0_ellipse_tmp[-6:] = 0
    # plt.plot(x_coord_fuselage, z_coord_fuselage_upper)
    # plt.plot(x_coord_fuselage, z_coord_fuselage_lower)
    # plt.plot(x_coord_fuselage, b_ellipse_tmp)
    # plt.plot(x_coord_fuselage, z_0_ellipse_tmp)
    # plt.show()
    # x_coord_fuselage = np.sort(x_coord_fuselage)
    return y_coord_fuselage, b_ellipse_tmp, z_0_ellipse_tmp

def generate_nonlifting_body_file():
    we = 0
    wn = 0

    # right wing
    nonlifting_body_node[wn:wn + n_node_main] = False
    we += n_elem_main
    wn += n_node_main

    # left wing
    nonlifting_body_node[wn:wn + n_node_main] = False
    we += n_elem_main
    wn += n_node_main -1

    #fuselage (beam?, body ID = 0)
    i_body = 0
    
    nonlifting_body_node[0] = True
    nonlifting_body_node[wn:wn + n_node_fuselage-1] = True
    nonlifting_body_distribution[we:we + n_elem_fuselage] = i_body
    nonlifting_body_m[i_body] = m_radial_fuselage
    #radius[wn:wn + n_node_fuselage] = get_ellipsoidal_geometry(x[wn:wn + n_node_fuselage], thickness_ratio_ellipse,0) #np.genfromtxt('radius_wanted.csv',delimiter=',')
    # radius_fuselage = create_fuselage_geometry()
    x_coord_fuselage = np.sort(x[nonlifting_body_node])
    idx_junction = find_index_of_closest_entry(x_coord_fuselage, x[0])
    x_coord_fuselage += abs(min(x_coord_fuselage))
    a_ellipse_tmp, b_ellipse_tmp, z_0_ellipse_tmp = generate_fuselage_geometry(x_coord_fuselage)
    a_ellipse[0] = a_ellipse_tmp[idx_junction]
    b_ellipse[0] = b_ellipse_tmp[idx_junction]
    z_0_ellipse[0] = z_0_ellipse_tmp[idx_junction]


    a_ellipse_tmp= np.delete(a_ellipse_tmp,idx_junction)
    b_ellipse_tmp= np.delete(b_ellipse_tmp,idx_junction)
    z_0_ellipse_tmp= np.delete(z_0_ellipse_tmp,idx_junction)
    a_ellipse[wn:wn + n_node_fuselage-1] =  a_ellipse_tmp
    b_ellipse[wn:wn + n_node_fuselage-1] =  b_ellipse_tmp
    z_0_ellipse[wn:wn + n_node_fuselage-1] =  z_0_ellipse_tmp
    # radius[0] = max(radius_fuselage)
    # radius_fuselage = np.delete(radius_fuselage,idx_junction)
    # radius[wn:wn + n_node_fuselage] = radius_fuselage #create_fuselage_geometry()

    with h5.File(route + '/' + case_name + '.nonlifting_body.h5', 'a') as h5file:
        h5file.create_dataset('shape', data='specific')
        h5file.create_dataset('a_ellipse', data=a_ellipse)
        h5file.create_dataset('b_ellipse', data=b_ellipse)
        h5file.create_dataset('z_0_ellipse', data=z_0_ellipse)
        nonlifting_body_m_input = h5file.create_dataset('surface_m', data=nonlifting_body_m)
        nonlifting_body_node_input = h5file.create_dataset('nonlifting_body_node', data=nonlifting_body_node)

        nonlifting_body_distribution_input = h5file.create_dataset('surface_distribution', data=nonlifting_body_distribution)
        
        # radius
        radius_input = h5file.create_dataset('radius', data=radius)
        dim_attr = radius_input.attrs['units'] = 'm'
    
    fuselage_shape = np.zeros((n_node, 4))
    fuselage_shape[:, 0] = x
    fuselage_shape[:, 1] = a_ellipse
    fuselage_shape[:, 2] = b_ellipse
    fuselage_shape[:, 3] = z_0_ellipse
    
    np.savetxt("fuselage_shape.csv", fuselage_shape)

    # right wing (surface 0, beam 0)

def generate_solver_file():
    file_name = route + '/' + case_name + '.sharpy'
    settings = dict()
    settings['SHARPy'] = {'case': case_name,
                          'route': route,
                          'flow': flow,
                          'write_screen': 'on',
                          'write_log': 'on',
                          'log_folder': route + '/output/',
                          'log_file': case_name + '.log'}

    settings['BeamLoader'] = {'unsteady': 'on',
                              'orientation': algebra.euler2quat(np.array([roll,
                                                                          alpha,
                                                                          beta]))}
    # settings['AerogridLoader'] = {'unsteady': 'on',
    #                               'aligned_grid': 'on',
    #                               'mstar': int(20/tstep_factor),
    #                               'freestream_dir': ['1', '0', '0']}
                                  
    # settings['AerogridLoader'] = {'unsteady': 'on',
    #                             'aligned_grid': 'on',
    #                             'mstar': wake_length*m, #int(20/tstep_factor),
    #                             'wake_shape_generator': 'StraightWake',
    #                             'wake_shape_generator_input': {
    #                                 'u_inf': u_inf,
    #                                 'u_inf_direction': [1., 0., 0.],
    #                                 'dt': dt,
    #                             },
    #                           }

    settings['AerogridLoader'] = {'unsteady': 'on',
                                'aligned_grid': 'on',
                                'mstar': int((1*chord_root)/(chord_tip/m)) + 9,
                                'wake_shape_generator': 'StraightWake',
                                    'wake_shape_generator_input': {
                                        'u_inf': u_inf,
                                        'u_inf_direction': [1., 0., 0.],
                                        'dt': dt,
                                        'dx1': chord_root/m,  # size of first wake panel. Here equal to bound panel
                                        'ndx1': int((1*chord_root)/(chord_tip/m)), 
                                        'r': 1.5,
                                        'dxmax':5*chord_tip}
                                    }
    print("Number of panels with size ``dx1`` = ", int((1*chord_root)/(chord_tip/m)) + 9)


    settings['AeroForcesCalculator'] = {'print_info': True,
                                    'coefficients': True,
                                    'S_ref': 2.54,
                                    'q_ref': 0.5*1.225*u_inf**2}
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
                              'num_cores': num_cores,
                              'n_rollup': 0,
                              'rollup_dt': dt,
                              'rollup_aic_refresh': 1,
                              'rollup_tolerance': 1e-4,
                              'velocity_field_generator': 'SteadyVelocityField',
                              'velocity_field_input': {'u_inf': u_inf,
                                                       'u_inf_direction': [1., 0, 0]},
                              'rho': rho,
                              'nonlifting_body_interactions': True}

    settings['StaticCoupled'] = {'print_info': 'off',
                                 'structural_solver': 'NonLinearStatic',
                                 'structural_solver_settings': settings['NonLinearStatic'],
                                 'aero_solver': 'StaticUvlm',
                                 'aero_solver_settings': settings['StaticUvlm'],
                                 'max_iter': 100,
                                 'n_load_steps': n_step,
                                 'tolerance': fsi_tolerance,
                                 'relaxation_factor': structural_relaxation_factor}

    settings['StaticTrim'] = {'solver': 'StaticCoupled',
                              'solver_settings': settings['StaticCoupled'],
                              'initial_alpha': alpha,
                              'initial_deflection': cs_deflection,
                              'initial_thrust': thrust,
                              'tail_cs_index': 4,
                              'fz_tolerance': 0.01,
                              'fy_tolerance': 0.01,
                              'm_tolerance': 0.01,
                              'initial_angle_eps': 0.05,
                              'initial_thrust_eps': 2.,
                              'save_info': True}

    settings['NonLinearDynamicCoupledStep'] = {'print_info': 'off',
                                               'max_iterations': 950,
                                               'delta_curved': 1e-1,
                                               'min_delta': tolerance,
                                               'newmark_damp': 5e-3,
                                               'gravity_on': gravity,
                                               'gravity': 9.81,
                                               'num_steps': n_tstep,
                                               'dt': dt,
                                               'initial_velocity': u_inf}

    settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'off',
                                           'max_iterations': 950,
                                           'delta_curved': 1e-1,
                                           'min_delta': tolerance,
                                           'newmark_damp': 5e-3,
                                           'gravity_on': gravity,
                                           'gravity': 9.81,
                                           'num_steps': n_tstep,
                                           'dt': dt,
                                           'initial_velocity': u_inf*int(free_flight)}

    relative_motion = 'off'
    if not free_flight:
        relative_motion = 'on'
    settings['StepUvlm'] = {'num_cores': num_cores,
                            'n_rollup': 0,
                            'convection_scheme': 2,
                            'gamma_dot_filtering': 0,
                            'cfl1': False,
                            'velocity_field_generator': 'GustVelocityField',
                            'velocity_field_input': {'u_inf': int(not free_flight)*u_inf,
                                                     'u_inf_direction': [1., 0, 0],
                                                     'gust_shape': '1-cos',
                                                     'gust_length': gust_length,
                                                     'gust_intensity': gust_intensity*u_inf,
                                                     'offset': gust_offset,
                                                     'span': span_main,
                                                     'relative_motion': relative_motion},
                            'rho': rho,
                            'n_time_steps': n_tstep,
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
                                  'final_relaxation_factor': 0.5,
                                  'n_time_steps': n_tstep,
                                  'dt': dt,
                                  'include_unsteady_force_contribution': 'on',
                                  'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot'],
                                  'postprocessors_settings': {'BeamLoads': {'csv_output': 'off'},
                                                              'BeamPlot': {'include_rbm': 'on',
                                                                           'include_applied_forces': 'on'},
                                                              'AerogridPlot': {
                                                                  'include_rbm': 'on',
                                                                  'include_applied_forces': 'on',
                                                                  'minus_m_star': 0},
                                                              }}

    settings['LiftDistribution'] = {'folder': route + '/output/',
                                    'normalise': True}
    settings['BeamLoads'] = {'csv_output': True}

    settings['BeamPlot'] = {'include_rbm': 'on',
                            'include_applied_forces': 'on',
                            'include_forward_motion': 'on'}

    settings['AerogridPlot'] = {'include_rbm': 'on',
                                'include_forward_motion': 'off',
                                'include_applied_forces': 'on',
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

    settings['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                    'linear_system_settings': {
                                        'beam_settings': {'modal_projection': False,
                                                          'inout_coords': 'nodes',
                                                          'discrete_time': True,
                                                          'newmark_damp': 0.05,
                                                          'discr_method': 'newmark',
                                                          'dt': dt,
                                                          'proj_modes': 'undamped',
                                                          'use_euler': 'off',
                                                          'num_modes': 40,
                                                          'print_info': 'on',
                                                          'gravity': 'on',
                                                          'remove_dofs': []},
                                        'aero_settings': {'dt': dt,
                                                          'integr_order': 2,
                                                          'density': rho,
                                                          'remove_predictor': False,
                                                          'use_sparse': True,
                                                          'rigid_body_motion': free_flight,
                                                          'use_euler': False,
                                                          'remove_inputs': ['u_gust']},
                                        'rigid_body_motion': free_flight}}

    settings['AsymptoticStability'] = {'sys_id': 'LinearAeroelastic',
                                        'print_info': 'on',
                                        'modes_to_plot': [],
                                        'display_root_locus': 'off',
                                        'frequency_cutoff': 0,
                                        'export_eigenvalues': 'off',
                                        'num_evals': 40,}


    import configobj
    config = configobj.ConfigObj()
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()



clean_test_files()
generate_fem()
generate_aero_file()

generate_nonlifting_body_file()
generate_solver_file()
generate_dyn_file()


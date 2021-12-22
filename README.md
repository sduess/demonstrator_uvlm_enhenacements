# UVLM modeling enhancements and demonstration
This repository contains the scripts for the generation of simulations on the FLEXOP aircraft using SHARPy. These setup has been used for steady-state aeroelastic results of this aircraft that will be presented at the SciTech 2022 Forum in San Diego, US.


## 01) Steady Verification
Generation and post-processing of steady-state aeroelastic results for the clamped FLEXOP aircraft for individual set simulation boundary conditions and settings.

Scripts:
* `generate_steady_results`: Run 'SHARPy' submodule with here generated input files containing the FLEXOP geoemtry and material properties plus individual set simulation settings. Output data is stored in the defined output folder. 
* `postprocess_steady_results`: Generates graphs to compare the achieved results with the ones obtained by [Sodja et al (2021)](https://arc.aiaa.org/doi/10.2514/1.C035955).

### SHARPy submodule
Nonlinear Aeroelastic simulation environment. Find the full documentation [here](https://ic-sharpy.readthedocs.io).
### FLEX OP submodule
Code to setup FLEXOP aircraft in 'SHARPy'.

## 02) FLEXOP Polar Generation
Generation of FLEXOP aircraft polars using airfoil polar data derived from XFOIL

Scripts:
* `generate_xfoil_polars`: Generates the XFOIL polar data for the FLEXOP airfoils and saves them to the FLEXOP submodule polar data folder. This uses the `xfoil_interface` module. See details below on using the provided python build of XFoil.
* `generate_flexop_polars`: Generates the FLEXOP aircraft polars given the settings at the end of the file
* `postprocess_flexop_polars`: Postprocess the forces and deflection of the aircraft cases previously ran and saves them to `polar_output_data`

### XFOIL submodule
`xfoil-python` is included as a submodule and enables to run XFoil viscous solver directly from Python. It is based on a compiled set of Fortran libraries, and the base code has been forked at [github.com/ngoiz/xfoil-python](github.com/ngoiz/xfoil-python). At the moment `pip install -e .` ran from the xfoil directory returns an error when trying to compile the Fortran library. Thus, `xfoil` needs to be added manually to the Python path in the very dirty `sys.path.append(path_to_xfoil)` way. The Fortran library must be compiled creating a `xfoil/build` directory, running `cmake ..` and `make`. A `libxfoil.so` library will be generated in `xfoil/lib` with OS data appended to it. This library then must be renamed to `libxfoil.so` only and moved to `xfoil/xfoil`

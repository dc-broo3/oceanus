Description of the parameters included in the .yaml files

- inpath : Input path to the folder location of input yaml parameters.
- snapname : Specific name of input parameter file.
- outpath : Output path where to save the resulting stream information.
- outname : Specific name of output parameter file.
- Tbegin : Time at which the progenitor begins to be tidally stripped in Gyr (negative).
- Tfinal :  Time at which the progenitor has been fully stripped in Gyr (negative).
- dtmin  : Minimum timestep for leapfrog integration in Gyr.
- prog_mass : Mass of the progenitor in solar masses.
- prog_scale : Scale radius (Plummer model) of the progenitor in kpc.
- prog_ics : Initial positions (in kpc, 0:3) and velocities (in km/s. 3:6) of the progenitor at present day.
- strip_rate : Number of particles ejected at each Lagrange point at each time step. Must be an even integer.
- haloflag : Flag to set the deformations to the MW dark matter halo.
- lmcflag : Flag to set the deformations to the MW disc.
- discflag : Flag to set the deformations to the LMC dark matter halo.
- discframe : Flag to specify whether to integrate in the inertial (False) or discframe (True)
- static_mwh : Flag to specify whether to create a static monopole MW dark matter halo (True/False)

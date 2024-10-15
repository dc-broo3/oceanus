import agama
import astropy.units as u
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
import gala.units as gu
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import sys
import numpy as np

mass_unit =232500
agama.setUnits(length=1, velocity=1, mass=mass_unit)
timeUnitGyr = agama.getUnits()['time'] / 1e3
agama.getUnits()

usys = gu.UnitSystem(u.kpc, 977.79222168*u.Myr, mass_unit*u.Msun, u.radian, u.km/u.s)

Integrators = {'Leapfrog': gi.LeapfrogIntegrator, 
                        'RK4': gi.Ruth4Integrator, 
                        'RK5': gi.RK5Integrator,
                        'DOPRI853': gi.DOPRI853Integrator}

rng = np.random.default_rng(seed=1)

def read_mass_scale_params(paramfile):
    """
    Read in the stream model parameters
    """
    with open(paramfile) as f:
        d = yaml.safe_load(f)
    
    IMF = d["IMF"]
    Noversample = d["Noversample"]
    Nsample = d["Nsample"]
    peri_cut = d["peri_cut"]
    apo_cut = d["apo_cut"]    
    return [IMF, Noversample, Nsample, peri_cut, apo_cut]

def read_pot_params(paramfile):
    """
    Read in the stream model parameters
    """
    with open(paramfile) as f:
        d = yaml.safe_load(f)

    inpath = d["inpath"]
    outpath = d["outpath"]
    Tbegin = d["Tbegin"]
    Tfinal = d["Tfinal"]
    num_particles = d["num_particles"]
    
    return [inpath, outpath, Tbegin, Tfinal, num_particles]

def F_rigid(t, w):
    pothalo = agama.Potential(type = 'Spheroid', mass = 6.5e11/mass_unit, scaleRadius  = 14,
                          outerCutoffRadius = 300,cutoffStrength = 4,gamma = 1,beta = 3)
    potbulge = agama.Potential(type='Spheroid', mass=51600, scaleRadius=0.2, outerCutoffRadius = 1.8,
                              gamma=0.0, beta=1.8, axisRatioZ=1.0)
    potdisc = agama.Potential(type='Disk', SurfaceDensity=3803.5, ScaleRadius=3.0,ScaleHeight= -0.4)
    totpot = agama.Potential(pothalo, potdisc, potbulge)
    
    wdot = np.zeros_like(w)
    wdot[3:] = totpot.force(w[:3].T).T
    wdot[:3] = w[3:]
    return wdot

def make_ics(params_mass_scale):
    
    IMF, Noversample, Nsample, peri_cut, apo_cut = params_mass_scale

    pothalo = agama.Potential(type = 'Spheroid', mass = 6.5e11/mass_unit, scaleRadius  = 14,
                          outerCutoffRadius = 300,cutoffStrength = 4,gamma = 1,beta = 3)

    potbulge = agama.Potential(type='Spheroid', mass=51600, scaleRadius=0.2, outerCutoffRadius = 1.8,
                              gamma=0.0, beta=1.8, axisRatioZ=1.0)
    
    potdisc = agama.Potential(type='Disk', SurfaceDensity=3803.5, ScaleRadius=3.0,ScaleHeight= -0.4)
    
    totpot = agama.Potential(pothalo, potdisc, potbulge)

    dens = agama.Density(type="Dehnen", scaleRadius=15)
    df = agama.DistributionFunction(type="QuasiSpherical", potential=totpot, density=dens)
    gm = agama.GalaxyModel(totpot, df)

    #--------------------------------------------------------------------------------------
    ### Lognormal (final) or truncated lognormal (initial) distribution for the GC masses
    #--------------------------------------------------------------------------------------
    if IMF==True:
        lower_bound = 1e4
        upper_bound = 1e7
        mu_mass_init = np.log(1e4) #solar masses 
        std_mass_init = np.log(3)
        samples_mass_init = rng.lognormal(mu_mass_init, std_mass_init, Noversample)[:,np.newaxis]
        samples_mass = samples_mass_init[(samples_mass_init > lower_bound) & (samples_mass_init < upper_bound)][:,np.newaxis]
        
    elif IMF==False:    
        mu_mass_final = np.log(1e5) #solar masses 
        std_mass_final = np.log(2)
        samples_mass = rng.lognormal(mu_mass_final, std_mass_final, Noversample)[:,np.newaxis]

    #--------------------------------------------------------------------------------------
    ### All GC scale radii set to 2 pc
    #--------------------------------------------------------------------------------------
    samples_scale = np.repeat(0.002, len(samples_mass))[:,np.newaxis]
    prog_mass_scale = np.concatenate([samples_mass, samples_scale], axis=1)

    #--------------------------------------------------------------------------------------
    ### Over-sample from the DF and integrate orbits to be able to find pericenters
    #--------------------------------------------------------------------------------------
    xv = gm.sample(len(samples_mass))[0]
    
    w0 = gd.PhaseSpacePosition.from_w(xv.T, units=usys)
    print("integrating orbits backwards in time...")
    integrator=Integrators['Leapfrog'](F_rigid, func_units=usys, progress=False)
    wf = integrator.run(w0,  dt= 1 * u.Myr, t1=0* u.Gyr, t2= 5 * u.Gyr)
    
    #--------------------------------------------------------------------------------------
    ### Cut on the pericentric and apocentric passage conditions
    #--------------------------------------------------------------------------------------
    print("finding pericenters and apocenters...")
    pericenters = wf.pericenter()
    apocenters = wf.apocenter()
    peri_mask = (pericenters > peri_cut[0]*u.kpc) & (pericenters < peri_cut[1]*u.kpc)
    apo_mask = (apocenters < apo_cut*u.kpc)
    
    print("cutting to sample DF...")
    cut_mass_scales = prog_mass_scale[peri_mask & apo_mask]
    mass_scales = rng.choice(cut_mass_scales, size=Nsample)
    
    cut_xv = xv[peri_mask & apo_mask]
    print("The number of streams passing critera: {}".format(len(cut_xv)))
    prog_ics = rng.choice(cut_xv, size=Nsample)
    
    print("re-finding pericenters and apocenters of cut sample...")
    w0 = gd.PhaseSpacePosition.from_w(prog_ics.T, units=usys)
    wf2 = integrator.run(w0,  dt= 1 * u.Myr, t1=0* u.Gyr, t2= 5 * u.Gyr)
    peris = wf2.pericenter()
    apos = wf2.apocenter()

    return prog_ics, mass_scales, peris, apos

def streamparams(ics, mass_scales, peris, apos):
    
    exts = "agama-mw"
    gens = "gen-params-agama.yaml"
    path = "/mnt/ceph/users/rbrooks/oceanus/ics/generation-files/"

    print("* Saving data in {}".format("/mnt/ceph/users/rbrooks/oceanus/ics/high-velids/" + exts))
    inpath, outpath, Tbegin, Tfinal, num_particles  = read_pot_params(path + gens)
    for i in range(len(ics)):
        yaml_data = {'Tbegin': Tbegin, 
                     'Tfinal': Tfinal, 
                     'prog_mass': mass_scales[i].tolist()[0],
                     'prog_scale': mass_scales[i].tolist()[1], 
                     'prog_ics': ics[i].tolist(),
                     'pericenter': peris[i].value.tolist(),
                     'apocenter': apos[i].value.tolist(),
                     'num_particles': num_particles,
                    'inpath':inpath,
                    'snapname':str("param_{}".format(i)),
                    'outpath':outpath,
                    'outname':str("stream_{}".format(i))}
        file_name = f"/mnt/home/rbrooks/ceph/oceanus/ics/high-vel-dis/{exts}/param_{i}.yaml"
        with open(file_name, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)
#-----------------------------------------------------------------------------------------    
# Run the script - python ics-agama.py ../ics/generation-files/mass-scale-agama.yaml
#-------------------------------------------------------------------------------------------------

params_mass_scales = read_mass_scale_params(sys.argv[1])   
prog_ics, mass_scales, peris, apos = make_ics(params_mass_scales)
streamparams(prog_ics, mass_scales, peris, apos)
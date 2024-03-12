import agama
import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import sys

import numpy as np
from gala.units import galactic
agama.setUnits(length=u.kpc, mass=u.Msun, time=u.Myr)

rng = np.random.default_rng(seed=1)

def make_ics(params):
    
    ext, inpath, outpath, Tbegin, Tfinal, dtmin, \
    haloflag, discflag, lmcflag, strip_rate, \
    discframe, static_mwh, lmc_switch, Noversample, Nsample, peri_cut, apo_cut = params

    #--------------------------------------------------------------------------------------
    ### Potential matching L23 (r_s = 12.8kpc, c = 15.3, v_c_solar = 184 km/s, V_vir = ˜130 km/s)
    #--------------------------------------------------------------------------------------
    gala_pot = gp.NFWPotential.from_circular_velocity(v_c=184 * u.km / u.s, r_s=8.249 * u.kpc, units=galactic)
    agama_pot = gala_pot.as_interop("agama")

    dens = agama.Density(type="Dehnen", scaleRadius=15)
    df = agama.DistributionFunction(type="QuasiSpherical", potential=agama_pot, density=dens)
    gm = agama.GalaxyModel(agama_pot, df)

    #--------------------------------------------------------------------------------------
    ### Lognormal distribution for the GC masses
    #--------------------------------------------------------------------------------------
    mu_mass = np.log(1e5) #solar masses 
    std_mass = np.log(3)
    samples_mass = rng.lognormal(mu_mass, std_mass, Noversample)[:,np.newaxis]

    #--------------------------------------------------------------------------------------
    ### Normal distribution for the GC scale radii
    #--------------------------------------------------------------------------------------
    mu_scale = np.log(1e-1) #kpc
    std_scale = np.log(2e0)
    samples_scale = rng.lognormal(mu_scale, std_scale, Noversample)[:,np.newaxis]

    prog_mass_scale = np.concatenate([samples_mass, samples_scale], axis=1)

    #--------------------------------------------------------------------------------------
    ### Over-sample from the DF and integrate orbits to be able to find pericenters
    #--------------------------------------------------------------------------------------
    xv = gm.sample(Noversample)[0]
    w0 = gd.PhaseSpacePosition.from_w(xv.T, units=galactic)
    wf = gala_pot.integrate_orbit(w0, dt=1 * u.Myr, t1=0, t2=6 * u.Gyr, store_all=True)

    #--------------------------------------------------------------------------------------
    ### Cut on the pericentric and apocentric passage conditions
    #--------------------------------------------------------------------------------------
    pericenters = wf.pericenter()
    apocenters = wf.apocenter()
    peri_mask = (pericenters < peri_cut[1]*u.kpc) & (pericenters < peri_cut[0]*u.kpc)
    apo_mask = (apocenters < apo_cut*u.kpc)
    
    pericenters_sample = pericenters[peri_mask]
    apocenters_sample = apocenters[apo_mask]
    
    peri_cut_sample = prog_mass_scale[peri_mask & apo_mask]
    prog_ics = rng.choice(peri_cut_sample, size=Nsample)

    #--------------------------------------------------------------------------------------
    ### Save the initial conditions data for the progenitors into a yaml file
    #--------------------------------------------------------------------------------------
    # for p in range(len(potentials)): something like this to run over each potential and just change flags, etc. Same x,v, M, a_s
    for i in range(len(prog_ics)):
        yaml_data = {'Tbegin': Tbegin, 
                     'Tfinal': Tfinal, 
                     'dtmin': dtmin, 
                     'prog_mass': prog_ics[i].tolist()[0],
                     'prog_scale': prog_ics[i].tolist()[1], 
                     'prog_ics':xv[i].tolist(),
                     'pericenter':pericenters_sample[i].value.tolist(),
                     'apocenter':apocenters_sample[i].value.tolist(),
                     'strip_rate': strip_rate,
                     'haloflag': haloflag,
                     'lmcflag': lmcflag,
                     'discflag': discflag,
                     'discframe': discframe,
                     'static_mwh': static_mwh,
                     'lmc_switch': lmc_switch, 
                    'inpath':inpath,
                    'snapname':str("param_{}".format(i)),
                    'outpath':outpath,
                    'outname':str("stream_{}".format(i))}
        file_name = f"/mnt/home/rbrooks/ceph/oceanus/ics/{ext}/param_{i}.yaml"
        with open(file_name, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)
                   
def readgenparams(paramfile):
    """
    Read in the stream model parameters
    """
    with open(paramfile) as f:
        d = yaml.safe_load(f)
    
    ext = d["ext"]
    inpath = d["inpath"]
    outpath = d["outpath"]
    Tbegin = d["Tbegin"]
    Tfinal = d["Tfinal"]
    dtmin  = d["dtmin"]
    haloflag = d["haloflag"]
    lmcflag = d["lmcflag"]
    discflag = d["discflag"]
    strip_rate = d["strip_rate"]
    discframe = d["discframe"]
    static_mwh = d["static_mwh"]
    lmc_switch = d["lmc_switch"]
    Noversample = d["Noversample"]
    Nsample = d["Nsample"]
    peri_cut = d["peri_cut"]
    apo_cut = d["apo_cut"]
    
    return [ext, inpath, outpath, Tbegin, Tfinal, dtmin, 
           haloflag, discflag, lmcflag, strip_rate, discframe, static_mwh, lmc_switch,
           Noversample, Nsample, peri_cut, apo_cut]
        
#-----------------------------------------------------------------------------------------    
# Run the script
#-----------------------------------------------------------------------------------------  

params = readgenparams(sys.argv[1])    
make_ics(params)
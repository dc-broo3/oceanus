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
    dtmin  = d["dtmin"]
    haloflag = d["haloflag"]
    lmcflag = d["lmcflag"]
    discflag = d["discflag"]
    strip_rate = d["strip_rate"]
    discframe = d["discframe"]
    static_mwh = d["static_mwh"]
    mwd_switch = d["mwd_switch"]
    lmc_switch = d["lmc_switch"]
    
    return [inpath, outpath, Tbegin, Tfinal, dtmin, 
           haloflag, discflag, lmcflag, strip_rate, discframe, static_mwh, mwd_switch, lmc_switch]

def make_ics(params_mass_scale):
    
    IMF, Noversample, Nsample, peri_cut, apo_cut = params_mass_scale

    gala_pot = gp.NFWPotential.from_circular_velocity(v_c=240 * u.km / u.s, r_s=10 * u.kpc, units=galactic)
    agama_pot = gala_pot.as_interop("agama")

    dens = agama.Density(type="Dehnen", scaleRadius=15)
    df = agama.DistributionFunction(type="QuasiSpherical", potential=agama_pot, density=dens)
    gm = agama.GalaxyModel(agama_pot, df)

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
    w0 = gd.PhaseSpacePosition.from_w(xv.T, units=galactic)
    wf = gala_pot.integrate_orbit(w0, dt=1. * u.Myr, t1=0, t2=6 * u.Gyr, store_all=True) 

    #--------------------------------------------------------------------------------------
    ### Cut on the pericentric and apocentric passage conditions
    #--------------------------------------------------------------------------------------
    pericenters = wf.pericenter()
    apocenters = wf.apocenter()
    peri_mask = (pericenters > peri_cut[0]*u.kpc) & (pericenters < peri_cut[1]*u.kpc)
    apo_mask = (apocenters < apo_cut*u.kpc)
    
    cut_mass_scales = prog_mass_scale[peri_mask & apo_mask]
    mass_scales = rng.choice(cut_mass_scales, size=Nsample)
    
    cut_prog_ics = xv[peri_mask & apo_mask]
    prog_ics = rng.choice(cut_prog_ics, size=Nsample)
    
    peris = pericenters[peri_mask & apo_mask]
    apos = apocenters[peri_mask & apo_mask]

    return prog_ics, mass_scales, peris, apos


def streamparams(ics, mass_scales, peris, apos):
    
    exts = ["static-mwh-only","rm-mwh-full-mwd-full-lmc", "em-mwh-full-mwd-full-lmc", "md-mwh-full-mwd-full-lmc", \
          "mq-mwh-full-mwd-full-lmc", "mdq-mwh-full-mwd-full-lmc",  "full-mwh-full-mwd-full-lmc", "full-mwh-full-mwd-no-lmc"]
    
    gens = ["gen-params-static-mwh-only.yaml", "gen-params-rm-mwh-mwd-lmc.yaml", "gen-params-em-mwh-mwd-lmc.yaml", \
            "gen-params-md-mwh-mwd-lmc.yaml", "gen-params-mq-mwh-mwd-lmc.yaml", "gen-params-mdq-mwh-mwd-lmc.yaml", \
            "gen-params-full-mwh-mwd-lmc.yaml", "gen-params-full-mwh-mwd-no-lmc.yaml"]
    
    path = "/mnt/ceph/users/rbrooks/oceanus/ics/generation-files/"
    
    for j in range(len(exts)): 
        
        print("* Saving data in {}".format("/mnt/ceph/users/rbrooks/oceanus/ics/" + exts[j]))
          
        inpath, outpath, Tbegin, Tfinal, dtmin, \
        haloflag, discflag, lmcflag, strip_rate, \
        discframe, static_mwh, mwd_switch, lmc_switch = read_pot_params(path + gens[j])
            
        for i in range(len(ics)):
            yaml_data = {'Tbegin': Tbegin, 
                         'Tfinal': Tfinal, 
                         'dtmin': dtmin, 
                         'prog_mass': mass_scales[i].tolist()[0],
                         'prog_scale': mass_scales[i].tolist()[1], 
                         'prog_ics': ics[i].tolist(),
                         'pericenter': peris[i].value.tolist(),
                         'apocenter': apos[i].value.tolist(),
                         'strip_rate': strip_rate,
                         'haloflag': haloflag,
                         'lmcflag': lmcflag,
                         'discflag': discflag,
                         'discframe': discframe,
                         'static_mwh': static_mwh,
                         'mwd_switch': mwd_switch,
                         'lmc_switch': lmc_switch, 
                        'inpath':inpath,
                        'snapname':str("param_{}".format(i)),
                        'outpath':outpath,
                        'outname':str("stream_{}".format(i))}
            file_name = f"/mnt/home/rbrooks/ceph/oceanus/ics/{exts[j]}/param_{i}.yaml"
            with open(file_name, 'w') as yaml_file:
                yaml.dump(yaml_data, yaml_file, default_flow_style=False)
#-----------------------------------------------------------------------------------------    
# Run the script - python initialconditions.py ../ics/generation-files/mass-scale.yaml
#-----------------------------------------------------------------------------------------  

params_mass_scales = read_mass_scale_params(sys.argv[1])   
prog_ics, mass_scales, peris, apos = make_ics(params_mass_scales)

streamparams(prog_ics, mass_scales, peris, apos)
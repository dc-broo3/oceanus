import numpy as np
import astropy.units as u
import yaml
import h5py
from astropy.constants import G
import os.path
import sys
from argparse import ArgumentParser
import pathlib
from mwlmc import model as mwlmc_model
Model = mwlmc_model.MWLMC()

def numerical_forceDerivs(positions, ts, mwhflag, mwdflag, lmcflag, epsilon=1e-3):
    
    """
    numerical_forceDerivs - takes the positions for the mock stream progenitor and returns the derivatives of the forces for each position 
                    and second derivative of the potential w.r.t the positons.
    
    Inputs
    - positions: The positions of the progenitor in kpc, Shape N x 3.
    - ts: Time steps corresponding to each position in Gyr.
    - mwhflag: flag to set which mw halo expansion orders to be non-zero.
    - mwdflag: flag to set which mw disc expansion orders to be non-zero.
    - lmcflag: flag to set which lmc halo expansion orders to be non-zero.
    - epsilon: (optional) The small value away from each position used find the derivative in kpc. Default is 1e-2 kpc.
    
    Retuns
    - Hess: The Hessian matrix of all force derivatives. Shape (len(times), 3, 3). ((xx, xy, xz), 
                                                                                    (yx, yy, yz), 
                                                                                    (zx, zy, zz))
    - d2Phi_d2r: The second derivative of the potential with repect to the position.
    """
    
    r_prog = np.linalg.norm(positions, axis=1)
    
    fxx_yx_zx = np.zeros(shape=(len(ts),3))
    fxy_yy_zy = np.zeros(shape=(len(ts),3))
    fxz_yz_zz = np.zeros(shape=(len(ts),3))

    for i in range(len(ts)):

        fxx_yx_zx[i] = (np.array(Model.all_forces(t=ts[i], x=positions[:,0][i]+epsilon, y=positions[:,1][i], z=positions[:,2][i],
                              mwhharmonicflag=mwhflag, 
                              mwdharmonicflag=mwdflag,
                              lmcharmonicflag=lmcflag)) \
                - np.array(Model.all_forces(t=ts[i], x=positions[:,0][i], y=positions[:,1][i], z=positions[:,2][i], 
                            mwhharmonicflag=mwhflag, 
                              mwdharmonicflag=mwdflag,
                              lmcharmonicflag=lmcflag)) ) \
              / (mag_(np.array([positions[:,0][i]+epsilon,positions[:,1][i], positions[:,2][i]]) - \
                     np.array([positions[:,0][i],positions[:,1][i], positions[:,2][i]])))
        
        fxy_yy_zy[i] = (np.array(Model.all_forces(t=ts[i], x=positions[:,0][i], y=positions[:,1][i]+epsilon, z=positions[:,2][i], 
                              mwhharmonicflag=mwhflag, 
                              mwdharmonicflag=mwdflag,
                              lmcharmonicflag=lmcflag)) \
                - np.array(Model.all_forces(t=ts[i], x=positions[:,0][i], y=positions[:,1][i], z=positions[:,2][i],
                              mwhharmonicflag=mwhflag, 
                              mwdharmonicflag=mwdflag,
                              lmcharmonicflag=lmcflag)) )  \
              / (mag_(np.array([positions[:,0][i],positions[:,1][i]+epsilon, positions[:,2][i]]) - \
                     np.array([positions[:,0][i],positions[:,1][i], positions[:,2][i]])))
        
        fxz_yz_zz[i] = (np.array(Model.all_forces(t=ts[i], x=positions[:,0][i], y=positions[:,1][i], z=positions[:,2][i]+epsilon,
                              mwhharmonicflag=mwhflag, 
                              mwdharmonicflag=mwdflag,
                              lmcharmonicflag=lmcflag)) \
                - np.array(Model.all_forces(t=ts[i], x=positions[:,0][i], y=positions[:,1][i], z=positions[:,2][i],
                              mwhharmonicflag=mwhflag, 
                              mwdharmonicflag=mwdflag,
                              lmcharmonicflag=lmcflag)) ) \
              / (mag_(np.array([positions[:,0][i],positions[:,1][i], positions[:,2][i]+epsilon]) - \
                     np.array([positions[:,0][i],positions[:,1][i], positions[:,2][i]])))
        
    Hess = np.zeros((len(ts), 3, 3))
    Hess[:, 0, :] = -np.array([fxx_yx_zx[:, 0], fxy_yy_zy[:, 0], fxz_yz_zz[:, 0]]).T
    Hess[:, 1, :] = -np.array([fxx_yx_zx[:, 1], fxy_yy_zy[:, 1], fxz_yz_zz[:, 1]]).T
    Hess[:, 2, :] = -np.array([fxx_yx_zx[:, 2], fxy_yy_zy[:, 2], fxz_yz_zz[:, 2]]).T
    
    r_hat = positions / r_prog[:,None]
    d2Phi_d2r = np.einsum('ki,kij,kj->k', r_hat, Hess, r_hat)
    
    return Hess, d2Phi_d2r

def plummer_force(r, m, b):
    G = 4.3e-6 #kpc/Msun * (km/s)^2
    dpot_dr = (G*m*r) / (r**2 + b**2)**1.5
    return -dpot_dr

def mag_(x):
    """
    mag_ - returns the sum in quadrature of a set of values.
    """
    return np.sqrt(np.sum(x**2)) #np.linalg.norm 

def lagrange_cloud_strip_adT(params, overwrite):  
    
    inpath, snapname, outpath, filename, \
    fc, Mprog, a_s, pericenter, apocenter, Tbegin, Tfinal, dtmin, \
    mwhflag, mwdflag, lmcflag, strip_rate, \
    discframe, static_mwh, lmc_switch = params
  
    fullfile_path = pathlib.Path(outpath) / filename

    if fullfile_path.exists() and not overwrite:
        return 
    #FIX THESE - astropy units 
    Gyr_to_s = 3.15576e16
    kpc_Gyr2_to_km_s2 = 3.0984366e-17
    km_to_kpc = 3.2407793e-17
    lambda_source = 1.2 # the multiplier of how far away from the tidal radius to strip from.
    nu=0.01
    max_steps = int((Tfinal - Tbegin) / dtmin) + 2 #allow for overrun
    max_particles = int((max_steps*strip_rate) + 1)
    
    # Model.reset_all_coefficients() - NEED TO BE ABLE TO ADD THIS IN HERE. CURRENTLY KILLS THE KERNEL WHEN CALLING A SUBSEQUENT MODEL.() FUNCTION
    # HAVE SUBMITTED A GITHUB ISSUE ON THIS
    if static_mwh==True:
        _, MWHcoeffs = Model.return_mw_coefficients()
        MWHcoeffs = np.array(MWHcoeffs)
        MWHcoeffs[:,0] = MWHcoeffs[:,0][0] 
        Model.install_mw_coefficients(MWHcoeffs)
        
    if lmc_switch==True:
        _, LMCcoeffs = Model.return_lmc_coefficients()
        LMCcoeffs = np.array(LMCcoeffs)
        LMCcoeffs *= 0 
        Model.install_lmc_coefficients(LMCcoeffs)
        
    orbit  = Model.rewind(fc[:3], fc[3:6],
                     mwhharmonicflag=mwhflag, 
                     mwdharmonicflag=mwdflag,
                     lmcharmonicflag=lmcflag,
                     rewindtime=np.abs(Tbegin),
                     dt=dtmin, 
                     discframe=discframe)
    prog_orbit = np.array(orbit)
    rewind_xs = prog_orbit[0:3]
    rewind_vs = prog_orbit[3:6]
    prog_fc = np.concatenate([rewind_xs.T[-1], rewind_vs.T[-1]]).reshape(1,6)
    
    xs_data = np.full(shape=(max_steps, max_particles, 3), fill_value=np.nan)
    vs_data = np.zeros(shape=(max_steps, max_particles, 3))
    ts = np.zeros(shape=(max_steps, ))

    prog_orbit = np.array(orbit)
    rewind_xs = prog_orbit[0:3]
    rewind_vs = prog_orbit[3:6]
    prog_fc = np.concatenate([rewind_xs.T[-1], rewind_vs.T[-1]]).reshape(1,6)

    xs = np.array(prog_fc[:,0:3])
    vs = -np.array(prog_fc[:,3:6])
    
    ts = np.array([Tbegin])
    t = ts[0]
    
    i = 0
    while t < Tfinal:
      
        prog_xs = xs[:, 0:3][0].reshape(1,3)
        prog_vs = vs[:, 0:3][0].reshape(1,3)

        # Calculate inital angular speed of progenitor (in units of km/s/kpc)
        r_prog = np.linalg.norm(prog_xs[0:3])
        L_prog = np.linalg.norm(np.cross(prog_xs[0:3], prog_vs[0:3]))
        Omega_prog = L_prog / r_prog**2
        r_hat = prog_xs[0:3]/r_prog
        num_hessian, num_d2Phi_dr2 = numerical_forceDerivs(prog_xs, np.array([t]), mwhflag, mwdflag, lmcflag,)

        # Calculate tidal radius
        r_t_max = a_s # Cap on tidal radius (used to replace nans).
        r_t = np.nan_to_num( (((G.to(u.kpc*(u.km/u.s)**2/u.Msun)* Mprog / (Omega_prog**2 - num_d2Phi_dr2))*u.Msun/(u.km/u.s/u.kpc)**2)**(1/3)).to(u.kpc).value, nan=r_t_max)
        r_t =  r_t * (1 - (np.abs(Tbegin) - np.abs(t))/np.abs(Tbegin) )**(1/3) 
        
        sigma_s = ( (G.to(u.kpc*(u.km/u.s)**2/u.Msun).value * Mprog * (1 - (np.abs(Tbegin) - np.abs(t))/np.abs(Tbegin)) ) / \
                    (r_t**2 + a_s**2)**0.5 )**0.5 #+ a_s**2

        # Calculate positions and velocities of points of particle release
        source_coords_in = prog_xs[0:3] - lambda_source * r_t[:, None] * r_hat
        source_coords_out = prog_xs[0:3] + lambda_source * r_t[:, None] * r_hat

        prog_velocity_r = np.sum(prog_vs[0:3]*r_hat)
        prog_velocity_tan = prog_vs[0:3] - prog_velocity_r * r_hat

        source_velocity_tan_in = prog_velocity_tan * (1 - 0.5 * r_t / r_prog)[:, None]
        source_velocity_tan_out = prog_velocity_tan * (1 + 0.5 * r_t / r_prog)[:, None]

        source_velocity_in = prog_velocity_r * r_hat + source_velocity_tan_in
        source_velocity_out = prog_velocity_r * r_hat + source_velocity_tan_out

        source_coords = np.zeros((len(source_coords_in)*2, 3))
        source_coords[::2] = source_coords_in
        source_coords[1::2] = source_coords_out

        source_velocity = np.zeros((len(source_velocity_in)*2, 3))
        source_velocity[::2] = source_velocity_in
        source_velocity[1::2] = source_velocity_out

        ic_source_coords = np.repeat(source_coords, strip_rate/2, axis=0)
        ic_source_velocities = np.repeat(source_velocity, strip_rate/2, axis=0)
 
        np.random.seed(0)
        xs = np.append(xs, ic_source_coords, axis=0)
        vs = np.append(vs, (ic_source_velocities + np.random.randn(len(ic_source_velocities), 3)*sigma_s), axis=0)
        
        #-----------------------------------------------------------------------------------------------------
        # LEAPFROG INTEGRATION
        #-----------------------------------------------------------------------------------------------------

        accs_mwhalo = Model.all_forces(t, x=xs[:,0], y=xs[:,1], z=xs[:,2],
                                      mwhharmonicflag=mwhflag, 
                                      mwdharmonicflag=mwdflag,
                                      lmcharmonicflag=lmcflag)
        accs_prog = plummer_force(xs - xs[0], Mprog*(1 - (np.abs(Tbegin) - np.abs(t))/np.abs(Tbegin)), a_s)
        accs = accs_mwhalo + accs_prog
           
        ## Adapative timestep
        # dt_adapt = np.nanmin([np.nanmin(nu*np.sqrt(np.sum(xs**2, axis=1)**0.5 / \
        #                                                np.sum(accs_mwhalo**2, axis=1)**0.5)), \
        #                       np.nanmin(nu*np.sqrt(np.sum((xs - prog_xs)[1:]**2, axis=1)**0.5 / \
        #                                               np.sum(accs_prog[1:]**2, axis=1)**0.5)) ])
        # if dt_adapt < dtmin:
        #     dt_adapt = dtmin
        dt_adapt = dtmin
        
     
        ts = np.append(ts, t+dt_adapt)
        t = t + dt_adapt
       
        v_halfs = vs + (accs*dt_adapt/2)
        xs = xs + v_halfs*(dt_adapt*Gyr_to_s*km_to_kpc)
        
        new_accs_mwhalo = Model.all_forces(t, x=xs[:,0], y=xs[:,1], z=xs[:,2],
                                          mwhharmonicflag=mwhflag, 
                                          mwdharmonicflag=mwdflag,
                                          lmcharmonicflag=lmcflag)
        new_accs_prog = plummer_force(xs - xs[0], Mprog*(1 - (np.abs(Tbegin) - np.abs(t))/np.abs(Tbegin)), a_s)
        new_accs = new_accs_mwhalo + new_accs_prog
        
        vs =  v_halfs + (new_accs*dt_adapt/2)

        xs_data[i], vs_data[i] = fill_with_zeros(xs, max_particles), fill_with_zeros(vs, max_particles)
        i = i + 1
    
    ts = np.repeat(ts, strip_rate)[:-strip_rate+1]
    ts = fill_with_nans_1d(ts, max_particles)
    
    mask = np.all(np.isnan(xs_data), axis=(1,2)) 
    
    # Save only every 100th time snapshot
    xs_snaps = xs_data[~mask][::100]
    vs_snaps = vs_data[~mask][::100]
    
    pot_label = harmonicflags_to_potlabel(mwhflag, mwdflag, lmcflag, static_mwh)    
    write_stream_hdf5(outpath, filename, 
                      xs_snaps, 
                      vs_snaps, 
                      ts,
                     pot_label, fc, Mprog, a_s, pericenter, apocenter, discframe)
    
    # return xs_data, vs_data, ts

def readparams(paramfile):
    """
    Read in the stream model parameters
    """
    with open(paramfile) as f:
        d = yaml.safe_load(f)
    
    inpath = d["inpath"]
    snapname = d["snapname"]
    outpath = d["outpath"]
    outname = d["outname"]
    prog_ics = np.array(d["prog_ics"])
    prog_mass = d["prog_mass"]
    prog_scale = d["prog_scale"]
    pericenter = d["pericenter"]
    apocenter = d["apocenter"]
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

    assert type(inpath)==str, "inpath parameter  must be a string"
    assert type(snapname)==str, "snapname parameter must be a string"
    assert type(outpath)==str, "outpath parameter must be a string"
    assert type(outname)==str, "outname parameter must be a string"
    # assert type(prog_ics)==list, "prog_scale parameter must be a "
    assert type(prog_mass)==float, "prog_mass parameter must be a float"
    assert type(prog_scale)==float, "prog_scale parameter must be a float"
    assert type(Tbegin)==float, "Tbegin parameter must be an float"
    assert type(Tfinal)==float, "Tfinal parameter must be an float"
    assert type(dtmin)==float, "dtmin parameter must be an float"
    assert type(haloflag)==int, "haloflag parameter must be an int"
    assert type(lmcflag)==int, "lmcflag parameter must be an int"
    assert type(discflag)==int, "discflag parameter must be an int"
    assert type(strip_rate)==int, "strip_rate parameter must be an int"
   
    return [inpath, snapname, outpath, outname, prog_ics ,prog_mass, prog_scale, pericenter, apocenter, Tbegin, Tfinal, dtmin, 
           haloflag, discflag, lmcflag, strip_rate, discframe, static_mwh, lmc_switch]

def write_stream_hdf5(outpath, filename, positions, velocities, times, potential, progics, progmass, progscale, pericenter, apocenter, frame):
    """
    Write stream into an hdf5 file
    
    """
    
    tmax = positions.shape[0]
    particlemax = positions.shape[1]
    
    print("* Writing stream: {}, for potential: {}".format(filename, potential))
    hf = h5py.File(outpath + filename + ".hdf5", 'w')
    hf.create_dataset('positions', data=positions, shape=(tmax, particlemax, 3))
    hf.create_dataset('velocities', data=velocities, shape=(tmax, particlemax, 3))
    hf.create_dataset('times', data=times)
    hf.create_dataset('potential', data=potential)
    hf.create_dataset('progenitor-ics', data=progics)
    hf.create_dataset('progenitor-mass', data=progmass)
    hf.create_dataset('progenitor-scale', data=progscale)
    hf.create_dataset('pericenter',data=pericenter)
    hf.create_dataset('apocenter', data=apocenter)
    hf.create_dataset('frame-of-reference', data=frame)
    #... flags to names for what ics we have used 
    hf.close()
    
def fill_with_zeros(arr, m):
    n = arr.shape[0]
    if m <= n:
        return arr
    else:
        filled_arr = np.zeros((m, 3))
        filled_arr[:n, :] = arr
        return filled_arr
    
def fill_with_nans_1d(arr, m):
    n = arr.shape[0]
    if m <= n:
        return arr
    else:
        filled_arr = np.full(m, np.nan)
        filled_arr[:n] = arr
        return filled_arr
    
    
def harmonicflags_to_potlabel(mwhflag, mwdflag, lmcflag, mwh_static):
    
    if mwhflag==0 and mwh_static==True and mwdflag==63 and lmcflag==63:
        label = 'rm-MWhalo-full-MWdisc-full-LMC'
    
    elif mwhflag==0 and mwh_static==False and mwdflag==63 and lmcflag==63:
        label = 'em-MWhalo-full-MWdisc-full-LMC'
    
    elif mwhflag==1 and mwdflag==63 and lmcflag==63:
        label = 'md-MWhalo-full-MWdisc-full-LMC'
        
    elif mwhflag==2 and mwdflag==63 and lmcflag==63:
        label = 'mq-MWhalo-full-MWdisc-full-LMC'
        
    elif mwhflag==3 and mwdflag==63 and lmcflag==63:
        label = 'mdq-MWhalo-full-MWdisc-full-LMC'
        
    elif mwhflag==63 and mwdflag==63 and lmcflag==63:
        label = 'Full-MWhalo-MWdisc-LMC'
        
    elif mwhflag==63 and mwdflag==63 and lmcflag==0:
        label = 'full-MWhalo-full-MWdisc-no-LMC'
      
    return label
    
#-----------------------------------------------------------------------------------------    
### Run the script
#-----------------------------------------------------------------------------------------  

parser = ArgumentParser()

parser.add_argument("-f", dest="param_file", required=True)

parser.add_argument("-o", "--overwrite", dest="overwrite", action="store_true", default=False)

args = parser.parse_args()

params = readparams(args.param_file)    

# change this to argparse (built in package): filename and flag (-o or --overwrite)

lagrange_cloud_strip_adT(params, overwrite=args.overwrite)
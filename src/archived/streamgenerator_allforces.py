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

gyr = 1*u.Gyr
km = 1*u.km

def plummer_force(r, m, b):
    newG = G.to(u.kpc / u.Msun * (u.km/u.s)**2)
    dpot_dr = (newG.value * m * r) / (r**2 + b**2)**1.5
    force = -dpot_dr
    return force

def numerical_forceDerivs(positions, ts, mwhflag, mwdflag, lmcflag, epsilon=1e-4):
    
    """
    numerical_forceDerivs - takes the positions for the mock stream progenitor and returns the derivatives of the forces for each position 
                    and second derivative of the potential w.r.t the positons.
    
    Inputs
    - positions: The positions of the progenitor in kpc, Shape N x 3.
    - ts: Time steps corresponding to each position in Gyr.
    - mwhflag: flag to set which mw halo expansion orders to be non-zero.
    - mwdflag: flag to set which mw disc expansion orders to be non-zero.
    - lmcflag: flag to set which lmc halo expansion orders to be non-zero.
    - epsilon: (optional) The small value away from each position used find the derivative in kpc. Default is 1e-3 kpc.
    
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
              / np.linalg.norm(np.array([positions[:,0][i]+epsilon,positions[:,1][i], positions[:,2][i]]) - \
                     np.array([positions[:,0][i],positions[:,1][i], positions[:,2][i]]))
        
        fxy_yy_zy[i] = (np.array(Model.all_forces(t=ts[i], x=positions[:,0][i], y=positions[:,1][i]+epsilon, z=positions[:,2][i], 
                              mwhharmonicflag=mwhflag, 
                              mwdharmonicflag=mwdflag,
                              lmcharmonicflag=lmcflag)) \
                - np.array(Model.all_forces(t=ts[i], x=positions[:,0][i], y=positions[:,1][i], z=positions[:,2][i],
                              mwhharmonicflag=mwhflag, 
                              mwdharmonicflag=mwdflag,
                              lmcharmonicflag=lmcflag)) )  \
              / np.linalg.norm((np.array([positions[:,0][i],positions[:,1][i]+epsilon, positions[:,2][i]]) - \
                     np.array([positions[:,0][i],positions[:,1][i], positions[:,2][i]])))
        
        fxz_yz_zz[i] = (np.array(Model.all_forces(t=ts[i], x=positions[:,0][i], y=positions[:,1][i], z=positions[:,2][i]+epsilon,
                              mwhharmonicflag=mwhflag, 
                              mwdharmonicflag=mwdflag,
                              lmcharmonicflag=lmcflag)) \
                - np.array(Model.all_forces(t=ts[i], x=positions[:,0][i], y=positions[:,1][i], z=positions[:,2][i],
                              mwhharmonicflag=mwhflag, 
                              mwdharmonicflag=mwdflag,
                              lmcharmonicflag=lmcflag)) ) \
              / np.linalg.norm((np.array([positions[:,0][i],positions[:,1][i], positions[:,2][i]+epsilon]) - \
                     np.array([positions[:,0][i],positions[:,1][i], positions[:,2][i]])))
        
    Hess = np.zeros((len(ts), 3, 3))
    Hess[:, 0, :] = -np.array([fxx_yx_zx[:, 0], fxy_yy_zy[:, 0], fxz_yz_zz[:, 0]]).T
    Hess[:, 1, :] = -np.array([fxx_yx_zx[:, 1], fxy_yy_zy[:, 1], fxz_yz_zz[:, 1]]).T
    Hess[:, 2, :] = -np.array([fxx_yx_zx[:, 2], fxy_yy_zy[:, 2], fxz_yz_zz[:, 2]]).T
    
    r_hat = positions / r_prog[:,None]
    d2Phi_d2r = np.einsum('ki,kij,kj->k', r_hat, Hess, r_hat)
    
    return d2Phi_d2r

def rewind_leap(fc_xs, fc_vs, mwhharmonicflag, mwdharmonicflag, lmcharmonicflag, Tend, dt):
    
    max_steps = int((np.abs(Tend) - 0.) / dt) + 2
    xs_orbit = np.zeros(shape=(max_steps,3))*u.kpc
    vs_orbit = np.zeros(shape=(max_steps,3))*(u.km/u.s) 
    
    mwd_v0 = np.array(Model.expansion_centre_velocities(0.)[9:12])*(u.km/u.s) 
    
    prog_xs0, prog_vs0 = fc_xs, -(fc_vs+mwd_v0)
    xs_orbit[0] = prog_xs0
    vs_orbit[0] = prog_vs0
    
    ts = np.array([0.])
    t = ts[0] 
    
    i=0
    while t > Tend:
        
        mwd_x_i = 0*np.array(Model.expansion_centres(t)[0:3])
        accs = Model.all_forces(t, x=xs_orbit[i][0].value + mwd_x_i[0], 
                                       y=xs_orbit[i][1].value + mwd_x_i[1], 
                                       z=xs_orbit[i][2].value + mwd_x_i[2],
                                      mwhharmonicflag=mwhharmonicflag, 
                                      mwdharmonicflag=mwdharmonicflag,
                                      lmcharmonicflag=lmcharmonicflag) * (u.km**2/(u.s**2*u.kpc))

        xs_orbit[i+1] = xs_orbit[i] + (vs_orbit[i] * (dt*u.Gyr).to(u.s)).to(u.kpc) + (0.5 * accs * ((dt*u.Gyr).to(u.s))**2).to(u.kpc)
   
        ts = np.append(ts, t - dt)
    
        t = t - dt
        mwd_x_ii = 0*np.array(Model.expansion_centres(t)[0:3])
        # Correct for already accounting for disc motion at the last timestep, adjust to current timestep
        new_accs = Model.all_forces(t, x=xs_orbit[i+1][0].value + mwd_x_ii[0] - mwd_x_i[0],
                                       y=xs_orbit[i+1][1].value + mwd_x_ii[1] - mwd_x_i[1], 
                                       z=xs_orbit[i+1][2].value + mwd_x_ii[2] - mwd_x_i[2],
                                      mwhharmonicflag=mwhharmonicflag, 
                                      mwdharmonicflag=mwdharmonicflag,
                                      lmcharmonicflag=lmcharmonicflag) * (u.km**2/(u.s**2*u.kpc))

        vs_orbit[i+1] = vs_orbit[i] + (0.5 * (accs + new_accs) * (dt*u.Gyr).to(u.s)).to(u.km/u.s)

        i = i+1
        
    mask = np.all(xs_orbit==0, axis=(1)) 

    return xs_orbit[~mask], vs_orbit[~mask], ts

def lagrange_cloud_strip_adT(params, overwrite):  
    
    inpath, snapname, outpath, filename, \
    fc, Mprog, a_s, pericenter, apocenter, Tbegin, Tfinal, dtmin, \
    mwhflag, mwdflag, lmcflag, strip_rate, \
    discframe, static_mwh, mwd_switch, lmc_switch = params
    
    fullfile_path = pathlib.Path(outpath) / filename

    if fullfile_path.exists() and not overwrite:
        return 
    
    new_G = G.to(u.kpc*(u.km/u.s)**2/u.Msun)
    Lunits = (u.kpc*u.km)/u.s
    lambda_source = 1. # the multiplier of how far away from the tidal radius to strip from.
    max_steps = int((Tfinal - Tbegin) / dtmin) + 2 
    max_particles = int( ((max_steps)*strip_rate) + 1)
    
    if static_mwh==True:
        _, MWHcoeffs = Model.return_mw_coefficients()
        MWHcoeffs = np.array(MWHcoeffs)
        MWHcoeffs[:,0] = MWHcoeffs[:,0][0] 
        MWHcoeffs[:,1:] = MWHcoeffs[:,1:]*0
        Model.install_mw_coefficients(MWHcoeffs)
        #Some line of code here to check they have been set to zero and reinstalled
        _, MWHcoeffs = Model.return_mw_coefficients()
        assert np.allclose(np.array(MWHcoeffs)[:,1:],0)==True, "MW halo coefficients need to be set to zero"
        
        
    if mwd_switch==True:
        MWDfloats, MWDctmp,MWDstmp = Model.return_disc_coefficients()
        ntimesteps = len(MWDctmp)
        MWDc, MWDs = np.zeros([ntimesteps, MWDctmp[0].shape[0],MWDctmp[0].shape[1]]), np.zeros([ntimesteps,MWDstmp[0].shape[0],MWDstmp[0].shape[1]])
        MWDc *= 0
        MWDs *= 0
        Model.install_disc_coefficients(MWDc,MWDs)
        #Some line of code here to check they have been set to zero and reinstalled.
        MWDfloats, MWDctmp,MWDstmp = Model.return_disc_coefficients()
        assert np.allclose(np.array(MWDctmp),0)==True, "MW disc coefficients (c) need to be set to zero"
        assert np.allclose(np.array(MWDstmp),0)==True, "MW disc coefficients (s) need to be set to zero"
        
    if lmc_switch==True:
        _, LMCcoeffs = Model.return_lmc_coefficients()
        LMCcoeffs = np.array(LMCcoeffs)
        LMCcoeffs *= 0 
        Model.install_lmc_coefficients(LMCcoeffs)
        #Some line of code here to check they have been set to zero and reinstalled.
        _, LMCcoeffs = Model.return_lmc_coefficients()
        assert np.allclose(np.array(LMCcoeffs),0)==True, "LMC coefficients need to be set to zero"
    
    prog_orbit = rewind_leap(fc[:3]*u.kpc, fc[3:6]*(u.km/u.s), 
                        mwhflag, 
                        mwdflag,
                        lmcflag, 
                        Tbegin,
                        dtmin)
    rewind_xs = prog_orbit[0]
    rewind_vs = prog_orbit[1]
    rewind_ts = prog_orbit[2]
    prog_fc = np.concatenate([rewind_xs[-1].value, rewind_vs[-1].value]).reshape(1,6)
    
    xs_data = np.full(shape=(max_steps, max_particles, 3), fill_value=np.nan)
    vs_data = np.zeros(shape=(max_steps, max_particles, 3))
    
    ts = np.array([rewind_ts[-1]])
    t = ts[0] 

    xs = np.array(prog_fc[:,0:3])*u.kpc
    vs = -np.array(prog_fc[:,3:6])*(u.km/u.s)
    
    sigma_vs = np.array([])
    tidal_radii = np.array([])
    
    i = 0
    while t < Tfinal:
        
        mwd_x_i = 0*np.array(Model.expansion_centres(t)[0:3]) 
        
        prog_xs = (xs[:, 0:3][0]).reshape(1,3).value
        prog_vs = (vs[:, 0:3][0]).reshape(1,3).value
     
        # Calculate inital angular speed of progenitor (in units of km/s/kpc)
        r_prog = np.linalg.norm(prog_xs[0:3])
        L_prog = np.linalg.norm(np.cross(prog_xs[0:3], prog_vs[0:3]))
        Omega_prog = L_prog*Lunits / (r_prog*u.kpc)**2
        r_hat = prog_xs[0:3]/r_prog
        num_d2Phi_dr2 = numerical_forceDerivs(prog_xs + mwd_x_i, np.array([t]), mwhflag, mwdflag, lmcflag)*(u.km/u.s/u.kpc)**2

        # Calculate tidal radius
        mass_frac = 1 - (np.abs(Tbegin) - np.abs(t))/np.abs(Tbegin)
        rt = ((new_G * (Mprog*mass_frac)*u.Msun) / (Omega_prog**2 - num_d2Phi_dr2) )**(1/3) 
        r_t = np.nan_to_num(rt.value, nan=a_s)
        tidal_radii = np.append(tidal_radii, r_t)
    
        sigma_s = ((new_G.value * Mprog * mass_frac) / (r_t**2 + a_s**2)**0.5)**0.5 
        sigma_vs = np.append(sigma_vs, sigma_s)

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
        xs = np.append(xs, ic_source_coords*u.kpc, axis=0)
        vs = np.append(vs, (ic_source_velocities + np.random.randn(len(ic_source_velocities), 3)*sigma_s)*(u.km/u.s), axis=0)
        
        #-----------------------------------------------------------------------------------------------------
        # LEAPFROG INTEGRATION - Drift-kick-drift to copy mwlmc rewind 
        #-----------------------------------------------------------------------------------------------------

        accs_mwhalo = Model.all_forces(t, x=xs[:,0].value + mwd_x_i[0], 
                                       y=xs[:,1].value + mwd_x_i[1], 
                                       z=xs[:,2].value + mwd_x_i[2],
                                      mwhharmonicflag=mwhflag, 
                                      mwdharmonicflag=mwdflag,
                                      lmcharmonicflag=lmcflag)*(u.km**2/(u.s**2*u.kpc))
        accs_prog = plummer_force((xs - xs[0]).value, Mprog * mass_frac, a_s)*(u.km**2/(u.s**2*u.kpc)) #sort this out
        accs = accs_mwhalo  #+ accs_prog
        
        
        xs = xs + (vs * (dtmin*u.Gyr).to(u.s)).to(u.kpc) + (0.5 * accs * ((dtmin*u.Gyr).to(u.s))**2).to(u.kpc)
        
        ts = np.append(ts, t+dtmin)
        t = t + dtmin
        
        mwd_x_ii = 0*np.array(Model.expansion_centres(t)[0:3])
        # Correct for already accounting for disc motion at the last timestep, adjust to current timestep
        new_accs_mwhalo = Model.all_forces(t, x=xs[:,0].value + mwd_x_ii[0] - mwd_x_i[0],
                                               y=xs[:,1].value + mwd_x_ii[1] - mwd_x_i[1], 
                                               z=xs[:,2].value + mwd_x_ii[2] - mwd_x_i[2],
                                              mwhharmonicflag=mwhflag, 
                                              mwdharmonicflag=mwdflag,
                                              lmcharmonicflag=lmcflag)*(u.km**2/(u.s**2*u.kpc))
        new_accs_prog = plummer_force((xs - xs[0]).value, Mprog * mass_frac, a_s)*(u.km**2/(u.s**2*u.kpc))
        new_accs = new_accs_mwhalo  #+ new_accs_prog
        
        vs = vs + (0.5 * (accs + new_accs) * (dtmin*u.Gyr).to(u.s)).to(u.km/u.s)

   
        xs_data[i], vs_data[i] = fill_with_zeros(xs, max_particles), fill_with_zeros(vs, max_particles)
        i = i + 1
    
    mask = np.all(np.isnan(xs_data), axis=(1,2)) 
    # Save only every 100th time snapshot - flipping to slice properly, flip back after
    xs_snaps_flip = np.flip(xs_data[~mask], axis=0)[::50]
    vs_snaps_flip = np.flip(vs_data[~mask], axis=0)[::50]
    
    xs_snaps = np.flip(xs_snaps_flip, axis=0)
    vs_snaps = np.flip(vs_snaps_flip, axis=0)
    
    pot_label = harmonicflags_to_potlabel(mwhflag, mwdflag, lmcflag, static_mwh)    
    write_stream_hdf5(outpath, filename, 
                      xs_snaps, 
                      vs_snaps, 
                      ts,
                     pot_label, fc, Mprog, a_s, pericenter, apocenter, discframe)
    
    # return xs_data[~mask], vs_data[~mask], ts, tidal_radii, sigma_vs, prog_orbit

# def lagrange_cloud_strip_adT(params, overwrite):  
    
#     inpath, snapname, outpath, filename, \
#     fc, Mprog, a_s, pericenter, apocenter, Tbegin, Tfinal, dtmin, \
#     mwhflag, mwdflag, lmcflag, strip_rate, \
#     discframe, static_mwh, mwd_switch, lmc_switch = params
  
#     fullfile_path = pathlib.Path(outpath) / filename

#     if fullfile_path.exists() and not overwrite:
#         return 
 
#     Gyr_to_s = gyr.to(u.s)
#     km_to_kpc = km.to(u.kpc)
#     new_G = G.to(u.kpc*(u.km/u.s)**2/u.Msun)
#     lambda_source = 1.2 # the multiplier of how far away from the tidal radius to strip from.
#     # nu=0.01 # For adaptive timestepping
#     max_steps = int((Tfinal - Tbegin) / dtmin) + 1 
#     max_particles = int( ((max_steps)*strip_rate) + 1)
    
#     if static_mwh==True:
#         _, MWHcoeffs = Model.return_mw_coefficients()
#         MWHcoeffs = np.array(MWHcoeffs)
#         MWHcoeffs[:,0] = MWHcoeffs[:,0][0] 
#         Model.install_mw_coefficients(MWHcoeffs)
#         #Some line of code here to check they have been set to zero and reinstalled.
        
#     if mwd_switch==True:
#         MWDfloats, MWDctmp,MWDstmp = Model.return_disc_coefficients()
#         ntimesteps = len(MWDctmp)
#         MWDc, MWDs = np.zeros([ntimesteps, MWDctmp[0].shape[0],MWDctmp[0].shape[1]]), np.zeros([ntimesteps,MWDstmp[0].shape[0],MWDstmp[0].shape[1]])
#         MWDc *= 0
#         MWDs *= 0
#         Model.install_disc_coefficients(MWDc,MWDs)
#         #Some line of code here to check they have been set to zero and reinstalled.
        
#     if lmc_switch==True:
#         _, LMCcoeffs = Model.return_lmc_coefficients()
#         LMCcoeffs = np.array(LMCcoeffs)
#         LMCcoeffs *= 0 
#         Model.install_lmc_coefficients(LMCcoeffs)
#         #Some line of code here to check they have been set to zero and reinstalled.
        
#     orbit  = Model.rewind(fc[:3], fc[3:6],
#                      mwhharmonicflag=mwhflag, 
#                      mwdharmonicflag=mwdflag,
#                      lmcharmonicflag=lmcflag,
#                      rewindtime=np.abs(Tbegin),
#                      dt=dtmin, 
#                      discframe=discframe)
#     prog_orbit = np.array(orbit)
#     rewind_xs = prog_orbit[0:3]
#     rewind_vs = prog_orbit[3:6]
#     prog_fc = np.concatenate([rewind_xs.T[-1], rewind_vs.T[-1]]).reshape(1,6)
    
#     xs_data = np.full(shape=(max_steps, max_particles, 3), fill_value=np.nan)
#     vs_data = np.zeros(shape=(max_steps, max_particles, 3))

#     xs = np.array(prog_fc[:,0:3])
#     vs = -np.array(prog_fc[:,3:6])
    
#     ts = np.array([Tbegin])
#     t = ts[0]
    
#     i = 0
#     while t < Tfinal:
        
#         mwd_x = np.array(Model.expansion_centres(t)[9:12])
      
#         prog_xs = (xs[:, 0:3][0]).reshape(1,3)
#         prog_vs = (vs[:, 0:3][0]).reshape(1,3)
     
#         # Calculate inital angular speed of progenitor (in units of km/s/kpc)
#         r_prog = np.linalg.norm(prog_xs[0:3])
#         L_prog = np.linalg.norm(np.cross(prog_xs[0:3], prog_vs[0:3]))
#         Omega_prog = L_prog / r_prog**2
#         r_hat = prog_xs[0:3]/r_prog
#         num_hessian, num_d2Phi_dr2 = numerical_forceDerivs(prog_xs + mwd_x, np.array([t]), mwhflag, mwdflag, lmcflag)

#         # Calculate tidal radius
#         mass_frac = 1 - (np.abs(Tbegin) - np.abs(t))/np.abs(Tbegin)
#         r_t = np.nan_to_num( (((new_G * Mprog / (Omega_prog**2 - num_d2Phi_dr2))*u.Msun/(u.km/u.s/u.kpc)**2)**(1/3)).to(u.kpc).value, nan=a_s)
#         r_t =  r_t * mass_frac**(1/3) 
    
#         sigma_s = ((new_G.value * Mprog * mass_frac) / (r_t**2 + a_s**2)**0.5)**0.5 

#         # Calculate positions and velocities of points of particle release
#         source_coords_in = prog_xs[0:3] - lambda_source * r_t[:, None] * r_hat
#         source_coords_out = prog_xs[0:3] + lambda_source * r_t[:, None] * r_hat

#         prog_velocity_r = np.sum(prog_vs[0:3]*r_hat)
#         prog_velocity_tan = prog_vs[0:3] - prog_velocity_r * r_hat

#         source_velocity_tan_in = prog_velocity_tan * (1 - 0.5 * r_t / r_prog)[:, None]
#         source_velocity_tan_out = prog_velocity_tan * (1 + 0.5 * r_t / r_prog)[:, None]

#         source_velocity_in = prog_velocity_r * r_hat + source_velocity_tan_in
#         source_velocity_out = prog_velocity_r * r_hat + source_velocity_tan_out

#         source_coords = np.zeros((len(source_coords_in)*2, 3))
#         source_coords[::2] = source_coords_in
#         source_coords[1::2] = source_coords_out

#         source_velocity = np.zeros((len(source_velocity_in)*2, 3))
#         source_velocity[::2] = source_velocity_in
#         source_velocity[1::2] = source_velocity_out

#         ic_source_coords = np.repeat(source_coords, strip_rate/2, axis=0)
#         ic_source_velocities = np.repeat(source_velocity, strip_rate/2, axis=0)
 
#         np.random.seed(0)
#         xs = np.append(xs, ic_source_coords, axis=0)
#         vs = np.append(vs, (ic_source_velocities + np.random.randn(len(ic_source_velocities), 3)*sigma_s), axis=0)
        
#         #-----------------------------------------------------------------------------------------------------
#         # LEAPFROG INTEGRATION
#         #-----------------------------------------------------------------------------------------------------

#         accs_mwhalo = Model.all_forces(t, x=xs[:,0] + mwd_x[0], 
#                                        y=xs[:,1] + mwd_x[1], 
#                                        z=xs[:,2] + mwd_x[2],
#                                       mwhharmonicflag=mwhflag, 
#                                       mwdharmonicflag=mwdflag,
#                                       lmcharmonicflag=lmcflag)
#         accs_prog = plummer_force(xs - xs[0], Mprog * mass_frac, a_s)
#         accs = accs_mwhalo + accs_prog
           
#         ## Adapative timestep
#         # dt_adapt = np.nanmin([np.nanmin(nu*np.sqrt(np.sum(xs**2, axis=1)**0.5 / \
#         #                                                np.sum(accs_mwhalo**2, axis=1)**0.5)), \
#         #                       np.nanmin(nu*np.sqrt(np.sum((xs - prog_xs)[1:]**2, axis=1)**0.5 / \
#         #                                               np.sum(accs_prog[1:]**2, axis=1)**0.5)) ])
#         # if dt_adapt < dtmin:
#         #     dt_adapt = dtmin
#         dt_adapt = dtmin
        
     
#         ts = np.append(ts, t+dt_adapt)
#         t = t + dt_adapt
        
#         mwd_x = np.array(Model.expansion_centres(t)[9:12])
       
#         v_halfs = vs + (accs * dt_adapt/2)
#         xs = xs + v_halfs * (dt_adapt * Gyr_to_s.value * km_to_kpc.value)
        
#         new_accs_mwhalo = Model.all_forces(t, x=xs[:,0] + mwd_x[0],
#                                            y=xs[:,1] + mwd_x[1], 
#                                            z=xs[:,2] + mwd_x[2],
#                                           mwhharmonicflag=mwhflag, 
#                                           mwdharmonicflag=mwdflag,
#                                           lmcharmonicflag=lmcflag)
#         new_accs_prog = plummer_force(xs - xs[0], Mprog * mass_frac, a_s)
#         new_accs = new_accs_mwhalo + new_accs_prog
        
#         vs =  v_halfs + (new_accs*dt_adapt/2)
   
#         xs_data[i], vs_data[i] = fill_with_zeros(xs, max_particles), fill_with_zeros(vs, max_particles)
#         i = i + 1

#     ts = np.repeat(ts, strip_rate)
#     ts = fill_with_nans_1d(ts, max_particles)
    
#     mask = np.all(np.isnan(xs_data), axis=(1,2)) 
#     # Save only every 100th time snapshot - flipping to slice properly, flip back after
#     xs_snaps_flip = np.flip(xs_data[~mask], axis=0)[::100]
#     vs_snaps_flip = np.flip(vs_data[~mask], axis=0)[::100]
    
#     xs_snaps = np.flip(xs_snaps_flip, axis=0)
#     vs_snaps = np.flip(vs_snaps_flip, axis=0)
    
#     pot_label = harmonicflags_to_potlabel(mwhflag, mwdflag, lmcflag, static_mwh)    
#     write_stream_hdf5(outpath, filename, 
#                       xs_snaps, 
#                       vs_snaps, 
#                       ts,
#                      pot_label, fc, Mprog, a_s, pericenter, apocenter, discframe)
    
#     # return xs_snaps, vs_snaps, ts

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
    prog_scale = d["prog_scale"] # kpc
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
    mwd_switch = d["mwd_switch"]
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
            haloflag, discflag, lmcflag, strip_rate, discframe, static_mwh, mwd_switch, lmc_switch]

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
        
    elif mwhflag==0 and mwdflag==0 and lmcflag==0:
        label = 'static-mwh-only'
      
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
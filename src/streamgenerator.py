import numpy as np
import astropy.units as u
import yaml
import h5py
from astropy.constants import G
import gala.integrate as gi
import gala.dynamics as gd
import gala.coordinates as gc
import gala.potential as gp
from gala.units import galactic
import os.path
import sys
from argparse import ArgumentParser
import pathlib

from scipy.spatial.transform import Rotation

from mwlmc import model as mwlmc_model
Model = mwlmc_model.MWLMC()

gyr = 1*u.Gyr
km = 1*u.km

def plummer_force(r, m, b):
    """
    plummer_force -  Returns the force of a Plummer sphere
    
    Inputs:
    - r: distance to compute force at from centre of potenial [kpc]
    - m: mass [Msolar]
    - b: scale radius [kpc]
    
    Outputs:
    - force in km^2 / (s^2 * kpc) - same as Lilleengen+23 simulations 
    """
    
    newG = G.to(u.kpc / u.Msun * (u.km/u.s)**2)
    dpot_dr = (newG * m*u.Msun * r*u.kpc) / ((r*u.kpc)**2 + (b*u.kpc)**2)**1.5
    force = -dpot_dr
    return force 

def num_summedforces(t, progxs, mwdharmonicflag, mwhharmonicflag, lmcharmonicflag):
    
    x0 = np.array(Model.expansion_centres(t))
    # disk 
    acc_disk = Model.mwd_fields(t, 
                                progxs[0, None].value - x0[:3][0, None],
                                progxs[1, None].value - x0[:3][1, None],
                                progxs[2, None].value - x0[:3][2, None],
                                mwdharmonicflag)[ :3]
    # halo
    acc_halo = Model.mwhalo_fields(t, 
                                   progxs[0, None].value - x0[3:6][0, None],
                                   progxs[1, None].value - x0[3:6][1, None],
                                   progxs[2, None].value - x0[3:6][2, None],
                                   mwhharmonicflag)[ :3]
    # lmc
    acc_lmc = Model.lmc_fields(t, 
                               progxs[0, None].value - x0[6:9][0, None],
                               progxs[1, None].value - x0[6:9][1, None],
                               progxs[2, None].value - x0[6:9][2, None],
                               lmcharmonicflag)[ :3]

    accs = (acc_disk + acc_halo + acc_lmc) * (u.km**2/(u.s**2*u.kpc))
    
    return accs

def numerical_forceDerivs(positions, t, mwhflag, mwdflag, lmcflag, epsilon=1e-4):
    
    """
    numerical_forceDerivs - takes the positions for the mock stream progenitor and returns the derivatives of the forces for each position 
                    and second derivative of the potential w.r.t the positons.
    
    Inputs
    - positions: The positions of the progenitor in kpc, Shape N x 3.
    - ts: Time steps corresponding to each position in Gyr.
    - mwhflag: flag to set which mw halo expansion orders to be non-zero.
    - mwdflag: flag to set which mw disc expansion orders to be non-zero.
    - lmcflag: flag to set which lmc halo expansion orders to be non-zero.
    - epsilon: (optional) The small value away from each position used find the derivative in kpc. Default is 0.01pc
    
    Retuns
    - Hess: The Hessian matrix of all force derivatives. Shape (len(times), 3, 3). ((xx, xy, xz), 
                                                                                    (yx, yy, yz), 
                                                                                    (zx, zy, zz))
    - d2Phi_d2r: The second derivative of the potential with repect to the position.
    """
    
    r_prog = np.linalg.norm(positions)
    
    fxx_yx_zx = np.zeros(shape=(1,3))
    fxy_yy_zy = np.zeros(shape=(1,3))
    fxz_yz_zz = np.zeros(shape=(1,3))
    
    positions *= u.kpc
    positions_dx = (positions + np.array([epsilon,0.,0.])*u.kpc )
    positions_dy = (positions + np.array([0.,epsilon,0.])*u.kpc )
    positions_dz = (positions + np.array([0.,0.,epsilon])*u.kpc )
    
    fxx_yx_zx = (num_summedforces(t, positions_dx, mwdflag, mwhflag, lmcflag) - num_summedforces(t, positions, mwdflag, mwhflag, lmcflag) ) \
                    / np.linalg.norm(positions_dx - positions)
        
    fxy_yy_zy = (num_summedforces(t, positions_dy, mwdflag, mwhflag, lmcflag) - num_summedforces(t, positions, mwdflag, mwhflag, lmcflag) ) \
                    / np.linalg.norm(positions_dy - positions)
        
    fxz_yz_zz = (num_summedforces(t, positions_dz, mwdflag, mwhflag, lmcflag) - num_summedforces(t, positions, mwdflag, mwhflag, lmcflag) ) \
                    / np.linalg.norm(positions_dz - positions)
        
    Hess = np.zeros((1, 3, 3))
    Hess[:, 0, :] = -np.array([fxx_yx_zx[:, 0], fxy_yy_zy[:, 0], fxz_yz_zz[:, 0]]).T
    Hess[:, 1, :] = -np.array([fxx_yx_zx[:, 1], fxy_yy_zy[:, 1], fxz_yz_zz[:, 1]]).T
    Hess[:, 2, :] = -np.array([fxx_yx_zx[:, 2], fxy_yy_zy[:, 2], fxz_yz_zz[:, 2]]).T
    
    r_hat = positions.reshape(1,3) / r_prog.reshape(1,)
    d2Phi_d2r = np.einsum('ki,kij,kj->k', r_hat, Hess, r_hat)
    
    return d2Phi_d2r.value*(u.km**2/(u.s**2*u.kpc**2))

def gala_F(t, w, mwdflag, mwhflag, lmcflag):
    t = t / 1e3  # Myr -> Gyr
    x0 = np.array(Model.expansion_centres(t))
    # disk
    # acc_disk = Model.mwd_fields(t, *(w[:3, :] - x0[:3, None]), mwdharmonicflag=mwdflag)[:, :3]
    MiyamotoNagai = gp.MiyamotoNagaiPotential(6.8e10*u.Msun, 3*u.kpc, 0.28*u.kpc, units=galactic, origin=x0[:3])
    acc_disk = MiyamotoNagai.acceleration(w[:3, :] - x0[:3, None]).to((u.km**2/(u.s**2*u.kpc))).value.T
    # halo
    acc_halo = Model.mwhalo_fields(t, *(w[:3] - x0[3:6, None]), mwhharmonicflag=mwhflag)[:, :3]
    # lmc
    acc_lmc = Model.lmc_fields(t, *(w[:3] - x0[6:9, None]), lmcharmonicflag=lmcflag)[:, :3]
    
    accs = (acc_disk + acc_halo + acc_lmc) * (u.km**2/(u.s**2*u.kpc))
    accs = accs.decompose(galactic).value
    return np.vstack((w[3:], accs.T))


def gala_rewind(Tbegin, Tend, dt, w, mwdflag, mwhflag, lmcflag):
    
    integrator = gi.LeapfrogIntegrator(gala_F, func_units=galactic, func_args=(mwdflag, mwhflag, lmcflag,))
    # integrator = gi.Ruth4Integrator(gala_F, func_units=galactic, func_args=(mwdflag, mwhflag, lmcflag,))
    
    mwd_x0 = np.array(Model.expansion_centres(0.)[:3])*u.kpc 
    mwd_v0 = np.array(Model.expansion_centre_velocities(0.)[:3])*(u.km/u.s) 
    w0 = gd.PhaseSpacePosition(pos=w.xyz + mwd_x0,
                               vel=(w.v_xyz.to(u.km/u.s) + mwd_v0).to(u.kpc/u.Myr) )
    orbit = integrator.run(w0, dt=-dt*u.Myr, t1=Tend*u.Gyr, t2=Tbegin*u.Gyr)
    # subtract these off
    disk_x0 = np.array([Model.expansion_centres(t)[:3] for t in orbit.t.to_value(u.Gyr)]) 
    disk_v0 = np.array([Model.expansion_centre_velocities(t)[:3] for t in orbit.t.to_value(u.Gyr)]) 
    pos=orbit.xyz - disk_x0.T*u.kpc,
    vel=orbit.v_xyz.to(u.km/u.s) - disk_v0.T*u.km/u.s,
    t=orbit.t
    return pos[0].to(u.kpc), vel[0].to(u.km/u.s), t.to(u.Gyr)

def energies_angmom(t, xs, vs, mwdflag, mwhflag, lmcflag):
    
    """
    calculate the energies and angular momenta of particles for a given time snapshot.
    """
    # Kinetic energy
    Ek = (.5 * np.linalg.norm(vs, axis=1)**2) * (u.km/u.s)**2
    # Potential energy
    x0 = np.array(Model.expansion_centres(t))
    pot_disk = Model.mwd_fields(t, 
                                xs[:,0] - x0[:3][0],
                                xs[:,1] - x0[:3][1],
                                xs[:,2] - x0[:3][2],
                                mwdflag)[:,4]
    pot_halo = Model.mwhalo_fields(t, 
                                   xs[:,0] - x0[3:6][0],
                                   xs[:,1] - x0[3:6][1],
                                   xs[:,2] - x0[3:6][2],
                                   mwhflag)[:,4]
    pot_lmc = Model.lmc_fields(t, 
                               xs[:,0] - x0[6:9][0],
                               xs[:,1] - x0[6:9][1],
                               xs[:,2] - x0[6:9][2],
                               lmcflag)[:,4]
    Ep = (pot_disk + pot_halo + pot_lmc) * (u.km/u.s)**2
    E = Ek + Ep
    # Angular momentum
    L = np.linalg.norm(np.cross(xs, vs), axis=1) * (u.kpc*u.km/u.s)
    Lz = np.cross(xs[:,0:2], vs[:,0:2]) * (u.kpc*u.km/u.s)
    
    return E, L, Lz

def orbpole(xs,vs):
    uu = np.cross(xs, vs, axis=1)
    uumag = np.linalg.norm(uu, axis=1)
    u = uu.T/uumag
    b = np.arcsin(u[2])
    sinl = u[1]/np.cos(b)
    cosl = u[0]/np.cos(b)
    ll = np.arctan2(sinl, cosl)
    gl = np.degrees(ll)
    gb = np.degrees(b)
    return gl, gb    

def lons_lats(pos, vel):
    prog = gd.PhaseSpacePosition(pos[0] * u.kpc, vel[0] * u.km / u.s)
    stream = gd.PhaseSpacePosition(pos[1:].T * u.kpc, vel[1:].T * u.km / u.s)
    R1 = Rotation.from_euler("z", -prog.spherical.lon.degree, degrees=True)
    R2 = Rotation.from_euler("y", prog.spherical.lat.degree, degrees=True)
    R_prog0 = R2.as_matrix() @ R1.as_matrix()  

    new_vxyz = R_prog0 @ prog.v_xyz
    v_angle = np.arctan2(new_vxyz[2], new_vxyz[1])
    R3 = Rotation.from_euler("x", -v_angle.to_value(u.degree), degrees=True)
    R = (R3 * R2 * R1).as_matrix()

    stream_rot = gd.PhaseSpacePosition(stream.data.transform(R))
    stream_sph = stream_rot.spherical
    lon = stream_sph.lon.wrap_at(180*u.deg).degree
    lat = stream_sph.lat.degree
    return lon, lat

def local_veldis(lons, vfs):
    # Compute percentiles
    lower_value = np.nanpercentile(lons, 0.1)
    upper_value = np.nanpercentile(lons, 99.9)
    # Filter lons_mainbody
    lons_mainbody = lons[(lons >= lower_value) & (lons <= upper_value)]
    vfs_mainbody = vfs[1:][(lons >= lower_value) & (lons <= upper_value)] #excludes progenitor [1:]
    # Create bins
    lon_bins = np.linspace(np.nanmin(lons_mainbody), np.nanmax(lons_mainbody), 50)
    # Compute absolute velocity norms
    vfs_absol = np.linalg.norm(vfs_mainbody, axis=1)
    # Slice lons_mainbody into bins
    bin_indices = np.digitize(lons_mainbody, lon_bins)
    # Create a mask array
    mask = np.zeros((len(lons_mainbody), len(lon_bins) - 1), dtype=bool)
    for i in range(1, len(lon_bins)):
        mask[:, i - 1] = (bin_indices == i)

    # Calculate standard deviation for each bin
    local_veldis = np.array([np.std(vfs_absol[m]) for m in mask.T])
    return np.nanmedian(local_veldis)

def widths_deforms(lons, lats):
    # Compute percentiles
    lower_value = np.nanpercentile(lons, 0.1)
    upper_value = np.nanpercentile(lons, 99.9)
    # Filter lons_mainbody
    lons_mainbody = lons[(lons >= lower_value) & (lons <= upper_value)]
    lats_mainbody = lats[(lons >= lower_value) & (lons <= upper_value)] 
    # Create bins
    lon_bins = np.linspace(np.nanmin(lons_mainbody), np.nanmax(lons_mainbody), 50)
    # Slice lons_mainbody into bins
    bin_indices = np.digitize(lons_mainbody, lon_bins)
    # Create a mask array
    mask = np.zeros((len(lons_mainbody), len(lon_bins) - 1), dtype=bool)
    for i in range(1, len(lon_bins)):
        mask[:, i - 1] = (bin_indices == i)

    # Calculate width for each bin
    local_width = np.array([np.nanstd(lats_mainbody[m]) for m in mask.T])
    track_deforms = np.array([np.abs(np.nanmedian(lats_mainbody[m])) for m in mask.T])
    return np.nanmedian(local_width), np.nanmedian(track_deforms)

def lagrange_cloud_strip_adT(params, overwrite):  
    
    inpath, snapname, outpath, filename, \
    fc, Mprog, a_s, pericenter, apocenter, Tbegin, Tfinal, dtmin, \
    mwhflag, mwdflag, lmcflag, strip_rate, \
    static_mwh, static_mwd, lmc_switch = params
    
    # fullfile_path = pathlib.Path(outpath) / filename + '.hdf5'
    fullfile_path = pathlib.Path(outpath) / pathlib.Path(filename + '.hdf5')

    if fullfile_path.exists() and not overwrite:
        print("Skipping as already exists and do not want to overwrite...")
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
        
    if static_mwd==True:
        MWDfloats, MWDctmp, MWDstmp = Model.return_disc_coefficients()
        MWDctmp, MWDstmp = np.array(MWDctmp), np.array(MWDstmp)
        MWDctmp[:,0], MWDstmp[:,0] = MWDctmp[:,0][0], MWDstmp[:,0][0] # Turn this on to get rigid monopole disc 
        # MWDctmp[:,0], MWDstmp[:,0] = MWDctmp[:,0][0]*0, MWDstmp[:,0][0]*0 # Turn this on to get rid of disc altogether
        MWDctmp[:,1:], MWDstmp[:,1:] = MWDctmp[:,1:]*0, MWDstmp[:,1:]*0
        Model.install_disc_coefficients(MWDctmp,MWDstmp)
        #Some line of code here to check they have been set to zero and reinstalled.
        MWDfloats, MWDctmp, MWDstmp = Model.return_disc_coefficients()
        assert np.allclose(np.array(MWDctmp)[:,1:],0)==True, "MW disc coefficients (c) need to be set to zero"
        assert np.allclose(np.array(MWDstmp)[:,1:],0)==True, "MW disc coefficients (s) need to be set to zero"
        
    if lmc_switch==True:
        _, LMCcoeffs = Model.return_lmc_coefficients()
        LMCcoeffs = np.array(LMCcoeffs)
        LMCcoeffs *= 0 
        Model.install_lmc_coefficients(LMCcoeffs)
        #Some line of code here to check they have been set to zero and reinstalled.
        _, LMCcoeffs = Model.return_lmc_coefficients()
        assert np.allclose(np.array(LMCcoeffs),0)==True, "LMC coefficients need to be set to zero"
    
    w0 = gd.PhaseSpacePosition.from_w(fc.T, units=galactic)
    print('rewinding progenitor...')
    prog_orbit = gala_rewind(Tbegin, Tfinal, dtmin*u.Gyr.to(u.Myr), w0, mwdflag, mwhflag, lmcflag)
    rewind_xs = prog_orbit[0].T #unpack tuple
    rewind_vs = prog_orbit[1].T #unpack tuple
    rewind_ts = prog_orbit[2]
    prog_ic = np.concatenate([rewind_xs[-1].value, rewind_vs[-1].value]).reshape(1,6)
    # w_ic = gd.PhaseSpacePosition(pos=rewind_xs[-1],vel=rewind_vs[-1]) 
    
    xs_data = np.full(shape=(max_steps, max_particles, 3), fill_value=np.nan)
    vs_data = np.zeros(shape=(max_steps, max_particles, 3))
    ts = np.array([rewind_ts[-1].value])
    t = ts[0] 
    
    print('forward integrating...')
    disk_xf = np.array(Model.expansion_centres(t)[:3]) 
    disk_vf = np.array(Model.expansion_centre_velocities(t)[:3]) 
    xs = np.array(prog_ic[:,0:3])*u.kpc + disk_xf*u.kpc
    vs = np.array(prog_ic[:,3:6])*(u.km/u.s) + disk_vf*(u.km/u.s)
    
    i = 0
    while t < Tfinal:
        
        prog_xs = (xs[:, 0:3][0]).reshape(1,3).value
        prog_vs = (vs[:, 0:3][0]).reshape(1,3).value
     
        # Calculate inital angular speed of progenitor (in units of km/s/kpc)
        r_prog = np.linalg.norm(prog_xs[0:3])
        L_prog = np.linalg.norm(np.cross(prog_xs[0:3], prog_vs[0:3]))
        Omega_prog = L_prog*Lunits / (r_prog*u.kpc)**2
        r_hat = prog_xs[0:3]/r_prog
        num_d2Phi_dr2 = numerical_forceDerivs(prog_xs.reshape(3,), t, mwhflag, mwdflag, lmcflag)

        # Calculate tidal radius
        mass_frac = 1 - (np.abs(Tbegin) - np.abs(t))/np.abs(Tbegin)
        rt = ((new_G * (Mprog*mass_frac)*u.Msun) / (Omega_prog**2 - num_d2Phi_dr2) )**(1/3) 
        r_t = np.nan_to_num(rt.value, nan=1) #kpc
        sigma_s = ((new_G.value * Mprog * mass_frac) / (r_t**2 + a_s**2)**0.5)**0.5

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
        # LEAPFROG INTEGRATION 
        #-----------------------------------------------------------------------------------------------------
        x0 = np.array(Model.expansion_centres(t))
        # disk 
        # acc_disk = Model.mwd_fields(t, 
        #                             xs[:,0].value - x0[:3][0, None],
        #                             xs[:,1].value - x0[:3][1, None],
        #                             xs[:,2].value - x0[:3][2, None],
        #                             mwdharmonicflag=mwdflag)[:, :3]
        MiyamotoNagai = gp.MiyamotoNagaiPotential(6.8e10*u.Msun, 3*u.kpc, 0.28*u.kpc, units=galactic, origin=x0[:3])
        acc_disk = MiyamotoNagai.acceleration((xs - x0[:3]*u.kpc).T).to((u.km**2/(u.s**2*u.kpc))).value.T   
        # halo
        acc_halo = Model.mwhalo_fields(t, 
                                       xs[:,0].value - x0[3:6][0, None],
                                       xs[:,1].value - x0[3:6][1, None],
                                       xs[:,2].value - x0[3:6][2, None],
                                       mwhharmonicflag=mwhflag)[:, :3]
        # lmc
        acc_lmc = Model.lmc_fields(t, 
                                   xs[:,0].value - x0[6:9][0, None],
                                   xs[:,1].value - x0[6:9][1, None],
                                   xs[:,2].value - x0[6:9][2, None],
                                   lmcharmonicflag=lmcflag)[:, :3]

        accs = (acc_disk + acc_halo + acc_lmc) * (u.km**2/(u.s**2*u.kpc))

        xs = xs + (vs * (dtmin*u.Gyr).to(u.s)).to(u.kpc) + (0.5 * accs * ((dtmin*u.Gyr).to(u.s))**2).to(u.kpc) 
        ts = np.append(ts, t+dtmin)
        t = t + dtmin

        x0 = np.array(Model.expansion_centres(t))
        # new_acc_disk = Model.mwd_fields(t, 
        #                             xs[:,0].value - x0[:3][0, None],
        #                             xs[:,1].value - x0[:3][1, None],
        #                             xs[:,2].value - x0[:3][2, None],
        #                             mwdharmonicflag=mwdflag)[:, :3]
        MiyamotoNagai = gp.MiyamotoNagaiPotential(6.8e10*u.Msun, 3*u.kpc, 0.28*u.kpc, units=galactic, origin=x0[:3])
        new_acc_disk = MiyamotoNagai.acceleration((xs - x0[:3]*u.kpc).T).to((u.km**2/(u.s**2*u.kpc))).value.T  
        # halo
        new_acc_halo = Model.mwhalo_fields(t, 
                                       xs[:,0].value - x0[3:6][0, None],
                                       xs[:,1].value - x0[3:6][1, None],
                                       xs[:,2].value - x0[3:6][2, None],
                                       mwhharmonicflag=mwhflag)[:, :3]
        # lmc
        new_acc_lmc = Model.lmc_fields(t, 
                                   xs[:,0].value - x0[6:9][0, None],
                                   xs[:,1].value - x0[6:9][1, None],
                                   xs[:,2].value - x0[6:9][2, None],
                                   lmcharmonicflag=lmcflag)[:, :3]
   
        new_accs = (new_acc_disk + new_acc_halo + new_acc_lmc) * (u.km**2/(u.s**2*u.kpc))
    
        vs = vs + (0.5 * (accs + new_accs) * (dtmin*u.Gyr).to(u.s)).to(u.km/u.s)

        xs_data[i], vs_data[i] = fill_with_zeros(xs, max_particles), fill_with_zeros(vs, max_particles)

        i += 1
         
    mask = np.all(np.isnan(xs_data), axis=(1,2)) 
    xs_data = xs_data[~mask]
    vs_data = vs_data[~mask]
    ts2 = np.repeat(ts, strip_rate)

    for i in range(len(xs_data)):
            
        disk_x0 = np.array(Model.expansion_centres(ts[i])[:3])
        disk_v0 = np.array(Model.expansion_centre_velocities(ts[i])[:3]) 
        xs_data[i] -= disk_x0
        vs_data[i] -= disk_v0
        
    # Save only every 500th/5000th time snapshot (10 total saved for dt=1/0.1Myr) - flipping to slice properly, flip back after
    xs_snaps = np.flip(np.flip(xs_data, axis=0)[::500], axis=0)
    vs_snaps = np.flip(np.flip(vs_data, axis=0)[::500], axis=0)
    ts_snaps = np.flip(np.flip(ts, axis=0)[::500])
    
    print("calculating energies, angular momenta, velocity dispersion, LMC separation...")
    Es = np.full(shape=(len(xs_snaps), max_particles), fill_value=np.nan)
    Ls = np.full(shape=(len(xs_snaps), max_particles), fill_value=np.nan)
    Lzs = np.full(shape=(len(xs_snaps), max_particles), fill_value=np.nan)
    
    lons, lats = lons_lats(xs_snaps[-1], vs_snaps[-1])
    sigma_v = local_veldis(lons, vs_snaps[-1])
    length = np.nanpercentile(lons, 95) - np.nanpercentile(lons, 5)
    width, track_deform = widths_deforms(lons, lats)
    med_lon, med_lat = np.nanmedian(lons), np.nanmedian(lats)
    
    lmc_sep = np.full(shape=(len(xs_snaps), max_particles), fill_value=np.nan)
    gls  =  np.full(shape=(len(xs_snaps), max_particles), fill_value=np.nan)
    gbs  =  np.full(shape=(len(xs_snaps), max_particles), fill_value=np.nan)
    
    for i in range(len(xs_snaps)):
        Es[i], Ls[i], Lzs[i] = energies_angmom(ts_snaps[i], xs_snaps[i], vs_snaps[i], mwdflag, mwhflag, lmcflag)
        lmc_sep[i] = np.linalg.norm(Model.expansion_centres(ts_snaps[i])[6:9]) - np.linalg.norm(xs_snaps[i], axis=1)
        gls[i], gbs[i] = orbpole(xs_snaps[i], vs_snaps[i])
    
    print("calculating LMC closest approach...")
    lmc_close_sep = np.nanmin(lmc_sep, axis=0)
        
    pot_label = harmonicflags_to_potlabel(mwhflag, mwdflag, lmcflag, static_mwh)    
    write_stream_hdf5(outpath, filename, xs_snaps, vs_snaps, ts2,
                      Es, Ls, Lzs, sigma_v, 
                      length, width, track_deform, med_lon, med_lat,
                      lmc_close_sep, gls, gbs,
                      pot_label, fc, Mprog, a_s, 
                      pericenter, apocenter)

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
    Tfinal =  d["Tfinal"]
    dtmin  = d["dtmin"]
    haloflag = d["haloflag"]
    lmcflag = d["lmcflag"]
    discflag = d["discflag"]
    strip_rate = d["strip_rate"]
    # discframe = d["discframe"]
    static_mwh = d["static_mwh"]
    static_mwd = d["mwd_switch"]
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
            haloflag, discflag, lmcflag, strip_rate, static_mwh, static_mwd, lmc_switch]

def write_stream_hdf5(outpath, filename, positions, velocities, times, 
                      energies, Ls, Lzs, sigma_v, 
                      length, width, track_deform, med_lon, med_lat,
                      lmc_sep, gls, gbs,
                      potential, progics, progmass, progscale, 
                      pericenter, apocenter):
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
    hf.create_dataset('energies', data=energies)
    hf.create_dataset('L', data=Ls)
    hf.create_dataset('Lz', data=Lzs)
    hf.create_dataset('loc_veldis', data=sigma_v)
    hf.create_dataset('lengths', data=length)
    hf.create_dataset('widths', data=width)
    hf.create_dataset('track_deform', data=track_deform)
    hf.create_dataset('av_lon', data=med_lon)
    hf.create_dataset('av_lat', data=med_lat)
    hf.create_dataset('lmc_sep', data=lmc_sep)
    hf.create_dataset('pole_l', data=gls)
    hf.create_dataset('pole_b', data=gbs)
    hf.create_dataset('potential', data=potential)
    hf.create_dataset('progenitor-ics', data=progics)
    hf.create_dataset('progenitor-mass', data=progmass)
    hf.create_dataset('progenitor-scale', data=progscale)
    hf.create_dataset('pericenter',data=pericenter)
    hf.create_dataset('apocenter', data=apocenter)
    hf.close()
    
def fill_with_zeros(arr, m):
    n = arr.shape[0]
    if m <= n:
        return arr
    else:
        filled_arr = np.full((m, 3), np.nan)
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
        
    elif mwhflag==63 and mwdflag==0 and lmcflag==63:
        label = 'full-mwh-no-mwd-full-lmc'  
         
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
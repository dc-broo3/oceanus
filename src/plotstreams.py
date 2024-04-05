from scipy.spatial.transform import Rotation
import numpy as np
import scipy
import pathlib
import h5py

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import median_absolute_deviation as mad_
from astropy.coordinates import CartesianRepresentation, SphericalRepresentation
import gala.dynamics as gd
import gala.coordinates as gc
from mwlmc import model as mwlmc_model
Model = mwlmc_model.MWLMC()
import yaml

import matplotlib
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
plt.style.use('/mnt/ceph/users/rbrooks/oceanus/analysis/my_standard.mplstyle')

galcen_v_sun = (11.1, 245, 7.3)*u.km/u.s
galcen_distance = 8.249*u.kpc

def plot_stream_frames(streams, path, plotname, savefig=False):
    
    potentials = list(['static-mwh-only.hdf5','rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5',\
                       'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5',\
                       'Full-MWhalo-MWdisc-LMC.hdf5', 'full-MWhalo-full-MWdisc-no-LMC.hdf5'])
    
    labels = list(['Static MW','Static Monopole', 'Evolving Monopole', 'Monopole + Dipole', 'Mono + Quadrupole', \
                   'Monopole + Dipole \n + Quadrupole', 'Full Expansion', 'Full Expansion \n (no LMC)'])
              
    t_idx = -1
    
    fig, ax = plt.subplots(len(streams), len(potentials), sharex='col', sharey='row', figsize=(21,6))
    plt.subplots_adjust(hspace=0, wspace=0.)
    
    for j in range(len(potentials)): 
        for i in range(len(streams)):   
            #-------------------------------------------------------------------------------------
            ### Read in the data
            #-------------------------------------------------------------------------------------
            data_path = pathlib.Path(path) / potentials[j]
            with h5py.File(data_path,'r') as file:
        
                prog = gd.PhaseSpacePosition(file[streams[i]]["positions"][t_idx, 0] * u.kpc, file[streams[i]]["velocities"][t_idx, 0] * u.km / u.s)
                stream = gd.PhaseSpacePosition(file[streams[i]]["positions"][t_idx, 1:].T * u.kpc, file[streams[i]]["velocities"][t_idx, 1:].T * u.km / u.s)
                start_times = np.array(file[streams[i]]['times'])
                prog_mass = np.array(file[streams[i]]['progenitor-mass']) * u.Msun
                rlmc = np.array(file[streams[i]]['lmc_sep'])
            #-------------------------------------------------------------------------------------
            ### Rotation matrix for progenitor to get it to near (X, 0, 0)
            #-------------------------------------------------------------------------------------
            R1 = Rotation.from_euler("z", -prog.spherical.lon.degree, degrees=True)
            R2 = Rotation.from_euler("y", prog.spherical.lat.degree, degrees=True)
            R_prog0 = R2.as_matrix() @ R1.as_matrix()  
            #-------------------------------------------------------------------------------------
            ### Rotate around new x axis so stream prog vel points along +y direction
            #-------------------------------------------------------------------------------------
            new_vxyz = R_prog0 @ prog.v_xyz
            v_angle = np.arctan2(new_vxyz[2], new_vxyz[1])
            R3 = Rotation.from_euler("x", -v_angle.to_value(u.degree), degrees=True)
            R = (R3 * R2 * R1).as_matrix()
            #-------------------------------------------------------------------------------------
            ### Rotate the whole stream by the final rotation matrix
            #-------------------------------------------------------------------------------------
            prog_rot = gd.PhaseSpacePosition(prog.data.transform(R))
            prog_sph = prog_rot.spherical
            stream_rot = gd.PhaseSpacePosition(stream.data.transform(R))
            stream_sph = stream_rot.spherical
            #-------------------------------------------------------------------------------------
            ### Plot the streams
            #-------------------------------------------------------------------------------------
            plt.sca(ax[i,j])
            print('* Plotting {} in potential {}'.format(streams[i], potentials[j]))
            plt.hlines(0, -200, 200, ls='dashed', color='lightgrey', lw=0.7, zorder=1)
            plt.vlines(0, -200, 200, ls='dashed', color='lightgrey', lw=0.7, zorder=1)
            plot=plt.scatter(stream_sph.lon.wrap_at(180*u.deg).degree[:-2], stream_sph.lat.degree[:-2], 
                             s=.5, c=start_times, cmap = 'viridis',rasterized=True, zorder=2)
            if j==0:
                name, ext = os.path.splitext(streams[i])
                plt.annotate(text='{}'.format(name), xy=(-180,65), fontsize=8 )
                plt.annotate(text=r'M = {} $\times \, 10^{{4}} \, \mathrm{{M}}_{{\odot}}$'.format(np.round(prog_mass.value/1e4, 1)),
                             xy=(-180, -80), fontsize=8)
            
    cb = fig.colorbar(plot,  ax=ax, location='right', aspect=30, pad=0.01)
    cb.set_label('Stripping time [Gyr]')
    cb.ax.tick_params(labelsize=12)
    
    #-------------------------------------------------------------------------------------
    ### Plot cosmetics
    #-------------------------------------------------------------------------------------
    for k in range(len(labels)):

        ax[0,k].set_title(labels[k])
        ax[len(streams)-1,k].set_xlabel(r'$\mathrm{lon}\,[^{\circ}]$')
        ax[len(streams)-1,k].set_xlim(-199,199)
        
    for l in range(len(streams)):
        ax[l, 0].set_ylabel(r'$\mathrm{lat}\,[^{\circ}]$')
        ax[l, 0].set_ylim(-99,99)

    if savefig==False:
        return
    elif savefig==True:
        savepath = '/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}'.format(plotname)
        print('* Saving figure at {}.pdf'.format(savepath))
        return plt.savefig(savepath)
    
    
def plot_stream_cartesian(streams, path, plotname, savefig=False):
    
    potentials = list(['static-mwh-only.hdf5','rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5', 'md-MWhalo-full-MWdisc-full-LMC.hdf5', \
                       'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5', 'Full-MWhalo-MWdisc-LMC.hdf5', \
                       'full-MWhalo-full-MWdisc-no-LMC.hdf5'])
    labels = list(['Static MW','Static Monopole', 'Evolving Monopole', 'Monopole + Dipole', 'Mono + Quadrupole', \
                      'Monopole + Dipole \n + Quadrupole', 'Full Expansion', 'Full Expansion \n (no LMC)'])
    
    fig_yz, ax = plt.subplots(len(streams), len(potentials), sharex='col', sharey='row', figsize=(19,5))
    plt.subplots_adjust(hspace=0, wspace=0.)
    
    for i in range(len(streams)): 
        for j in range(len(potentials)):   
    
            #-------------------------------------------------------------------------------------
            ### Read in the data
            #-------------------------------------------------------------------------------------
            data_path = pathlib.Path(path) / potentials[j]
            with h5py.File(data_path,'r') as file:
    
                pos = np.array(file[streams[i]]['positions'])
                vel = np.array(file[streams[i]]['velocities'])
                start_times = np.array(file[streams[i]]['times'])
                prog_mass = np.array(file[streams[i]]['progenitor-mass'])
                  
            #-------------------------------------------------------------------------------------
            ### Find the progentior Galactic coordiantes
            #-------------------------------------------------------------------------------------
            t0_pos, t0_vel = pos[-1], vel[-1]
            t0_prog_pos, t0_prog_vel = t0_pos[0], t0_vel[0]
            prog_xyz = CartesianRepresentation(t0_prog_pos[0]*u.kpc, y=t0_prog_pos[1]*u.kpc, z=t0_prog_pos[2]*u.kpc)
            
            #-------------------------------------------------------------------------------------
            ### Plot the streams
            #------------------------------------------------------------------------------------- 
            plt.sca(ax[i,j])
            plot=plt.scatter(t0_pos[:,1][1:-2],t0_pos[:,2][1:-2], s=1, c=start_times, cmap = 'viridis',rasterized=True)
            plt.scatter(prog_xyz.y, prog_xyz.z, s=50, edgecolors='k', facecolor='orange',marker='*', label='Prog.', rasterized=True)
            
    cb = fig_yz.colorbar(plot,  ax=ax, location='right', aspect=30, pad=0.01)
    cb.set_label('Stripping time [Gyr]')
    cb.ax.tick_params(labelsize=12)
    
    #-------------------------------------------------------------------------------------
    ### Plot cosmetics
    #-------------------------------------------------------------------------------------

    for k in range(len(labels)):

        ax[0,k].set_title(labels[k])
        ax[len(streams)-1,k].set_xlabel(r'$y\,[\mathrm{kpc}]$')
        
    for l in range(len(streams)):
        ax[l, 0].set_ylabel(r'$z\,[\mathrm{kpc}]$')

    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}'.format(plotname + '_yz'))
        
    fig_xy, ax = plt.subplots(len(streams), len(potentials), sharex='col', sharey='row', figsize=(17,5))
    plt.subplots_adjust(hspace=0, wspace=0.)
    
    for i in range(len(streams)): 
        for j in range(len(potentials)):   
    
            #-------------------------------------------------------------------------------------
            ### Read in the data
            #-------------------------------------------------------------------------------------
            data_path = pathlib.Path(path) / potentials[j]
            with h5py.File(data_path,'r') as file:
    
                pos = np.array(file[streams[i]]['positions'])
                vel = np.array(file[streams[i]]['velocities'])
                start_times = np.array(file[streams[i]]['times'])
                prog_mass = np.array(file[streams[i]]['progenitor-mass'])
                  
            #-------------------------------------------------------------------------------------
            ### Find the progentior Galactic coordiantes
            #-------------------------------------------------------------------------------------
            t0_pos, t0_vel = pos[-1], vel[-1]
            t0_prog_pos, t0_prog_vel = t0_pos[0], t0_vel[0]
            prog_xyz = CartesianRepresentation(t0_prog_pos[0]*u.kpc, y=t0_prog_pos[1]*u.kpc, z=t0_prog_pos[2]*u.kpc)
            
            #-------------------------------------------------------------------------------------
            ### Plot the streams
            #-------------------------------------------------------------------------------------
                
            plt.sca(ax[i,j])
            # print('* Plotting {} in potential {}'.format(streams[i], potentials[j]))
            plot=plt.scatter(t0_pos[:,0][1:-2],t0_pos[:,1][1:-2], s=1, c=start_times, cmap = 'viridis',rasterized=True)
            plt.scatter(prog_xyz.x, prog_xyz.y, s=50, edgecolors='k', facecolor='orange',marker='*', label='Prog.', rasterized=True)
            
    cb = fig_xy.colorbar(plot,  ax=ax, location='right', aspect=30, pad=0.01)
    cb.set_label('Stripping time [Gyr]')
    cb.ax.tick_params(labelsize=12)
    
    #-------------------------------------------------------------------------------------
    ### Plot cosmetics
    #-------------------------------------------------------------------------------------
            
    for k in range(len(labels)):

        ax[0,k].set_title(labels[k])
        ax[len(streams)-1,k].set_xlabel(r'$x\,[\mathrm{kpc}]$')
        
    for l in range(len(streams)):
        ax[l, 0].set_ylabel(r'$y\,[\mathrm{kpc}]$')
        
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}'.format(plotname + '_xy'))
    
    fig_xz, ax = plt.subplots(len(streams), len(potentials), sharex='col', sharey='row', figsize=(17,5))
    plt.subplots_adjust(hspace=0, wspace=0.)
    
    for i in range(len(streams)): 
        for j in range(len(potentials)):   
    
            #-------------------------------------------------------------------------------------
            ### Read in the data
            #-------------------------------------------------------------------------------------
            data_path = pathlib.Path(path) / potentials[j]
            with h5py.File(data_path,'r') as file:
    
                pos = np.array(file[streams[i]]['positions'])
                vel = np.array(file[streams[i]]['velocities'])
                start_times = np.array(file[streams[i]]['times'])
                prog_mass = np.array(file[streams[i]]['progenitor-mass'])
                  
            #-------------------------------------------------------------------------------------
            ### Find the progentior Galactic coordiantes
            #-------------------------------------------------------------------------------------
            t0_pos, t0_vel = pos[-1], vel[-1]
            t0_prog_pos, t0_prog_vel = t0_pos[0], t0_vel[0]
            prog_xyz = CartesianRepresentation(t0_prog_pos[0]*u.kpc, y=t0_prog_pos[1]*u.kpc, z=t0_prog_pos[2]*u.kpc)
            #-------------------------------------------------------------------------------------
            ### Plot the streams
            #------------------------------------------------------------------------------------- 
            plt.sca(ax[i,j])
            plot=plt.scatter(t0_pos[:,0][1:-2],t0_pos[:,2][1:-2], s=1, c=start_times, cmap = 'viridis',rasterized=True)
            plt.scatter(prog_xyz.x, prog_xyz.z, s=50, edgecolors='k', facecolor='orange',marker='*', label='Prog.', rasterized=True)
            
    cb = fig_xz.colorbar(plot,  ax=ax, location='right', aspect=30, pad=0.01)
    cb.set_label('Stripping time [Gyr]')
    cb.ax.tick_params(labelsize=12)
    
    #-------------------------------------------------------------------------------------
    ### Plot cosmetics
    #-------------------------------------------------------------------------------------
     
    for k in range(len(labels)):

        ax[0,k].set_title(labels[k])
        ax[len(streams)-1,k].set_xlabel(r'$x\,[\mathrm{kpc}]$')
        
    for l in range(len(streams)):
        ax[l, 0].set_ylabel(r'$z\,[\mathrm{kpc}]$')

    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}'.format(plotname + '_xz'))
        
def rv_dispersion(p, v):
    
    galcen_v_sun = (11.1, 245, 7.3)*u.km/u.s
    galcen_distance = 8.249*u.kpc
    
    positions = p + Model.expansion_centres(0.)[:3]
    velocities = v + Model.expansion_centre_velocities(0.)[:3]
    
    posvel_gc = SkyCoord(x=positions[:,0]*u.kpc, y=positions[:,1]*u.kpc, z=positions[:,2]*u.kpc,
                         v_x=velocities[:,0]*u.km/u.s, v_y=velocities[:,1]*u.km/u.s, v_z=velocities[:,2]*u.km/u.s ,
                         frame='galactocentric', galcen_distance=galcen_distance, galcen_v_sun=galcen_v_sun)
    posvel_galactic = posvel_gc.transform_to('galactic')
    posvel_galactic_rc = gc.reflex_correct(posvel_galactic)
    rvs =  posvel_galactic_rc.radial_velocity
    
    return np.nanstd(rvs) 

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
        
def pole_hist(path, plotname, savefig=False):
    
    potentials = list(['static-mwh-only.hdf5','rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5', 
                       'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5', \
                       'Full-MWhalo-MWdisc-LMC.hdf5', 'full-MWhalo-full-MWdisc-no-LMC.hdf5'])
    labels = list(['Static MW','Static Monopole', 'Evolving Monopole', 'Monopole + Dipole', 'Mono + Quadrupole', \
                      'Monopole + Dipole \n + Quadrupole', 'Full Expansion', 'Full Expansion \n (no LMC)'])
    fig, ax = plt.subplots(1,1, figsize=(5,2.5))

    Nstreams = 128
    for j in range(len(potentials)):    
        data_path = pathlib.Path(path) / potentials[j]
        pole_b = []
        for i in range(Nstreams):
            with h5py.File(data_path,'r') as file:

                pole_b.append(np.nanmedian(np.array(file['stream_{}'.format(i)]['pole_b'])[-1]))
                
        sinb = np.sin((pole_b*u.deg).to(u.rad)).value        
        plt.sca(ax)
        plt.hist(sinb, bins=np.linspace(-1,1,20), histtype='step', fill=False, label=labels[j])

    plt.xlabel('$\sin(b_{\mathrm{pole}})$')
    plt.ylabel('N')
    plt.legend(bbox_to_anchor=(1.45,1.), fontsize=9)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}'.format(plotname))
    
        
def radialphase_peris_veldis(galdist, pericenters, apocenters, sigmavs, mass,plotname, potential, savefig=False):

    f = (np.array(galdist) - np.array(pericenters)) / (np.array(apocenters) - np.array(pericenters))
    fig, ax = plt.subplots(1,2, figsize=(9,2.75), sharey='row')
    
    plt.subplots_adjust(wspace=0.)
    plt.sca(ax[0])
    plot=plt.scatter(f, sigmavs, c=mass, cmap='plasma_r', edgecolor='k',
                   norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e6))

    plt.xlabel(r'$\frac{r_{\mathrm{gal}} - r_p}{r_a - r_p}$')
    plt.ylabel('$\sigma_v$ [km/s]')
    plt.xlim(-0.2,1.2)

    plt.sca(ax[1])
    plot=plt.scatter(pericenters, sigmavs, c=mass, cmap='plasma_r', edgecolor='k',
                   norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e6))
    plt.xlabel('$r_{p}$ [kpc]')
    plt.xlim(6,30)
    plt.ylim(0,30)
    
    cb = fig.colorbar(plot, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01)
    cb.set_label(r'$M_{\mathrm{prog}}\,[\mathrm{M}_{\odot}$]')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
        
def poledisp_peri(poledis_l, poledis_b, pericenters, mass, plotname, potential, savefig=False):

    fig, ax = plt.subplots(1,2, figsize=(9,2.75), sharey='row')

    plt.subplots_adjust(wspace=0.)
    plt.sca(ax[0])
    plot=plt.scatter(poledis_l, pericenters, c=mass, cmap='plasma_r', edgecolor='k',
                   norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e6))

    plt.xlabel(r'$\sigma_{l,\mathrm{pole}}\,[^\circ]$')
    plt.xscale('log')
    plt.xlim(1e-1,5e2)
    plt.ylim(9,26)
    plt.ylabel('$r_p$ [kpc]')

    plt.sca(ax[1])
    plot=plt.scatter(poledis_b, pericenters, c=mass, cmap='plasma_r', edgecolor='k',
                   norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e6))
    plt.xlabel(r'$\sigma_{b,\mathrm{pole}}\,[^\circ]$')
    plt.xscale('log')
    plt.xlim(1.01e-1,5e1)

    cb = fig.colorbar(plot, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01)
    cb.set_label(r'$M_{\mathrm{prog}}\,[\mathrm{M}_{\odot}$]')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
        
def poledisp_distance(poledis_l, poledis_b, distances, mass, plotname, potential, savefig=False):
    
    fig, ax = plt.subplots(1,2, figsize=(9,2.75), sharey='row')

    plt.subplots_adjust(wspace=0.)
    plt.sca(ax[0])
    plot=plt.scatter(poledis_l, distances, c=mass, cmap='plasma_r', edgecolor='k',
                   norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e6))

    plt.xlabel(r'$\sigma_{l,\mathrm{pole}}\,[^\circ]$')
    plt.xscale('log')
    plt.ylabel('$r_{\mathrm{gal}}$ [kpc]')
    plt.xlim(1.01e-1,5e2)
    plt.ylim(0,55)

    plt.sca(ax[1])
    plot=plt.scatter(poledis_b, distances, c=mass, cmap='plasma_r', edgecolor='k',
                   norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e6))
    plt.xlabel(r'$\sigma_{b,\mathrm{pole}}\,[^\circ]$')
    plt.xscale('log')
    plt.xlim(1.01e-1,5e1)

    cb = fig.colorbar(plot, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01)
    cb.set_label(r'$M_{\mathrm{prog}}\,[\mathrm{M}_{\odot}$]')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
        
def mollewide_poles_distance(polel, poleb, distance, plotname, potential, savefig=False):
    
    plt.figure(figsize=(8,5))
    plt.subplot(projection="mollweide")
    plt.grid(alpha=.25)
    sc=plt.scatter((polel*u.deg).to(u.rad), (poleb*u.deg).to(u.rad),
               c=distance, cmap='plasma_r', edgecolor='k')

    cb=plt.colorbar(sc,location='right', aspect=30, pad=0.02, shrink=.65)
    cb.set_label(r'Distance [kpc]')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
        
def width_length(width, length, mass, plotname, potential, savefig=False):
    
    fig, ax = plt.subplots(1,1, figsize=(5,3))
    plot=ax.scatter(width, length, c=mass, cmap='plasma_r', edgecolor='k',
                   norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e6))
    plt.sca(ax)
    plt.xlabel('$w\,[^{\circ}]$')
    plt.ylabel('$l$ [kpc]')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-2,3e1)
    plt.ylim(5e-1,1e2)

    cb = fig.colorbar(plot, ax=ax,location='right', aspect=30, pad=0.01)
    cb.set_label(r'$M_{\mathrm{prog}}\,[\mathrm{M}_{\odot}$]')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
        
def av_lon_lat(lons, lats, mass, plotname, potential, savefig=False):
    
    fig, ax = plt.subplots(1,1, figsize=(5,3))
    plot=ax.scatter(lons, lats, c=mass, cmap='plasma_r', edgecolor='k',
                   norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e6))

    plt.sca(ax)
    plt.xlabel(r'$\bar{\psi_{1}}\,[^{\circ}]$')
    plt.ylabel(r'$\bar{\psi_{2}}\,[^{\circ}]$')
    plt.xlim(-20,20)
    plt.ylim(-20,20)

    cb = fig.colorbar(plot, ax=ax,location='right', aspect=30, pad=0.01)
    cb.set_label(r'$M_{\mathrm{prog}}\,[\mathrm{M}_{\odot}$]')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()

    
def veldis_mstellar_rel(m):
    return (-5.28*np.log10(m)) + 53.55

def stellarmass_veldis(mass, veldis, plotname, potential, savefig=False):
    
    fig, ax = plt.subplots(1,1, figsize=(5,3))
    plt.scatter(np.array(masses)[veldis < veldis_mstellar_rel(mass)], np.array(veldis)[veldis < veldis_mstellar_rel(mass)], c='k', s=5)
    plt.scatter(np.array(masses)[veldis > veldis_mstellar_rel(mass)], np.array(veldis)[veldis > veldis_mstellar_rel(mass)], c='r', s=5)

    plt.sca(ax)
    plt.xlabel(r'$M_{\mathrm{*}}\,[\mathrm{M}_{\odot}]$')
    plt.ylabel(r'$\sigma_v$ [km/s]')
    plt.xscale('log')
    
    ms = np.logspace(4,6, 20, base=10.0)
    plt.plot(ms, veldis_mstellar_rel(ms), c='r')
    plt.xlim(1e4,1e6)
    plt.ylim(0.1,110)
    
    Nstreams = len(np.array(masses)[veldis < veldis_mstellar_rel(mass)])
    fsurvived = np.round((Nstreams/len(masses)), 3)*100
    
    plt.annotate('phase-mixed', xy=(5.83e5,45), xytext=(4e5, 26), color='r', fontsize=8,
                arrowprops=dict(color='r', arrowstyle='->'))
    plt.annotate('stream', xy=(5.9e5,3), xytext=(4.75e5, 18), color='k', fontsize=8,
                arrowprops=dict(color='k', arrowstyle='->'))
    plt.annotate(r'$f_{\mathrm{survived}}$' + '= {}\%'.format(fsurvived), xy=(2.75e5, 90), 
                 xytext=(2.75e5, 100), color='k', fontsize=10,)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
    
def rlmc_veldis(rlmc, veldis, pericenters, plotname, potential, savefig=False):

    fig, ax = plt.subplots(1,1, figsize=(5,3)) 
    plot=plt.scatter(np.nanmin(rlmc,axis=1), np.array(veldis), c=pericenters, 
                     edgecolor='k',cmap='plasma_r', vmin=10, vmax=25)

    plt.xlabel(r'Closest approach to LMC [kpc]')
    plt.ylabel(r'$\sigma_v$ [km/s]')
    plt.xlim(0,49)
    plt.ylim(0,31)
    
    cb = fig.colorbar(plot, ax=ax,location='right', aspect=30, pad=0.01)
    cb.set_label(r'$r_{p}\,[\mathrm{kpc}]$')
    cb.ax.tick_params(labelsize=12)

    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
        
###-------------------------------------------------------------------------------
### run the script
###--------------------------------------------------------------------------------

path = '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/'
# streams = list(['stream_0', 'stream_1','stream_2','stream_3','stream_4']) 
# plot_stream_frames(streams, path, 'plot_stream_coords', True)
        
# pole_hist(path, 'sinbpole-histogram', True)

potentials_list = list(['static-mwh-only.hdf5','rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5', \
                   'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5', \
                   'Full-MWhalo-MWdisc-LMC.hdf5', 'full-MWhalo-full-MWdisc-no-LMC.hdf5'])

for potential in potentials_list:

    rgal = []
    peris = []
    apos = []
    widths = []
    lengths = []
    av_lon = []
    av_lat = []
    loc_veldis = []
    lmc_sep = []
    pole_b = []
    pole_b_dis = []
    pole_l = []
    pole_l_dis = []
    masses = []
    energy = []
    Ls = []
    Lzs = []
    
    Nstreams = 128
    for i in range(Nstreams):
        data_path = pathlib.Path(path) / potential 
        with h5py.File(data_path,'r') as file:

            if i ==1:
                pot_folder = file['stream_{}'.format(i)]['potential'][()].decode('utf-8')

            pos = np.array(file['stream_{}'.format(i)]['positions'])[-1]
            vel = np.array(file['stream_{}'.format(i)]['velocities'])[-1]
            lons, lats = lons_lats(pos, vel)
            loc_veldis.append(local_veldis(lons, vel))
            rgal.append( np.nanmedian(np.linalg.norm(np.array(file['stream_{}'.format(i)]['positions'])[-1],axis=1)) )
            peris.append(np.array(file['stream_{}'.format(i)]['pericenter']))
            apos.append(np.array(file['stream_{}'.format(i)]['apocenter']))
            widths.append(np.array(file['stream_{}'.format(i)]['width']))
            lengths.append(np.array(file['stream_{}'.format(i)]['length']))
            av_lon.append(np.array(file['stream_{}'.format(i)]['av_lon']))
            av_lat.append(np.array(file['stream_{}'.format(i)]['av_lat']))
            lmc_sep.append(np.array(file['stream_{}'.format(i)]['lmc_sep']))
            pole_b.append(np.nanmedian(np.array(file['stream_{}'.format(i)]['pole_b'])[-1]))
            pole_l.append(np.nanmedian(np.array(file['stream_{}'.format(i)]['pole_l'])[-1]))
            pole_b_dis.append(np.nanstd(np.array(file['stream_{}'.format(i)]['pole_b'])[-1]))
            pole_l_dis.append(np.nanstd(np.array(file['stream_{}'.format(i)]['pole_l'])[-1]))
            masses.append(np.array(file['stream_{}'.format(i)]['progenitor-mass']))
            energy.append(np.nanmedian(np.array(file['stream_{}'.format(i)]['energies'])[-1]))
            Ls.append(np.nanmedian(np.array(file['stream_{}'.format(i)]['L'])[-1]))
    print('* Saving figures for potential: {}'.format(potential))
    
    radialphase_peris_veldis(rgal, peris, apos, loc_veldis, masses, 'radialphase_peris_veldis', pot_folder, True)
    # poledisp_peri(pole_l_dis, pole_b_dis, peris, masses, 'poledisp_peri', pot_folder, True)
    # poledisp_distance(pole_l_dis, pole_b_dis, rgal, masses, 'poledisp_distance', pot_folder, True)
    # mollewide_poles_distance(pole_l, pole_b, rgal, 'mollewide_poles_distance', pot_folder, True)
    # width_length(widths, lengths, masses, 'width_length', pot_folder, True)
    # av_lon_lat(av_lon, av_lat, masses, 'av_lon_lat', pot_folder, True)
    # stellarmass_veldis(masses, loc_veldis, 'stellarmass_veldis', pot_folder, True)
    # rlmc_veldis(lmc_sep, loc_veldis, peris, 'rlmc_veldis', pot_folder, True)
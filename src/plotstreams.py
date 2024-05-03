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
        
def pole_hist(path, plotname, savefig=False):
    
    potentials = list(['static-mwh-only.hdf5','rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5', 
                       'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5', \
                       'Full-MWhalo-MWdisc-LMC.hdf5', 'full-MWhalo-full-MWdisc-no-LMC.hdf5'])
    labels = list(['Static MW','Static Monopole', 'Evolving Monopole', 'Monopole + Dipole', 'Mono + Quadrupole', \
                      'Monopole + Dipole \n + Quadrupole', 'Full Expansion', 'Full Expansion \n (no LMC)'])
    fig, ax = plt.subplots(1,1, figsize=(5,2.5))

    Nstreams = 1024
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
    plt.close()
    
        
def radialphase_peris_veldis(galdist, pericenters, apocenters, sigmavs, mass,plotname, potential, savefig=False):

    f = (np.array(galdist) - np.array(pericenters)) / (np.array(apocenters) - np.array(pericenters))
    fig, ax = plt.subplots(1,2, figsize=(9,2.75), sharey='row')

    plt.subplots_adjust(wspace=0.)
    plt.sca(ax[0])
    
    x_range, xbins = (0, 1) , 20
    y_range, ybins = (0, 50) , 20
    plot = plt.hexbin(f, sigmavs, cmap='magma',
                      gridsize=(xbins, ybins), extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                      norm=matplotlib.colors.LogNorm(vmin=1e0, vmax=1e2))
    
    plt.xlabel(r'$\frac{r_{\mathrm{gal}} - r_p}{r_a - r_p}$')
    plt.ylabel('$\sigma_{v,\,\mathrm{loc}}$ [km/s]')
    plt.xlim(-0.05,1.05)

    plt.sca(ax[1])
    x_range, xbins = (5, 25) , 20
    y_range, ybins = (0, 50) , 20
    plot = plt.hexbin(pericenters, sigmavs, cmap='magma',
                      gridsize=(xbins, ybins), extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                      norm=matplotlib.colors.LogNorm(vmin=1e0, vmax=1e2)) 
    plt.xlabel('$r_{p}$ [kpc]')
    plt.xlim(1,26)
    plt.ylim(0,50)
    
    cb = fig.colorbar(plot, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01)
    cb.set_label('Number counts')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
    
def peri_veldis_scatter(rlmc, veldis, peris, plotname, potential, savefig=False):
    
    fig, ax = plt.subplots(1,1, figsize=(5,3)) 
    plot = plt.scatter(peris, veldis, c=np.nanmin(rlmc,axis=1), 
                       cmap='viridis_r', edgecolor='k', rasterized=True,
                     vmin=0, vmax=25)
    
    plt.xlabel(r'$r_p$ [kpc]')
    plt.ylabel(r'$\sigma_v$ [km/s]')
    plt.xlim(1,26)
    plt.ylim(0,100)
 
    cb = fig.colorbar(plot, ax=ax,location='right', aspect=30, pad=0.01)
    cb.set_label(r'Closest approach to LMC [kpc]')
    cb.ax.tick_params(labelsize=12)

    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
    
def peri_veldis_dist_scatter(veldis, peris, distance, plotname, potential, savefig=False):
        
    fig, ax = plt.subplots(1,1, figsize=(5,3)) 
    plot = plt.scatter(peris, veldis, c=distance, 
                       cmap='viridis_r', edgecolor='k', rasterized=True,
                     vmin=0,)
    
    plt.xlabel(r'$r_p$ [kpc]')
    plt.ylabel(r'$\sigma_v$ [km/s]')
    plt.xlim(1,26)
    plt.ylim(0,100)
 
    cb = fig.colorbar(plot, ax=ax,location='right', aspect=30, pad=0.01)
    cb.set_label(r'$\bar{d}$ [kpc]')
    cb.ax.tick_params(labelsize=12)

    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
        
def poledisp_peri(poledis_l, poledis_b, pericenters, mass, plotname, potential, savefig=False):

    fig, ax = plt.subplots(1,2, figsize=(9,2.75), sharey='row')

    plt.subplots_adjust(wspace=0.)
    plt.sca(ax[0])
    x_bins_log = np.logspace(np.log10(0.1), np.log10(250), 25)
    y_range, ybins = (9, 26) , 20
    plot = plt.hexbin(np.log10(poledis_l), pericenters, cmap='magma',
                      gridsize=(x_bins_log.size, ybins),
                      norm=matplotlib.colors.LogNorm(vmin=1e0, vmax=1e2))

    plt.xlabel(r'$\log_{10}(\sigma_{l,\mathrm{pole}})\,[^\circ]$')
    plt.xlim(np.log10(0.1),np.log10(300))
    plt.ylim(1,26)
    plt.ylabel('$r_p$ [kpc]')

    plt.sca(ax[1])
    x_bins_log = np.logspace(np.log10(0.1), np.log10(50), 20)
    y_range, ybins = (5, 25) , 20
    plot = plt.hexbin(np.log10(poledis_b), pericenters, cmap='magma',
                      gridsize=(x_bins_log.size, ybins),
                      norm=matplotlib.colors.LogNorm(vmin=1e0, vmax=1e2))
    plt.xlim(np.log10(0.1),np.log10(50))
    plt.xlabel(r'$\log_{10}(\sigma_{b,\mathrm{pole}})\,[^\circ]$')

    cb = fig.colorbar(plot, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01)
    cb.set_label('Number counts')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
        
def poledisp_distance(poledis_l, poledis_b, distances, mass, plotname, potential, savefig=False):
    
    fig, ax = plt.subplots(1,2, figsize=(9,2.75), sharey='row')

    plt.subplots_adjust(wspace=0.)
    plt.sca(ax[0])
    x_bins_log = np.logspace(np.log10(0.1), np.log10(250), 25)
    y_range, ybins = (0, 55) , 20
    plot = plt.hexbin(np.log10(poledis_l), distances, cmap='magma',
                      gridsize=(x_bins_log.size, ybins),
                      norm=matplotlib.colors.LogNorm(vmin=1e0, vmax=1e2))

    plt.xlabel(r'$\log_{10}(\sigma_{l,\mathrm{pole}})\,[^\circ]$')
    plt.ylabel('$r_{\mathrm{gal}}$ [kpc]')
    plt.xlim(np.log10(0.1),np.log10(300))
    plt.ylim(0,55)

    plt.sca(ax[1])
    x_bins_log = np.logspace(np.log10(0.1), np.log10(50), 20)
    y_range, ybins = (9, 26) , 20
    plot = plt.hexbin(np.log10(poledis_b), distances, cmap='magma',
                      gridsize=(x_bins_log.size, ybins),
                      norm=matplotlib.colors.LogNorm(vmin=1e0, vmax=1e2))
    plt.xlabel(r'$\log_{10}(\sigma_{b,\mathrm{pole}})\,[^\circ]$')
    plt.xlim(np.log10(0.1),np.log10(50))

    cb = fig.colorbar(plot, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01)
    cb.set_label('Number counts')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
        
def mollewide_poles_distance(polel, poleb, distance, plotname, potential, savefig=False):
    
    plt.figure(figsize=(8,5))
    plt.subplot(projection="mollweide")
    plt.grid(alpha=.25)
    sc=plt.scatter((polel*u.deg).to(u.rad), (poleb*u.deg).to(u.rad),
               c=distance, cmap='plasma_r', edgecolor='k', rasterized=True)

    cb=plt.colorbar(sc,location='right', aspect=30, pad=0.02, shrink=.65)
    cb.set_label(r'Distance [kpc]')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
        
def width_length(width, length, mass, plotname, potential, savefig=False):
    
    fig, ax = plt.subplots(1,1, figsize=(5,3))
    x_bins_log = np.logspace(np.log10(1e-2), np.log10(1e2), 30)
    y_bins_log = np.logspace(np.log10(5e-1), np.log10(360), 30)
    plot = plt.hexbin(np.log10(width), np.log10(length), cmap='magma',
                      gridsize=(x_bins_log.size, y_bins_log.size),
                      norm=matplotlib.colors.LogNorm(vmin=1e0, vmax=1e2))
    plt.sca(ax)
    plt.xlabel('$\log_{10}(w)\,[^{\circ}]$')
    plt.ylabel('$\log_{10}(l)\,[^{\circ}]$')
    plt.xlim(np.log10(1e-2),np.log10(1e2))
    plt.ylim(np.log10(3),np.log10(360))

    cb = fig.colorbar(plot, ax=ax,location='right', aspect=30, pad=0.01)
    cb.set_label('Number counts')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
        
def av_lon_lat(lons, lats, mass, plotname, potential, savefig=False):
    
    fig, ax = plt.subplots(1,1, figsize=(5,3))
    
    x_range, xbins = (-5, 5) , 25
    y_range, ybins = (-5, 5) , 20
    plot = plt.hexbin(lons, lats, cmap='magma',
                      gridsize=(xbins, ybins), extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                      norm=matplotlib.colors.LogNorm(vmin=1e0, vmax=1e2))

    plt.sca(ax)
    plt.xlabel(r'$\bar{\psi_{1}}\,[^{\circ}]$')
    plt.ylabel(r'$\bar{\psi_{2}}\,[^{\circ}]$')
    plt.xlim(-5,5)
    plt.ylim(-5,5)

    cb = fig.colorbar(plot, ax=ax,location='right', aspect=30, pad=0.01)
    cb.set_label('Number counts')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()

    
def veldis_mstellar_rel(m):
    return (-5.28*np.log10(m)) + 53.55

def stellarmass_veldis(mass, veldis, plotname, potential, savefig=False):
    
    fig, ax = plt.subplots(1,1, figsize=(5,3))
    plt.scatter(np.array(masses)[veldis < veldis_mstellar_rel(mass)], np.array(veldis)[veldis < veldis_mstellar_rel(mass)], c='k', s=5, rasterized=True)
    plt.scatter(np.array(masses)[veldis > veldis_mstellar_rel(mass)], np.array(veldis)[veldis > veldis_mstellar_rel(mass)], c='r', s=5, rasterized=True)

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
    
def rlmc_veldis(rlmc, veldis, plotname, potential, savefig=False):
    
    fig, ax = plt.subplots(1,1, figsize=(5,3)) 
    
    x_range, xbins = (0, 50) , 25
    y_range, ybins = (0, 50) , 20
    plot = plt.hexbin(np.nanmin(rlmc,axis=1), veldis, cmap='magma',
                      gridsize=(xbins, ybins), extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                      norm=matplotlib.colors.LogNorm(vmin=1e0, vmax=1e2))

    plt.xlabel(r'Closest approach to LMC [kpc]')
    plt.ylabel(r'$\sigma_v$ [km/s]')
    plt.xlim(0,49)
    plt.ylim(0,50)
 
    cb = fig.colorbar(plot, ax=ax,location='right', aspect=30, pad=0.01)
    cb.set_label('Number counts')
    cb.ax.tick_params(labelsize=12)

    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}/{}'.format(potential, plotname + '_' + potential))
    plt.close()
    
def plt_1dhists(path, plotname, savefig=False):
    fig, ax = plt.subplots(2,3, figsize=(13,5.5))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    potentials = list(['static-mwh-only.hdf5','rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5',\
                           'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5',\
                           'Full-MWhalo-MWdisc-LMC.hdf5', 'full-MWhalo-full-MWdisc-no-LMC.hdf5'])

    labels = list(['Static MW','Static Monopole', 'Evolving Monopole', 'Monopole + Dipole', 'Mono + Quadrupole', \
                   'Monopole + Dipole + Quadrupole', 'Full Expansion', 'Full Expansion (no LMC)'])

    for j in range(len(potentials)): 

        with h5py.File(path + potentials[j],'r') as file:
            lengths = np.array(file['lengths'])
            widths = np.array(file['widths'])
            loc_veldis = np.array(file['loc_veldis'])
            energies = np.array(file['energies'])
            track_deform = np.array(file['track_deform'])
            Lzs = np.array(file['Lz'])

        # lengths
        plt.sca(ax[0,0])
        h, bin_edges = np.histogram(lengths, bins=np.linspace(-1, 360, 10))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        plt.plot(bin_mids, h, label=labels[j])
        plt.xlabel(r'$l\,[^{\circ}]$')
        plt.ylabel('Counts')
        plt.xlim(0, 360)
        plt.ylim(0.1,)
        plt.legend(frameon=False, ncol=4, fontsize=12, bbox_to_anchor=(3.6,1.35))

        #widths
        plt.sca(ax[1,0])
        h, bin_edges = np.histogram(widths, bins=np.linspace(-0., 3, 15))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        plt.plot(bin_mids, h)
        plt.xlabel(r'$w\,[^{\circ}]$')
        plt.ylabel('Counts')
        plt.xlim(0,)
        plt.ylim(0.1,)

        # velocity dispersion
        plt.sca(ax[0,1])
        h, bin_edges = np.histogram(loc_veldis, bins=np.linspace(-0, 50, 25))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        plt.plot(bin_mids, h)
        plt.xlabel(r'$\sigma_{v}\,[\mathrm{km}\,\mathrm{s}^{-1}]$')
        plt.ylabel('Counts')
        plt.xlim(0,50)
        plt.ylim(0.1,)
        
        # track deformation
        plt.sca(ax[1,1])
        h, bin_edges = np.histogram(track_deform, bins=np.linspace(-0., 10, 25))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        plt.plot(bin_mids, h)
        plt.xlabel(r'$\bar{\delta}\,[^{\circ}]$')
        plt.ylabel('Counts')
        plt.ylim(0.1,)

        # median energies
        plt.sca(ax[0,2])
        h, bin_edges = np.histogram(np.log10(-energies), bins=np.linspace(4, 5.5, 30))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        plt.plot(bin_mids, h)
        plt.xlabel(r'$\log_{10}(\bar{E})\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$')
        plt.ylabel('Counts')
        plt.xlim(4,5.5)
        plt.ylim(0.1,)
        
        # median Lz
        plt.sca(ax[1,2])
        h, bin_edges = np.histogram(Lzs, bins=np.linspace(-6000, 6000, 30))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        plt.plot(bin_mids, h)
        plt.xlabel(r'$L_{z}\,[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{kpc}]$')
        plt.ylabel('Counts')
        plt.ylim(0.1,)
        
    if savefig==False:
        return
    elif savefig==True:
        savepath = '/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}'.format(plotname)
        print('* Saving figure at {}.pdf'.format(savepath))
        plt.savefig(savepath)
    plt.close()
    
    
def plt_1dhists_quadrants(path, plotname, savefig=False):
    
    potentials = list(['static-mwh-only.hdf5','rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5',\
                           'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5',\
                           'Full-MWhalo-MWdisc-LMC.hdf5', 'full-MWhalo-full-MWdisc-no-LMC.hdf5'])
    
    labels = list(['Static MW','Static Monopole', 'Evolving Monopole', 'Monopole + Dipole', 'Mono + Quadrupole', \
                   'Monopole + Dipole \n + Quadrupole', 'Full Expansion', 'Full Expansion \n (no LMC)'])
    
    quadrants = list(['Q1', 'Q2', 'Q3', 'Q4'])
    
    fig, ax = plt.subplots(6,len(potentials), figsize=(24,12))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for j in range(len(potentials)): 

        with h5py.File(path + potentials[j],'r') as file:
            
            l_gc,b_gc  = np.array(file['l_gc']), np.array(file['b_gc'])
            mask_q1 = ( (l_gc > 0) & (l_gc < 180) & (b_gc > 0) & (b_gc < 90) )
            mask_q2 = ( (l_gc > 180) & (l_gc < 360) & (b_gc > 0) & (b_gc < 90) )
            mask_q3 = ( (l_gc > 0) & (l_gc < 180) & (b_gc > -90) & (b_gc < 0) )
            mask_q4 = ( (l_gc > 180) & (l_gc < 360) & (b_gc > -90) & (b_gc < 0) )
            
            masks = [mask_q1, mask_q2, mask_q3, mask_q4]
            
            lengths = np.array(file['lengths'])
            widths = np.array(file['widths'])
            loc_veldis = np.array(file['loc_veldis'])
            energies = np.array(file['energies'])
            track_deforms = np.array(file['track_deform'])
            Ls = np.array(file['L'])
            Lzs = np.array(file['Lz'])
            
            # print(track_deforms)
            
        for m in range(len(masks)):
        
            # lengths
            plt.sca(ax[0,j])
            h, bin_edges = np.histogram(lengths[masks[m]], bins=np.linspace(-1, 360, 10))
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
            plt.plot(bin_mids, h, label=quadrants[m])
            plt.xlabel(r'$l\,[^{\circ}]$')
            plt.xlim(0, 360)
            plt.ylim(0.1,)
            if j==0:
                plt.legend(frameon=False,fontsize=10)

            #widths
            plt.sca(ax[1,j])
            h, bin_edges = np.histogram(widths[masks[m]], bins=np.linspace(0, 3, 15))
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
            plt.plot(bin_mids, h)
            plt.xlabel(r'$w\,[^{\circ}]$', fontsize=14)
            plt.xlim(0,)
            plt.ylim(0.1,)

            # velocity dispersion
            plt.sca(ax[2,j])
            h, bin_edges = np.histogram(loc_veldis[masks[m]], bins=np.linspace(0, 20, 15))
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
            plt.plot(bin_mids, h)
            plt.xlabel(r'$\sigma_{v}\,[\mathrm{km}\,\mathrm{s}^{-1}]$', fontsize=14)
            plt.xlim(0,20)
            plt.ylim(0.1,)
            
            # track deforms
            plt.sca(ax[3,j])
            h, bin_edges = np.histogram(track_deforms[masks[m]], bins=np.linspace(-0., 10, 20))
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
            plt.plot(bin_mids, h)
            plt.xlabel(r'$\bar{\delta}\,[^{\circ}]$', fontsize=14)
            plt.ylim(0.1,)

            # median energies
            plt.sca(ax[4,j])
            h, bin_edges = np.histogram(np.log10(-energies)[masks[m]], bins=np.linspace(4, 5.5, 25))
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
            plt.plot(bin_mids, h)
            plt.xlabel(r'$\log_{10}(\bar{E})\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$', fontsize=14)
            plt.xlim(4,5.5)
            plt.ylim(0.1,)

            # # median L
            # plt.sca(ax[4,j])
            # h, bin_edges = np.histogram(Ls[masks[m]], bins=np.linspace(0, 6500, 20))
            # bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
            # plt.plot(bin_mids, h)
            # plt.xlabel(r'$L\,[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{kpc}]$', fontsize=14)
            # plt.ylim(0.1,)

            # median Lz
            plt.sca(ax[5,j])
            h, bin_edges = np.histogram(Lzs[masks[m]], bins=np.linspace(-6000, 6000, 20))
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
            plt.plot(bin_mids, h)
            plt.xlabel(r'$L_{z}\,[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{kpc}]$', fontsize=14)
            plt.ylim(0.1,)
        
    #-------------------------------------------------------------------------------------
    ### Plot cosmetics
    #-------------------------------------------------------------------------------------
    for k in range(len(labels)):
        ax[0,k].set_title(labels[k], fontsize=14)
        
    for l in range(6):
        ax[l, 0].set_ylabel('Counts', fontsize=14)
        
    if savefig==False:
        return
    elif savefig==True:
        savepath = '/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}'.format(plotname)
        print('* Saving figure at {}.pdf'.format(savepath))
        plt.savefig(savepath)
    plt.close()
        
###-------------------------------------------------------------------------------
### run the script
###--------------------------------------------------------------------------------

# streams = list(['stream_0', 'stream_1','stream_2','stream_3','stream_4']) 
# plot_stream_frames(streams, '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/', 'plot_stream_coords', True)
# pole_hist(path, 'sinbpole-histogram', True)

potentials_list = list(['static-mwh-only.hdf5','rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5', \
                   'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5', \
                   'Full-MWhalo-MWdisc-LMC.hdf5', 'full-MWhalo-full-MWdisc-no-LMC.hdf5'])

pot_folders = list(['static-mwh-only', 'rm-MWhalo-full-MWdisc-full-LMC', 'em-MWhalo-full-MWdisc-full-LMC',
                  'md-MWhalo-full-MWdisc-full-LMC', 'mq-MWhalo-full-MWdisc-full-LMC', 'mdq-MWhalo-full-MWdisc-full-LMC',
                  'Full-MWhalo-MWdisc-LMC', 'full-MWhalo-full-MWdisc-no-LMC'])

# path = '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/plotting_data/'
path = '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/plotting_data/1024-dthalfMyr-10rpmin-75ramax/'
for (potential, folder) in zip(potentials_list, pot_folders):
    with h5py.File(path + potential,'r') as file:
            pot_folder = folder
            rgal = np.array(file['ds'])
            peris = np.array(file['pericenter'])
            apos = np.array(file['apocenter'])
            loc_veldis = np.array(file['loc_veldis'])
            masses = np.array(file['mass'])
            
            pole_l = np.array(file['pole_l'])[:,-1]
            pole_l_dis =np.nanstd(pole_l, axis=1)
            pole_b = np.array(file['pole_b'])[:,-1]
            pole_b_dis = np.nanstd(pole_b, axis=1)
        
            widths = np.array(file['widths'])
            lengths = np.array(file['lengths'])
            av_lon = np.array(file['av_lon'])
            av_lat = np.array(file['av_lat'])
            lmc_sep = np.array(file['lmc_sep'])
                 
    print('* Saving figures for potential: {}'.format(potential))     
    radialphase_peris_veldis(rgal, peris, apos, loc_veldis, masses, 'radialphase_peris_veldis', pot_folder, True)
    peri_veldis_scatter(lmc_sep, loc_veldis, peris, 'peri_veldis_scatter', pot_folder, True)
    peri_veldis_dist_scatter(loc_veldis, peris, rgal, 'peri_veldis_dist_scatter', pot_folder, savefig=True)
    poledisp_peri(pole_l_dis, pole_b_dis, peris, masses, 'poledisp_peri', pot_folder, True)
    poledisp_distance(pole_l_dis, pole_b_dis, rgal, masses, 'poledisp_distance', pot_folder, True)
    # mollewide_poles_distance(pole_l, pole_b, rgal, 'mollewide_poles_distance', pot_folder, True)
    width_length(widths, lengths, masses, 'width_length', pot_folder, True)
    av_lon_lat(av_lon, av_lat, masses, 'av_lon_lat', pot_folder, True)
    stellarmass_veldis(masses, loc_veldis, 'stellarmass_veldis', pot_folder, True)
    rlmc_veldis(lmc_sep, loc_veldis, 'rlmc_veldis', pot_folder, True)

# plt_1dhists('/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/plotting_data/', '1d-hists' , True)
# plt_1dhists_quadrants('/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/plotting_data/', '1d-hists-quad' , True)
from scipy.spatial.transform import Rotation
import scipy
import pathlib
import h5py

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, SphericalRepresentation
import yaml

import matplotlib
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
plt.style.use('my_standard.mplstyle')

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
        return plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}'.format(plotname))
    
    
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
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
    
    potentials = list(['rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5', 'md-MWhalo-full-MWdisc-full-LMC.hdf5', \
                       'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5', 'Full-MWhalo-MWdisc-LMC.hdf5', \
                       'full-MWhalo-full-MWdisc-no-LMC.hdf5'])
    labels = list(['Static Monopole', 'Evolving Monopole', 'Monopole + Dipole', 'Mono + Quadrupole', \
                      'Monopole + Dipole \n + Quadrupole', 'Full Expansion', 'Full Expansion \n (no LMC)'])
    
    fig, ax = plt.subplots(len(streams), len(potentials), sharex='col', sharey='row', figsize=(17,5))
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
            prog_Vxyz = CartesianRepresentation(t0_prog_vel[0], y=t0_prog_vel[1], z=t0_prog_vel[2])
            GC_sph = prog_xyz.represent_as(SphericalRepresentation)
            prog_lon, prog_lat = GC_sph.lon.deg, GC_sph.lat.deg        
     
            #-------------------------------------------------------------------------------------
            ### Rotation matrix for progenitor to get it to near (X, 0, 0)
            #-------------------------------------------------------------------------------------
            R1 = Rotation.from_euler('z', -prog_lon, degrees=True)
            R2 = Rotation.from_euler('y', prog_lat, degrees=True)
            R_prog0 = R2.as_matrix() @ R1.as_matrix() 
            
            #-------------------------------------------------------------------------------------
            ### Rotate the progenitor xyx and V_xyz
            #-------------------------------------------------------------------------------------
            rot_prog_pos = R_prog0 @ t0_prog_pos
            rot_prog_vel = R_prog0 @ t0_prog_vel
            
            #-------------------------------------------------------------------------------------
            ### Rotate around new x axis so stream prog vel points along +y direction
            #-------------------------------------------------------------------------------------
            vel_rot = (np.arctan(rot_prog_vel[2]/rot_prog_vel[1])*u.rad).to(u.deg)
            R3 = Rotation.from_euler('x', vel_rot.value, degrees=True).as_matrix() 
            R_progf = R3 @ R_prog0
            
            #-------------------------------------------------------------------------------------
            ### Rotate the whole stream by the final rotation matrix
            #-------------------------------------------------------------------------------------
            
            rot_stream_xs = np.dot(R_progf, t0_pos.T).T
            rot_stream_vs = np.dot(R_progf, t0_vel.T).T
            
            #-------------------------------------------------------------------------------------
            ### Plot the streams
            #-------------------------------------------------------------------------------------
                
            plt.sca(ax[i,j])
            print('* Plotting {} in potential {}'.format(streams[i], potentials[j]))
            plt.hlines(0, -200, 200, ls='dashed', color='lightgrey', lw=0.7)
            plt.vlines(0, -200, 200, ls='dashed', color='lightgrey', lw=0.7)
            plot=plt.scatter(rot_stream_xs[:,1],rot_stream_xs[:,2], s=1, c=start_times, cmap = 'viridis',rasterized=True)
            plt.scatter(rot_stream_xs[0][1],rot_stream_xs[0][2], s=200, edgecolors='k', facecolor='orange',marker='*', label='Prog.', rasterized=True)
            
            if j==0:
                name, ext = os.path.splitext(streams[i])
                plt.annotate(text='{}'.format(name), xy=(-35,15), fontsize=8)
            
    cb = fig.colorbar(plot,  ax=ax, location='right', aspect=30, pad=0.01)
    cb.set_label('Stripping time [Gyr]')
    cb.ax.tick_params(labelsize=12)
    
    #-------------------------------------------------------------------------------------
    ### Plot cosmetics
    #-------------------------------------------------------------------------------------
    
    lgd = ax[2,0].legend(frameon=False, fontsize=8, loc='lower left')
    lgd.legend_handles[0]._sizes = [75]
            
    for k in range(len(labels)):

        ax[0,k].set_title(labels[k])
        ax[len(streams)-1,k].set_xlabel(r'$y^{\prime}\,[\mathrm{kpc}]$')
        ax[len(streams)-1,k].set_xlim(-39,39)
        
    for l in range(len(streams)):
        ax[l, 0].set_ylabel(r'$z^{\prime}\,[\mathrm{kpc}]$')
        ax[l, 0].set_ylim(-19,19)

    if savefig==False:
        return
    elif savefig==True:
        return plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/{}'.format(plotname))
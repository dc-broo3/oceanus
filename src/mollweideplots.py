print("Loading modules and packages...")

from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
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
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.patches import Polygon
import os
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
plt.style.use('/mnt/ceph/users/rbrooks/oceanus/analysis/my_standard.mplstyle')

galcen_v_sun = (11.1, 245, 7.3)*u.km/u.s
galcen_distance = 8.249*u.kpc

###--------------------------------------------------------------------------------------------
### Functions that plotting functions rely upon.
###--------------------------------------------------------------------------------------------

def orbpole(xs,vs):
    uu = np.cross(xs, vs)
    uumag = np.linalg.norm(uu)
    u = uu.T/uumag
    b = np.arcsin(u[2])
    sinl = u[1]/np.cos(b)
    cosl = u[0]/np.cos(b)
    ll = np.arctan2(sinl, cosl)
    gl = np.degrees(ll)
    gb = np.degrees(b)
    return gl, gb   

def galactic_coords(p, v):
    
    galcen_v_sun = (11.1, 245, 7.3)*u.km/u.s
    galcen_distance = 8.249*u.kpc
    
    posvel_gc = SkyCoord(x=p[:,0]*u.kpc, y=p[:,1]*u.kpc, z=p[:,2]*u.kpc,
                         v_x=v[:,0]*u.km/u.s, v_y=v[:,1]*u.km/u.s, v_z=v[:,2]*u.km/u.s ,
                         frame='galactocentric', galcen_distance=galcen_distance, galcen_v_sun=galcen_v_sun)
    posvel_galactic = posvel_gc.transform_to('galactic')
    posvel_galactic_rc = gc.reflex_correct(posvel_galactic)
    l, b, d = posvel_galactic_rc.l, posvel_galactic_rc.b, posvel_galactic_rc.distance
    pm_l_cosb, pm_b, rvs = posvel_galactic_rc.pm_l_cosb, posvel_galactic_rc.pm_b, posvel_galactic_rc.radial_velocity
    
    return l.value, b.value, d.value, pm_l_cosb.value, pm_b.value, rvs.value

def rotation_matrix_from_vectors(v1, v2):
    # Normalize the vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Calculate the cosine and sine of the angle between the vectors
    cos_theta = np.dot(v1, v2)
    sin_theta = np.cross(v1, v2)
    
    # Construct the rotation matrix
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta, cos_theta]])
    return R

disc_xs, disc_vs = Model.expansion_centres(0.)[:3], Model.expansion_centre_velocities(0.)[:3]
disc_lpole, disc_bpole = orbpole(np.array(disc_xs), np.array(disc_vs))
v1_disc = np.array([disc_lpole, disc_bpole])
v2 = np.array([0, 90])
rotation_matrix_disc = rotation_matrix_from_vectors(v1_disc, v2)

def mollweide(l, b, bmin, bmax, title="", unit="", nside=24, smooth=1, q=[0], sub=421, **kwargs):

    mwlmc_indices = hp.ang2pix(nside,  (90-b)*np.pi/180., l*np.pi/180.)
    npix = hp.nside2npix(nside)
 
    idx, counts = np.unique(mwlmc_indices, return_counts=True)
    degsq = hp.nside2pixarea(nside, degrees=True)
    # filling the full-sky map
    hpx_map = np.zeros(npix, dtype=float)
    if q[0] != 0 :    
        counts = np.zeros_like(idx, dtype=float)
        k=0
        for i in idx:
            pix_ids = np.where(mwlmc_indices==i)[0]
            counts[k] = np.mean(q[pix_ids])
            k+=1
        hpx_map[idx] = counts
    else:
        hpx_map[idx] = counts/degsq
    
    map_smooth = hp.smoothing(hpx_map, fwhm=smooth*np.pi/180)
    
    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
    else:
        cmap='viridis'
    
       
    projview(
      map_smooth,
      coord=["G"], # Galactic
      graticule=True,
      graticule_labels=True,
      rot=(0, 0, 0),
      unit=unit,
      sub=sub,
      xlabel="Galactic Longitude (l) ",
      ylabel="Galactic Latitude (b)",
      flip='astro',
      cb_orientation="horizontal",
      min=bmin,
      max=bmax,
      latitude_grid_spacing=45,
      projection_type="mollweide",
      title=title,
      cmap=cmap,
      label='Colourbar label',
      fontsize={"xlabel": 15, "ylabel": 15,"xtick_label": 10,"ytick_label": 10,
              "title": 15, "cbar_label": 12,"cbar_tick_label": 12},
      override_plot_properties={"cbar_shrink":.5, "figure_size_ratio":1,
                              "cbar_pad":.1,'cbar_label_pad': 1} )
	
    if 'l2' in kwargs.keys():
        l2 = kwargs['l2']
        b2 = kwargs['b2']
        newprojplot(theta=np.radians(90-(b2)), phi=np.radians(l2),
                    marker="o",color='k', markersize=0., lw=2, mfc='k')
        newprojplot(theta=np.radians(90-(b2[-1])), phi=np.radians(l2[-1]), marker="*",color='k', markersize=15, lw=0, mfc='lightgrey')
        
###--------------------------------------------------------------------------------------------
### Plotting functions.
###--------------------------------------------------------------------------------------------

def mollewide_gc(path_data, pots, cbar, pot_labels, plotname, savefig=False):

    pltidx = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1)]
    
    fig, ax = plt.subplots(4, 2, subplot_kw={'projection': 'mollweide'}, figsize=(14, 16))
    
    plt.subplots_adjust(hspace=-0.1, wspace=0.1)

    for j in range(len(pots)): 

        with h5py.File(path_data + potentials[j],'r') as file:
            
            t_idx = -1

            l_gc = np.array(file['l_gc'])
            b_gc = np.array(file['b_gc'])
            ds = np.array(file['ds'])
            loc_veldis = np.array(file['loc_veldis'])
            widths = np.array(file['widths'])
            track_deform = np.array(file['track_deform'])
            pm_ang = np.array(file['pm_misalignment'])
      
        rng = np.random.default_rng(seed=1)

        wrapped_ls = np.where(l_gc>=180, l_gc - 360, l_gc)
        plt.sca(ax[pltidx[j]])
        ax[pltidx[j]].tick_params(labelsize=8)
        plt.grid(alpha=.25)
        rng = np.random.default_rng(seed=1)
        
        ts=np.linspace(-5, 0, 1000)
        lmc_xs = [Model.expansion_centres(t)[6:9] for t in ts]
        lmc_vs =  [Model.expansion_centre_velocities(t)[6:9] for t in ts]
        lmc_l_gc, lmc_b_gc, *_ = galactic_coords(np.array(lmc_xs), np.array(lmc_vs))
        
        if cbar=='veldis':
            sc=plt.scatter(-wrapped_ls*u.deg.to(u.rad), b_gc*u.deg.to(u.rad), 
                           c=loc_veldis, cmap='magma_r', s=.5,rasterized=True, vmin=0,vmax=20,  zorder=1)
            if j==0:
                cb=plt.colorbar(sc, ax=[ax[i] for i in pltidx], location='right', aspect=40, pad=0.05, shrink=.5)
                cb.ax.tick_params(labelsize=14)
                cb.set_label(r'$\sigma_{v, \mathrm{loc}}\,[\mathrm{km}\,\mathrm{s}^{-1}]$')

        elif cbar=='pms':
            sc=plt.scatter(-wrapped_ls*u.deg.to(u.rad), b_gc*u.deg.to(u.rad), 
                           c=pm_ang, cmap='magma_r', s=.5,rasterized=True, vmin=0,vmax=90,  zorder=1)
            if j==0:
                cb=plt.colorbar(sc, ax=[ax[i] for i in pltidx], location='right', aspect=40, pad=0.05, shrink=.5)
                cb.ax.tick_params(labelsize=14)
                cb.set_label(r'$\bar{\vartheta}\,[^{\circ}]$')
            
        elif cbar=='deviation':
            sc=plt.scatter(-wrapped_ls*u.deg.to(u.rad), b_gc*u.deg.to(u.rad), 
                           c=track_deform, cmap='magma_r', s=.5,rasterized=True, vmin=0,vmax=10,  zorder=1)
            if j==0:
                cb=plt.colorbar(sc, ax=[ax[i] for i in pltidx], location='right', aspect=40, pad=0.05, shrink=.5)
                cb.ax.tick_params(labelsize=14)
                cb.set_label(r'$\bar{\delta}\,[^{\circ}]$')
                
        elif cbar=='widths':
            sc=plt.scatter(-wrapped_ls*u.deg.to(u.rad), b_gc*u.deg.to(u.rad), 
                           c=widths, cmap='magma_r', s=.5,rasterized=True, vmin=0,vmax=3,  zorder=1)
            if j==0:
                cb=plt.colorbar(sc, ax=[ax[i] for i in pltidx], location='right', aspect=40, pad=0.05, shrink=.5)
                cb.ax.tick_params(labelsize=14)
                cb.set_label(r'$w\,[^{\circ}]$')
        
        lmc_l_gc_wrap = np.where(lmc_l_gc >= 180, lmc_l_gc - 360, lmc_l_gc)

        plt.scatter(-lmc_l_gc_wrap[-1]*u.deg.to(u.rad), lmc_b_gc[-1]*u.deg.to(u.rad), s=100,  zorder=2,
                            edgecolors='k', facecolor='lightgrey',marker='*', label='LMC', rasterized=True)
  
        plt.scatter((-lmc_l_gc_wrap) * u.deg.to(u.rad), lmc_b_gc * u.deg.to(u.rad), rasterized=True,
                     s=5,c='lightgrey', zorder=1)

        plt.title(labels[j], loc='left', fontsize=10)
        if j ==0:
            plt.legend(frameon=False, fontsize=7, loc='upper right')
        
     # Flip lon labels
    for axs in ax.flat:
        x_labels = axs.get_xticks() * 180/np.pi
        axs.set_xticklabels(['{:.0f}'.format(-label) + r'$^{\circ}$' for label in x_labels])
        
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/mollweide/{}'.format(plotname + '-' + cbar))
    plt.close()
     
def mollweide_gc_Ebins(data_path, potential, cbar, plotname, savefig=False):
    
    labels = list([r'$\mathcal{E} \in \{-11, -9\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$', 
             r'$\mathcal{E} \in \{-9, -8\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$', 
             r'$\mathcal{E} \in \{-8, -7\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$',
             r'$\mathcal{E} \in \{-7, -6\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$', 
             r'$\mathcal{E} \in \{-6, -5\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$',
             r'$\mathcal{E} \in \{-5, -4\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$'])
   
    with h5py.File(data_path + potential,'r') as file:

        energies = np.array(file['energies'])
        l_gc = np.array(file['l_gc'])
        b_gc = np.array(file['b_gc'])
        # ds = np.array(file['ds'])
        widths = np.array(file['widths'])
        loc_veldis = np.array(file['loc_veldis'])
        track_deform = np.array(file['track_deform'])
        pm_ang = np.array(file['pm_misalignment'])
        
        t_idx = -1
        l_pole = np.array(file['pole_l'])[:,t_idx]
        b_pole = np.array(file['pole_b'])[:,t_idx]
        
    poles =  np.stack((l_pole, b_pole))
    rot_pole = np.array([rotation_matrix_disc @ poles[:,i] for i in range(len(l_pole))])
    l_pole_std, b_pole_std = np.nanstd(rot_pole[:,0],axis=1), np.nanstd(rot_pole[:,1],axis=1)
        
    E_bins = np.delete(np.linspace(-11,-4,8), 1)
    hist, bins = np.histogram(energies/1e4, E_bins)

    # Initialize a list to store indices
    indices_in_bins = []
    for i in range(len(bins) - 1):
        # Select indices of data points that fall within the current bin
        indices = np.where((energies/1e4 >= bins[i]) & (energies/1e4 < bins[i+1]))[0]
        indices_in_bins.append(indices)
        
        
    pltidx = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

    ts=np.linspace(-5, 0, 1000)
    lmc_xs = [Model.expansion_centres(t)[6:9] for t in ts]
    lmc_vs =  [Model.expansion_centre_velocities(t)[6:9] for t in ts]
    lmc_l_gc, lmc_b_gc, *_ = galactic_coords(np.array(lmc_xs), np.array(lmc_vs))
    fig, ax = plt.subplots(3, 2, subplot_kw={'projection': 'mollweide'}, figsize=(12, 12))
    plt.subplots_adjust(hspace=-0.25, wspace=0.1)

    j = 0
    for idx in indices_in_bins:

        ls_bin = l_gc[idx]
        bs_bin = b_gc[idx]

        wrapped_ls = np.where(ls_bin>=180, ls_bin - 360, ls_bin)

        plt.sca(ax[pltidx[j]])
        ax[pltidx[j]].tick_params(labelsize=8)
        plt.grid(alpha=.25)

        
        if cbar=='veldis':
            veldis_bin = loc_veldis[idx]
            sc=plt.scatter(-wrapped_ls*u.deg.to(u.rad), bs_bin*u.deg.to(u.rad), 
                           c=veldis_bin, cmap='magma_r', s=.5,rasterized=True, vmin=0,vmax=20, zorder=2)
            if j==0:
                cb=plt.colorbar(sc, ax=[ax[0,0], ax[0,1], ax[1,0], ax[1,1], ax[2,0], ax[2,1]], 
                                location='right', aspect=40, pad=0.05, shrink=.5)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(r'$\sigma_{v, \mathrm{loc}}\,[\mathrm{km}\,\mathrm{s}^{-1}]$')
            
        elif cbar=='deviation':  
            trackdeform_bin = track_deform[idx]
            sc=plt.scatter(-wrapped_ls*u.deg.to(u.rad), bs_bin*u.deg.to(u.rad), 
                           c=trackdeform_bin, cmap='magma_r', s=.5,rasterized=True, vmin=0, vmax=10, zorder=2)
            if j==0:
                cb=plt.colorbar(sc, ax=[ax[0,0], ax[0,1], ax[1,0], ax[1,1], ax[2,0], ax[2,1]], 
                                location='right', aspect=40, pad=0.05, shrink=.5)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(r'$\bar{\delta}\,[^{\circ}]$')
            
        elif cbar=='pms':    
            pm_ang_bin = pm_ang[idx]
            sc=plt.scatter(-wrapped_ls*u.deg.to(u.rad), bs_bin*u.deg.to(u.rad), 
                           c=pm_ang_bin, cmap='magma_r', s=.5,rasterized=True, vmin=0, vmax=90, zorder=2)
            if j==0:
                cb=plt.colorbar(sc, ax=[ax[0,0], ax[0,1], ax[1,0], ax[1,1], ax[2,0], ax[2,1]], 
                                location='right', aspect=40, pad=0.05, shrink=.5)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(r'$\bar{\vartheta}\,[^{\circ}]$')

        elif cbar=='widths':
            widths_bin = widths[idx]
            sc=plt.scatter(-wrapped_ls*u.deg.to(u.rad), bs_bin*u.deg.to(u.rad), 
                           c=widths_bin, cmap='magma_r', s=.5,rasterized=True, vmin=0, vmax=3, zorder=2)
            if j==0:
                cb=plt.colorbar(sc, ax=[ax[0,0], ax[0,1], ax[1,0], ax[1,1], ax[2,0], ax[2,1]], 
                                location='right', aspect=40, pad=0.05, shrink=.5)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(r'$w\,[^{\circ}]$')
            
        elif cbar=='lon_pole':
            lpolestd_bin = l_pole_std[idx]
            sc=plt.scatter(-wrapped_ls*u.deg.to(u.rad), bs_bin*u.deg.to(u.rad), 
                           c=lpolestd_bin, cmap='magma_r', s=.5,rasterized=True, norm=LogNorm(vmin=0.1, vmax=200), zorder=2)
            if j==0:
                cb=plt.colorbar(sc, ax=[ax[0,0], ax[0,1], ax[1,0], ax[1,1], ax[2,0], ax[2,1]], 
                                location='right', aspect=40, pad=0.05, shrink=.5)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(r'$\sigma_{l^{\prime},\mathrm{pole}}\,[^{\circ}]$')

        elif cbar=='lat_pole':
            bpolestd_bin = b_pole_std[idx]
            sc=plt.scatter(-wrapped_ls*u.deg.to(u.rad), bs_bin*u.deg.to(u.rad), 
                           c=bpolestd_bin, cmap='magma_r', s=.5,rasterized=True, norm=LogNorm(vmin=0.1, vmax=200), zorder=2)
            if j==0:
                cb=plt.colorbar(sc, ax=[ax[0,0], ax[0,1], ax[1,0], ax[1,1], ax[2,0], ax[2,1]], 
                                location='right', aspect=40, pad=0.05, shrink=.5)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(r'$\sigma_{b^{\prime},\mathrm{pole}}\,[^{\circ}]$')

        lmc_l_gc_wrap = np.where(lmc_l_gc >= 180, lmc_l_gc - 360, lmc_l_gc)
        plt.scatter(-lmc_l_gc_wrap[-1]*u.deg.to(u.rad), lmc_b_gc[-1]*u.deg.to(u.rad), s=100,  zorder=2,
                            edgecolors='k', facecolor='lightgrey',marker='*', label='LMC', rasterized=True)
  
        plt.scatter((-lmc_l_gc_wrap) * u.deg.to(u.rad), lmc_b_gc * u.deg.to(u.rad), rasterized=True,
                     s=5,c='lightgrey', zorder=1)
        
        ax[pltidx[j]].set_title(labels[j], fontsize=10, loc='right')
        j += 1
        
    # Flip lon labels
    for axs in ax.flat:
        x_labels = axs.get_xticks() * 180/np.pi
        axs.set_xticklabels(['{:.0f}'.format(-label) + r'$^{\circ}$' for label in x_labels])
        
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/mollweide/ebins/{}'.format(plotname + '-' + cbar))
    plt.close()
        
def smooth_mollweide_gc(path_data, pots, cbar, pot_labels, plotname, savefig=False):

    subidx = [421, 422, 423, 424, 425, 426, 427, 428]
    fig, ax = plt.subplots(4, 2, subplot_kw={'projection': 'mollweide'}, figsize=(14, 16))
    ax[0,0].set_visible(False)
    ax[0,1].set_visible(False)
    ax[1,0].set_visible(False)
    ax[1,1].set_visible(False)
    ax[2,0].set_visible(False)
    ax[2,1].set_visible(False)
    ax[3,0].set_visible(False)
    ax[3,1].set_visible(False)
    
    plt.subplots_adjust(hspace=0.2, wspace=-0.1)

    for j in range(len(pots)): 

        with h5py.File(path_data + potentials[j],'r') as file:
            
            t_idx = -1

            l_gc = np.array(file['l_gc'])
            b_gc = np.array(file['b_gc'])
            ds = np.array(file['ds'])
            loc_veldis = np.array(file['loc_veldis'])
            widths = np.array(file['widths'])
            track_deform = np.array(file['track_deform'])
            pm_ang = np.array(file['pm_misalignment'])

        wrapped_ls = np.where(l_gc>=180, l_gc - 360, l_gc)
        
        ts=np.linspace(-5, 0, 1000)
        lmc_xs = [Model.expansion_centres(t)[6:9] for t in ts]
        lmc_vs =  [Model.expansion_centre_velocities(t)[6:9] for t in ts]
        lmc_l_gc, lmc_b_gc, *_ = galactic_coords(np.array(lmc_xs), np.array(lmc_vs))
        lmc_l_gc_wrap = np.where(lmc_l_gc >= 180, lmc_l_gc - 360, lmc_l_gc)
        
        if cbar=='veldis':
            mollweide(l=wrapped_ls, b=b_gc, bmin=0, bmax=15, nside=30, smooth=7, 
                      sub=subidx[j], title=pot_labels[j], q=loc_veldis, 
                      cmap='magma_r', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                     unit=r'$\sigma_{v, \mathrm{loc}}\,[\mathrm{km}\,\mathrm{s}^{-1}]$')

        elif cbar=='pms':
            mollweide(l=wrapped_ls, b=b_gc, bmin=0, bmax=50, nside=30, smooth=7, 
                      sub=subidx[j], title=pot_labels[j], q=pm_ang, 
                      cmap='magma_r', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                     unit = r'$\bar{\vartheta}\,[^{\circ}]$')
            
        elif cbar=='deviation':
            mollweide(l=wrapped_ls, b=b_gc, bmin=0, bmax=10, nside=30, smooth=7, 
                      sub=subidx[j], title=pot_labels[j], q=track_deform, 
                      cmap='magma_r', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                     unit = r'$\bar{\delta}\,[^{\circ}]$')
                
        elif cbar=='widths':
            mollweide(l=wrapped_ls, b=b_gc, bmin=0, bmax=3, nside=30, smooth=7, 
                      sub=subidx[j], title=pot_labels[j], q=widths, 
                      cmap='magma_r', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                     unit=r'$w\,[^{\circ}]$')
        
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/mollweide/{}'.format(plotname + '-' + cbar))
    plt.close()
    
def smooth_mollweide_gc_Ebins(data_path, potential, cbar, plotname, savefig=False):
    
    labels = list([r'$\mathcal{E} \in \{-11, -9\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$', 
             r'$\mathcal{E} \in \{-9, -8\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$', 
             r'$\mathcal{E} \in \{-8, -7\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$',
             r'$\mathcal{E} \in \{-7, -6\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$', 
             r'$\mathcal{E} \in \{-6, -5\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$',
             r'$\mathcal{E} \in \{-5, -4\}\,[(\mathrm{km}\,\mathrm{s}^{-1})^2]$'])
   
    with h5py.File(data_path + potential,'r') as file:

        energies = np.array(file['energies'])
        l_gc = np.array(file['l_gc'])
        b_gc = np.array(file['b_gc'])
        # ds = np.array(file['ds'])
        widths = np.array(file['widths'])
        loc_veldis = np.array(file['loc_veldis'])
        track_deform = np.array(file['track_deform'])
        pm_ang = np.array(file['pm_misalignment'])
        
        t_idx = -1
        l_pole = np.array(file['pole_l'])[:,t_idx]
        b_pole = np.array(file['pole_b'])[:,t_idx]
        
    poles =  np.stack((l_pole, b_pole))
    rot_pole = np.array([rotation_matrix_disc @ poles[:,i] for i in range(len(l_pole))])
    l_pole_std, b_pole_std = np.nanstd(rot_pole[:,0],axis=1), np.nanstd(rot_pole[:,1],axis=1)
        
    E_bins = np.delete(np.linspace(-11,-4,8), 1)
    hist, bins = np.histogram(energies/1e4, E_bins)

    # Initialize a list to store indices
    indices_in_bins = []
    for i in range(len(bins) - 1):
        # Select indices of data points that fall within the current bin
        indices = np.where((energies/1e4 >= bins[i]) & (energies/1e4 < bins[i+1]))[0]
        indices_in_bins.append(indices)
        
        
    subidx = [421, 422, 423, 424, 425, 426, 427, 428]

    ts=np.linspace(-5, 0, 1000)
    lmc_xs = [Model.expansion_centres(t)[6:9] for t in ts]
    lmc_vs =  [Model.expansion_centre_velocities(t)[6:9] for t in ts]
    lmc_l_gc, lmc_b_gc, *_ = galactic_coords(np.array(lmc_xs), np.array(lmc_vs))
    lmc_l_gc_wrap = np.where(lmc_l_gc >= 180, lmc_l_gc - 360, lmc_l_gc)
        
    fig, ax = plt.subplots(3, 2, subplot_kw={'projection': 'mollweide'}, figsize=(14, 16))
    plt.subplots_adjust(hspace=0.2, wspace=-0.1)
    
    ax[0,0].set_visible(False)
    ax[0,1].set_visible(False)
    ax[1,0].set_visible(False)
    ax[1,1].set_visible(False)
    ax[2,0].set_visible(False)
    ax[2,1].set_visible(False)

    j = 0
    for idx in indices_in_bins:

        ls_bin = l_gc[idx]
        bs_bin = b_gc[idx]

        wrapped_ls = np.where(ls_bin>=180, ls_bin - 360, ls_bin)

        if cbar=='veldis':
            veldis_bin = loc_veldis[idx]
            mollweide(l=wrapped_ls, b=bs_bin, bmin=0, bmax=15, nside=30, smooth=7, 
                      sub=subidx[j], title=labels[j], q=veldis_bin, 
                      cmap='magma_r', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                     unit=r'$\sigma_{v, \mathrm{loc}}\,[\mathrm{km}\,\mathrm{s}^{-1}]$')
            
        elif cbar=='deviation':  
            trackdeform_bin = track_deform[idx]
            mollweide(l=wrapped_ls, b=bs_bin, bmin=0, bmax=10, nside=30, smooth=7, 
                      sub=subidx[j], title=labels[j], q=trackdeform_bin, 
                      cmap='magma_r', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                     unit=r'$\bar{\delta}\,[^{\circ}]$')
            
        elif cbar=='pms':    
            pm_ang_bin = pm_ang[idx]
            mollweide(l=wrapped_ls, b=bs_bin, bmin=0, bmax=50, nside=30, smooth=7, 
                      sub=subidx[j], title=labels[j], q=pm_ang_bin, 
                      cmap='magma_r', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                     unit= r'$\bar{\vartheta}\,[^{\circ}]$')

        elif cbar=='widths':
            widths_bin = widths[idx]
            mollweide(l=wrapped_ls, b=bs_bin, bmin=0, bmax=3, nside=30, smooth=7, 
                      sub=subidx[j], title=labels[j], q=widths_bin, 
                      cmap='magma_r', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                     unit=r'$w\,[^{\circ}]$')
            
        elif cbar=='lon_pole':
            lpolestd_bin = l_pole_std[idx]
            mollweide(l=wrapped_ls, b=bs_bin, bmin=0, bmax=100, nside=30, smooth=7, 
                          sub=subidx[j], title=labels[j], q=lpolestd_bin, 
                          cmap='magma_r', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                     unit=r'$\sigma_{l^{\prime},\mathrm{pole}}\,[^{\circ}]$')

        elif cbar=='lat_pole':
            bpolestd_bin = b_pole_std[idx]
            mollweide(l=wrapped_ls, b=bs_bin, bmin=0, bmax=100, nside=30, smooth=7, 
                          sub=subidx[j], title=labels[j], q=bpolestd_bin, 
                          cmap='magma_r', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                     unit=r'$\sigma_{b^{\prime},\mathrm{pole}}\,[^{\circ}]$')
        
        j += 1
        
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/mollweide/ebins/{}'.format(plotname + '-' + cbar))
    plt.close()
    
def mollweide_pot_enhancment(cbar, savefig=False):
    
    if cbar=='density':
        fields_slice = 3
        unit = r"$\delta \rho$"
        plotname='dens-enhancement'
        bmin, bmax= -.5, .5
    if cbar=='potential':
        fields_slice = 4
        unit = r"$\delta \Phi$"
        plotname='pot-enhancement'
        bmin, bmax= -.1, .1
    
    subidx = [421, 422, 423, 424, 425, 426]
    
    d_ranges = [(10,20), (20,40), (40,60), (60,80), (80,100), (100,150)]
    labels = list([r'$d \in \{10, 20\}\,[\mathrm{kpc}]$', 
                   r'$d \in \{20, 40\}\,[\mathrm{kpc}]$',
                  r'$d \in \{40, 60\}\,[\mathrm{kpc}]$',
                  r'$d \in \{60, 80\}\,[\mathrm{kpc}]$',
                  r'$d \in \{80, 100\}\,[\mathrm{kpc}]$',
                  r'$d \in \{100, 150\}\,[\mathrm{kpc}]$'])

    ts=np.linspace(-5, 0, 1000)
    lmc_xs = [Model.expansion_centres(t)[6:9] for t in ts]
    lmc_vs =  [Model.expansion_centre_velocities(t)[6:9] for t in ts]
    lmc_l_gc, lmc_b_gc, *_ = galactic_coords(np.array(lmc_xs), np.array(lmc_vs))
    lmc_l_gc_wrap = np.where(lmc_l_gc >= 180, lmc_l_gc - 360, lmc_l_gc)
    
    ls = np.linspace(-180, 180, 100)
    bs = np.linspace(-90, 90, 100)
    Lsm, Bsm = np.meshgrid(ls, bs)
    L, B =  Lsm.ravel(), Bsm.ravel()
    
    fig, ax = plt.subplots(3, 2, subplot_kw={'projection': 'mollweide'}, figsize=(14, 16))
    plt.subplots_adjust(hspace=0.2, wspace=-0.1)
    
    ax[0,0].set_visible(False)
    ax[0,1].set_visible(False)
    ax[1,0].set_visible(False)
    ax[1,1].set_visible(False)
    ax[2,0].set_visible(False)
    ax[2,1].set_visible(False)
    
    for j in range(len(subidx)):
        
        dens0_arr = np.zeros(len(L))
        dens_mwhalo_arr = np.zeros(len(L))
        
        d_range = d_ranges[j]
        for d in range(d_range[0], d_range[1], 2):
        
            ds = np.ones(len(L)) * d

            lb_galactic = SkyCoord(l=L*u.deg, b=B*u.deg, distance=ds*u.kpc, frame='galactic')
            pos_gc = lb_galactic.transform_to('galactocentric')
            dens0 = Model.mwhalo_fields(ts[-1], pos_gc.x.value, pos_gc.y.value, pos_gc.z.value, mwhharmonicflag=0)[:,fields_slice]
            dens_mwhalo = Model.mwhalo_fields(ts[-1], pos_gc.x.value, pos_gc.y.value, pos_gc.z.value, mwhharmonicflag=63)[:,fields_slice]

            dens0_arr =+ dens0
            dens_mwhalo_arr =+ dens_mwhalo
        
        delta_rho = dens_mwhalo_arr/dens0_arr - 1
        
        mollweide(l=L, b=B,  bmin=bmin, bmax=bmax,  nside=20, smooth=10, 
                  q=delta_rho, cmap='coolwarm', l2=lmc_l_gc_wrap , b2=lmc_b_gc,
                  sub=subidx[j], title=labels[j], unit=unit)
        
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/mollweide/{}'.format(plotname))
    plt.close()
###---------------------------------------------------------------------------------------------------------------- 
# Run the script and plot scripts 
###---------------------------------------------------------------------------------------------------------------- 
print("Script is running...")

data_path = '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/plotting_data/16384-dt1Myr/'
    
### All potentials, all energies
print("Plotting mollewide_gc...")

potentials = list(['rigid-mw.hdf5','rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5',\
                       'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5',\
                        'full-MWhalo-full-MWdisc-no-LMC.hdf5', 'full-MWhalo-full-MWdisc-full-LMC.hdf5'])
labels = list(['Rigid MW without motion','Rigid Monopole \& LMC', 'Evolving Monopole \& LMC', 'Monopole + Dipole \& LMC', 'Monopole + Quadrupole \& LMC', \
               'Monopole + Dipole + Quadrupole \& LMC',  'Full Expansion (no LMC)', 'Full Expansion \& LMC'])

# mollewide_gc(data_path, potentials, 'veldis', labels, 'mollweide-gc', True)
# mollewide_gc(data_path, potentials, 'pms', labels, 'mollweide-gc', True)
# mollewide_gc(data_path, potentials, 'deviation', labels, 'mollweide-gc', True)
# mollewide_gc(data_path, potentials, 'widths', labels, 'mollweide-gc', True)

# smooth_mollweide_gc(data_path, potentials, 'veldis', labels, 'smooth-mollweide-gc', True)
# smooth_mollweide_gc(data_path, potentials, 'pms', labels, 'smooth-mollweide-gc', True)
# smooth_mollweide_gc(data_path, potentials, 'deviation', labels, 'smooth-mollweide-gc', True)
# smooth_mollweide_gc(data_path, potentials, 'widths', labels, 'smooth-mollweide-gc', True)

### All potentials, all energies
print("Plotting mollewide_gc_Ebins...")
pot = 'full-MWhalo-full-MWdisc-full-LMC.hdf5'
# pot = 'full-MWhalo-full-MWdisc-no-LMC.hdf5'

# mollweide_gc_Ebins(data_path, pot, 'veldis','gc-ebins-fullexp', True)
# mollweide_gc_Ebins(data_path, pot, 'pms','gc-ebins-fullexp', True)
# mollweide_gc_Ebins(data_path, pot, 'deviation','gc-ebins-fullexp', True)
# mollweide_gc_Ebins(data_path, pot, 'widths','gc-ebins-fullexp', True)
# mollweide_gc_Ebins(data_path, pot, 'lon_pole','gc-ebins-fullexp', True)
# mollweide_gc_Ebins(data_path, pot, 'lat_pole','gc-ebins-fullexp', True)

# smooth_mollweide_gc_Ebins(data_path, pot, 'veldis', 'smooth-gc-ebins-fullexp', True)
# smooth_mollweide_gc_Ebins(data_path, pot, 'pms', 'smooth-gc-ebins-fullexp', True)
# smooth_mollweide_gc_Ebins(data_path, pot, 'deviation', 'smooth-gc-ebins-fullexp', True)
# smooth_mollweide_gc_Ebins(data_path, pot, 'widths', 'smooth-gc-ebins-fullexp', True)
# smooth_mollweide_gc_Ebins(data_path, pot, 'lon_pole','smooth-gc-ebins-fullexp', True)
# smooth_mollweide_gc_Ebins(data_path, pot, 'lat_pole','smooth-gc-ebins-fullexp', True)
 
                      
### Plotting potential constrast
print("Plotting mollweide_pot_enhancment...")
mollweide_pot_enhancment('density', True)
mollweide_pot_enhancment('potential', True)
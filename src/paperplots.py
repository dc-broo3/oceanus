print("Loading modules and packages...")

from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import seaborn as sns
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
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
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

def rubinlsst_mask(path, potl):

    rubin_radec = np.load('/mnt/ceph/users/rbrooks/oceanus/analysis/RUBIN_ra_dec_footprint.npy')
    # Create an approximate outline around the edges of Rubin LSST footprint
    hull = ConvexHull(rubin_radec)
    rubin_outline_radec = rubin_radec[hull.vertices]
    
    outline_path = Path(rubin_outline_radec)
    # outline_patch = patches.PathPatch(outline_path, facecolor='none', edgecolor='b', lw=3)

    with h5py.File(path + potl, 'r') as file:
        energies = np.array(file['energies'])
        pm_ang = np.array(file['pm_misalignment'])
        l_gc = np.array(file['l_gc'])
        b_gc = np.array(file['b_gc'])

    lb_streams = SkyCoord(l=l_gc*u.deg,
    b=b_gc*u.deg, frame='galactic')

    radec_streams = lb_streams.transform_to('icrs')
    radec_streams = np.vstack([radec_streams.ra.value, radec_streams.dec.value]).T

    lsst_mask = outline_path.contains_points(radec_streams)
    return lsst_mask

disc_xs, disc_vs = Model.expansion_centres(0.)[:3], Model.expansion_centre_velocities(0.)[:3]
disc_lpole, disc_bpole = orbpole(np.array(disc_xs), np.array(disc_vs))
v1_disc = np.array([disc_lpole, disc_bpole])
v2 = np.array([0, 90])
rotation_matrix_disc = rotation_matrix_from_vectors(v1_disc, v2)

def plot_metric_QUAD(ax, indices_in_bins, metric, threshold, y_label, title, colors, labels, bin_mids, mask_idx):
    ax.tick_params(axis='x',which='both', top=False)
    
    frac = []
    uncert = []
    for idx in indices_in_bins:
        metric_bin = metric[idx]
        if len(metric_bin) == 0:
            frac_high = 0
            uncert_high = 0
        else:
            above = len(metric_bin[metric_bin > threshold])
            if above == 0:
                above = 1  # to avoid division by 0 below
            total = len(metric_bin)
            frac_high = above / total
            uncert_high = frac_high * ((1 / above) + (1 / total))**0.5
        frac.append(frac_high)
        uncert.append(uncert_high)

    if mask_idx == 3: #mask_idx == 2 
        ax.plot(bin_mids, frac, lw=3, label=labels[mask_idx], c=colors[mask_idx])
        ax.fill_between(bin_mids, np.array(frac) - np.array(uncert), np.array(frac) + np.array(uncert),
                        alpha=0.3, ec='None', fc=colors[mask_idx])
    else:
        ax.plot(bin_mids, frac, lw=1, label=labels[mask_idx], c=colors[mask_idx])
        ax.fill_between(bin_mids, np.array(frac) - np.array(uncert), np.array(frac) + np.array(uncert),
                        fc=colors[mask_idx], alpha=0.2, ec='None',)

    ax.set_xlabel(r'$E\,[10^{4}\,(\mathrm{km}\,\mathrm{s}^{-1})^2]$', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlim(-9.99, -4.21)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=14)
    if mask_idx==0:
        secax = ax.secondary_xaxis('top', functions=(interp_E_to_r, interp_r_to_E))
        secax.set_xlabel('Galactocentric radius [kpc]', color='grey', fontsize=12) 
        secax.tick_params(axis='x',which='both', colors='grey')

t_init=-5 #Gyr
rs = np.linspace(1, 500, 500)
rs_zeros = np.zeros(shape=(500,))
mwh_sph_pot = Model.mwhalo_fields(t_init, rs, rs_zeros, rs_zeros, 0)[:,4]/1e4 #divided by 10^4
interp_E_to_r = interp1d(mwh_sph_pot, rs, kind='cubic', fill_value="extrapolate")
interp_r_to_E = interp1d(rs, mwh_sph_pot, kind='cubic', fill_value="extrapolate")

###--------------------------------------------------------------------------------------------
### DES data
###--------------------------------------------------------------------------------------------

DES_df = pd.read_csv('/mnt/ceph/users/rbrooks/oceanus/analysis/DES-pm-misaligns.csv')

frac_DES_pms = len(DES_df[DES_df['pm_angle'] > 10]) / len(DES_df)
mean_distance = np.mean(DES_df['distance_kpc'])
std_distance = np.std(DES_df['distance_kpc'])
gc_l = DES_df['gc_l']
gc_b = DES_df['gc_b']

DES_plot_data = {"frac" : frac_DES_pms,
                "med_d" : mean_distance,
                "std_d" : std_distance,
                "gc_l": gc_l,
                "gc_b": gc_b}
        
###--------------------------------------------------------------------------------------------
### Plotting functions.
###--------------------------------------------------------------------------------------------

def fig1(streams, path, potentials, labels, figsize, plotname, savefig=False):
              
    t_idx = -1
    
    fig, ax = plt.subplots(len(streams), len(potentials), sharex='col', sharey='row', figsize=figsize)
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
            lon = stream_sph.lon.wrap_at(180*u.deg).degree[:-2]
            lat = stream_sph.lat.degree[:-2]
            #-------------------------------------------------------------------------------------
            ### Plot the streams
            #-------------------------------------------------------------------------------------
            plt.sca(ax[i,j])
            # print('* Plotting {} in potential {}'.format(streams[i], potentials[j]))
            plt.hlines(0, -200, 200, ls='dashed', color='lightgrey', lw=0.7, zorder=1)
            plt.vlines(0, -200, 200, ls='dashed', color='lightgrey', lw=0.7, zorder=1)
            plot=plt.scatter(lon[:-2], lat[:-2], s=.5, c=start_times, cmap = 'viridis',rasterized=True, zorder=2)
            
            if j==0:
                name, ext = os.path.splitext(streams[i])
                plt.annotate(text='{}'.format(name), xy=(-170,19), fontsize=10 )
                plt.annotate(text=r'M = {} $\times \, 10^{{4}} \, \mathrm{{M}}_{{\odot}}$'.format(np.round(prog_mass.value/1e4, 1)),
                             xy=(-170, -25), fontsize=10)
            
    cb = fig.colorbar(plot,  ax=ax, location='right', aspect=30, pad=0.01)
    cb.set_label('Stripping time [Gyr]')
    cb.ax.tick_params(labelsize=12)
    
    #-------------------------------------------------------------------------------------
    ### Plot cosmetics
    #-------------------------------------------------------------------------------------
    for k in range(len(labels)):

        ax[0,k].set_title(labels[k])
        ax[len(streams)-1,k].set_xlabel(r'$\mathrm{lon}\,[^{\circ}]$')
        ax[len(streams)-1,k].set_xlim(-189,189)
        
    for l in range(len(streams)):
        ax[l, 0].set_ylabel(r'$\mathrm{lat}\,[^{\circ}]$')
        ax[l, 0].set_ylim(-29,29)

    if savefig==False:
        return
    elif savefig==True:
        return plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/paper-figs/{}'.format(plotname))
    plt.close()

def fig3_cdf(path, plotname, savefig=False):
    fig, ax = plt.subplots(3,3, figsize=(12.5,8.75))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    potentials = list(['rigid-mw.hdf5','static-mw.hdf5', 'rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5',\
                           'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5',\
                            'full-MWhalo-full-MWdisc-no-LMC.hdf5', 'full-MWhalo-full-MWdisc-full-LMC.hdf5'])

    labels = list(['Rigid MW without motion (no LMC)', 'Rigid MW + motion (no LMC)', 'Rigid Monopole \& LMC', 'Evolving Monopole \& LMC', \
                   'Monopole + Dipole \& LMC', 'Monopole + Quadrupole \& LMC',\
                   'Monopole + Dipole + Quadrupole \& LMC', 'Full Expansion (no LMC)', 'Full Expansion \& LMC'])

    for j in range(len(potentials)): 

        with h5py.File(path + potentials[j],'r') as file:
            lengths = np.array(file['lengths'])
            widths = np.array(file['widths'])
            loc_veldis = np.array(file['loc_veldis'])
            track_deform = np.array(file['track_deform'])
            pm_ang = np.array(file['pm_misalignment'])
            
            lons = np.array(file['lons'])
            l_lead = np.nanpercentile(lons, 95, axis=1)
            l_trail = np.nanpercentile(lons, 5, axis=1)
            asymmetry = np.abs(l_lead/l_trail)
            
            t_idx = -1
            l_pole = np.array(file['pole_l'])[:,t_idx]
            b_pole = np.array(file['pole_b'])[:,t_idx]
        
            poles =  np.stack((l_pole, b_pole))
        rot_pole = np.array([rotation_matrix_disc @ poles[:,i] for i in range(len(l_pole))])
        
        rot_bpole_i = np.where(rot_pole[:,1] > 90, rot_pole[:,1] - 180, rot_pole[:,1])
        rot_bpole_wrapped = np.where(rot_bpole_i < -90, rot_bpole_i + 180, rot_bpole_i)
        cos_bpole = np.cos(rot_bpole_wrapped * np.pi/180)
        cos_bpole_prog = cos_bpole[:,0]
        
        l_pole_std, b_pole_std = np.nanstd(rot_pole[:,0],axis=1) * cos_bpole_prog, np.nanstd(rot_pole[:,1],axis=1)

        Nstreams= len(lengths)
        # lengths
        plt.sca(ax[0,0])
        h, bin_edges = np.histogram(lengths, bins=np.logspace(-1, np.log10(360), 200))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        if j==8:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2.5, color='k', label=labels[j], zorder=1)
        elif j==7:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2, ls='dashed', color='k', label=labels[j])
        else:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=1, label=labels[j], zorder=2)
        plt.xlabel(r'$l_{\mathrm{stream}}\,[^{\circ}]$')
        plt.ylabel('CDF')
        plt.xlim(1.1, 360)
        plt.ylim(0, 1)
        plt.xscale('log')
        plt.title('Length')

        # asymmetry
        plt.sca(ax[0,1])
        h, bin_edges = np.histogram(asymmetry, bins=np.logspace(-1, 1, 300))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        if j==8:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2.5, color='k', label=labels[j], zorder=1)
        elif j==7:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2, ls='dashed', color='k', label=labels[j])
        else:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=1, label=labels[j], zorder=2)
        plt.xlabel(r'$l_{\mathrm{leading}}/l_{\mathrm{trailing}}$')
        plt.ylabel('CDF')
        plt.xlim(0.09, 11)
        plt.xscale('log')
        plt.ylim(0, 1)
        plt.title('Asymmetry')

        # widths
        plt.sca(ax[0,2])
        h, bin_edges = np.histogram(widths, bins=np.logspace(np.log10(0.05), np.log10(3), 100))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        if j==8:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2.5, color='k', label=labels[j], zorder=1)
        elif j==7:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2, ls='dashed', color='k', label=labels[j])
        else:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=1, label=labels[j], zorder=2)
        plt.vlines(0.5, 0, 1, color='lightgrey', ls='solid', lw=.75, zorder=0.5)
        
        plt.xlabel(r'$w\,[^{\circ}]$')
        plt.ylabel('CDF')
        plt.xlim(5e-2,3)
        plt.ylim(0,1)
        plt.xscale('log')
        plt.title('Width')
        
        # track deviation
        plt.sca(ax[1,0])
        h, bin_edges = np.histogram(track_deform, bins=np.logspace(-2, 1, 100))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        if j==8:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2.5, color='k', label=labels[j], zorder=1)
        elif j==7:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2, ls='dashed', color='k', label=labels[j])
        else:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=1, label=labels[j], zorder=2)
        plt.vlines(2, 0, 1, color='lightgrey', ls='solid', lw=.75, zorder=0.5)
        
        plt.xlabel(r'$\bar{\delta}\,[^{\circ}]$')
        plt.ylabel('CDF')
        plt.xlim(5e-2,10)
        plt.ylim(0,1)
        plt.xscale('log')
        plt.title('Deviation from Great Circle')
    
        # velocity dispersion
        plt.sca(ax[1,1])
        h, bin_edges = np.histogram(loc_veldis, bins=np.logspace(-1, np.log10(20), 100))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        if j==8:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2.5, color='k', label=labels[j], zorder=1)
        elif j==7:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2, ls='dashed', color='k', label=labels[j])
        else:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=1, label=labels[j], zorder=2)
        # plt.vlines(2.5, 0, 16394, color='lightgrey', ls='solid', lw=.75, zorder=0.5)
        
        plt.xlabel(r'$\sigma_{v}\,[\mathrm{km}\,\mathrm{s}^{-1}]$')
        plt.ylabel('CDF')
        plt.xlim(2e-1,20)
        plt.ylim(0,1)
        plt.xscale('log')
        plt.title('Local velocity dispersion')
        
        # pm angle
        plt.sca(ax[1,2])
        h, bin_edges = np.histogram(pm_ang, bins=np.logspace(np.log10(0.5), np.log10(90), 500))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        if j==8:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2.5, color='k', label=labels[j], zorder=1)
        elif j==7:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2, ls='dashed', color='k', label=labels[j])
        else:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=1, label=labels[j], zorder=2)

        plt.vlines(10, 0, 1, color='lightgrey', ls='solid', lw=.75, zorder=0.5)

        plt.xlabel(r'$\bar{\vartheta} \,[^{\circ}]$')
        plt.ylabel('CDF')
        plt.xlim(0.8,90)
        plt.xscale('log')
        plt.ylim(0, 1)
        plt.title('Proper motion misalignment')
        
        # median l pole spread
        plt.sca(ax[2,0])
        h, bin_edges = np.histogram(l_pole_std, bins=np.logspace(-2, 2, 100))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        if j==8:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2.5, color='k', label=labels[j], zorder=1)
        elif j==7:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2, ls='dashed', color='k', label=labels[j])
        else:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=1, label=labels[j], zorder=2)
        plt.vlines(2, 0, 16394, color='lightgrey', ls='solid', lw=.75, zorder=0.5)
        
        # plt.xlabel(r'$\sigma_{l^{\prime},\,{\mathrm{pole}}}[^{\circ}]$')
        plt.xlabel(r'$\sigma_{l^{\prime}\,{\mathrm{pole}}} \cos(b^{\prime}_{\mathrm{pole}})\,[^{\circ}]$')
        plt.ylabel('CDF')
        plt.xlim(0.05,150)
        plt.ylim(0,1)
        plt.xscale('log')
        plt.title('Longitudinal pole dispersion')
        
        # median b pole spread
        plt.sca(ax[2,1])
        h, bin_edges = np.histogram(b_pole_std, bins=np.logspace(-2, 2, 100))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) /2
        if j==8:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2.5, color='k', label=labels[j], zorder=1)
        elif j==7:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=2, ls='dashed', color='k', label=labels[j])
        else:
            plt.plot(bin_mids, np.cumsum(h)/Nstreams, lw=1, label=labels[j], zorder=2)
        plt.vlines(2, 0, 16394, color='lightgrey', ls='solid', lw=.75, zorder=0.5)
        
        plt.xlabel(r'$\sigma_{b^{\prime},\,{\mathrm{pole}}}[^{\circ}]$')
        plt.ylabel('CDF')
        plt.xlim(0.05,150)
        plt.ylim(0,1)
        plt.xscale('log')
        plt.title('Latitudinal pole dispersion')
        plt.legend(frameon=False, ncol=1, fontsize=12, bbox_to_anchor=(1.1,1.15))
        
        ax[2,2].set_visible(False)
        
    if savefig==False:
        return
    elif savefig==True:
        savepath = '/mnt/ceph/users/rbrooks/oceanus/analysis/figures/paper-figs/{}'.format(plotname)
        print('* Saving figure at {}.pdf'.format(savepath))
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def fig3_pdf(path, plotname, savefig=False):
    fig, ax = plt.subplots(3,3, figsize=(12.5,8.75))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    potentials = list(['rigid-mw.hdf5','static-mw.hdf5', 'rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5',\
                           'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5',\
                            'full-MWhalo-full-MWdisc-no-LMC.hdf5', 'full-MWhalo-full-MWdisc-full-LMC.hdf5'])

    labels = list(['Rigid MW without motion (no LMC)', 'Rigid MW + motion (no LMC)', 'Rigid Monopole \& LMC', 'Evolving Monopole \& LMC', \
                   'Monopole + Dipole \& LMC', 'Monopole + Quadrupole \& LMC',\
                   'Monopole + Dipole + Quadrupole \& LMC', 'Full Expansion (no LMC)', 'Full Expansion \& LMC'])

    for j in range(len(potentials)): 

        with h5py.File(path + potentials[j],'r') as file:
            lengths = np.array(file['lengths'])
            widths = np.array(file['widths'])
            loc_veldis = np.array(file['loc_veldis'])
            track_deform = np.array(file['track_deform'])
            pm_ang = np.array(file['pm_misalignment'])
            
            lons = np.array(file['lons'])
            l_lead = np.nanpercentile(lons, 95, axis=1)
            l_trail = np.nanpercentile(lons, 5, axis=1)
            asymmetry = np.abs(l_lead/l_trail)
            
            t_idx = -1
            l_pole = np.array(file['pole_l'])[:,t_idx]
            b_pole = np.array(file['pole_b'])[:,t_idx]
        
            poles =  np.stack((l_pole, b_pole))
        rot_pole = np.array([rotation_matrix_disc @ poles[:,i] for i in range(len(l_pole))])
        
        rot_bpole_i = np.where(rot_pole[:,1] > 90, rot_pole[:,1] - 180, rot_pole[:,1])
        rot_bpole_wrapped = np.where(rot_bpole_i < -90, rot_bpole_i + 180, rot_bpole_i)
        cos_bpole = np.cos(rot_bpole_wrapped * np.pi/180)
        cos_bpole_prog = cos_bpole[:,0]
        
        l_pole_std, b_pole_std = np.nanstd(rot_pole[:,0],axis=1) * cos_bpole_prog, np.nanstd(rot_pole[:,1],axis=1)

        # lengths
        plt.sca(ax[0,0])
        if j==8:
            kde=sns.kdeplot(data=lengths, bw_adjust=1, log_scale=True, lw=2.5, color='k', label=labels[j], zorder=1 )
 
        elif j==7:
            kde=sns.kdeplot(data=lengths, bw_adjust=1, log_scale=True, lw=2, ls='dashed', color='k', label=labels[j])
 
        else:
            kde=sns.kdeplot(data=lengths, bw_adjust=1, log_scale=True,  lw=1, label=labels[j], zorder=2)
   
        plt.xlabel(r'$l_{\mathrm{stream}}\,[^{\circ}]$')
        plt.ylabel('PDF')
        plt.xlim(1.1, 360)
        plt.ylim(0,.99)
        plt.xscale('log')
        plt.title('Length')

        # asymmetry
        plt.sca(ax[0,1])
        if j==8:
            kde=sns.kdeplot(data=asymmetry, bw_adjust=1, log_scale=True,lw=2.5, color='k', label=labels[j], zorder=1 )
 
        elif j==7:
            kde=sns.kdeplot(data=asymmetry, bw_adjust=1, log_scale=True,lw=2, ls='dashed', color='k', label=labels[j])
 
        else:
            kde=sns.kdeplot(data=asymmetry, bw_adjust=1, log_scale=True,lw=1, label=labels[j], zorder=2)
 
        plt.xlabel(r'$l_{\mathrm{leading}}/l_{\mathrm{trailing}}$')
        plt.ylabel('PDF')
        plt.xlim(0.09, 11)
        plt.xscale('log')
        plt.ylim(0, 2.69)
        plt.title('Asymmetry')

        #widths
        plt.sca(ax[0,2])
        if j==8:
            kde=sns.kdeplot(data=widths, bw_adjust=1,log_scale=True,lw=2.5, color='k', label=labels[j], zorder=1 )
 
        elif j==7:
            kde=sns.kdeplot(data=widths, bw_adjust=1, log_scale=True,lw=2, ls='dashed', color='k', label=labels[j])
 
        else:
            kde=sns.kdeplot(data=widths, bw_adjust=1, log_scale=True,lw=1, label=labels[j], zorder=2)
 
        plt.vlines(0.5, 0, 4, color='lightgrey', ls='solid', lw=.75, zorder=0.5, clip_on=True)
        
        plt.xlabel(r'$w\,[^{\circ}]$')
        plt.ylabel('PDF')
        plt.xlim(5e-2,3)
        plt.ylim(0, 2.49)
        plt.xscale('log')
        plt.title('Width')

        # deviation from Great Circle
        plt.sca(ax[1,0])
        if j==8:
            kde=sns.kdeplot(data=track_deform, bw_adjust=1, log_scale=True,lw=2.5, color='k', label=labels[j], zorder=1 )
 
        elif j==7:
            kde=sns.kdeplot(data=track_deform, bw_adjust=1, log_scale=True,lw=2, ls='dashed', color='k', label=labels[j])
 
        else:
            kde=sns.kdeplot(data=track_deform, bw_adjust=1, log_scale=True,lw=1, label=labels[j], zorder=2)
 
        plt.vlines(2, 0, 4, color='lightgrey', ls='solid', lw=.75, zorder=0.5)
        
        plt.xlabel(r'$\bar{\delta}\,[^{\circ}]$')
        plt.ylabel('PDF')
        plt.xlim(5e-2,10)
        plt.ylim(0,1.29)
        plt.xscale('log')
        plt.title('Deviation from Great Circle')

        # velocity dispersion
        plt.sca(ax[1,1])
        loc_veldis[loc_veldis == 0] = 0.0001
        if j==8:
            kde=sns.kdeplot(data=loc_veldis, bw_adjust=1, log_scale=True,lw=2.5, color='k', label=labels[j], zorder=1 )
 
        elif j==7:
            kde=sns.kdeplot(data=loc_veldis, bw_adjust=1, log_scale=True,lw=2, ls='dashed', color='k', label=labels[j])
 
        else:
            kde=sns.kdeplot(data=loc_veldis, bw_adjust=1, log_scale=True,lw=1, label=labels[j], zorder=2)
 
        # plt.vlines(2.5, 0, 4, color='lightgrey', ls='solid', lw=.75, zorder=0.5)
        
        plt.xlabel(r'$\sigma_{v}\,[\mathrm{km}\,\mathrm{s}^{-1}]$')
        plt.ylabel('PDF')
        plt.xlim(2e-1,20)
        plt.ylim(0,2.29)
        plt.xscale('log')
        plt.title('Local velocity dispersion')
    
        # pm angle
        plt.sca(ax[1,2])
        if j==8:
            kde=sns.kdeplot(data=pm_ang, bw_adjust=1, log_scale=True,lw=2.5, color='k', label=labels[j], zorder=1 )
 
        elif j==7:
            kde=sns.kdeplot(data=pm_ang, bw_adjust=1, log_scale=True,lw=2, ls='dashed', color='k', label=labels[j])
 
        else:
            kde=sns.kdeplot(data=pm_ang, bw_adjust=1, log_scale=True,lw=1, label=labels[j], zorder=2)
 
        plt.vlines(10, 0, 4, color='lightgrey', ls='solid', lw=.75, zorder=0.5)

        plt.xlabel(r'$\bar{\vartheta} \,[^{\circ}]$')
        plt.ylabel('PDF')
        plt.xlim(0.8,90)
        plt.xscale('log')
        plt.ylim(0,1.29)
        plt.title('Proper motion misalignment')
        
        # median l pole spread
        plt.sca(ax[2,0])
        if j==8:
            kde=sns.kdeplot(data=l_pole_std, bw_adjust=1,log_scale=True,lw=2.5, color='k', label=labels[j], zorder=1 )

        elif j==7:
            kde=sns.kdeplot(data=l_pole_std, bw_adjust=1,log_scale=True, lw=2, ls='dashed', color='k', label=labels[j])

        else:
            kde=sns.kdeplot(data=l_pole_std, bw_adjust=1,log_scale=True, lw=1, label=labels[j], zorder=2)
        plt.vlines(2, 0, 4, color='lightgrey', ls='solid', lw=.75, zorder=0.5)
        
        plt.xlabel(r'$\sigma_{l^{\prime}\,{\mathrm{pole}}} \cos(b^{\prime}_{\mathrm{pole}})\,[^{\circ}]$')
        plt.ylabel('PDF')
        plt.xlim(0.05,150)
        plt.ylim(0, 1.79)
        plt.xscale('log')
        plt.title('Longitudinal pole dispersion')
        
        # median b pole spread
        plt.sca(ax[2,1])
        if j==8:
            kde=sns.kdeplot(data=b_pole_std, bw_adjust=1,log_scale=True, lw=2.5, color='k', label=labels[j], zorder=1 )

        elif j==7:
            kde=sns.kdeplot(data=b_pole_std, bw_adjust=1,log_scale=True,lw=2, ls='dashed', color='k', label=labels[j])

        else:
            kde=sns.kdeplot(data=b_pole_std, bw_adjust=1, log_scale=True,lw=1, label=labels[j], zorder=2)
        plt.vlines(2, 0, 4, color='lightgrey', ls='solid', lw=.75, zorder=0.5)
        
        plt.xlabel(r'$\sigma_{b^{\prime},\,{\mathrm{pole}}}[^{\circ}]$')
        plt.ylabel('PDF')
        plt.xlim(0.05,150)
        plt.ylim(0,2.49)
        plt.xscale('log')
        plt.title('Latitudinal pole dispersion')
        plt.legend(frameon=False, ncol=1, fontsize=12, bbox_to_anchor=(1.1,1.15))
        
        ax[2,2].set_visible(False)
        
    if savefig==False:
        return
    elif savefig==True:
        savepath = '/mnt/ceph/users/rbrooks/oceanus/analysis/figures/paper-figs/{}'.format(plotname)
        print('* Saving figure at {}.pdf'.format(savepath))
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()
       
def fig4(path_data, pots, pot_labels, plotname, savefig=False):
    fig, axs = plt.subplots(3, 2, figsize=(10, 11))
    plt.subplots_adjust(hspace=0.65, wspace=0.3)

    for j in range(len(pots)):
        with h5py.File(path_data + pots[j], 'r') as file:
            energies = np.array(file['energies'])
            loc_veldis = np.array(file['loc_veldis'])
            widths = np.array(file['widths'])
            track_deform = np.array(file['track_deform'])
            pm_ang = np.array(file['pm_misalignment'])

            t_idx = -1
            l_pole = np.array(file['pole_l'])[:,t_idx]
            b_pole = np.array(file['pole_b'])[:,t_idx]
        
            poles =  np.stack((l_pole, b_pole))
        rot_pole = np.array([rotation_matrix_disc @ poles[:,i] for i in range(len(l_pole))])
        
        rot_bpole_i = np.where(rot_pole[:,1] > 90, rot_pole[:,1] - 180, rot_pole[:,1])
        rot_bpole_wrapped = np.where(rot_bpole_i < -90, rot_bpole_i + 180, rot_bpole_i)
        cos_bpole = np.cos(rot_bpole_wrapped * np.pi/180)
        cos_bpole_prog = cos_bpole[:,0]
        
        l_pole_std, b_pole_std = np.nanstd(rot_pole[:,0],axis=1) * cos_bpole_prog, np.nanstd(rot_pole[:,1],axis=1)
        # l_pole_std, b_pole_std = np.nanstd(rot_pole[:,0],axis=1), np.nanstd(rot_pole[:,1],axis=1)

        E_bins = np.linspace(-11, -3, 15)
        hist, bins = np.histogram(energies / 1e4, E_bins)

        bin_mids = [(E_bins[i] + E_bins[i + 1]) / 2 for i in range(len(E_bins) - 1)]
        indices_in_bins = [np.where((energies / 1e4 >= bins[i]) & (energies / 1e4 < bins[i + 1]))[0] for i in range(len(bins) - 1)]
        
      
        def plot_metric(ax, metric, threshold, y_label, title):
            ax.tick_params(axis='x',which='both', top=False)

            frac = []
            uncert = []
            for idx in indices_in_bins:
                metric_bin = metric[idx]
                if len(metric_bin) == 0:
                    frac_high = 0
                    uncert_high = 0
                else:
                    above = len(metric_bin[metric_bin > threshold])
                    if above==0:
                        above=1 # to avoid divison by 0 below
                    total = len(metric_bin)
                    frac_high = above / total
                    uncert_high = frac_high * ((1 / above) + (1 / total))**0.5
                frac.append(frac_high)
                uncert.append(uncert_high)
            if j == (len(pots) - 1):
                plt.plot(bin_mids, frac, c='k', lw=2.5, label=pot_labels[j], zorder=1)
                plt.fill_between(bin_mids, np.array(frac) - np.array(uncert), np.array(frac) + np.array(uncert),
                                 color='k', alpha=0.3, edgecolor='None')
            elif j == (len(pots) - 2):
                plt.plot(bin_mids, frac, c='k', lw=2, ls='dashed', label=pot_labels[j])
                plt.fill_between(bin_mids, np.array(frac) - np.array(uncert), np.array(frac) + np.array(uncert),
                                 color='k', alpha=0.2, edgecolor='None')
            else:
                plt.plot(bin_mids, frac, lw=1, label=pot_labels[j], zorder=2)
                plt.fill_between(bin_mids, np.array(frac) - np.array(uncert), np.array(frac) + np.array(uncert),
                                 alpha=0.2, edgecolor='None')
        
            plt.xlabel(r'$E\,[10^{4}\,(\mathrm{km}\,\mathrm{s}^{-1})^2]$', fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.xlim(-9.99,-4.21)
            plt.ylim(-0.02, 1.02)
            plt.title(title, fontsize=14, fontweight='bold')
            
            if j==0:
                secax = ax.secondary_xaxis('top', functions=(interp_E_to_r, interp_r_to_E))
                secax.set_xlabel('Galactocentric radius [kpc]', color='grey', fontsize=12) 
                secax.tick_params(axis='x',which='both', colors='grey')
        
        # plt.sca(axs[0, 0])
        # plot_metric(axs[0, 0], loc_veldis, 2.5, r'$f\left(E\,;\,\sigma_v > 2.5\,\mathrm{km}\,\mathrm{s}^{-1} \right)$', 'Local velocity dispersion')
        # plt.legend(frameon=False, ncol=2, bbox_to_anchor=(2.3, 1.85), fontsize=13)

        plt.sca(axs[0, 0])
        plot_metric(axs[0, 0], track_deform, 2, r'$f\left(E\,;\,\bar{\delta} > 2^{\circ} \right)$', 'Deviation from Great Circle')
    
        plt.sca(axs[0, 1])
        plot_metric(axs[0, 1], pm_ang, 10, r'$f\left(E\,;\,\bar{\vartheta} > 10^{\circ} \right)$', 'Proper motion misalignment')

        plt.sca(axs[1, 0])
        plot_metric(axs[1, 0], l_pole_std, 2, r'$f\left(E\,;\,\sigma_{l^{\prime}\,{\mathrm{pole}}} \cos(b^{\prime}_{\mathrm{pole}}) > 2^{\circ} \right)$', 'Longitudinal pole spread')

        plt.sca(axs[1, 1])
        plot_metric(axs[1, 1], b_pole_std, 2, r'$f\left(E\,;\,\sigma_{b^{\prime},\,\mathrm{pole}} > 2^{\circ} \right)$', 'Latitudinal pole spread')
        
        plt.sca(axs[2, 0])
        plot_metric(axs[2, 0], widths, 0.5, r'$f\left(E\,;\,w > 0.5^{\circ} \right)$', 'Width')
        plt.legend(frameon=False, ncol=1, bbox_to_anchor=(1.2, 1.15), fontsize=12)
       
    axs[2, 1].set_visible(False)

    if savefig:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/paper-figs/{}'.format(plotname), bbox_inches='tight')
    plt.close()

def fig5(path_data, potl, real_data, pot_name, labels, plotname, savefig=False):

    ts=np.linspace(-2.5, 0, 1000)
    lmc_xs = [Model.expansion_centres(t)[6:9] for t in ts]
    lmc_vs =  [Model.expansion_centre_velocities(t)[6:9] for t in ts]
    lmc_l_gc, lmc_b_gc, *_ = galactic_coords(np.array(lmc_xs), np.array(lmc_vs))

    disc_xs = [Model.expansion_centres(t)[:3] for t in ts]
    lmc_dist_gc = np.linalg.norm(np.array(lmc_xs) - np.array(disc_xs), axis=1)
    
    fig, ax = plt.subplots(3, 2, figsize=(10, 11))
    mollweide_ax = fig.add_subplot(3, 2, 6, projection='mollweide')
    plt.subplots_adjust(hspace=0.55, wspace=0.3)

    print("Reading data...")
    with h5py.File(path_data + potl, 'r') as file:
        energies = np.array(file['energies'])
        loc_veldis = np.array(file['loc_veldis'])
        widths = np.array(file['widths'])
        track_deform = np.array(file['track_deform'])
        pm_ang = np.array(file['pm_misalignment'])

        t_idx = -1
        l_pole = np.array(file['pole_l'])[:, t_idx]
        b_pole = np.array(file['pole_b'])[:, t_idx]
        
        l_gc = np.array(file['l_gc'])
        b_gc = np.array(file['b_gc'])

        poles = np.stack((l_pole, b_pole))
    print("Data read...")    
    rot_pole = np.array([rotation_matrix_disc @ poles[:, i] for i in range(len(l_pole))])
    l_pole_std, b_pole_std = np.nanstd(rot_pole[:, 0], axis=1), np.nanstd(rot_pole[:, 1], axis=1)
    
    wrapped_l_gc = -np.where(l_gc >= 180, l_gc - 360, l_gc)
            
    mask_q1 = ((wrapped_l_gc > -180) & (wrapped_l_gc < 0) & (b_gc > 0) & (b_gc < 90))
    mask_q2 = ((wrapped_l_gc > 0) & (wrapped_l_gc < 180) & (b_gc > 0) & (b_gc < 90))
    mask_q3 = ((wrapped_l_gc > -180) & (wrapped_l_gc < 0) & (b_gc > -90) & (b_gc < 0))
    mask_q4 = ((wrapped_l_gc > 0) & (wrapped_l_gc < 180) & (b_gc > -90) & (b_gc < 0))
    masks = [mask_q1, mask_q2, mask_q3, mask_q4]

    E_bins = np.linspace(-11, -3, 15)
    bin_mids = [(E_bins[i] + E_bins[i + 1]) / 2 for i in range(len(E_bins) - 1)]

    # Define the colors for each quadrant
    colors = ['#EE7733', '#009988', '#AA4499', '#EE3377']
    
    plt.suptitle(pot_name, y=0.97, fontweight='bold')
    
    for m in range(len(masks)):
        print(f"Plotting Q{m + 1}")
        Es = energies[masks[m]]
        hist, bins = np.histogram(Es / 1e4, E_bins)

        indices_in_bins = []
        for i in range(len(bins) - 1):
            indices = np.where((Es / 1e4 > bins[i]) & (Es / 1e4 < bins[i + 1]))[0]
            indices_in_bins.append(indices)

        # plot_metric_QUAD(ax[0, 0], indices_in_bins, loc_veldis[masks[m]], 2.5,
        #             r'$f\left(E\,;\,\sigma_v > 2.5\,\mathrm{km}\,\mathrm{s}^{-1} \right)$', 'Local velocity dispersion',
        #             colors, labels, bin_mids, m)
        
      
        plot_metric_QUAD(ax[0, 0], indices_in_bins, track_deform[masks[m]], 2,
                    r'$f\left(E\,;\,\bar{\delta} > 2^{\circ} \right)$', 'Deviation from Great Circle',
                    colors, labels, bin_mids, m)
        
        plot_metric_QUAD(ax[0, 1], indices_in_bins, pm_ang[masks[m]], 10,
                    r'$f\left(E\,;\,\bar{\vartheta} > 10^{\circ} \right)$', 'Proper motion misalignment',
                    colors, labels, bin_mids, m)
        
        plot_metric_QUAD(ax[1, 0], indices_in_bins, l_pole_std[masks[m]], 2,
                    r'$f\left(E\,;\,\sigma_{l^{\prime}\,{\mathrm{pole}}} \cos(b^{\prime}_{\mathrm{pole}}) > 2^{\circ} \right)$', 'Longitudinal pole spread',
                    colors, labels, bin_mids, m)
        
        plot_metric_QUAD(ax[1, 1], indices_in_bins, b_pole_std[masks[m]], 2,
                    r'$f\left(E\,;\,\sigma_{b^{\prime},\,\mathrm{pole}} > 2^{\circ} \right)$', 'Latitudinal pole spread',
                    colors, labels, bin_mids, m)
        
        plot_metric_QUAD(ax[2, 0], indices_in_bins, widths[masks[m]], 0.5,
                    r'$f\left(E\,;\,w > 0.5^{\circ} \right)$', 'Width', colors, labels, bin_mids, m)

        ax[2, 1].set_visible(False)

    # DES data
    # plt.sca(ax[0, 1])
    # xerr = np.array([interp_r_to_E(real_data['med_d'] + real_data['std_d']),
    #                    interp_r_to_E(real_data['med_d'] - real_data['std_d'])] )[:,np.newaxis]
    # plt.errorbar(interp_r_to_E(real_data['med_d']), real_data['frac'],xerr=np.abs(xerr - interp_r_to_E(real_data['med_d'])),
    #              marker='o', linestyle='None', ecolor='k',
    #              capsize=5, mfc='red', mec='k', ms=5, label='DES streams')
    # plt.legend(frameon=False, ncol=1, fontsize=10)
        
    plt.sca(mollweide_ax)
    plt.grid(alpha=.25)    

    # Define the vertices for the quadrants
    quadrants = [
        Polygon([[-np.pi, 0], [0, 0], [0, np.pi / 2], [-np.pi, np.pi / 2]], closed=True),  # Top-left
        Polygon([[0, 0], [np.pi, 0], [np.pi, np.pi / 2], [0, np.pi / 2]], closed=True),    # Top-right
        Polygon([[-np.pi, -np.pi / 2], [0, -np.pi / 2], [0, 0], [-np.pi, 0]], closed=True),  # Bottom-left
        Polygon([[0, -np.pi / 2], [np.pi, -np.pi / 2], [np.pi, 0], [0, 0]], closed=True),    # Bottom-right
    ]

    # Fill each quadrant with the respective color
    for color, quad in zip(colors, quadrants):
        mollweide_ax.add_patch(quad)
        quad.set_facecolor(color)
        quad.set_alpha(0.2)

    lmc_l_gc_wrap = np.where(lmc_l_gc >= 180, lmc_l_gc - 360, lmc_l_gc)
    plt.scatter((-lmc_l_gc_wrap[-1]) * u.deg.to(u.rad), lmc_b_gc[-1] * u.deg.to(u.rad), s=100,
                edgecolors='k', facecolor='deepskyblue', marker='*', rasterized=True, zorder=2)
    sc = plt.scatter((-lmc_l_gc_wrap) * u.deg.to(u.rad), lmc_b_gc * u.deg.to(u.rad), rasterized=True,
                     s=5, c=lmc_dist_gc, cmap='Greys_r', norm=LogNorm(vmin=45, vmax=500), lw=0, zorder=1)
    cb=plt.colorbar(sc,location='bottom', aspect=30, pad=0.1, shrink=.6)
    cb.set_label(r'$\mathbf{r}_{\mathrm{LMC}}\,[\mathrm{kpc}]$')
    cb.ax.tick_params(labelsize=10)

    # DES data
    # plt.scatter(-real_data['gc_l']* u.deg.to(u.rad), 
    #             real_data['gc_b']* u.deg.to(u.rad), 
    #             facecolor='r', edgecolor='k')

    mollweide_ax.annotate('LMC orbit', (-2.2*np.pi/6, -1.5*np.pi/8),
                         fontsize=9)
    
    mollweide_ax.annotate('Q1', (-5*np.pi/6, np.pi/8))
    mollweide_ax.annotate('Q2', (4*np.pi/6, np.pi/8))
    mollweide_ax.annotate('Q3', (-5*np.pi/6, -np.pi/8))
    mollweide_ax.annotate('Q4', (4*np.pi/6, -np.pi/8))
    
    mollweide_ax.tick_params( labelsize=8)
    
    x_labels = mollweide_ax.get_xticks() * 180/np.pi
    mollweide_ax.set_xticklabels(['{:.0f}'.format(-label) + r'$^{\circ}$' for label in x_labels])

    ax[2, 0].legend(frameon=False, ncol=1, fontsize=10)

    if savefig:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/paper-figs/{}'.format(plotname), bbox_inches='tight')
        print('Figure saved.')
    plt.close()


def fig6(path_data, potls, real_data, quadlabels, plotname, savefig=False):

    ### ADAPTED QUADRANT PLOTTING FUNCTION USED IN FIG 6 ONLY.
    def plot_metric_Q4(ax, indices_in_bins, metric, threshold, y_label, colors, labels, bin_mids, pot_idx):
        ax.tick_params(axis='x',which='both', top=False)
        frac = []
        uncert = []
        for idx in indices_in_bins:
            metric_bin = metric[idx]
            if len(metric_bin) == 0:
                frac_high = 0
                uncert_high = 0
            else:
                above = len(metric_bin[metric_bin > threshold])
                if above == 0:
                    above = 1  # to avoid division by 0 below
                total = len(metric_bin)
                frac_high = above / total
                uncert_high = frac_high * ((1 / above) + (1 / total))**0.5
            frac.append(frac_high)
            uncert.append(uncert_high)
    
        if pot_idx == 0: #mask_idx == 2 
            ax.plot(bin_mids, frac, lw=3, label=labels[pot_idx], c=colors[pot_idx])
            ax.fill_between(bin_mids, np.array(frac) - np.array(uncert), np.array(frac) + np.array(uncert),
                            alpha=0.2, ec='None', fc=colors[pot_idx])
    
        elif pot_idx == 1: #mask_idx == 2 
            ax.plot(bin_mids, frac, lw=1, ls='dashed', label=labels[pot_idx], c=colors[pot_idx])
            ax.fill_between(bin_mids, np.array(frac) - np.array(uncert), np.array(frac) + np.array(uncert),
                            alpha=0.1, ec='None', fc=colors[pot_idx])
    
        elif pot_idx == 2: #mask_idx == 2 
            ax.plot(bin_mids, frac, lw=1, ls='dotted', label=labels[pot_idx], c=colors[pot_idx])
            ax.fill_between(bin_mids, np.array(frac) - np.array(uncert), np.array(frac) + np.array(uncert),
                            alpha=0.1, ec='None', fc=colors[pot_idx])
    
        ax.set_xlabel(r'$E\,[10^{4}\,(\mathrm{km}\,\mathrm{s}^{-1})^2]$', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlim(-9.99, -4.21)
        ax.set_ylim(-0.02, 1.02)
        if pot_idx==0:
            secax = ax.secondary_xaxis('top', functions=(interp_E_to_r, interp_r_to_E))
            secax.set_xlabel('Galactocentric radius [kpc]', color='grey', fontsize=12) 
            secax.tick_params(axis='x',which='both', colors='grey')
        

    ### GETTING THE LMC'S PAST ORBIT IN GC COORDINATES
    ts=np.linspace(-2.5, 0, 1000)
    lmc_xs = [Model.expansion_centres(t)[6:9] for t in ts]
    lmc_vs =  [Model.expansion_centre_velocities(t)[6:9] for t in ts]
    lmc_l_gc, lmc_b_gc, *_ = galactic_coords(np.array(lmc_xs), np.array(lmc_vs))
    
    disc_xs = [Model.expansion_centres(t)[:3] for t in ts]
    lmc_dist_gc = np.linalg.norm(np.array(lmc_xs) - np.array(disc_xs), axis=1)

    ### BEGINING OF PLOTTING 
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3))
    plt.subplots_adjust(wspace=0.1)
    ax[1].set_visible(False)
    mollweide_ax = fig.add_subplot(1, 2, 2, projection='mollweide')

    ### PLOTTING THE QUADRANTS DATA FOR THE FIDUCIAL 'FULL EXPANSION + LMC' POTENTIAL
    print("Reading data...")
    for j in range(0, len(potls)): 
        potl=potls[j]
        with h5py.File(path_data + potl, 'r') as file:
            energies = np.array(file['energies'])
            pm_ang = np.array(file['pm_misalignment'])
            l_gc = np.array(file['l_gc'])
            b_gc = np.array(file['b_gc'])

        wrapped_l_gc = -np.where(l_gc >= 180, l_gc - 360, l_gc)
    
        # Masks for quadrants data
        mask_q4 = ((wrapped_l_gc > 0) & (wrapped_l_gc < 180) & (b_gc > -90) & (b_gc < 0))

        E_bins = np.linspace(-11, -3, 15)
        bin_mids = [(E_bins[i] + E_bins[i + 1]) / 2 for i in range(len(E_bins) - 1)]
    
        # Define the colors for each quadrant
        colors = ['#EE3377', '#EE3377', '#EE3377']
        plt.sca(ax[0])
    
        # Plotting masked quadrant
        Es = energies[mask_q4]
        hist, bins = np.histogram(Es / 1e4, E_bins)

        indices_in_bins = []
        for i in range(len(bins) - 1):
            indices = np.where((Es / 1e4 > bins[i]) & (Es / 1e4 < bins[i + 1]))[0]
            indices_in_bins.append(indices)
        
        plot_metric_Q4(ax[0], indices_in_bins, pm_ang[mask_q4], 10,
                    r'$f\left(E\,;\,\bar{\vartheta} > 10^{\circ} \right)$',
                    colors, quadlabels, bin_mids, j)

    # Plotting DES data
    xerr = np.array([interp_r_to_E(real_data['med_d'] + real_data['std_d']),
                           interp_r_to_E(real_data['med_d'] - real_data['std_d'])] )[:,np.newaxis]

    n_sample = 7 
    yerr = ((real_data['frac'] * (1 - real_data['frac'])) / n_sample )**.5
    
    plt.errorbar(interp_r_to_E(real_data['med_d']), real_data['frac'],
                 xerr=np.abs(xerr - interp_r_to_E(real_data['med_d'])),
                 yerr = yerr,
                 marker='o', linestyle='None', ecolor='k',
                 capsize=2, mfc='red', mec='k', ms=5, label='DES streams')

    # Plotting LSST prediction
    with h5py.File(path_data + potls[0], 'r') as file:
            energies = np.array(file['energies'])
            pm_ang = np.array(file['pm_misalignment'])
    lsst_mask = rubinlsst_mask(path_data, potls[0])
    E_lsst = energies[lsst_mask]
    pmang_lsst = pm_ang[lsst_mask]
    hist, bins = np.histogram(E_lsst / 1e4, E_bins)
    
    indices_in_bins = []
    for i in range(len(bins) - 1):
        indices = np.where((E_lsst / 1e4 > bins[i]) & (E_lsst / 1e4 < bins[i + 1]))[0]
        indices_in_bins.append(indices)

    frac = []
    uncert = []

    for idx in indices_in_bins:
        pm_ang_bin = pmang_lsst[idx]
        if len(pm_ang_bin)==0:
            frac_high = np.nan
            uncert_high = 0
        else:
            above = len(pm_ang_bin[pm_ang_bin > 10])
            if above==0:
                above=1 # to avoid divison by 0 below
            total = len(pm_ang_bin)
            frac_high = above / total
            uncert_high = frac_high * ((1/above) + (1/total))**0.5
        frac.append(frac_high)
        uncert.append(uncert_high)

    plt.plot(bin_mids, frac, c='k', lw=1, ls='solid', label='LSST', zorder=2)
    plt.fill_between(bin_mids, np.array(frac)-np.array(uncert), np.array(frac)+np.array(uncert),
                     color='k', alpha=0.2, edgecolor='None')

    plt.legend(frameon=False, ncol=1, fontsize=10)


    ### PLOTTING THE MOLLWIEDE PROJECTION 
    
    plt.sca(mollweide_ax)
    plt.grid(alpha=.25)    
    colors = ['#EE7733', '#009988', '#AA4499', '#EE3377']
    # Define the vertices for the quadrants
    quadrants = [
        Polygon([[-np.pi, 0], [0, 0], [0, np.pi / 2], [-np.pi, np.pi / 2]], closed=True),  # Top-left
        Polygon([[0, 0], [np.pi, 0], [np.pi, np.pi / 2], [0, np.pi / 2]], closed=True),    # Top-right
        Polygon([[-np.pi, -np.pi / 2], [0, -np.pi / 2], [0, 0], [-np.pi, 0]], closed=True),  # Bottom-left
        Polygon([[0, -np.pi / 2], [np.pi, -np.pi / 2], [np.pi, 0], [0, 0]], closed=True),    # Bottom-right
    ]

    # Fill each quadrant with the respective color
    for color, quad in zip(colors, quadrants):
        mollweide_ax.add_patch(quad)
        quad.set_facecolor(color)
        quad.set_alpha(0.2)

    # Past orbit of LMC
    lmc_l_gc_wrap = np.where(lmc_l_gc >= 180, lmc_l_gc - 360, lmc_l_gc)
    plt.scatter((-lmc_l_gc_wrap[-1]) * u.deg.to(u.rad), lmc_b_gc[-1] * u.deg.to(u.rad), s=100,
                edgecolors='k', facecolor='deepskyblue', marker='*', rasterized=True, zorder=2)
    sc = plt.scatter((-lmc_l_gc_wrap) * u.deg.to(u.rad), lmc_b_gc * u.deg.to(u.rad), rasterized=True,
                     s=5, c=lmc_dist_gc, cmap='Greys_r', norm=LogNorm(vmin=45, vmax=500), lw=0, zorder=1)
    cb=plt.colorbar(sc,location='bottom', aspect=30, pad=0.1, shrink=.6)
    cb.set_label(r'$\mathbf{r}_{\mathrm{LMC}}\,[\mathrm{kpc}]$')
    cb.ax.tick_params(labelsize=10)

    # DES data
    plt.scatter(-real_data['gc_l']* u.deg.to(u.rad), 
                real_data['gc_b']* u.deg.to(u.rad), 
                facecolor='r', edgecolor='k')

    mollweide_ax.annotate('LMC orbit', (-2.2*np.pi/6, -1.5*np.pi/8),
                         fontsize=9)
    
    mollweide_ax.annotate('Q1', (-5*np.pi/6, np.pi/8))
    mollweide_ax.annotate('Q2', (4*np.pi/6, np.pi/8))
    mollweide_ax.annotate('Q3', (-5*np.pi/6, -np.pi/8))
    mollweide_ax.annotate('Q4', (4*np.pi/6, -np.pi/8))
    
    mollweide_ax.tick_params( labelsize=9)
    
    x_labels = mollweide_ax.get_xticks() * 180/np.pi
    mollweide_ax.set_xticklabels(['{:.0f}'.format(-label) + r'$^{\circ}$' for label in x_labels])
    
    if savefig:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/paper-figs/{}'.format(plotname), bbox_inches='tight')
        print('Figure saved.')
    plt.close()
    
###---------------------------------------------------------------------------------------------------------------- 
# Run the script and plot scripts 
###---------------------------------------------------------------------------------------------------------------- 
print("Script is running...")

### Figure 1
fig1_data_path = '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/16384-dt1Myr/'
streams_fig1 = list(['stream_9', 'stream_12680', 'stream_16212']) 
plotname_fig1a, plotname_fig1b = 'fig1a', 'fig1b'
fs_fig1a, fs_fig1b = (13,4.5), (16,4.5)

potentials_fig1a = list(['rigid-mw.hdf5','static-mw.hdf5', 'rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5'])
labels_fig1a = list(['Rigid MW \n without motion (no LMC)', 'Rigid MW \n + motion (no LMC)', 'Rigid Monopole \n \& LMC', 'Evolving Monopole \n \& LMC'])

potentials_fig1b = list(['md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5',\
                        'full-MWhalo-full-MWdisc-no-LMC.hdf5', 'full-MWhalo-full-MWdisc-full-LMC.hdf5'])
labels_fig1b = list(['Monopole + Dipole \n \& LMC', 'Monopole + Quadrupole \n \& LMC', 'Monopole + Dipole \n + Quadrupole \& LMC', \
               'Full Expansion \n(no LMC)', 'Full Expansion \n \& LMC'])

# fig1(streams_fig1, fig1_data_path, potentials_fig1a, labels_fig1a, fs_fig1a, plotname_fig1a, True)
# fig1(streams_fig1, fig1_data_path, potentials_fig1b, labels_fig1b, fs_fig1b, plotname_fig1b, True)

data_path = '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/plotting_data/16384-dt1Myr/'
    
### Figure 3
print("Plotting figure 3...")
plotname_fig3 = 'fig3-pdf' 
# fig3_cdf(data_path, plotname_fig3, False)
# fig3_pdf(data_path, plotname_fig3, True)

### Figure 4
print("Plotting figure 4...")
potentials_fig4 = list(['rigid-mw.hdf5','static-mw.hdf5', 'rm-MWhalo-full-MWdisc-full-LMC.hdf5', 'em-MWhalo-full-MWdisc-full-LMC.hdf5',\
               'md-MWhalo-full-MWdisc-full-LMC.hdf5', 'mq-MWhalo-full-MWdisc-full-LMC.hdf5', 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5',\
                'full-MWhalo-full-MWdisc-no-LMC.hdf5', 'full-MWhalo-full-MWdisc-full-LMC.hdf5'])
labels_fig4 = list(['Rigid MW without motion (no LMC)', 'Rigid MW + motion (no LMC)', 'Rigid Monopole \& LMC', 'Evolving Monopole \& LMC', \
       'Monopole + Dipole \& LMC', 'Monopole + Quadrupole \& LMC', 'Monopole + Dipole + Quadrupole \& LMC', 'Full Expansion (no LMC)', 'Full Expansion \& LMC'])
# fig4(data_path, potentials_fig4, labels_fig4, 'fig4', True)

### Figure 5
print("Plotting figure 5...")
potential_fig5 = 'full-MWhalo-full-MWdisc-full-LMC.hdf5'
potential_name_fig5 = 'Full Expansion \& LMC'
labels_fig5 = list(['Q1','Q2','Q3','Q4'])
# fig5(data_path, potential_fig5, potential_name_fig5, labels_fig5, 'fig5', True)
fig5(data_path, potential_fig5, DES_plot_data, potential_name_fig5, labels_fig5, 'fig5', True)


### Figure 6
print("Plotting figure 6...")
data_path='/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/plotting_data/16384-dt1Myr/'
pots_fig6 = list(['full-MWhalo-full-MWdisc-full-LMC.hdf5', 'full-MWhalo-full-MWdisc-no-LMC.hdf5',
                 'rm-MWhalo-full-MWdisc-full-LMC.hdf5'])
quadlabels_fig6 = list(['Q4 Full Expansion \& LMC','Q4 Full Expansion (no LMC)','Q4 Rigid Monopole \& LMC'])

fig6(data_path, pots_fig6, DES_plot_data, quadlabels_fig6, 'fig6', True)

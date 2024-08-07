from scipy.spatial.transform import Rotation
import numpy as np
import pathlib
import h5py
import astropy.units as u
import gala.dynamics as gd
import gala.coordinates as gc
import yaml
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
plt.style.use('/mnt/ceph/users/rbrooks/oceanus/analysis/my_standard.mplstyle')

def histhexplt(plttype,xvals,yvals,cvals=None,hexfn=np.median,
               f=None,ax=None,fs=None,cbarlabel='Cbar label',
               xlim=(None,None),ylim=(None,None),vlim=(None,None),xbins=None,ybins=None,
               gsxdiv=100,gsydiv=100,logtog=0,densitytog=0,
               cmapp='coolwarm',xlabel=None,ylabel=None):
    #set gridsize from the values being plotted, use gridsize dividers (gsydiv, gsxdiv) to set size dynamcially
    if ybins is None:
        if gsydiv is not None:
            ybins = int(np.floor(len(np.unique(yvals))/gsydiv))
        else: 
            ybins = int(len(np.unique(yvals)))
    if xbins is None:
        if gsxdiv is not None:
            xbins = int(np.floor(len(np.unique(xvals))/gsxdiv))
        else: 
            xbins = int(len(np.unique(xvals)))
    gs_bs = (xbins,ybins) #number of bins in x and y

    #set extent and range by xlim and ylim if given, otherwise it is set by inar params later
    if xlim[0] is None:
        xv = xvals[np.isfinite(xvals)]
        xmin = np.amin(np.unique(xv))
    else:
        xmin = xlim[0]
    if xlim[1] is None:
        xv = xvals[np.isfinite(xvals)]
        xmax = np.amax(np.unique(xv))
    else:
        xmax = xlim[1]
    
    if ylim[0] is None:
        yv = yvals[np.isfinite(yvals)]
        ymin = np.amin(np.unique(yv))
    else:
        ymin = ylim[0]
    if ylim[1] is None:
        yv = yvals[np.isfinite(yvals)]
        ymax = np.amax(np.unique(yv))
    else:
        ymax = ylim[1]
    ex_rg = ([xmin,xmax],[ymin,ymax])

    if plttype == 'hist':
        if logtog == 1:
            normtog = mpl.colors.LogNorm()
        else:
            normtog=None
        if densitytog == 1:## are these indexings are wrong
            todump=ax.hist2d(xvals.flatten(),yvals.flatten(),bins=gs_bs,range=ex_rg,norm=normtog,vmin=vlim[0],vmax=vlim[1],cmap=cmapp)
            
            ax.pcolormesh(todump[1], todump[2], (todump[0]/np.sum(todump[0])).T,vmin=vlim[0],vmax=vlim[1],cmap=cmapp)        
        else:
            histout=ax.hist2d(xvals.flatten(),yvals.flatten(),bins=gs_bs,range=ex_rg,norm=normtog,vmin=vlim[0],vmax=vlim[1],cmap=cmapp)

        if densitytog == 1:
            f.colorbar(histout,ax=ax,label=cbarlabel)
        else:
            f.colorbar(histout[3],ax=ax,label=cbarlabel)

    elif plttype == 'hex':
        if logtog == 1:
            binstog ='log'
        else:
            binstog=None
        hexout=ax.hexbin(xvals,yvals,C=cvals,reduce_C_function=hexfn,gridsize=gs_bs,
                         bins=binstog,extent=list(np.array(ex_rg).flat),vmin=vlim[0],vmax=vlim[1],cmap=cmapp)
        if densitytog == 1:
            outar = hexout.get_array()
            hexout.set_array(outar/np.sum(outar))
            if vlim[0] is None:
                hexout.set_clim(np.amin(hexout.get_array()),np.amax(hexout.get_array()))
            else:
                hexout.set_clim(vlim[0],vlim[1])
        if densitytog == 1:
            f.colorbar(hexout,ax=ax, label=cbarlabel)
        else:
            f.colorbar(hexout,ax=ax, label=cbarlabel)#

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(ptitle)

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

def RELHIST_radialphase_peris_veldis(galdist, pericenters, apocenters, sigmavs, plotname, savefig=True):

    f = (galdist[0] - pericenters[0]) / (apocenters[0] - pericenters[0])
    f0 = (galdist[1] - pericenters[1]) / (apocenters[1] - pericenters[1])
    fig, ax = plt.subplots(1,2, figsize=(10,3), sharey='row')

    plt.subplots_adjust(wspace=0.)
    plt.sca(ax[0])
    
    xbins = np.linspace(-0.05, 1.05, 40) 
    ybins = np.linspace(0, 40, 35)
    h, xedges, yedges, image = plt.hist2d(f, sigmavs[0], bins=(xbins, ybins))
    h0, xedges1, yedges1, image1 = plt.hist2d(f0, sigmavs[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / h0
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 100
    plot=plt.pcolormesh(xedges, yedges, rel_diff.T, cmap='coolwarm', vmin=-2, vmax=2, rasterized=True)
    
    plt.xlabel(r'$\frac{r_{\mathrm{gal}} - r_p}{r_a - r_p}$', fontsize=14)
    plt.ylabel('$\sigma_{v}$ [km/s]', fontsize=12)
    plt.xlim(-0.05,1.05)
    plt.xticks([0, 0.5, 1])

    plt.sca(ax[1])    
    xbins = np.linspace(9, 26 , 40) 
    ybins = np.linspace(0, 40 , 35)
    h, xedges, yedges, image = plt.hist2d(pericenters[0],sigmavs[0], bins=(xbins, ybins))
    h0, xedges1, yedges1, image1 = plt.hist2d(pericenters[1],sigmavs[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / h0
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 100
    plot=plt.pcolormesh(xedges, yedges, rel_diff.T, cmap='coolwarm', vmin=-2, vmax=2, rasterized=True)
    
    plt.xlabel('$r_{p}$ [kpc]', fontsize=12)
    plt.xlim(9,26)
    plt.ylim(0,30)
    
    cb = fig.colorbar(plot, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01)
    cb.set_label(r'$\left( \mathcal{N}_{\mathrm{LMC}} - \mathcal{N}_{\mathrm{no\,LMC}} \right) / \mathcal{N}_{\mathrm{no\,LMC}}$')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/lmceffects/{}'.format(plotname))
    plt.close()
    
def RELHEX_radialphase_peris_veldis(galdist, pericenters, apocenters, sigmavs, plotname, savefig=True):
    
    f = (galdist[0] - pericenters[0]) / (apocenters[0] - pericenters[0])
    f0 = (galdist[1] - pericenters[1]) / (apocenters[1] - pericenters[1])
    fig, ax = plt.subplots(1,2, figsize=(10,3), sharey='row')
    plt.subplots_adjust(wspace=0.)

    ### Proxy for where stream is on orbit vs veldis
    xbins = np.linspace(-0.05, 1.05, 40) 
    ybins = np.linspace(0, 40, 35)
    h, xedges, yedges = np.histogram2d(f, sigmavs[0], bins=(xbins, ybins))
    h0, xedges1, yedges1 = np.histogram2d(f0, sigmavs[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / len(h0)
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 1

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xgrid, ygrid = np.meshgrid(xcenters, ycenters) 
    xbins = int(len(np.unique(xgrid.flatten())))
    ybins = int(len(np.unique(ygrid.flatten())))
    gs_bs = (xbins,ybins) 
    ex_rg = ([-0.05,1.05],[0,30])
    plt.sca(ax[0])
    plt.hexbin(xgrid.flatten(), ygrid.flatten(), C=rel_diff.T.flatten(), gridsize=gs_bs,
                     extent=list(np.array(ex_rg).flat), vmin=-1, vmax=1, cmap='seismic')
    plt.xlabel(r'$\frac{r_{\mathrm{gal}} - r_p}{r_a - r_p}$')
    plt.ylabel('$\sigma_{v}$ [km/s]')
   
    ### Peris vs veldis
    xbins = np.linspace(9, 26 , 40) 
    ybins = np.linspace(0, 40 , 35)
    h, xedges, yedges =np.histogram2d(pericenters[0],sigmavs[0], bins=(xbins, ybins))
    h0, xedges1, yedges1 = np.histogram2d(pericenters[1],sigmavs[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / len(h0)
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 1
    
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xgrid, ygrid = np.meshgrid(xcenters, ycenters) 
    xbins = int(len(np.unique(xgrid.flatten())))
    ybins = int(len(np.unique(ygrid.flatten())))
    gs_bs = (xbins,ybins) 
    ex_rg = ([9,26],[0,30])
    plt.sca(ax[1])
    hexplt=plt.hexbin(xgrid.flatten(), ygrid.flatten(), C=rel_diff.T.flatten(), gridsize=gs_bs,
                     extent=list(np.array(ex_rg).flat), vmin=-1, vmax=1, cmap='seismic')
    plt.xlabel('$r_{p}$ [kpc]')
    
    fig.colorbar(hexplt, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01,
                 label=r'$\mathcal{N}_{\mathrm{LMC}} - \mathcal{N}_{\mathrm{no\,LMC}} $')
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/lmceffects/{}'.format(plotname))
    plt.close()
    
def RELHIST_poledisp_peri(poledis_l, poledis_b, pericenters, plotname, savefig=True):

    fig, ax = plt.subplots(1,2, figsize=(9,2.75), sharey='row')

    plt.subplots_adjust(wspace=0.)
    plt.sca(ax[0])
    
    xbins = np.linspace(np.log10(0.1), np.log10(250), 30) 
    ybins = np.linspace(9, 26 , 25)
    h, xedges, yedges, image = plt.hist2d(np.log10(poledis_l[0]), pericenters[0], bins=(xbins, ybins))
    h0, xedges1, yedges1, image1 = plt.hist2d(np.log10(poledis_l[1]), pericenters[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / h0
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 100
    plot=plt.pcolormesh(xedges, yedges, rel_diff.T, cmap='coolwarm', vmin=-2, vmax=2, rasterized=True)
    
    plt.xlabel(r'$\log_{10}(\sigma_{l,\mathrm{pole}})\,[^\circ]$')
    plt.xlim(np.log10(0.1),np.log10(300))
    plt.ylim(9,26)
    plt.ylabel('$r_p$ [kpc]')

    plt.sca(ax[1])
    xbins = np.linspace(np.log10(0.1), np.log10(50), 30) 
    ybins = np.linspace(9, 26, 25)
    h, xedges, yedges, image = plt.hist2d(np.log10(poledis_b[0]), pericenters[0], bins=(xbins, ybins))
    h0, xedges1, yedges1, image1 = plt.hist2d(np.log10(poledis_b[1]), pericenters[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / h0
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 100
    plot=plt.pcolormesh(xedges, yedges, rel_diff.T, cmap='coolwarm', vmin=-2, vmax=2, rasterized=True)
    
    plt.xlim(np.log10(0.1),np.log10(50))
    plt.xlabel(r'$\log_{10}(\sigma_{b,\mathrm{pole}})\,[^\circ]$')

    cb = fig.colorbar(plot, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01)
    cb.set_label(r'$\left( \mathcal{N}_{\mathrm{LMC}} - \mathcal{N}_{\mathrm{no\,LMC}} \right) / \mathcal{N}_{\mathrm{no\,LMC}}$')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/lmceffects/{}'.format(plotname))
    plt.close()
      
def RELHEX_poledisp_peri(poledis_l, poledis_b, pericenters, plotname, savefig=True):

    fig, ax = plt.subplots(1,2, figsize=(9,2.75), sharey='row')

    plt.subplots_adjust(wspace=0.)
    
    ### pole_l dispersion vs pericenter
    xbins = np.linspace(np.log10(0.1), np.log10(250), 30) 
    ybins = np.linspace(9, 26 , 25)
    h, xedges, yedges = np.histogram2d(np.log10(poledis_l[0]), pericenters[0], bins=(xbins, ybins))
    h0, xedges1, yedges1 = np.histogram2d(np.log10(poledis_l[1]), pericenters[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / len(h0)
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 1

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xgrid, ygrid = np.meshgrid(xcenters, ycenters) 
    xbins = int(len(np.unique(xgrid.flatten())))
    ybins = int(len(np.unique(ygrid.flatten())))
    gs_bs = (xbins,ybins) 
    ex_rg = ([np.log10(0.1),np.log10(300)],[9,26])
    plt.sca(ax[0])
    hexplt=plt.hexbin(xgrid.flatten(), ygrid.flatten(), C=rel_diff.T.flatten(), gridsize=gs_bs,
                     extent=list(np.array(ex_rg).flat), vmin=-1, vmax=1, cmap='seismic')
    plt.xlabel(r'$\log_{10}(\sigma_{l,\mathrm{pole}})\,[^\circ]$')
    plt.ylabel('$r_p$ [kpc]')

    plt.sca(ax[1])
    
    ### pole_b dispersion vs pericenter
    xbins = np.linspace(np.log10(0.1), np.log10(50), 30) 
    ybins = np.linspace(9, 26, 25)
    h, xedges, yedges = np.histogram2d(np.log10(poledis_b[0]), pericenters[0], bins=(xbins, ybins))
    h0, xedges1, yedges1 = np.histogram2d(np.log10(poledis_b[1]), pericenters[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / len(h0)
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 1

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xgrid, ygrid = np.meshgrid(xcenters, ycenters) 
    xbins = int(len(np.unique(xgrid.flatten())))
    ybins = int(len(np.unique(ygrid.flatten())))
    gs_bs = (xbins,ybins) 
    ex_rg = ([np.log10(0.1),np.log10(50)],[9,26])
    hexplt=plt.hexbin(xgrid.flatten(), ygrid.flatten(), C=rel_diff.T.flatten(), gridsize=gs_bs,
                     extent=list(np.array(ex_rg).flat), vmin=-1, vmax=1, cmap='seismic')
    plt.xlabel(r'$\log_{10}(\sigma_{b,\mathrm{pole}})\,[^\circ]$')

    cb = fig.colorbar(hexplt, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01,
                      label=r'$\mathcal{N}_{\mathrm{LMC}} - \mathcal{N}_{\mathrm{no\,LMC}}$')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/lmceffects/{}'.format(plotname))
    plt.close()
    
def RELHIST_poledisp_distance(poledis_l, poledis_b, distances,plotname, savefig=True):
    
    fig, ax = plt.subplots(1,2, figsize=(9,2.75), sharey='row')

    plt.subplots_adjust(wspace=0.)
    plt.sca(ax[0])
    
    xbins = np.linspace(np.log10(0.1), np.log10(250), 30) 
    ybins = np.linspace(0, 55, 25)
    h, xedges, yedges, image = plt.hist2d(np.log10(poledis_l[0]), distances[0], bins=(xbins, ybins))
    h0, xedges1, yedges1, image1 = plt.hist2d(np.log10(poledis_l[1]), distances[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / h0
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 100
    plot=plt.pcolormesh(xedges, yedges, rel_diff.T, cmap='coolwarm', vmin=-2, vmax=2, rasterized=True)

    plt.xlabel(r'$\log_{10}(\sigma_{l,\mathrm{pole}})\,[^\circ]$')
    plt.ylabel('$r_{\mathrm{gal}}$ [kpc]')
    plt.xlim(np.log10(0.1),np.log10(300))
    plt.ylim(0,55)

    plt.sca(ax[1])
    
    xbins = np.linspace(np.log10(0.1), np.log10(50), 30) 
    ybins = np.linspace(0, 55, 25)
    h, xedges, yedges, image = plt.hist2d(np.log10(poledis_b[0]), distances[0], bins=(xbins, ybins))
    h0, xedges1, yedges1, image1 = plt.hist2d(np.log10(poledis_b[1]), distances[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / h0
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 100
    plot=plt.pcolormesh(xedges, yedges, rel_diff.T, cmap='coolwarm', vmin=-2, vmax=2, rasterized=True)
    
    plt.xlabel(r'$\log_{10}(\sigma_{b,\mathrm{pole}})\,[^\circ]$')
    plt.xlim(np.log10(0.1),np.log10(50))

    cb = fig.colorbar(plot, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01)
    cb.set_label(r'$\left( \mathcal{N}_{\mathrm{LMC}} - \mathcal{N}_{\mathrm{no\,LMC}} \right) / \mathcal{N}_{\mathrm{no\,LMC}}$')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/lmceffects/{}'.format(plotname))
    plt.close()
    
def RELHEX_poledisp_distance(poledis_l, poledis_b, distances, plotname, savefig=True):
    
    fig, ax = plt.subplots(1,2, figsize=(9,2.75), sharey='row')
    plt.subplots_adjust(wspace=0.)
    
    ### pole_l dispersion vs pericenter
    xbins = np.linspace(np.log10(0.1), np.log10(250), 30) 
    ybins = np.linspace(0, 55, 25)
    h, xedges, yedges = np.histogram2d(np.log10(poledis_l[0]), distances[0], bins=(xbins, ybins))
    h0, xedges1, yedges1 = np.histogram2d(np.log10(poledis_l[1]), distances[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / len(h0)
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 1

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xgrid, ygrid = np.meshgrid(xcenters, ycenters) 
    xbins = int(len(np.unique(xgrid.flatten())))
    ybins = int(len(np.unique(ygrid.flatten())))
    gs_bs = (xbins,ybins) 
    ex_rg = ([np.log10(0.1),np.log10(300)],[0,55])
    plt.sca(ax[0])
    hexplt=plt.hexbin(xgrid.flatten(), ygrid.flatten(), C=rel_diff.T.flatten(), gridsize=gs_bs,
                     extent=list(np.array(ex_rg).flat), vmin=-1, vmax=1, cmap='seismic')
    plt.xlabel(r'$\log_{10}(\sigma_{l,\mathrm{pole}})\,[^\circ]$')
    plt.ylabel('$r_{\mathrm{gal}}$ [kpc]')
    
    ### pole_b dispersion vs pericenter
    xbins = np.linspace(np.log10(0.1), np.log10(50), 30) 
    ybins = np.linspace(0, 55, 25)
    h, xedges, yedges = np.histogram2d(np.log10(poledis_b[0]), distances[0], bins=(xbins, ybins))
    h0, xedges1, yedges1 = np.histogram2d(np.log10(poledis_b[1]), distances[1], bins=(xbins, ybins))
    
    rel_diff = (h - h0) / len(h0)
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 1

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xgrid, ygrid = np.meshgrid(xcenters, ycenters) 
    xbins = int(len(np.unique(xgrid.flatten())))
    ybins = int(len(np.unique(ygrid.flatten())))
    gs_bs = (xbins,ybins) 
    ex_rg = ([np.log10(0.1),np.log10(50)],[0,55])
    plt.sca(ax[1])
    hexplt=plt.hexbin(xgrid.flatten(), ygrid.flatten(), C=rel_diff.T.flatten(), gridsize=gs_bs,
                     extent=list(np.array(ex_rg).flat), vmin=-1, vmax=1, cmap='seismic')
    plt.xlabel(r'$\log_{10}(\sigma_{b,\mathrm{pole}})\,[^\circ]$')

    cb = fig.colorbar(hexplt, ax=[ax[0], ax[1]],location='right', aspect=30, pad=0.01,
                      label=r'$\mathcal{N}_{\mathrm{LMC}} - \mathcal{N}_{\mathrm{no\,LMC}}$')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/lmceffects/{}'.format(plotname))
    plt.close()
    
def RELHIST_width_length(width, length, plotname, savefig=True):
    
    fig, ax = plt.subplots(1,1, figsize=(5,3))
    plt.sca(ax)
    
    xbins = np.linspace(np.log10(1e-2), np.log10(3e1), 35) 
    ybins = np.linspace(np.log10(5e-1), np.log10(1e2), 30)
    h, xedges, yedges, image = plt.hist2d(np.log10(width[0]), np.log10(length[0]), bins=(xbins, ybins))
    h0, xedges1, yedges1, image1 = plt.hist2d(np.log10(width[1]), np.log10(length[1]), bins=(xbins, ybins))
    
    rel_diff = (h - h0) / h0
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 100
    plot=plt.pcolormesh(xedges, yedges, rel_diff.T, cmap='coolwarm', vmin=-2, vmax=2, rasterized=True)
    
    plt.xlabel('$\log_{10}(w)\,[^{\circ}]$')
    plt.ylabel('$\log_{10}(l)$ [kpc]')
    plt.xlim(np.log10(1e-2),np.log10(3e1))
    plt.ylim(np.log10(5e-1),np.log10(1e2))

    cb = fig.colorbar(plot, ax=ax,location='right', aspect=30, pad=0.01)
    cb.set_label(r'$\left( \mathcal{N}_{\mathrm{LMC}} - \mathcal{N}_{\mathrm{no\,LMC}} \right) / \mathcal{N}_{\mathrm{no\,LMC}}$')
    cb.ax.tick_params(labelsize=12)
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/lmceffects/{}'.format(plotname))
    plt.close()
    
def RELHEX_width_length(width, length, plotname, savefig=True):
    
    fig, ax = plt.subplots(1,1, figsize=(5,3))
  
    xbins = np.linspace(np.log10(1e-2), np.log10(3e1), 35) 
    ybins = np.linspace(np.log10(5e-1), np.log10(1e2), 30)

    h, xedges, yedges = np.histogram2d(np.log10(widths_joint[0]), np.log10(lengths_joint[0]), bins=(xbins, ybins))
    h0, xedges1, yedges1 = np.histogram2d(np.log10(widths_joint[1]), np.log10(lengths_joint[1]), bins=(xbins, ybins))
    
    rel_diff = (h - h0) / len(h0)
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 1
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xgrid, ygrid = np.meshgrid(xcenters, ycenters) 
    
    xbins = int(len(np.unique(xgrid.flatten())))
    ybins = int(len(np.unique(ygrid.flatten())))
    gs_bs = (xbins,ybins) 
    ex_rg = ([np.log10(1e-2), np.log10(3e1)],[np.log10(5e-1),np.log10(1e2)])
    
    hexplt=plt.hexbin(xgrid.flatten(), ygrid.flatten(), C=rel_diff.T.flatten(), gridsize=gs_bs,
                     extent=list(np.array(ex_rg).flat), vmin=-1, vmax=1, cmap='seismic')
    plt.xlabel(r'$\log_{10}(w)\,[^{\circ}]$')
    plt.ylabel('$\log_{10}(l)$ [kpc]')

    fig.colorbar(hexplt, ax=ax, location='right', aspect=30, pad=0.02,
                 label=r'$\mathcal{N}_{\mathrm{LMC}} - \mathcal{N}_{\mathrm{no\,LMC}} $')
    
    if savefig==True:
        plt.savefig('/mnt/ceph/users/rbrooks/oceanus/analysis/figures/lmceffects/{}'.format(plotname))
    plt.close()
    
    
###-------------------------------------------------------------------------------
### run the script
###--------------------------------------------------------------------------------

path = '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/'

rgal_0 = []
peris_0 = []
apos_0 = []
widths_0 = []
lengths_0 = []
loc_veldis_0 = []
lmc_sep_0 = []
pole_b_dis_0 = []
pole_l_dis_0 = []

Nstreams = 1024
for i in range(Nstreams):
    data_path = pathlib.Path(path) / 'full-MWhalo-full-MWdisc-no-LMC.hdf5' 
    with h5py.File(data_path,'r') as file:
        
        pos = np.array(file['stream_{}'.format(i)]['positions'])[-1]
        vel = np.array(file['stream_{}'.format(i)]['velocities'])[-1]
        lons, lats = lons_lats(pos, vel)
        loc_veldis_0.append(local_veldis(lons, vel))
        rgal_0.append( np.nanmedian(np.linalg.norm(np.array(file['stream_{}'.format(i)]['positions'])[-1],axis=1)) )
        peris_0.append(np.array(file['stream_{}'.format(i)]['pericenter']))
        apos_0.append(np.array(file['stream_{}'.format(i)]['apocenter']))
        widths_0.append(np.array(file['stream_{}'.format(i)]['width']))
        lengths_0.append(np.array(file['stream_{}'.format(i)]['length']))
        pole_b_dis_0.append(np.nanstd(np.array(file['stream_{}'.format(i)]['pole_b'])[-1]))
        pole_l_dis_0.append(np.nanstd(np.array(file['stream_{}'.format(i)]['pole_l'])[-1]))


rgal = []
peris = []
apos = []
widths = []
lengths = []
loc_veldis = []
pole_b_dis = []
pole_l_dis = []
masses = []

for i in range(Nstreams):
    data_path = pathlib.Path(path) / 'Full-MWhalo-MWdisc-LMC.hdf5' 
    with h5py.File(data_path,'r') as file:
        
        pos = np.array(file['stream_{}'.format(i)]['positions'])[-1]
        vel = np.array(file['stream_{}'.format(i)]['velocities'])[-1]
        lons, lats = lons_lats(pos, vel)
        loc_veldis.append(local_veldis(lons, vel))
        rgal.append( np.nanmedian(np.linalg.norm(np.array(file['stream_{}'.format(i)]['positions'])[-1],axis=1)) )
        peris.append(np.array(file['stream_{}'.format(i)]['pericenter']))
        apos.append(np.array(file['stream_{}'.format(i)]['apocenter']))
        widths.append(np.array(file['stream_{}'.format(i)]['width']))
        lengths.append(np.array(file['stream_{}'.format(i)]['length']))
        pole_b_dis.append(np.nanstd(np.array(file['stream_{}'.format(i)]['pole_b'])[-1]))
        pole_l_dis.append(np.nanstd(np.array(file['stream_{}'.format(i)]['pole_l'])[-1]))
        masses.append(np.array(file['stream_{}'.format(i)]['progenitor-mass']))

loc_veldis_joint = np.array([loc_veldis, loc_veldis_0]) 
rgal_joint = np.array([rgal,rgal_0])
peris_joint = np.array([peris, peris_0])
apos_joint = np.array([apos, apos_0])
widths_joint = np.array([widths, widths_0])
lengths_joint = np.array([lengths, lengths_0])
pole_b_dis_joint  = np.array([pole_b_dis, pole_b_dis_0])
pole_l_dis_joint  = np.array([pole_l_dis, pole_l_dis_0])    

print("* saving figures...")
    
# RELHIST_radialphase_peris_veldis(rgal_joint, peris_joint, apos_joint, loc_veldis_joint, 'radialphase_peris_veldis', True)
# RELHIST_poledisp_peri(pole_l_dis_joint, pole_b_dis_joint, peris_joint, 'poledisp_peri', True)
# RELHIST_poledisp_distance(pole_l_dis_joint, pole_b_dis_joint, rgal_joint, 'poledisp_distance', True)
# RELHIST_width_length(widths_joint, lengths_joint, 'width_length', True)

RELHEX_radialphase_peris_veldis(rgal_joint, peris_joint, apos_joint, loc_veldis_joint, 'radialphase_peris_veldis_hex', True)
RELHEX_poledisp_peri(pole_l_dis_joint, pole_b_dis_joint, peris_joint, 'poledisp_peri_hex', True)
RELHEX_poledisp_distance(pole_l_dis_joint, pole_b_dis_joint, rgal_joint, 'poledisp_distance_hex', True)
RELHEX_width_length(widths_joint, lengths_joint, 'width_length_hex', True)
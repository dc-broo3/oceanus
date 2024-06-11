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

galcen_v_sun = (11.1, 245, 7.3)*u.km/u.s
galcen_distance = 8.249*u.kpc

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

def pm_misalignment(lons, xfs, vfs):
    
    # Compute percentiles
    lower_value = np.nanpercentile(lons, 5)
    upper_value = np.nanpercentile(lons, 95)
    
    # Filter away outlier particles
    lons_mainbody = lons[(lons >= lower_value) & (lons <= upper_value)]
    xfs_mainbody = xfs[1:][(lons >= lower_value) & (lons <= upper_value)] 
    vfs_mainbody = vfs[1:][(lons >= lower_value) & (lons <= upper_value)] 
    
    # Create bins
    lon_bins = np.linspace(np.nanmin(lons_mainbody), np.nanmax(lons_mainbody), 50)
    
    # Compute angular momentum vectors and normalise
    L = np.cross(xfs_mainbody, vfs_mainbody, axis=1)
    Lmag = np.linalg.norm(L, axis=1)
    L_norm = L.T/Lmag
    
    # Slice lons_mainbody into bins
    bin_indices = np.digitize(lons_mainbody, lon_bins)
    # Create a mask array for lon bins
    mask = np.zeros((len(lons_mainbody), len(lon_bins) - 1), dtype=bool)
    for i in range(1, len(lon_bins)):
        mask[:, i - 1] = (bin_indices == i)

    # Calculate median angular momentum vector and pole track vector for each bin
    Lnorm_bins = np.array([np.nanmedian(L_norm.T[m], axis=0) for m in mask.T])[1:]
    
    xs_bins = np.array([np.nanmedian(xfs_mainbody[m], axis=0) for m in mask.T])
    J_bins = np.array([np.cross(xs_bins[i], xs_bins[i+1]) for i in range(len(xs_bins) - 1)])
    Jmag_bins = np.linalg.norm(J_bins, axis=1)
    Jnorm_bins = (J_bins.T / Jmag_bins).T

    #Calculate the angluar separation by the dot product and arccos()
    L_dot_J_bins = np.einsum('ij,ij->i', Jnorm_bins, Lnorm_bins) 
    
    pm_angles_rad = np.arccos(L_dot_J_bins) * u.rad
    pm_angles_deg = pm_angles_rad.to(u.deg)
    
    med_pm_angle = np.nanmedian(pm_angles_deg)
    
    return med_pm_angle


def galactic_coords(p, v):
    
    galcen_v_sun = (11.1, 245, 7.3)*u.km/u.s
    galcen_distance = 8.249*u.kpc
    
    # positions = p + Model.expansion_centres(0.)[:3]
    # velocities = v + Model.expansion_centre_velocities(0.)[:3]
    
    posvel_gc = SkyCoord(x=p[:,0]*u.kpc, y=p[:,1]*u.kpc, z=p[:,2]*u.kpc,
                         v_x=v[:,0]*u.km/u.s, v_y=v[:,1]*u.km/u.s, v_z=v[:,2]*u.km/u.s ,
                         frame='galactocentric', galcen_distance=galcen_distance, galcen_v_sun=galcen_v_sun)
    posvel_galactic = posvel_gc.transform_to('galactic')
    posvel_galactic_rc = gc.reflex_correct(posvel_galactic)
    l, b, d = np.nanmedian(posvel_galactic_rc.l), np.nanmedian(posvel_galactic_rc.b), np.nanmedian(posvel_galactic_rc.distance)
    pm_l_cosb, pm_b, rvs = np.nanmedian(posvel_galactic_rc.pm_l_cosb), np.nanmedian(posvel_galactic_rc.pm_b), np.nanmedian(posvel_galactic_rc.radial_velocity)
    
    sigma_rv = np.nanstd(posvel_galactic_rc.radial_velocity)
    
    return l.value, b.value, d.value, pm_l_cosb.value, pm_b.value, rvs.value, sigma_rv.value


def write_pltoutputs_hdf5(outpath, filename, 
                         l_gc, b_gc, ds, pm_l_cosb_gc, pm_b_gc, vlos, sigma_los,
                         peris, apos,
                         widths, lengths, track_deform, lmc_sep,
                         lons, lats, pm_misalign, loc_veldis,
                         pole_b, pole_l,
                         pole_b_dis, pole_l_dis,
                         masses, energy, Ekinetic, Ls, Lzs):
    print("* Writing data for potential {}".format(filename))
    hf = h5py.File(outpath + filename + ".hdf5", 'w')
    hf.create_dataset('l_gc', data=l_gc)
    hf.create_dataset('b_gc', data=b_gc)
    hf.create_dataset('ds', data=ds)
    hf.create_dataset('pm_l_cosb', data=pm_l_cosb_gc)
    hf.create_dataset('pm_b', data=pm_b_gc)
    hf.create_dataset('vlos', data=vlos)
    hf.create_dataset('sigmavlos', data=sigma_los)
                      
    hf.create_dataset('pericenter',data=peris)
    hf.create_dataset('apocenter', data=apos)
                
    hf.create_dataset('lengths', data=lengths)
    hf.create_dataset('widths', data=widths)
    hf.create_dataset('track_deform', data=track_deform)
    hf.create_dataset('lmc_sep', data=lmc_sep)
    hf.create_dataset('lons', data=lons)
    hf.create_dataset('lats', data=lats)
    # hf.create_dataset('av_lon', data=av_lon)
    # hf.create_dataset('av_lat', data=av_lat)
    hf.create_dataset('loc_veldis', data=loc_veldis)
    hf.create_dataset('pm_misalignment', data=pm_misalign)
                      
    hf.create_dataset('pole_l', data=pole_l)
    hf.create_dataset('pole_b', data=pole_b)                 
    hf.create_dataset('sigma_pole_l', data=pole_l_dis)
    hf.create_dataset('sigma_pole_b', data=pole_b_dis)
                      
    hf.create_dataset('mass', data=masses)           
    hf.create_dataset('energies', data=energy)
    hf.create_dataset('Eks', data=Ekinetic)
    hf.create_dataset('L', data=Ls)
    hf.create_dataset('Lz', data=Lzs)
    hf.close()

###----------------------------------------------------------------------------------
### Running the script - change potential to desired run
###----------------------------------------------------------------------------------
    
l_gc, b_gc, ds, pm_l_cosb_gc, pm_b_gc, vlos, sigma_los = [], [], [], [], [], [], []
peris, apos = [], []
widths, lengths, track_deforms, lmc_sep = [], [], [], []
lons, lats = [], []
pm_angles = []
# av_lon, av_lat = [], []
loc_veldis = []
pole_b, pole_l = [], []
pole_b_dis, pole_l_dis = [], []
masses = []
energy, Eks, Ls, Lzs = [], [], [], []

path = '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/1024-dthalfMyr-10rpmin-75ramax/'
# potential = 'rigid-mw.hdf5'
# potential = 'static-mw.hdf5'
# potential = 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5' 
# potential = 'Full-MWhalo-MWdisc-LMC.hdf5' 
potential = 'full-MWhalo-full-MWdisc-no-LMC.hdf5' 


Nstreams = 1024 
for i in range(Nstreams):
    data_path = pathlib.Path(path) / potential 
    with h5py.File(data_path,'r') as file:
        
        if i ==0:
            filename_end = file['stream_{}'.format(i)]['potential'][()].decode('utf-8')
        
        pos = np.array(file['stream_{}'.format(i)]['positions'])[-1]
        vel = np.array(file['stream_{}'.format(i)]['velocities'])[-1]
        l, b, d,  pm_l_cosb, pm_b, rvs, sigma_rv = galactic_coords(pos, vel)
        l_gc.append(l), b_gc.append(b), ds.append(d)
        pm_l_cosb_gc.append(pm_l_cosb), pm_b_gc.append(pm_b), vlos.append(rvs)
        sigma_los.append(sigma_rv)

        peris.append(np.array(file['stream_{}'.format(i)]['pericenter']))
        apos.append(np.array(file['stream_{}'.format(i)]['apocenter']))
        lmc_sep.append(np.array(file['stream_{}'.format(i)]['lmc_sep']))
        
        widths.append(np.array(file['stream_{}'.format(i)]['widths']))
        lengths.append(np.array(file['stream_{}'.format(i)]['lengths']))
        track_deforms.append(np.array(file['stream_{}'.format(i)]['track_deform']))
  
        # av_lon.append(np.array(file['stream_{}'.format(i)]['av_lon']))
        # av_lat.append(np.array(file['stream_{}'.format(i)]['av_lat']))
        
        lon, lat = lons_lats(pos, vel)
        lons.append(lon) 
        lats.append(lat)
        loc_veldis.append(local_veldis(lon, vel))
        
        pm_angle = pm_misalignment(lon, pos, vel)
        pm_angles.append(pm_angle.value)
        
        pole_b.append(np.array(file['stream_{}'.format(i)]['pole_b']))
        pole_l.append(np.array(file['stream_{}'.format(i)]['pole_l']))
        pole_b_dis.append(np.nanstd(np.array(file['stream_{}'.format(i)]['pole_b'])))
        pole_l_dis.append(np.nanstd(np.array(file['stream_{}'.format(i)]['pole_l'])))
        
        masses.append(np.array(file['stream_{}'.format(i)]['progenitor-mass']))
        energy.append(np.array(file['stream_{}'.format(i)]['energies'])[-1, 0])
        Ls.append(np.array(file['stream_{}'.format(i)]['L'])[-1, 0])
        Lzs.append(np.array(file['stream_{}'.format(i)]['Lz'])[-1, 0])
        
        Ek_prog = (.5 * np.linalg.norm(vel[0], axis=0)**2)
        Eks.append(Ek_prog)
           
out_path = '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/plotting_data/1024-dthalfMyr-10rpmin-75ramax/'

write_pltoutputs_hdf5(out_path, filename_end,
                      l_gc, b_gc, ds, pm_l_cosb_gc, pm_b_gc, vlos, sigma_los,
                      peris, apos,
                     widths, lengths, track_deforms, lmc_sep,
                     lons, lats, pm_angles, loc_veldis,
                     pole_b, pole_l,
                     pole_b_dis, pole_l_dis,
                     masses, energy, Eks, Ls, Lzs)
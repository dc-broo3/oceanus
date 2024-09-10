import agama, numpy as np, scipy.integrate, scipy.special
import gala.potential as gp
import gala.dynamics as gd
import gala.integrate as gi
import gala.units as gu
import matplotlib, matplotlib.pyplot as plt
import astropy.units as u
import os
import os.path
import yaml
import h5py
import sys
from argparse import ArgumentParser
import pathlib
from scipy.spatial.transform import Rotation

print("setting mass units for agama and gala...")
mass_unit =232500
agama.setUnits(length=1, velocity=1, mass=mass_unit)
timeUnitGyr = agama.getUnits()['time'] / 1e3

usys = gu.UnitSystem(u.kpc, 977.79222168*u.Myr, mass_unit*u.Msun, u.radian, u.km/u.s)

print("loading agama MW--LMC model from Eugene's Tango for Three paper...")
pot_frozen   = agama.Potential('../analysis/potentials_triax/potential_frozen.ini')   # fixed analytic potentials
pot_evolving = agama.Potential('../analysis/potentials_triax/potential_evolving.ini') # time-dependent multipole potentials

def energies_angmom(xs, vs):
    
    """
    calculate the energies and angular momenta of particles for a given time snapshot. Galactocentric frame.
    """
    # Kinetic energy
    Ek = (.5 * np.linalg.norm(vs, axis=1)**2) * (u.km/u.s)**2
    # Potential energy
    Ep = pot_frozen.potential(xs) * (u.km/u.s)**2
    
    E = Ek + Ep
    # Angular momentum
    L = np.linalg.norm(np.cross(xs, vs), axis=1) * (u.kpc*u.km/u.s)
    Lz = np.cross(xs[:,0:2], vs[:,0:2]) * (u.kpc*u.km/u.s)
    Lx = np.cross(xs[:,1:], vs[:,1:]) * (u.kpc*u.km/u.s)

    Ly = (L**2 - Lx**2 - Lz**2)**0.5
    
    return E, L, Lx, Ly, Lz

def orbpole(xs, vs):
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
    lower_value = np.nanpercentile(lons, 5)
    upper_value = np.nanpercentile(lons, 95)
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
    lower_value = np.nanpercentile(lons, 5)
    upper_value = np.nanpercentile(lons, 95)
    # Filter lons_mainbody
    lons_mainbody = lons[(lons >= lower_value) & (lons <= upper_value)]
    lats_mainbody = lats[(lons >= lower_value) & (lons <= upper_value)] 
    # Create bins
    lon_bins, Delta_phi1 = np.linspace(np.nanmin(lons_mainbody), np.nanmax(lons_mainbody), 50, retstep=True)
    # Slice lons_mainbody into bins
    bin_indices = np.digitize(lons_mainbody, lon_bins)
    # Create a mask array
    mask = np.zeros((len(lons_mainbody), len(lon_bins) - 1), dtype=bool)
    for i in range(1, len(lon_bins)):
        mask[:, i - 1] = (bin_indices == i)

    # Calculate width for each bin
    local_width = np.array([np.nanstd(lats_mainbody[m]) for m in mask.T])
    dev_bins = np.array([np.abs(np.nanmedian(lats_mainbody[m])) for m in mask.T])

    # Calculate gradient of the deviation for adjacent bins)
    Delta_dev_bins = np.diff(dev_bins)
    ddev_dphi1 = Delta_dev_bins / Delta_phi1

    return np.nanmedian(local_width), np.nanmedian(dev_bins), np.percentile(ddev_dphi1, 95)

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

def createICforParticleSpray(pot_host, orb_sat, mass_sat, gala_modified=True):
    """
    Create initial conditions for particles escaping through Largange points,
    using the method of Fardal+2015
    Arguments:
        pot_host:  an instance of agama.Potential.
        orb_sat:   the orbit of the satellite, an array of shape (N, 6).
        mass_sat:  the satellite mass (a single number or an array of length N).
        gala_modified:  if True, use modified parameters as in Gala, otherwise the ones from the original paper.
    Return:
        initial conditions for stream particles, an array of shape (2*N, 6) - 
        two points for each point on the original satellite trajectory.
    """
    N = len(orb_sat)
    x, y, z, vx, vy, vz = orb_sat.T
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = (x*x + y*y + z*z)**0.5
    L = (Lx*Lx + Ly*Ly + Lz*Lz)**0.5
    # rotation matrices transforming from the host to the satellite frame for each point on the trajectory
    R = np.zeros((N, 3, 3))
    R[:,0,0] = x/r
    R[:,0,1] = y/r
    R[:,0,2] = z/r
    R[:,2,0] = Lx/L
    R[:,2,1] = Ly/L
    R[:,2,2] = Lz/L
    R[:,1,0] = R[:,0,2] * R[:,2,1] - R[:,0,1] * R[:,2,2]
    R[:,1,1] = R[:,0,0] * R[:,2,2] - R[:,0,2] * R[:,2,0]
    R[:,1,2] = R[:,0,1] * R[:,2,0] - R[:,0,0] * R[:,2,1]
    # compute  the second derivative of potential by spherical radius
    der = pot_host.forceDeriv(orb_sat[:,0:3])[1]
    d2Phi_dr2 = -(x**2  * der[:,0] + y**2  * der[:,1] + z**2  * der[:,2] +
                  2*x*y * der[:,3] + 2*y*z * der[:,4] + 2*z*x * der[:,5]) / r**2
    # compute the Jacobi radius and the relative velocity at this radius for each point on the trajectory
    Omega = L / r**2
    rj = (agama.G * mass_sat / (Omega**2 - d2Phi_dr2))**(1./3)
    vj = Omega * rj
    # assign positions and velocities (in the satellite reference frame) of particles
    # leaving the satellite at both lagrange points.
    rj = np.repeat(rj, 2) * np.tile([1, -1], N)
    vj = np.repeat(vj, 2) * np.tile([1, -1], N)
    mean_x  = 2.0
    disp_x  = 0.5 if gala_modified else 0.4
    disp_z  = 0.5
    mean_vy = 0.3
    disp_vy = 0.5 if gala_modified else 0.4
    disp_vz = 0.5
    rx  = np.random.normal(size=2*N) * disp_x + mean_x
    rz  = np.random.normal(size=2*N) * disp_z * rj
    rvy =(np.random.normal(size=2*N) * disp_vy + mean_vy) * vj * (rx if gala_modified else 1)
    rvz = np.random.normal(size=2*N) * disp_vz * vj
    rx *= rj
    ic_stream = np.tile(orb_sat, 2).reshape(2*N, 6)
    ic_stream[:,0:3] += np.einsum('ni,nij->nj',
        np.column_stack([rx,  rx*0, rz ]), np.repeat(R, 2, axis=0))
    ic_stream[:,3:6] += np.einsum('ni,nij->nj',
        np.column_stack([rx*0, rvy, rvz]), np.repeat(R, 2, axis=0))
    return ic_stream

def createStreamParticleSpray(time_total, num_particles, pot_host, posvel_sat, mass_sat, gala_modified=True):
    
    # integrate the orbit of the progenitor from its present-day posvel (at time t=0)
    # back in time for an interval time_total, storing the trajectory at num_steps points
    print(posvel_sat.shape)
    time_sat, orbit_sat = agama.orbit(potential=pot_host, ic=posvel_sat,
        time=time_total, trajsize=num_particles//2)
    # reverse the arrays to make them increasing in time
    time_sat  = time_sat [::-1]
    orbit_sat = orbit_sat[::-1]

    # at each point on the trajectory, create a pair of seed initial conditions
    # for particles released at Lagrange points
    ic_stream = createICforParticleSpray(pot_host, orbit_sat, mass_sat, gala_modified=gala_modified)
    time_seed = np.repeat(time_sat, 2)
    xv_stream = np.vstack(agama.orbit(potential=pot_host,
        ic=ic_stream, time=-time_seed, timestart=time_seed, trajsize=1)[:,1])
    return time_sat, orbit_sat, xv_stream, ic_stream

def lagrange_cloud_strip_adT(params, overwrite):  
    
    inpath, snapname, outpath, filename, \
    fc, Mprog, a_s, pericenter, apocenter, Tbegin, Tfinal, num_particles = params
    
    print("Parameters present!")
    fullfile_path = pathlib.Path(outpath) / pathlib.Path(filename + '.hdf5')

    if fullfile_path.exists() and not overwrite:
        print("Skipping as already exists and do not want to overwrite...")
        return 
    
    print("Begining stream generation process...")
    mass_sat   = Mprog/mass_unit  # in Msun
    time_total = Tbegin/timeUnitGyr.value  # in time units (0.978 Gyr)
    ts, orbit_sat, xv_stream, ic_stream = createStreamParticleSpray(time_total, 
                                                                    num_particles, 
                                                                    pot_frozen, 
                                                                    np.array(fc),
                                                                    mass_sat)
    
    xv_prog_stream = np.concatenate([orbit_sat[0].reshape(1,6), xv_stream])
    print("stream generated!")
    xs, vs = xv_prog_stream[:,:3], xv_prog_stream[:,3:6] # only have the final snapshot
    
    print("calculating energies, angular momenta, velocity dispersion, orbital poles...")
    Es = np.full(shape=(1, xs.shape[0]), fill_value=np.nan)
    Ls = np.full(shape=(1, xs.shape[0]), fill_value=np.nan)
    Lxs = np.full(shape=(1, xs.shape[0]), fill_value=np.nan)
    Lys = np.full(shape=(1, xs.shape[0]), fill_value=np.nan)
    Lzs = np.full(shape=(1, xs.shape[0]), fill_value=np.nan)
    gls = np.full(shape=(1, xs.shape[0]), fill_value=np.nan)
    gbs = np.full(shape=(1, xs.shape[0]), fill_value=np.nan)
    
    lons, lats = lons_lats(xs, vs)
    sigma_v = local_veldis(lons, vs)
    length = np.nanpercentile(lons, 95) - np.nanpercentile(lons, 5)
    width, track_deform, grad_track_deform = widths_deforms(lons, lats)
    pm_angle = pm_misalignment(lons, xs, vs)
    
    for i in range(1):
        Es[i], Ls[i], Lxs[i], Lys[i], Lzs[i] = energies_angmom(xs, vs)
        gls[i], gbs[i] = orbpole(xs, vs)

    write_stream_hdf5(outpath, filename, xs, vs, ts,
                      sigma_v, length, width, track_deform, grad_track_deform, pm_angle, fc, Mprog, a_s, 
                      pericenter, apocenter,
                     Es, Ls, Lxs, Lys, Lzs, gls, gbs)

def readparams(paramfile):
    """
    Read in the stream model parameters
    """
    
    print("Opening parameter yaml file...")
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
    num_particles = d["num_particles"]
    
    print("Read yaml contents and returning function...")

    return [inpath, snapname, outpath, outname, prog_ics ,prog_mass, prog_scale, pericenter, apocenter, Tbegin, Tfinal, num_particles]

def write_stream_hdf5(outpath, filename, positions, velocities, times, 
                      sigma_v, length, width, track_deform, grad_track_deform, pm_angles, progics, progmass, progscale, 
                      pericenter, apocenter,
                     energies, Ls, Lxs, Lys, Lzs, gls, gbs):
    """
    Write stream into an hdf5 file
    
    """
    tmax = 1
    particlemax = positions.shape[0]
    
    print("* Writing stream: {}".format(filename))
    hf = h5py.File(outpath + filename + ".hdf5", 'w')
    hf.create_dataset('positions', data=positions, shape=(tmax, particlemax, 3))
    hf.create_dataset('velocities', data=velocities, shape=(tmax, particlemax, 3))
    hf.create_dataset('times', data=times)
    
    hf.create_dataset('energies', data=energies)
    hf.create_dataset('L', data=Ls)
    hf.create_dataset('Lx', data=Lxs)
    hf.create_dataset('Ly', data=Lys)
    hf.create_dataset('Lz', data=Lzs)
    
    hf.create_dataset('loc_veldis', data=sigma_v)
    hf.create_dataset('lengths', data=length)
    hf.create_dataset('widths', data=width)
    hf.create_dataset('track_deform', data=track_deform)
    hf.create_dataset('grad_track_deform', data=grad_track_deform)
    hf.create_dataset('pm_misalignment', data=pm_angles) 

    hf.create_dataset('pole_l', data=gls)
    hf.create_dataset('pole_b', data=gbs)
    
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
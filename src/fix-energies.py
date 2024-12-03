import numpy as np
import h5py
import astropy.units as u
import pathlib

from mwlmc import model as mwlmc_model
Model = mwlmc_model.MWLMC()

_, MWHcoeffs = Model.return_mw_coefficients()
MWHcoeffs = np.array(MWHcoeffs)
MWHcoeffs[:,0] = MWHcoeffs[:,0][0] 
MWHcoeffs[:,1:] = MWHcoeffs[:,1:]*0
Model.install_mw_coefficients(MWHcoeffs)

MWDfloats, MWDctmp, MWDstmp = Model.return_disc_coefficients()
MWDctmp, MWDstmp = np.array(MWDctmp), np.array(MWDstmp)
MWDctmp[:,0], MWDstmp[:,0] = MWDctmp[:,0][0], MWDstmp[:,0][0]
MWDctmp[:,1:], MWDstmp[:,1:] = MWDctmp[:,1:]*0, MWDstmp[:,1:]*0
Model.install_disc_coefficients(MWDctmp,MWDstmp)

_, LMCcoeffs = Model.return_lmc_coefficients()
LMCcoeffs = np.array(LMCcoeffs)
LMCcoeffs *= 0 
Model.install_lmc_coefficients(LMCcoeffs)

def energies_(t, xs, vs, mwdflag, mwhflag, lmcflag, motion):
    
    """
    calculate the energies and angular momenta of particles for a given time snapshot. Galactocentric frame.
    """
    # Kinetic energy
    Ek = (.5 * np.linalg.norm(vs, axis=1)**2) * (u.km/u.s)**2
    # Potential energy
    x0 = np.array(Model.expansion_centres(t))
    if motion == False:
        x0 *= 0 
    
    x_lmc_GC = x0[6:9] - x0[:3]
    
    pot_disk = Model.mwd_fields(t, 
                                xs[:,0],
                                xs[:,1],
                                xs[:,2],
                                mwdflag)[:,4]
    pot_halo = Model.mwhalo_fields(t, 
                                   xs[:,0],
                                   xs[:,1],
                                   xs[:,2],
                                   mwhflag)[:,4]
    pot_lmc = Model.lmc_fields(t, 
                               xs[:,0] - x_lmc_GC[0],
                               xs[:,1] - x_lmc_GC[1],
                               xs[:,2] - x_lmc_GC[2],
                               lmcflag)[:,4]
    Ep = (pot_disk + pot_halo + pot_lmc) * (u.km/u.s)**2
    E = Ek + Ep

    return E

# Nstreams= 1024
# for i in range(Nstreams):
#     stream = "stream_{}.hdf5".format(i)
#     with h5py.File("/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/em-mwh-full-mwd-full-lmc/" + stream, 'a') as file:
#         mwdflag, mwhflag, lmcflag, motion = 63, 0, 63, True
        
#         xs_snaps = file["positions"]
#         vs_snaps = file["velocities"]
#         ts =  np.array(file["times"])[2::2]
    
#         ### 10kpc cut samples
#         ts_snaps = np.flip(np.flip(ts, axis=0)[::1000])
        
#         Es = np.full(shape=(xs_snaps.shape[0], xs_snaps.shape[1]), fill_value=np.nan)
#         for j in range(xs_snaps.shape[0]):
#             Es[j] = energies_(ts_snaps[j], xs_snaps[j], vs_snaps[j], mwdflag, mwhflag, lmcflag, motion)

#         if 'energies' in file:
#             del file['energies']
            
#         # Write the datasets
        # file.create_dataset('energies', data=Es)
    
path = '/mnt/ceph/users/rbrooks/oceanus/analysis/stream-runs/combined-files/1024-dthalfMyr-10rpmin-75ramax/'
potential = 'rigid-mw.hdf5'

Nstreams = 1024 
for i in range(Nstreams):
    data_path = pathlib.Path(path) / potential 
    
    mwdflag, mwhflag, lmcflag, motion = 0, 0, 0, False
    with h5py.File(data_path,'r+') as file:
      
        xs_snaps = file['stream_{}'.format(i)]["positions"]
        vs_snaps = file['stream_{}'.format(i)]["velocities"]
        ts =  np.array(file['stream_{}'.format(i)]["times"])[2::2]
        ts_snaps = np.flip(np.flip(ts, axis=0)[::1000])
        
        saved_Es = np.array(file['stream_{}'.format(i)]["energies"])
        
        Es = np.full(shape=(xs_snaps.shape[0], xs_snaps.shape[1]), fill_value=np.nan)
    
        for j in range(xs_snaps.shape[0]):
            Es[j] = energies_(ts_snaps[j], xs_snaps[j], vs_snaps[j], mwdflag, mwhflag, lmcflag, motion)
        
        if 'energies' in file['stream_{}'.format(i)]:
            del file['stream_{}'.format(i)]['energies']
            
        # Write the datasets
        file['stream_{}'.format(i)].create_dataset('energies', data=Es)
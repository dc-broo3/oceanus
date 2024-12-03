import numpy as np
import astropy.units as u
import yaml
import h5py
from astropy.constants import G
import os.path
import sys

def what_potential(filename):
    with h5py.File(filename, 'r') as file:
        potential = file['potential'][()].decode('utf-8')  # Retrieve the value associated with the 'potential' dataset
    return "combined-files/16384-dt1Myr/{}.hdf5".format(potential)   
    # return "combined-files/128-dt1Myr-5kpcrpmin/{}.hdf5".format(potential)   

# Function to combine HDF5 files into a single master HDF5 file
def combine_hdf5_files(input_dir, input_sub, output_file):
    output_path = input_dir + output_file
    if os.path.exists(output_path):
        master = h5py.File(output_path, 'a')
    else:
        master = h5py.File(output_path, 'w')    
    with master as master_file:
        for filename in os.listdir(input_dir + input_sub):  
            if filename.endswith(".hdf5"):
                input_file = os.path.join(input_dir + input_sub, filename)
                with h5py.File(input_file, 'r') as f:
                    group_name = os.path.splitext(filename)[0]
                    f.copy('/', master_file, name=group_name)
        master.close()
                    
# Input directory containing HDF5 files
# ending = "full-mwh-full-mwd-no-lmc/"
# ending = "full-mwh-full-mwd-full-lmc/"
# ending = "static-mw/"
ending = "rigid-mw/"

input_ =  "/mnt/home/rbrooks/ceph/oceanus/analysis/stream-runs/" 
# Output master HDF5 file
output_ = what_potential(input_ + ending + "stream_0.hdf5") #all streams will have same label, saves doing same steps for 10^X streams

# Agama streams
# input_ =  "/mnt/home/rbrooks/ceph/oceanus/analysis/stream-runs/" 
# ending = "high-vel-dis/agama-mw/"
# output_ = "combined-files/1024-agama.hdf5"

# Combine HDF5 files
combine_hdf5_files(input_, ending, output_)
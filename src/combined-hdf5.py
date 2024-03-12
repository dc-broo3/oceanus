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
    return "combined-files/{}.hdf5".format(potential)   

    
# Function to combine HDF5 files into a single master HDF5 file
def combine_hdf5_files(input_dir, input_sub, output_file):
    output_path = input_ + output_file
    if os.path.exists(output_path):
        master = h5py.File(output_path, 'a')
    else:
        master = h5py.File(output_path, 'w')
            
    with master as master_file:
        
        for filename in os.listdir(input_dir + input_sub):
            # print(input_dir + input_sub)
            
            if filename.endswith(".hdf5"):
                input_file = os.path.join(input_dir + input_sub, filename)
                # print(input_file)
    
                with h5py.File(input_file, 'r') as f:
                    group_name = os.path.splitext(filename)[0]
                    f.copy('/', master_file, name=group_name)

        master.close()
        
    Remove all files in the directory after processing
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith(".hdf5"):
            os.remove(file_path)

                    
# Input directory containing HDF5 files
ending = "mdq-mwh-full-mwd-full-lmc/"
input_ =  "/mnt/home/rbrooks/ceph/oceanus/analysis/stream-runs/" 

# Output master HDF5 file
output_ = what_potential(input_ + ending + "stream_0.hdf5") #all streams will have same label, saves doing same steps for 10^X streams
# output_ = "combined-files/full-MWhalo-full-MWdisc-no-LMC.hdf5"
# Combine HDF5 files

print(output_)
combine_hdf5_files(input_, ending, output_)
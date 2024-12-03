import h5py
import numpy as np
file_name = 'rigid-mw.hdf5'
# file_name ='static-mw.hdf5'
# file_name = 'rm-MWhalo-full-MWdisc-full-LMC.hdf5'
# file_name = 'em-MWhalo-full-MWdisc-full-LMC.hdf5'    
# file_name = 'md-MWhalo-full-MWdisc-full-LMC.hdf5' 
# file_name = 'mq-MWhalo-full-MWdisc-full-LMC.hdf5'   
# file_name = 'mdq-MWhalo-full-MWdisc-full-LMC.hdf5'  
# file_name ='full-MWhalo-full-MWdisc-no-LMC.hdf5' 
# file_name = 'full-MWhalo-full-MWdisc-full-LMC.hdf5'


# Open the existing HDF5 file in read mode
with h5py.File('../analysis/stream-runs/combined-files/16384-dt1Myr/' + file_name, 'r') as input_file:
    # Create a new HDF5 file to store the selected datasets
    with h5py.File('../analysis/stream-runs/combined-files/zenodo-data/' + file_name, 'w') as output_file:
        # Iterate over all the groups in the input file
        for group_name in input_file.keys():
            group = input_file[group_name]  # Access each group
            
            # Define the datasets you want to keep from this group
            datasets_to_keep = ['positions', 'velocities', 'progenitor-ics', 'progenitor-mass', 'progenitor-scale']  # example datasets you want
            
            # Create a new group in the output file with the same group name
            new_group = output_file.create_group(group_name)
            
           # Iterate over the datasets within the group
            for dataset_name in datasets_to_keep:
                if dataset_name in group.keys():  # Check if the dataset exists
                    dataset = group[dataset_name]
                    
                    # Check if the dataset is 3D
                    if len(dataset.shape) == 3:
                        # If 3D, extract the final 2D "slice" (along the first dimension)
                        final_slice = dataset[-1, :, :]  # Last "row" in the first dimension
                         # Cast to 32-bit floating point
                        final_slice_32bit = final_slice.astype(np.float32)

                        # Copy the final 2D slice to the new group in the output file
                        new_group.create_dataset(dataset_name, data=final_slice_32bit)
                        print(f"Copied final slice of dataset: {dataset_name} from group: {group_name}")
                    else:
                        # If not 3D, just copy the entire dataset
                        data = dataset[...]  # Load the whole dataset
                        data_32bit = data.astype(np.float32)
                        new_group.create_dataset(dataset_name, data=data_32bit)
                        print(f"Copied entire dataset: {dataset_name} from group: {group_name}")

        print("Selected data has been copied to the new file.")
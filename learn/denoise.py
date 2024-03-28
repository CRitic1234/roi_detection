import numpy as np
import nrrd
from glob import glob
import scipy.ndimage as ndi
import os

def skull_id(labels_out):
    # Reshape labels_out to a 1D array
    labels_out = labels_out.reshape((1, -1))
    labels_out = labels_out[0, :]
    # Find unique labels and their counts
    label, counts = np.unique(labels_out, return_counts=True)
    # Get the largest label (most common)
    largest_label = label[np.argmax(counts)]
    return largest_label

if __name__ == '__main__':
    # Directory of original nrrd files
    data_dir = "C:/Users/USER/OneDrive/Desktop/randomdata"
    data_list = glob('{}/*.nrrd'.format(data_dir))
    # Directory to save the cleaned nrrd files
    save_dir = "C:/Users/USER/OneDrive/Desktop/learn/cleaned"
    
    for data_file in data_list:
        print('Current data to clean:', data_file)
        # Read nrrd file: data is the skull volume, header is the nrrd header
        data, header = nrrd.read(data_file)
        # Get all the connected components in data
        labels_out, _ = ndi.label(data.astype('int32'))
        # Select the largest connected component (skull)
        skull_label = skull_id(labels_out)
        # Keep only the largest connected component (remove other components)
        skull = (labels_out == skull_label)
        # File name of the cleaned skull
        filename = os.path.join(save_dir, os.path.basename(data_file)[:-5] + '_cleaned.nrrd')
        print('Writing the cleaned skull to nrrd...')
        nrrd.write(filename, skull.astype('uint8'), header=header)
        print('Writing done...')

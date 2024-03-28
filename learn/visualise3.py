import numpy as np
import nrrd
import matplotlib.pyplot as plt
from glob import glob

# Directory containing the NRRD files
data_dir = 'C:/Users/USER/OneDrive/Desktop/learn/cleaned'

# Get the list of NRRD files
data_files = glob(data_dir + '/*.nrrd')

# Read and visualize each NRRD file
for i, data_file in enumerate(data_files[:10]):  # Selecting the first 11 files
    print(f'Visualizing file {i + 1}: {data_file}')
    
    # Load the NRRD file
    data, header = nrrd.read(data_file)

    # Display a slice of the volume
    plt.subplot(2, 5, i + 1)  # Adjust the subplot layout as needed
    plt.imshow(data[:, :, 50], cmap='gray')  # Adjust the slice index as needed
    plt.title(f'File {i + 1}')
    plt.axis('off')

plt.tight_layout()  # Adjust the layout
plt.show()

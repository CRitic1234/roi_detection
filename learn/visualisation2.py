import numpy as np
import nrrd
import matplotlib.pyplot as plt

# Load the NRRD file
data, header = nrrd.read('C:/Users/USER/OneDrive/Desktop/randomdata/000.nrrd')

# Display a slice of the volume
plt.imshow(data[:, :, 50], cmap='gray')  # Adjust the slice index as needed
plt.axis('off')
plt.show()

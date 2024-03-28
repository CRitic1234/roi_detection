import numpy as np
import nrrd
from glob import glob
import random

def generate_cube_defect(volume, defect_size):
    x_max, y_max, z_max = volume.shape
    x = random.randint(defect_size, x_max - defect_size)
    y = random.randint(defect_size, y_max - defect_size)
    z = random.randint(int(z_max / 2), z_max - defect_size)  # Avoid the lower part of the skull
    volume[x - defect_size//2:x + defect_size//2, 
           y - defect_size//2:y + defect_size//2, 
           z:] = 0  # Set the region inside the cube defect to 0
    return volume

def generate_sphere_defect(volume, defect_size):
    x_max, y_max, z_max = volume.shape
    x = random.randint(z_max + defect_size - z_max, x_max - (z_max + defect_size - z_max))
    y = random.randint(z_max + defect_size - z_max, y_max - (z_max + defect_size - z_max))
    z = random.randint(int(z_max / 2), z_max - (z_max + defect_size - z_max))  # Avoid the lower part of the skull
    radius = defect_size // 2
    xx, yy, zz = np.ogrid[:x_max, :y_max, :z_max]
    sphere_mask = (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2 <= radius ** 2
    volume[sphere_mask] = 0  # Set the region inside the sphere defect to 0
    return volume

def save_defected_volume(volume, filename):
    nrrd.write(filename, volume.astype(np.uint8))  # Assuming the volume contains integer values
    
if __name__ == "__main__":
    # Directory of the cleaned skull volumes
    print(1)
    cleaned_dir = "C:/Users/USER/OneDrive/Desktop/learn/cleaned"
    print(11)
    cleaned_files = glob(cleaned_dir + "*.nrrd")
    print(2)
    # Directory to save the defected volumes
    defected_dir = "C:/Users/USER/OneDrive/Desktop/learn/defected"
    
    # Generating defects for each cleaned volume
    print(10)
    
    for file in cleaned_files:
        print(3)
        print('Generating defects for:', file)
        volume, header = nrrd.read(file)
        
        # Choose between cube and sphere defects randomly
        defect_type = random.choice(["cube", "sphere"])
        
        if defect_type == "cube":
            defect_size = random.randint(10, 50)  # Adjust the range as per requirement
            volume = generate_cube_defect(volume, defect_size)
            print(4)
        else:
            defect_size = random.randint(10, 50)  # Adjust the range as per requirement
            volume = generate_sphere_defect(volume, defect_size)
            print(5)
        
        # Save the defected volume
        filename = defected_dir + file.split("/")[-1]
        save_defected_volume(volume, filename)
    print(6)
    print("Defected volumes saved successfully.")

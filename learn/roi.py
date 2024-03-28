import cc3d
import numpy as np
import nrrd
from glob import glob
import open3d as o3d
import math
import stl
from stl import mesh
import os
import sys
import scipy
from skimage import measure, morphology
import pymrt.geometry

def skull_id(labels_out):
    labels_out = labels_out.reshape((1, -1))
    labels_out = labels_out[0, :]
    label = np.unique(labels_out)
    hist, bin_edges = np.histogram(labels_out, bins=label)
    hist = np.ndarray.tolist(hist)
    hist_ = hist
    hist_ = np.array(hist_)
    hist.sort(reverse=True)
    idx = (hist_ == hist[1])
    idx = idx + 1 - 1
    idx_ = np.sum(idx * label[0:len(idx)])
    print('idx', idx_)
    return idx_

def find_mins_maxs(obj):
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    return minx, maxx, miny, maxy, minz, maxz

def re_sample(image, current_spacing, new_spacing):
    resize_factor = current_spacing / new_spacing
    new_shape = image.shape * resize_factor
    new_shape = np.round(new_shape)
    actual_resize_factor = new_shape / image.shape
    new_spacing = current_spacing / actual_resize_factor
    image_resized = scipy.ndimage.interpolation.zoom(image, actual_resize_factor)
    return image_resized, new_spacing

def generate_real_hole(data, radius, loc, loc1, idxxx):
    x_ = data.shape[0]
    y_ = data.shape[1]
    z_ = data.shape[2]
    full_masking = np.zeros(shape=(x_, y_, z_))

    masked_x = int(x_ * 1 / 3)
    masked_y = int(y_ * 1 / 3)
    masked_z = int(z_ / 3)

    cylinder1 = pymrt.geometry.cylinder((masked_x, masked_y, masked_z), int(z_), radius, 2, position=(1 / 6, 1 / 6, 1))
    cylinder2 = pymrt.geometry.cylinder((masked_x, masked_y, masked_z), int(z_), radius, 2, position=(1 / 6, 5 / 6, 1))
    cylinder3 = pymrt.geometry.cylinder((masked_x, masked_y, masked_z), int(z_), radius, 2, position=(5 / 6, 1 / 6, 1))
    cylinder4 = pymrt.geometry.cylinder((masked_x, masked_y, masked_z), int(z_), radius, 2, position=(5 / 6, 5 / 6, 1))

    cylinder1 = cylinder1 + 1 - 1
    cylinder2 = cylinder2 + 1 - 1
    cylinder3 = cylinder3 + 1 - 1
    cylinder4 = cylinder4 + 1 - 1

    cube = np.zeros(shape=(masked_x, masked_y, masked_z))
    cube[int((1 / 6) * masked_x):int((5 / 6) * masked_x), int((1 / 6) * masked_y):int((5 / 6) * masked_y),
         0:masked_z] = 1

    combined = cube + cylinder1 + cylinder2 + cylinder3 + cylinder4
    combined = (combined != 0)
    combined = combined + 1 - 1

    if idxxx == 1:
        full_masking[int((loc / 4) * x_):int((loc / 4) * x_) + masked_x,
                     int((1 / 2) * y_):int((1 / 2) * y_) + masked_y, z_ - masked_z:z_] = combined

    return full_masking

if __name__ == '__main__':
    # Directory of original nrrd files
    data_dir = "D:/skull-nrrd"
    data_list = glob('{}/*.nrrd'.format(data_dir))

    # Directory to save the cleaned nrrd file
    save_dir = "D:/skull-nrrd/cleaned/"

    for i in range(len(data_list)):
        print('current data to clean:', data_list[i])

        # Read nrrd file. data: skull volume. header: nrrd header
        data, header = nrrd.read(data_list[i])

        # Get all the connected components in data
        labels_out = cc3d.connected_components(data.astype('int32'))

        # Select the index of the second largest connected component 
        # in the data (the largest connected component is the background).
        skull_label = skull_id(labels_out)

        # Keep only the second largest connected components (and remove other components)
        skull = (labels_out == skull_label)
        skull = skull + 1 - 1

        # File name of the cleaned skull
        filename = save_dir + data_list[i][-10:-5] + '.nrrd'
        print('writing the cleaned skull to nrrd...')
        nrrd.write(filename, skull, header)
        print('writing done...')

from glob import glob
import numpy as np
import nrrd
from scipy.ndimage import zoom
import random
import pymrt.geometry

def generate_cude(size):
    for i in range(len(pair_list)):
        print('generating data:', pair_list[i])
        temp, header = nrrd.read(pair_list[i])

        full_masking = generate_hole_implants(temp, size)

        c_masking_1 = (full_masking == 1)
        c_masking_1 = c_masking_1 + 1 - 1

        defected_image = c_masking_1 * temp

        c_masking = (full_masking == 0)
        c_masking = c_masking + 1 - 1
        implants = c_masking * temp

        f1 = defected_dir + pair_list[i][-10:-5] + '.nrrd'
        f2 = implant_dir + pair_list[i][-10:-5] + '.nrrd'
        nrrd.write(f1, defected_image, header)
        nrrd.write(f1, defected_image, header)
        nrrd.write(f2, implants, header)

def generate_hole_implants(data, cube_dim):
    x_ = data.shape[0]
    y_ = data.shape[1]
    z_ = data.shape[2]
    full_masking = np.ones(shape=(x_, y_, z_))
    x = random.randint(int(cube_dim / 2), x_ - int(cube_dim / 2))
    y = random.randint(int(cube_dim / 2), y_ - int(cube_dim / 2))
    z = int(z_ * (3 / 4))
    cube_masking = np.zeros(shape=(cube_dim, cube_dim, z_ - z))
    full_masking[x - int(cube_dim / 2):x + int(cube_dim / 2), y - int(cube_dim / 2):y + int(cube_dim / 2),
                 z:z_] = cube_masking
    return full_masking

def generate_sphere_hole_implants(data, size):
    x_ = data.shape[0]
    y_ = data.shape[1]
    z_ = data.shape[2]
    z = int(z_ * (3 / 4))
    x = random.randint(z_ + size - z, x_ - (z_ + size - z))
    y = random.randint(z_ + size - z, y_ - (z_ + size - z))
    arr = sphere((x_, y_, z_ + size), z_ + size - z, (x, y, z))
    return arr

def generate_sphere(size1):
    for i in range(len(pair_list)):
        size = size1
        print('generating data:', pair_list[i])
        temp = nrrd.read(pair_list[i])[0]
        temp_ = np.zeros(shape=(temp.shape[0], temp.shape[1], temp.shape[2] + size))
        temp_[:, :, 0:temp.shape[2]] = temp
        arr = generate_sphere_hole_implants(temp, size)
        arr = (arr == 1)
        arr = arr + 1 - 1
        implants = arr * temp_
        arr = (arr == 0)
        arr = arr + 1 - 1
        defected_image = arr * temp_
        f1 = defected_dir + pair_list[i][-10:-5] + '.nrrd'
        f2 = implant_dir + pair_list[i][-10:-5] + '.nrrd'
        nrrd.write(f1, defected_image[:, :, 0:temp.shape[2]].astype('float64'))
        nrrd.write(f2, implants[:, :, 0:temp.shape[2]].astype('float64'))
        print(defected_image[:, :, 0:temp.shape[2]].shape)

def generate_real(temp, size, loc, loc1, idxxx):
    full_masking = generate_real_hole(temp, size, loc, loc1, idxxx)
    full_masking = (full_masking == 1)
    full_masking = full_masking + 1 - 1
    implants = full_masking * temp

    c_masking = (full_masking == 0)
    c_masking = c_masking + 1 - 1
    defected_image = c_masking * temp
    return implants, defected_image

if __name__ == "__main__":
    # Directory of the healthy skull
    pair_list = glob('{}/*.nrrd'.format('C:/Users/Jianning/Desktop'))

    defected_dir = 'C:/Users/Jianning/Desktop/1/'
    implant_dir = 'C:/Users/Jianning/Desktop/2/'

    generate_cude(128)
    # generate_sphere(20)

    data, header = nrrd.read('../sample_case.nrrd')
    defected_skull_dir = '../save_defected_skull_dir/'
    implant_dir = '../save_implant_dir/'
    f = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    for i in range(10):
        size = np.random.randint(6, 14, 1)[0]
        loc = np.random.randint(1, 3, 1)[0]
        loc1 = np.random.randint(1, 3, 1)[0]
        idxxx = np.random.randint(1, 10, 1)[0]
        print(idxxx)
        f1 = defected_skull_dir + 'sample_case_Defects' + f[i] + '.nrrd'
        f2 = implant_dir + 'sample_case_Implants' + f[i] + '.nrrd'
        implants, defected_image = generate_real(data, size, loc, loc1, idxxx)
        nrrd.write(f1, defected_image, header)
        nrrd.write(f2, implants, header)

# Python scripts for skull segmentation from CT scan
# Read nrrd files

import numpy as np
import nrrd
from glob import glob

if __name__ == '__main__':
    # directory of original nrrd files
    data_dir = "D:/skull-nrrd"
    data_list=glob('{}/*.nrrd'.format(data_dir))

    # Directory to save the segmented nrrd file
    save_dir = "D:/skull-nrrd/segmented/"

for i in range(len(data_list)):
    print('current data to segment:', data_list[i])

    # Read nrrd file. data: skull volume. header: nrrd header
    data, header = nrrd.read(data_list[i])

    # Set threshold, 100--max
    segmented_data = (data >= 100)
    segmented_data = segmented_data + 1 - 1

    # File name of the cleaned skull
    filename = save_dir + data_list[i][-10:-5] + '.nrrd'
    print('writing the cleaned skull to nrrd...')
    nrrd.write(filename, segmented_data, h)
    print('writing done...')


#     import open3d
# import matplotlib.pylab as plt
# import sys
# import numpy as np
# import nrrd
# import os
# import scipy
# from skimage import measure, morphology

if __name__ == '__main__':
    ct_data,ct_header=nrrd.read('000.nrrd')


    ct_spacing = np.asarray([ct_header['space directions'][0, 0],
                             ct_header['space directions'][1, 1],
                             ct_header['space directions'][2, 2]])

    ct_origin = np.asarray([ct_header['space origin'][0],
                            ct_header['space origin'][1],
                            ct_header['space origin'][2]])

    if ct_spacing[2] > 0:
        num_slices = int(180 / ct_spacing[2])
        ct_data = ct_data[:, :, -num_slices:]

    # resample data to uniform spacing of 1x1x1mm
    if ct_spacing[2] > 0:
        ct_data_resampled, _ = re_sample(ct_data, ct_spacing, new_spacing=[1, 1, 1])
    else:
        ct_data_resampled = ct_data

    skin_masked = ct_data_resampled


    skin_verts, skin_faces, skin_norm, skin_val = measure.marching_cubes_lewiner(skin_masked,step_size=1)
    skin_verts = skin_verts
    skin_points = skin_verts[:, [1, 0, 2]]
    # create mesh
    skin_mesh = open3d.geometry.TriangleMesh()
    skin_mesh.vertices = open3d.utility.Vector3dVector(skin_points)
    skin_mesh.triangles = open3d.utility.Vector3iVector(skin_faces)
    skin_mesh.compute_vertex_normals()

    # write mesh
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
    open3d.io.write_triangle_mesh('filemane' + '.stl', skin_mesh)
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Info)

# import numpy as np
# import nrrd
# from glob import glob
# import open3d as o3d
# import math
# import stl
# from stl import mesh
# import numpy

# import os
# import sys


if __name__ == '__main__':
    # where the STL files are stored
    base_dir='D:/skull-volume/TU_200/Segmentiert/TU_fertig/stl'
    data_list=glob('{}/*.stl'.format(base_dir))
    x=[]
    y=[]
    z=[]

    for i in range(len(data_list)):
    	main_body = mesh.Mesh.from_file(data_list[i])
        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(main_body)
        x.append(maxx-minx)
    	y.append(maxy-miny)
    	z.append(maxz-minz)


    x=np.array(x)
    y=np.array(y)
    z=np.array(z)

    print('x min',x.min())
    print('x max',x.max())


    print('y min',y.min())
    print('y max',y.max())

    print('z min',z.min())
    print('z max',z.max())





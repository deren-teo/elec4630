import os

from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np
import vtk

from tqdm import tqdm

### GLOBAL VARIABLES ###########################################################

# Path to the project root, i.e. "a2"
A2_ROOT = Path(__file__).parent.parent.resolve()

# Dinosaur bounding box coordinates
MIN_X = -180; MIN_Y = -80; MIN_Z =  20
MAX_X =   90; MAX_Y =  70; MAX_Z = 460

### TYPEDEFS ###################################################################

# Define an "image" type, which is more descriptive than "np.ndarray"
Image = np.ndarray

### DATA PARSING ###############################################################

def parse_projections(fp: Path) -> np.ndarray:
    '''
    Parse a text file containing 3x4 projection matrices into a 3D array.
    '''
    projn_array = []; tmp_array: np.ndarray = None; idx = 0

    with open(fp, 'r') as f:
        for line in f.readlines():
            line = line.strip()

            # Exit on finish condition
            if line.endswith('...'):
                break

            # Parse first line of array
            if line.startswith('P'):
                tmp_array = np.empty((3, 4))
                line = line.split('[')[1]

            # Parse data lines (incl. first line) of array
            if line.endswith(';'):
                line = line.replace(';', '').replace(']', '')
                tmp_array[idx] = np.array(list(map(float, line.split(' '))))
                idx += 1
                continue

            # Parse blank line after array
            if line == '':
                projn_array.append(tmp_array)
                idx = 0

    return np.array(projn_array)

### IMAGE PROCESSING ###########################################################

def clear_borders(img: Image) -> Image:
    '''
    Uses flood fill to clear white components conected to the border
    in a binary image.

    Adapted from: https://stackoverflow.com/a/65544104
    '''
    pad = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=255)
    h, w = pad.shape
    mask = np.zeros([h + 2, w + 2], np.uint8)
    img = cv.floodFill(pad, mask, (0,0), 0, (5), (0), flags=8)[1]
    return img[1:-1, 1:-1]

def segment_dino(img: Image) -> Image:
    '''
    Process the image to segment the dinosaur from the background.
    '''
    # Segment the dinosaur using two hue thresholds
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 0]
    _, img_h = cv.threshold(img, 130, 255, cv.THRESH_BINARY)
    _, img_l = cv.threshold(img,  75, 255, cv.THRESH_BINARY_INV)

    # Combine images and clear border-connected white regions
    img = clear_borders(img_h + img_l)

    return img

def shape_from_silhouette(
        silhouettes: List[Image], projections: List[np.ndarray]) -> np.ndarray:
    '''
    Define a bounding box encompassing the dinosaur, and iterate through all
    silhouette and projection matrix pairs to "carve away" empty voxels.

    This code is optimised for speed using various NumPy functionalities,
    which obfuscates the intuition somewhat.

    This code is adapted from the SpaceCarving program, by Matthieu Zins (2019):
        https://github.com/zinsmatt/SpaceCarving
    '''
    # Create voxel grid (projective coordinates)
    x, y, z = np.mgrid[MIN_X:MAX_X, MIN_Y:MAX_Y, MIN_Z:MAX_Z]
    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(float)
    pts = np.vstack((pts, np.ones((1, pts.shape[1]))))

    # For each silhouette, store an array of voxel states
    voxel_states = []

    # Use each silhouette and projection pair to "carve" the volume
    for mask, P in tqdm(list(zip(silhouettes, projections))):

        # Project all voxels onto the silhouette surface at once
        uvs = P @ pts
        uvs /= uvs[2, :]
        uvs = np.round(uvs).astype(int)

        # Remove projected points outside of the silhouette image
        x_ok = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < mask.shape[1])
        y_ok = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < mask.shape[0])
        xy_ok = np.logical_and(x_ok, y_ok)
        idx_ok = np.where(xy_ok)[0]
        uvs_ok = uvs[:2, idx_ok]

        # Identify the voxels that lie inside and outside the silhouette mask
        states = np.zeros(uvs.shape[1])
        states[idx_ok] = mask[uvs_ok[1, :], uvs_ok[0, :]]

        voxel_states.append(states)

    # In the for loop, empty voxels are marked for each silhouette individually.
    # Here, the information is combined to remove all empty voxels.
    voxel_states = np.vstack(voxel_states)
    occupancy = np.min(voxel_states, axis=0)

    return pts[:3].T, occupancy

def export_point_cloud(fp: Path, voxels: np.ndarray, occupancy: np.ndarray):
    '''
    Export a 3D array as a rectilinear grid to enable viewing of the iso-volume
    as a mesh using ParaView. The file is written to the working directory.

    This code is adapted from the SpaceCarving program, by Matthieu Zins (2019):
        https://github.com/zinsmatt/SpaceCarving
    '''
    # Define VTK arrays for Cartesian coordinates and for occupancy state
    x_coords = vtk.vtkFloatArray()
    y_coords = vtk.vtkFloatArray()
    z_coords = vtk.vtkFloatArray()
    scalars  = vtk.vtkFloatArray()

    x = np.arange(MIN_X, MAX_X)
    y = np.arange(MIN_Y, MAX_Y)
    z = np.arange(MIN_Z, MAX_Z)

    for i in x:
        x_coords.InsertNextValue(i)

    for j in y:
        y_coords.InsertNextValue(j)

    for k in z:
        z_coords.InsertNextValue(k)

    # For the matrix multiplication, the voxels are ordered:
    #   [(x0, y0, z0), (x0, y0, z1), ..., (x0, y0, zn), ...]
    #
    # but when inserting into the scalar array, VTK requires the ordering:
    #  [(x0, y0, z0), (x1, y0, z0), ..., (xn, y0, z0), ...]
    #
    # so the indexing is a bit complex to facilitate this
    a = MAX_X - MIN_X; b = MAX_Y - MIN_Y; c = MAX_Z - MIN_Z
    for p in range(c):
        for q in range(b):
            for r in range(a):
                idx = p + c * q + b * c * r
                scalars.InsertNextValue(occupancy[idx])

    # Configure rectilinear grid and set occupancy data
    rgrid = vtk.vtkRectilinearGrid()
    rgrid.SetDimensions(len(x), len(y), len(z))
    rgrid.SetXCoordinates(x_coords)
    rgrid.SetYCoordinates(y_coords)
    rgrid.SetZCoordinates(z_coords)
    rgrid.GetPointData().SetScalars(scalars)

    # Write to file
    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(str(Path(fp, 'reconstruction.vtr')))
    writer.SetInputData(rgrid)
    writer.Write()

### ENTRYPOINT #################################################################

def main():

    # Get list of image filepaths
    frames_fp = Path(A2_ROOT, 'data', 'dino', 'images')
    for _, _, files in os.walk(frames_fp):
        frame_fps = sorted(files, key=lambda f: int(f[-6:-4]))

    # Read in projection matrices from text file
    projns_fp = Path(A2_ROOT, 'data', 'dino', 'Dino projection matrices.txt')
    projections = parse_projections(projns_fp)
    assert len(frame_fps) == len(projections)

    # Apply shape-from-silhouette to reconstruct a model
    imgs = [cv.imread(str(Path(frames_fp, frame_fp))) for frame_fp in frame_fps]
    silhouettes = [segment_dino(img) for img in imgs]
    voxels, occupancy = shape_from_silhouette(silhouettes, projections)

    # Export the final volume as a point cloud to be viewed using ParaView
    export_fp = Path(A2_ROOT, 'output', 'dino')
    export_point_cloud(export_fp, voxels, occupancy)

    # Clean up
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import List

from tqdm import tqdm

### GLOBAL VARIABLES ###########################################################

# Path to the project root, i.e. "a2"
A2_ROOT = Path(__file__).parent.parent.resolve()

### TYPEDEFS ###################################################################

# Define a "contour" type, which is more descriptive than "np.ndarray"
Contour = np.ndarray

# Define an "image" type, which is more descriptive than "np.ndarray"
Image = np.ndarray

### VIDEO UTILITIES ############################################################

def frames2video(frames: List[Image], video_fp: str, fps: float):
    '''
    Combine a series of frames into a video at the given FPS rate.
    '''
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    frame_size = frames[0].shape[:2][::-1]
    vw = cv.VideoWriter(video_fp, fourcc, fps, frame_size, isColor=True)

    for frame in frames:
        vw.write(frame)

    vw.release()
    cv.destroyAllWindows()

### CONTOUR PROPERTIES #########################################################

def contour_aspect(contour: Contour) -> float:
    '''
    Return the ratio of width to height of a bounding rectangle around
    the contour.
    '''
    _, _, w, h = cv.boundingRect(contour)
    return w / h

def contour_extent(contour: Contour) -> float:
    '''
    Return the radio of contour area to bounding rectangle area.
    '''
    _, _, w, h = cv.boundingRect(contour)
    return cv.contourArea(contour) / (w * h)

def contour_solidity(contour: Contour) -> float:
    '''
    Return the ratio of contour area to convex hull area.
    '''
    hull_area = cv.contourArea(cv.convexHull(contour))
    return 0 if not hull_area else cv.contourArea(contour) / hull_area

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

def crop_frame(frame: Image) -> Image:
    '''
    Crop the frame to remove the black border. The cropping parameters were
    determined using the x- and y-axes (pixels) produced by matplotlib.
    '''
    return frame[65:-65, 151:-151]

def remove_small_contours(img: Image, min_area: float) -> Image:
    '''
    Fill in contours in a binary image with area less than min_area.
    First fills with black, then repeats with white.
    '''

    # Find small contours and fill with black
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) < min_area]
    img = cv.drawContours(img, contours, -1, color=0, thickness=-1)

    # Repeat but fill with white
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) < min_area]
    img = cv.drawContours(img, contours, -1, color=255, thickness=-1)

    return img

def remove_noncircular_contours(img: Image, min_diff: float) -> Image:
    '''
    Removes contours with an aspect ratio between min_diff on
    either side of 1.0.
    '''
    min_val = 1 - min_diff; max_val = 1 + min_diff

    # Find non-circular contours and fill with black
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if contour_aspect(c) < min_val or
                                       contour_aspect(c) > max_val]
    img = cv.drawContours(img, contours, -1, color=0, thickness=-1)

    return img

def segment_outer_wall(frame: Image) -> Image:
    '''
    Process the frame to enable contour detection of the outer ventricle wall.
    '''
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Use an adaptive threshold to binarize the image
    # frame = cv.adaptiveThreshold(frame, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv.THRESH_BINARY_INV, blockSize=101, C=20)
    frame = cv.adaptiveThreshold(frame, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv.THRESH_BINARY_INV, blockSize=71, C=10)

    # Clear white regions connected to the border to remove large noise
    frame = clear_borders(frame)

    # Prune the contours to remove small and non-circular regions (noise)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    frame = cv.morphologyEx(frame, cv.MORPH_ERODE, kernel)
    frame = remove_noncircular_contours(frame, min_diff=0.4)
    frame = cv.morphologyEx(frame, cv.MORPH_DILATE, kernel)
    frame = remove_small_contours(frame, min_area=5000)

    #
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    frame = cv.morphologyEx(frame, cv.MORPH_DILATE, kernel)

    #
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (80, 80))
    frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)

    # plt.imshow(frame, cmap='gray')
    # plt.show()

    return frame

def segment_inner_wall(frame: Image) -> Image:
    '''
    Process the frame to enable contour detection of the inner ventricle wall.
    '''
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Threshold the frame to segment the area inside the ventricle
    _, frame = cv.threshold(frame, 70, 255, cv.THRESH_BINARY)

    return frame

def contour_outer_wall(frame: Image) -> Contour:
    '''
    Return a single contour representing the left ventricle outer wall.
    '''
    # frame = segment_outer_wall(frame)

    # contours, _ = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours = [c for c in contours if contour_solidity(c) > 0.8]
    # contours = sorted(contours, key=lambda c: cv.contourArea(c), reverse=True)

    # frame_rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    # cv.drawContours(frame_rgb, contours, 0, (255, 0, 0), 2)

    # plt.imshow(frame_rgb)
    # plt.show()

    # return cv.convexHull(contours[0])

def contour_inner_wall(frame: Image) -> Contour:
    '''
    Return a single contour representing the left ventricle inner wall.
    '''
    frame = segment_inner_wall(frame)

    # Idenfify contours in the frame matching the inner wall
    contours, _ = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if contour_extent(c) > 0.4]
    contours = sorted(contours, key=lambda c: cv.contourArea(c), reverse=True)

    # Return the convex hull of the contour to eliminate noisy blobs
    return cv.convexHull(contours[0])

### DRAWING & PLOTTING #########################################################

def draw_contours(img: Image, contours: List) -> Image:
    '''
    Draw the given contours on the given BGR image in red and with
    2-pixel line thickness.
    '''
    return cv.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)

def draw_frame_no(img: Image, frame_no: int) -> Image:
    '''
    Draw the frame number on the given BGR image in the bottom right corner.
    '''
    return cv.putText(img, text=f'Frame: {frame_no}', org=(10, 245),
        fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(255, 255, 255))

def plot_ventricle_area(values: List[float], save_fp: str, fps: float = 30.0):
    '''
    Plot the given values over time, representing the area inside the inner
    wall of a left ventricle. Save the plot to the given filepath.
    '''
    plt.rc('font', family='serif', size=10)
    plt.rc('text', usetex=1)

    _, ax = plt.subplots(figsize=(8, 2))

    # Generate a vector of timestamps corresponding to each value
    t = np.linspace(start=0, stop=len(values) / fps, num=len(values))
    sns.lineplot(x=t, y=values, ax=ax, color='k')

    ax.set_title('Left ventricle area')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Area (pixels)')

    plt.savefig(save_fp, bbox_inches='tight', dpi=300)

### ENTRYPOINT #################################################################

representative_frames = [0, 19, 24, 27, 39, 67, 83, 99]

def main():

    # Get list of frame filepaths
    frames_fp = Path(A2_ROOT, 'data', 'cardiac_mri', 'frames')
    for _, _, files in os.walk(frames_fp):
        frame_fps = sorted(files, key=lambda f: int(f[-8:-4]))

    # Identify the area of the left ventricle in each frame
    ventricle_area = []; annotated_frames = []
    for i, frame_fp in enumerate(tqdm(frame_fps)):

        # if not i in representative_frames:
        #     continue

        # Read in the frame then crop away the black border
        frame = crop_frame(cv.imread(str(Path(frames_fp, frame_fp))))

        # Identify the left ventricle as a contour in the segmented frame
        contours = [
            # contour_outer_wall(frame),
            contour_inner_wall(frame),
        ]
        # annotated_frames.append(cv.cvtColor(segment_outer_wall(frame), cv.COLOR_GRAY2BGR))

        # Draw the inner and outer wall contours and frame number on each frame
        annotated_frame = draw_contours(frame, contours)
        annotated_frame = draw_frame_no(frame, i + 1)
        annotated_frames.append(annotated_frame)
        # plt.imshow(annotated_frame, cmap='gray')
        # plt.show()

        # Calculate the area enclosed by the contour
        ventricle_area.append(cv.contourArea(contours[-1]))

    # Plot the area inside the inner wall of the left ventricle over time
    save_fp = str(Path(A2_ROOT, 'output', 'cardiac_mri', 'ventricle_area.png'))
    plot_ventricle_area(ventricle_area, save_fp)

    # Save the annotated frames as a video
    video_fp = str(Path(A2_ROOT, 'output', 'cardiac_mri', 'processed_result.mp4'))
    frames2video(annotated_frames, video_fp, fps=10.0)

    # Clean up
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
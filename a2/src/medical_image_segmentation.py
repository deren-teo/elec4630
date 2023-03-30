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

# Define an "image" type, which is more intuitive than "np.ndarray"
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

### IMAGE PROCESSING ###########################################################

def crop_frame(frame: Image) -> Image:
    '''
    Crop the frame to remove the black border. The cropping parameters were
    determined using the x- and y-axes (pixels) produced by matplotlib.
    '''
    return frame[15:-15, 51:-55]

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

def segment_frame(frame: Image) -> Image:
    '''
    TODO
    '''
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # frame = cv.medianBlur(frame, ksize=21)
    frame = cv.adaptiveThreshold(frame, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv.THRESH_BINARY_INV, blockSize=101, C=13)
    # frame = cv.morphologyEx(frame, cv.MORPH_ERODE, kernel=np.ones((3, 3)))
    # frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel=np.ones((3, 3)))
    # frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel=np.ones((3, 3)))
    # frame = clear_borders(frame)

    return frame

def contour_inner_wall(frame: Image) -> np.array:
    '''
    TODO
    '''
    contours, _ = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv.contourArea(c), reverse=True)[:5]

    return contours

### DRAWING & PLOTTING #########################################################

# def draw_contour(img: Image, contour: np.array) -> Image:
#     '''
#     Draw the given contour on the given BGR image.
#     '''
#     return cv.drawContours(img, [contour], 0, color=(0, 0, 255), thickness=2)

def draw_contours(img: Image, contours: List) -> Image:
    ''''''
    return cv.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)

def draw_frame_no(img: Image, frame_no: int) -> Image:
    '''
    Draw the frame number on the given BGR image in the bottom right corner.
    '''
    return cv.putText(img, text=f'Frame: {frame_no}', org=(10, 25),
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
    ax.set_xlabel('TIme (s)')
    ax.set_ylabel('Area (sq. pixels)')

    plt.savefig(save_fp, bbox_inches='tight', dpi=300)

### ENTRYPOINT #################################################################

def main():

    # Get list of frame filepaths
    frames_fp = Path(A2_ROOT, 'data', 'cardiac_mri', 'frames')
    for _, _, files in os.walk(frames_fp):
        frame_fps = sorted(files, key=lambda f: int(f[-8:-4]))

    # Identify the area of the left ventricle in each frame
    ventricle_area = []; annotated_frames = []
    for i, frame_fp in enumerate(tqdm(frame_fps)):

        # Read in the frame then crop away the black border
        frame = crop_frame(cv.imread(str(Path(frames_fp, frame_fp))))

        # Segment the frame to enable identification of the left ventricle
        frame_segmented = segment_frame(frame)

        # Identify the left ventricle as a contour in the segmented frame
        contours = contour_inner_wall(frame_segmented)

        # Draw the left ventricle contour frame number on the frame
        annotated_frame = draw_contours(frame, contours)
        annotated_frame = draw_frame_no(frame, i + 1)
        annotated_frames.append(annotated_frame)
        # plt.imshow(annotated_frame, cmap='gray')
        # plt.show()

        # Calculate the area enclosed by the contour

    # # Plot the area inside the inner wall of the left ventricle over time
    # save_fp = str(Path(A2_ROOT, 'output', 'cardiac_mri', 'ventricle_area.png'))
    # plot_ventricle_area(ventricle_area, save_fp)

    # Save the annotated frames as a video
    video_fp = str(Path(A2_ROOT, 'output', 'cardiac_mri', 'processed_result.mp4'))
    frames2video(annotated_frames, video_fp, fps=10.0)

    # Clean up
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
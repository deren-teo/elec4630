import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

### GLOBAL VARIABLES ###########################################################

# Path to the project root, i.e. "a1"
A1_ROOT = Path(__file__).parent.parent.resolve()

# Define an "image" type, which is more intuitive than "np.ndarray"
Image = np.ndarray

### VIDEO UTILITIES ############################################################

def video2frames(video_fp: str, frames_fp: str):
    '''
    Save every frame of a video to the given directory. Directory must exist.
    '''
    vc = cv.VideoCapture(video_fp); frame_no = 0

    success, image = vc.read()
    while success:
        cv.imwrite(str(Path(frames_fp, f'frame{frame_no}.jpg')), image)
        success, image = vc.read()
        frame_no += 1

    vc.release()
    cv.destroyAllWindows()


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

def extract_pantograph_template_A(frame0: Image) -> Image:
    '''
    Create a template of the pantograph from the first frame of the video.
    The cropping parameters are determined by trial and error.
    '''
    return frame0[215:-185, 235:-260] # cropping in pixels


def extract_pantograph_template_B(frame1200: Image) -> Image:
    '''
    Create a template of the pantograph from the 1200th frame of the video.
    The cropping parameters are determined by trial and error.
    '''
    return frame1200[140:-260, 243:-268]


def crop_frame(frame: Image) -> Image:
    '''
    Crop the frame to remove the black border and watermark. The cropping
    parameters were determined by trial and error.
    '''
    return frame[40:-130, 140:-140] # cropping in pixels


def find_pantograph(frame: Image, templates: List[Image]) -> Tuple[int, int]:
    '''
    Find the top left and bottom right positions of the pantograph template
    in the frame. Takes the best match between all provided templates.
    '''
    match_val = -1; match_loc = None
    for template in templates:
        match = cv.matchTemplate(frame, template, cv.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(match)

        if max_val > match_val:
            match_val = max_val
            match_loc = max_loc

    x, y = match_loc
    h, w = template.shape[:2]

    return x, y, x + w, y + h


def crop_above_pantograph(frame: Image, pantograph_position: Tuple) -> Image:
    '''
    Crop the frame, leaving only the area above the identified pantograph.
    '''
    x1, y1, x2, _ = pantograph_position
    return frame[:y1, x1:x2]


def find_contact_point(frame: Image, thresh=75) -> Image:
    '''
    Process the frame to produce a binary image isolating the overhead lines.
    Apply the Hough (line) transform to locate the lines in the frame.
    Return...

    Code adapted from: https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
    '''
    h, w = frame.shape[:2]

    # Convert the frame to HSV and extract the V-channel, as the powerlines
    # stand out better against the sky
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)[:, :, 2]

    # Binarize the frame to isolate the powerlines from the foreground
    maxval = np.amax(frame)
    frame_binarized = cv.threshold(frame, thresh, maxval, cv.THRESH_BINARY_INV)[1]

    # Apply the Hough transform to identify powerline candidates; the threshold
    # is set to be 80% of the frame height, to reduce false positives
    lines = cv.HoughLines(frame_binarized, 1, np.pi / 180, int(0.6 * h))
    if lines is None:
        return None

    # Sort lines by theta; power line generally has smaller theta value but
    # is always greater than about 80 degrees from the horizontal
    lines = sorted(lines, key=lambda l: l[0][1]); powerline = None
    for line in lines:
        _, theta = line[0]
        if theta <= np.radians(15) or theta >= np.radians(165):
            powerline = line
            break
    if powerline is None:
        return None

    # Calculate the equation of the line formed by rho and theta
    rho, theta = powerline[0]

    # If the line is not vertical, use the equation of a line to calculate the
    # intersection of the powerline with the frame; this is approximately
    # the intersection with the pantograph
    if theta != 0:
        a = np.cos(theta)
        b = np.sin(theta)
        m = -a / b
        c = rho * (b + a**2 / b)
        if 0 <= (x := int((h - c) / m)) <= w:
            return (x, h)

    # Else if the line is vertical, then the intersection is simply (rho, h)
    return (int(rho), h)

### DRAWING & PLOTTING #########################################################

def draw_contact_point(frame: Image, point: Tuple[int, int], x_offset: int) -> Image:
    '''
    Draw the contact point on the frame with the given x-offset to correct for
    cropping done by "crop_above_pantograph".
    '''
    point = (point[0] + x_offset, point[1])
    return cv.circle(frame, point, radius=3, color=(0, 0, 255), thickness=-1)

def plot_intersection_x(values: List[int], save_fp: str, fps: float = 30.0):
    '''
    Plot the x-values of the intersection between the pantograph and the
    powerline over time. Save the plot to the given filepath.
    '''
    plt.rc('font', family='serif', size=10)
    plt.rc('text', usetex=1)

    _, ax = plt.subplots()

    # Generate a vector of timestamps corresponding to each value
    t = np.linspace(start=0, stop=len(values) / fps, num=len(values))
    sns.lineplot(x=t, y=values, ax=ax, color='k')

    ax.set_title('Horizontal intersection of powerline and pantograph')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pixels from left side')

    plt.savefig(save_fp, bbox_inches='tight', dpi=300)

### ENTRYPOINT #################################################################

def main():

    # Get list of frame filepaths
    frames_fp = Path(A1_ROOT, 'data', 'rail_pantograph', 'frames')
    for _, _, files in os.walk(frames_fp):
        frame_fps = sorted(files, key=lambda f: int(f[5:-4]))

    # Extract pantograph template from first frame
    frame0    = cv.imread(str(Path(frames_fp, 'frame0.jpg')))
    frame1200 = cv.imread(str(Path(frames_fp, 'frame1200.jpg')))
    template_A = extract_pantograph_template_A(frame0)
    template_B = extract_pantograph_template_B(frame1200)
    templates = [template_A, template_B]

    # Identify intersection of power line and pantograph in each frame
    intersection_x = []; # annotated_frames = []
    for frame_fp in tqdm(frame_fps):
        frame = cv.imread(str(Path(frames_fp, frame_fp)))

        # Crop away the black border and watermark
        frame_cropped = crop_frame(frame)

        # Identify the pantograph and isolate the region directly above it
        pantograph_position = find_pantograph(frame_cropped, templates)
        frame_overhead = crop_above_pantograph(frame_cropped, pantograph_position)

        # Identify the intersection of the pantograph and powerline
        contact_point = find_contact_point(frame_overhead)
        intersection_x.append(contact_point[0])

        # Draw the intersection point between the pantograph and powerline
        # if contact_point:
        #     x_offset = pantograph_position[0]
        #     annotated_frame = draw_contact_point(frame_cropped, contact_point, x_offset)
        # else:
        #     annotated_frame = frame_cropped

        # annotated_frames.append(annotated_frame)

    # Save a video with the powerline contact point drawn
    # video_fp = str(Path(A1_ROOT, 'output', 'rail_pantograph', 'pantograph_intersection.mp4'))
    # frames2video(annotated_frames, video_fp, fps=30)

    # Plot the horizontal position of the intersection over time
    save_fp = str(Path(A1_ROOT, 'output', 'rail_pantograph', 'intersection.png'))
    plot_intersection_x(intersection_x, save_fp)

    # Clean up
    cv.destroyAllWindows()


if __name__ == '__main__':

    # Save each frame of the pantograph video into a directory
    video_fp  = str(Path(A1_ROOT, 'data', 'rail_pantograph', 'Panto2023.mp4'))
    frames_fp = str(Path(A1_ROOT, 'data', 'rail_pantograph', 'frames_tmp'))
    video2frames(video_fp, frames_fp)

    # Run the detection program
    main()
import os
import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import List, Tuple

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

### VITERBI ALGORITHM ##########################################################

def viterbi(obs: List, states: List, start_p: np.array, trans_p: np.array,
        emit_p:np.array) -> List:
    '''
    Implementation of a generic Viterbi algorithm, adapted from Wikipedia:
    https://en.wikipedia.org/wiki/Viterbi_algorithm.

    Parameters:
        obs - List of observations, one for each time step
        states - Enumeration of all possible states; zero-indexed
        start_p - Initial probability of each state
        trans_p - Transition probabilities between states
        emit_p - Probability of each possible observation given each state

    Returns:
        Most likely sequence of states in chronological order.
    '''
    # Initialise an array to store probabilities and previous states
    V = np.empty((len(obs), len(states), 2), dtype=np.float64)

    # Calculate the intial probabilities
    for s in states:
        V[0, s] = np.array([start_p[s] * emit_p[s, obs[0]], np.nan])

    # Run the Viterbi algorithm for t > 0
    for t in range(1, len(obs)):
        for s in states:
            max_prob = 0; s_prev_best = None
            for s_prev in states:
                prob = V[t - 1, s_prev, 0] * trans_p[s_prev, s] * emit_p[s, obs[t]]
                if prob > max_prob:
                    max_prob = prob
                    s_prev_best = s_prev
            V[t, s] = np.array([max_prob, s_prev_best])

    # Return the most probable sequence of states in chronological order
    sequence = [np.argmax(V[-1, :, 0]),]

    for t in range(len(obs) - 1, 0, -1):
        s_prev = V[t, sequence[-1], 1]
        sequence.append(int(s_prev))

    return sequence[::-1]

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

def segment_outer_wall(frame: Image) -> Image:
    '''
    Process the frame to enable contour detection of the outer ventricle wall.
    '''
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Use an adaptive threshold to binarize the image
    frame = cv.adaptiveThreshold(
        frame, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY_INV, blockSize=71, C=10)

    thresholded_copy = frame.copy()

    # Black out white regions connected to the border then remove remaining
    # small contours to reduce noise
    frame = clear_borders(frame)
    frame = remove_small_contours(frame, min_area=4000)

    # If either of the above steps removed everything in the frame,
    # then apply a different strategy based on morphological opening
    if np.all(frame == 0):

        frame = thresholded_copy

        # Fill in the largest contour with white; this is usually the inner
        # area of the ventricle, which allows a large morphological opening
        # to be applied without completely losing thin wall sections
        contours, _ = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv.contourArea(c))
        cv.drawContours(frame, contours[:-1], -1, color=255, thickness=-1)

        # Apply morphological opening using a large kernel to remove noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)

        # Remove small contours isolated by the morph opening
        frame = remove_small_contours(frame, 4000)

    return frame

def segment_inner_wall(frame: Image) -> Image:
    '''
    Process the frame to enable contour detection of the inner ventricle wall.
    '''
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Threshold the frame to segment the area inside the ventricle
    _, frame = cv.threshold(frame, 70, 255, cv.THRESH_BINARY)

    return frame

def extract_states(
        frame: Image,
        search_origin: Tuple,
        radial_resolution: int,
        distal_resolution: int,
        max_radius: int,
        search_dir: int
    ) -> np.array:
    '''
    Extract states from a segmented frame for the Viterbi algorithm.
    A state is a combination of a radial and distal position.

    Parameters:
        frame - binary segmented image
        radial_resolution - angular resolution of states
        distal_resolution - distal resolution of states, along an angle
        max_radius - maximum distance from origin to search
        search_dir - if 1, searches outward from origin; if -1, searches
            inward from max_radius

    Returns:
        Array of length equal to radial resolution; distal states.
    '''
    # Initialise an array to store a state number for each angle
    states = np.empty(radial_resolution, dtype=np.uint8)

    # Define an innermost distance; don't check the search origin
    distal_origin = max_radius // distal_resolution

    # Define possible angles and radii based on given resolutions
    angles = np.linspace(0, 360, radial_resolution, endpoint=False)
    radii  = np.linspace(distal_origin, max_radius, distal_resolution)

    # Invert the search direction if requested
    radii = radii[::-1] if search_dir < 0 else radii

    # Search the segmented image to populate the states
    for i, angle in enumerate(angles):
        for j, radius in enumerate(radii):
            x = int(radius * np.cos(angle) + search_origin[0])
            y = int(radius * np.sin(angle) + search_origin[1])
            if frame[x, y] == 255:
                states[i] = j
                break

    return states

def contour_ventricle_wall(obs_array: np.array, states: List,
        angles: List, contour_origin: Tuple) -> np.array:
    '''
    Apply the Viterbi algorithm to construct a contour at each time step
    representing the most likely configuration of the ventricle wall.

    Parameters:
        obs_array -
        states -
        angles -
        contour_origin -

    Returns:

    '''
    # Define Viterbi initial state probability uniformly over all states
    n_states = len(states)
    start_p = (1 / n_states) * np.ones(n_states)

    # Define Viterbi transition probability; states are likely
    # to transition to close by states and not those far away
    trans_p = np.eye(n_states)
    for i in range(n_states):
        for j in range(i + 1, n_states):
            trans_p[i, j] = 1 / abs(states[i] - states[j])
    for i in range(1, n_states):
        for j in range(i):
            trans_p[i, j] = trans_p[j, i]
    trans_p *= np.sum(trans_p)

    # Define Viterbi emittance probability; observations are likely to
    # occur near where the true state is (same as transition probability)
    emit_p = trans_p

    #
    angle_states = np.empty(obs_array.shape)
    for i in range(obs_array.shape[1]):
        obs = obs_array[:, i]
        states = list(range(n_states))
        angle_states[:, i] = viterbi(obs, states, start_p, trans_p, emit_p)

    # Convert from a list of states per angle at each time step to
    # a set of pixel locations defining a contour for each frame
    contours = np.empty((*obs_array.shape, 1, 2), dtype=np.int32)
    for t, obs in enumerate(obs_array):
        for angle_idx, state_idx in enumerate(obs):
            x = int(states[state_idx] * np.cos(angles[angle_idx]) + contour_origin[0])
            y = int(states[state_idx] * np.sin(angles[angle_idx]) + contour_origin[1])
            contours[t, angle_idx, 0] = np.array([x, y])

    return contours

### DRAWING & PLOTTING #########################################################

def draw_contours(img: Image, contours: List) -> Image:
    '''
    Draw the given contours on the given BGR image in red and with
    2-pixel line thickness.
    '''
    try:
        return cv.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)
    except cv.error as e:
        print(contours)
        raise e

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

def main():

    # Get list of frame filepaths
    frames_fp = Path(A2_ROOT, 'data', 'cardiac_mri', 'frames')
    for _, _, files in os.walk(frames_fp):
        frame_fps = sorted(files, key=lambda f: int(f[-8:-4]))

    # Define configuration parameters for the state-extraction step
    h, w = crop_frame(cv.imread(str(Path(frames_fp, frame_fps[0])))).shape[:2]
    viterbi_configuration = {
        'search_origin':     (origin := (w // 2, h // 2)),
        'radial_resolution': (radial_resolution := 3),
        'distal_resolution': (distal_resolution := 20),
        'max_radius':        (max_radius := 100),
    }

    # Process each frame to identify states to use with Viterbi
    frames = []
    viterbi_states_outer = np.empty((len(frame_fps), radial_resolution), dtype=np.uint8)
    viterbi_states_inner = np.empty((len(frame_fps), radial_resolution), dtype=np.uint8)
    for i, frame_fp in enumerate(tqdm(frame_fps)):

        # Read in the frame then crop away the black border
        frame = crop_frame(cv.imread(str(Path(frames_fp, frame_fp))))
        frames.append(frame)

        # Segment the frame then record the edge states for Viterbi
        viterbi_states_outer[i] = extract_states(segment_outer_wall(frame),
            **viterbi_configuration, search_dir=-1)
        viterbi_states_inner[i] = extract_states(segment_inner_wall(frame),
            **viterbi_configuration, search_dir=1)

    # Run the Viterbi algorithm on the collected states and create
    # contours based on the returned sequence of states
    distal_origin = max_radius // distal_resolution
    contour_config = {
        'states': np.linspace(distal_origin, max_radius, distal_resolution),
        'angles': np.linspace(0, 360, radial_resolution, endpoint=False),
        'contour_origin': origin,
    }

    contours_outer = contour_ventricle_wall(viterbi_states_outer, **contour_config)
    contours_inner = contour_ventricle_wall(viterbi_states_inner, **contour_config)

    # Identify the area of the left ventricle in each frame
    ventricle_area = []; annotated_frames = []
    for i, frame in enumerate(frames):

        # Draw the inner and outer wall contours and frame number on each frame
        contours = [contours_outer[i], contours_inner[i]]
        annotated_frame = draw_contours(frame, contours)
        annotated_frame = draw_frame_no(frame, i + 1)
        annotated_frames.append(annotated_frame)

        # Calculate the area enclosed by the inner contour
        ventricle_area.append(cv.contourArea(contours[-1]))

    # Plot the area inside the inner wall of the left ventricle over time
    save_fp = str(Path(
        A2_ROOT, 'output', 'cardiac_mri', 'ventricle_area_viterbi.png'))
    plot_ventricle_area(ventricle_area, save_fp)

    # Save the annotated frames as a video
    video_fp = str(Path(
        A2_ROOT, 'output', 'cardiac_mri', 'processed_result_viterbi.mp4'))
    frames2video(annotated_frames, video_fp, fps=10.0)

    # Clean up
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
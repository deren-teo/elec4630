import os

from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np
import scipy.stats as stats

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

def frames2video(frames: List[Image], video_fp: Path, fps: float):
    '''
    Combine a series of frames into a video at the given FPS rate.
    '''
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    frame_size = frames[0].shape[:2][::-1]
    vw = cv.VideoWriter(str(video_fp), fourcc, fps, frame_size, isColor=True)

    for frame in frames:
        vw.write(frame)

    vw.release()
    cv.destroyAllWindows()

### VITERBI ALGORITHM ##########################################################

def viterbi(obs: List, states: List, start_p: np.ndarray, trans_p: np.ndarray,
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
    frame = cv.adaptiveThreshold(
        frame, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY_INV, blockSize=71, C=10)

    # Remove small contours to reduce noise
    frame = remove_small_contours(frame, min_area=4000)

    return frame

def extract_states(segmented_frame: Image, radial_dir: int) -> np.ndarray:
    '''
    Extract states from a segmented frame to be used for the Viterbi algorithm.

    Within the context of the ventricle wall segmentation problem, a state is
    defined as a discretised distance from an origin point somewhere in the
    segmented frame. The origin is defined globally by FRAME_ORIGIN.

    The state represents the distance to the outer/inner ventricle wall in
    the discretised space, as represented by the first white pixel found
    along a line from the origin and at some angle to the horizontal.

    Depending on radial_dir, the points along the line from the origin can be
    searched outwards from the origin, or inwards towards the origin. This
    allows the function to be used for both the inner and outer ventricle walls.

    The resolution of the discretised space along the line from the origin
    is defined globally by RADIAL_RESOLUTION. This integer value defines the
    number of equally-spaced points along the line, between the origin and
    some maximum radial distance, also defined globally, by MAX_RADIUS.

    Of course, the contour of the ventricle wall cannot be adequately described
    by a single point on the coincident with a line from the origin at some
    arbitrary angle to the horizontal. Therefore, the process of finding the
    first white pixel along a line from the origin is repeated for many lines
    of equally spaced angles. The number of lines is defined globally by
    ANGULAR_RESOLUTION.

    As a final note, this function does not return the actual states extracted
    for each angle, which would be either a polar or Cartesian coordinate.
    Instead, for each angle, the index of the state is stored based on the
    enumeration of all possible states. This index depends on radial_dir.
    '''
    # Initialise an array to store a state number for each angle
    extracted_states = np.empty(ANGULAR_RESOLUTION, dtype=np.uint8)

    # Define an innermost distance; don't check the search origin
    distal_origin = MAX_RADIUS // RADIAL_RESOLUTION

    # Define all angles and radii based on angular and radial resolutions
    angles = np.linspace(0, 360, ANGULAR_RESOLUTION, endpoint=False)
    radii  = np.linspace(distal_origin, MAX_RADIUS, RADIAL_RESOLUTION)

    # Invert the search direction if required
    radii = radii[::radial_dir]; assert radial_dir in (-1, 1)

    # Search the segmented image to populate the states, starting either
    # from inside out or outside in based on the radial_dir parameter
    for i, angle in enumerate(angles):
        for j, radius in enumerate(radii):
            x = int(radius * np.cos(np.radians(angle)) + FRAME_ORIGIN[0])
            y = int(radius * np.sin(np.radians(angle)) + FRAME_ORIGIN[1])
            if segmented_frame[y, x] == 255:
                extracted_states[i] = j
                break

    return extracted_states

def extract_outer_states(frame: Image) -> np.ndarray:
    '''
    Segment the frame using the algorithm for the outer wall, then extract
    states for the Viterbi algorithm from the outside in.
    '''
    return extract_states(segment_outer_wall(frame), radial_dir=-1)

def extract_inner_states(frame: Image) -> np.ndarray:
    '''
    Segment the frame using the algorithm for the inner wall, then extract
    states for the Viterbi algorithm from the inside out.
    '''
    return extract_states(segment_inner_wall(frame), radial_dir=1)

def generate_viterbi_sequences(observation_array: np.ndarray) -> np.ndarray:
    '''
    Runs the Viterbi algorithm on each column of the 2D observation_array.

    Each row of observation_array contains the set of states extracted from a
    single frame for either the inner or outer ventricle wall.

    Each column of observation_array contains the chronological ordering of all
    states observed for a single radial line, the angle of which is defined
    by the angular resolution global parameter and indexed respectively.

    Returns:
        2D array with same dimensions as the input observation_array. Instead
        of observations, the return array describes the most likely state at
        each timestep for each angle based on the Viterbi algorithm.
    '''
    # The Viterbi algorithm requires a vector defining the initial
    # probabilities of being in each state; since the initial configuration
    # of the ventricle is unknown, a uniform distribution is used
    start_p = (1 / RADIAL_RESOLUTION) * np.ones(RADIAL_RESOLUTION)

    # Define a transition probability matrix, which specifies the probability
    # of transitioning from one state to another between consecutive timesteps
    # A normal distribution is used, which reflects the fact that the motion of
    # the ventricle wall is continuous in reality, therefore probability of a
    # transitioning depends on the distance between states
    trans_p = np.eye(RADIAL_RESOLUTION); trans_norm = stats.norm(0, 1)
    # Populate the upper triangle first, then copy to the lower triangle
    for i in range(RADIAL_RESOLUTION):
        for j in range(i + 1, RADIAL_RESOLUTION):
            trans_p[i, j] = trans_norm.pdf(abs(i - j))
    # Copy and transpose the upper triangle to the lower triangle
    for i in range(1, RADIAL_RESOLUTION):
        for j in range(i):
            trans_p[i, j] = trans_p[j, i]

    # Define an emittance probability matrix, which specifies the probability
    # of each possible observation given a particular (actual but hidden) state
    # As with the transition probabilities, a normal distribution is used, which
    # reflects the fact that an observation is likely to be made close to the
    # actual state, but is more lenient due to the known presence of noise.
    emit_p = np.eye(RADIAL_RESOLUTION); emit_norm = stats.norm(0, 5)
    # Populate the upper triangle first, then copy to the lower triangle
    for i in range(RADIAL_RESOLUTION):
        for j in range(i + 1, RADIAL_RESOLUTION):
            emit_p[i, j] = emit_norm.pdf(abs(i - j))
    # Copy and transpose the upper triangle to the lower triangle
    for i in range(1, RADIAL_RESOLUTION):
        for j in range(i):
            emit_p[i, j] = emit_p[j, i]

    # Run the Viterbi algorithm on each set of observations, corresponding
    # to each angle in the angular resolution, to generate a chronological
    # ordering of most probable states for each angle over time
    probable_states = np.empty(observation_array.shape, dtype=np.uint8)

    state_enum = list(range(RADIAL_RESOLUTION))

    for angle_idx in range(observation_array.shape[1]):
        obs = observation_array[:, angle_idx]
        probable_states[:, angle_idx] = viterbi(
            obs, state_enum, start_p, trans_p, emit_p)

    return probable_states

def reconstruct_contours(sequences: np.ndarray, radial_dir: int) -> List[Contour]:
    '''
    Reconstruct a sequence of contours from multiple sequences of states
    generated by runs of the Viterbi algorithm for each increment in
    the angular resolution.

    The input array should have dimensions NxM, where N is the number of
    timesteps (i.e. number of frames), and M is the angular resolution.

    Parameters:
        sequences - input array as described above
        radial_dir - inside out (=1) or outside in (=-1) state reconstruction

    Returns:
        List of contours of length N; one contour per timestep (/frame).
    '''
    contours = []

    # Define all angles and radii based on angular and radial resolutions
    distal_origin = MAX_RADIUS // RADIAL_RESOLUTION
    angles = np.linspace(0, 360, ANGULAR_RESOLUTION, endpoint=False)
    radii  = np.linspace(distal_origin, MAX_RADIUS, RADIAL_RESOLUTION)

    # Invert the reconstruction direction if required
    radii = radii[::radial_dir]; assert radial_dir in (-1, 1)

    # For each timestep, convert all state (i.e. a distance from the origin)
    # and angle pairs into a Cartesian pixel location to form a contour
    for observation in sequences:
        new_contour = np.empty((ANGULAR_RESOLUTION, 1, 2), dtype=np.int32)

        for angle_idx, state_idx in enumerate(observation):
            x = int(radii[state_idx] * np.cos(np.radians(angles[angle_idx])))
            y = int(radii[state_idx] * np.sin(np.radians(angles[angle_idx])))

            new_contour[angle_idx, 0] = np.array([x, y]) + np.array(FRAME_ORIGIN)

        contours.append(cv.convexHull(new_contour))

    return contours

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

### ENTRYPOINT #################################################################

def define_state_generator_globals():
    '''
    Define global parameters to be used for discretising segmented frames
    into sets of states for the Viterbi algorithm.
    '''
    # Load and crop frame0 to determine the origin of the radial states
    frames_fp = str(Path(A2_ROOT, 'data', 'cardiac_mri', 'frames'))
    frame0_fp = str(Path(frames_fp, 'Cardiac MRI Left ventricle0000.jpg'))
    frame0 = crop_frame(cv.imread(frame0_fp))
    h, w = frame0.shape[:2]

    # Define several global configuration parameters
    global FRAME_ORIGIN, ANGULAR_RESOLUTION, RADIAL_RESOLUTION, MAX_RADIUS

    # Frame origin is the middle of the frame; we look for the ventricle wall
    # by searching radially outward (/inward) from (/to) this origin point
    FRAME_ORIGIN = (w // 2, h // 2)

    # Angular resolution defines the number of angles within 360 degrees to
    # search along; e.g. radial resolution of 36 means the ventricle wall
    # is searched for along radial lines every 10 degrees apart
    ANGULAR_RESOLUTION = 36

    # Radial resolution defines the number of points along each radial line,
    # also corresponding to the number of Viterbi states per angle.
    # The pixel under each point in the segmented frame is determined as either
    # black or white. The first white point is the observed Viterbi state.
    RADIAL_RESOLUTION = 40

    # Max radius defines the maximum distance in pixels away from the frame
    # origin to search for the ventricle wall. This value should be large
    # enough to enclose the entire ventricle in all frames, but should not
    # exceed (half of) the minimum dimension of any frame.
    MAX_RADIUS = 120

def main():

    # Get list of frame filepaths
    frames_fp = Path(A2_ROOT, 'data', 'cardiac_mri', 'frames')
    for _, _, files in os.walk(frames_fp):
        frame_fps = sorted(files, key=lambda f: int(f[-8:-4]))

    # Store an array of cropped frames in memory for easier manipulation
    read_frame = lambda frame_fp: cv.imread(str(Path(frames_fp, frame_fp)))
    frames = [crop_frame(read_frame(frame_fp)) for frame_fp in frame_fps]

    # Define parameters for discretising segmented frames into sets of states
    # for use with the Viterbi algorithm, then for converting the Viterbi
    # results into back into points to define contours
    define_state_generator_globals()

    # Extract outer/inner wall states for each frame for the Viterbi algorithm
    outer_states = np.empty((len(frame_fps), ANGULAR_RESOLUTION), dtype=np.uint8)
    inner_states = np.empty((len(frame_fps), ANGULAR_RESOLUTION), dtype=np.uint8)

    for i, frame in enumerate(tqdm(frames)):
        outer_states[i] = extract_outer_states(frame)
        inner_states[i] = extract_inner_states(frame)

    # Run the Viterbi algorithm on the extracted states; the algorithm needs
    # to be run once for every increment in angular resolution and returns a
    # sequence of states representing the most likely position of the outer/
    # inner ventricle wall along a line at each angle, for each time step
    outer_state_sequences = generate_viterbi_sequences(outer_states)
    inner_state_sequences = generate_viterbi_sequences(inner_states)

    # Reconstruct the sequences of states generated by the Viterbi algorithm
    # into a sequence of contours; an inner and outer contour for each frame
    outer_contours = reconstruct_contours(outer_state_sequences, radial_dir=-1)
    inner_contours = reconstruct_contours(inner_state_sequences, radial_dir=1)

    # For each frame, draw the contours produced by the Viterbi algorithm
    # onto the frame, as well as the frame number. Calculate the area enclosed
    # by the inner wall contour and store a list of ventricle area values
    ventricle_area = []; annotated_frames = []
    for i, frame in enumerate(frames):

        # Draw the inner and outer wall contours and frame number on each frame
        contours = [outer_contours[i], inner_contours[i]]
        annotated_frame = draw_contours(frame, contours)
        annotated_frame = draw_frame_no(frame, i + 1)
        annotated_frames.append(annotated_frame)

        # Calculate the area enclosed by the inner contour
        ventricle_area.append(cv.contourArea(contours[-1]))

    # Export the raw ventricle area data
    save_fp = Path(A2_ROOT, 'output', 'cardiac_mri', 'ventricle_area_viterbi.txt')
    np.savetxt(save_fp, np.array(ventricle_area))

    # Save the annotated frames as a video
    video_fp = Path(A2_ROOT, 'output', 'cardiac_mri', 'processed_result_viterbi.mp4')
    frames2video(annotated_frames, video_fp, fps=29.97)

    # Clean up
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

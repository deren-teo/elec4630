import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

# Path to the project root, i.e. "a1"
A1_ROOT = Path(__file__).parent.parent.resolve()

# Define an "image" type, which is more intuitive than "np.ndarray"
Image = np.ndarray


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

    for frame in tqdm(frames):
        vw.write(frame)

    vw.release()
    cv.destroyAllWindows()


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


def mark_frame(frame: Image, pantograph_position: Tuple) -> Image:
    '''
    Mark the position of the pantograph in the frame using the template size.
    '''
    x1, y1, x2, y2 = pantograph_position
    cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255))
    return frame


def crop_above_pantograph(frame: Image, pantograph_position: Tuple) -> Image:
    '''
    Crop the image, leaving only the area above the identified pantograph.
    '''
    x1, y1, x2, _ = pantograph_position
    return frame[:y1, x1:x2]


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
    marked_frames = []
    for frame_fp in tqdm(frame_fps):
        frame = cv.imread(str(Path(frames_fp, frame_fp)))
        frame_cropped = crop_frame(frame)
        pantograph_position = find_pantograph(frame_cropped, templates)
        frame_marked = mark_frame(frame_cropped, pantograph_position)
        marked_frames.append(frame_marked)

    # Save a video with the pantograph position marked
    video_fp = str(Path(A1_ROOT, 'data', 'rail_pantograph', 'pantograph_tracking.mp4'))
    frames2video(marked_frames, video_fp, fps=30)

    # Plot the horizontal position of the intersection over time

    # Clean up
    cv.destroyAllWindows()


if __name__ == '__main__':
    
    # Save each frame of the pantograph video into a directory
    # video_fp  = str(Path(A1_ROOT, 'data', 'rail_pantograph', 'Panto2023.mp4'))
    # frames_fp = str(Path(A1_ROOT, 'data', 'rail_pantograph', 'frames'))
    
    # video2frames(video_fp, frames_fp)

    # Run the detection program
    main()
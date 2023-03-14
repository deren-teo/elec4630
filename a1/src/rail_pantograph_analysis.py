import cv2 as cv

from pathlib import Path

# Path to the project root, i.e. "a1"
A1_ROOT = Path(__file__).parent.parent.resolve()


def extract_frames(video_fp: str, frames_fp: str):
    '''
    Save every frame of a video to the given directory. Directory must exist.
    '''
    vc = cv.VideoCapture(video_fp); frame_no = 0

    success, image = vc.read()
    while success:
        cv.imwrite(str(Path(frames_fp, f'frame{frame_no}.jpg')), image)
        success, image = vc.read()
        frame_no += 1


def main():
    pass


if __name__ == '__main__':
    
    # Save each frame of the pantograph video into a directory
    video_fp  = str(Path(A1_ROOT, 'data', 'rail_pantograph', 'Panto2023.mp4'))
    frames_fp = str(Path(A1_ROOT, 'data', 'rail_pantograph', 'frames'))
    
    extract_frames(video_fp, frames_fp)

    # Run the detection program
    # main()
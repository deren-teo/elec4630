import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Tuple

### GLOBAL VARIABLES ###########################################################

# Path to the project root, i.e. "a1"
A1_ROOT = Path(__file__).parent.parent.resolve()

### TYPEDEFS ###################################################################

# Define an "image" type, which is more intuitive than "np.ndarray"
Image = np.ndarray

# Define a "line" type, describing a line between two Cartesian points
Line = Tuple[int, int, int, int] # (x1, y1, x2, y2)

# Define a "circle" type, describing a circle of given centre and radius
Circle = Tuple[int, int, int] # (x, y, r)

### IMAGE PROCESSING ###########################################################

def contour_aspect(contour: np.ndarray) -> float:
    '''
    Returns width to height ratio of a bounding rectangle around the contour.
    '''
    _, _, w, h = cv.boundingRect(contour)
    return float(w) / h

def find_road_edges(img: Image) -> List[Line]:
    '''
    Detect the position of the road edge in the provided image. It is expected
    that the road edge will be segmented by the car in the foreground.

    Returns a list of lines (defined between two Cartesian points) indicating
    the sections of identified road edge.
    '''

    # Convert the image to HSV and extract the saturation channel, which
    # contrasts the road well against the grass
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 1]

    # Binarize the frame to enhance the road
    img = cv.threshold(img, thresh=100, maxval=255, type=cv.THRESH_BINARY)[1]

    # Morphologically close the image using a very large (gigantic) kernel to
    # absolutely eradicate every last detail except the road edge border
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=np.ones((121, 121)))
    img = np.invert(img)

    # Split the image into left and right halves to identify road extent
    _, w = img.shape
    img_l = img[:, :int(w / 2)]
    img_r = img[:, int(w / 2):]

    # For each half, creating a bounding rectangle defining the road extent
    _, _, extent_l, _ = cv.boundingRect(img_l)
    _, _, extent_r, _ = cv.boundingRect(img_r)

    # Run edge detection to identify the road edge border
    img = cv.Canny(img_l, 100, 200)

    # Apply the Hough line transform to identify the road edge
    # Threshold should be large enough to ignore noise
    line = cv.HoughLines(img, rho=1, theta=(np.pi / 180), threshold=100)[-1]

    # Calculate the equation of the line formed by rho and theta
    rho, theta = line[0]

    # Use the equation of a line to calculate the section of the road between
    # the left and right borders and extents
    a = np.cos(theta)
    b = np.sin(theta)
    m = -a / b
    c = rho * (b + a**2 / b)

    # Define the lines of the road edge, not overlapping the foreground
    edge_l = (0, int(c), extent_l, int(m * extent_l + c))
    edge_r = (w - extent_r, int(m * (w - extent_r) + c), w, int(m * w + c))

    return [edge_l, edge_r]

def find_broomstick(img: Image) -> List[Line]:
    '''
    Detect the position of the broomstick in the provided image.

    Returns a list of one line (defined between two Cartesian points)
    indicating the position of the broomstick.
    '''

def find_wheelhubs(img: Image) -> List[Circle]:
    '''
    Detect the position and size of the wheel hubs in the provided image.

    Returns a list of circles (defined by a Cartesian centre and radius)
    indicating the outline of the identified wheel hubs.
    '''

    # Binarize the image to enhance the wheelhub outlines
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.threshold(img, thresh=50, maxval=255, type=cv.THRESH_BINARY)[1]

    # Morphologically close image to remove noisy details
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=np.ones((11, 11)))

    # Find contours in the image, with the aim of identifying circles
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours by area to leave only wheel-sized contours
    # For reference, the wheel hubs in the sample have ~3000 sq. pixels
    contours = tuple(c for c in contours if 2500 < cv.contourArea(c) < 3500)

    # Further filter contours by aspect ratio, which should be sufficient
    # to distinguish wheel hubs
    contours = tuple(c for c in contours if 0.8 < contour_aspect(c) < 1.2)

    # Record the position and (assumed) radius of all remaining contours
    wheelhubs_circles: List[Circle] = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour); r = int((w + h) / 4)
        wheelhubs_circles.append((x + r, y + r, r))

    return wheelhubs_circles

### ENTRYPOINT #################################################################

def main():

    # Read in the sample image
    img_fp = str(Path(A1_ROOT, 'data', 'mr_bean', 'MrBean2023.jpg'))
    img = cv.imread(img_fp)

    # Maintain an RGB version of the image for drawing on
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Identify and draw the road edge in the image
    road_edge_lines = find_road_edges(img)
    for x1, y1, x2, y2 in road_edge_lines:
        cv.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), thickness=5)

    # Identify and draw the broomstick in the image
    broomstick_line = find_broomstick(img)
    for x1, y1, x2, y2 in broomstick_line:
        cv.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), thickness=5)

    # Identify and draw the wheel hub outlines in the image
    wheelhub_circles = find_wheelhubs(img)
    for x, y, r in wheelhub_circles:
        cv.circle(img_rgb, (x, y), r, (255, 0, 0), thickness=5)

    # Save the final result
    save_fp = str(Path(A1_ROOT, 'output', 'mr_bean', 'detected_features.png'))
    cv.imwrite(save_fp, img_rgb)


if __name__ == '__main__':
    main()
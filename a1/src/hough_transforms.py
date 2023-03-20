import cv2 as cv
import numpy as np

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

def hough_line_equation(line: np.ndarray) -> Tuple[float, float]:
    '''
    Calculate the gradient and y-intercept from the rho and theta values
    returned by a Hough line transform.
    '''
    rho, theta = line[0]

    a = np.cos(theta)
    b = np.sin(theta)

    m = -a / b
    c = rho * (b + a**2 / b)

    return m, c

def contour_height(contour: np.ndarray) -> int:
    '''
    Returns the height of a bounding rectangle around the contour.
    '''
    _, _, _, h = cv.boundingRect(contour)
    return h

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

    # Perform Canny edge detection to identify road edges
    edges = cv.Canny(img, 100, 200)

    # Apply the Hough line transform to identify the road edge
    # Threshold should be large enough to ignore noise
    line = cv.HoughLines(edges, rho=1, theta=(np.pi / 180), threshold=100)[0]

    # Calculate the gradient and y-intercept of the line
    m, c = hough_line_equation(line)

    # Use contours to identify sections of road not obstructed by foreground
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # For each road section, define the line representing the road edge
    road_edges = []
    for contour in contours:
        x, _, w, _ = cv.boundingRect(contour)
        road_edges.append((x, int(m * x + c), x + w, int(m * (x + w) + c)))

    return road_edges

def find_broomstick(img: Image) -> List[Line]:
    '''
    Detect the position of the broomstick in the provided image.

    Returns a list of one line (defined between two Cartesian points)
    indicating the position of the broomstick.
    '''

    # Convert the image to HSV and extract the saturation channel, which
    # contrasts the broomstick reasonably well against the grass
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 1]

    # Binarize image to enhance the broomstick outline
    img = cv.threshold(img, thresh=150, maxval=255, type=cv.THRESH_BINARY_INV)[1]

    # Apply morphological tophat transform to black out large white areas
    img = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel=np.ones((11, 7)))

    # Morphologically open the image using a vertical kernel to remove as much
    # noise as possible without erasing the thin, vertical broomstick line
    img_opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel=np.ones((11, 1)))

    # Apply the Hough line transform to identify the road edge
    # Threshold should be large enough to ignore noise
    line = cv.HoughLines(img_opened, rho=1, theta=(np.pi / 180), threshold=125)[0]

    # Calculate the gradient and y-intercept of the line
    m, c = hough_line_equation(line)

    # Morphologically open the image to remove noise, but this time less than
    # before to maintain a continuous broomstick contour
    img_opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel=np.ones((9, 1)))

    # Find contours in the image, with the aim of identifying broomstick shapes
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Sort the contours by vertical length to find the broomstick contour
    broomstick_contour = sorted(contours, key=lambda c: contour_height(c))[-1]

    # Get the position and size of the broomstick contour
    _, y, _, h = cv.boundingRect(broomstick_contour)

    # Contour height offset: the contour method tends to include Mr Bean's hand
    # and not the top of the broomstick, so an offset is added
    y += -20

    # Define and return the extents of the line
    return [(int((y - c) / m), y, int((y + h - c) / m), y + h),]

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

    # Apply the Hough circle transform to identify circles in the image
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=50, param2=30)

    return list(tuple(map(int, circle)) for circle in circles[0])

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
        cv.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

    # Identify and draw the broomstick in the image
    broomstick_line = find_broomstick(img)
    for x1, y1, x2, y2 in broomstick_line:
        cv.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

    # Identify and draw the wheel hub outlines in the image
    wheelhub_circles = find_wheelhubs(img)
    for x, y, r in wheelhub_circles:
        cv.circle(img_rgb, (x, y), r, (255, 0, 0), thickness=3)

    # Save the final result
    save_fp = str(Path(A1_ROOT, 'output', 'mr_bean', 'detected_features.png'))
    cv.imwrite(save_fp, cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
import os

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

# Define a "rectangle" type, describing a top-left corner and width and height
Rectangle = Tuple[int, int, int, int]

### TEMPLATE MANAGEMENT ########################################################

def template_diamond() -> Image:
    '''
    Create a solid diamond template.
    '''

    # Initialise the template canvas
    template = np.zeros((100, 100), dtype=np.uint8)

    # Draw the diamond based on linear inequalities
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            if i <= 50 and i > 50 - j and i > j - 50:
                template[i][j] = 1
            elif i > 50 and i < j + 50 and i < 150 - j:
                template[i][j] = 1

    return template

def template_triangle() -> Image:
    '''
    Create an inverted triangle outline template.
    '''

    # Initialise the template canvas
    template = np.zeros((110, 120), dtype=np.uint8)

    # Draw the triangle based on linear equations
    cv.line(template, ( 10, 10), ( 56, 100), color=255, thickness=11)
    cv.line(template, (111, 10), ( 56, 100), color=255, thickness=11)
    cv.line(template, ( 10, 10), (111,  10), color=255, thickness=11)

    # Inset the template into a 5-pixel border all around
    border_description = (5, 5, 5, 5, cv.BORDER_CONSTANT)
    template = cv.copyMakeBorder(template, *border_description, value=0)

    # Blur the template to match slightly distorted signs
    return cv.blur(template, ksize=(11, 11))

def template_rectangle(aspect_ratio: float) -> Image:
    '''
    Create a solid rectangle template with the given aspect ratio.
    '''

    # Aspect ratio > 1; wide rectangle
    if aspect_ratio > 1:
        h = int(50 / aspect_ratio) * 2
        rectangle = np.ones((h, 100), dtype=np.uint8)

    # Aspect ratio <= 1; tall rectangle
    else:
        w = int(50 * aspect_ratio) * 2
        rectangle = np.ones((100, w), dtype=np.uint8)

    # Inset the rectangle in a 5-pixel upper and lower border
    border_description = (5, 5, 0, 0, cv.BORDER_CONSTANT)
    return cv.copyMakeBorder(rectangle, *border_description, value=0)

def scale_template(template: Image, scaling_factor: float) -> Image:
    '''
    Scales a 100x100 template by the scaling factor.
    '''
    scaled_w = int(template.shape[1] * scaling_factor)
    scaled_h = int(template.shape[0] * scaling_factor)
    interpolation = cv.INTER_AREA if (scaling_factor < 1) else cv.INTER_CUBIC

    return cv.resize(template, (scaled_w, scaled_h), interpolation)

def template_too_large(img: Image, template: Image) -> bool:
    '''
    If any of the template is larger than the image, returns true. Else false.
    '''
    h, w = img.shape[:2]; t_h, t_w = template.shape[:2]

    return any([h < t_h, w < t_w])

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

def process_reddish(img: Image) -> Image:
    '''
    Process the image to enhance red/yellow signs using the saturation channel.
    This is effective at removing background, but also removes white signs.
    '''
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 1]
    img = cv.bilateralFilter(img, d=11, sigmaColor=50, sigmaSpace=75)
    img = cv.threshold(img, thresh=200, maxval=255, type=cv.THRESH_BINARY)[1]
    img = cv.morphologyEx(img, cv.MORPH_DILATE, kernel=np.ones((3, 3)))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel=np.ones((7, 7)))
    contours, _ = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contour_convex_ratio = \
        lambda c: cv.contourArea(cv.convexHull(c)) / cv.contourArea(c)
    contours = [c for c in contours if 0.8 < contour_convex_ratio(c) < 1.2]
    cv.drawContours(img, contours, -1, 255, -1)
    img = cv.GaussianBlur(img, ksize=(9, 9), sigmaX=0)

    return img

def process_whitish(img: Image) -> Image:
    '''
    Process the image to enhance white signs using a band-pass threshold.
    This is can separate white signs from background, but is relatively noisy.
    '''
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.bilateralFilter(img, d=11, sigmaColor=25, sigmaSpace=75)
    imgA = cv.threshold(img, thresh=150, maxval=255, type=cv.THRESH_BINARY)[1]
    imgB = cv.threshold(img, thresh=230, maxval=255, type=cv.THRESH_BINARY_INV)[1]
    img = clear_borders(imgA & imgB)
    img = cv.GaussianBlur(img, ksize=(11, 11), sigmaX=0)

    return img

def rectangle_overlap(rectA: Rectangle, rectB: Rectangle) -> float:
    '''
    Return the number of overlapping pixels as a fraction of the smaller area.

    TODO: optimise this
    '''
    x1, y1, w1, h1 = rectA; area1 = w1 * h1
    x2, y2, w2, h2 = rectB; area2 = w2 * h2

    bounding_w = max(x1 + w1, x2 + w2) - min(x1, x2)
    bounding_h = max(y1 + h1, y2 + h2) - min(y1, y2)

    bounding_array_1 = np.zeros((bounding_w, bounding_h), dtype=np.uint8)
    bounding_array_2 = np.zeros((bounding_w, bounding_h), dtype=np.uint8)

    x_offset = min(x1, x2)
    y_offset = min(y1, y2)

    x1_offset = x1 - x_offset; y1_offset = y1 - y_offset
    x2_offset = x2 - x_offset; y2_offset = y2 - y_offset

    bounding_array_1[x1_offset:(x1_offset + w1), y1_offset:(y1_offset + h1)] = 1
    bounding_array_2[x2_offset:(x2_offset + w2), y2_offset:(y2_offset + h2)] = 1

    overlap = np.sum(bounding_array_1 & bounding_array_2)

    return overlap / min(area1, area2)

def remove_overlapping_rectangles(rect_value_pairs: List,
        max_overlap: float) -> List[Rectangle]:
    '''
    Remove overlapping rectangles from the list of (rectangle, value) pairs.
    Rectangles are sorted by value, and the highest value non-overlapping
    rectangles are returned.
    '''
    rect_value_pairs = sorted(rect_value_pairs, key=lambda t: t[1], reverse=True)

    # Store a list of confirmed non-overlapping rectangles
    non_overlapping = [rect_value_pairs[0][0],]

    # For each subsequent rectangle, check if it overlaps more than allowed
    for rect, _ in rect_value_pairs[1:]:
        for existing_rect in non_overlapping:
            if rectangle_overlap(rect, existing_rect) > max_overlap:
                break
        else:
            non_overlapping.append(rect)

    # Return the non-overlapping rectangles (without associated values)
    return non_overlapping

def match_signs(img: Image, templates: List) -> List[Rectangle]:
    '''
    Use template matching to identify the templates in the image.
    '''
    overall_max = 0

    # Store a list of matches above the threshold, to be sorted and filtered
    match_results = []

    # Iterate through all templates and scaling factors
    for template, scale_range in templates:
        for scale in scale_range:

            # Scale the template to match different sizes
            scaled_template = scale_template(template, scale)

            # Check that the scaled template is still smaller than the image
            if template_too_large(img, scaled_template):
                break

            # Attempt to match the template
            match = cv.matchTemplate(img, scaled_template, cv.TM_CCORR_NORMED)
            match_loc = np.where(match >= 0.85)

            _, max_val, _, _ = cv.minMaxLoc(match)
            overall_max = max(overall_max, max_val)

            # Define bounding rectangles for each match based on template size
            h, w = scaled_template.shape[:2]
            for y, x in zip(*match_loc):
                # Include the match value to be sorted on
                match_results.append(((x, y, w, h), match[y, x]))

    print('Overal max:', overall_max)

    # If no matches were found, return empty list
    if match_results == []:
        return []

    # Else, remove overlapping matches in the match results by match value
    return remove_overlapping_rectangles(match_results, max_overlap=0.20)

def find_signs(img: Image) -> List[Rectangle]:
    '''
    Attempt to find all red, yellow and white signs in the given image, and
    return a list of rectangles denoting the position of each sign.
    '''

    # Collect all detected signs in one list
    detected_signs: List[Rectangle] = []

    # Define templates for each red/yellow sign shape
    templates_reddish = [
        (template_diamond(),                 np.arange(0.45, 1.45, 0.10)),
        (template_triangle(),                np.arange(0.95, 1.45, 0.05)),
        (template_rectangle(aspect_ratio=5), np.arange(1.15, 1.30, 0.05)),
    ]

    # Attempt to match red/yellow signs
    detected_signs += match_signs(process_reddish(img), templates_reddish)

    # Define templates for each white sign shape
    templates_whitish = [
        (template_rectangle(aspect_ratio=2.6), np.arange(1.75, 1.95, 0.05)),
        (template_rectangle(aspect_ratio=0.8), np.arange(0.95, 1.05, 0.05)),
    ]

    # Attempt to match white signs
    detected_signs += match_signs(process_whitish(img), templates_whitish)

    # Return all detected signs
    return detected_signs


### TESTING & DEBUGGING ########################################################

def show_identification_basis(imgs: List[Image]):
    '''
    Show all 11 signs on one figure with the two types of processsing applied.
    This may help understand why a technique does or does not work.
    '''

    # Show all 11 images with saturation processing
    fig, axs = plt.subplots(3, 4)
    fig.tight_layout()
    for i, ax in enumerate(axs.flat):
        if i < 11:
            img = process_reddish(imgs[i])
            ax.imshow(img, cmap='gray')
    plt.show()

    # Show all 11 images with grayscale processing
    fig, axs = plt.subplots(3, 4)
    fig.tight_layout()
    for i, ax in enumerate(axs.flat):
        if i < 11:
            img = process_whitish(imgs[i])
            ax.imshow(img, cmap='gray')
    plt.show()

### ENTRYPOINT #################################################################

def main():

    # plt.imshow(template_triangle(), cmap='gray')
    # plt.show()

    # Get the filepath of the sample images
    imgsrc_fp = str(Path(A1_ROOT, 'data', 'street_signs'))
    for _, _, files in os.walk(imgsrc_fp):
        imgs_fp = sorted(files, key=lambda f: int(f[6:-4]))
        break

    # Read in the sample images
    imgs = [cv.imread(str(Path(imgsrc_fp, img_fp))) for img_fp in imgs_fp]

    # show_identification_basis(imgs)

    fig, axs = plt.subplots(3, 4)
    fig.tight_layout()
    for i, ax in enumerate(axs.flat):
        if i < 11:
            sign_rects = find_signs(imgs[i])
            for x, y, w, h in sign_rects:
                cv.rectangle(imgs[i], (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            ax.imshow(cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB), cmap='gray')
    plt.show()

    # # Find the signs in each image and draw a red box around each
    # for i, img in enumerate(imgs):
    #     sign_rects = find_signs(img)
    #     for x, y, w, h in sign_rects:
    #         cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255))

    #     # Show the annotated image
    #     plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    #     plt.show()

    #     # # Save the annotated image
    #     # save_fp = str(Path(A1_ROOT, 'output', 'street_signs', f'output{i}.png'))
    #     # cv.imwrite(save_fp, cv.cvtColor(img, cv.COLOR_BGR2RGB))


if __name__ == '__main__':
    main()
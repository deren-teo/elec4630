import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Path to the project root, i.e. "a1"
A1_ROOT = Path(__file__).parent.parent.resolve()


def process_image(img, thresh=0):
    '''
    Attempt to process an image containing one or more street signs into a
    binary image, where the area of any street signs is white, and the
    remaining area is black.
    '''
    
    # Yellow signs tend to have high saturation relative to background
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_sat = img_hsv[:, :, 1]

    # Blur saturation image
    # img_blurred = cv.blur(img_sat, ksize=(7, 7))
    # img_blurred = cv.GaussianBlur(img_sat, ksize=(11, 11), sigmaX=0)
    # img_blurred = cv.medianBlur(img_sat, ksize=7)
    img_blurred = cv.bilateralFilter(img_sat, d=9, sigmaColor=75, sigmaSpace=75)

    # Binarize saturation image
    maxval = np.amax(img_blurred)
    _, img_binarized = cv.threshold(img_blurred, thresh, maxval, cv.THRESH_BINARY)

    # img_blurred = cv.blur(img_binarized, ksize=(7, 7))

    kernel = np.ones((3, 3), np.uint8)
    img_opened = cv.morphologyEx(img_binarized, cv.MORPH_OPEN, kernel)

    return img_opened

# def segment_image(img):
#     '''TODO: documentation'''

#     kernel = np.ones((71, 71), np.uint8)
#     sure_bg = cv.dilate(img, kernel)

#     kernel = np.ones((71, 71), np.uint8)
#     sure_fg = cv.erode(img, kernel)

#     sure_fg = np.uint8(sure_fg)
#     unknown = cv.subtract(sure_bg, sure_fg)

#     _, markers = cv.connectedComponents(sure_fg)
#     markers = markers + 1
#     markers[unknown==255] = 0

#     img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#     markers = cv.watershed(img, markers)
#     img[markers==-1] = [255, 0, 0]

#     return markers


def bounding_rect(img, dst):
    ''''''
    x, y, w, h = cv.boundingRect(img)
    cv.rectangle(dst, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return dst


def main():
    '''TODO: documentation'''

    img_fp = Path(A1_ROOT, 'data', 'street_signs', 'images0.jpg')
    img_raw = cv.imread(str(img_fp))
    img_rgb = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)
    img_processed = process_image(img_raw, thresh=200)
    # img_segmented = bounding_rect(img_processed, img_rgb)

    plt.imshow(img_processed, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
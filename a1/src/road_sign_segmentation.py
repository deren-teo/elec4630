import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Path to the project root, i.e. "a1"
A1_ROOT = Path(__file__).parent.parent.resolve()


def process_image(img, alpha=0.4, beta=0.6, d=15, sigmaColor=25, sigmaSpace=75, thresh=185):
    '''
    TODO: documentation
    '''
    img_red = img[:, :, 2]

    # Yellow signs tend to have high saturation relative to background
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_sat = img_hsv[:, :, 1]

    # Combine red channel and saturation channel to reduce intensity of 
    # saturated but non-reddish (incl. non-yellow) background
    img_prod = cv.addWeighted(img_red, alpha, img_sat, beta, 0)

    # Blur saturation image using bilateral filter to preserve edges
    img_blurred = cv.bilateralFilter(img_prod, d, sigmaColor, sigmaSpace)

    # Binarize saturation image
    maxval = np.amax(img_blurred)
    _, img_binarized = cv.threshold(img_blurred, thresh, maxval, cv.THRESH_BINARY)

    return img_binarized


def bounding_rect(img, dst):
    '''TODO: documentation'''
    x, y, w, h = cv.boundingRect(img)
    cv.rectangle(dst, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return dst


def main():
    '''TODO: documentation'''

    fig, axs = plt.subplots(3, 4)
    
    for i, ax in enumerate(axs.flat):
        ax.axis('off')
        if i < 11:
            img_fp = Path(A1_ROOT, 'data', 'street_signs', f'images{i}.jpg')

            img_raw = cv.imread(str(img_fp))
            img_rgb = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)

            img_processed = process_image(img_raw)
            img_segmented = bounding_rect(img_processed, img_rgb)

            ax.imshow(img_segmented, cmap='gray')
    
    plt.show()


if __name__ == '__main__':
    main()
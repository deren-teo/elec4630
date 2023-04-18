import os

from pathlib import Path
from typing import List

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC

### GLOBAL VARIABLES ###########################################################

# Path to the project root, i.e. "a2"
A2_ROOT = Path(__file__).parent.parent.resolve()

### TYPEDEFS ###################################################################

# Define an "image" type, which is more descriptive than "np.ndarray"
Image = np.ndarray

### IMAGE PROCESSING ###########################################################

def load_images(filepath: Path) -> List[Image]:
    '''
    Load all files in the given filepath as images in sorted order.
    '''
    for _, _, files in os.walk(filepath):
        filenames = sorted(files)

    return [cv.imread(str(Path(filepath, filename))) for filename in filenames]

def prepare_images(images: List[Image]) -> np.ndarray:
    '''
    Convert a list of 3-channel grayscale images to single-channel,
    then flatten and stack into an array with one image per row.
    '''
    # Convert images from 3-channel grayscale to single-channel
    images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]

    # Flatten images and stack into an array
    return np.vstack(image.flatten() for image in images)

### ENTRYPOINT #################################################################

def main():

    # Load training images and prepare for PCA
    train_filepath = Path(A2_ROOT, 'data', 'face_recognition', 'eig')
    train_imgs = load_images(train_filepath)
    X_train = prepare_images(train_imgs)
    y_train = np.array(range(len(X_train)))

    # Load testing images and prepare for PCA; the test images are arranged
    # into folders, with the same faces grouped together
    X_test = []; y_test = []
    for i in range(6):
        test_i_filepath = Path(A2_ROOT, 'data', 'face_recognition', str(i + 1))
        test_i_imgs = load_images(test_i_filepath)
        X_test_i = prepare_images(test_i_imgs)
        X_test.append(X_test_i)
        y_test.append([i] * len(X_test_i))

    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    # Apply PCA to the training images to identify principal components
    pca = PCA().fit(X_train)

    # Use the PCA model to reduce the dimensionality of all images
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train an SVM classifier on the training images
    svm = SVC().fit(X_train_pca, y_train)

    # Test the trained classifier on the testing images
    y_pred = svm.predict(X_test_pca)

    # Observe the results
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()

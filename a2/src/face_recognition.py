import os

from pathlib import Path
from typing import List

import cv2 as cv
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg

from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.utils.extmath import svd_flip

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
    return np.vstack([image.flatten() for image in images])

### EIGENFACE PCA ##############################################################

class ManualPCA:
    '''
    An implementation based on the sklearn.decomposition.PCA class.
    Intentionally adopts a very similar but simpler interface.
    '''

    # Principle components in feature space, representing the directions of
    # maximum variance in the data. Equivalently, the right singular vectors
    # of the centred input data, parallel to its eigenvectors. The components
    # are sorted by decreasing `explained_variance_`.
    components_: np.ndarray = None

    # The amount of variance explained by each of the selected components.
    # The variance estimation uses `n_samples - 1` degrees of freedom.
    #
    # Equal to n_components largest eigenvalues of the covariance matrix of X.
    explained_variance_: np.ndarray = None

    def fit(self, X: np.ndarray):
        '''
        Fit the model with X.

        Parameters:
            X - training data; rows are samples and columns are features.

        Returns:
            Instance itself (of the PCA class).
        '''
        # Centre data
        X = X - np.mean(X, axis=0)

        # Singular value decomposition
        U, S, V = linalg.svd(X, full_matrices=False)

        # Flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)

        # Eigenvectors are the rows of V
        self.components_ = V

        # Get variance explained by singular values
        self.explained_variance_ = (S**2) / (X.shape[0] - 1)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        '''
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters:
            X - New data; rows are samples and columns are features.

        Returns:
            Projection of X in the first principal components, where the number
            of columns is reduced to the number of components.
        '''
        return X @ self.components_.T

### GUI FUNCTIONALITY ##########################################################

def face_classifier(face: Image) -> Image:
    '''
    Wrapper for classification functionality to use with Gradio GUI.

    Given an input image from the testing set, the corresponding image in
    the training set is returned.

    Parameters:
        face - an image in numpy array format from the testing set

    Returns:
        Matching face (image) in numpy array format from the training set.
    '''
    face = prepare_images([face,])
    face_pca = pca.transform(face)
    pred_idx = int(svm.predict(face_pca))
    return train_imgs[pred_idx]

### ENTRYPOINT #################################################################

def main():

    # Load training images and prepare for PCA
    train_filepath = Path(A2_ROOT, 'data', 'face_recognition', 'eig')
    global train_imgs
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

    # Apply PCA to the training images to identify principal components;
    # define it as a global variable to be accessed by the GUI
    global pca
    pca = PCA().fit(X_train)

    for ef in pca.components_:
        plt.imshow(ef.reshape((128, 128)), cmap='gray')
        plt.show()

    # Use the PCA model to reduce the dimensionality of all images
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train an SVM classifier on the training images; define it as a global
    # variable to be accessed by the GUI
    global svm
    svm = SVC().fit(X_train_pca, y_train)

    # Test the trained classifier on the testing images
    y_pred = svm.predict(X_test_pca)

    # Observe the results
    print(classification_report(y_test, y_pred))

    # Create the GUI
    app = gr.Interface(fn=face_classifier, inputs="image", outputs="image")

    # Launch the GUI
    app.launch()


if __name__ == '__main__':

    # Prompt the user to select a PCA implementation
    print('Select a PCA implementation:')
    print('[1] sklearn.decomposition.PCA')
    print('[2] Own manual implementation')
    print('Enter either "1" or "2": ')

    user_input = input()
    if user_input == "1":
        PCA = SklearnPCA
    elif user_input == "2":
        PCA = ManualPCA
    else:
        raise ValueError("Invalid input")

    main()

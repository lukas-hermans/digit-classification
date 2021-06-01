import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    """
    Load data from csv-file.

    Parameters
    ----------
    path : str
        Full path of the location of the file (e.g., "../data/mnist_test.csv").
        The first column of the file should contain the labels,
        while the remaining columns are expected to be the features.

    Returns
    -------
    feats : numpy.ndarray of shape (n, p)
        Features, where n is the number of examples
        and p the number of features.
    label : numpy.ndarray of shape (n, 1)
        Label.

    """

    data = np.loadtxt(path, skiprows=1, delimiter=",")

    feats = data[:1000, 1:]
    label = data[:1000, 0]

    return feats, label


def make_binary(label, which):
    """
    Transforms label in one vs. all encoding for binary classification.

    Parameters
    ----------
    label : numpy.ndarray of shape (n, 1)
        Label.
    which : int
        Specifies which class is interpreted as 1.
        All other classes are set to -1.

    Returns
    -------
    binary_label : numpy.ndarray of shape (n, 1)
        Binary label.

    """

    binary_label = label.copy()

    binary_label[np.argwhere(label == which)] = 1
    binary_label[np.argwhere(label != which)] = -1

    return binary_label


def plot_examples(feats, path):
    """
    Plot the images from the features (each feature should be a brightness
                                       value between 0 and 255).

    Parameters
    ----------
    feats : numpy.ndarray of shape (n, p)
        Features, where n is the number of examples
        and p the number of features.
    path : str
        Path where the image should be saved
        (e.g., "figures/").

    Returns
    -------
    None.

    """

    n = np.shape(feats)[0]

    fig, ax = plt.subplots(1, n, figsize=(10, 3))

    # add each image to its own panel in figure
    for i in range(n):
        # take ith example and bring it into correct shape
        img = feats[i]
        n_pxl = len(img)
        n_pxl_side = int(np.sqrt(n_pxl))
        img = img.reshape((n_pxl_side, n_pxl_side))

        ax[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        ax[i].set_axis_off()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0)
    plt.savefig(path + "plot_examples.pdf",
                bbox_inches="tight", pad_inches=0.0)

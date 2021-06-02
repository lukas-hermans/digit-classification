"""
    imports
"""

import data
import perceptron
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")  # close all open plot windows


"""
    user input
"""

path_train = "../data/mnist_train.csv"
path_test = "../data/mnist_test.csv"

path_figs = "figures/"

n_epoch = np.arange(1, 11)  # number of epochs over whole training set
epoch_size = 10000
deg = np.arange(1, 7)  # degree of polynomial kernel


"""
    load data
"""

# load training and test data
feats_train, label_train = data.load_data(path_train)
feats_test, label_test = data.load_data(path_test)

# plot examples for digits from 0 to 9
example_ind = [3, 2, 1, 18, 24, 23, 21, 17, 61, 62]
# data.plot_examples(feats_train[example_ind, :], path_figs)

# plot histograms of data
# data.plot_hist(label_train, path_figs, "plot_hist_train.pdf")
# data.plot_hist(label_test, path_figs, "plot_hist_test.pdf")


"""
    train & test kernel perceptron
"""

# lists to store errors for three different classifier types
training_error = []
training_error_best = []
training_error_mean = []
test_error = []
test_error_best = []
test_error_mean = []

kernel_perceptron = perceptron.KernelPerceptron(
    feats_train, label_train)

for i in n_epoch:
    for j in deg:
        kernel_perceptron.reset(i, epoch_size, j)
        kernel_perceptron.train()
        test_error.append(kernel_perceptron.compute_error(
            feats_test, label_test, "best"))
        training_error.append(kernel_perceptron.compute_error(
            feats_train, label_train, "best"))

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

n_epoch = np.arange(3, 4)  # number of epochs over whole training set
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


"""
    train & test kernel perceptron
"""

training_error = []
test_error = []

kernel_perceptron = perceptron.KernelPerceptron(
    feats_train, label_train)

for i in n_epoch:
    for j in deg:
        kernel_perceptron.reset(i, j)
        kernel_perceptron.train()
        test_error.append(kernel_perceptron.compute_error(feats_test, label_test, "best"))
        training_error.append(kernel_perceptron.compute_error(feats_train, label_train, "best"))
        
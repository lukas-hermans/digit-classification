"""
    imports
"""

import data
import perceptron
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")  # close all open plot windows

# set seed
np.random.seed(123)


"""
    data paths
"""

path_train = "../data/mnist_train.csv"
path_test = "../data/mnist_test.csv"

path_figs = "figures/"
path_results = "results/"


"""
    load data
"""

# load training and test data
feats_train, label_train = data.load_data(path_train)
feats_test, label_test = data.load_data(path_test)

# plot examples for digits from 0 to 9
example_ind = [3, 2, 1, 18, 24, 23, 21, 17, 61, 62]
data.plot_examples(feats_train[example_ind, :], path_figs)

# plot histograms of data
data.plot_hist(label_train, path_figs, "plot_hist_train.pdf")
data.plot_hist(label_test, path_figs, "plot_hist_test.pdf")


"""
    train & test kernel perceptron
"""

# initialize kernel perceptron instance
kernel_perceptron = perceptron.KernelPerceptron(
    feats_train, label_train)

### compute dependence on partition in epochs and sample size ###

# parameters (user input)
n_epoch = [1, 2, 4, 8, 16]
n_sample = [2000, 1000, 500, 250, 125]
deg = 3

# repeat computation three times to cover statistics
for rep in range(3):
    # lists to store errors for three different classifier types
    training_error = []
    training_error_min = []
    training_error_avg = []
    test_error = []
    test_error_min = []
    test_error_avg = []

    # computation of errors for different combinations of n_epoch and n_sample
    # leading to the same total number of training examples
    # errors are computed from n_rep repetitions of the computation
    for i in range(len(n_epoch)):
        kernel_perceptron.reset(n_epoch[i], n_sample[i], deg)
        kernel_perceptron.train()
        test_error.append(kernel_perceptron.compute_error(
            feats_test, label_test, "standard"))
        training_error.append(kernel_perceptron.compute_error(
            feats_train, label_train, "standard"))
        test_error_min.append(kernel_perceptron.compute_error(
            feats_test, label_test, "min"))
        training_error_min.append(kernel_perceptron.compute_error(
            feats_train, label_train, "min"))
        test_error_avg.append(kernel_perceptron.compute_error(
            feats_test, label_test, "avg"))
        training_error_avg.append(kernel_perceptron.compute_error(
            feats_train, label_train, "avg"))

    # save results in file
    header = "deg = " + str(deg) + "\n" + \
        "n_epoch * n_sample: " + str(n_epoch) + " * " + str(n_sample) + "\n" +\
        "The first three rows contain the test errors of the different" +\
        "while the last three contain their training errors."
    data = np.array([training_error, training_error_min, training_error_avg,
                     test_error, test_error_min, test_error_avg])
    np.savetxt(path_results + "partition/partition" +
               str(rep) + ".txt", data,  header=header)

### training and test error for different choices of hyperparameters ###

# parameters (user input)
n_epoch = np.arange(1, 11)
n_sample = 1000
deg = np.arange(1, 7)

# compute errors for different hyperparameters
for i in deg:
    # lists to store errors for three different classifier types
    training_error = []
    training_error_min = []
    training_error_avg = []
    test_error = []
    test_error_min = []
    test_error_avg = []

    for j in n_epoch:
        kernel_perceptron.reset(i, n_sample, j)
        # output binary training error in one particular case
        if i == 3 and j == 1:
            kernel_perceptron.reset(i, n_sample, j, True, 1)
        kernel_perceptron.train()
        test_error.append(kernel_perceptron.compute_error(
            feats_test, label_test, "standard", path_results, True))
        training_error.append(kernel_perceptron.compute_error(
            feats_train, label_train, "standard"))
        test_error_min.append(kernel_perceptron.compute_error(
            feats_test, label_test, "min"))
        training_error_min.append(kernel_perceptron.compute_error(
            feats_train, label_train, "min", path_results, True))
        test_error_avg.append(kernel_perceptron.compute_error(
            feats_test, label_test, "avg"))
        training_error_avg.append(kernel_perceptron.compute_error(
            feats_train, label_train, "avg", path_results, True))

    # save results of current degree in file
    data = np.array([training_error, training_error_min, training_error_avg,
                     test_error, test_error_min, test_error_avg])
    np.savetxt(path_results + "performance/performance_deg=" + str(i) +
               ".txt", data,
               delimiter=",", header="n_epoch: " + str(n_epoch))

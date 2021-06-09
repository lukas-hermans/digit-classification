"""
    imports
"""

import data
import perceptron
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")  # close all open plot windows


"""
    data paths
"""

path_train = "../data/mnist_train.csv"
path_test = "../data/mnist_test.csv"

path_figs = "./figures/"
path_results = "./results/"


"""
    load data
"""

# load training and test data
feats_train, label_train = data.load_data(path_train)
feats_test, label_test = data.load_data(path_test)

# plot examples for digits from 0 to 9
# example_ind = [3, 2, 1, 18, 24, 23, 21, 17, 61, 62]
# data.plot_examples(feats_train[example_ind, :], path_figs)

# plot histograms of data
# data.plot_hist(label_train, path_figs, "plot_hist_train.pdf")
# data.plot_hist(label_test, path_figs, "plot_hist_test.pdf")


"""
    train & test kernel perceptron
"""

# initialize kernel perceptron instance
kernel_perceptron = perceptron.KernelPerceptron(feats_train, label_train,
                                                feats_test, label_test)

# hyperparameters
n_epoch = 10
deg_list = np.arange(1, 7)

# compute errors for different hyperparameters
for deg in deg_list:

    # set seed
    np.random.seed(deg)

    # lists to store errors for three different classifier types
    training_error = []
    training_error_min = []
    training_error_avg = []
    test_error = []
    test_error_min = []
    test_error_avg = []

    # set degree of kernel perceptron instance
    kernel_perceptron.reset(deg)

    # run kernel perceptron and store training and test error after each epoch
    for j in range(n_epoch):
        # train kernel perceptron over all training examples (shuffled)
        kernel_perceptron.train()

        # compute training and test error for different ways
        test_error.append(
            kernel_perceptron.compute_error("test", "standard",
                                            path_results, True))
        training_error.append(
            kernel_perceptron.compute_error("train", "standard"))
        test_error_min.append(
            kernel_perceptron.compute_error("test", "min",
                                            path_results, True))
        training_error_min.append(
            kernel_perceptron.compute_error("train", "min"))
        test_error_avg.append(
            kernel_perceptron.compute_error("test", "avg",
                                            path_results, True))
        training_error_avg.append(
            kernel_perceptron.compute_error("train", "avg"))

    # save results of current degree of kernel in file
    data = np.array([
        training_error, training_error_min, training_error_avg, test_error,
        test_error_min, test_error_avg
    ])
    np.savetxt(path_results + "performance/deg=" + str(deg) + ".txt",
               data,
               delimiter=", ",
               header="n_epoch = " + str(n_epoch) +
               "\nfile contains errors as a function of n_epoch" +
               "\nrows: training error (for fin, min, avg), " +
               "test error (for fin, min, avg)")

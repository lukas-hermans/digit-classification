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
example_ind = [3, 2, 1, 18, 24, 23, 21, 17, 61, 62]
# data.plot_examples(feats_train[example_ind, :], path_figs)

# plot histograms of data
# data.plot_hist(label_train, path_figs, "plot_hist_train.pdf")
# data.plot_hist(label_test, path_figs, "plot_hist_test.pdf")


"""
    train & test kernel perceptron
"""

### compute dependence on partition in epochs and sample size ###

# # set seed
# np.random.seed(123)

# # parameters
# n_epoch = [1, 2, 4, 8]
# n_sample = [20000, 10000, 5000, 2500]
# deg = 4

# initialize kernel perceptron instance
kernel_perceptron = perceptron.KernelPerceptron(feats_train, label_train)

# # repeat computation three times to cover statistics
# for rep in range(3):
#     # lists to store errors for three different classifier types
#     training_error = []
#     training_error_min = []
#     training_error_avg = []
#     test_error = []
#     test_error_min = []
#     test_error_avg = []

#     # computation of errors for different combinations of n_epoch and n_sample
#     # leading to the same total number of training examples
#     # errors are computed from n_rep repetitions of the computation
#     for i in range(len(n_epoch)):
#         kernel_perceptron.reset(n_sample[i], deg, n_epoch[i])

#         for j in range(n_epoch[i]):
#             kernel_perceptron.train()

#         test_error.append(kernel_perceptron.compute_error(
#             feats_test, label_test, "standard"))
#         training_error.append(kernel_perceptron.compute_error(
#             feats_train, label_train, "standard"))
#         test_error_min.append(kernel_perceptron.compute_error(
#             feats_test, label_test, "min"))
#         training_error_min.append(kernel_perceptron.compute_error(
#             feats_train, label_train, "min"))
#         test_error_avg.append(kernel_perceptron.compute_error(
#             feats_test, label_test, "avg"))
#         training_error_avg.append(kernel_perceptron.compute_error(
#             feats_train, label_train, "avg"))

#     # save results in file
#     header = "deg = " + str(deg) + "\n" + \
#         "n_epoch * n_sample: " + str(n_epoch) + " * " + str(n_sample) + "\n" +\
#         "The first three rows contain the training errors of the different" +\
#         " predictors" +\
#         "while the last three contain their test errors."
#     data = np.array([training_error, training_error_min, training_error_avg,
#                      test_error, test_error_min, test_error_avg])
#     np.savetxt(path_results + "partition/partition" +
#                str(rep) + ".txt", data,  header=header)

### training and test error for different choices of hyperparameters ###

# parameters(user input)
n_epoch = 50
n_sample = 5000
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

    # set seed
    np.random.seed(i)

    kernel_perceptron.reset(n_sample, i, n_epoch)
    # output binary training error in one particular case
    #if i == 3:
    #    kernel_perceptron.reset(n_sample, i, n_epoch, True, 1)

    for j in range(n_epoch):
        kernel_perceptron.train()
        test_error.append(
            kernel_perceptron.compute_error(feats_test, label_test, "standard",
                                            path_results, True))
        training_error.append(
            kernel_perceptron.compute_error(feats_train, label_train,
                                            "standard"))
        test_error_min.append(
            kernel_perceptron.compute_error(feats_test, label_test, "min",
                                            path_results, True))
        training_error_min.append(
            kernel_perceptron.compute_error(feats_train, label_train, "min"))
        test_error_avg.append(
            kernel_perceptron.compute_error(feats_test, label_test, "avg",
                                            path_results, True))
        training_error_avg.append(
            kernel_perceptron.compute_error(feats_train, label_train, "avg"))

    # save results of current degree of kernel in file
    data = np.array([
        training_error, training_error_min, training_error_avg, test_error,
        test_error_min, test_error_avg
    ])
    np.savetxt(path_results + "performance/performance_deg=" + str(i) + ".txt",
               data,
               delimiter=", ",
               header="n_epoch = " + str(n_epoch) +
               " with n_sample =" + str(n_sample))

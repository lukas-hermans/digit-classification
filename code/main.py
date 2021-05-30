"""
    imports
"""

import data
import perceptron
import matplotlib.pyplot as plt

plt.close("all")


"""
    user input
"""

path_train = "../data/mnist_train.csv"
path_test = "../data/mnist_test.csv"

path_figs = "figures/plot_examples.pdf"

# load training and test data
feats_train, label_train = data.load_data(path_train)
feats_test, label_test = data.load_data(path_test)

# plot examples for digits from 0 to 9
example_ind = [3, 2, 1, 18, 24, 23, 21, 17, 61, 62]
# data.plot_examples(feats_train[example_ind, 1:], path_figs)

"""
    train kernel perceptron
"""

label = data.make_binary(label_train, 3)
kernel_perceptron = perceptron.KernelPerceptron(feats_train, label)
kernel_perceptron.train(2, 1)

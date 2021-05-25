'''
    imports
'''

import data
import matplotlib.pyplot as plt

plt.close("all")


'''
    user input
'''

path_train = "../data/mnist_test.csv"
path_test = "../data/mnist_test.csv"

path_figs = "figures/"


# load training and test data
feats_train, label_train = data.load_data(path_train)
feats_test, label_test = data.load_data(path_test)

# plot examples for digits from 0 to 9
example_ind = [3, 2, 1, 18, 24, 23, 21, 17, 61, 62]
data.plot_examples(feats_train[example_ind, 1:], path_figs)

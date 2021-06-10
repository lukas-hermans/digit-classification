import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn


plt.close("all")


"""
    comparison of different predictor types
"""

a = 5  # plots for digit a
deg_list = np.arange(1, 7)
n_epoch = np.arange(1, 11)
figsize = (5, 4)
size = 13

path_results = "./results/bin_pred/"
path_figs = "./figures/bin_pred/"

# training error
for deg in deg_list:
    training_error, test_error,\
        training_error_min, test_error_min,\
        training_error_avg, test_error_avg =\
        np.loadtxt(path_results + "digit=" + str(a) + "_deg=" + str(deg)
                   + ".txt",
                   delimiter=",", unpack=True,
                   skiprows=1)

    plt.figure(figsize=figsize)

    plt.plot(n_epoch, training_error * 100,
             label=r"final: $\vec{\alpha}_{fin}$",
             color="r", linestyle="solid")
    plt.plot(n_epoch, training_error_min * 100,
             label=r"minimizing: $\vec{\alpha}_{min}$",
             color="b", linestyle="dotted")
    plt.plot(n_epoch, training_error_avg * 100,
             label=r"average: $\langle\vec{\alpha}\rangle$",
             color="green", linestyle="dashed")

    plt.tight_layout()
    plt.ylim(0)
    plt.grid()
    plt.legend(fontsize=size)
    plt.xticks(n_epoch)
    plt.xlabel(r"$N_{epoch}$", size=size)
    plt.ylabel(
        r"binary training error $\hat{\ell}_{D^{(5)}}\,[\%]$", size=size)
    plt.tight_layout()
    plt.savefig(path_figs + "training_digit=" + str(a) + "_deg=" + str(deg)
                + ".pdf",
                bbox_inches="tight", pad_inches=0.0)

# test error
for deg in deg_list:
    training_error, test_error,\
        training_error_min, test_error_min,\
        training_error_avg, test_error_avg =\
        np.loadtxt(path_results + "digit=" + str(a) + "_deg=" + str(deg)
                   + ".txt",
                   delimiter=",", unpack=True,
                   skiprows=1)

    plt.figure(figsize=figsize)

    plt.plot(n_epoch, test_error * 100,
             label=r"final: $\vec{\alpha}_{fin}$",
             color="r", linestyle="solid")
    plt.plot(n_epoch, test_error_min * 100,
             label=r"minimizing: $\vec{\alpha}_{min}$",
             color="b", linestyle="dotted")
    plt.plot(n_epoch, test_error_avg * 100,
             label=r"average: $\langle\vec{\alpha}\rangle$",
             color="green", linestyle="dashed")

    plt.tight_layout()
    plt.ylim(0)
    plt.grid()
    plt.legend(fontsize=size)
    plt.xticks(n_epoch)
    plt.xlabel(r"$N_{epoch}$", size=size)
    plt.ylabel(
        r"binary test error $\hat{\ell}_{D^{(5)}}\,[\%]$", size=size)
    plt.tight_layout()
    plt.savefig(path_figs + "test_digit=" + str(a) + "_deg=" + str(deg)
                + ".pdf",
                bbox_inches="tight", pad_inches=0.0)


"""
    find predictors with lowest test errors
"""

path_results = "./results/performance/"
path_figs = "./figures/performance/"

training_error, test_error = [], []
training_error_min, test_error_min = [], []
training_error_avg, test_error_avg = [], []

for deg in deg_list:
    data = np.loadtxt(path_results + "deg=" + str(deg) +
                      ".txt", skiprows=2, delimiter=",")

    training_error.append(data[0]*100)
    training_error_min.append(data[1]*100)
    training_error_avg.append(data[2]*100)

    test_error.append(data[3]*100)
    test_error_min.append(data[4]*100)
    test_error_avg.append(data[5]*100)

print("minimal test error rates:" + "\n")

ind_list, ind_min_list, ind_avg_list = [], [], []

# find index of minimum test error for each degree
for i, deg in enumerate(deg_list):
    ind = np.argmin(test_error[i])
    ind_min = np.argmin(test_error_min[i])
    ind_avg = np.argmin(test_error_avg[i])

    ind_list.append(ind)
    ind_min_list.append(ind_min)
    ind_avg_list.append(ind_avg)

    print("deg=" + str(deg))
    print("final (epoch=" + str(ind + 1) + ")): test=" + str(test_error[i][ind]) +
          ", training=" + str(training_error[i][ind]))
    print("minimizing (epoch=" + str(ind_min + 1) + ")): test=" + str(test_error_min[i][ind_min]) +
          ", training=" + str(training_error_avg[i][ind_min]))
    print("average (epoch=" + str(ind_avg + 1) + ")): test=" + str(test_error_avg[i][ind_avg]) +
          ", training=" + str(training_error_avg[i][ind_avg]) + "\n")


"""
    performance
"""

# training error
for i, deg in enumerate(deg_list):
    data = np.loadtxt(path_results + "deg=" + str(deg) +
                      ".txt", skiprows=2, delimiter=",")

    plt.figure(figsize=figsize)

    plt.plot(n_epoch, data[0]*100, label=r"final: $\vec{\alpha}_{fin}$",
             color="r", linestyle="solid")
    plt.plot(
        n_epoch, data[1]*100, label=r"minimizing: $\vec{\alpha}_{min}$",
        color="b", linestyle="dotted")
    plt.plot(
        n_epoch, data[2]*100, label=r"average: $\langle\vec{\alpha}\rangle$",
        color="green", linestyle="dashed")

    plt.tight_layout()
    plt.grid()
    plt.ylim(0)
    plt.xticks(n_epoch)
    plt.xlabel(r"$N_{epoch}$", size=size)
    plt.ylabel(r"multiclass training error $\hat{\ell}_S\,[\%]$")
    plt.legend(fontsize=size)
    plt.tight_layout()
    plt.savefig(path_figs + "training_deg=" + str(deg) + ".pdf",
                bbox_inches="tight", pad_inches=0.0)

# test error
for i, deg in enumerate(deg_list):
    data = np.loadtxt(path_results + "deg=" + str(deg) +
                      ".txt", skiprows=1, delimiter=",")

    plt.figure(figsize=figsize)

    plt.plot(n_epoch, data[3]*100, label=r"final: $\vec{\alpha}_{fin}$",
             color="r", linestyle="solid")
    plt.vlines(n_epoch[ind_list[i]], 0, 1.5*np.max(data[3:]*100),
               color="r", linestyle="solid", linewidth=2)
    plt.plot(
        n_epoch, data[4]*100, label=r"minimizing: $\vec{\alpha}_{min}$",
        color="b", linestyle="dotted")
    plt.vlines(n_epoch[ind_min_list[i]], 0, 1.5*np.max(data[3:]*100),
               color="b", linestyle="dotted", linewidth=4)
    plt.plot(
        n_epoch, data[5]*100, label=r"average: $\langle\vec{\alpha}\rangle$",
        color="green", linestyle="dashed")
    plt.vlines(n_epoch[ind_avg_list[i]], 0, 1.5*np.max(data[3:]*100),
               color="green", linestyle="dashed", linewidth=3)

    plt.tight_layout()
    plt.grid()
    plt.xticks(n_epoch)
    plt.xlabel(r"$N_{epoch}$", size=size)
    plt.ylabel(r"multiclass test error $\hat{\ell}_D\,[\%]$", size=size)
    plt.legend(fontsize=size)
    plt.tight_layout()
    plt.savefig(path_figs + "test_deg=" + str(deg) + ".pdf",
                bbox_inches="tight", pad_inches=0.0)


"""
    confusion matrix
"""

path_results = "./results/conf_mat/"
path_figs = "./figures/conf_mat/"

load_file = ["conf_mat_standard_n_epoch=9_deg=3.txt",
             "conf_mat_min_n_epoch=6_deg=3.txt",
             "conf_mat_avg_n_epoch=10_deg=4.txt"]
save_file = ["predictor.pdf",
             "min_predictor.pdf",
             "avg_predictor.pdf"]

for i in range(len(load_file)):
    conf_mat = pd.read_csv(
        path_results + load_file[i], header=None)  # load matrix

    plt.figure()
    ax = sn.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d",
                    annot_kws={"size": size})

    # add border to plot
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    plt.xlabel(r"predicted label $\hat{y}$", size=size)
    plt.ylabel(r"true label $y$", size=size)
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig(path_figs + save_file[i],
                bbox_inches="tight", pad_inches=0.0)

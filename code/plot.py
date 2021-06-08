import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn


plt.close("all")


"""
    confusion matrix
"""

path_results = "./results/conf_mat/"
path_figs = "./figures/conf_mat/"

load_file = ["conf_mat_avg_n_epoch=1_deg=1.txt",
             "conf_mat_avg_n_epoch=1_deg=2.txt",
             "conf_mat_avg_n_epoch=10_deg=3.txt"]
save_file = ["predictor.pdf",
             "min_predictor.pdf",
             "avg_predictor.pdf"]

for i in range(len(load_file)):
    conf_mat = pd.read_csv(
        path_results + load_file[i], header=None)  # load matrix

    plt.figure()
    ax = sn.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d")

    # add border to plot
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    plt.xlabel(r"predicted label $\hat{y}$")
    plt.ylabel(r"true label $y$")
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig(path_figs + save_file[i],
                bbox_inches="tight", pad_inches=0.0)

"""
    comparison of different predictor types
"""

# path_results = "./results/"
# path_figs = "./figures/"

# iteration, training_error = np.loadtxt(path_results + "output_bin.txt",
#                                        delimiter=",", unpack=True,
#                                        skiprows=3)

# plt.figure()

# plt.plot(iteration[:], training_error[:],
#          label=r"final: $\vec{\alpha}_{fin}$",
#          color="r", linestyle="solid")

# plt.tight_layout()
# plt.grid()
# plt.legend()
# plt.xlabel(r"# iteration")
# plt.ylabel(r"binary training error $\hat{\ell}_{S^{(a)}}$")
# plt.tight_layout()
# plt.savefig(path_figs + "output_bin.pdf",
#             bbox_inches="tight", pad_inches=0.0)


"""
    performance
"""

# path_results = "./results/performance/"
# path_figs = "./figures/performance/"

# n_epoch = np.arange(1, 11)
# deg = np.arange(1, 7)

# # training error
# for i in deg:
#     data = np.loadtxt(path_results + "deg=" + str(i) +
#                       ".txt", skiprows=2, delimiter=",")

#     plt.figure(figsize=(7, 4))

#     plt.plot(n_epoch, data[0], label=r"final: $\vec{\alpha}_{fin}$",
#              color="r", linestyle="solid")
#     plt.plot(
#         n_epoch, data[1], label=r"minimizing: $\vec{\alpha}_{min}$",
#         color="b", linestyle="dotted")
#     plt.plot(
#         n_epoch, data[2], label=r"average: $\langle\vec{\alpha}\rangle$",
#         color="green", linestyle="dashed")

#     plt.tight_layout()
#     plt.grid()
#     plt.xlabel(r"$n_{epoch}$")
#     plt.ylabel(r"training error $\hat{\ell}_S$")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(path_figs + "training_performance_deg=" + str(i),
#                 bbox_inches="tight", pad_inches=0.0)

# # test error
# for i in deg:
#     data = np.loadtxt(path_results + "deg=" + str(i) +
#                       ".txt", skiprows=1, delimiter=",")

#     plt.figure(figsize=(7, 4))

#     plt.plot(n_epoch, data[3], label=r"final: $\vec{\alpha}_{fin}$",
#              color="r", linestyle="solid")
#     plt.plot(
#         n_epoch, data[4], label=r"minimizing: $\vec{\alpha}_{min}$",
#         color="b", linestyle="dotted")
#     plt.plot(
#         n_epoch, data[5], label=r"average: $\langle\vec{\alpha}\rangle$",
#         color="green", linestyle="dashed")

#     plt.tight_layout()
#     plt.grid()
#     plt.xlabel(r"$n_{epoch}$")
#     plt.ylabel(r"test error $\hat{\ell}_D$")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(path_figs + "test_performance_deg=" + str(i),
#                 bbox_inches="tight", pad_inches=0.0)

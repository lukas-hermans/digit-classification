import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn


plt.close("all")


"""
    confusion matrix
"""

# path_results = "./results/conf_mat/"
# path_figs = "./figures/conf_mat/"

# load_file = ["conf_mat_avg_n_epoch=1_deg=1.txt",
#              "conf_mat_avg_n_epoch=1_deg=2.txt",
#              "conf_mat_avg_n_epoch=10_deg=3.txt"]
# save_file = ["predictor.pdf",
#              "min_predictor.pdf",
#              "avg_predictor.pdf"]

# for i in range(len(load_file)):
#     conf_mat = pd.read_csv(
#         path_results + load_file[i], header=None)  # load matrix

#     plt.figure()
#     ax = sn.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d")

#     # add border to plot
#     for _, spine in ax.spines.items():
#         spine.set_visible(True)

#     plt.xlabel(r"predicted label $\hat{y}$")
#     plt.ylabel(r"true label $y$")
#     ax.xaxis.tick_top()  # x axis on top
#     ax.xaxis.set_label_position('top')
#     plt.tight_layout()
#     plt.savefig(path_figs + save_file[i],
#                 bbox_inches="tight", pad_inches=0.0)


"""
    performance
"""

path_results = "./results/performance/"
path_figs = "./figures/performance/"

n_epoch = np.arange(1, 11)
deg = np.arange(1, 7)

for i in n_epoch:
    data = np.loadtxt(path_results + "performance_n_epoch=" + str(i) +
                      ".txt", skiprows=1, delimiter=",")

    plt.figure(figsize=(7, 4))

    plt.plot(deg, data[3], label=r"final: $\vec{\alpha}_{fin}$", color="r")
    plt.plot(
        deg, data[4], label=r"average: $\langle\vec{\alpha}\rangle$", color="b")
    plt.plot(
        deg, data[5], label=r"minimizing: $\vec{\alpha}_{min}$", color="green")

    plt.tight_layout()
    plt.grid()
    plt.xlabel(r"degree $p$ of polynomial kernel")
    plt.ylabel(r"test error $\hat{\ell}_D$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_figs + "performance_deg=" + str(i),
                bbox_inches="tight", pad_inches=0.0)

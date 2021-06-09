import numpy as np
from data import make_binary


class KernelPerceptron:

    def __init__(self, feats_train, label_train,
                 feats_test, label_test):
        """
        Initialization of multiclass kernel perceptron.
        Class is written for application on recognition of handwritten digits.

        Parameters
        ----------
        feats_train : numpy.ndarray of shape (n, p)
            Training features, where n is the number of examples
            and p the number of features.
        label_train : numpy.ndarray of shape (n, 1)
            Training label.
        feats_test : numpy.ndarray of shape (m, p)
            Test features, where m is the number of examples
            and p the number of features.
        label_test : numpy.ndarray of shape (n, 1)
            Test label.

        Returns
        -------
        None.

        """

        self.feats_train = feats_train
        self.label_train = label_train
        self.feats_test = feats_test
        self.label_test = label_test
        self.n_train = len(label_train)  # total number of training examples
        self.n_test = len(label_test)  # total number of test examples

        # possible digits
        self.digits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # binary training data for each digit
        self.z = np.zeros([self.n_train, len(self.digits)])
        for digit in self.digits:
            self.z[:, digit] = make_binary(self.label_train, digit)

        # binary test data for each digit
        self.z_test = np.zeros([self.n_test, len(self.digits)])
        for digit in self.digits:
            self.z_test[:, digit] = make_binary(self.label_test, digit)

        print("initialization of kernel perceptron successfull\n")

    def reset(self, deg):
        """
        Reset kernel perceptron instance for a new run.

        Parameters
        ----------
        deg : int
            Degree of polynomial kernel.

        Returns
        -------
        None.

        """

        self.deg = deg
        self.n_epoch = 1  # number of current epoch

        # alpha lists contain numpy.ndarray of shape (n, 1) for every digit
        # final multiclass predictor
        self.alpha_list = len(self.digits) * [np.zeros(self.n_train)]
        # minimizing multiclass predictor w.r.t. training error
        self.alpha_min_list = len(self.digits) * [np.zeros(self.n_train)]
        self.alpha_prev_list = len(self.digits) * [np.zeros(self.n_train)]
        # average multiclass predictor from all epochs
        self.alpha_avg_list = len(self.digits) * [np.zeros(self.n_train)]

        # lists for (on the fly) computation of minimizing predictor
        self.mistakes_min_list = len(self.digits) * [self.n_train]
        self.g_min_list = len(self.digits) * [np.zeros(self.n_train)]

    def compute_kernel(self, x1, x2):
        """
        Compute polynomial kernel of degree specified in reset method.

        Parameters
        ----------
        x1 : numpy.ndarray of shape (n, p)
            n examples with p features.
        x2 : numpy.ndarray of shape (m, p)
            m examples with p features.

        Returns
        -------
        kernel : numpy.ndarray of shape (n, m)
            Kernel matrix computed from x1 and x2.

        """

        kernel = (1 + np.dot(x1, x2.T))**self.deg

        return kernel

    def propose_new_min(self, alpha_min_prop, alpha_prev,
                        g_min_current, mistakes_min_current, which):
        """
        Propose a new minimizing alpha classifier (w.r.t. training error).
        For on the fly computation.

        Parameters
        ----------
        alpha_min_prop : numpy.ndarray of shape (n, 1)
            Proposed new minimizing alpha classifier.
            n is the number of training examples.
        alpha_prev : numpy.ndarray of shape (n, 1)
            Previous tested alpha classifier.
        g_min_current : numpy.ndarray of shape (epoch_size, 1)
            Current prediction for the n training features
            (inside sgn-function).
        mistakes_min_current : int
            Current total count of mistakes on training data.
        which : int
            Number of digit that is predicted.

        Returns
        -------
        g_min_prop : numpy.ndarray of shape (n_sample, 1)
            Proposed prediction for the n training features.
        mistakes_min_prop : int
            Proposed total count of mistakes on training data.

        """

        # index of training example where proposed and current alpha differ
        S = np.argwhere(alpha_min_prop != alpha_prev).flatten()

        # only index s gives a new contribution of proposed
        g_min_prop = g_min_current.copy()
        g_min_prop +=\
            np.sum((alpha_min_prop[S] - alpha_prev[S]) *
                   self.z[S, which] *
                   self.kernel_train[S, :].T,
                   axis=1)

        z_hat_min_prop = np.sign(g_min_prop)

        mistakes_min_prop = np.sum(
            z_hat_min_prop != self.z[:, which])
        np.sum(z_hat_min_prop != self.z[:, which])

        return g_min_prop, mistakes_min_prop

    def write_output_bin(self, which, alpha, alpha_min, alpha_avg):
        """
        Compute the training and test error for every input predictor
        and write to file "./results/bin_pred/output_bin.txt".

        Parameters
        ----------
        which : int
            Which digit.
        alpha_min : numpy.ndarray of shape (n, 1)
            Final predictor.
        alpha_min : numpy.ndarray of shape (n, 1)
            Predictor that minimizes the training error.
        alpha_avg : numpy.ndarray of shape (n, 1)
            Average predictor.

        Returns
        -------
        None.

        """

        # reset output file for first epoch
        if self.n_epoch == 1:
            file = open("./results/bin_pred/digit=" +
                        str(which) + "_deg=" +
                        str(self.deg) + ".txt", "w")
            file.write("tr. err. fin. pred., test err. fin. pred., " +
                       "tr. err. min. pred., test err. min. pred., " +
                       "tr. err. avg. pred., test  err. avg. pred.\n")

        else:
            file = open("./results/bin_pred/digit=" +
                        str(which) + "_deg=" +
                        str(self.deg) + ".txt", "a")

        # compute training and test error
        # for every predictor in A and write to file
        for i, al in enumerate([alpha, alpha_min, alpha_avg]):
            S = np.argwhere(al != 0).flatten()

            # training error
            z_hat = np.sum(al[S] *
                           self.z[S, which] *
                           self.kernel_train[S, :].T, axis=1)
            z_hat = np.sign(z_hat)
            training_error = np.sum(
                z_hat != self.z[:, which]) / self.n_train

            file.write(str(training_error) + ", ")

            # test error
            z_hat = np.sum(al[S] *
                           self.z[S, which] *
                           self.kernel_test[S, :].T, axis=1)
            z_hat = np.sign(z_hat)
            test_error = np.sum(
                z_hat != self.z_test[:, which]) / self.n_test

            file.write(str(test_error))

            if i < 2:
                file.write(", ")
            else:
                file.write("\n")

        file.close()

    def train(self):
        """
        Train the kernel perceptron on the given training set.
        This method uses polynomial kernel.
        Value deg can be specified using the reset method.
        The results are stored in the alpha lists.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        # compute kernel matrix
        if self.n_epoch == 1:
            self.kernel_train = self.compute_kernel(
                self.feats_train, self.feats_train)
            self.kernel_test = self.compute_kernel(
                self.feats_train, self.feats_test)

        print("***start training of multiclass classifier for deg = "
              + str(self.deg) + ", current epoch: "
              + str(self.n_epoch) + "***")

        # loop over all digits to find binary classifier w.r.t that digit
        for digit in self.digits:
            print(str(digit))  # loading bars

            # alpha vectors of current digit
            alpha_temp = self.alpha_list[digit].copy()
            alpha_min_temp = self.alpha_min_list[digit].copy()
            alpha_prev_temp = self.alpha_prev_list[digit].copy()
            alpha_avg_temp = self.alpha_avg_list[digit].copy()

            # for finding (on the fly) minimizing predictor
            mistakes_min_temp = self.mistakes_min_list[digit]
            g_min_temp = self.g_min_list[digit].copy()

            # shuffle training indexes
            ind_train = np.arange(self.n_train)
            np.random.shuffle(ind_train)

            # loop over all examples in training data
            # (in order of shuffled indexes = random permutation)
            for t in ind_train:
                # compute predicted label for training example with index t
                S = np.argwhere(alpha_temp != 0).flatten()
                z_hat_t = np.sum(
                    alpha_temp[S] * self.z[S, digit] * self.kernel_train[S, t])

                z_hat_t = np.sign(z_hat_t)

                if z_hat_t != self.z[t, digit]:
                    alpha_temp[t] += 1
                    # check if new alpha is better than old one
                    # (in terms of training error)
                    g_min_prop, mistakes_min_prop = self.propose_new_min(
                        alpha_temp, alpha_prev_temp,
                        g_min_temp, mistakes_min_temp, digit)
                    g_min_temp = g_min_prop.copy()
                    alpha_prev_temp = alpha_temp.copy()

                    if mistakes_min_prop < mistakes_min_temp:
                        mistakes_min_temp = mistakes_min_prop
                        alpha_min_temp = alpha_temp.copy()

                alpha_avg_temp += alpha_temp

            # add alpha vectors for current digitto corresponding lists
            self.alpha_list[digit] = alpha_temp.copy()
            self.alpha_min_list[digit] = alpha_min_temp.copy()
            self.alpha_prev_list[digit] = alpha_prev_temp.copy()
            self.alpha_avg_list[digit] = alpha_avg_temp.copy()

            # update lists for on the fly computation
            self.mistakes_min_list[digit] = mistakes_min_temp
            self.g_min_list[digit] = g_min_temp.copy()

            self.write_output_bin(
                digit, alpha_temp, alpha_min_temp,
                alpha_avg_temp / (self.n_epoch * self.n_train))

        print("\n***training of multiclass classifier for deg = "
              + str(self.deg) + ", current epoch: "
              + str(self.n_epoch) + " finished***\n")

        self.n_epoch += 1

    def predict(self, feats_pred, which_alpha):
        """
        Predict label using kernel perceptron.

        Parameters
        ----------
        feats_pred : str
            Prediction features,.
            Either "train" or "test".
        which_alpha : str
            Specifies type of predictor.
            Can be "standard", "min" or "avg".

        Returns
        -------
        label_pred : numpy.ndarray of shape (n, 1)
            Prediction label.

        """

        if feats_pred == "train":
            n_pred = self.n_train
            kernel = self.kernel_train
        if feats_pred == "test":
            n_pred = self.n_test
            kernel = self.kernel_test

        label_pred = np.zeros(n_pred)

        z_hat = np.zeros([n_pred, len(self.digits)])

        # compute binary prediction for every digit
        for digit in self.digits:
            if which_alpha == "standard":
                alpha_temp = self.alpha_list[digit]
            if which_alpha == "min":
                alpha_temp = self.alpha_min_list[digit]
            if which_alpha == "avg":
                alpha_temp = self.alpha_avg_list[digit] / \
                    (self.n_epoch * self.n_train)
            S = np.argwhere(alpha_temp != 0).flatten()
            z_hat[:, digit] +=\
                np.sum(alpha_temp[S] * self.z[S, digit] *
                       kernel[S, :].T, axis=1)
        label_pred = self.digits[np.argmax(z_hat, axis=1)]

        return label_pred

    def compute_error(self, feats_pred, which_alpha,
                      path_results="./results/", conf_mat=False):
        """
        Compute relative error (using zero-one loss) and confusion matrix.

        Parameters
        ----------
        feats_pred : str
            Prediction features,.
            Either "train" or "test".
        which_alpha : str
            Specifies type of predictor.
            Can be "standard", "min" or "avg".
        path_results : str
            Path where results should be saved
            (e.g., "./results/")
        conf_mat : bool
            Set to True if confusion matrix of prediction w.r.t.
            true label should be computed.

        Returns
        -------
        error : float
            Relative error.

        """

        if feats_pred == "train":
            n_pred = self.n_train
            label_true = self.label_train
        if feats_pred == "test":
            n_pred = self.n_test
            label_true = self.label_test

        label_pred = self.predict(feats_pred, which_alpha)

        # compute confusion matrix
        if conf_mat:
            conf_mat = np.zeros([len(self.digits), len(self.digits)])
            for i in self.digits:
                for j in self.digits:
                    # number of times, where digit j is predicted
                    # and i is the true label
                    n_ij = np.sum((label_pred == j) * (label_true == i))
                    conf_mat[i, j] = n_ij
            np.savetxt(path_results + "conf_mat/conf_mat_" + which_alpha +
                       "_n_epoch=" + str(self.n_epoch - 1) + "_deg=" +
                       str(self.deg) + ".txt",
                       conf_mat.astype(int), fmt='%i', delimiter=",")

        if feats_pred == "train":
            label_true = self.label_train
        if feats_pred == "test":
            label_true = self.label_test

        mistakes = np.sum(label_pred != label_true)
        error = mistakes / n_pred

        return error

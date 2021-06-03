import numpy as np
from data import make_binary


class KernelPerceptron:

    def __init__(self, feats_train, label_train):
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

        Returns
        -------
        None.

        """

        self.feats_train = feats_train
        self.label_train = label_train
        self.n_train = len(label_train)  # total number of training examples

        # possible digits
        self.digits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # binary training data for each digit
        self.z = np.zeros([self.n_train, len(self.digits)])
        for digit in self.digits:
            self.z[:, digit] = make_binary(self.label_train, digit)

        # alpha lists contain numpy.ndarray of shape (n, 1) for every digit
        self.alpha_list = []  # multiclass predictor after run
        # minimizing multiclass predictor w.r.t. training error
        self.alpha_min_list = []
        # average multiclass predictor from all epochs
        self.alpha_avg_list = []

        print("initialization of kernel perceptron successfull\n")

    def reset(self, n_epoch, n_sample, deg,
              output_bin=False, output_digit=1):
        """
        Reset kernel perceptron for another run.

        Parameters
        ----------
        n_epoch : int
            Number of cycles over the whole training set.
            For each cycle, the training set is permutated randomly.
        n_sample : int
            Number of training examples sampled with replacement in each epoch.
        deg : int
            Degree of polynomial kernel.
        output_bin : bool
            Specifies if training error for a binary classifier is written
            to "results/bin_class.txt".
        output_digit : int
            Specifies the digit for which the data is stored.

        Returns
        -------
        None.

        """

        self.n_epoch = n_epoch
        self.n_sample = n_sample
        self.deg = deg

        self.output_bin = output_bin
        self.output_digit = output_digit

        self.alpha_list = []
        self.alpha_min_list = []
        self.alpha_avg_list = []

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

    def propose_new_min(self, alpha_min_prop, alpha_min_current,
                        g_min_current, mistakes_min_current, which,
                        n_sample):
        """
        Propose a new minimizing alpha classifier (w.r.t. training error).

        Parameters
        ----------
        alpha_min_prop : numpy.ndarray of shape (n, 1)
            Proposed new minimizing alpha classifier.
            n is the number of training examples.
        alpha_min_current : numpy.ndarray of shape (n, 1)
            Current minimizing alpha classifier.
        g_min_current : numpy.ndarray of shape (epoch_size, 1)
            Current prediction for the n training features
            (inside sgn-function).
        mistakes_min_current : int
            Current total count of mistakes on training data.
        which : int
            Number of digit that is predicted.
        n_sample : int
            Number of training examples sampled with replacement in each epoch.
            Here, number of training examples used
            for computation of training error.

        Returns
        -------
        g_min_prop : numpy.ndarray of shape (n_sample, 1)
            Proposed prediction for the n training features.
        mistakes_min_prop : int
            Proposed total count of mistakes on training data.

        """

        # index of training example where proposed and current alpha differ
        S = np.argwhere(alpha_min_prop != alpha_min_current).flatten()

        # only index s gives a new contribution of proposed
        g_min_prop = g_min_current + np.sum(
            (alpha_min_prop[S] - alpha_min_current[S]) *
            self.z[S, which] *
            self.compute_kernel(
                self.feats_train[S, :], self.feats_train[:n_sample, :]).T,
            axis=1)

        z_hat_min_prop = np.sign(g_min_prop)

        mistakes_min_prop = np.sum(
            z_hat_min_prop != self.z[:n_sample, which])
        np.sum(z_hat_min_prop != self.z[:n_sample, which])

        return g_min_prop, mistakes_min_prop

    def write_output_bin(self, A, alpha_min, alpha_avg):
        """
        Compute the training error for every input predictor
        and write to file "./results/output_bin.txt".

        Parameters
        ----------
        A : list of numpy.ndarray of shape (n, 1)
            List of predictors, where n is the number of training examples.
            Usually, A contains n_epoch * n_sample predictors.
        alpha_min : numpy.ndarray of shape (n, 1)
            Predictor that minimizes the training error.
        alpha_avg : numpy.ndarray of shape (n, 1)
            Average predictor.

        Returns
        -------
        None.

        """

        A.append(alpha_min)
        A.append(alpha_avg)

        with open("./results/output_bin.txt", "w") as file:
            file.write("n_epoch = " + str(self.n_epoch) +
                       ", deg = " + str(self.deg) +
                       "\nlast to training errors correspond" +
                       "to minimizing and average predictor" +
                       "\niteration, training error")

            # compute training error for every predictor in A and write to file
            for i, alpha in enumerate(A):
                S = np.argwhere(alpha != 0).flatten()
                z_hat = np.sum(alpha[S] * self.z[S, self.output_digit] *
                               self.compute_kernel(self.feats_train[S, :],
                                                   self.feats_train).T, axis=1)
                training_error = np.sum(
                    z_hat != self.z[:, self.output_digit]) / self.n_train
                file.write(str(i + 1) + ", " + str(training_error) + "\n")

        file.close()

    def train(self):
        """
        Train the kernel perceptron on the given training set.
        This method uses polynomial kernel.
        Values of n_epoch and deg can be specified using the reset method.
        The results are stored in the alpha lists.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        print("***start training of multiclass classifier for deg = "
              + str(self.deg)
              + " and " + str(self.n_epoch)
              + " epochs over random permutations of training data***")

        # loop over all digits to find binary classifier w.r.t that digit
        for digit in self.digits:
            if digit != self.digits[0]:
                print("")
            print("train binary classifier for digit = " + str(digit))

            # temporary alpha vectors for current digit
            alpha_temp = np.zeros(self.n_train)
            alpha_min_temp = np.zeros(self.n_train)
            alpha_avg_temp = np.zeros(self.n_train)

            # initial number of mistakes of minimizing alpha classifier
            mistakes_min = self.n_sample  # since sgn(0) = 0
            # initial prediction of minimizing alpha classifier
            # (before y_hat = sgn(y_in))
            g_min = np.zeros(self.n_sample)

            if self.output_bin:
                A = []

            # loop over desired number of epochs over training data
            for i in range(self.n_epoch):
                print(u"\u2588", end='')  # loading bars
                # shuffle training indexes
                ind_train = np.arange(self.n_train)
                ind_train = np.random.choice(
                    ind_train, self.n_sample, replace=False)

                # loop over all examples in training data
                # (in order of shuffled indexes = random permutation)
                for t in ind_train:
                    # compute predicted label for training example with index t
                    S = np.argwhere(alpha_temp != 0).flatten()
                    z_hat_t = np.sum(alpha_temp[S] * self.z[S, digit] *
                                     self.compute_kernel(
                        self.feats_train[S, :], self.feats_train[t, :]))

                    z_hat_t = np.sign(z_hat_t)

                    if z_hat_t != self.z[t, digit]:
                        alpha_temp[t] += 1

                        # check if new alpha is better than old one
                        # (in terms of training error)
                        g_min_prop, mistakes_min_prop = self.propose_new_min(
                            alpha_temp, alpha_min_temp,
                            g_min, mistakes_min, digit, self.n_sample)
                        if mistakes_min_prop < mistakes_min:
                            mistakes_min = mistakes_min_prop
                            g_min = g_min_prop.copy()
                            alpha_min_temp = alpha_temp.copy()

                    alpha_avg_temp += alpha_temp

                    if self.output_bin:
                        if digit == self.output_digit:
                            A.append(alpha_temp)

            # add alpha vectors to corresponding lists for current digit
            self.alpha_list.append(alpha_temp)
            self.alpha_min_list.append(alpha_min_temp)
            self.alpha_avg_list.append(
                alpha_avg_temp / (self.n_epoch * self.n_train))

            if self.output_bin:
                if digit == self.output_digit:
                    self.write_output_bin(A, alpha_min_temp, alpha_avg_temp)

        print("\n***training of multiclass classifier for deg = "
              + str(self.deg)
              + " and " + str(self.n_epoch)
              + " epochs over random permutations"
              + " of training data completed***\n")

    def predict(self, feats_pred, which_alpha):
        """
        Predict label using kernel perceptron.

        Parameters
        ----------
        feats_pred : numpy.ndarray of shape (n, p)
            Prediction features, where n is the number of examples
            and p the number of features.
        which_alpha : str
            Specifies type of predictor.
            Can be "standard", "min" or "avg".

        Returns
        -------
        label_pred : numpy.ndarray of shape (n, 1)
            Prediction label.

        """

        n_pred = np.shape(feats_pred)[0]  # number of prediction examples
        label_pred = np.zeros(n_pred)

        z_hat = np.zeros([n_pred, len(self.digits)])

        # compute binary prediction for every digit
        for digit in self.digits:
            if which_alpha == "standard":
                alpha_temp = self.alpha_list[digit]
            if which_alpha == "min":
                alpha_temp = self.alpha_min_list[digit]
            if which_alpha == "avg":
                alpha_temp = self.alpha_avg_list[digit]
            S = np.argwhere(alpha_temp != 0).flatten()
            z_hat[:, digit] = np.sum(alpha_temp[S] * self.z[S, digit] *
                                     self.compute_kernel(
                self.feats_train[S, :], feats_pred).T, axis=1)
        label_pred = self.digits[np.argmax(z_hat, axis=1)]

        return label_pred

    def compute_error(self, feats_pred, label_true, which_alpha,
                      path_results="./results/", conf_mat=False):
        """
        Compute relative error (using zero-one loss) and confusion matrix.

        Parameters
        ----------
        feats_pred : numpy.ndarray of shape (n, p)
            Prediction features, where n is the number of examples
            and p the number of features.
        label_true : numpy.ndarray of shape (n, 1)
            True label.
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

        n_pred = np.shape(feats_pred)[0]  # number of prediction examples

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
                       "_n_epoch=" + str(self.n_epoch) + "_deg=" +
                       str(self.deg) + ".txt",
                       conf_mat.astype(int), fmt='%i', delimiter=",")

        mistakes = np.sum(label_pred != label_true)
        error = mistakes / n_pred

        return error

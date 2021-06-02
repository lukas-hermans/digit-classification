import numpy as np
from data import make_binary
import time


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

        self.digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # possible digits

        # binary training data for each digit
        self.y = np.zeros([self.n_train, len(self.digits)])
        for digit in self.digits:
            self.y[:, digit] = make_binary(self.label_train, digit)

        # alpha lists contain numpy.ndarray of shape (n, 1) for every digit
        self.alpha_list = []  # multiclass predictor after run
        # best multiclass predictor w.r.t. training error
        self.alpha_best_list = []
        # average multiclass predictor from all epochs
        self.alpha_mean_list = []

        print("initialization of kernel perceptron successfull\n")

    def reset(self, n_epoch, epoch_size, deg):
        """
        Reset kernel perceptron for another run.

        Parameters
        ----------
        n_epoch : int
            Number of cycles over the whole training set.
            For each cycle, the training set is permutated randomly.
        epoch_size : int
            Number of training examples considered in each epoch.
        deg : int
            Degree of polynomial kernel.

        Returns
        -------
        None.

        """

        self.n_epoch = n_epoch
        self.epoch_size = epoch_size
        self.deg = deg

        self.alpha_list = []
        self.alpha_best_list = []
        self.alpha_mean_list = []

    def compute_kernel(self, x1, x2):
        """
        Compute polynomial kernel.

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

    def propose_new_best(self, alpha_best_prop, alpha_best_current,
                         y_in_best_current, mistakes_best_current, which,
                         epoch_size):
        """
        Propose a new best alpha classifier (w.r.t. training error).

        Parameters
        ----------
        alpha_best_prop : numpy.ndarray of shape (n, 1)
            Proposed new best alpha classifier.
            n is the number of training examples.
        alpha_best_current : numpy.ndarray of shape (n, 1)
            Current best alpha classifier.
        y_in_best_current : numpy.ndarray of shape (epoch_size, 1)
            Current prediction for the n training features.
        mistakes_best_current : int
            Current total count of mistakes on training data.
        which : int
            Number of digit that is predicted.
        epoch_size : int
            Number of training examples considered in each epoch.
            Here, number of training examples used
            for computation of training error.

        Returns
        -------
        y_in_best_prop : numpy.ndarray of shape (epoch_size, 1)
            Proposed prediction for the n training features.
        mistakes_best_prop : int
            Proposed total count of mistakes on training data.

        """

        # index of training example where proposed and current alpha differ
        s = np.argwhere(alpha_best_prop != alpha_best_current)[0]

        # only index s gives a new contribution of proposed alpha
        y_in_best_prop = y_in_best_current + self.y[s, which] * \
            self.compute_kernel(
                self.feats_train[s, :], self.feats_train[:epoch_size, :])

        y_hat_best_prop = np.sign(y_in_best_prop)

        mistakes_best_prop = \
            np.sum(y_hat_best_prop != self.y[:epoch_size, which])

        return y_in_best_prop, mistakes_best_prop

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
            alpha_best_temp = np.zeros(self.n_train)
            alpha_mean_temp = np.zeros(self.n_train)

            # initial number of mistakes of best alpha classifier
            mistakes_best = self.epoch_size  # since sgn(0) = 0
            # initial prediction of best alpha classifier
            # (before y_hat = sgn(y_in))
            y_in_best = np.zeros(self.epoch_size)  # since sgn(0) = 0

            # loop over desired number of epochs over training data
            for i in range(self.n_epoch):
                print(u"\u2588", end='')  # loading bars
                time_a = time.time()
                # shuffle training indexes
                ind_train = np.arange(self.n_train)
                ind_train = np.random.choice(
                    ind_train, self.epoch_size, replace=False)

                # loop over all examples in training data
                # (in order of shuffled indexes = random permutation)
                for t in ind_train:
                    # compute predicted label for training example with index t
                    S = np.argwhere(alpha_temp != 0).flatten()
                    y_hat_t = np.sum(alpha_temp[S] * self.y[S, digit] *
                                     self.compute_kernel(
                        self.feats_train[S, :], self.feats_train[t, :]))

                    y_hat_t = np.sign(y_hat_t)

                    if y_hat_t != self.y[t, digit]:
                        alpha_temp[t] += 1

                        # check if new alpha is better than old one
                        # (in terms of training error)
                        y_in_best_prop, mistakes_best_prop = \
                            self.propose_new_best(
                                alpha_temp, alpha_best_temp,
                                y_in_best, mistakes_best, digit, self.epoch_size)
                        if mistakes_best_prop < mistakes_best:
                            mistakes_best = mistakes_best_prop
                            y_in_best = y_in_best_prop.copy()
                            alpha_best_temp = alpha_temp.copy()

                    alpha_mean_temp += alpha_temp

                print(time.time() - time_a)

            # add alpha vectors to corresponding lists for current digit
            self.alpha_list.append(alpha_temp)
            self.alpha_best_list.append(alpha_best_temp)
            self.alpha_mean_list.append(
                alpha_mean_temp / (self.n_epoch * self.n_train))

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
            Can be "standard", "best" or "mean".

        Returns
        -------
        label_pred : numpy.ndarray of shape (n, 1)
            Prediction label.

        """

        n_pred = np.shape(feats_pred)[0]  # number of prediction examples
        label_pred = np.zeros(n_pred)

        # compute prediction for every example
        for t in range(n_pred):
            # compute binary prediction for every digit
            y_hat = []
            for digit in self.digits:
                y_hat_temp = 0
                if which_alpha == "standard":
                    alpha_temp = self.alpha_list[digit]
                if which_alpha == "best":
                    alpha_temp = self.alpha_best_list[digit]
                if which_alpha == "mean":
                    alpha_temp = self.alpha_mean_list[digit]
                S = np.argwhere(alpha_temp != 0).flatten()
                y_hat = np.sum(alpha_temp[S] * self.y[S, digit] *
                               self.compute_kernel(
                    self.feats_train[S, :], feats_pred[t, :]))
                y_hat.append(y_hat_temp)
            label_pred[t] = self.digits[np.argmax(y_hat)]

        return label_pred

    def compute_error(self, feats_pred, label_true, which_alpha):
        """
        Compute relative error (using zero-one loss).

        Parameters
        ----------
        feats_pred : numpy.ndarray of shape (n, p)
            Prediction features, where n is the number of examples
            and p the number of features.
        label_true : numpy.ndarray of shape (n, 1)
            True label.
        which_alpha : str
            Specifies type of predictor.
            Can be "standard", "best" or "mean".

        Returns
        -------
        error : float
            Relative error.

        """

        n_pred = np.shape(feats_pred)[0]  # number of prediction examples

        label_pred = self.predict(feats_pred, which_alpha)

        mistakes = np.sum(label_pred != label_true)
        error = mistakes / n_pred

        return error

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

        self.digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # possible digits
        
        # binary training data for each digit
        self.y = np.zeros([self.n_train, len(self.digits)])
        for digit in self.digits:
            self.y[:, digit] = make_binary(self.label_train, digit)

        # alpha lists contain numpy.ndarray of shape (n, 1) for every digit
        self.alpha_list = []  # multiclass predictor after run
        self.alpha_best_list = []  # best multiclass predictor w.r.t. training error
        self.alpha_mean_list = []  # average multiclass predictor from all epochs

        print("initialization of kernel perceptron successfull\n")
        
    def reset(self, n_epoch, deg):
        """
        Reset kernel perceptron for another run.

        Parameters
        ----------
        n_epoch : int
            Number of cycles over the whole training set.
            For each cycle, the training set is permutated randomly.
        deg : int
            Degree of polynomial kernel.

        Returns
        -------
        None.

        """
        
        self.n_epoch = n_epoch
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

    def train(self):
        """
        Train the kernel perceptron on the given training set.
        Method uses polynomial kernel.
        The results are stored in self.alpha_best_list and self.alpha_mean_list.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        
        print("***start training of multiclass classifier for deg = " + str(self.deg)
              + " and " + str(self.n_epoch)
              + " epochs over random permutations of training data***")

        # loop over all digits to find binary classifier w.r.t that digit
        for digit in self.digits:
            if digit != self.digits[0]:
                print("")
            print("train binary classifier for digit = " + str(digit))

            alpha_temp = np.zeros(self.n_train)  # temporary alpha vector
            alpha_best_temp = np.zeros(self.n_train)
            alpha_mean_temp = np.zeros(self.n_train) # temporary mean alpha vector
            
            best_mistakes = self.n_train # initial number of mistakes of best alpha classifier (since sign(0) = 0)
            best_<

            # loop over desired number of epochs over training data
            for i in range(self.n_epoch):
                print(u"\u2588", end='')
                
                # shuffle training indexes
                ind_train = np.arange(self.n_train)
                np.random.shuffle(ind_train)

                # loop over all examples in training data (in order of shuffled indexes = random permutation)
                for t in ind_train:
                    # compute predicted label for training example with index t
                    y_hat_t = 0
                    S = np.argwhere(alpha_temp != 0)
                    for s in S:
                        y_hat_t += alpha_temp[s] * self.y[s, digit] * \
                            self.compute_kernel(self.feats_train[s,:], self.feats_train[t,:])
                    y_hat_t = np.sign(y_hat_t)

                    if y_hat_t != self.y[t, digit]:
                        alpha_temp[t] += 1
                        
                        

                    alpha_mean_temp += alpha_temp

            self.alpha_list.append(alpha_temp)
            self.alpha_best_list.append(alpha_best_temp)
            self.alpha_mean_list.append(alpha_mean_temp / (self.n_epoch * self.n_train))

        print("\n***training of multiclass classifier for deg = " + str(self.deg)
              + " and " + str(self.n_epoch)
              + " epochs over random permutations"
              + " of training data completed***\n")

    def compute_binary_mistakes(self, alpha, which):
        """
        Computes the number of mistakes of binary classifier.
        Always refers to prediction of TRAINING EXAMPLES!!!

        Parameters
        ----------
        alpha : numpy.ndarray of shape (n, 1)
            alpha vector which describes the perceptron.
            n is the number of training examples.
        which : int
            Digit that is predicted.

        Returns
        -------
        mistkaes : float
            Total number of mistakes.

        """
        
        mistakes = 0
        
        # compute prediction for each training example using alpha
        S = np.argwhere(alpha != 0)
        for t in range(self.n_train):
            y_hat = 0
            
            for s in S:
                y_hat += alpha[s] * self.y[s, which] * \
                    self.compute_kernel(self.feats_train[s,:], self.feats_train[t, :])
            
            y_hat = np.sign(y_hat)
            
            if y_hat != self.y[t, which]:
                mistakes += 1
        
        return mistakes

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
                    alpha_temp = self.alpha_mean_list[digit]
                if which_alpha == "best":
                    alpha_temp = self.alpha_best_list[digit]
                if which_alpha == "mean":
                    alpha_temp = self.alpha_mean_list[digit]
                S = np.argwhere(alpha_temp != 0)
                for s in S:
                    y_hat_temp += alpha_temp[s] * self.y[s, digit] * \
                        self.compute_kernel(self.feats_train[s,:], feats_pred[t, :])
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

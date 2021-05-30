import numpy as np


class KernelPerceptron:

    def __init__(self, feats_train, label_train):
        self.feats_train = feats_train
        self.label_train = label_train
        self.n_example = len(label_train)  # total number of training examples

        self.S = []  # training examples for update

    def train(self, deg, n_epoch):

        for i in range(n_epoch):

            print("epoch nr.: " + str(i + 1) + "\n")

            for t in range(self.n_example):

                if (t + 1) % int(self.n_example / 6) == 0:
                    print("example nr.: " + str(t + 1) + "\n")

                label_pred_t = self.predict(self.feats_train[t, :])

                if self.label_train[t] != label_pred_t:
                    self.S.append(t)

    def predict(self, feats_pred):

        label_pred = 0
        for s in self.S:
            label_pred += self.label_train[s] * \
                (1 + np.dot(self.feats_train[s, :], feats_pred.T))
        label_pred = np.sign(label_pred)

        return label_pred

    def test_error(self, feats, label):

        label_pred = self.predict(feats)

        return np.sum(label_pred == label) / len(label)

import numpy as np
import cvxopt


class SVM:
    def __init__(self, X, y, S_=2):
        """
        Args:
            X: training predictor data set (np.array of shape n×p)
            y: training outputs data set (np.array of shape n×1)
            S_: number of observations closest of the hyperplane taken into account when calculating margin (int)
        """
        self.X = X
        self.y = y
        self.S_ = S_
        self.alphas = None
        self.w = None
        self.b = None
        self.M = None

    def train(self):
        """
        Function trains SVM model on input training data - it calculates hyperplane params and max. margin
        Returns: w (np.array; hyperplane coefficients), b (float; hyperplane bias), M (float; max. margin)
        """
        n = self.X.shape[0]
        H = np.dot(self.y * self.X, (self.y * self.X).T)
        q = np.repeat([-1.0], n)[..., None]
        A = self.y.reshape(1, -1)
        b = 0.0
        G = np.negative(np.eye(n))
        h = np.zeros(n)
        P = cvxopt.matrix(H)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
        # solution of optimization problem using cvxopt's QP optimization
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol["x"])
        # calculation of w vector
        self.w = np.dot((self.y * self.alphas).T, self.X)[0]
        # calculation of bias b
        # pomen parametra S!!
        S = np.where(self.alphas.flatten() > sorted(self.alphas.flatten())[-self.S_])
        self.b = np.mean(self.y[S] - self.X[S, :] @ self.w.T)
        # margin calculaton
        self.M = 1 / np.sqrt(sum(self.w ** 2))

    def predict(self, X_test):
        """

        Args:
            X_test: predictor values for which we would like to make prediction
        Returns: prediction

        """
        pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            value = self.get_SVM_val(X_test[i,:])
            if value > 0:
                pred[i] = 1
            else:
                pred[i] = -1
        return pred

    def get_SVM_val(self, X_test):
        """
        Function calculates testing value (hyperplane equation value) for given input data X
        Args:
            X_test: predictor values for which we would like to make prediction

        Returns: testing value

        """
        return sum(X_test * self.w) + self.b

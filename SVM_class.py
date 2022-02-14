import numpy as np
import cvxopt
import my_statistics as ms
from tqdm.notebook import tqdm


class SVM:
    """
    Currently actually applies the Maximum Margin Classifier, and not the actual SVM.
    """

    def __init__(self, X, y, C=0, kernel='linear', d=None, gamma=None, show_progress=False):
        """
        Args:
            X: training predictor data set (np.array of shape n×p)
            y: training outputs data set (np.array of shape n×1)
            C: soft margin parameter -> if 0 hard margin is calculated. Defaults to 0.
            kernel: type of kernel: linear, polynomial, radial (str)
            d: order of polynomial for polynomial kernel (int)
            gamma: positive constant for radial kernel (float)
            S_: number of observations closest of the hyperplane taken into account when calculating margin (int)
            show_progress: if show_progress is True, the progress during QP optimization is printed (bool)
        """
        self.X = X
        self.y = y
        self.alphas = None
        self.w, self.b, self.M = None, None, None
        self.show_progress = show_progress
        self.C = C
        self.d = d
        self.gamma = gamma
        self.kernel = kernel
        self.sv = None
        if kernel == 'linear':
            self.K = self.linear_kernel(self.X, self.X)
        elif kernel == 'polynomial':
            self.K = self.polynomial_kernel(self.X, self.X)
        elif kernel == 'radial':
            self.K = self.radial_kernel(self.X, self.X)
        else:
            print('Invalid kernel value inserted.')

    def train(self):
        """
        Function trains SVM model on input training data - it calculates hyperplane params and max. margin
        Returns: w (np.array; hyperplane weight vector), b (float; hyperplane bias), M (float; max. margin)
        """
        n = self.X.shape[0]
        H = np.outer(self.y, self.y) * self.K
        q = np.repeat([-1.0], n)[..., None]
        A = self.y.reshape(1, -1)
        b = 0.0
        if (self.C is not None) and (self.C != 0):
            G = np.vstack((np.eye(n) * -1, np.eye(n)))
            h = np.hstack((np.zeros(n), np.ones(n) * self.C))
        else:
            G = np.negative(np.eye(n))
            h = np.zeros(n)
        P = cvxopt.matrix(H)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
        # solution of optimization problem using cvxopt's QP optimization
        cvxopt.solvers.options['show_progress'] = self.show_progress
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol["x"])
        # determination of support vectors
        if self.kernel == 'linear':
            # calculation of w vector
            self.w = np.dot((self.y * self.alphas).T, self.X)[0]
        # calculation of bias b
        # pomen parametra S!!
        self.sv = np.where(self.alphas > 1e-5)[0]
        if self.kernel == 'linear':
            self.b = np.mean(self.y[self.sv] - self.X[self.sv, :] @ self.w.T)
        else:
            y_sv = self.y[self.sv]
            self.b = 0
            for i in range(self.X[self.sv].shape[0]):
                if self.kernel == 'polynomial':
                    self.b += y_sv[i] - sum(self.alphas[self.sv][i]*y_sv[i]*self.polynomial_kernel(self.X[self.sv][i], self.X[self.sv]))
                elif self.kernel == 'radial':
                    self.b += y_sv[i] - sum(self.alphas[self.sv][i]*y_sv[i]*self.radial_kernel(self.X[self.sv][i], self.X[self.sv]))
            self.b /= self.X[self.sv].shape[0]
        # margin calculaton
        self.M = 1 / np.sqrt(sum((np.dot((self.y * self.alphas).T, self.X)[0]) ** 2))

    def predict(self, X_test):
        """

        Args:
            X_test: predictor values for which we would like to make prediction
        Returns: prediction

        """
        return np.sign(self.get_SVM_val(X_test))

    def get_SVM_val(self, X_test):
        """
        Function calculates testing value (hyperplane equation value) for given input data X
        Args:
            X_test: predictor values for which we would like to make prediction

        Returns: testing value

        """
        if self.kernel == 'linear':
            return np.sum(X_test * self.w, axis=1) + self.b
        elif self.kernel == 'polynomial':
            return self.b + np.sum(self.alphas.flatten()[self.sv] * self.y[self.sv].flatten() *
                                   self.polynomial_kernel(X_test, self.X[self.sv]), axis=1)
        elif self.kernel == 'radial':
            return self.b + np.sum(self.alphas.flatten()[self.sv] * self.y[self.sv].flatten() *
                                   self.radial_kernel(X_test, self.X[self.sv]), axis=1)

    @staticmethod
    def linear_kernel(X1, X2):
        """
        Function implements linear kernel.
        Returns: kernel values for input X data.
        """
        return np.sum(np.einsum('ik,jk->ijk', X1, X2), axis=-1)

    def radial_kernel(self, X1, X2):
        """
        Function implements polynomial kernel.
        Returns: kernel values for input X data.
        """
        if self.gamma is None:
            print('Add radial kernel parameter gamma')
        exponent_matrix = np.zeros((X1.shape[0], X2.shape[0], X1.shape[1]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                exponent_matrix[i, j, :] = (X1[i, :] - X2[j, :])**2
        exponents = -self.gamma * np.sum(exponent_matrix, axis=2)
        return np.exp(exponents)

    def polynomial_kernel(self, X1, X2):
        """
        Function implements polynomial kernel.
        Returns: kernel values for input X data.
        """
        if self.d is None:
            print('Add polynomial order parameter d')
        if X1.shape[0] == X2.shape[1]:
            return (1 + np.sum(np.einsum('k,jk->jk', X1, X2), axis=-1)) ** self.d
        else:
            return (1 + np.sum(np.einsum('ik,jk->ijk', X1, X2), axis=-1)) ** self.d




def k_fold_CV_for_SVM(inputs_, outputs_, C, k=10, kernel='linear'):
    """Function performs k-fold Cross Validation for the purpose of optimal C value determination in soft margin SVM
    machine learning method.

    Args:
        inputs_ (numpy array): numeric array of input values (n×p matrix) where n is sample size and p is number of
                               predictors.
        outputs_ (numpy array): numeric array of output values
        k (int, optional): number of folds. Defaults to 10.
        C (list/numpy array, optional): Possible C values. Defaults to np.arange(1,10).

    Returns:
        list: error rates corresponding to the input K values
        int (optional): optimal K value
    """
    # data split
    input_f, output_f = ms.k_fold_split(inputs_, outputs_, k=k)
    error_rates = []
    for k_ in tqdm(C, desc=f'Main loop'):
        errors_ = []
        for i in tqdm(range(k), desc=f'loop for K = {k_}: ', leave=False):
            test_x_fold, test_y_fold = input_f[i], output_f[i]
            ind_list = np.logical_not(np.arange(k) == i)
            training_x_folds = np.concatenate(np.array(input_f)[ind_list].squeeze(), axis=0)
            training_y_folds = np.concatenate(np.array(output_f)[ind_list].squeeze(), axis=0)
            svm_temp = SVM(training_x_folds, training_y_folds[:, np.newaxis], C=k_)
            svm_temp.train()
            prediction_k = svm_temp.predict(test_x_fold)
            errors_.append(ms.classification_error_rate(prediction_k, test_y_fold))
        error_rates.append(np.average(errors_))
    C_opt = C[np.argmin(error_rates)]
    return error_rates, C_opt

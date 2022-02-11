import numpy as np
import cvxopt
# TODO: Pack functions into class.


def SVM(X, y, S_):
    """
    function determine SVM separation hyperplane parameters w and b
    Args:
        X: training predictor data
        y: training reference output data

    Returns: w, bias b, margin M

    """
    n = X.shape[0]
    H = np.dot(y*X, (y*X).T)
    q = np.repeat([-1.0], n)[..., None]
    A = y.reshape(1, -1)
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
    alphas = np.array(sol["x"])
    # calculation of w vector
    w = np.dot((y * alphas).T, X)[0]
    # calculation of bias b
    S = np.where(alphas.flatten() > sorted(alphas.flatten())[-S_]) # alphas of support vectors (1e1,1e2) are much bigger then the other (1e-8,1e-9)
    b = np.mean(y[S] - X[S, :]@w.T)
    # margin calculaton
    M = 1/np.sqrt(sum(w**2))
    return w, b, M


def get_SVM_val(X, w, b):
    """
    Function calculates testing value for given input data X
    Args:
        X: input data
        w: hyperplane parameter
        b: bias - hyperplane parameter

    Returns: testing value

    """
    return sum(X*w) + b


def SVM_predict(X, w, b):
    """
    Function makes prediction for input data X
    Args:
        X: input data
        w: hyperplane parameter
        b: bias - hyperplane parameter

    Returns: prediction

    """
    pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        value = get_SVM_val(X[i, :], w, b)
        if value > 0:
            pred[i] = 1
        else:
            pred[i] = -1
    return pred

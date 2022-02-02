import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
import sys
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import prettytable
from tqdm.notebook import tqdm
import random


def MSE(y_pred, y_real):
    """
    Funcition calculates Mean Squared Error (MSE) for inserted data sets.
    :param: y_pred - predicted/aproximated values data set (np.array)
    :param: y_real - true values data set (np.array)
    :return: MSE
    """
    return (sum((y_pred-y_real)**2))/len(y_pred)


def mean(X):
    """
    Function calculates mean value for the data set X.
    Args:
        X: data set (numpy array)

    Returns: mean of X

    """
    return sum(X)/len(X)


def RSS(y_pred, y_real):
    """Function calculates and returns the Residual Sum of Squares (RSS).

    Args:
        y_pred (np.array): numeric array of predicted values
        y_real (np.array): numeric array of actual values

    Returns: RSS for input data
    """
    return sum((y_real-y_pred)**2)



def stdev(X, type_='sample'):
    """
    Function calculates standard deviation for the data set X.
    Args:
        X: data set (numpy array)
        type_: 'sample' (N-1 in denominator) or 'popular' (N in denominator) (string)

    Returns: strandar deviation

    """
    den = None
    if type_ == 'popular':
        den = len(X)
    elif type_ == 'sample':
        den = len(X)-1
    return np.sqrt(sum((X-mean(X))**2)/den)


def cov(X, Y):
    """
    Function calculates the covariance for data sets X and Y.
    Args:
        X: data set (numpy array)
        Y: data set (numpy array)

    Returns: correlation

    """
    return sum((X-mean(X))*(Y-mean(Y)))/(len(X))


def corr(X, Y):
    """
    Function calculates the correlation for data sets X and Y.
    Args:
        X: data set (numpy array)
        Y: data set (numpy array)

    Returns: correlation

    """
    return cov(X, Y)/(stdev(X)*stdev(Y))


def euclidean(P1, P2):
    """
    Function calculates the Euclidean distance between points P1 and P2.
    Args:
        P1: point 1 coordinates (numpy array)
        P2: point 2 coordinates (numpy array)

    Returns: euclidean distance

    """
    return np.sqrt(sum(P1**2+P2**2))


def confusion_matrix(predictions, references, print_=False, ROC_=False):
    """
    Function calculates values of confusion matrix values together with sensitivity, specificity and overall accuracy and prints the result in the table form.
    :param predictions: array of predictions - values 0 or 1 (numpy array)
    :param references: array of references - values 0 or 1 (numpy array)

    return table, sensitivity, specificity, overall accuracy
    """
    TN = np.count_nonzero((predictions==0)&(references==0))
    TP = np.count_nonzero((predictions==1)&(references==1))
    FN = np.count_nonzero((predictions==0)&(references==1))
    FP = np.count_nonzero((predictions==1)&(references==0))
    sensitivity = TP/(TP + FN)
    specificity = TN/(TN + FP)
    overall = (TP + TN)/(TP+TN+FP+FN)
    table = prettytable.PrettyTable()
    table.field_names = ['True status','No','Yes']
    table.add_rows([['Prediciton','',''],
               ['No', f' TN = {TN}', f' FN = {FN}'],
               ['Yes', f' FP = {FP}', f' TP = {TP}'],
                  ['', f'specificity = {specificity*100:.2f} %', f'sensitivity = {sensitivity*100:.2f} %']])
    if print_:
        print(table)

    return table, sensitivity, specificity, overall


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# STATISTICAL LEARNING
def KNN(test_point_, input_data_, output_data, K, normalize=False, show=False, dummy=False):
    """
    Function applies K Nearest Neighbors method.
    Args:
        test_point_: test point location (np.array)
        input_data_: 1D or 2D array of feature values (1 column for individual feature)
        output_data: output for given input_data (np.array)
        K: number of points taken into account by KNN method
        normalize: iput_data is normalized to the values from 0 to 1
        show: Plot graph of points

    Returns: category of test_point

    """
    if normalize:
        norm_basis = abs(input_data_).max(axis=0) # values for normalization
        input_data = input_data_.copy()
        test_point = test_point_.copy()
        to_norm = norm_basis != 0 # predictors to be normalized
        input_data[:,to_norm] = input_data_[:,to_norm] / norm_basis[to_norm]
        test_point[to_norm] = test_point_[to_norm] / norm_basis[to_norm]                    
    else:
        input_data = input_data_.copy()
        test_point = test_point_.copy()

    # calculate distances
    distances = np.sqrt((np.sum(abs(input_data**2-test_point**2), axis=1)).astype(float))

    distances_sorted = sorted(distances)

    output_data_ = output_data.copy()
    if dummy:
        # dummy variables for output data:
        output_set = list(set(output_data))
        for i, j in enumerate(output_data):
            for k, l in enumerate(output_set):  # k predstavlja indeks elementa v naboru možnih izhodnih podatkov
                if output_data[i] == l:  # preverjanje če je posamezni element enak elementu l nabora možnih izhodnih
                    # podatkov
                    output_data_[i] = k  # pripis dummy variable k elementa l i-temu izhodnemu podatku

    # identify K closest points
    closest = output_data_[np.where(distances <= distances_sorted[K - 1])[0]]
    # determine class with highest probability
    count = np.unique(closest, return_counts=True)

    if show:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(test_point[0], test_point[1], test_point[2], facecolor='black', edgecolor='black', label='test')
        for i in range(len(output_data)):
            if i in np.where(distances <= distances_sorted[K - 1])[0]:
                ax.scatter(input_data[i, 0], input_data[i, 1], input_data[i, 2], facecolor='blue',
                           edgecolors=output_data[i], label='K closest')
            else:
                ax.scatter(input_data[i, 0], input_data[i, 1], input_data[i, 2], facecolor=output_data[i],
                           edgecolors=output_data[i], label='other points')
        ax.legend()

    return count[0][np.argmax(count[1])]


# Linear Discriminant Analysis (LDA)
def lin_discriminant_function(x, μ, Σ, π):
    """Function calculates the value of linear disciriminant function for given x.

    Args:
        x (np.array): vector of predictor values for which we would like to perform a prediction
        μ (np.array): vector of mean values for individual predictors
        Σ (np.array): covariance matrix
        π (float): prior probability

    Returns:
        float: values of discriminant function
    """
    return x.T@np.linalg.inv(Σ)@μ-0.5*μ.T@np.linalg.inv(Σ)@μ + np.log(π)

def linear_discriminant_analysis(input_data, predictors, outputs):
    """Function applies LDA method. It learns from predictors and outputs data and make predictions for input_data.

    Args:
        input_data (np.array): predictor values for which we would like to perfrom predictions (1 column for 1 predictor) (1D or 2D np.array)
        predictors (np.array): numeric array of predictor values (1 column for 1 predictor) (1D or 2D np.array)
        outputs (np.array): numeric array of known system outputs
        
    Returns:
        [type]: [description]
    """
    categories = np.unique(outputs)
    mu = np.zeros((len(categories), predictors.shape[1]))
    pi = np.zeros(len(categories))
    for j, i in enumerate(categories):
        avg = np.average(predictors[outputs==i], axis=0)
        mu[j, :] = avg
        pi[j] = predictors[outputs==i].shape[0]/predictors.shape[0]
    p_matrix = np.cov(predictors.astype(float).T)
    # discriminant function
    predictions_ = []
    for j,i in enumerate(input_data):
        delta_ = []
        for k in range(len(categories)):
            delta_.append(lin_discriminant_function(i, mu[k,:], p_matrix, pi[k]))
        category_ind = np.argmax(delta_)
        predictions_.append(categories[category_ind])
    
    return np.array(predictions_)


# LDA method for 2 class preditions which enables Bayes decision boundary modification
def LDA_decision_boundary_mod(input_data, predictors, outputs,
                              decision_boundary=(1, .5)):
    """LDE for 2 class prediction which enables Bayes decision boundary modification.
    
    Args:
        input_data (np.array): predictor values for which we would like to perfrom predictions (1 column for 1 predictor)
                               (1D or 2D np.array)
        predictors (np.array): numeric array of predictor values (1 column for 1 predictor) (1D or 2D np.array)
        outputs (np.array): numeric array of known system outputs
        decision_boundary (tuple of int,float; optional): Bayes decision boundary modification - tuple
                                (class for which we are setting decision 
                                boundary [0 or 1], decision boundary
                                [0 to 1])
    Returns:
        numpy array: predicition by LDA
    :param 
    """
    categories = np.unique(outputs)
    mu = np.zeros((len(categories), predictors.shape[1]))
    pi = np.zeros(len(categories))
    for j, i in enumerate(categories):
        avg = np.average(predictors[outputs==i], axis=0)
        mu[j, :] = avg
        pi[j] = predictors[outputs==i].shape[0]/predictors.shape[0]
    p_matrix = np.cov(predictors.astype(float).T)
    # discriminant function
    predictions_ = []
    for j,i in enumerate(input_data):
        delta_ = []
        for k in range(len(categories)):
            delta_.append(lin_discriminant_function(i, mu[k,:],
                                                    p_matrix, pi[k]))
        delta_ratio = delta_[decision_boundary[0]]/(delta_[decision_boundary[0]] + delta_[not decision_boundary[0]])
        if delta_ratio > decision_boundary[1]:
            predictions_.append(categories[decision_boundary[0]])
        else:
            predictions_.append(categories[int(not decision_boundary[0])])
    return np.array(predictions_)


# Quadratic Discriminant Analysis (QDA)
def quad_discriminant_function(x, μ, Σ, π):
    """Function calculates the value of quadratic disciriminant function for given x.

    Args:
        x (np.array): vector of predictor values for which we would like to perform a prediction
        μ (np.array): vector of mean values for individual predictors
        Σ (np.array): covariance matrix
        π (float): prior probability

    Returns:
        float: values of discriminant function
    """
    return -0.5*x.T@np.linalg.inv(Σ)@x+\
            x.T@np.linalg.inv(Σ)@μ-0.5*μ.T@np.linalg.inv(Σ)@μ -\
            0.5*np.log(np.linalg.det(Σ)) + np.log(π)

def quadratic_discriminant_analysis(input_data, predictors, outputs):
    """Function applies QDA method for making predictions

    Args:
        input_data (np.array): predictor values for which we would like to perfrom predictions (1 column for 1 predictor)
                               (1D or 2D np.array)
        predictors (np.array): numeric array of predictor values (1 column for 1 predictor) (1D or 2D np.array)
        outputs (np.array): numeric array of known system outputs
        
    Returns:
        numpy array: predicition by QDA
    """
    categories = np.unique(outputs)
    mu = np.zeros((len(categories), predictors.shape[1]))
    pi = np.zeros(len(categories))
    p_matrix = np.zeros((len(categories), predictors.shape[1], predictors.shape[1]))
    for j, i in enumerate(categories):
        avg = np.average(predictors[outputs==i], axis=0)
        mu[j, :] = avg
        pi[j] = predictors[outputs==i].shape[0]/predictors.shape[0]
        p_matrix[j, :, :] = np.cov(predictors[outputs==i].astype(float).T)
    # discriminant function
    predictions_ = []
    for j,i in enumerate(tqdm(input_data)):
        delta_ = []
        for k in range(len(categories)):
            delta_.append(quad_discriminant_function(i, mu[k,:],
                                                    p_matrix[k,:,:], pi[k]))
        category_ind = np.argmax(delta_)
        predictions_.append(categories[category_ind])
    
    return np.array(predictions_)


# decision trees
def find_s(x_in, y_ref, s):
    """
    Function determines 's' parameter for indicidual predictor by minimization of RSS.
    """
    RSS_min = np.array([np.inf])
    s_min = min(s)
    for i in s:
        R1_ind = np.argwhere(x_in<i)
        R2_ind = np.argwhere(x_in>=i)
        if (len(R1_ind) !=0) & (len(R2_ind)!= 0):
            R1_mean = np.average(y_ref[R1_ind])
            R2_mean = np.average(y_ref[R2_ind])
            R1_RSS = sum((x_in[R1_ind]-R1_mean)**2)
            R2_RSS = sum((x_in[R2_ind]-R2_mean)**2)
            RSS_s = R1_RSS + R2_RSS
            if RSS_s<RSS_min:
                RSS_min = RSS_s
                s_min = i
        else:
            pass
    return RSS_min, s_min


def split(x_in, y_ref, s_partitions):
    """
    Function finds split parameters for individual split according to recursice binary splitting method.
    x_in: pandas dataframe of input predictors
    y_ref: pandas series of reference responses
    s_partitions: number of partition between min and max predicor values when searching for minimum s value for individual predictor.
    
    return: index of predictor with min RSS, s_value for predictor with min RSS.
    """
    pred_list = list(x_in.columns)
    ref = np.array(y_ref)
    RSS_min = []
    s_min = []
    for i in pred_list:
        d_series = np.array(x_in[i])
        s_values = np.linspace(min(d_series), max(d_series),s_partitions)[1:-1] # s values for individual predictor
        RSS_, s_ = find_s(d_series, ref, s_values)
        RSS_min.append(RSS_)
        s_min.append(s_)
    min_ind = np.argmin(RSS_min)
    pred_min = pred_list[min_ind]
    s_min = s_min[min_ind]
    return pred_min, s_min


def initial(x_in):
    predictors = list(x_in.columns)
    R_dict = {}
    for i in predictors:
        R_dict[i] = [min(x_in[i]),max(x_in[i])]
    return R_dict

# ne dela še
def decision_tree(x_in, y_ref, s_partitions, R_size_max):
    R_list = {'R0': initial(x_in)}
    R_x_data = {'R0':x_in}
    R_x_temp = R_x_data.copy()
    R_y_data = {'R0':y_ref}
    predictors = list(x_in.columns)
    R_size=np.inf
    name_ = 0
    while R_size > R_size_max:
        sizes=[]
        for j, i in enumerate(R_x_temp):
            print(j,i)
            if R_x_data[i].shape[0]>R_size_max:
                pred_, s_ = split(R_x_data[i], R_y_data[i], s_partitions=s_partitions)
                old = R_list[i][pred_].copy()
                name_+=1
                R_list[f'R{name_}'] = R_list[i].copy()
                R_list[f'R{name_}'][pred_] = [s_, old[1]]
                mask_1 = list(((R_x_data[i][pred_]>=R_list[f'R{name_}'][pred_][0]) & (R_x_data[i][pred_]<=R_list[f'R{name_}'][pred_][1])))
                R_x_data[f'R{name_}'] = R_x_data[i][mask_1]
                R_y_data[f'R{name_}'] = R_y_data[i][mask_1]
                size_0 = R_x_data[f'R{name_}'].shape[0]
                if size_0 == 0:
                    del R_x_data[f'R{name_}']
                    del R_y_data[f'R{name_}']
                else:
                    sizes.append(size_0)
                name_+=1
                R_list[f'R{name_}'] = R_list[i].copy()
                R_list[f'R{name_}'][pred_] = [old[0],s_]
                mask_2 = list(((R_x_data[i][pred_]>=R_list[f'R{name_}'][pred_][0]) & (R_x_data[i][pred_]<R_list[f'R{name_}'][pred_][1])))
                R_x_data[f'R{name_}'] = R_x_data[i][mask_2]
                R_y_data[f'R{name_}'] = R_y_data[i][mask_2]
                size_1 = R_x_data[f'R{name_}'].shape[0]
                if size_1 ==0:
                    del R_x_data[f'R{name_}']
                    del R_y_data[f'R{name_}']
                else:
                    sizes.append(size_1)
                del R_list[i]
                del R_x_data[i]
                del R_y_data[i]
            else:
                sizes.append(R_x_data[i].shape[0])
                pass
        R_x_temp = R_x_data.copy()
        R_size=max(sizes)
        print(list(R_list.keys()), sizes, sum(sizes))
        print('-----------------------------------------')
    print(R_list)
    
    return R_list


# k-fold Cross Validation

def k_fold_split(input_, output_, k):
    indices = list(np.arange(input_.shape[0], dtype=int))
    fold_size = int(input_.shape[0]/k)
    input_folds = []
    output_folds = []
    for i in range(k-1):
        k_ind = np.zeros(fold_size, dtype=int)
        for j in range(fold_size):
            k_ind[j] = random.choice(indices)
            indices.remove(k_ind[j])
        input_folds.append(input_[k_ind])
        output_folds.append(output_[k_ind])
    input_folds.append(input_[indices])
    output_folds.append(output_[indices])
    return np.array(input_folds), np.array(output_folds)


def classification_error_rate(pred_, ref_):
    n = ref_.shape[0]
    return 1/n*(n-np.count_nonzero(pred_==ref_))

    
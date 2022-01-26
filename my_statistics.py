import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
import sys
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import prettytable


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


def KNN(test_point, input_data_, output_data, K, normalize=False, show=False, dummy=False):
    """
    Function applies K Nearest Neighbors method.
    Args:
        test_point: test point location (np.array)
        input_data_: 1D or 2D array of feature values (1 column for individual feature)
        output_data: output for given input_data (np.array)
        K: number of points taken into account by KNN method
        normalize: iput_data is normalized to the values from 0 to 1
        show: Plot graph of points

    Returns: category of test_point

    """
    if normalize:
        input_data = input_data_ / input_data_.max(axis=0)
    else:
        input_data = input_data_

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
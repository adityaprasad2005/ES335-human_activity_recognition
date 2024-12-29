""" Complete the performance metrics functions in this file"""

from typing import Union
import numpy as np
import pandas as pd

# y_hat= predicted values and y= ground truth values

def get_accuracy(y_hat, y):
    """
    Inputs:
        y_hat: numpy array of predicted values
        y: numpy array of actual value

    Outputs:
        acc: accuracy
    """

    assert len(y_hat) == len(y)

    acc= sum(y_hat==y)/len(y)

    return acc

def get_precision(y_hat, y, cls) :
    """
    Inputs:
        y_hat: numpy array of predicted values
        y: numpy array of actual value
        cls: class

    Outputs:
        pres_cls: precision
    """

    assert y_hat.size == y.size 

    if cls not in np.unique(y_hat):
        return None

    y_hat_cls= y_hat[y_hat == cls]
    y_cls= y[y_hat == cls]

    pres_cls = sum(y_hat_cls== y_cls)/len(y_hat_cls)

    return pres_cls


def get_recall(y_hat, y, cls):
    """
    Inputs:
        y_hat: numpy array of predicted values
        y: numpy array of actual value
        cls: class

    Outputs:
        rec_cls: recall
    """

    assert y_hat.size == y.size 

    if cls not in np.unique(y):
        return None

    y_cls= y[y==cls]
    y_hat_cls= y_hat[y==cls]

    rec_cls= sum(y_cls== y_hat_cls)/len(y_cls)

    return rec_cls

def get_f1_score(y_hat, y, cls) :
    """
    Inputs:
        y_hat: numpy array of predicted values
        y: numpy array of actual value
        cls: class

    Outputs:
        f1_cls: f1 score
    """

    assert y_hat.size == y.size 

    if cls not in np.unique(y_hat):
        return None

    pres_cls= get_precision(y_hat, y, cls)
    rec_cls= get_recall(y_hat, y, cls)

    f1_cls= 2*pres_cls*rec_cls/(pres_cls+rec_cls)

    return f1_cls

def get_rmse(y_hat, y) :
    """
    Inputs:
        y_hat: numpy array of predicted values
        y: numpy array of actual value

    Outputs:
        rmse: root mean square error
    """
    assert y_hat.size == y.size 

    sum_sq_err= np.square(y_hat - y).sum()

    mean_sq_err= sum_sq_err/len(y)

    return np.sqrt(mean_sq_err)


def get_mae(y_hat, y) :
    """
    Inputs:
        y_hat: numpy array of predicted values
        y: numpy array of actual value

    Outputs:
        mae: mean absolute error
    """

    assert y_hat.size == y.size 

    sum_abs_err= np.abs(y_hat-y).sum()

    mean_abs_err= sum_abs_err/y.size

    return mean_abs_err

def get_mcc_score(y_hat, y) :

    """Function to calculate the mathew's correlation coefficient"""

    assert y_hat.size == y.size
    assert np.unique(y_hat).size == 2
    
    cls1 = np.unique(y_hat)[0]
    cls2 = np.unique(y_hat)[1]

    TP = sum( (y_hat == cls2) & (y == cls2))
    FP = sum( (y_hat == cls2) & (y == cls1))
    TN = sum( (y_hat == cls1) & (y == cls1))
    FN = sum( (y_hat == cls1) & (y == cls2))

    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return MCC
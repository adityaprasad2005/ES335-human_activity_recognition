"""Complete all utility functions."""

import pandas as pd
import numpy as np

def get_info_gain_real_ip_dis_op(X, y, attr_num, sep_val):    # For real input data and discrete output

    """
    Inputs:
    X : numpy array of shape (num_samples, num_features)
    y : numpy array of shape (num_samples, )
    attr_num : int column number
    sep_val : float separating value for the attribute

    Outputs:
    info_gain : float
    """
    assert X.shape[0] == y.shape[0]

    entropy_initial = get_entropy(y)

    entropy_final = 0

    # X_left = X[X[:, attr_num] <= sep_val]
    y_left = y[X[:, attr_num] <= sep_val]
    w = len(y_left)/len(y)
    entropy_final += w*get_entropy(y_left)

    # X_right = X[X[:, attr_num] > sep_val]
    y_right = y[X[:, attr_num] > sep_val]
    w = len(y_right)/len(y) 
    entropy_final += w*get_entropy(y_right)
    
    info_gain = entropy_initial - entropy_final

    return info_gain

def get_info_gain_real_ip_real_op(X, y, sep_val, feat_):    # For real input data and real output

    """
    Inputs:
    X : numpy array of shape (num_samples, 1)
    y : numpy array of shape (num_samples, )
    sep_val : float separating value
    feat_ : int column number

    Outputs:
    info_gain : float
    """
    assert X.shape[0] == y.shape[0]

    rmse_initial = get_rmse(y)

    rmse_final = 0

    # X_left = X[X[:, 0] <= sep_val]
    y_left = y[X[:, feat_] <= sep_val]
    rmse_final += len(y_left)/len(y)*get_rmse(y_left)

    # X_right = X[X[:, 0] > sep_val]
    y_right = y[X[:, feat_] > sep_val]
    rmse_final += len(y_right)/len(y)*get_rmse(y_right)
    
    info_gain = rmse_initial - rmse_final

    return info_gain


def get_info_gain_dis_ip_real_op(X, y, attr_num):    # For Discrete input data and real output

    """
    Inputs:
    X : numpy array of shape (num_samples, num_features)
    y : numpy array of shape (num_samples, )
    attr_num : int column number

    Outputs:
    info_gain : float
    """
    assert X.shape[0] == y.shape[0]

    rmse_initial = get_rmse(y)

    rmse_final = 0

    unique_vals = np.unique(X[:, attr_num])
    for cls in unique_vals:

        y_temp = y[X[:, attr_num] == cls]
        weight = len(y_temp)/len(y)
        r = get_rmse(y_temp)
        rmse_final += weight*r
    
    info_gain = rmse_initial - rmse_final

    return info_gain
    
def get_info_gain_dis_ip_dis_op(X, y, attr_num):    # For Discrete input data and discrete output

    """
    Inputs:
    X : numpy array of shape (num_samples, num_features)
    y : numpy array of shape (num_samples, )
    attr_num : int column number

    Outputs:
    info_gain : float
    """
    assert X.shape[0] == y.shape[0]

    entropy_initial = get_entropy(y)

    entropy_final = 0

    unique_vals = np.unique(X[:, attr_num])
    for cls in unique_vals:

        y_temp = y[X[:, attr_num] == cls]
        weight = len(y_temp)/len(y)
        ent = get_entropy(y_temp)
        entropy_final += weight*ent
    
    info_gain = entropy_initial - entropy_final

    return info_gain
    
def get_entropy(y):
    """
    Inputs :
    y : numpy array of shape (num_samples, )

    Outputs:
    entropy : float
    """

    entropy= 0
    for cls in np.unique(y):
        prob_cls= len(y[y==cls])/len(y)

        if prob_cls == 0:     # to avoid log(0)
            prob_cls = 1e-6

        entropy += -prob_cls*np.log2(prob_cls)

    return entropy

def get_rmse(y):
    """
    Inputs:
    y : numpy array of shape (num_samples, )

    Outputs:
    rmse : float
    """
    
    sum_squared_err= sum((y-np.mean(y))**2)
    mean_squared_err= sum_squared_err/len(y)
    rmse= np.sqrt(mean_squared_err)

    return rmse

def get_gini_index(y):
    """
    Inputs:
    y : numpy array of shape (num_samples, )

    Outputs:
    gini : float
    """

    gini= 1

    for cls in np.unique(y):
        prob_cls= len(y[y==cls])/len(y)
        gini += -prob_cls**2

    return gini


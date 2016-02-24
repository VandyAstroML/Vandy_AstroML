#! /usr/bin/env python

# Victor Calderon
# Vanderbilt University
# Project: Digits dataset for VandyAstroML project

"""
Description: Decision-tree algorithm on the dataset "Digits" from Scikit-learn.
"""
# modules for Python2/3
from __future__ import division, print_function, absolute_import

# Importing modules
import numpy as num
import random
from sklearn import datasets
from collections import Counter

class Best_Choices():
    def __init_(self, results=None, true_tree=None, false_tree=None,col=-1 ):
        self.results    = results # Dictionary of results
        self.true_tree  = None    # Subtree for the `true` tree
        self.false_tree = None    # SubTree for the `false` tree
        self.col        = -1      # Chosen column number

def splitting_branches(data, ncol, val):
    """
    Separatates `ncol` from data into two branches, depending on the conditions 
    based on `val`

    Parameters
    ----------
    data: array_like
        N-dimensional array with the data

    ncol: int
        column number used to categorized the two output sets (arrays).

    val: string, int, or float
        value used to categorize `data`

    Returns
    -------
    set_arrs: array_like
        array containing both sets, i.e. arr1 and arr2, based on the criteria of 
        `val`.
    """
    # Checking if value is an integer, a float, or something else
    if isinstance(val, int) or 
        val_type = lambda data: data[ncol] >= val # for numbers
    else:
        val_type = lambda data: data[ncol] == val # for strings or other
    # Saving into arrays (and making sure they are lists, and not arrays)
    set_corr = [row for row in data if val_type(row)]
    set_fail = [row for row in data if not val_type(row)]
    set_arrs = (set_corr, set_fail)

    return set_arrs

def counter_of_targets(data):
    """
    Calculates the total number of occurences for each of the possible values of 
    the targets for each (sub)array.

    Parameters
    ----------
    data: array_like
        N-dimensional array or 1-d array

    Returns
    -------
    Count_dict: dictionary
        Dictionary with possible values and their counts
    """
    ## Assuming it's a list
    Count_obj = Counter(num.array(data).T[-1])
    Count_dict = {}
    for key in Count_obj.keys():
    Count_dict[key] = Count_obj[key]

    return Count_dict

def entropy_calculation(data):
    """
    Calculated the `entropy` of the dataset. This is a measure of how unpure 
    the mix of `target` values is. Ideally, entropy==0.

    Parameters
    ----------
    data_dict: dictionary
        dictionary containing counts for possible `target` values

    Returns
    -------
    ent_val: float
        value for the estimated entropy of the given `data`
    """
    counts  = counter_of_targets(data)
    ent_val = 0.
    # Looping over all `target` values in `counts`
    for key in counts.keys():
        p_val = counts[key]/float(len(data))
        # Entropy formula
        ent_val -= p_val*num.log2(p_val)

    return ent_val

def Tree_training(data, estimator=entropy_calculation):
    """
    This function iterates over all possible values for each of the columns 
    of `data` and constructs the whole Decision Tree structure

    Parameters
    ----------
    data: array_like
        N-dimensional array, with `targets` being the last column.

    estimator: function
        Estimator that determines the cost function of the data, i.e. how 
        (un)pure `data` is

    Returns
    -------
    Best_vals: class
        This class contains information on the best values obtained through the 
        Decision Tree constructed.
    """
    

def new_data_classification(data, tree_obj):
    """
    Classifies (predicts) `data` according to `tree_obj`

    Parameters
    ----------
    data: array_like
        N-dimensional array containing values to be classified

    tree_obj: DecisionTree object after it was trained

    Returns
    -------
    obj_class: dictionary
        Prediciton value for `data` using `tree_obj`
    """


def main():
    ## Loading dataset
    digits = datasets.load_digits()
    # Digits - Data
    digits_dat = digits['data']
    n_samples  = digits_dat.shape[0]
    n_features = digits_dat.shape[1]
    # Digits - target names
    digits_tnames = digits['target_names'].astype(int)
    # Digits - target values
    digits_target = digits['target']

    ## Training and control samples
    # Selects 80 percent of sample as training sample 
    # and the other 20 percent as the control sample.
    # Want to select 80 percent of all types of targets
    train_frac = 0.8
    # training set - indices
    train_idx  = []
    for target in digits_tnames:
        tnames_idx    = num.where(digits_target==target)[0]
        tnames_size   = int(tnames_idx.size*train_frac)
        tnames_sample = random.sample(tnames_idx, tnames_size)
        train_idx.extend(tnames_sample)
    train_idx = num.array(train_idx)
    # Control set - indices
    cont_idx  = set(range(len(digits_target))).difference(train_idx)
    cont_idx  = num.array(list(cont_idx))
    # Training set - dictionary
    train_dict = {}
    train_dict['data'  ] = digits_dat   [train_idx]
    train_dict['target'] = digits_target[train_idx]
    train_data_target    = num.column_stack((train_dict['data'],
        train_dict['target']))
    # Combining data and target
    # Control set - dictionary
    cont_dict = {}
    cont_dict['data'  ] = digits_dat   [cont_idx]
    cont_dict['target'] = digits_target[cont_idx]
    cont_data_target    = num.column_stack((cont_dict['data'],
        cont_dict['target']))



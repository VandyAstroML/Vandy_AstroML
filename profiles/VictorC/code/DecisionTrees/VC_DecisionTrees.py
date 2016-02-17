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
from sklearn import datasets

def main():
	## Loading dataset
	digits = datasets.load_digits()
	# Digits - Data
	digits_dat = digits['data']
	n_samples  = digits_dat.shape[0]
	n_features = digits_dat.shape[1]
	# Digits - target names
	digits_tnames = digits['target_names']
	# Digits - target values
	digits_target = digits['target']



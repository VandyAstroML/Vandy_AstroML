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
	(arr1,arr2): array_like
		array containing both sets, i.e. arr1 and arr2, based on the criteria of 
		`val`.
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



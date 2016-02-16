import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# Load digits data -- all numpy arrays
# _data        = raw pixel values
# _targetnames = possible numbers (0-9)
# _target      = "truth values"
digits             = load_digits()
digits_data        = digits['data']
digits_targetnames = digits['target_names']
digits_target      = digits['target']


# Choose fraction of training examples

frac_train = 0.8

# Create lists on indices for each target type so we can perform data splitting
target_index = {}

# initialize a list for each target number
for i in range(len(digits_targetnames)):
    key               = str(i)
    target_index[key] = []


for i in range(len(digits_target)):

    for j in digits_targetnames:
        if digits_target[i] == j:
            key = str(j)
            target_index[key].append(i)

# print(target_index)

print(digits_target[5], digits_target[33], digits_target[162])
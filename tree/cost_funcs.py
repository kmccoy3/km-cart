#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python module containing cost functions to be used within decision tree.
"""


# =============================================================================
# AUTHOR INFORMATION
# =============================================================================

__author__ = "Kevin McCoy"
__copyright__ = "Copyright 2024, McCoy and Peterson"
__credits__ = ["Kevin McCoy", "Christine Peterson"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Christine Peterson"
__email__ = ["CBPeterson@mdanderson.org", "kmccoy1@rice.edu"]
__status__ = "development"
__date__ = "2024-02-08" # Last modified date

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

# =============================================================================
# COST FUNCTIONS
# =============================================================================

def gini_index(data):
    gini = 1
    for label in data['label'].unique():
        gini -= (len(data[data['label'] == label]) / len(data)) ** 2
    return gini


def group_gini_index(data):
    gini = 0
    for group in data['group'].unique():
        gini += (len(data[data['group'] == group]) / len(data)) ** 2

    return gini


def sum_of_squares(data):
    return np.sum((data['label'] - np.mean(data['label'])) ** 2)


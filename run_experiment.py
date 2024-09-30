#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python module containing functions to run single experiment.
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

# Standard libraries
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score

# Local imports
from tree.tree_node import tree_node
from tree.cost_funcs import *
from tree.data_funcs import *


# =============================================================================
# FUNCTIONS
# =============================================================================


def hyperparameter_tuning(df_train, df_val, deltas):
    """_summary_

    Args:
        df_train (_type_): _description_
        df_val (_type_): _description_
        deltas (_type_): _description_

    Returns:
        _type_: _description_
    """

    best_delta = 0
    best_score = 0

    for delta in deltas:
        tree = tree_node(df_train)
        tree.build_tree('modified_gini', delta=delta)

        y_pred = tree.evaluate(df_val)
        y_true = df_val['label']

        score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_delta = delta

    return best_delta, best_score



def run_experiment(sd, rs=1):
    """_summary_

    Args:
        sd (_type_): _description_
    """

    # TODO: add parameters from terminal

    data = generate_data(seed=sd, p=5, n=25, k=25, random_sigma=rs, noise_sigma=0.2, gen_type="Friedman")

    data = data.drop(columns=['random_effect', 'noise'])

    data = center_groups(data)

    df_train, df_val, df_test = split_data(data, 0.2, 0.2)

    df_train_val = pd.concat([df_train, df_val])



    ############################
    # Regular Gini

    tree = tree_node(df_train_val)
    tree.build_tree('gini')


    predictions = tree.evaluate(df_test)

    
    new_entry1 = pd.DataFrame([[sd, "gini", "accuracy", accuracy_score(df_test['label'], predictions)]],
                             columns=['sd','method','metric', 'value'])
    new_entry2 = pd.DataFrame([[sd, "gini", "precision", precision_score(df_test['label'], predictions, zero_division=np.nan)]],
                             columns=['sd','method','metric', 'value'])
    new_entry3 = pd.DataFrame([[sd, "gini", "recall", recall_score(df_test['label'], predictions, zero_division=np.nan)]],
                             columns=['sd','method','metric', 'value'])
    new_entry4 = pd.DataFrame([[sd, "gini", "auprc", average_precision_score(df_test['label'], predictions)]],
                             columns=['sd','method','metric', 'value'])
    
    test_results_df = pd.concat([new_entry1, new_entry2, new_entry3, new_entry4])

    ############################
    # Modified Gini

    deltas = np.linspace(0, 0.5, 11)

    best_delta, _ = hyperparameter_tuning(df_train, df_val, deltas)

    tree = tree_node(df_train_val)
    tree.build_tree('modified_gini', delta=best_delta)



    predictions = tree.evaluate(df_test)

    new_entry1 = pd.DataFrame([[sd, "modified_gini", "accuracy", accuracy_score(df_test['label'], predictions)]],
                             columns=['sd','method','metric', 'value'])
    new_entry2 = pd.DataFrame([[sd, "modified_gini", "precision", precision_score(df_test['label'], predictions, zero_division=np.nan)]],
                             columns=['sd','method','metric', 'value'])
    new_entry3 = pd.DataFrame([[sd, "modified_gini", "recall", recall_score(df_test['label'], predictions, zero_division=np.nan)]],
                             columns=['sd','method','metric', 'value'])
    new_entry4 = pd.DataFrame([[sd, "modified_gini", "auprc", average_precision_score(df_test['label'], predictions)]],
                             columns=['sd','method','metric', 'value'])
    
    test_results_df = pd.concat([test_results_df, new_entry1, new_entry2, new_entry3, new_entry4])

    test_results_df.to_csv("."+f"/results/random_sigma_{rs}".replace('.', '-')+".csv", index=False, mode='a', header=False)


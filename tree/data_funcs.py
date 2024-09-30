#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python module containing functions to generate and plot simulated data.
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import matrix_normal, invwishart, uniform
from scipy.special import expit


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def sign(num):
    return 0 if num < 0 else 1


def generate_data(seed=0, p=2, n=20, k=5, random_sigma=0.2, noise_sigma=0.2, gen_type="Linear"):
    """Generate synthetic data with mixed effects.

    Args:
        seed (int, optional): Random number generator seed. Defaults to 0.
        p (int, optional): Number of dimensions of observable input data. Defaults to 2.
        n (int, optional): Number of observations per group. Defaults to 120.
        k (int, optional): Number of groups. Defaults to 6.
        random_sigma (float, optional): _description_. Defaults to 0.2.
        noise_sigma (float, optional): _description_. Defaults to 0.2.
        mu (float, optional): _description_. Defaults to 0.

    Returns:
        pd.DataFrame: Generated data with columns 'X1', 'X2', 'random_effect', 'noise', 'group', 'label'
    """

    # TODO: Add random effect slopes in addition to intercepts (change beta)
    # TODO: Add interaction terms (Friedman function)
    # TODO: Add option to generate continuous labels
    # TODO: Add categorical input variables

    # Set random seed
    np.random.seed(seed)

    # Generate main covariates
    # M = np.random.uniform(-1, 1, size=(k, p)) # group means
    # U = np.identity(k) # row covariance (group covariance)
    # V = invwishart.rvs(df=p+1, scale=np.identity(p)) # column covariance (covariance of features)

    # M = np.zeros((k, p))
    # V = np.identity(p)



    # X = matrix_normal.rvs(mean=M, rowcov=U, colcov=V, size=n)

    # # Reshape X to be a 2D array
    # X = X.reshape((n*k, p))

    X = uniform.rvs(size=(n*k, p))

    # Generate group labels
    K = np.tile(np.arange(k), n)

    # Create dataframe
    df = pd.DataFrame(X, columns=["X" + str(i) for i in range(1, p+1)])
 
    # Generate random effects and common noise terms
    random_effect = np.tile(np.random.normal(0,random_sigma,k), n)
    noise = np.random.normal(0,noise_sigma,n*k)



    match gen_type:
        case "Friedman":
            # Generate label based on Friedman function
            if p != 5:
                raise ValueError("Friedman function requires 5 covariates!")

            probs = np.sin(np.pi * df['X1'] * df['X2']) + 2 * (df['X3'] - 0.5)**2 + df['X4'] + 0.5 * df['X5'] 
        case "Linear":
            # Generate label based on linear function
            beta = np.random.uniform(-1, 1, size=p)
            probs = np.matmul(df.values, beta) + random_effect + noise
        case _:
            raise ValueError("Invalid gen_type. Must be 'Friedman' or 'Linear'")

    # Add random effects and noise columns to dataframe
    df['random_effect'] = random_effect
    df['noise'] = noise

    # Generate probabilities for binary label
    probs = probs - np.mean(probs) # center around 0
    probs = probs / np.std(probs) # scale to unit variance

    probs = probs + random_effect + noise



    # probs = expit(*probs)

    # Generate binary labels
    df['group'] = K
    # df['label'] = np.random.binomial(1, probs)
    df['label'] = [sign(prob) for prob in probs]


    # Output final dataframe
    return df


def plot_data(df, filename=None):
    """Plot synthetic data with mixed effects.

    Args:
        pd.dataframe: Dataframe containing synthetic data with columns 'X1', 
            'X2', 'random_effect', 'noise', 'group', 'label'
    """

    # Check that dataframe has the correct number of columns
    if 'X3' in df.columns:
        raise ValueError("Dataframe must only have 2 covariates in order to plot!")
        # TODO: Add PCA capability

    # Plot data
    fig, ax = plt.subplots(figsize=(8, 8))

    # FIXME: palette / legend not working
    sns.scatterplot(data=df, x='X1', y='X2', hue='group', style='label', palette=sns.color_palette("Spectral", as_cmap=True))

    plt.axis('equal')
    plt.legend(fontsize=14)
    plt.title('Synthetic Mixed Effect Data', fontsize=14)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Save plot if filename is provided
    if filename:
        plt.savefig(filename, dpi=600)

    plt.show()

    return fig, ax


def split_data(data, val_size=0.2, test_size=0.2):
    """_summary_

    Args:
        data (_type_): _description_
        val_size (float, optional): _description_. Defaults to 0.2.
        test_size (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """

    n = len(data['group'].unique())
    val_size = int(n * val_size)
    test_size = int(n * test_size)
    train_size = n - val_size - test_size

    df_train = data[data['group'] < train_size]
    df_val = data[(data['group'] >= train_size) & (data['group'] < train_size + val_size)]
    df_test = data[data['group'] >= train_size + val_size]

    return df_train, df_val, df_test

def center_groups(data):
    # TODO: Add support for >2 covariates
    for group in data['group'].unique():
        idx = data['group'] == group
        data.loc[idx, 'X1'] = data.loc[idx, 'X1'] - np.mean(data.loc[idx, 'X1'])
        data.loc[idx, 'X2'] = data.loc[idx, 'X2'] - np.mean(data.loc[idx, 'X2'])
    
    return data



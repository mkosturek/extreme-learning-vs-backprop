import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as posthocs


def prepare_measures_for_friedman_test(results_df: pd.DataFrame,
                                       hyperparameter_name: str = 'activation_fn',
                                       metric: str = 'accuracy'):

    measures_df = results_df.pivot_table(index=['dataset_name'],
                                         columns=[hyperparameter_name],
                                         values=metric)
    measures = []
    for i in range(len(measures_df.columns)):
        measures.append(measures_df.values[:, i])
    return measures


def friedman_test(results_df: pd.DataFrame,
                  hyperparameter_name: str = 'activation_fn',
                  metric: str = 'accuracy'):

    measures = prepare_measures_for_friedman_test(results_df, hyperparameter_name, metric)
    return friedmanchisquare(*measures)


def nemenyi_test(results_df: pd.DataFrame,
                 hyperparameter_name: str = 'activation_fn',
                 metric: str = 'accuracy'):

    mdf = results_df.groupby(['dataset_name',
                              hyperparameter_name]).mean().reset_index()

    nemenyi = posthocs.posthoc_nemenyi_friedman(mdf, melted=True,
                                                y_col=metric,
                                                group_col=hyperparameter_name,
                                                block_col='dataset_name')
    return nemenyi

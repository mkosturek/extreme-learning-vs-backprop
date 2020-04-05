import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict


def total_variance(results: pd.DataFrame,
                   variables: List[str],
                   metric: str = 'accuracy') -> float:
    df = results.groupby(variables).mean().reset_index()
    return np.var(df[metric], ddof=0)


def decompose_variance(results: pd.DataFrame,
                       variables: List[str],
                       metric: str = 'accuracy') -> dict:

    df = results.groupby(variables).mean().reset_index()

    orders = range(1, len(variables)+1)

    V = dict()
    V[''] = np.var(df[metric], ddof=0)

    for order in orders:
        for combination in combinations(variables, r=order):
            v = np.var(df.groupby(list(combination)).mean()[metric], ddof=0)

            for ord in range(1,order):
                for lower_order_elements in combinations(combination, r=ord):
                    v -= V[lower_order_elements]

            V[combination] = v

    return V


def sensitivity_indices(decomposed_variances: dict, overall_variance: float):

    S = dict()
    for k, v in decomposed_variances.items():
        if k != '':
            S[k] = v / overall_variance

    return S



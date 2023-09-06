import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import *
from itertools import combinations
from baynet import DAG

"""
Adding pruning rules (see arXiv:1707.06194) to benchmark queries
"""

def entropy(col: pd.Series) -> float:
    counts = col.value_counts().to_numpy()
    p_arr = counts/col.size
    #h = - np.sum(p * np.log2(p))
    h = - np.sum([p * np.log2(p) for p in p_arr if p > 0])
    return h

def per_var(data: pd.DataFrame) -> List[int]:
    """
    Theorem 3 pruning: per-variable in-degree cap using each variable's entropy.
    """
    log_b = np.log2
    variables = data.columns
    n = data.shape[0]
    entropies = []
    levels = []
    for var in variables:
        levels.append(len(data[var].cat.categories))
        entropies.append(entropy(data[var]))
    caps = []
    for x_idx in range(data.shape[1]-1):
        other_vars = list(range(data.shape[1]))
        other_vars.remove(x_idx)
        H_x = entropies[x_idx]
        x_cap = 0
        for y_idx in other_vars:
            H_y = entropies[y_idx]
            x_cap_candidate = np.ceil(1.0 + np.log2(min(H_x, H_y)/((levels[y_idx]-1)*(levels[x_idx]-1))) + np.log2(n) - np.log2(log_b(n)))
            x_cap = max(x_cap, x_cap_candidate)
        caps.append(int(x_cap))
    return caps

def global_cap(data: pd.DataFrame) -> int:
    """
    Corollary 1
    """
    n = data.shape[0]
    log_b = np.log2
    cap = int(np.ceil(1 + np.log2(n) - np.log2(log_b(n))))
    return cap

def pruning_comparison(data: pd.DataFrame, percentages: bool) -> List[int]:
    """
    Compute number of scores/entropies to be computed based on pruning available
    """
    C_global = global_cap(data)
    C_x_list = per_var(data)

    d = len(data.columns)
    naive_scores = d * 2**(d-1)
    naive_entropies = 2*naive_scores
    naive_intersections = 0

    entropy_sets = set()
    scores = []

    global_entropy_sets = set()
    global_scores = []

    for x_idx in range(data.shape[1]-1):
        other_vars = list(range(data.shape[1]))
        other_vars.remove(x_idx)
        for set_size in range(len(data.columns)+1):
            for Pi_x in combinations(other_vars, set_size):
                naive_intersections += (len(Pi_x)*2-1)
                if set_size <= C_global:
                    global_entropy_sets.add(frozenset(Pi_x))
                    global_entropy_sets.add(frozenset(list(Pi_x) + [x_idx]))
                    global_scores.append(frozenset(list(Pi_x) + [x_idx])) #(f"{x_idx}|{Pi_x}")
                if set_size <= C_x_list[x_idx]:
                    entropy_sets.add(frozenset(Pi_x))
                    entropy_sets.add(frozenset(list(Pi_x) + [x_idx]))
                    scores.append(frozenset(list(Pi_x) + [x_idx])) #(f"{x_idx}|{Pi_x}")

    intersections = sum(len(x)-1 for x in entropy_sets)
    sabna_intersections = sum(len(x)-1 for x in scores)
    global_intersections = sum(len(x)-1 for x in global_entropy_sets)
    sabna_global_intersections = sum(len(x)-1 for x in global_scores)

    if not percentages:
        # return [len(global_entropy_sets), len(global_scores), global_intersections, sabna_global_intersections, 
        #         len(entropy_sets), len(scores), intersections, sabna_intersections]
        return [len(entropy_sets), len(scores), intersections, sabna_intersections]

    return [
        #len(global_entropy_sets)/naive_entropies, len(global_scores)/naive_scores, global_intersections/naive_intersections,
        #sabna_global_intersections/naive_intersections,
        len(entropy_sets)/naive_entropies, len(scores)/naive_scores, intersections/naive_intersections,
        sabna_intersections/naive_intersections,
        naive_entropies, naive_scores, naive_intersections
        ]


def percentage_formatter(f: float) -> str:
    return f"{100*f:.2f}"


if __name__ == "__main__":
    dataset_names = ["asia", "sachs"]
    sample_sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    results = []
    for dataset_name in dataset_names:
        dag = DAG.from_bif(dataset_name)
        for n in sample_sizes:
            data = dag.sample(n, seed=n)
            # caps = per_var(data)
            results.append([dataset_name, n] + pruning_comparison(data, True))
    results = pd.DataFrame.from_records(results, columns=[
        "dataset", "samples", 
        #"C1, entropies", "C1, scores", "C1, cached intersections", "C1, SABNA intersections",
        "T3, entropies", "T3, scores", "T3, cached intersections", "T3, SABNA intersections",
        "naive entropies", "naive scores", "naive intersections",
        ])
    print(results.to_latex(index=False, float_format=percentage_formatter))
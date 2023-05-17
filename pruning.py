import pandas as pd
import numpy as np
from typing import *
from baynet import DAG

"""
Adding pruning rules (see arXiv:1707.06194) to benchmark queries
"""

def entropy(col: pd.Series) -> float:
    counts = col.value_counts().to_numpy()
    p = counts/col.size
    h = - np.sum(p * np.log2(p))
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

if __name__ == "__main__":
    sachs_dag = DAG.from_bif("sachs")
    data = sachs_dag.sample(1000)
    caps = per_var(data)
    print(f"Theorem 3 in-degree caps: {caps}")

    print(f"Corollary 1 global in-degree cap = {global_cap(data)}")

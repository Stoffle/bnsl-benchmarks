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

def pruning_comparison(data: pd.DataFrame) -> Tuple[int, int]:
    """
    Compute number of scores/entropies to be computed based on pruning available
    """
    C_global = global_cap(data)
    C_x_list = per_var(data)

    entropy_sets = set()
    scores = []

    global_entropy_sets = set()
    global_scores = []

    for x_idx in range(data.shape[1]-1):
        other_vars = list(range(data.shape[1]))
        other_vars.remove(x_idx)
        for set_size in range(min(C_global + 1, len(data.columns))):
            for Pi_x in combinations(other_vars, set_size):
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

    return (len(global_entropy_sets), len(global_scores), global_intersections, sabna_global_intersections, 
            len(entropy_sets), len(scores), intersections, sabna_intersections)



if __name__ == "__main__":
    results = []
    dag = DAG.from_bif("sachs")
    #dag = DAG.generate("forest fire", 15, seed=0).generate_discrete_parameters(seed=0)
    d = len(dag.nodes)
    for n in [1000, 10_000, 100_000, 1_000_000]:
        data = dag.sample(n)
        # caps = per_var(data)
        results.append(pruning_comparison(data))
    results = pd.DataFrame.from_records(results, columns=[
        "C1, entropies", "C1, scores", "C1, cached intersections", "C1, SABNA intersections",
        "T3, entropies", "T3, scores", "T3, cached intersections", "T3, SABNA intersections",
        ])
    results["EG, entropies"] = [995, 1607, 1982, 2046]
    results["EG, scores"] = [4106, 7872, 10666, 11253]
    results["x"] = [1000, 10_000, 100_000, 1_000_000]
    #results.plot(x="x", y=["C1, entropies", "C1, scores", "T3, entropies", "T3, scores", "EG, entropies", "EG, scores"])
    results.plot(x="x", y=["C1, entropies", "T3, entropies", "EG, entropies"])
    ax = plt.gca()
    ax.set_xscale('log')
    plt.xlabel("Sample size")
    #ax.set_yscale('log')
    plt.legend()
    plt.show()

    # Intersections comparison; subplot for each of SABNA, cached, and EG
    results["EG intersections"] = results["EG, entropies"]
    #results.plot(x="x", y=["C1, cached, intersections", "C1, SABNA intersections", "T3, cached, intersections", "T3, SABNA intersections", "EG, intersections"])
    #ax = plt.gca()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_xscale('log')
    ax1.plot(results["x"], results["C1, SABNA intersections"], label="Corollary 1")
    ax1.plot(results["x"], results["T3, SABNA intersections"], label="Theorem 3")
    ax1.set_title("SABNA intersections")
    ax1.set_ylim(bottom=0)
    ax1.legend()

    ax2.plot(results["x"], results["C1, cached intersections"], label="Corollary 1")
    ax2.plot(results["x"], results["T3, cached intersections"], label="Theorem 3")
    ax2.set_title("Cached entropy intersections")
    ax2.set_xlabel("Sample size")
    ax2.set_xscale('log')
    ax2.set_ylim(bottom=0)
    ax2.legend()

    ax3.plot(results["x"], results["EG intersections"], label="Theorem 2")
    ax3.set_title("Entropy graph intersections")
    ax3.set_xscale('log')
    ax3.set_ylim(bottom=0)
    ax3.legend()

    
    ax1.set_xscale('log')
    plt.xlabel("Sample size")
    ax.set_yscale('log')
    plt.legend()
    plt.show()


    # print(f"Number of variables: {d}")
    # print(f"Theorem 3 in-degree caps: {caps}")

    # print(f"Corollary 1 global in-degree cap = {global_cap(data)}")

    # sizes = pruning_comparison(data)
    # print(f"Naive: |Entropies| = {2**d}, |Scores| = {d*2**(d-1)}")
    # print(f"Global cap, caching: |Entropies| = {sizes[0]}, |Scores| = {sizes[1]}, intersections = {sizes[2]}, sabna intersections = {sizes[3]}")
    # print(f"Per-var cap, caching: |Entropies| = {sizes[4]}, |Scores| = {sizes[5]}, intersections = {sizes[6]}, sabna intersections = {sizes[7]}")

import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from tqdm import tqdm

import pybnsl
from baynet import DAG, metrics
from generate_data import generate_data


# def bootstrap_sl(data: pd.DataFrame, repeats: int, bootstrap_proportion: float) -> DAG:
#     # ideally bootstrapping would be done in rust, but could work for A at least
#     for _ in range(repeats):
#         bootstrap_data = data.copy() #TODO
#         ms = pybnsl.sl(bootstrap_data.values.astype(np.uint8), list(bootstrap_data.columns), True).strip('\"')
#         learnt = DAG.from_modelstring(ms)
#         a_parents = learnt.get_ancestors("A")["name"]

def run_sl(data: pd.DataFrame) -> DAG:
    ms = pybnsl.sl(data.values.astype(np.uint8), list(data.columns), True)
    ms = ms.strip('\"')
    return DAG.from_modelstring(ms)

def sl_on_dag(dag: DAG, n: int, seed: int = 0) -> DAG:
    data = generate_data(dag, n, seed)
    start_time = timeit.default_timer()
    learnt = run_sl(data)
    duration = timeit.default_timer() - start_time
    # print(f"SL took {duration}s")
    return learnt

def linear_dist_dag(d: int, max_parents: int, seed:int) -> DAG:
    assert max_parents <= d-1
    in_degree = max_parents
    amat = np.zeros((d, d))
    gen = np.random.default_rng(seed=seed)
    for var in range(d):
        if in_degree == 0: continue
        parents = gen.choice(np.arange(var+1, d), in_degree, replace=False)
        amat[var][parents] = 1
        in_degree -= 1
    amat = amat.T
    # node_names = [_name_node(i) for i in range(d)]
    return DAG.from_amat(amat).generate_discrete_parameters(max_levels=2, seed=seed)

if __name__=='__main__':
    true_indegrees = np.arange(1, 11)
    sample_sizes = np.logspace(1, 3, num=5, endpoint=True).astype(int)
    # sample_sizes = [500, 1000]
    for n in tqdm(sample_sizes):
        learnt_degree_means = []
        learnt_degree_upperq = []
        learnt_degree_lowerq = []
        for in_deg in tqdm(true_indegrees, leave=False):
            inner_learnt_degrees = []
            for seed in tqdm(range(10), leave=False):
                dag = linear_dist_dag(in_deg+5, in_deg, seed=seed)
                learnt = sl_on_dag(dag, n, seed=seed)
                correct_neighbours = [neighbour["name"] for neighbour in learnt.get_node(0).neighbors() if dag.are_neighbours(0, dag.get_node_index(neighbour["name"]))]
                inner_learnt_degrees.append(len(correct_neighbours))
            learnt_degree_means.append(np.median(inner_learnt_degrees))
            learnt_degree_upperq.append(np.quantile(inner_learnt_degrees, 0.75))
            learnt_degree_lowerq.append(np.quantile(inner_learnt_degrees, 0.25))
        plt.plot(true_indegrees, learnt_degree_means, label=f"n = {n}")
        plt.fill_between(true_indegrees, learnt_degree_lowerq, learnt_degree_upperq, alpha=0.3)
    plt.xlabel("Ground truth DAG, node A in-degree")
    plt.ylabel("Learnt DAG, node A degree")
    plt.legend()
    plt.show()
    # dag = linear_dist_dag(15, 6, seed=0)
    # dag.plot()
    # learnt = sl_on_dag(dag, 400, seed=0)
    # dag.compare(learnt).plot()


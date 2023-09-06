import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from tqdm import tqdm

import pybnsl
from baynet import DAG, metrics
from baynet.structure import _name_node
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
        print(parents)
        amat[var][parents] = 1
        in_degree -= 1
    amat = amat.T
    print(amat)
    # node_names = [_name_node(i) for i in range(d)]
    return DAG.from_amat(amat)

if __name__=='__main__':
    # d = DAG.from_bif("sachs")
    dags = []
    in_degrees = []
    sample_sizes = [2000]
    learnt_neighbours = []
    # for n in tqdm(sample_sizes):
    #     for d in tqdm(range(10, 15)):
    #         for seed in tqdm(range(15), leave=False):
    #             dag = DAG.generate("forest fire", 13, fw_prob=0.5, bw_factor=0.5, seed=seed).generate_discrete_parameters(seed=seed)
    #             dags.append(dag)
    #             in_degrees.append(dag.get_node(0).indegree())
    #             learnt = sl_on_dag(dag, n, seed)
    #             learnt_neighbours.append(len(learnt.get_node(0).neighbors()))
    # d = DAG.generate("ide_cozman", 15, seed=seed).generate_discrete_parameters(seed=seed)
    # print(f"nodes = {len(dag.nodes)}")
    # print(f"edges = {len(dag.edges)}")
    # print(f"node A in-degree = {dag.indegree()[0]}")
    # learnt = sl_on_dag(d, 3000, seed)
    # d.compare(learnt).plot()
    # print(f"v-structure recall = {metrics.v_recall(d, learnt)}")
    # dag.plot()
    # print(in_degrees)
    # print(sorted(in_degrees))
    # plt.scatter(in_degrees, learnt_neighbours)
    # plt.show()
    dag = linear_dist_dag(15, 6, seed=2)
    dag.plot()

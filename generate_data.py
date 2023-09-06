from baynet import DAG
from pandas import DataFrame
from pathlib import Path
from pruning import per_var, entropy

def df_cat_to_int(df: DataFrame) -> DataFrame:
    out_df = DataFrame()
    for col in df.columns:
        out_df[col] = df[col].cat.codes
    return out_df

def generate_data(dag: DAG, n:int, seed=0) -> DataFrame:
    # n_str = f"{n/1000:.0f}k"
    df = df_cat_to_int(dag.sample(n, seed=seed))
    return df

def write_ordered_columns(data:DataFrame, filename_prefix: str) -> None:
    cols = data.columns.tolist()
    cols.sort(key=lambda x: -entropy(data[x]), reverse=True) # ascending entropy
    df = data[cols]
    #df.to_csv(f"{filename_prefix}_sorted.csv", index=False)
    df.to_csv(f"{filename_prefix}_reversed.csv", index=False)


def write_data(data:DataFrame, filename_prefix: str) -> None:
    data.to_csv(f"{filename_prefix}.csv", index=False)
    data.transpose().to_csv(f"{filename_prefix}_T.txt", sep=" ", header=False, index=False)

if __name__ == "__main__":
    sachs_dag = DAG.from_bif("sachs")
    child_dag = DAG.from_bif("child")
    asia_dag = DAG.from_bif("asia")
    ff15_dag = DAG.generate("forest fire", 15, seed=0).generate_discrete_parameters(seed=0)
    print(f"ff15 modelstring: {ff15_dag.get_modelstring()}")
    p = Path().resolve() / "data"
    p.mkdir(exist_ok=True)
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        if n >= 1000:
            n_dir = p / f"{n/1000:.0f}k"
        else:
            n_dir = p / f"{n}"
        n_dir.mkdir(exist_ok=True)
        # write_data(generate_data(sachs_dag, n), f"{n_dir}/sachs")
        # write_data(generate_data(child_dag, n), f"{n_dir}/child")
        # write_data(generate_data(ff15_dag, n), f"{n_dir}/forest_fire_15")
        #write_data(generate_data(asia_dag, n), f"{n_dir}/asia")
        write_ordered_columns(generate_data(sachs_dag, n), f"{n_dir}/sachs")
        write_ordered_columns(generate_data(child_dag, n), f"{n_dir}/child")
        write_ordered_columns(generate_data(ff15_dag, n), f"{n_dir}/forest_fire_15")
        write_ordered_columns(generate_data(asia_dag, n), f"{n_dir}/asia")
        


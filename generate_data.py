from baynet import DAG
from pandas import DataFrame
from pathlib import Path

def df_cat_to_int(df: DataFrame) -> DataFrame:
    out_df = DataFrame()
    for col in df.columns:
        out_df[col] = df[col].cat.codes
    return out_df

def generate_data(dag: DAG, n:int, filename_prefix: str) -> None:
    # n_str = f"{n/1000:.0f}k"
    sachs_1k_df = df_cat_to_int(dag.sample(n, seed=0))
    sachs_1k_df.to_csv(f"{filename_prefix}.csv", index=False)
    sachs_1k_df.transpose().to_csv(f"{filename_prefix}_T.txt", sep=" ", header=False, index=False)

if __name__ == "__main__":
    sachs_dag = DAG.from_bif("sachs")
    ff15_dag = DAG.generate("forest fire", 15, seed=0).generate_discrete_parameters(seed=0)
    print(f"ff15 modelstring: {ff15_dag.get_modelstring()}")
    p = Path().resolve() / "data"
    p.mkdir(exist_ok=True)
    for n in [1_000, 10_000, 100_000]:
        n_dir = p / f"{n/1000:.0f}k"
        n_dir.mkdir(exist_ok=False)
        generate_data(sachs_dag, n, f"{n_dir}/sachs")

        generate_data(ff15_dag, n, f"{n_dir}/forest_fire_15")

# %%
import polars as pl
from pathlib import Path
from mpipi_lammps_gen.generate_lammps_files import parse_cif, ProteinData
from dataclasses import asdict

# %%
input = Path("./alpha_fold_query_experiment_rg_opt.parquet")
df = (
    pl.scan_parquet(input)
    .select("alpha_fold_db_id", "cif_text", "pae_matrix")
    .collect()
)

prot_data_list = []

for cif_text, pae in zip(df["cif_text"], df["pae_matrix"]):
    prot_data = parse_cif(cif_text, method="Ca")
    prot_data.pae = pae.to_list()
    prot_data.residue_positions = None
    prot_data.sequence_three_letter = None
    prot_data_list.append(ProteinData(**asdict(prot_data)))


series = pl.Series("prot_data", prot_data_list)

df.select(pl.col("alpha_fold_db_id").alias("key")).insert_column(
    -1, series
).write_parquet("alpha_fold_db_rgopt.parquet")


df.select(pl.col("alpha_fold_db_id").alias("key")).insert_column(-1, series).write_ipc(
    "alpha_fold_db_rgopt.arrow"
)

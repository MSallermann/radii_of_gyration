import polars as pl
from pathlib import Path

from mpipi_lammps_gen.generate_lammps_files import parse_cif, ProteinData
from dataclasses import asdict

from pl_schema_from_dataclass import dataclass_to_polars_struct

df_total = None

output = Path("alpha_fold_query_exp.parquet")
if output.exists():
    output.unlink()


struct = dataclass_to_polars_struct(ProteinData)


for p in Path(".").glob("alpha_fold_query*.parquet"):
    df = (
        pl.read_parquet(p, low_memory=True, use_pyarrow=True)
        .drop("pdb_text")
        .with_columns(
            pl.col("cif_text")
            .map_elements(
                lambda cif_text: asdict(parse_cif(cif_text)), return_dtype=struct
            )
            .alias("prot_data")
        )
        .drop("cif_text")
    )
    if df_total is None:
        df_total = df

    df_total = pl.concat((df_total, df), how="vertical_relaxed")

df_total.write_parquet(output, compression="zstd")

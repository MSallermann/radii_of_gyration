import logging
from collections.abc import Iterable
from typing import Any
from pathlib import Path

import polars as pl

from mpipi_lammps_gen import alpha_fold_query

logger = logging.getLogger(__name__)


def find_next_free_file(file: Path, n_max: int = 1000) -> Path:
    test_name = Path()

    for i in range(1, n_max + 1):
        test_name = file.with_name(file.with_suffix("").name + f"_{i}").with_suffix(
            file.suffix
        )
        if not test_name.exists():
            return test_name

    msg = (
        f"Could not find free file within {n_max} attempts. Last attempt `{test_name}`"
    )
    raise Exception(msg)


def save(
    query_results: Iterable[Any],
    output_path: Path,
):
    if query_results == 0:
        return

    # Convert the new results to a dataframe
    df_out_new = pl.DataFrame(query_results)

    # If no query results, just exit
    if len(df_out_new) == 0:
        return

    # If something already exists at the output path we try to read it in as a dataframe
    if output_path.exists():
        logger.info(f"`{output_path}` exists. Trying to read as parquet.")
        try:
            df_out_old = pl.read_parquet(output_path)
            logger.info(f"... success! There are {len(df_out_old)} rows in the df")
        except Exception as e:
            df_out_old = None
            logger.exception("... failed!", stack_info=True)
    else:
        df_out_old = None

    if df_out_old is None:
        df_out = df_out_new
    else:
        logger.info("Trying to concat dataframes.")
        try:
            # In case we could read it in as a dataframe, we try to concatenate the two frames
            df_out = pl.concat((df_out_old, df_out_new), how="vertical_relaxed")
            logger.info("... success!")
        except Exception as e:
            # If the concatenation fails, we change the output path so that we do not lose the old data
            df_out = df_out_new
            output_path = find_next_free_file(output_path)
            logger.exception(
                "Cannot concatenate data frames. Changing output path to not overwrite data"
            )

    logger.info(f"Saving {len(df_out)} rows to `{output_path}`")
    df_out.write_parquet(output_path)


def main(accessions: Iterable[str], output_path: Path, n_flush: int = 100, **kwargs):
    query_results = []

    accessions_list = list(accessions)

    for idx, a in enumerate(accessions_list):
        logger.info(f"Queried {a} [{idx} / {len(accessions_list)}]")

        query_result = alpha_fold_query.query_alphafold(a, **kwargs)

        if query_result is not None:
            query_results.extend(query_result)

        if idx != 0 and idx % n_flush == 0:
            save(
                query_results=query_results,
                output_path=find_next_free_file(output_path),
            )
            query_results = []

    save(query_results=query_results, output_path=find_next_free_file(output_path))


if __name__ == "__main__":
    from logging import FileHandler
    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(), FileHandler("query_alpha_fold.log")],
    )

    # ids_to_query = pl.read_parquet("./data_idrs.parquet")["uniprot_id"]
    # ids_to_query = pl.read_parquet(
    #     "/home/sie/Biocondensates/exp_data/rg_experimental.parquet"
    # )["uniprot_accession"].unique()

    ids_to_query = (
        pl.read_csv(
            "/home/sie/Biocondensates/prepare_samples_for_rgopt/db_rgopt_test_all_sasbdb_pae.csv"
        )
        .select(pl.col("af_id").str.split("-").list.slice(1, 1).list[0])["af_id"]
        .to_list()
    )

    output_path = Path("./alpha_fold_query_experiment_rg_opt.parquet")

    print(ids_to_query[:3])

    ids_to_query = set(ids_to_query)

    # .difference(
    #     pl.read_parquet(output_path)["accession"]
    # )

    main(accessions=ids_to_query, output_path=output_path, n_flush=50, get_pdb=False)

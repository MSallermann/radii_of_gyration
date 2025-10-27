from mpipi_lammps_gen.sasbdb_query import query_sasbdb, SASBDBQueryResult
import json
import polars as pl
import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


def save(
    query_results: Iterable[SASBDBQueryResult],
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
        try:
            df_out_old = pl.read_parquet(output_path)
        except Exception as e:
            df_out_old = None
    else:
        df_out_old = None

    if df_out_old is None:
        df_out = df_out_new
    else:
        try:
            # In case we could read it in as a dataframe, we try to concatenate the two frames
            df_out = pl.concat((df_out_old, df_out_new))
        except Exception as e:
            # If the concatenation fails, we change the output path so that we do not lose the old data
            df_out = df_out_new
            logger.exception(
                "Cannot concatenate data frames. Changing output path to not overwrite data"
            )
            output_path = output_path.with_name(
                output_path.with_suffix("").name + "_new"
            ).with_suffix(output_path.suffix)

    logger.info(f"Saving to {output_path}")
    df_out.write_parquet(output_path)


def main(accessions: Iterable[str], output_path: Path, n_flush: int = 100):
    query_results = []

    accessions_list = list(accessions)

    for idx, a in enumerate(accessions_list):
        logger.info(f"Queried {a}, [{idx} / {len(accessions_list)}]")
        query_results.extend(query_sasbdb(a))

        if idx % n_flush == 0:
            save(query_results=query_results, output_path=output_path)
            query_results = []

    save(query_results=query_results, output_path=output_path)


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

    path = "/home/sie/Biocondensates/radii_of_gyration/prepare_input/uniprotkb_database_sasbdb.json"
    with open(path, "r") as f:
        data = json.load(f)

    accessions = [d["primaryAccession"] for d in data["results"]]

    main(
        accessions=accessions,
        output_path=Path("sasbdb_query_results.parquet"),
        n_flush=50,
    )

from mpipi_lammps_gen.sasbdb_query import query_sasbdb
from run_alpha_fold_query import save

import json
import polars as pl
import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


def main(accessions: Iterable[str], output_path: Path, n_flush: int = 100):
    query_results = []

    accessions_list = list(accessions)

    for idx, a in enumerate(accessions_list):
        logger.info(f"Queried {a} [{idx} / {len(accessions_list)}]")
        query_results.extend(query_sasbdb(a))

        if idx != 0 and idx % n_flush == 0:
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
        handlers=[RichHandler(), FileHandler("query_sasbdb.log")],
    )

    path = "/home/sie/Biocondensates/radii_of_gyration/prepare_input/uniprotkb_database_sasbdb.json"
    with open(path, "r") as f:
        data = json.load(f)

    accessions = [d["primaryAccession"] for d in data["results"]]

    main(
        accessions=accessions,
        output_path=Path("sasbdb_query_results.parquet"),
        n_flush=100,
    )

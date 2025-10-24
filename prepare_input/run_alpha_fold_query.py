import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from mpipi_lammps_gen import alpha_fold_query

logger = logging.getLogger(__name__)


def main(uniprot_ids_in: Iterable[str], output_path: Path):
    uniprot_ids_in = list(uniprot_ids_in)

    if output_path.exists():
        df_out = pl.read_parquet(output_path)
        seen = set(df_out["accession"])
    else:
        df_out = None
        seen = set()

    # Remove all the ids we have seen already
    uniprot_ids_in_unique = set(uniprot_ids_in)
    ids_to_query = uniprot_ids_in_unique.difference(seen)

    logger.info(
        f"Of {len(uniprot_ids_in)} input ids, {len(uniprot_ids_in_unique)} are unique. Out of these {len(seen)} are already in the output file '{output_path}'."
    )
    logger.info(f"Therefore I will query {len(ids_to_query)} ids.")

    query_result_generator = alpha_fold_query.query_alphafold_bulk(
        list(ids_to_query),
        get_cif=True,
        get_pae_matrix=True,
        get_pdb=False,
        timeout=30,
        retries=4,
        backoff_time=5,
    )

    query_results = []

    idx = 0
    try:
        for query_result in query_result_generator:
            idx += 1

            logger.info(
                f"Queried {query_result.accession} [{idx} / {len(ids_to_query)}]"
            )

            if query_result.http_status != 200:
                continue

            query_results.append(query_result)

    except BaseException as e:
        output_path = output_path.with_name("saved_after_exc").with_suffix(".parquet")

        logger.exception(
            f"Encountered exception {e}. Will try to save data to {output_path}",
            stack_info=True,
            stacklevel=1,
        )

        raise e
    finally:
        df_out_new = pl.DataFrame(query_results)

        try:
            df_out = df_out_new if df_out is None else pl.concat((df_out, df_out_new))
            logger.info(f"Saving to {output_path}")
            df_out.write_parquet(output_path)
        except Exception as e:
            logger.exception(
                f"Exception when trying to concat new results to old results. Saving new results in {output_path.with_suffix("new.parquet")} "
            )
            df_out_new.write_parquet(output_path.with_suffix(".new.parquet"))


if __name__ == "__main__":
    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    # ids_to_query = pl.read_parquet("./data_idrs.parquet").sample(8000)["uniprot_id"]
    # ids_to_query = ["P37840", "P35637"]
    # ids_to_query = ["P37840", "Q01718", "Q5A5Q6", "P06971", "P13468"]
    # ids_to_query = ["Q9ULK0"]

    ids_to_query = pl.read_parquet(
        "/home/moritz/Biocondensates/predictor/lammps_input_generator_for_mpipi/rg_workflow/samples.parquet"
    )["accession"]

    output_path = Path("./samples_with_pae.parquet")

    main(uniprot_ids_in=ids_to_query, output_path=output_path)

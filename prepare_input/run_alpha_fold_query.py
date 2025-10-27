import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from mpipi_lammps_gen import alpha_fold_query

logger = logging.getLogger(__name__)


def save(
    query_results: Iterable[alpha_fold_query.AlphaFoldQueryResult],
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


def main(uniprot_ids_in: Iterable[str], output_path: Path, n_flush: int = 100):
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

            if idx % n_flush == 0:
                logger.info("Flushing results")
                save(query_results=query_results, output_path=output_path)
                query_results.clear()

    except BaseException as e:
        output_path = output_path.with_name("saved_after_exc").with_suffix(".parquet")

        logger.exception(
            f"Encountered exception {e}. Will try to save data to {output_path}",
            stack_info=True,
            stacklevel=1,
        )

        raise e
    finally:
        logger.info("Run finished. Saving")
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

    ids_to_query = pl.read_parquet("./data_idrs.parquet")["uniprot_id"]

    output_path = Path("./samples_with_pae.parquet")

    main(uniprot_ids_in=ids_to_query, output_path=output_path, n_flush=200)

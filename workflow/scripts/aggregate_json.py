import json
from pathlib import Path

import pandas as pd


def main(
    input_paths: list[Path | str],
    output_path: Path | str,
    added_columns: dict | None = None,
    ignore_keys: list[str] | None = None,
):
    output_path = Path(output_path)
    data = []

    for idx, _ip in enumerate(input_paths):
        ip = Path(_ip)

        if ip.exists():
            with ip.open("r") as f:
                record = json.load(f)
            record["file"] = str(ip)
        else:
            record = {"file": None}

        if added_columns is not None:
            for k in added_columns:
                item = added_columns[k][idx]
                record[k] = item

        if ignore_keys is not None:
            for k in ignore_keys:
                record.pop(k)

        data.append(record)

    df = pd.DataFrame.from_records(data)

    with output_path.open("w") as f:
        if output_path.suffix == ".csv":
            df.to_csv(f)
        elif output_path.suffix == ".json":
            df.to_json(f, indent=4)
        else:
            msg = f"{output_path.suffix} is not a valid file extension"
            raise Exception(msg)


if __name__ == "__main__":
    from snakemake.script import snakemake

    main(
        snakemake.input,
        snakemake.output[0],
        snakemake.params.get("add_columns", None),
        snakemake.params.get("ignore_columns", None),
    )

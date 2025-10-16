import json
from pathlib import Path

import pandas as pd
from snakemake.script import snakemake


def main(
    input_paths: list[Path | str],
    output_path: Path | str,
    added_columns: dict | None = None,
    ignore_keys: list[str] | None = None,
):
    output_path = Path(output_path)
    data = {}

    for idx, _ip in enumerate(input_paths):
        ip = Path(_ip)
        with ip.open("r") as f:
            res = json.load(f)

        res["file"] = str(ip)

        if added_columns is not None:
            for k in added_columns:
                item = added_columns[k][idx]
                if k in data:
                    data[k].append(item)
                else:
                    data[k] = [item]

        for k, v in res.items():
            if ignore_keys is not None and k in ignore_keys:
                continue
            if k in data:
                data[k].append(v)
            else:
                data[k] = [v]

    df = pd.DataFrame(data)
    with output_path.open("w") as f:
        if output_path.suffix == ".csv":
            df.to_csv(f)
        elif output_path.suffix == ".json":
            df.to_json(f, indent=4)
        else:
            msg = f"{output_path.suffix} is not a valid file extension"
            raise Exception(msg)


if __name__ == "__main__":
    main(
        snakemake.input,
        snakemake.output[0],
        snakemake.params.get("add_columns", None),
        snakemake.params.get("ignore_columns", None),
    )

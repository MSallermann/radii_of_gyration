import numpy as np
from pathlib import Path
import json


def average_radius_of_gyration(input_file: Path, n_skip: int, out_json: Path | str):
    out_json = Path(out_json)

    rg = np.atleast_1d(np.loadtxt(input_file))[n_skip:]

    rg_mean = np.mean(rg)
    rg_std = np.std(rg)
    n_samples = len(rg)

    with out_json.open("w") as f:
        json.dump({"rg_mean": rg_mean, "rg_std": rg_std, "n_samples": n_samples}, f)


if __name__ == "__main__":
    try:
        from snakemake.script import snakemake
    except ImportError:
        ...

    average_radius_of_gyration(
        input_file=snakemake.input[0],
        n_skip=snakemake.params["n_skip"],
        out_json=snakemake.output["rg_json"],
    )

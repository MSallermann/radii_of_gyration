import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from lammps_logfile import File


def moving_avg(x):
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def analyse_lammps_log(log_file: Path, n_skip: int, output_json: Path):
    # Read the log file into a DataFrame
    # This example uses the 'crack_log.lammps' file found in 'examples/logfiles/'
    lf = File(log_file)

    cols = ["PotEng", "Press", "KinEng", "Temp"]

    avg_data = {}

    steps = lf.get("Step")

    for col in cols:
        fig, ax = plt.subplots()

        col_vals = lf.get(col)

        avg_data[col] = np.mean(col_vals[n_skip:])

        ax.plot(
            steps[:n_skip],
            col_vals[:n_skip],
            color="grey",
            marker=".",
            ls="None",
            label="skipped",
        )

        ax.plot(
            steps[n_skip:],
            col_vals[n_skip:],
            color="blue",
            marker=".",
            ls="None",
            label="computed",
        )

        ax.plot(steps, moving_avg(col_vals), label="running mean", color="black")

        ax.axhline(avg_data[col], ls="--", color="black", label="total mean")

        ax.legend()
        ax.set_xlabel("Step")
        ax.set_ylabel(col)

        fig.savefig(output_json.parent / f"{col}.png", dpi=300)

    with open(output_json, "w") as f:
        json.dump(avg_data, f, indent=4)


if __name__ == "__main__":
    try:
        from snakemake.script import snakemake
    except ImportError:
        ...

    analyse_lammps_log(
        log_file=Path(snakemake.input[0]),
        n_skip=snakemake.params["n_skip"],
        output_json=Path(snakemake.output[0]),
    )

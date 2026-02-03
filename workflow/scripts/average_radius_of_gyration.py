import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt


def moving_avg(x):
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def average_radius_of_gyration(input_file: Path, n_skip: int, out_json: Path | str):
    out_json = Path(out_json)

    rg_full = np.atleast_1d(np.loadtxt(input_file))

    rg = rg_full[n_skip:]

    rg_mean = np.mean(rg)
    rg_std = np.std(rg)
    n_samples = len(rg)
    rg_err = rg_std / np.sqrt(n_samples)

    running_avg = moving_avg(rg)

    # the last 20% of the computed values
    running_avg_tail = running_avg[int(n_samples * 0.8) :]
    tail_deviation = (np.mean(running_avg_tail) - rg_mean) / rg_mean

    with out_json.open("w") as f:
        json.dump(
            {
                "rg_mean": rg_mean,
                "rg_std": rg_std,
                "n_samples": n_samples,
                "rg_err": rg_err,
                "n_skip": n_skip,
                "tail_deviation": tail_deviation,
            },
            f,
        )

    fig, ax = plt.subplots()

    steps = np.arange(1, len(rg_full) + 1)

    ax.plot(
        steps[:n_skip],
        rg_full[:n_skip],
        color="grey",
        marker=".",
        ls="None",
        label="skipped",
    )

    ax.plot(
        steps[n_skip:],
        rg_full[n_skip:],
        color="blue",
        marker=".",
        ls="None",
        label="computed",
    )

    ax.fill_between(
        steps[n_skip + int(0.8 * n_samples) :],
        y1=rg_mean + tail_deviation * rg_mean,
        y2=rg_mean,
        alpha=0.2,
    )

    ax.plot(steps[n_skip:], running_avg, label="running mean", color="black")
    ax.axhline(float(rg_mean), ls="--", color="black", label="total mean")
    ax.legend()
    ax.set_xlabel("steps")
    ax.set_ylabel("Rg [Angs.]")

    ax.set_title(
        f"mean={rg_mean:.2e}, err={rg_err:.2e}, std={rg_std:.2e}\n"
        f"n_samples={n_samples:.0f}, n_skip={n_skip:.0f}"
    )

    fig.tight_layout()

    fig.savefig(out_json.parent / "plot.png", dpi=300)


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

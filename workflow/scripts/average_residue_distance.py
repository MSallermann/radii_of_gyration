import numpy as np
import MDAnalysis as mda
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import numpy.typing as npt


def get_r2(y_data: npt.ArrayLike, y_fit: npt.ArrayLike) -> float:
    y_mean = np.mean(y_data)

    ssres = np.sum((y_data - y_fit) ** 2)
    sstot = np.sum((y_data - y_mean) ** 2)

    return 1.0 - ssres / sstot


def pair_distance(
    output_csv: Path,
    output_json: Path,
    lammps_data_file: Path,
    lammps_traj_file: Path,
    n_skip: int,
):
    u = mda.Universe(
        lammps_data_file, lammps_traj_file, format="LAMMPSDUMP"
    )  # adjust filenames
    atoms = u.atoms

    N = len(atoms)
    max_sep = N - 1
    sum_dist = np.zeros(max_sep + 1)
    count = np.zeros(max_sep + 1, dtype=int)

    for ts in u.trajectory[n_skip:]:
        pos = atoms.positions  # (N,3) in Å by default in MDAnalysis for LAMMPS dumps
        for sep in range(1, max_sep + 1):
            d = pos[sep:] - pos[:-sep]  # vectors between i and i+sep
            r = np.linalg.norm(d, axis=1)  # distances
            sum_dist[sep] += r.sum()
            count[sep] += r.size

    R_sep = sum_dist / count  # <R_ij> as a function of |i-j|

    df = pl.DataFrame({"sequence_dist": range(max_sep + 1), "r_ij_avg": R_sep})
    df = df.fill_nan(0.0)
    df.write_csv(output_csv)

    # Fit
    def _fit_fun(ij: float, b: float, nu: float) -> float:
        return b * ij**nu

    fit_fun = np.vectorize(_fit_fun)

    # fit with a fixed kuhn distance
    b_fixed = 5.5
    popt_fixed_kuhn, pcov_fixed_kuhn = curve_fit(
        lambda ij, nu: fit_fun(ij, b=b_fixed, nu=nu),
        xdata=df["sequence_dist"],
        ydata=df["r_ij_avg"],
        p0=[0.5],
    )
    nu_fixed = popt_fixed_kuhn[0]

    # fit with a variable kuhn distance
    popt_variable_kuhn, pcov_variable_kuhn = curve_fit(
        fit_fun,
        xdata=df["sequence_dist"],
        ydata=df["r_ij_avg"],
        p0=[5.5, 0.5],
    )
    b_variable = popt_variable_kuhn[0]
    nu_variable = popt_variable_kuhn[1]

    y_fit_variable = fit_fun(df["sequence_dist"], b=b_variable, nu=nu_variable)
    r2_variable = get_r2(y_data=df["r_ij_avg"].to_numpy(), y_fit=y_fit_variable)

    y_fit_fixed = fit_fun(df["sequence_dist"], b=b_fixed, nu=nu_fixed)
    r2_fixed = get_r2(y_data=df["r_ij_avg"].to_numpy(), y_fit=y_fit_fixed)

    with output_json.open("w") as f:
        json.dump(
            {
                "b_fixed": b_fixed,
                "nu_fixed": nu_fixed,
                "b_variable": b_variable,
                "nu_variable": nu_variable,
                "r2_fixed": r2_fixed,
                "r2_variable": r2_variable,
            },
            f,
        )

    ### Make a quick plot
    fig, ax = plt.subplots()
    ax.plot(
        df["sequence_dist"], df["r_ij_avg"], color="black", marker=".", label="data"
    )

    ax.plot(
        df["sequence_dist"],
        y_fit_variable,
        color="blue",
        label=f"Kuhn fit, $R^2$={r2_variable:.2f}, (b={b_variable:.2f}, nu={nu_variable:.2f})",
    )

    ax.plot(
        df["sequence_dist"],
        y_fit_fixed,
        color="red",
        label=f"Kuhn fit, b={b_fixed:.2f} (fix), $R^2$={r2_fixed:.2f}, (nu={nu_fixed:.2f})",
    )

    ax.set_xlabel("|i-j|")
    ax.set_ylabel("|Rij| [Angs.]")
    ax.legend()
    fig.savefig(output_csv.parent / "plot.png", dpi=300)


if __name__ == "__main__":
    try:
        from snakemake.script import snakemake
    except ImportError:
        ...

    pair_distance(
        output_csv=Path(snakemake.output["output_csv"]),
        output_json=Path(snakemake.output["output_json"]),
        lammps_data_file=Path(snakemake.input["lammps_data_file"]),
        lammps_traj_file=Path(snakemake.input["lammps_traj_file"]),
        n_skip=snakemake.params["n_skip"],
    )

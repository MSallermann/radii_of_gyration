from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer
from scipy.optimize import curve_fit

from mpipi_lammps_gen.globular_domains import GlobularDomain, protein_topology
from mpipi_lammps_gen.shortest_path_graph import (
    PathProperties,
    build_shortest_path_graph,
)
from mpipi_lammps_gen.shortest_path_graph_cached import (
    build_path_query_cache,
    get_path_properties_cached,
)


import logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Pair-distance analysis for LAMMPS trajectories.",
)


def get_r2(y_data: np.ndarray, y_fit: np.ndarray) -> float:
    y_data_arr = np.asarray(y_data, dtype=float)
    y_fit_arr = np.asarray(y_fit, dtype=float)

    y_mean = np.mean(y_data_arr)
    ssres = np.sum((y_data_arr - y_fit_arr) ** 2)
    sstot = np.sum((y_data_arr - y_mean) ** 2)

    return 1.0 - ssres / sstot


def _loop_correction(
    loop: tuple[int, int] | None, kuhn_length: float, bond_length: float
) -> float:
    """
    Preserve the old loop correction structure, but apply it to the new
    start_loop / end_loop representation.

    loop = (idx_within_loop, n_loop_residues)
    """
    if loop is None:
        return 0.0

    idx_in_loop, loop_length = loop
    n_segments = loop_length + 1
    idx_segment = idx_in_loop + 1

    n_segments_kuhn = n_segments * bond_length / kuhn_length
    idx_segment_kuhn = idx_segment * bond_length / kuhn_length

    return (
        kuhn_length**2
        * abs(idx_segment_kuhn)
        * abs((n_segments_kuhn - idx_segment_kuhn) / n_segments_kuhn)
    )


def path_prop_to_r2(
    path_prop: PathProperties, kuhn_length: float, bond_length: float, nu: float = 1.0
) -> float:
    """
    Convert path properties into an expected <R_ij^2>.
    """
    expected_r2 = (
        kuhn_length
        * bond_length
        * (path_prop.random_walk_contour_length / bond_length) ** nu
    )

    for dist in path_prop.fixed_distances:
        expected_r2 += dist**2

    expected_r2 += _loop_correction(
        path_prop.start_loop, kuhn_length=kuhn_length, bond_length=bond_length
    )
    expected_r2 += _loop_correction(
        path_prop.end_loop, kuhn_length=kuhn_length, bond_length=bond_length
    )

    return expected_r2


def pair_distance(
    output_csv: Path,
    output_json: Path,
    lammps_data_file: Path,
    lammps_traj_file: Path,
    n_skip: int,
    domains: Iterable[GlobularDomain] | None = None,
    kuhn_fixed: float = 5.5,
    bond_length: float = 3.81,
    max_sequence_dist_fit: int | None = None,
    log_progress: bool = False,
    n_pairs_log: int = 10000,
) -> None:
    u = mda.Universe(lammps_data_file, lammps_traj_file, format="LAMMPSDUMP")
    atoms = u.atoms

    if len(u.trajectory) == 0:
        raise ValueError(f"Trajectory {lammps_traj_file} contains no frames.")

    # Use frame 0 coordinates as the reference residue positions.
    u.trajectory[0]
    residue_positions = np.asarray(atoms.positions, dtype=float).copy()

    n_residues = len(atoms)
    domains_list = list(domains) if domains is not None else []

    ######################################
    # Topology
    ######################################

    logger.info("... building topology")
    topology = protein_topology(n_residues=n_residues, domains=domains_list)

    logger.info("... building shortest path graph")
    shortest_path_graph = build_shortest_path_graph(
        topology,
        residue_positions,
        bond_length=bond_length,
        segment_length=None,
    )

    logger.info("... building path query cache")
    cache = build_path_query_cache(
        topology=topology,
        residue_positions=residue_positions,
        bond_length=bond_length,
        shortest_path_graph=shortest_path_graph,
    )

    logger.info("... recording per residue path properties")
    path_properties_per_residue: dict[tuple[int, int], PathProperties] = {}

    n_pairs = int(n_residues * (n_residues - 1) / 2)
    n_pairs_done = 0
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            path_properties_per_residue[(i, j)] = get_path_properties_cached(
                cache=cache,
                i1=i,
                i2=j,
            )
            n_pairs_done += 1

            if log_progress and n_pairs_done % n_pairs_log == 0:
                logger.info(
                    f"{n_pairs_done} of {n_pairs} done ({n_pairs_done / n_pairs * 100:.1f}%)"
                )

    logger.info("done")

    ######################################
    # Ideal theta curve
    ######################################
    max_sep = n_residues - 1
    separations = list(range(1, max_sep + 1))
    rij2_ideal: list[float] = []

    for sep in separations:
        r2_sum = 0.0
        count = 0

        for i in range(n_residues - sep):
            j = i + sep
            path_prop = path_properties_per_residue[(i, j)]
            r2_sum += path_prop_to_r2(
                path_prop, kuhn_length=kuhn_fixed, bond_length=bond_length
            )
            count += 1

        rij2_ideal.append(r2_sum / count)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(
        separations,
        rij2_ideal,
        color="blue",
        lw=3,
        label="MDP corrections",
    )
    ax.plot(
        separations,
        kuhn_fixed * bond_length * np.array(separations),
        color="grey",
        label="IDP behaviour",
    )

    np.savetxt(output_csv.parent / "rij2_ideal.txt", rij2_ideal)
    np.savetxt(output_csv.parent / "separations.txt", separations)

    ax.legend()
    ax.set_ylabel(r"$<R_{ij}^2>$")
    ax.set_xlabel("|i-j|")
    fig.savefig(output_csv.parent / "ideal_theta.png", dpi=300)
    plt.close(fig)

    ######################################
    # Trajectory averaging
    ######################################
    sum_mean = np.zeros(max_sep + 1)
    sum_mean_sq = np.zeros(max_sep + 1)
    n_frames = 0

    for _ts in u.trajectory[n_skip:]:
        pos = atoms.positions
        frame_mean = np.zeros(max_sep + 1)

        for sep in range(1, max_sep + 1):
            r2 = np.linalg.norm(pos[:-sep] - pos[sep:], axis=1) ** 2
            frame_mean[sep] = r2.mean()

        sum_mean += frame_mean
        sum_mean_sq += frame_mean**2
        n_frames += 1

    if n_frames == 0:
        raise ValueError(
            f"No trajectory frames available after skipping n_skip={n_skip} frames."
        )

    mean_r2 = sum_mean / n_frames
    r_sep = np.sqrt(mean_r2)

    if n_frames > 1:
        var_r2 = (sum_mean_sq - n_frames * mean_r2**2) / (n_frames - 1)
        var_r2 = np.clip(var_r2, 0.0, None)
        stderr_mean_r2 = np.sqrt(var_r2 / n_frames)

        # Propagate uncertainty through r = sqrt(mean_r2)
        r_sep_err = np.zeros_like(r_sep)
        mask = r_sep > 0
        r_sep_err[mask] = stderr_mean_r2[mask] / (2.0 * r_sep[mask])
    else:
        r_sep_err = np.zeros_like(r_sep)

    df = pl.DataFrame(
        {
            "sequence_dist": range(max_sep + 1),
            "r_ij_avg": r_sep,
            "r_ij_err": r_sep_err,
            "r_ij2_avg": mean_r2,
            "r_ij2_err": stderr_mean_r2,
        }
    ).fill_nan(0.0)

    df.write_csv(output_csv)

    ######################################
    # Prepare fitting data
    ######################################

    if max_sequence_dist_fit is None:
        max_sequence_dist_fit = max_sep

    fit_df = df.filter(
        (pl.col("sequence_dist") > 0)
        & (pl.col("sequence_dist") < max_sequence_dist_fit)
    )

    ######################################
    # Fit with fixed kuhn length
    ######################################
    def _fit_fun_fixed_kuhn(ij: int, nu: float) -> float:
        """For an IDP, this is equivalent to
        return kuhn_fixed * bond_length * ij**nu
        """

        r2 = 0.0
        count = 0

        sep = round(ij)

        if sep == 0:
            return 0.0

        for i in range(n_residues - sep):
            j = i + sep
            path_prop = path_properties_per_residue[(i, j)]
            r2 += path_prop_to_r2(
                path_prop=path_prop,
                kuhn_length=kuhn_fixed,
                bond_length=bond_length,
                nu=nu,
            )
            count += 1

        return r2 / count

    fit_fun_fixed_kuhn = np.vectorize(_fit_fun_fixed_kuhn)

    popt_fixed_kuhn, _pcov_fixed_kuhn = curve_fit(
        fit_fun_fixed_kuhn,
        xdata=fit_df["sequence_dist"].to_numpy(),
        ydata=fit_df["r_ij2_avg"].to_numpy(),
        p0=[1.0],
    )
    nu_fixed = float(popt_fixed_kuhn[0])

    y_fit_fixed = fit_fun_fixed_kuhn(df["sequence_dist"].to_numpy(), *popt_fixed_kuhn)
    r2_fixed = get_r2(y_data=df["r_ij2_avg"].to_numpy(), y_fit=y_fit_fixed)

    ######################################
    # Fit with variable kuhn length
    ######################################

    def _fit_fun_var_kuhn(ij: int, kuhn: float, nu: float) -> float:
        """For an IDP, this is equivalent to
        return kuhn * bond_length * ij**nu
        """

        r2 = 0.0
        count = 0

        sep = round(ij)

        if sep == 0:
            return 0.0

        for i in range(n_residues - sep):
            j = i + sep
            path_prop = path_properties_per_residue[(i, j)]
            r2 += path_prop_to_r2(
                path_prop=path_prop,
                kuhn_length=kuhn,
                bond_length=bond_length,
                nu=nu,
            )
            count += 1

        return r2 / count

    fit_fun_var_kuhn = np.vectorize(_fit_fun_var_kuhn)

    popt_var_kuhn, _pcov_var_kuhn = curve_fit(
        fit_fun_var_kuhn,
        xdata=fit_df["sequence_dist"].to_numpy(),
        ydata=fit_df["r_ij2_avg"].to_numpy(),
        p0=[kuhn_fixed, 1.0],
    )
    kuhn_var = float(popt_var_kuhn[0])
    nu_var = float(popt_var_kuhn[1])

    y_fit_var = fit_fun_var_kuhn(df["sequence_dist"].to_numpy(), *popt_var_kuhn)
    r2_var = get_r2(y_data=df["r_ij2_avg"].to_numpy(), y_fit=y_fit_var)

    with output_json.open("w") as f:
        json.dump(
            {
                "bond_length": bond_length,
                "kuhn_length": kuhn_fixed,
                "nu_fixed": nu_fixed,
                "r2_fixed": r2_fixed,
                "nu_var": nu_var,
                "kuhn_var": kuhn_var,
                "r2_var": r2_var,
            },
            f,
            indent=4,
        )

    ######################################
    # Plot
    ######################################
    fig, ax = plt.subplots()

    x = df["sequence_dist"].to_numpy()
    y = df["r_ij2_avg"].to_numpy()
    yerr = df["r_ij2_err"].to_numpy()

    ax.fill_between(x, y - yerr, y + yerr, color="grey", alpha=0.3)

    ax.plot(x, y, color="black", marker=".", label="data")

    ax.plot(
        np.array(separations),
        rij2_ideal,
        color="black",
        ls="-",
        label="Ideal with MDP corrections",
    )

    ax.plot(
        x,
        y_fit_fixed,
        color="red",
        ls="--",
        label="Fit fixed kuhn",
    )

    ax.plot(
        x,
        y_fit_var,
        color="blue",
        ls="--",
        label="Fit variable kuhn",
    )

    ax.set_xlabel("|i-j|")
    ax.set_ylabel(r"$<R_{ij}^2>$")
    ax.legend()
    fig.savefig(output_csv.parent / "plot.png", dpi=300)


def run_from_snakemake(snakemake) -> None:
    domains_file = snakemake.input.get("globular_domains_file")
    if domains_file is None:
        domains = None
    else:
        with open(domains_file, "r") as f:
            domains = [GlobularDomain(**t) for t in json.load(f)]

    pair_distance(
        output_csv=Path(snakemake.output["output_csv"]),
        output_json=Path(snakemake.output["output_json"]),
        lammps_data_file=Path(snakemake.input["lammps_data_file"]),
        lammps_traj_file=Path(snakemake.input["lammps_traj_file"]),
        n_skip=snakemake.params["n_skip"],
        domains=domains,
        kuhn_fixed=snakemake.params.get("kuhn_length", 5.5),
        bond_length=snakemake.params.get("bond_length", 3.81),
    )


if __name__ == "__main__":
    try:
        from snakemake.script import snakemake  # type: ignore[name-defined]
    except ImportError:
        snakemake = None

    run_from_snakemake(snakemake)

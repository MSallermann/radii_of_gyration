from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import MDAnalysis as mda
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.optimize import curve_fit

from mpipi_lammps_gen.generate_lammps_files import ProteinData
from mpipi_lammps_gen.globular_domains import GlobularDomain, protein_topology

# Adjust these imports if your package layout differs
from mpipi_lammps_gen.shortest_path_graph import (
    PathProperties,
    build_shortest_path_graph,
    get_path_properties,
)

# Your old code imported Graph from networkx, but the API used below
# matches netgraph.Graph, not networkx.Graph.
from netgraph import Graph


def plot_network(
    graph: nx.MultiGraph, ax, prot_data: ProteinData, use_geom_layout: bool
) -> None:
    if use_geom_layout:
        edge_length = {
            k: v / (2 * len(prot_data.sequence_one_letter))
            for k, v in nx.get_edge_attributes(graph, "length").items()
        }

        Graph(
            graph,
            node_labels=True,
            edge_labels=False,
            ax=ax,
            node_layout="geometric",
            node_layout_kwargs={"edge_length": edge_length},
            node_size=1.5,
            node_edge_width=0.1,
            edge_width=0.2,
        )
    else:
        Graph(
            graph,
            node_labels=True,
            edge_labels=False,
            ax=ax,
            node_size=1.5,
            node_edge_width=0.1,
            edge_width=0.2,
        )


def get_r2(y_data: npt.ArrayLike, y_fit: npt.ArrayLike) -> float:
    y_data_arr = np.asarray(y_data, dtype=float)
    y_fit_arr = np.asarray(y_fit, dtype=float)

    y_mean = np.mean(y_data_arr)
    ssres = np.sum((y_data_arr - y_fit_arr) ** 2)
    sstot = np.sum((y_data_arr - y_mean) ** 2)

    return 1.0 - ssres / sstot


def _loop_correction(loop: tuple[int, int] | None, b_fixed: float) -> float:
    """
    Preserve the old loop correction structure, but apply it to the new
    start_loop / end_loop representation.

    loop = (idx_within_loop, n_loop_residues)
    """

    if loop is None:
        return 0.0

    idx_in_loop, loop_length = loop

    # convert to segment counts
    n_segments = loop_length + 1
    idx_segment = idx_in_loop + 1

    return b_fixed**2 * abs(idx_segment) * abs((n_segments - idx_segment) / n_segments)


def path_prop_to_r2(path_prop: PathProperties, b_fixed: float) -> float:
    """
    Convert path properties into an expected <R_ij^2>.

    Contributions:
      - random-walk contour pieces
      - fixed rigid-domain shortcuts
      - loop endpoint corrections

    IMPORTANT:
    `random_walk_contour_length` is a contour length, not a count.

    If get_path_properties(...) was called with bond_length=b_fixed, then:
        n_random_segments = random_walk_contour_length / b_fixed
    and therefore:
        b_fixed**2 * n_random_segments = b_fixed * random_walk_contour_length
    """
    expected_r2 = b_fixed * path_prop.random_walk_contour_length

    for dist in path_prop.fixed_distances:
        expected_r2 += dist**2

    expected_r2 += _loop_correction(path_prop.start_loop, b_fixed)
    expected_r2 += _loop_correction(path_prop.end_loop, b_fixed)

    return expected_r2


def pair_distance(
    output_csv: Path,
    output_json: Path,
    lammps_data_file: Path,
    lammps_traj_file: Path,
    n_skip: int,
    prot_data: ProteinData,
    domains: Iterable[GlobularDomain] | None = None,
    b_fixed: float = 3.81,
) -> None:
    u = mda.Universe(lammps_data_file, lammps_traj_file, format="LAMMPSDUMP")
    atoms = u.atoms

    # number of residues
    n_residues = len(atoms)

    domains_list = list(domains) if domains is not None else []

    ############## Topology ###########################

    topology = protein_topology(n_residues=n_residues, domains=domains_list)

    # fig, ax = plt.subplots()
    # plot_network(topology, ax, prot_data, use_geom_layout=False)
    # fig.savefig(output_csv.parent / "network.png", dpi=300)
    # plt.close(fig)

    assert prot_data.residue_positions is not None
    residue_positions = prot_data.residue_positions

    shortest_path_graph = build_shortest_path_graph(
        topology,
        residue_positions,
        bond_length=b_fixed,
        segment_length=None,
    )

    path_properties_per_residue: dict[tuple[int, int], PathProperties] = {}
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            path_properties_per_residue[(i, j)] = get_path_properties(
                topology=topology,
                i1=i,
                i2=j,
                residue_positions=residue_positions,
                shortest_path_graph=shortest_path_graph,
                bond_length=b_fixed,
                segment_length=None,
            )

    ############## Ideal theta curve ###########################
    max_sep = n_residues - 1
    separations = list(range(1, max_sep + 1))
    rij2_ideal: list[float] = []

    for sep in separations:
        r2_sum = 0.0
        count = 0

        for i in range(n_residues - sep):
            j = i + sep
            path_prop = path_properties_per_residue[(i, j)]
            r2_sum += path_prop_to_r2(path_prop, b_fixed)
            count += 1

        rij2_ideal.append(r2_sum / count)

    fig, ax = plt.subplots()
    ax.plot(
        separations, np.sqrt(rij2_ideal), color="blue", lw=3, label="MDP corrections"
    )
    ax.plot(
        separations,
        b_fixed * np.sqrt(separations),
        color="grey",
        label="IDP behaviour",
    )

    np.savetxt(output_csv.parent / "rij2_ideal.txt", rij2_ideal)
    np.savetxt(output_csv.parent / "separations.txt", separations)

    ax.legend()
    ax.set_ylabel(r"$\sqrt{<R_{ij}^2>}$")
    ax.set_xlabel("|i-j|")
    fig.savefig(output_csv.parent / "ideal_theta.png", dpi=300)
    plt.close(fig)

    ##############################################################

    sum_mean = np.zeros(max_sep + 1)
    sum_mean_sq = np.zeros(max_sep + 1)
    n_frames = 0

    # Iterate over frames in the trajectory
    for ts in u.trajectory[n_skip:]:
        pos = atoms.positions

        # mean of r_ij^2 per frame
        frame_mean = np.zeros(max_sep + 1)

        for sep in range(1, max_sep + 1):
            r2 = np.linalg.norm(pos[:-sep] - pos[sep:], axis=1) ** 2
            frame_mean[sep] = r2.mean()

        sum_mean += frame_mean
        sum_mean_sq += frame_mean**2
        n_frames += 1

    r_sep = np.sqrt(sum_mean / n_frames)
    var = np.sqrt((sum_mean_sq - n_frames * r_sep**2) / (n_frames - 1))
    r_sep_err = np.sqrt(var / n_frames)

    df = pl.DataFrame(
        {
            "sequence_dist": range(max_sep + 1),
            "r_ij_avg": r_sep,
            "r_ij_err": r_sep_err,
        }
    )

    df = df.fill_nan(0.0)
    df.write_csv(output_csv)

    # Fit
    def _fit_fun(ij: float, b: float, nu: float) -> float:
        return b * ij**nu

    fit_fun = np.vectorize(_fit_fun)

    # fit with a fixed Kuhn distance
    popt_fixed_kuhn, _pcov_fixed_kuhn = curve_fit(
        lambda ij, nu: fit_fun(ij, b=b_fixed, nu=nu),
        xdata=df["sequence_dist"],
        ydata=df["r_ij_avg"],
        p0=[0.5],
    )
    nu_fixed = popt_fixed_kuhn[0]

    y_fit_fixed = fit_fun(df["sequence_dist"], b=b_fixed, nu=nu_fixed)
    r2_fixed = get_r2(y_data=df["r_ij_avg"].to_numpy(), y_fit=y_fit_fixed)

    with output_json.open("w") as f:
        json.dump(
            {
                "b_fixed": b_fixed,
                "nu_fixed": nu_fixed,
                "r2_fixed": r2_fixed,
            },
            f,
        )

    ### Make a quick plot
    fig, ax = plt.subplots()

    ax.fill_between(
        df["sequence_dist"],
        df["r_ij_avg"] - df["r_ij_err"],
        df["r_ij_avg"] + df["r_ij_err"],
        color="grey",
        alpha=0.3,
    )

    ax.plot(
        df["sequence_dist"],
        df["r_ij_avg"],
        color="black",
        marker=".",
        label="data",
    )

    ax.plot(
        separations,
        np.sqrt(rij2_ideal),
        color="black",
        ls="-",
        label="Ideal with MDP corrections",
    )

    ax.plot(
        separations,
        b_fixed * np.sqrt(separations),
        color="black",
        ls="--",
        label="Ideal (IDP)",
    )

    ax.plot(
        df["sequence_dist"],
        y_fit_fixed,
        color="red",
        label=f"Kuhn fit, b={b_fixed:.2f} (fix), $R^2$={r2_fixed:.2f}, (nu={nu_fixed:.2f})",
    )

    ax.set_xlabel("|i-j|")
    ax.set_ylabel(r"$\sqrt{<R_{ij}^2>}$")
    ax.legend()
    fig.savefig(output_csv.parent / "plot.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    try:
        from snakemake.script import snakemake
    except ImportError:
        pass

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
        prot_data=ProteinData(**snakemake.params["prot_data"]),
    )

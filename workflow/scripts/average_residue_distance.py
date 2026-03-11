import numpy as np
import MDAnalysis as mda
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import numpy.typing as npt
from mpipi_lammps_gen.globular_domains import (
    GlobularDomain,
    protein_topology,
    build_protein_graph,
    get_path_properties,
    PathProperties,
    shortest_path_matrix,
)
from mpipi_lammps_gen.generate_lammps_files import ProteinData
from collections.abc import Iterable
from networkx import Graph
import networkx as nx


def plot_network(graph, ax, prot_data: ProteinData, use_geom_layout: bool):

    if use_geom_layout:
        # compute edge lengths
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
    prot_data: ProteinData,
    domains: Iterable[GlobularDomain] | None = None,
    b_fixed: float = 3.81,
):
    u = mda.Universe(lammps_data_file, lammps_traj_file, format="LAMMPSDUMP")
    atoms = u.atoms

    # number of residues
    N = len(atoms)

    ############## Topology ###########################

    graph = protein_topology(n_residues=N, domains=list(domains))

    fig, ax = plt.subplots()
    plot_network(graph, ax, prot_data, use_geom_layout=False)
    fig.savefig(output_csv.parent / "network.png", dpi=300)

    assert prot_data.residue_positions is not None

    path_properties_per_residue: dict[tuple[int, int], PathProperties] = {}
    for i in range(N):
        for j in range(i + 1, N):
            path_properties_per_residue[(i, j)] = get_path_properties(
                graph, i, j, prot_data.residue_positions
            )

    def path_prop_to_r2(path_prop: PathProperties, b_fixed: float) -> float:
        expected_r2 = b_fixed**2 * path_props.n_random_segments

        for dist in path_props.fixed_distances:
            expected_r2 += dist**2

        for loop in path_props.loops:
            expected_r2 += (
                b_fixed**2 * np.abs(loop[0]) * np.abs((loop[1] - loop[0]) / loop[0])
            )

        for loop in path_props.loops:
            expected_r2 += (
                b_fixed**2 * np.abs(loop[0]) * np.abs((loop[1] - loop[0]) / loop[0])
            )

        return expected_r2

    ############## Ideal theta curve ###########################
    max_sep = N - 1
    separations = list(range(1, max_sep))
    rij2_ideal = []
    for sep in separations:
        r2_sum = 0
        count = 0
        for i in range(N - sep):
            j = i + sep
            if j < N:
                path_props = path_properties_per_residue[i, j]
                r2_sum += path_prop_to_r2(path_props, b_fixed)
                count += 1

        rij2_ideal.append(r2_sum / count)

    fig, ax = plt.subplots()
    ax.plot(separations, np.sqrt(rij2_ideal), color="blue", label="MDP corrections")
    ax.plot(
        separations, b_fixed * np.sqrt(separations), color="grey", label="IDP behaviour"
    )
    ax.legend()
    ax.set_ylabel(r"$\sqrt{<R_{ij}^2>}$")
    ax.set_xlabel("|i-j|")
    fig.savefig(output_csv.parent / "ideal_theta.png", dpi=300)
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
            # for each `sep`, we grab the indices in the sequence which are separated by that much

            r2 = np.linalg.norm(pos[:-sep] - pos[sep:], axis=1) ** 2
            frame_mean[sep] = r2.mean()

        sum_mean += frame_mean
        sum_mean_sq += frame_mean**2
        n_frames += 1

    R_sep = np.sqrt(sum_mean / n_frames)
    var = np.sqrt((sum_mean_sq - n_frames * R_sep**2) / (n_frames - 1))
    R_sep_err = np.sqrt(var / n_frames)

    df = pl.DataFrame(
        {"sequence_dist": range(max_sep + 1), "r_ij_avg": R_sep, "r_ij_err": R_sep_err}
    )

    df = df.fill_nan(0.0)
    df.write_csv(output_csv)

    # Fit
    def _fit_fun(ij: float, b: float, nu: float) -> float:
        return b * ij**nu

    fit_fun = np.vectorize(_fit_fun)

    # fit with a fixed kuhn distance
    popt_fixed_kuhn, pcov_fixed_kuhn = curve_fit(
        lambda ij, nu: fit_fun(ij, b=b_fixed, nu=nu),
        xdata=df["sequence_dist"],
        ydata=df["r_ij_avg"],
        p0=[0.5],
    )
    nu_fixed = popt_fixed_kuhn[0]

    # fit with a variable kuhn distance
    # popt_variable_kuhn, pcov_variable_kuhn = curve_fit(
    #     fit_fun,
    #     xdata=df["sequence_dist"],
    #     ydata=df["r_ij_avg"],
    #     p0=[b_fixed, 0.5],
    # )
    # b_variable = popt_variable_kuhn[0]
    # nu_variable = popt_variable_kuhn[1]

    # y_fit_variable = fit_fun(df["sequence_dist"], b=b_variable, nu=nu_variable)
    # r2_variable = get_r2(y_data=df["r_ij_avg"].to_numpy(), y_fit=y_fit_variable)

    y_fit_fixed = fit_fun(df["sequence_dist"], b=b_fixed, nu=nu_fixed)
    r2_fixed = get_r2(y_data=df["r_ij_avg"].to_numpy(), y_fit=y_fit_fixed)

    with output_json.open("w") as f:
        json.dump(
            {
                "b_fixed": b_fixed,
                "nu_fixed": nu_fixed,
                # "b_variable": b_variable,
                # "nu_variable": nu_variable,
                "r2_fixed": r2_fixed,
                # "r2_variable": r2_variable,
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
        df["sequence_dist"], df["r_ij_avg"], color="black", marker=".", label="data"
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

    # ax.plot(
    #     df["sequence_dist"],
    #     y_fit_variable,
    #     color="blue",
    #     label=f"Kuhn fit, $R^2$={r2_variable:.2f}, (b={b_variable:.2f}, nu={nu_variable:.2f})",
    # )

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


if __name__ == "__main__":
    try:
        from snakemake.script import snakemake
    except ImportError:
        ...

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
        use_graph_distance=snakemake.params["use_graph_distance"],
    )

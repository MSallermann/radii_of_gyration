from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import logging

from mpipi_lammps_gen.generate_lammps_files import (
    generate_lammps_data,
    get_lammps_group_definition,
    get_lammps_minimize_command,
    get_lammps_nvt_command,
    get_lammps_npt_command,
    get_lammps_viscous_command,
    write_lammps_data_file,
    trim_protein,
    LammpsData,
    ProteinData,
)
from mpipi_lammps_gen.util import (
    coordination_numbers_from_distance_matrix,
    group_distance_matrix,
)
from mpipi_lammps_gen.globular_domains import (
    decide_globular_domains_from_sequence,
    merge_domains,
    GlobularDomain,
)
from mpipi_lammps_gen.render_jinja2 import render_jinja2
from mpipi_lammps_gen.generate_pair_interactions import (
    get_wf_pairs_str,
    generate_wf_interactions,
)

logger = logging.getLogger(__name__)


@dataclass
class Params:
    temp: float
    ionic_strength: float

    n_steps: int
    timestep: float

    n_save_rg: int
    n_save_traj: int
    n_save_density: int

    nvt_seed: int

    #### criterion
    # the plddts threshold
    threshold: float
    # the minimum length of a domain in the sequence based criterion
    minimum_domain_length: int
    # the minimum lenght of an IDR between two domains in the sequence based criterion
    minimum_idr_length: int
    # if the minimum pae between a pair of groups is below this, they are merged
    min_pae_cutoff: float | None
    # if the mean pae between a pair of groups is below this, they are merged
    mean_pae_cutoff: float | None
    # if the minimum distance between a pair of groups is below this, they are merged
    min_distance_cutoff: float | None
    # if the maximum coordination number between a pair of groups is greater than or equal to this, they are merged
    max_coordination_cutoff: int | None
    # the distance cutoff used to compute coordination numbers
    coordination_distance_cutoff: float | None

    n_proteins_x: int = 1
    n_proteins_y: int = 1
    n_proteins_z: int = 1

    box_buffer: float | None = 1.0

    dt_ramp_up: list[float] = field(default_factory=lambda: [0.1, 1.0])
    steps_per_stage: int = 10000

    press: float | None = None

    intermediate_press: float | None = None
    n_steps_intermediate_press: int | None = None

    # slicing up the sequence
    start_idx: int | None = None
    end_idx: int | None = None


def create_lammps_data(params: Params, protein_data: ProteinData) -> LammpsData:
    # Optionally slice up the proteins
    if params.start_idx is not None or params.end_idx is not None:
        if params.start_idx is None:
            params.start_idx = 0
        if params.end_idx is None:
            params.end_idx = len(protein_data.sequence_one_letter)

        protein_data = trim_protein(
            protein_data, params.start_idx - 1, params.end_idx - 1
        )

    # Decide groups based on sequence
    assert protein_data.plddts is not None
    globular_domains = decide_globular_domains_from_sequence(
        plddts=protein_data.plddts,
        threshold=params.threshold,
        minimum_domain_length=params.minimum_domain_length,
        minimum_idr_length=params.minimum_idr_length,
    )

    # Merge groups based on PAE
    if params.min_pae_cutoff is not None or params.mean_pae_cutoff is not None:
        assert protein_data.pae is not None

        pae_matrix_arr = np.array(protein_data.pae)

        def merge_based_on_pae(g1: GlobularDomain, g2: GlobularDomain) -> bool:
            sub_pae = pae_matrix_arr[
                g1.start_idx() : g1.end_idx(), g2.start_idx() : g2.end_idx()
            ]

            min_pae = np.min(np.ravel(sub_pae))

            if params.min_pae_cutoff is not None and min_pae < params.min_pae_cutoff:
                return True

            mean_pae = np.mean(np.ravel(sub_pae))

            if params.mean_pae_cutoff is not None and mean_pae < params.mean_pae_cutoff:
                return True

            return False

        globular_domains = merge_domains(
            globular_domains, should_be_merged=merge_based_on_pae
        )

    # Merge groups based on distance
    if (
        params.min_distance_cutoff is not None
        or params.max_coordination_cutoff is not None
    ):
        # if we want to look at the coordination numbers we also need the cutoff for that
        if params.max_coordination_cutoff is not None:
            assert params.coordination_distance_cutoff is not None

        residue_positions = protein_data.get_residue_positions()
        assert residue_positions is not None

        def merge_based_on_distance(g1: GlobularDomain, g2: GlobularDomain) -> bool:
            distance_matrix = group_distance_matrix(residue_positions, g1, g2)

            if params.min_distance_cutoff is not None:
                min_distance = np.min(np.ravel(distance_matrix))
                if min_distance <= params.min_distance_cutoff:
                    return True

            if (
                params.max_coordination_cutoff is not None
                and params.coordination_distance_cutoff is not None
            ):
                coord1, coord2 = coordination_numbers_from_distance_matrix(
                    distance_matrix=distance_matrix,
                    cutoff=params.coordination_distance_cutoff,
                )
                if (
                    np.max(coord1) >= params.max_coordination_cutoff
                    or np.max(coord2) >= params.max_coordination_cutoff
                ):
                    return True

            return False

        globular_domains = merge_domains(
            globular_domains, should_be_merged=merge_based_on_distance
        )

    if params.box_buffer is None:
        box_buffer = 0.0
    else:
        box_buffer = params.box_buffer

    lammps_data = generate_lammps_data(
        protein_data,
        globular_domains,
        box_buffer=box_buffer,
        n_proteins_x=params.n_proteins_x,
        n_proteins_y=params.n_proteins_y,
        n_proteins_z=params.n_proteins_z,
    )

    return lammps_data


def create_lammps_files(
    script_path: Path,
    data_file_path: Path,
    jinja_workdir: Path,
    params: Params,
    script_template_file: Path,
    lammps_data: LammpsData,
    pairs_str: str,
):
    # First write the data file to the output folder
    data_file_str = write_lammps_data_file(lammps_data)
    data_file_path.parent.mkdir(exist_ok=True, parents=True)
    with data_file_path.open("w") as f:
        f.write(data_file_str)

    # get the group definition string
    groups_definition_str = get_lammps_group_definition(lammps_data)

    min_cmd = get_lammps_minimize_command(
        lammps_data, etol=0.0, ftol=0.0001, maxiter=10000, max_eval=40000, timestep=0.1
    )

    viscous_cmd = get_lammps_viscous_command(
        lammps_data,
        n_time_steps=params.steps_per_stage,
        timestep=0.1,
        damp=1000,
        limit=0.01,
    )

    if params.press is None:
        production_run = get_lammps_nvt_command(
            lammps_data,
            timestep=params.timestep,
            temp=params.temp,
            n_time_steps=params.n_steps,
            dt_ramp_up=params.dt_ramp_up,
            steps_per_stage=params.steps_per_stage,
            seed=params.nvt_seed,
        )
    else:
        # else we have to run npt

        n_steps_small_nvt = params.steps_per_stage
        production_run = get_lammps_nvt_command(
            lammps_data,
            timestep=params.timestep,
            temp=params.temp,
            n_time_steps=n_steps_small_nvt,
            dt_ramp_up=params.dt_ramp_up,
            steps_per_stage=params.steps_per_stage,
            seed=params.nvt_seed,
        )

        if params.intermediate_press is not None:
            assert params.n_steps_intermediate_press is not None

            production_run += f"# Running {params.n_steps_intermediate_press} steps at intermediate pressure of {params.intermediate_press} atm\n"
            production_run += get_lammps_npt_command(
                lammps_data,
                timestep=params.timestep,
                temp=params.temp,
                press=params.intermediate_press,
                n_time_steps=params.n_steps_intermediate_press,
                dt_ramp_up=params.dt_ramp_up,
                steps_per_stage=params.steps_per_stage,
                seed=params.nvt_seed,
            )

            # We dont need the ramp up phase twice
            params.dt_ramp_up = []

        production_run += get_lammps_npt_command(
            lammps_data,
            timestep=params.timestep,
            temp=params.temp,
            press=params.press,
            n_time_steps=params.n_steps,
            dt_ramp_up=params.dt_ramp_up,
            steps_per_stage=params.steps_per_stage,
            seed=params.nvt_seed,
        )

    run_str = min_cmd + viscous_cmd + production_run

    variables: dict[str, Any] = {
        "input": {"template": script_template_file},
        "params": {
            "temperature": params.temp,
            "n_steps": params.n_steps,
            "ionic_strength": params.ionic_strength,
            "groups_definition_str": groups_definition_str,
            "data_file_name": data_file_path.name,
            "run_str": run_str,
            "pairs": pairs_str,
            "n_save_rg": params.n_save_rg,
            "n_save_traj": params.n_save_traj,
            "n_save_density": params.n_save_density,
        },
    }

    render_jinja2(
        working_directory=jinja_workdir,
        template_file=script_template_file,
        included_files=[],
        variables=variables,
        output=script_path,
    )


if __name__ == "__main__":
    try:
        from snakemake.script import snakemake
    except ImportError:
        ...

    template_file = snakemake.input["template_file"]

    params = Params(
        temp=snakemake.params["temp"],
        press=snakemake.params["press"],
        ionic_strength=snakemake.params["ionic_strength"],
        n_steps=snakemake.params["n_steps"],
        timestep=snakemake.params["timestep"],
        threshold=snakemake.params["threshold"],
        minimum_domain_length=snakemake.params["minimum_domain_length"],
        minimum_idr_length=snakemake.params["minimum_idr_length"],
        start_idx=snakemake.params.get("start_idx"),
        end_idx=snakemake.params.get("end_idx"),
        min_pae_cutoff=snakemake.params.get("min_pae_cutoff"),
        mean_pae_cutoff=snakemake.params.get("mean_pae_cutoff"),
        min_distance_cutoff=snakemake.params.get("min_distance_cutoff"),
        max_coordination_cutoff=snakemake.params.get("max_coordination_cutoff"),
        coordination_distance_cutoff=snakemake.params.get(
            "coordination_distance_cutoff"
        ),
        box_buffer=snakemake.params.get("box_buffer", 0.0),
        n_save_traj=snakemake.params["n_save_traj"],
        n_save_rg=snakemake.params["n_save_rg"],
        n_save_density=snakemake.params["n_save_density"],
        nvt_seed=snakemake.params["nvt_seed"],
        n_proteins_x=snakemake.params.get("n_proteins_x", 1),
        n_proteins_y=snakemake.params.get("n_proteins_y", 1),
        n_proteins_z=snakemake.params.get("n_proteins_z", 1),
        intermediate_press=snakemake.params.get("intermediate_pressure"),
        n_steps_intermediate_press=snakemake.params.get(
            "n_steps_intermediate_pressure"
        ),
    )

    protein_data_dict = snakemake.params["prot_data"]
    protein_data = ProteinData(**protein_data_dict)

    residue_location = snakemake.params.get("residue_location", "Ca")

    if residue_location is not None:
        if protein_data.atom_xyz is None:
            msg = "Tried to use the `residue_location` parameter, but no atomic coordinates are given in the coarse grained protein_data (prot_data.atom_xyz is None)"
            raise Exception(msg)

        if residue_location == "calvados":
            assert protein_data.plddts is not None
            # In CALVADOS, residues in glob domains are at the center of mass position, while residues in IDRs are at Ca
            method = [
                "com" if plddt > params.threshold else "Ca"
                for plddt in protein_data.plddts
            ]
        else:
            method = residue_location

        residue_positions = (
            protein_data.compute_residue_positions(method=method) is not None
        )
    else:
        residue_positions = protein_data.get_residue_positions()

    assert residue_positions is not None

    lammps_data = create_lammps_data(params, protein_data)

    wf_interactions = generate_wf_interactions(
        idr_glob_scaling=snakemake.params["idr_glob_scaling"],
        glob_glob_scaling=snakemake.params["glob_glob_scaling"],
    )
    pairs_str = get_wf_pairs_str(interactions=wf_interactions)

    script_path = Path(snakemake.output["script"])
    data_file_path = Path(snakemake.output["data_file"])

    output_folder = data_file_path.parent

    create_lammps_files(
        script_path=script_path,
        data_file_path=data_file_path,
        jinja_workdir=output_folder / ".jinjaworkdirs",
        params=params,
        script_template_file=template_file,
        lammps_data=lammps_data,
        pairs_str=pairs_str,
    )

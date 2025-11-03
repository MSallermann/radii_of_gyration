import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import logging

logger = logging.getLogger(__name__)

from mpipi_lammps_gen.generate_lammps_files import (
    generate_lammps_data,
    get_lammps_group_definition,
    get_lammps_minimize_command,
    get_lammps_nvt_command,
    get_lammps_viscous_command,
    parse_cif,
    write_lammps_data_file,
    trim_protein,
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


@dataclass
class Params:
    temp: float
    ionic_strength: float
    n_steps: int
    timestep: float

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

    box_buffer: float = 100.0

    # slicing up the sequence
    start_idx: int | None = None
    end_idx: int | None = None


def coarse_grain(
    output: Path,
    params: Params,
    template_file: Path,
    protein_data: ProteinData | None,
    cif_text: str | None = None,
):
    if protein_data is None and cif_text is None:
        raise Exception("Protein data and cif_text cannot both be None")

    if protein_data is None and cif_text is not None:
        protein_data = parse_cif(cif_text)

    assert protein_data is not None  # this is mainly for pyright

    output.mkdir(parents=True, exist_ok=True)

    data_file_path = output / "data_file.data"

    workdir = output / "workdir"
    workdir.mkdir(exist_ok=True)

    script_path = output / "script.lmp"

    # Optionally slice up the proteins
    if params.start_idx is not None or params.end_idx is not None:
        if params.start_idx is None:
            params.start_idx = 0
        if params.end_idx is None:
            params.end_idx = len(protein_data.sequence_one_letter)

        protein_data = trim_protein(protein_data, params.start_idx, params.end_idx)

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

    lammps_data = generate_lammps_data(
        protein_data, globular_domains, box_buffer=params.box_buffer
    )

    data_file_str = write_lammps_data_file(lammps_data)
    with data_file_path.open("w") as f:
        f.write(data_file_str)

    groups_definition_str = get_lammps_group_definition(lammps_data)

    min_cmd = get_lammps_minimize_command(
        lammps_data, etol=0.0, ftol=0.0001, maxiter=10000, max_eval=40000, timestep=0.1
    )

    viscous_cmd = get_lammps_viscous_command(
        lammps_data, n_time_steps=10000, timestep=0.1, damp=1000, limit=0.01
    )

    nvt_cmd = get_lammps_nvt_command(
        lammps_data,
        timestep=params.timestep,
        temp=params.temp,
        n_time_steps=params.n_steps,
        dt_ramp_up=[0.1, 1.0],
        steps_per_stage=10000,
    )

    run_str = min_cmd + viscous_cmd + nvt_cmd

    variables: dict[str, Any] = {
        "input": {"template": template_file},
        "params": {
            "temperature": params.temp,
            "n_steps": params.n_steps,
            "ionic_strength": params.ionic_strength,
            "groups_definition_str": groups_definition_str,
            "data_file_name": data_file_path.name,
            "run_str": run_str,
        },
    }

    render_jinja2(
        working_directory=workdir,
        template_file=template_file,
        included_files=[],
        variables=variables,
        output=script_path,
    )

    shutil.rmtree(workdir)


if __name__ == "__main__":
    from snakemake.script import snakemake

    template_file = snakemake.input["template_file"]

    params = Params(
        temp=snakemake.params["temp"],
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
    )

    protein_data_dict = snakemake.params.get("prot_data")

    if protein_data_dict is None:
        protein_data = parse_cif(snakemake.params["cif_text"])
    else:
        protein_data = ProteinData(**protein_data_dict)

    coarse_grain(
        output=Path(snakemake.output[0]),
        params=params,
        template_file=template_file,
        protein_data=protein_data,
    )

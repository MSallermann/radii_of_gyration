import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mpipi_lammps_gen.generate_lammps_files import (
    generate_lammps_data,
    get_lammps_group_definition,
    get_lammps_minimize_command,
    get_lammps_nvt_command,
    get_lammps_viscous_command,
    parse_cif,
    write_lammps_data_file,
)
from mpipi_lammps_gen.globular_domains import decide_globular_domains_from_sequence
from mpipi_lammps_gen.render_jinja2 import render_jinja2


@dataclass
class Params:
    temp: float = 293.0
    ionic_strength: float = 150.0
    n_steps: int = int(10e6)
    timestep: float = 10.0

    threshold: float = 70.0
    minimum_domain_length: int = 3
    minimum_idr_length: int = 3

    box_buffer: float = 100.0


def coarse_grain(
    output: Path,
    params: Params,
    template_file: Path,
    plddts: list[float],
    cif_text: str,
):

    output.mkdir(parents=True, exist_ok=True)

    data_file_path = output / "data_file.data"

    workdir = output / "workdir"
    workdir.mkdir(exist_ok=True)

    script_path = output / "script.lmp"

    protein_data = parse_cif(cif_text=cif_text)

    globular_domains = decide_globular_domains_from_sequence(
        plddts=plddts,
        threshold=params.threshold,
        minimum_domain_length=params.minimum_domain_length,
        minimum_idr_length=params.minimum_idr_length,
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

    plddts = snakemake.params["plddts"]

    params = Params(
        temp=snakemake.params["temp"],
        ionic_strength=snakemake.params["ionic_strength"],
        n_steps=snakemake.params["n_steps"],
        timestep=snakemake.params["timestep"],
        threshold=snakemake.params["threshold"],
        minimum_domain_length=snakemake.params["minimum_domain_length"],
        minimum_idr_length=snakemake.params["minimum_idr_length"],
    )

    coarse_grain(
        output=Path(snakemake.output[0]),
        params=params,
        template_file=template_file,
        plddts=plddts,
        cif_text=snakemake.params["cif_text"],
    )

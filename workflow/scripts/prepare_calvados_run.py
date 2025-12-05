from calvados.cfg import Config, Components

from mpipi_lammps_gen.generate_lammps_files import (
    ProteinData,
    trim_protein,
)
from mpipi_lammps_gen.mdtraj_conversion import protein_to_mdtraj, plddt_to_bfactor
import json
from dataclasses import dataclass

import numpy as np
from pathlib import Path


@dataclass
class CalvadosParams:
    name: str

    temperature_k: float
    ionic_strength_mm: float
    n_steps: int
    n_save: int
    start_idx: int | None
    end_idx: int | None
    box_buffer: float | None
    prot_data: ProteinData

    rg_skip: int

    residues_file: Path


def write_calvados_inp_files(output_path: Path, params: CalvadosParams):
    output_path = output_path.absolute()
    output_path.mkdir(exist_ok=True)

    start_idx = params.start_idx - 1 if params.start_idx is not None else 0
    end_idx = (
        params.end_idx - 1
        if params.end_idx is not None
        else len(params.prot_data.sequence_one_letter)
    )

    prot_data = trim_protein(params.prot_data, start_idx, end_idx)

    res_pos = np.array(prot_data.get_residue_positions())
    x_lo = (np.min(res_pos[:, 0]) - params.box_buffer) / 10.0  # convert to nm
    x_hi = (np.max(res_pos[:, 0]) + params.box_buffer) / 10.0  # convert to nm
    y_lo = (np.min(res_pos[:, 1]) - params.box_buffer) / 10.0  # convert to nm
    y_hi = (np.max(res_pos[:, 1]) + params.box_buffer) / 10.0  # convert to nm
    z_lo = (np.min(res_pos[:, 2]) - params.box_buffer) / 10.0  # convert to nm
    z_hi = (np.max(res_pos[:, 2]) + params.box_buffer) / 10.0  # convert to nm

    box = np.array([x_hi - x_lo, y_hi - y_lo, z_hi - z_lo]).tolist()

    # set the saving interval (number of integration steps)

    calvados_config = Config(
        sysname=params.name,
        box=box,
        temp=params.temperature_k,
        ionic=params.ionic_strength_mm / 1000.0,  # conver to molar
        pH=7.5,
        topol="center",
        fresidues=params.residues_file.as_posix(),
        wfreq=params.n_save,  # every ns
        steps=params.n_steps,
        platform="CPU",
        restart="checkpoint",
        frestart="restart.chk",
        verbose=True,
        runtime=0,  # overwrites 'steps' keyword if > 0
    )

    data_path = output_path / "data"
    data_path.mkdir(exist_ok=True)

    analysis_output_path: str = output_path.relative_to(output_path).as_posix()
    analysis_data_folder: str = data_path.relative_to(output_path).as_posix()

    analyses = f"""
    from calvados.analysis import save_conf_prop
    import polars as pl
    import json

    save_conf_prop(path="{analysis_output_path:s}",name="{params.name:s}",residues_file="{params.residues_file.as_posix():s}",output_path="{analysis_data_folder}",start={params.rg_skip},is_idr=True,select='all')
    
    df = pl.read_csv("{analysis_data_folder}/conf_prop.csv")
    rg_mean = df.filter( pl.col("") == "Rg" ).row(0, named=True)["value"]
    rg_err = df.filter( pl.col("") == "Rg" ).row(0, named=True)["error"]

    with open("./rg.json", "w") as f:
        json.dump({{"rg_mean": rg_mean*10.0, "rg_err": rg_err*10.0}}, f) # convert to angs and save
    """

    output_path.mkdir(exist_ok=True, parents=True)
    calvados_config.write(output_path, name="config.yaml", analyses=analyses)

    pdb_folder = (output_path / "pdbs").absolute()
    pdb_folder.mkdir(exist_ok=True)

    components = Components(
        # Defaults
        molecule_type="protein",
        nmol=1,  # number of molecules
        restraint=True,  # apply restraints
        charge_termini="both",  # charge N or C or both
        # INPUT
        fresidues=params.residues_file.as_posix(),  # residue definitions
        pdb_folder=pdb_folder.as_posix(),  # directory for pdb and PAE files
        # RESTRAINTS
        restraint_type="go",  # harmonic or go
        use_com=True,  # apply on centers of mass instead of CA
        colabfold=0,  # PAE format (EBI AF=0, Colabfold=1&2)
        k_go=15.0,  # Restraint force constant
    )
    components.add(name=params.name)
    components.write(output_path, name="components.yaml")

    # Write the pdb file
    traj = protein_to_mdtraj(prot_data)

    traj.unitcell_lengths = [
        b * 10.0 for b in box
    ]  # have to convert back to angstrom for pdb
    traj.unitcell_angles = [90.0] * 3

    traj.save_pdb(
        pdb_folder / f"{params.name}.pdb", bfactors=plddt_to_bfactor(prot_data)
    )

    # Write the pae file
    pae_matrix = np.array(prot_data.pae)
    pae_dict = {
        "predicted_aligned_error": prot_data.pae,
        "max_predicted_aligned_error": float(np.max(np.ravel(pae_matrix))),
    }

    with open(pdb_folder / f"{params.name}.json", "w") as f:
        json.dump([pae_dict], f)


if __name__ == "__main__":
    # from snakemake.script import snakemake

    params = CalvadosParams(
        name=snakemake.params["name"],
        temperature_k=snakemake.params["temp"],
        ionic_strength_mm=snakemake.params["ionic_strength"],
        n_steps=snakemake.params["n_steps"],
        n_save=snakemake.params["n_save"],
        start_idx=snakemake.params.get("start_idx"),
        end_idx=snakemake.params.get("end_idx"),
        box_buffer=snakemake.params.get("box_buffer", 1000.0),
        prot_data=ProteinData(**snakemake.params["prot_data"]),
        residues_file=Path(snakemake.params["residues_file"]),
        rg_skip=snakemake.params["rg_skip"],
    )

    write_calvados_inp_files(output_path=Path(snakemake.output[0]), params=params)

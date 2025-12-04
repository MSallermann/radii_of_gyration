from calvados.cfg import Config, Components

from mpipi_lammps_gen.generate_lammps_files import (
    ProteinData,
    trim_protein,
    parse_cif_from_path,
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
    box_buffer: float
    prot_data: ProteinData

    residues_file: Path


def write_calvados_inp_files(output_path: Path, params: CalvadosParams):
    output_path = output_path.absolute()
    output_path.mkdir(exist_ok=True)

    print(output_path)

    start_idx = params.start_idx - 1 if params.start_idx is not None else 0
    end_idx = (
        params.end_idx - 1
        if params.end_idx is not None
        else len(params.prot_data.sequence_one_letter)
    )

    prot_data = trim_protein(params.prot_data, start_idx, end_idx)

    res_pos = np.array(prot_data.residue_positions)
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

    analyses = f"""

    from calvados.analysis import save_conf_prop

    save_conf_prop(path="{output_path.as_posix():s}",name="{params.name:s}",residues_file="{params.residues_file.as_posix():s}",output_path="{data_path.as_posix()}",start=0,is_idr=True,select='all')
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

    traj.unitcell_lengths = box
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
    prot_data = parse_cif_from_path(
        Path(
            "/home/sie/Biocondensates/lammps_input_generator_for_mpipi/tests/res/Q9ULK0.cif"
        )
    )
    n_res = len(prot_data.sequence_one_letter)
    prot_data.pae = (np.random.uniform(size=(n_res, n_res)) * 10.0).tolist()

    params = CalvadosParams(
        name="testing",
        temperature_k=300,
        ionic_strength_mm=150,
        n_steps=10000,
        n_save=10,
        start_idx=20,
        end_idx=30,
        box_buffer=100.0,
        prot_data=prot_data,
        residues_file=Path(
            "/home/sie/Biocondensates/CALVADOS_stuff/CALVADOS/examples/single_AF_CALVADOS/input/residues.csv"
        ),
    )

    write_calvados_inp_files(output_path=Path("./test_output"), params=params)

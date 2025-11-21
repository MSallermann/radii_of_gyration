import json
import sys
from pathlib import Path

script_folder = Path(__file__).parent.parent / "workflow" / "scripts"
sys.path.insert(0, script_folder.as_posix())
import aggregate_json  # type: ignore # noqa: E402


def aggregate_from_run_folder(run_folder: Path):
    with (run_folder / "samples.json").open("rb") as f:
        samples = json.load(f)

    aggregate_json.main(
        input_paths=[run_folder / f"results/rg/{sample}/rg.json" for sample in samples],
        output_path=run_folder / "aggregated.csv",
        added_columns={
            "accession": [samples[sample]["accession"] for sample in samples],
            "threshold": [samples[sample]["threshold"] for sample in samples],
            "temperature": [samples[sample]["temperature"] for sample in samples],
            "ionic_strength": [samples[sample]["ionic_strength"] for sample in samples],
        },
        ignore_keys=["file"],
    )


if __name__ == "__main__":
    run_folder = Path("/gpfs/projects/ucm96/moritz/rg_workflow/runs/test100")
    aggregate_from_run_folder(run_folder)

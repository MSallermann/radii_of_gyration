import json
import sys
from pathlib import Path

script_folder = Path(__file__).parent.parent / "workflow" / "scripts"
sys.path.insert(0, script_folder.as_posix())
import aggregate_json  # type: ignore # noqa: E402


def aggregate_from_run_folder(run_folder: Path, output_file: Path | None = None):
    with (run_folder / "samples.json").open("rb") as f:
        samples = json.load(f)

    keys = set()
    for samp in samples.values():
        keys.update(samp.keys())

    added_columns = {
        key: [sample.get(key) for sample in samples.values()] for key in keys
    }

    if output_file is None:
        output_file = run_folder / "aggregated.csv"

    print(f"Aggregating from {run_folder} to {output_file}")

    aggregate_json.main(
        input_paths=[run_folder / f"results/rg/{sample}/rg.json" for sample in samples],
        output_path=output_file,
        added_columns={**added_columns, "sample_key": list(samples.keys())},
        ignore_keys=["file"],
    )


if __name__ == "__main__":
    run_folder = Path("/gpfs/projects/ucm96/moritz/rg_workflow/runs/test100")

    for run_folder in Path("/work/e280/e280/moritzsa/rg_opt_runs_55").glob("*"):
        if run_folder.is_dir():
            aggregate_from_run_folder(
                run_folder,
                output_file=Path("/work/e280/e280/moritzsa/rg_opt_runs_55")
                / f"{run_folder.name}.csv",
            )

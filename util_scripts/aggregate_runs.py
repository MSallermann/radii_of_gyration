import json
import sys
from pathlib import Path

script_folder = Path(__file__).parent.parent / "workflow" / "scripts"
sys.path.insert(0, script_folder.as_posix())
import aggregate_json  # type: ignore # noqa: E402

RUN_FOLDER = Path("/gpfs/projects/ucm96/moritz/rg_workflow/runs/test100")
with (RUN_FOLDER / "samples.json").open("rb") as f:
    SAMPLES = json.load(f)

aggregate_json.main(
    input_paths=[RUN_FOLDER / f"results/rg/{sample}/rg.json" for sample in SAMPLES],
    output_path=RUN_FOLDER / "aggregated.csv",
    added_columns={
        "accession": [SAMPLES[sample]["accession"] for sample in SAMPLES],
        "threshold": [SAMPLES[sample]["threshold"] for sample in SAMPLES],
        "temperature": [SAMPLES[sample]["temperature"] for sample in SAMPLES],
        "ionic_strength": [SAMPLES[sample]["ionic_strength"] for sample in SAMPLES],
    },
    ignore_keys=["file"],
)

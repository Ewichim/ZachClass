"""Small shared helpers for paths, reports, and the OpenMC geometry."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parent


def load_config():
    with (ROOT / "config.yaml").open() as stream:
        return yaml.safe_load(stream)


def output_paths(config):
    output = ROOT / config["output_directory"]
    paths = {"root": output}
    for name in ("reports", "dagmc", "plots", "simulation"):
        paths[name] = output / name
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nuclear_data_path(config):
    return os.environ.get("OPENMC_CROSS_SECTIONS") or config["openmc"].get("cross_sections")


def input_fingerprint(config):
    sources = {}
    for component in config["components"]:
        path = ROOT / config["step_directory"] / component["file"]
        sources[component["file"]] = file_hash(path)
    # Runtime controls and image resolution do not change the CAD or mesh.
    geometry_config = {key: config[key] for key in (
        "step_directory", "components", "units", "inspection", "mesh")}
    digest = hashlib.sha256(json.dumps(geometry_config, sort_keys=True).encode()).hexdigest()
    return {"sources": sources, "config_sha256": digest}


def require_current_inspection(config, paths):
    report = json.loads((paths["reports"] / "step_inspection.json").read_text())
    if report["fingerprint"] != input_fingerprint(config):
        raise RuntimeError("STEP files or config changed. Run inspect_step.py again.")
    if not report["conversion_ready"]:
        repair_path = paths["reports"] / "topology_repair.json"
        if not repair_path.is_file():
            raise RuntimeError("CAD inspection failed. See outputs/reports/geometry_report.md and repair_step.py.")
        repair = json.loads(repair_path.read_text())
        if (not repair["verified"] or repair["fingerprint"] != report["fingerprint"]
                or repair["resolved_errors"] != report["errors"]
                or file_hash(Path(repair["assembly_path"])) != repair["assembly_sha256"]):
            raise RuntimeError("CAD inspection failed and no matching verified topology copy is available.")
    return report


def require_validation(config, paths):
    require_current_inspection(config, paths)
    report = json.loads((paths["reports"] / "dagmc_validation.json").read_text())
    if report["h5m_sha256"] != file_hash(paths["dagmc"] / "tokamak.h5m"):
        raise RuntimeError("H5M changed. Run validate_dagmc.py again.")
    if not report["geometry_checks_passed"]:
        raise RuntimeError("DAGMC validation failed. Read dagmc_validation.json.")
    return report


def archive_outputs(paths, names):
    """Preserve previous results before rerunning a stage."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for name in names:
        folder = paths[name]
        if any(folder.iterdir()):
            destination = paths["root"] / "previous" / stamp / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(folder), destination)
            folder.mkdir()


def build_geometry(config, paths):
    import openmc

    dagmc = openmc.DAGMCUniverse(paths["dagmc"] / "tokamak.h5m")
    universe = dagmc.bounded_universe(
        bounded_type="box", boundary_type="vacuum",
        padding_distance=config["openmc"]["boundary_padding_cm"],
    )
    return openmc.Geometry(universe), dagmc

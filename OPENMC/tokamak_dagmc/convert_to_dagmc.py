# %%
"""Convert only CAD that passed inspection. All source placements are preserved."""

import contextlib
import json
from pathlib import Path
import traceback

from cad_to_dagmc import CadToDagmc
import cad_to_dagmc

from workflow import (ROOT, archive_outputs, file_hash, load_config, output_paths,
                      require_current_inspection, write_json)
from mesh_checks import fix_faceting_tag

config = load_config()
paths = output_paths(config)
mesh = config["mesh"]["profiles"][config["mesh"]["profile"]]


def convert_components(components, destination, inspection):
    converter = CadToDagmc()
    import cadquery as cq

    repair_path = paths["reports"] / "topology_repair.json"
    if repair_path.is_file():
        repair = json.loads(repair_path.read_text())
        if not repair["verified"] or repair["fingerprint"] != inspection["fingerprint"]:
            raise RuntimeError("Topology copy is stale or unverified.")
        if file_hash(Path(repair["assembly_path"])) != repair["assembly_sha256"]:
            raise RuntimeError("Topology copy changed after verification.")
        repaired = cq.Shape.importBrep(repair["assembly_path"])
        # Scale the compound together to preserve its shared face topology.
        repaired = repaired.scale(config["units"]["imported_to_cm"])
        selected_files = [component["file"] for component in components]
        selected_solids = []
        selected_tags = []
        for solid, item in zip(repaired.Solids(), repair["solids"]):
            if item["file"] in selected_files:
                selected_solids.append(solid)
                selected_tags.append(item["tag"])
        converter.add_cadquery_object(cq.Compound.makeCompound(selected_solids), material_tags=selected_tags)
        print("Using separately verified BREP topology copy, with all original placements.")
    else:
        add_original_steps(converter, components, inspection)
    mesh_converter(converter, components, destination)


def add_original_steps(converter, components, inspection):
    counts = {}
    for record in inspection["components"]:
        counts[record["file"]] = record["solid_count"]
    for component in components:
        filename = component["file"]
        count = converter.add_stp_file(
            filename=str(ROOT / config["step_directory"] / filename),
            material_tags=[component["tag"]] * counts[filename],
            scale_factor=config["units"]["imported_to_cm"],
        )
        if count != counts[filename]:
            raise RuntimeError(f"{filename}: imported solid count changed ({count}).")
        print(f"Tagged all {count} solids in {filename}: {component['tag']}", flush=True)


def mesh_converter(converter, components, destination):
    import cadquery as cq

    # Keep an assembled, scaled copy for debugging; the originals are untouched.
    assembled = cq.Compound.makeCompound(converter.parts)
    assembled.exportBrep(str(destination.with_suffix(".brep")))
    from cad_to_dagmc.core import imprint_assembly

    assembly = cq.Assembly()
    for number, solid in enumerate(converter.parts):
        assembly.add(solid, name=str(number))
    imprinted, origins = imprint_assembly(assembly, threads=config["mesh"]["threads"], glue="partial")
    if not imprinted.isValid() or len(imprinted.Solids()) != len(converter.parts):
        raise RuntimeError("Imprinting produced invalid topology or changed the solid count. No mesh exported.")
    if any(len(names) != 1 for names in origins.values()):
        raise RuntimeError("Imprinting mixed input volumes. Review CAD overlaps before meshing.")
    sizing = {}
    for component in components:
        if component["file"] == "first_wall.step":
            sizing[component["tag"]] = mesh["first_wall_cm"]
    # cad_to_dagmc 0.15 changed imprint=True to imprint='partial'/'full'.
    # 'partial' glues shared topology while retaining individual material volumes.
    converter.export_dagmc_h5m_file(
        filename=str(destination), meshing_backend="gmsh", h5m_backend="h5py",
        imprint="partial", imprint_threads=config["mesh"]["threads"],
        threads=config["mesh"]["threads"], method="file", mesh_algorithm=6,
        min_mesh_size=mesh["min_cm"], max_mesh_size=mesh["max_cm"],
        set_size=sizing or None,
    )
    faceting_estimate = fix_faceting_tag(destination)
    write_json(destination.with_suffix(".metadata.json"), {
        "change": "Add FACETING_TOL as a sparse tag on the file meshset; retain the writer's FACETING_TOLERANCE tag.",
        "reason": "DAGMC reads FACETING_TOL; the pinned writer emits a different name on the root set.",
        "faceting_estimate_cm": faceting_estimate, "triangle_coordinates_changed": False,
        "limit": "This value is the writer's extent-based estimate, not a measured maximum CAD-to-mesh deviation.",
        "reference": "https://github.com/svalinn/DAGMC/blob/develop/src/dagmc/DagMC.cpp",
    })


# %%
if __name__ == "__main__":
    inspection = require_current_inspection(config, paths)
    archive_outputs(paths, ["dagmc"])
    result = {"fingerprint": inspection["fingerprint"],
              "cad_to_dagmc_version": cad_to_dagmc.__version__,
              "mesh_profile": config["mesh"]["profile"], "mesh_sizes_cm": mesh,
              "success": False}
    logfile = paths["dagmc"] / "conversion.log"
    try:
        with logfile.open("w") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            convert_components(config["components"], paths["dagmc"] / "tokamak.h5m", inspection)
        result.update(success=True, h5m_sha256=file_hash(paths["dagmc"] / "tokamak.h5m"))
    except Exception as error:
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        # These are diagnostics only. No partial model replaces the full model.
        stages = [
            ["Vacuum", "first_wall", "shielding", "vacuum_vessel"],
            ["Vacuum", "first_wall", "shielding", "vacuum_vessel", "cs_magnet"],
            ["Vacuum", "first_wall", "shielding", "vacuum_vessel", "cs_magnet", "pf_coils"],
        ]
        result["diagnostic_stages"] = []
        for number, tags in enumerate(stages, 1):
            destination = paths["dagmc"] / f"diagnostic_{number}.h5m"
            selected = [component for component in config["components"] if component["tag"] in tags]
            stage = {"files": [component["file"] for component in selected]}
            try:
                with (paths["dagmc"] / f"diagnostic_{number}.log").open("w") as stream:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        convert_components(selected, destination, inspection)
                stage["success"] = True
            except Exception as stage_error:
                stage.update(success=False, error=str(stage_error))
            result["diagnostic_stages"].append(stage)
    write_json(paths["reports"] / "conversion.json", result)
    print(f"Conversion success: {result['success']}; log: {logfile}")
    if not result["success"]:
        raise SystemExit("Full conversion failed. See conversion.json and diagnostic logs.")

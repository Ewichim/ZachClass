# %%
"""Share two coincident interfaces in a BREP copy; do not move CAD surfaces."""

import json

import cadquery as cq
import numpy as np
from OCP.BRepTools import BRepTools_ReShape
from cad_to_dagmc.core import imprint_assembly

from inspect_step import bounding_box, import_shape
from workflow import (ROOT, file_hash, input_fingerprint, load_config,
                      output_paths, write_json)

config = load_config()
paths = output_paths(config)
repair_directory = paths["root"] / "repaired"
repair_directory.mkdir(parents=True, exist_ok=True)
tolerance_mm = config["inspection"]["boolean_tolerance_mm"]


def compare_shells(first, second):
    """Require matching face type, area, bounds, center, and sampled trim edges."""
    if len(first.Faces()) != len(second.Faces()):
        raise RuntimeError("Interface shells have different face counts.")
    unused = second.Faces()
    comparisons = []
    for face in first.Faces():
        matches = []
        for other in unused:
            if face.geomType() != other.geomType():
                continue
            relative_area = abs(face.Area() - other.Area()) / face.Area()
            center_difference = np.linalg.norm(np.array(face.Center().toTuple()) - other.Center().toTuple())
            first_box = bounding_box(face)
            second_box = bounding_box(other)
            bounds_difference = np.max(np.abs(np.array([first_box["min"], first_box["max"]]) -
                                             np.array([second_box["min"], second_box["max"]])))
            if relative_area < 1e-10 and center_difference < tolerance_mm and bounds_difference < tolerance_mm:
                matches.append((other, relative_area, float(bounds_difference)))
        if len(matches) != 1:
            raise RuntimeError("Interface face matching is ambiguous or outside tolerance.")
        other, relative_area, bounds_difference = matches[0]
        maximum_distance = 0.0
        # Check both directions, so matching only a subset of a face is insufficient.
        for a, b in ((face, other), (other, face)):
            boundary = cq.Compound.makeCompound(b.Edges())
            for edge in a.Edges():
                for fraction in np.linspace(0.0, 1.0, 17):
                    point = edge.positionAt(float(fraction))
                    distance = cq.Vertex.makeVertex(*point.toTuple()).distance(boundary)
                    maximum_distance = max(maximum_distance, distance)
        if maximum_distance > tolerance_mm:
            raise RuntimeError("Coincident-face trim boundaries disagree.")
        comparisons.append({"surface_type": face.geomType(), "relative_area_difference": relative_area,
                            "maximum_bbox_difference_mm": bounds_difference,
                            "maximum_sampled_edge_distance_mm": maximum_distance})
        unused.remove(other)
    return comparisons


def repair_interfaces():
    inspection = json.loads((paths["reports"] / "step_inspection.json").read_text())
    fingerprint = input_fingerprint(config)
    if inspection["fingerprint"] != fingerprint:
        raise RuntimeError("Run inspect_step.py on the current inputs first.")
    allowed_error = "Intersection unresolved: Blanket.step:1 / first_wall.step:1."
    if any(error != allowed_error for error in inspection["errors"]):
        raise RuntimeError("This limited repair cannot resolve the other inspection errors.")
    shapes = {}
    for component in config["components"]:
        shapes[component["file"]] = import_shape(ROOT / config["step_directory"] / component["file"])
    blanket = shapes["Blanket.step"].Solids()[0]
    wall = shapes["first_wall.step"].Solids()[0]
    vessel = shapes["Vaccume Vessel.step"].Solids()[0]
    report = {"fingerprint": fingerprint, "verified": False, "source_files_modified": False,
              "operation": "Reuse matching shell topology, with opposite shell orientation across interfaces.",
              "coordinate_transforms": "none", "matching_tolerance_mm": tolerance_mm,
              "interfaces": [], "resolved_errors": []}
    reshape = BRepTools_ReShape()
    # Both blanket shells coincide with existing shells in neighbouring parts.
    for target, reference, description in [
        (blanket.Shells()[1], wall.Shells()[0], "blanket inner / first-wall outer"),
        (blanket.Shells()[0], vessel.Shells()[1], "blanket outer / vessel inner"),
    ]:
        comparison = compare_shells(target, reference)
        reshape.Replace(target.wrapped, reference.wrapped.Reversed())
        report["interfaces"].append({"name": description, "face_comparisons": comparison})
    repaired = cq.Shape.cast(reshape.Apply(blanket.wrapped))
    relative_volume = abs(repaired.Volume() - blanket.Volume()) / blanket.Volume()
    report["blanket_volume_before_mm3"] = blanket.Volume()
    report["blanket_volume_after_mm3"] = repaired.Volume()
    report["blanket_relative_volume_change"] = relative_volume
    if not repaired.isValid() or relative_volume > 1e-10:
        raise RuntimeError("Shell replacement changed the solid or produced invalid topology.")
    shapes["Blanket.step"] = repaired
    for filename in ("first_wall.step", "Vaccume Vessel.step"):
        common = repaired.intersect(shapes[filename], tol=tolerance_mm)
        if not common.isValid() or abs(common.Volume()) > config["inspection"]["overlap_absolute_tolerance_mm3"]:
            raise RuntimeError(f"Repaired interface still has an unresolved intersection: {filename}.")
    all_solids = []
    manifest = []
    assembly = cq.Assembly()
    for component in config["components"]:
        for number, solid in enumerate(shapes[component["file"]].Solids(), 1):
            index = len(all_solids)
            all_solids.append(solid)
            manifest.append({"index": index, "file": component["file"], "solid_number": number,
                             "tag": component["tag"], "volume_mm3": solid.Volume()})
            assembly.add(solid, name=str(index))
    imprinted, origins = imprint_assembly(assembly, threads=config["mesh"]["threads"], glue="partial")
    if not imprinted.isValid() or len(imprinted.Solids()) != len(all_solids):
        raise RuntimeError("Full repaired assembly failed imprint validity or solid-count checks.")
    if any(len(names) != 1 for names in origins.values()):
        raise RuntimeError("Full imprint assigned a volume to more than one original solid.")
    # Verify each body's mass using the imprint's provenance, not only total mass.
    maximum_volume_change = 0.0
    for solid, names in origins.items():
        index = int(names[0].rsplit("/", 1)[-1])
        original_volume = manifest[index]["volume_mm3"]
        maximum_volume_change = max(maximum_volume_change, abs(solid.Volume() - original_volume) / original_volume)
    if maximum_volume_change > 1e-10:
        raise RuntimeError("Imprinting changed an individual solid volume.")
    destination = repair_directory / "assembly_shared_interfaces_mm.brep"
    cq.Compound.makeCompound(all_solids).exportBrep(str(destination))
    repaired.exportBrep(str(repair_directory / "Blanket_shared_interfaces_mm.brep"))
    if input_fingerprint(config) != fingerprint:
        raise RuntimeError("Original inputs changed during repair.")
    report.update(verified=True, resolved_errors=inspection["errors"], solids=manifest,
                  assembly_path=str(destination), assembly_sha256=file_hash(destination),
                  full_imprint_solid_count=len(imprinted.Solids()), full_imprint_valid=True,
                  maximum_individual_volume_change_in_imprint=maximum_volume_change,
                  remaining_overlaps="No positive intersections at the inspection tolerance; formerly unresolved interfaces now return valid zero-volume intersections.",
                  limits="Face matching uses geometric invariants and sampled trim distances. This is a topology repair within the stated tolerance, not a CAD redesign.")
    write_json(paths["reports"] / "topology_repair.json", report)
    print(f"Verified topology copy: {destination}; {len(all_solids)} solids.")
    return report


# %%
if __name__ == "__main__":
    repair_report = repair_interfaces()

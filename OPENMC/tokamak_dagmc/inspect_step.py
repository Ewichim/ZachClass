# %%
"""Inspect original STEP solids before meshing. Lengths below are imported mm."""

import itertools
import json
import math
import re

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.STEPControl import STEPControl_Reader

from workflow import (ROOT, archive_outputs, input_fingerprint, load_config,
                      output_paths, write_json)

config = load_config()
paths = output_paths(config)


def bounding_box(shape):
    box = shape.BoundingBox()
    return {"min": [box.xmin, box.ymin, box.zmin],
            "max": [box.xmax, box.ymax, box.zmax],
            "size": [box.xlen, box.ylen, box.zlen]}


def import_shape(path):
    workplane = cq.importers.importStep(str(path))
    return cq.Compound.makeCompound(workplane.vals())


def declared_units(path):
    contents = path.read_text()
    units = re.findall(r"SI_UNIT\(\s*([^,]+),\s*\.METRE\.\)", contents)
    return sorted(set(units))


def surface_axes(shape):
    """Axes of analytic surfaces help detect translated or rotated exports."""
    axes = []
    for face in shape.Faces():
        surface = BRepAdaptor_Surface(face.wrapped)
        kind = face.geomType()
        if kind == "TORUS":
            axis = surface.Torus().Axis()
        elif kind == "CYLINDER":
            axis = surface.Cylinder().Axis()
        else:
            continue
        axes.append({"surface": kind,
                     "origin_mm": list(axis.Location().Coord()),
                     "direction": list(axis.Direction().Coord())})
    return axes


def inspect_geometry():
    archive_outputs(paths, ["reports"])
    tolerance = config["inspection"]["boolean_tolerance_mm"]
    report = {"fingerprint": input_fingerprint(config), "components": [],
              "pairs": [], "errors": [], "warnings": []}
    solids = []
    reader = STEPControl_Reader()
    # OpenCascade reports the target unit in millimetres per model unit.
    report["reader_system_length_unit_mm"] = reader.SystemLengthUnit()
    if not math.isclose(reader.SystemLengthUnit(), 1.0):
        report["errors"].append("STEP importer is not using mm; review scaling.")
    if config["units"]["imported_length_unit"] != "mm" or not math.isclose(
            config["units"]["imported_to_cm"], 0.1):
        report["errors"].append("Configured scale disagrees with the mm importer.")

    for component in config["components"]:
        filename = component["file"]
        path = ROOT / config["step_directory"] / filename
        record = {"file": filename, "tag": component["tag"],
                  "declared_step_length_prefixes": declared_units(path),
                  "imported_length_unit": "mm", "solids": []}
        report["components"].append(record)
        try:
            shape = import_shape(path)
            record.update(import_success=True, solid_count=len(shape.Solids()),
                          shell_count=len(shape.Shells()), face_count=len(shape.Faces()),
                          bbox_mm=bounding_box(shape),
                          center_of_mass_mm=list(shape.Center().toTuple()),
                          analytic_surface_axes=surface_axes(shape))
            solid_shells = sum(len(s.Shells()) for s in shape.Solids())
            record["unattached_shell_count"] = len(shape.Shells()) - solid_shells
            if not shape.isValid() or record["unattached_shell_count"] or not shape.Solids():
                report["errors"].append(f"{filename}: invalid shape or unattached shells.")
            for number, solid in enumerate(shape.Solids(), 1):
                closed = all(BRep_Tool.IsClosed_s(s.wrapped) for s in solid.Shells())
                faces = solid.Faces()
                areas = [face.Area() for face in faces]
                lengths = [edge.Length() for edge in solid.Edges()]
                invalid_faces = [i for i, face in enumerate(faces, 1) if not face.isValid()]
                item = {"id": f"{filename}:{number}", "number": number,
                        "valid": solid.isValid(), "closed_shells": closed,
                        "volume_mm3": solid.Volume(), "bbox_mm": bounding_box(solid),
                        "center_of_mass_mm": list(solid.Center().toTuple()),
                        "face_count": len(faces), "invalid_face_ids": invalid_faces,
                        "minimum_face_area_mm2": min(areas),
                        "minimum_edge_length_mm": min(lengths),
                        "small_face_ids": [i for i, a in enumerate(areas, 1)
                                           if a < config["inspection"]["small_face_mm2"]],
                        "small_edge_count": sum(0 < length < config["inspection"]["small_edge_mm"]
                                                for length in lengths),
                        # For a broad thin sheet, 2 V / A is a thickness estimate.
                        "thickness_proxy_mm": 2 * solid.Volume() / solid.Area()}
                record["solids"].append(item)
                solids.append((solid, item, component))
                if not item["valid"] or not closed or invalid_faces or solid.Volume() <= 0:
                    report["errors"].append(f"{item['id']}: invalid/open/zero-volume solid.")
                if item["small_face_ids"] or item["small_edge_count"]:
                    report["warnings"].append(f"{item['id']}: small features; inspect before fine meshing.")
            record["total_volume_mm3"] = sum(s["volume_mm3"] for s in record["solids"])
            print(f"{filename}: {record['solid_count']} solids, "
                  f"{record['total_volume_mm3'] / 1e9:.6g} m3", flush=True)
        except Exception as error:
            record.update(import_success=False, error=str(error))
            report["errors"].append(f"{filename}: {error}")

    print("Checking bounding-box candidates with OpenCascade intersections...", flush=True)
    for (first, a, ca), (second, b, cb) in itertools.combinations(solids, 2):
        dimensions = np.minimum(a["bbox_mm"]["max"], b["bbox_mm"]["max"]) - np.maximum(
            a["bbox_mm"]["min"], b["bbox_mm"]["min"])
        if np.any(dimensions < -tolerance):
            continue
        pair = {"first": a["id"], "second": b["id"],
                "different_tags": ca["tag"] != cb["tag"]}
        try:
            intersection = first.intersect(second, tol=tolerance)
            volume = sum(abs(s.Volume()) for s in intersection.Solids())
            threshold = max(config["inspection"]["overlap_absolute_tolerance_mm3"],
                            config["inspection"]["overlap_relative_tolerance"] *
                            min(a["volume_mm3"], b["volume_mm3"]))
            if not intersection.isValid() or volume > min(a["volume_mm3"], b["volume_mm3"]) + threshold:
                pair.update(boolean_valid=False, raw_signed_volume_mm3=intersection.Volume(),
                            raw_absolute_volume_mm3=volume)
                raise RuntimeError("OpenCascade returned an invalid intersection; its volume is not a physical overlap measurement.")
            pair.update(intersection_mm3=volume, tolerance_mm3=threshold,
                        boolean_valid=True, overlap=volume > threshold)
            if pair["overlap"]:
                smaller = min(a["volume_mm3"], b["volume_mm3"])
                pair["fraction_of_smaller"] = volume / smaller
                pair["duplicate_candidate"] = all(
                    abs(volume - s["volume_mm3"]) < threshold for s in (a, b))
                report["errors"].append(
                    f"Overlap {a['id']} / {b['id']}: {volume / 1e3:.6g} cm3.")
            report["pairs"].append(pair)
        except Exception as error:
            pair["error"] = str(error)
            report["pairs"].append(pair)
            report["errors"].append(f"Intersection unresolved: {a['id']} / {b['id']}.")

    if solids:
        assembly = cq.Compound.makeCompound([s[0] for s in solids])
        report["assembly_bbox_mm"] = bounding_box(assembly)
        diameter_m = max(report["assembly_bbox_mm"]["size"][:2]) / 1000
        report["overall_diameter_m"] = diameter_m
        low, high = config["units"]["expected_overall_diameter_m"]
        if not low <= diameter_m <= high:
            report["errors"].append("Overall dimensions fail configured metre-scale plausibility check.")
        report["alignment_note"] = (
            "All placements retained. Analytic surface axes, bounding boxes, equatorial "
            "radial samples, and angular probes are recorded; these are diagnostic checks, "
            "not proof of the intended Fusion dimensions.")

    # Sample the layer sequence in the original equatorial plane (z=0).
    radius_mm = max(report.get("assembly_bbox_mm", {"size": [0]})["size"]) / 2
    samples = []
    for radius in np.arange(0.0, radius_mm + 2, 2.0):
        inside = []
        for solid, item, component in solids:
            if solid.isInside((float(radius), 0, 0), tolerance):
                inside.append(item["id"])
        if not samples or samples[-1]["inside"] != inside:
            samples.append({"from_radius_mm": float(radius), "inside": inside})
    report["equatorial_radial_transitions_2mm_grid"] = samples

    # For each axisymmetric component, find an interior point in its xz section
    # and rotate it through 360 degrees. Discrete TF coils are checked separately.
    for record in report["components"]:
        members = [s[0] for s in solids if s[2]["file"] == record["file"]]
        if not members:
            continue
        bb = record["bbox_mm"]
        seed = None
        for z in np.linspace(bb["min"][2], bb["max"][2], 9)[1:-1]:
            for radius in np.linspace(0, max(bb["max"][:2]), 300)[1:]:
                if any(s.isInside((float(radius), 0, float(z)), tolerance) for s in members):
                    seed = [float(radius), 0, float(z)]
                    break
            if seed:
                break
        if seed:
            angles = np.linspace(0, 2 * math.pi, config["inspection"]["sample_angles"], endpoint=False)
            count = 0
            for angle in angles:
                point = (seed[0] * math.cos(angle), seed[0] * math.sin(angle), seed[2])
                count += any(s.isInside(point, tolerance) for s in members)
            record["angular_probe"] = {"seed_mm": seed, "inside_count": count,
                                       "angle_count": len(angles)}
            if record["tag"] != "tf_coils" and count != len(angles):
                report["errors"].append(f"{record['file']}: partial torus or axis mismatch at angular probe.")
        else:
            report["warnings"].append(f"{record['file']}: no interior angular-probe seed found.")

    report["conversion_ready"] = not report["errors"]
    write_json(paths["reports"] / "step_inspection.json", report)
    write_markdown(report)
    print(f"Inspection complete: {len(report['errors'])} blocking issues.", flush=True)
    return report


def write_markdown(report):
    lines = ["# STEP geometry inspection", "", "Original STEP files were not modified.", "",
             f"Conversion precheck passed: **{report['conversion_ready']}**.", "",
             "## Components", "", "| File | Solids | Shells | Faces | Volume (m³) | Tag |",
             "|---|---:|---:|---:|---:|---|"]
    for component in report["components"]:
        if component["import_success"]:
            lines.append(f"| {component['file']} | {component['solid_count']} | "
                         f"{component['shell_count']} | {component['face_count']} | "
                         f"{component['total_volume_mm3'] / 1e9:.7g} | {component['tag']} |")
    lines += ["", "## Units and placement", "",
              "STEP length prefixes are recorded per file (`$` means metres). "
              "The OpenCascade importer target is mm; imported coordinates are multiplied "
              "by 0.1 once to obtain cm. No component is translated or rotated.", "",
              f"Assembly bounding box (mm): `{report.get('assembly_bbox_mm')}`.", "",
              "Full-turn angular probes and analytic surface axes are in the JSON. "
              "A successful probe checks the sampled circle; it does not prove every feature is complete.", "",
              "## Blocking issues", ""]
    lines += [f"- {error}" for error in report["errors"]] or ["None found at the stated tolerances."]
    lines += ["", "## Warnings", ""]
    lines += [f"- {warning}" for warning in report["warnings"]] or ["None."]
    lines += ["", "## Numerical limits", "",
              f"Boolean fuzzy tolerance: {config['inspection']['boolean_tolerance_mm']} mm. "
              f"Overlap threshold: max({config['inspection']['overlap_absolute_tolerance_mm3']} mm³, "
              f"{config['inspection']['overlap_relative_tolerance']} × smaller solid volume). "
              "Every bounding-box candidate is intersected, including same-material pairs. "
              "The equatorial layer scan uses 2 mm spacing and can miss narrower gaps. "
              "The 2V/A thickness proxy is an average sheet estimate, not a minimum thickness. "
              "Small topological edges/faces are recorded; smooth thin regions still need mesh convergence.", "",
              "## Next steps", "",
              "If overlaps are listed, resolve the named bodies in Fusion using explicit cuts "
              "or correct their placement, then re-export only into a separate reviewed input set. "
              "No automatic subtraction or geometric repair is performed here. "
              "DAGMC mesh checks and OpenMC transport are separate later stages."]
    (paths["reports"] / "geometry_report.md").write_text("\n".join(lines) + "\n")


# %%
if __name__ == "__main__":
    inspection = inspect_geometry()

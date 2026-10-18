# %%
"""Check the full DAGMC mesh before plotting or transport."""

from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import openmc
import openmc.lib

from materials import make_materials
from mesh_checks import check_volume_mesh, export_surface_vtk, read_mesh
from workflow import (build_geometry, file_hash, load_config, output_paths,
                      require_current_inspection, write_json)

config = load_config()
paths = output_paths(config)


def run_check(command, log_path):
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command}
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
        text = completed.stdout + completed.stderr
        log_path.write_text(text)
        return {"available": True, "command": command,
                "returncode": completed.returncode, "log": str(log_path)}
    except subprocess.TimeoutExpired:
        return {"available": True, "command": command, "error": "Timed out after 180 seconds."}


def validate():
    filename = paths["dagmc"] / "tokamak.h5m"
    report = {"h5m_exists": filename.is_file(), "h5m_sha256": None,
              "geometry_checks_passed": False, "transport_validated": False, "errors": []}
    if not filename.is_file():
        report["errors"].append("Full-model tokamak.h5m is absent. CAD prechecks must pass before conversion.")
        write_json(paths["reports"] / "dagmc_validation.json", report)
        print(report["errors"][0])
        return report
    report["h5m_sha256"] = file_hash(filename)
    try:
        inspection = require_current_inspection(config, paths)
        geometry, dagmc = build_geometry(config, paths)
        report.update(python_h5m_readable=True, n_cells_including_implicit_complement=dagmc.n_cells,
                      n_surfaces=dagmc.n_surfaces, material_names=dagmc.material_names,
                      bounding_box_cm=[list(dagmc.bounding_box.lower_left), list(dagmc.bounding_box.upper_right)],
                      openmc_dagmc_enabled=openmc.lib._dagmc_enabled())
        coordinates, triangles, volumes, senses = read_mesh(filename)
        checks = [check_volume_mesh(coordinates, volume) for volume in volumes]
        report["volume_mesh_checks"] = checks
        for check in checks:
            if check["nonmanifold_or_open_edges"] or check["degenerate_triangles"] or check["signed_volume_cm3"] <= 0:
                report["errors"].append(f"Volume {check['id']}: unsealed, degenerate, or inverted mesh.")
        actual = Counter(v["material"] for v in volumes)
        expected = Counter()
        cad_volume_cm3 = Counter()
        for component in inspection["components"]:
            expected[component["tag"]] += component["solid_count"]
            cad_volume_cm3[component["tag"]] += component["total_volume_mm3"] * config["units"]["imported_to_cm"]**3
        report["expected_volume_counts_by_tag"] = dict(expected)
        report["actual_volume_counts_by_tag"] = dict(actual)
        if actual != expected:
            report["errors"].append("Material groups or per-component solid counts disagree with STEP inspection.")
        if len(set(v["id"] for v in volumes)) != len(volumes) or any(v["id"] == 10000 for v in volumes):
            report["errors"].append("DAGMC volume IDs are duplicated or collide with the CSG boundary cell.")
        if dagmc.n_surfaces >= 10000:
            report["errors"].append("Review CSG surface IDs before exporting this large model.")
        report["volumes_by_material"] = {}
        for tag, cad_volume in cad_volume_cm3.items():
            mesh_volume = sum(v["signed_volume_cm3"] for v in checks if v["material"] == tag)
            relative_error = abs(mesh_volume - cad_volume) / cad_volume
            report["volumes_by_material"][tag] = {"cad_cm3": cad_volume, "mesh_cm3": mesh_volume,
                                                   "relative_error": relative_error}
            if relative_error > 0.02:
                report["errors"].append(f"{tag}: mesh volume differs from CAD by more than 2%.")
        bb = inspection["assembly_bbox_mm"]
        expected_bounds = np.array([bb["min"], bb["max"]]) * config["units"]["imported_to_cm"]
        difference = np.abs(np.array(report["bounding_box_cm"]) - expected_bounds)
        report["maximum_bbox_difference_cm"] = float(difference.max())
        if difference.max() > 1.0:
            report["errors"].append("Mesh bounds differ from the scaled CAD bounds by more than 1 cm.")
        # Run tools on a disposable copy, preserving the production mesh.
        check_copy = paths["reports"] / "mesh_check_copy.h5m"
        shutil.copy2(filename, check_copy)
        report["watertight_tool"] = run_check(["check_watertight", str(check_copy)], paths["reports"] / "watertight.log")
        report["overlap_tool"] = run_check(["overlap_check", "-p", "1", "-t", "2", str(check_copy)], paths["reports"] / "mesh_overlap.log")
        for key in ("watertight_tool", "overlap_tool"):
            tool = report[key]
            if not tool.get("available") or tool.get("returncode") != 0:
                report["errors"].append(f"{key}: unavailable or failed; review its log.")
        overlap_log = paths["reports"] / "mesh_overlap.log"
        if not overlap_log.is_file() or "No overlaps were found." not in overlap_log.read_text():
            report["errors"].append("DAGMC overlap checker did not explicitly report no overlaps.")
        if not report["openmc_dagmc_enabled"]:
            report["errors"].append("OpenMC's C API lacks DAGMC support.")
        report["geometry_checks_passed"] = not report["errors"]
        report["limits"] = ("Edge closure, signed mesh volumes, CAD intersections and the non-exhaustive "
                            "DAGMC overlap tool are complementary checks. Mesh convergence and transport "
                            "remain separate requirements. Python H5M metadata access is not C++ transport initialization.")
        if report["geometry_checks_passed"]:
            geometry.export_to_xml(paths["simulation"] / "geometry.xml")
            make_materials(config).export_to_xml(paths["simulation"] / "materials.xml")
            export_surface_vtk(paths["plots"] / "tokamak_surfaces.vtk", coordinates, volumes)
    except Exception as error:
        report["errors"].append(str(error))
    report["geometry_checks_passed"] = not report["errors"]
    write_json(paths["reports"] / "dagmc_validation.json", report)
    print(json.dumps(report, indent=2))
    return report


# %%
if __name__ == "__main__":
    validation = validate()
    if not validation["geometry_checks_passed"]:
        raise SystemExit("Full DAGMC geometry did not pass validation.")

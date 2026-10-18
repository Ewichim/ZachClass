# %%
"""Record Boolean/interface failures and convert safe subsets for diagnosis."""

import contextlib
import json

import cadquery as cq
from cad_to_dagmc.core import imprint_assembly

from convert_to_dagmc import convert_components
from inspect_step import import_shape
from workflow import ROOT, load_config, output_paths, write_json

config = load_config()
paths = output_paths(config)


def diagnose():
    inspection = json.loads((paths["reports"] / "step_inspection.json").read_text())
    result = {"source_files_modified": False, "repairs_applied": False,
              "interface_checks": [], "subset_conversions": []}
    filenames = ["Blanket.step", "first_wall.step"]
    solids = []
    assembly = cq.Assembly()
    for filename in filenames:
        solid = import_shape(ROOT / config["step_directory"] / filename).Solids()[0]
        solids.append(solid)
        assembly.add(solid, name=filename)
    for tolerance in (1e-7, 1e-5, 0.01):
        common = solids[0].intersect(solids[1], tol=tolerance)
        result["interface_checks"].append({
            "method": "OpenCascade common", "files": filenames,
            "fuzzy_tolerance_mm": tolerance, "result_valid": common.isValid(),
            "signed_solid_volumes_mm3": [s.Volume() for s in common.Solids()],
        })
    imprinted, mapping = imprint_assembly(assembly, threads=2, glue="partial")
    result["partial_imprint"] = {"valid": imprinted.isValid(),
                                 "solid_count": len(imprinted.Solids()),
                                 "signed_solid_volumes_mm3": [s.Volume() for s in imprinted.Solids()]}
    # This invalid result is kept only as a BREP for troubleshooting.
    imprinted.exportBrep(str(paths["dagmc"] / "INVALID_blanket_wall_imprint_mm.brep"))
    for name, files in [
        ("diagnostic_plasma", ["Plamsa.step"]),
        ("diagnostic_plasma_first_wall", ["Plamsa.step", "first_wall.step"]),
    ]:
        selected = [c for c in config["components"] if c["file"] in files]
        stage = {"name": name, "files": files, "full_model": False}
        try:
            with (paths["dagmc"] / f"{name}.log").open("w") as stream:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    convert_components(selected, paths["dagmc"] / f"{name}.h5m", inspection)
            stage["success"] = True
        except Exception as error:
            stage.update(success=False, error=str(error))
        result["subset_conversions"].append(stage)
        print(stage, flush=True)
    write_json(paths["reports"] / "interface_diagnostics.json", result)
    return result


# %%
if __name__ == "__main__":
    diagnosis = diagnose()

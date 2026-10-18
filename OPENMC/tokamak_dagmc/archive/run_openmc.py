# %%
"""Small fixed-source test. Geometry validation is required before input export."""

from datetime import datetime, timezone
import contextlib
import json
import math
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmc

from materials import make_materials
from mesh_checks import read_mesh
from plot_geometry import make_plots
from workflow import (ROOT, build_geometry, load_config, output_paths,
                      file_hash, nuclear_data_path, require_validation, write_json)

config = load_config()
paths = output_paths(config)
run_parameters = config["openmc"]


def check_nuclear_data(materials):
    filename = nuclear_data_path(config)
    report = {"OPENMC_CROSS_SECTIONS": os.environ.get("OPENMC_CROSS_SECTIONS"),
              "selected_existing_index": filename, "missing": []}
    if not filename or not Path(filename).is_file():
        report["missing"].append("A readable nuclear-data cross_sections.xml index.")
        return report
    library = openmc.data.DataLibrary.from_xml(filename)
    for material in materials:
        for nuclide in material.get_nuclides():
            entry = library.get_by_material(nuclide)
            if entry is None or not Path(entry["path"]).is_file():
                report["missing"].append(nuclide)
    return report


def prepare_model():
    require_validation(config, paths)
    materials = make_materials(config)
    geometry, dagmc = build_geometry(config, paths)
    inspection = json.loads((paths["reports"] / "step_inspection.json").read_text())
    plasma = next(component for component in inspection["components"] if component["file"] == "Plamsa.step")
    lower = np.array(plasma["bbox_mm"]["min"]) * config["units"]["imported_to_cm"]
    upper = np.array(plasma["bbox_mm"]["max"]) * config["units"]["imported_to_cm"]
    center = (lower + upper) / 2
    radius = max(upper[0] - lower[0], upper[1] - lower[1]) / 2
    coordinates, triangles, volumes, senses = read_mesh(paths["dagmc"] / "tokamak.h5m")
    plasma_ids = [volume["id"] for volume in volumes if volume["material"] == plasma["tag"]]

    # r dr dphi dz is uniform in cylinder volume. Reject points outside the
    # actual DAGMC plasma cell, producing a distributed toroidal neutron source.
    space = openmc.stats.CylindricalIndependent(
        r=openmc.stats.PowerLaw(0.0, radius, 1.0),
        phi=openmc.stats.Uniform(0.0, 2 * math.pi),
        z=openmc.stats.Uniform(lower[2], upper[2]),
        origin=(center[0], center[1], 0.0),
    )
    # These cells carry IDs only for source-domain serialization, not CSG regions.
    plasma_domains = [openmc.Cell(cell_id=cell_id) for cell_id in plasma_ids]
    source = openmc.IndependentSource(
        space=space, angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([14.1e6], [1.0]),
        constraints={"domains": plasma_domains, "rejection_strategy": "resample"},
    )
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = source
    settings.particles = run_parameters["particles"]
    settings.batches = run_parameters["batches"]
    settings.seed = run_parameters["seed"]
    settings.max_lost_particles = 1
    settings.rel_max_lost_particles = 1e-9
    settings.statepoint = {"batches": [settings.batches]}
    mesh = openmc.RegularMesh()
    mesh.dimension = run_parameters["mesh_dimension"]
    mesh.lower_left = coordinates.min(axis=0)
    mesh.upper_right = coordinates.max(axis=0)
    tally = openmc.Tally(name="neutron_flux")
    tally.filters = [openmc.MeshFilter(mesh), openmc.ParticleFilter(["neutron"])]
    tally.scores = ["flux"]
    model = openmc.Model(geometry, materials, settings, openmc.Tallies([tally]))

    run_name = datetime.now(timezone.utc).strftime("test_%Y%m%dT%H%M%S%fZ")
    case = paths["simulation"] / "cases" / run_name
    case.mkdir(parents=True)
    shutil.copy2(paths["dagmc"] / "tokamak.h5m", case / "tokamak.h5m")
    dagmc.filename = case / "tokamak.h5m"
    shutil.copy2(ROOT / "config.yaml", case / "config.yaml")
    model.export_to_xml(case)
    for name in ("geometry.xml", "materials.xml", "settings.xml", "tallies.xml"):
        shutil.copy2(case / name, paths["simulation"] / name)
    data_report = check_nuclear_data(materials)
    write_json(paths["reports"] / "nuclear_data.json", data_report)
    plot_report_path = paths["reports"] / "openmc_plots.json"
    plot_report = json.loads(plot_report_path.read_text()) if plot_report_path.is_file() else {}
    if plot_report.get("h5m_sha256") == file_hash(paths["dagmc"] / "tokamak.h5m"):
        shutil.copy2(paths["plots"] / "tokamak_xz.png", case / "setup_xz.png")
    else:
        make_plots(geometry, materials, paths["dagmc"] / "tokamak.h5m", case / "setup_plots")
    source_report = {"plasma_cell_ids": plasma_ids, "proposal_radius_cm": radius,
                     "proposal_z_cm": [float(lower[2]), float(upper[2])],
                     "spatial_density": "uniform in DAGMC plasma volume, by cylindrical rejection",
                     "neutron_energy_eV": 14.1e6, "isotropic": True,
                     "approximations": "Uniform emissivity; no temperature or density profile."}
    write_json(case / "source_parameters.json", source_report)
    return model, case, mesh, data_report


# %%
# Model setup and export.
if __name__ == "__main__":
    try:
        model, case_directory, flux_mesh, nuclear_data = prepare_model()
    except (RuntimeError, FileNotFoundError) as error:
        write_json(paths["reports"] / "transport.json", {"completed": False, "reason": str(error)})
        raise SystemExit(str(error))


# %%
# Dedicated transport cell. Set run_transport in config.yaml to export only.
if __name__ == "__main__":
    statepoint_path = None
    transport_report = {"completed": False, "case_directory": str(case_directory),
                        "particles_per_batch": model.settings.particles,
                        "batches": model.settings.batches,
                        "geometry_debug": run_parameters["geometry_debug"]}
    if run_parameters["run_transport"] and not nuclear_data["missing"] and shutil.which("openmc"):
        try:
            with (case_directory / "openmc.log").open("w") as log, contextlib.redirect_stdout(log):
                statepoint_path = model.run(cwd=case_directory, threads=run_parameters["threads"],
                                            geometry_debug=run_parameters["geometry_debug"])
            transport_report["completed"] = True
        except RuntimeError as error:
            transport_report["reason"] = str(error)
    else:
        transport_report["reason"] = "Transport disabled, executable absent, or nuclear data missing; XML exported."
    write_json(paths["reports"] / "transport.json", transport_report)
    write_json(case_directory / "run_summary.json", transport_report)
    print(f"Transport completed: {transport_report['completed']}; outputs: {case_directory}")
    if transport_report.get("reason"):
        print(transport_report["reason"])


# %%
# Analysis cell. Flux is per source neutron, with no assumed fusion power.
if __name__ == "__main__" and statepoint_path is not None:
    with openmc.StatePoint(statepoint_path) as statepoint:
        tally = statepoint.get_tally(name="neutron_flux")
        table = tally.get_pandas_dataframe()
        table.to_csv(case_directory / "flux_tracklength_per_source.csv", index=False)
        volume_cm3 = np.prod(flux_mesh.width)
        flux = tally.mean.ravel().reshape(tuple(flux_mesh.dimension), order="F") / volume_cm3
    figure, axis = plt.subplots()
    image = axis.imshow(flux[:, :, flux.shape[2] // 2].T, origin="lower",
                        extent=[flux_mesh.lower_left[0], flux_mesh.upper_right[0],
                                flux_mesh.lower_left[1], flux_mesh.upper_right[1]])
    axis.set_xlabel("x (cm)")
    axis.set_ylabel("y (cm)")
    figure.colorbar(image, ax=axis, label="Flux (cm⁻² per source neutron)")
    figure.savefig(case_directory / "flux_xy.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    shutil.copy2(case_directory / "flux_tracklength_per_source.csv", paths["simulation"] / "flux_tracklength_per_source.csv")
    shutil.copy2(case_directory / "flux_xy.png", paths["simulation"] / "flux_xy.png")

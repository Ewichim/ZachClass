# %%
"""First-wall neutron flux in the validated tokamak, with a timed batch loop."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import openmc
import openmc.lib

try:
    STUDY_DIRECTORY = Path(__file__).resolve().parent
except NameError:
    # VS Code may start the notebook kernel at the workspace root.
    STUDY_DIRECTORY = Path.cwd()
    if (STUDY_DIRECTORY / "OPENMC" / "first_wall_flux").is_dir():
        STUDY_DIRECTORY = STUDY_DIRECTORY / "OPENMC" / "first_wall_flux"
    elif (STUDY_DIRECTORY / "first_wall_flux").is_dir():
        STUDY_DIRECTORY = STUDY_DIRECTORY / "first_wall_flux"

GEOMETRY_DIRECTORY = STUDY_DIRECTORY.parent / "tokamak_dagmc"
sys.path.insert(0, str(GEOMETRY_DIRECTORY))
from materials import make_materials
from workflow import (file_hash, load_config, nuclear_data_path, output_paths,
                      require_validation, write_json)


# %%
# Simulation parameters. Running all cells starts the timed calculation.
target_seconds = 60 * 60
particles_per_batch = 5000
maximum_batches = 10000
minimum_batches = 2  # At least two realizations are needed for an uncertainty.
threads = 2
seed = 42
checkpoint_seconds = 10 * 60
source_strength_n_per_s = None  # Set only if a physical neutron rate is known.
run_transport = True

energy_edges_eV = np.concatenate(([0.0], np.geomspace(1e-5, 1e8, 101)))


# %%
# Model setup: validated geometry, source, settings, and first-wall tallies.
def prepare_model(duration_seconds, batch_particles, batch_limit, thread_count,
                  source_strength=None, run_label="first_wall"):
    if duration_seconds <= 0 or batch_particles <= 0 or batch_limit < minimum_batches:
        raise ValueError("Use a positive duration/particle count and at least two batches.")
    if thread_count <= 0 or (source_strength is not None and source_strength <= 0):
        raise ValueError("Thread count and any supplied source strength must be positive.")
    if not openmc.lib._dagmc_enabled():
        raise RuntimeError("This OpenMC installation does not support DAGMC.")

    config = load_config()
    geometry_paths = output_paths(config)
    validation = require_validation(config, geometry_paths)
    inspection = json.loads((geometry_paths["reports"] / "step_inspection.json").read_text())
    wall = next(c for c in config["components"] if c["file"] == "first_wall.step")
    plasma = next(c for c in inspection["components"] if c["file"] == "Plamsa.step")
    wall_mesh = [v for v in validation["volume_mesh_checks"] if v["material"] == wall["tag"]]
    plasma_ids = [v["id"] for v in validation["volume_mesh_checks"] if v["material"] == plasma["tag"]]
    if not wall_mesh or not plasma_ids:
        raise RuntimeError("The validated mesh is missing the first wall or plasma.")
    wall_volume_cm3 = sum(v["signed_volume_cm3"] for v in wall_mesh)
    if wall_volume_cm3 <= 0:
        raise RuntimeError("The first-wall mesh has a nonpositive volume.")
    if sum(c["tag"] == wall["tag"] for c in config["components"]) != 1:
        raise RuntimeError("The first-wall material tag must be unique to the first wall.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = STUDY_DIRECTORY / "outputs" / f"{run_label}_{stamp}"
    case_directory = run_directory / "cases" / "dt_14p1MeV"
    case_directory.mkdir(parents=True)
    mesh_file = case_directory / "tokamak.h5m"
    shutil.copy2(geometry_paths["dagmc"] / "tokamak.h5m", mesh_file)
    if file_hash(mesh_file) != validation["h5m_sha256"]:
        raise RuntimeError("The copied mesh does not match the validated geometry.")

    # Materials retain the six provisional compositions from the geometry study.
    materials = make_materials(config)
    wall_material = next(material for material in materials if material.name == wall["tag"])
    cross_sections = nuclear_data_path(config)
    if not cross_sections or not Path(cross_sections).is_file():
        raise RuntimeError("The existing nuclear-data index is unavailable.")
    library = openmc.data.DataLibrary.from_xml(cross_sections)
    for material in materials:
        for nuclide in material.get_nuclides():
            entry = library.get_by_material(nuclide)
            if entry is None or not Path(entry["path"]).is_file():
                raise RuntimeError(f"Nuclear data are missing for {nuclide}.")

    # The immutable case copy keeps this run reproducible after later remeshing.
    dagmc = openmc.DAGMCUniverse(mesh_file)
    geometry = openmc.Geometry(dagmc.bounded_universe(
        bounded_type="box", boundary_type="vacuum",
        padding_distance=config["openmc"]["boundary_padding_cm"],
    ))

    # A uniform cylinder proposal, rejected outside the actual toroidal plasma.
    lower = np.array(plasma["bbox_mm"]["min"]) * config["units"]["imported_to_cm"]
    upper = np.array(plasma["bbox_mm"]["max"]) * config["units"]["imported_to_cm"]
    center = (lower + upper) / 2
    radius_cm = max(upper[0] - lower[0], upper[1] - lower[1]) / 2
    space = openmc.stats.CylindricalIndependent(
        r=openmc.stats.PowerLaw(0.0, radius_cm, 1.0),
        phi=openmc.stats.Uniform(0.0, 2 * math.pi),
        z=openmc.stats.Uniform(lower[2], upper[2]),
        origin=(center[0], center[1], 0.0),
    )
    source_domains = [openmc.Cell(cell_id=cell_id) for cell_id in plasma_ids]
    source = openmc.IndependentSource(
        space=space, angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([14.1e6], [1.0]),
        constraints={"domains": source_domains, "rejection_strategy": "resample"},
    )

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = source
    settings.particles = int(batch_particles)
    settings.batches = int(batch_limit)
    settings.seed = seed
    settings.max_lost_particles = 1
    settings.rel_max_lost_particles = 1e-9
    settings.output = {"path": str(case_directory), "summary": True, "tallies": False}
    settings.statepoint = {"batches": [int(batch_limit)]}
    settings.sourcepoint = {"write": False}

    # One material bin collects every first-wall solid into the same realization.
    # This preserves covariance if the wall is later split into multiple solids.
    wall_filter = openmc.MaterialFilter([wall_material])
    neutron_filter = openmc.ParticleFilter(["neutron"])
    total_flux = openmc.Tally(name="first_wall_total_flux")
    total_flux.filters = [wall_filter, neutron_filter]
    total_flux.scores = ["flux"]
    total_flux.estimator = "tracklength"
    spectrum = openmc.Tally(name="first_wall_energy_flux")
    spectrum.filters = [wall_filter, neutron_filter, openmc.EnergyFilter(energy_edges_eV)]
    spectrum.scores = ["flux"]
    spectrum.estimator = "tracklength"
    model = openmc.Model(geometry, materials, settings, openmc.Tallies([total_flux, spectrum]))
    model.export_to_xml(case_directory)

    metadata = {
        "status": "prepared", "target_seconds": duration_seconds,
        "particles_per_batch": batch_particles, "maximum_batches": batch_limit,
        "minimum_batches": minimum_batches, "threads": thread_count, "seed": seed,
        "checkpoint_seconds": checkpoint_seconds, "openmc_version": openmc.__version__,
        "python_executable": sys.executable, "h5m_sha256": validation["h5m_sha256"],
        "first_wall_tag": wall["tag"], "first_wall_material_id": wall_material.id,
        "first_wall_cell_ids": [v["id"] for v in wall_mesh],
        "first_wall_mesh_volume_cm3": wall_volume_cm3,
        "plasma_cell_ids": plasma_ids, "proposal_radius_cm": radius_cm,
        "proposal_z_cm": [float(lower[2]), float(upper[2])],
        "source_energy_eV": 14.1e6, "source_isotropic": True,
        "source_strength_n_per_s": source_strength, "cross_sections": cross_sections,
        "geometry_debug": False, "case_directory": str(case_directory),
        "normalization": "track-length tally / actual first-wall mesh volume; per source neutron",
        "assumptions": "Uniform plasma emissivity; provisional materials and coarse CAD surface mesh.",
    }
    write_json(run_directory / "run_summary.json", metadata)
    for name in ("config.yaml", "first_wall_flux.py"):
        source_file = GEOMETRY_DIRECTORY / name if name == "config.yaml" else STUDY_DIRECTORY / name
        shutil.copy2(source_file, run_directory / ("geometry_config.yaml" if name == "config.yaml" else name))
    for name in ("dagmc_validation.json", "topology_repair.json"):
        shutil.copy2(geometry_paths["reports"] / name, run_directory / name)
    make_setup_plot(geometry_paths, run_directory, validation["h5m_sha256"])
    return model, run_directory, metadata


# %%
def make_setup_plot(geometry_paths, run_directory, mesh_sha256):
    plot_report = json.loads((geometry_paths["reports"] / "openmc_plots.json").read_text())
    if plot_report.get("h5m_sha256") != mesh_sha256:
        raise RuntimeError("Regenerate the geometry plots for the current mesh first.")
    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    for axis, name, title in zip(axes, ("tokamak_xz.png", "poloidal_zoom.png"),
                                 ("Full tokamak", "First-wall layers")):
        axis.imshow(mpimg.imread(geometry_paths["plots"] / name))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("First-wall tally: red region; 14.1 MeV isotropic source in the cyan plasma")
    figure.savefig(run_directory / "setup.png", dpi=140, bbox_inches="tight")
    plt.close(figure)


# %%
def run_timed_transport(model, run_directory, metadata):
    if openmc.lib.is_initialized:
        raise RuntimeError("Finalize the existing OpenMC library session before starting this study.")
    case = Path(metadata["case_directory"])
    statepoint = case / "statepoint.final.h5"
    started = time.monotonic()
    last_checkpoint = started
    metadata["status"] = "running"
    write_json(run_directory / "run_summary.json", metadata)
    history_path = run_directory / "flux_history.csv"
    try:
        with history_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["batch", "source_histories", "elapsed_seconds", "flux_per_source_cm2",
                             "std_dev_per_source_cm2", "relative_std_dev"])
            with openmc.lib.run_in_memory(args=["-s", str(metadata["threads"]), str(case)]):
                openmc.lib.simulation_init()
                live_tally = openmc.lib.tallies[model.tallies[0].id]
                for _ in openmc.lib.iter_batches():
                    batch = openmc.lib.current_batch()
                    now = time.monotonic()
                    elapsed = now - started
                    flux = float(live_tally.mean.ravel()[0]) / metadata["first_wall_mesh_volume_cm3"]
                    deviation = None
                    relative_error = None
                    if live_tally.num_realizations >= 2:
                        deviation = float(live_tally.std_dev.ravel()[0]) / metadata["first_wall_mesh_volume_cm3"]
                        relative_error = deviation / flux if flux > 0 else None
                    writer.writerow([batch, batch * metadata["particles_per_batch"], elapsed,
                                     flux, deviation, relative_error])
                    stream.flush()
                    if now - last_checkpoint >= metadata["checkpoint_seconds"]:
                        openmc.lib.statepoint_write(str(case / f"checkpoint.{batch}.h5"), write_source=False)
                        last_checkpoint = time.monotonic()
                    if elapsed >= metadata["target_seconds"] and batch >= metadata["minimum_batches"]:
                        metadata["stop_reason"] = "time target reached at a completed batch"
                        break
                else:
                    metadata["stop_reason"] = "maximum batch count reached"
                openmc.lib.statepoint_write(str(statepoint), write_source=False)
                openmc.lib.simulation_finalize()
        metadata.update(status="completed", elapsed_seconds=time.monotonic() - started,
                        completed_batches=batch, completed_source_histories=batch * metadata["particles_per_batch"],
                        statepoint=str(statepoint))
    except BaseException as error:
        metadata.update(status="failed or interrupted", elapsed_seconds=time.monotonic() - started,
                        error=f"{type(error).__name__}: {error}")
        write_json(run_directory / "run_summary.json", metadata)
        raise
    write_json(run_directory / "run_summary.json", metadata)
    print(f"Completed {batch} batches in {metadata['elapsed_seconds'] / 60:.2f} minutes.")
    return statepoint


# %%
def analyze_results(statepoint_path, run_directory, metadata):
    volume_cm3 = metadata["first_wall_mesh_volume_cm3"]
    with openmc.StatePoint(statepoint_path, autolink=False) as statepoint:
        if statepoint.n_realizations < 2:
            raise RuntimeError("At least two tally realizations are needed for uncertainty estimates.")
        total = statepoint.get_tally(name="first_wall_total_flux")
        spectrum = statepoint.get_tally(name="first_wall_energy_flux")
        mean = float(total.mean.ravel()[0]) / volume_cm3
        deviation = float(total.std_dev.ravel()[0]) / volume_cm3
        spectrum_mean = spectrum.mean.ravel() / volume_cm3
        spectrum_deviation = spectrum.std_dev.ravel() / volume_cm3
        bins = spectrum.find_filter(openmc.EnergyFilter).bins
        realizations = int(statepoint.n_realizations)
    if not math.isclose(float(spectrum_mean.sum()), mean, rel_tol=1e-9, abs_tol=1e-16):
        raise RuntimeError("The energy-bin fluxes do not sum to the total first-wall flux.")
    result = {"first_wall_volume_cm3": volume_cm3, "flux_per_source_cm2": mean,
              "std_dev_per_source_cm2": deviation,
              "relative_std_dev": deviation / mean if mean > 0 else None,
              "tally_realizations": realizations, "uncertainty": "one standard deviation of the mean"}
    rate = metadata["source_strength_n_per_s"]
    if rate is not None:
        result.update(source_strength_n_per_s=rate, flux_n_cm2_s=mean * rate,
                      std_dev_n_cm2_s=deviation * rate)
    write_json(run_directory / "first_wall_flux.json", result)
    with (run_directory / "first_wall_flux.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    with (run_directory / "energy_spectrum.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["energy_low_eV", "energy_high_eV", "flux_per_source_cm2", "std_dev_per_source_cm2"])
        for limits, flux, error in zip(bins, spectrum_mean, spectrum_deviation):
            writer.writerow([limits[0], limits[1], flux, error])
    positive = (bins[:, 0] > 0) & (spectrum_mean > 0)
    energies_MeV = np.sqrt(bins[positive, 0] * bins[positive, 1]) / 1e6
    figure, axis = plt.subplots()
    axis.errorbar(energies_MeV, spectrum_mean[positive], yerr=spectrum_deviation[positive],
                  fmt=".", markersize=3, linewidth=0.7)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Neutron energy (MeV)")
    axis.set_ylabel("Bin-integrated flux (cm⁻² per source neutron)")
    axis.set_title("Volume-averaged first-wall neutron spectrum; error bars: 1σ")
    figure.savefig(run_directory / "energy_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    metadata["first_wall_flux"] = result
    metadata["analysis_status"] = "completed"
    write_json(run_directory / "run_summary.json", metadata)
    print(f"First-wall average flux: {mean:.6g} ± {deviation:.2g} cm^-2 per source neutron (1σ).")
    if rate is not None:
        print(f"At {rate:.6g} n/s: {mean * rate:.6g} ± {deviation * rate:.2g} n/cm^2/s.")
    print(f"Results: {run_directory}")
    return result


# %%
# Setup and model export. Each execution has its own folder and geometry copy.
if __name__ == "__main__":
    model, run_directory, run_metadata = prepare_model(
        target_seconds, particles_per_batch, maximum_batches, threads, source_strength_n_per_s)
    print(f"Inputs and setup plot: {run_directory}")
    print(f"Transport target: {target_seconds / 60:g} minutes; {threads} threads.")


# %%
# Transport cell. Set run_transport=False to prepare inputs without launching.
if __name__ == "__main__":
    statepoint_path = None
    if run_transport:
        statepoint_path = run_timed_transport(model, run_directory, run_metadata)


# %%
# Analysis cell. The one-hour computational budget is not a physical irradiation time.
if __name__ == "__main__" and statepoint_path is not None:
    flux_result = analyze_results(statepoint_path, run_directory, run_metadata)

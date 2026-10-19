# %% [markdown]
# # Neutron flux in the tokamak first wall
#
# This calculation follows 14.1 MeV neutrons born in the plasma and measures
# their flux in the surrounding first wall. It uses the CAD geometry already
# converted and checked in `../tokamak_dagmc/`.
#
# The main result is the **scalar neutron flux averaged over the whole wall**.
# Neutrons moving in every direction contribute, including scattered neutrons.
# The energy spectrum shows how that flux is distributed among energy bins.
#
# To use this notebook, select the **ZClass** kernel, review the parameters below,
# and run the cells in order. The last three code cells prepare the model, run
# transport, and analyze the results. **Run All starts a roughly one-hour run.**
# Set `run_transport = False` to prepare the inputs and setup image first.
#
# All lengths passed to OpenMC are in **cm**, and neutron energies are in **eV**.
# With no source rate supplied, results are reported per source neutron.

# %% [markdown]
# ## 1. Load the tools and find the existing geometry
#
# OpenMC handles neutron transport. NumPy and Matplotlib handle the results.
# The sibling `tokamak_dagmc` folder supplies the material definitions and
# checks that the STEP files, repaired geometry, and mesh still match.

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
# Save figures without needing a desktop, so the script also works over SSH.
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
# Reuse the geometry study's materials and validation checks.
sys.path.insert(0, str(GEOMETRY_DIRECTORY))
from materials import make_materials
from workflow import (file_hash, load_config, nuclear_data_path, output_paths,
                      require_validation, write_json)


# %% [markdown]
# ## 2. Choose the run parameters
#
# A **history** starts with one sampled source neutron and follows its transport
# and any secondary particles. A **batch** groups histories so OpenMC can
# estimate the mean flux and its statistical uncertainty.
#
# The timer is checked after each batch. The run finishes the current batch
# before stopping, so it may take a little longer than an hour. Initialization
# counts toward the hour; setup, final file writing, and analysis add time.
# The batch limit is a second stopping condition if it is reached first.
#
# `source_strength_n_per_s` is the total physical neutron emission rate of the
# plasma. Leave it as `None` for a result per source neutron, or enter a known
# rate to also obtain n/cm²/s. The hour of computing time is not an irradiation
# time and does not set the physical source strength.

# %%
target_seconds = 60 * 60  # One hour of initialization and neutron transport.
particles_per_batch = 5000  # Source histories per batch.
maximum_batches = 10000  # Upper limit, even if the timer has not expired.
minimum_batches = 2  # Needed to estimate uncertainty; two alone give a poor estimate.
threads = 2  # CPU threads used by OpenMC.
seed = 42  # Fix the random seed to help reproduce a run with the same settings.
checkpoint_seconds = 10 * 60  # Save accumulated tallies about every ten minutes.
source_strength_n_per_s = None  # Total plasma neutron rate, if known [neutrons/s].
run_transport = True  # False prepares the model without starting transport.

# Logarithmic energy bins span 1e-5 eV to 100 MeV. Include a first bin from zero
# so very low-energy neutrons still contribute to the spectrum.
energy_edges_eV = np.concatenate(([0.0], np.geomspace(1e-5, 1e8, 101)))


# %% [markdown]
# ## 3. Define the model setup
#
# This cell defines a helper; the model is built in the setup cell near the end.
# It copies the checked DAGMC surface mesh into a new run folder, assigns the
# existing materials, and defines the source and two first-wall tallies.
# DAGMC follows particles through the triangular CAD surfaces.
#
# The source is **uniform throughout the plasma volume** and **isotropic**
# (all directions are equally likely), with a single energy of **14.1 MeV**.
# These are simple D–T source assumptions, rather than a fitted plasma profile.
# The material compositions and coarse surface mesh are also approximations.
#
# The first tally measures total wall flux. The second sorts the same flux
# into energy bins. Both include every part carrying the first-wall material
# tag. The outer vacuum boundary removes particles that leave the model.

# %%
def prepare_model(duration_seconds, batch_particles, batch_limit, thread_count,
                  source_strength=None, run_label="first_wall"):
    """Build the fixed-source model and save its inputs in a new run folder."""
    if duration_seconds <= 0 or batch_particles <= 0 or batch_limit < minimum_batches:
        raise ValueError("Use a positive duration/particle count and at least two batches.")
    if thread_count <= 0 or (source_strength is not None and source_strength <= 0):
        raise ValueError("Thread count and any supplied source strength must be positive.")
    if not openmc.lib._dagmc_enabled():
        raise RuntimeError("This OpenMC installation does not support DAGMC.")

    config = load_config()
    geometry_paths = output_paths(config)
    # Stop here if the source CAD, repair, or mesh differs from the checked model.
    validation = require_validation(config, geometry_paths)
    inspection = json.loads((geometry_paths["reports"] / "step_inspection.json").read_text())
    wall = next(c for c in config["components"] if c["file"] == "first_wall.step")
    plasma = next(c for c in inspection["components"] if c["file"] == "Plamsa.step")
    wall_mesh = [v for v in validation["volume_mesh_checks"] if v["material"] == wall["tag"]]
    plasma_ids = [v["id"] for v in validation["volume_mesh_checks"] if v["material"] == plasma["tag"]]
    if not wall_mesh or not plasma_ids:
        raise RuntimeError("The validated mesh is missing the first wall or plasma.")
    # Normalize using the volume enclosed by the triangles that OpenMC uses,
    # rather than the slightly different volume of the original curved CAD.
    wall_volume_cm3 = sum(v["signed_volume_cm3"] for v in wall_mesh)
    if wall_volume_cm3 <= 0:
        raise RuntimeError("The first-wall mesh has a nonpositive volume.")
    # A material filter would also count other components if they shared this tag.
    if sum(c["tag"] == wall["tag"] for c in config["components"]) != 1:
        raise RuntimeError("The first-wall material tag must be unique to the first wall.")

    # A separate folder for each execution keeps previous results available.
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
    # Use the installed nuclear data; check that every material nuclide is covered.
    library = openmc.data.DataLibrary.from_xml(cross_sections)
    for material in materials:
        for nuclide in material.get_nuclides():
            entry = library.get_by_material(nuclide)
            if entry is None or not Path(entry["path"]).is_file():
                raise RuntimeError(f"Nuclear data are missing for {nuclide}.")

    # Use the saved mesh copy so later remeshing cannot change this run's geometry.
    dagmc = openmc.DAGMCUniverse(mesh_file)
    geometry = openmc.Geometry(dagmc.bounded_universe(
        bounded_type="box", boundary_type="vacuum",
        padding_distance=config["openmc"]["boundary_padding_cm"],
    ))

    # First sample a cylinder enclosing the plasma, then reject points outside
    # the actual plasma cells. Accepted points are uniform in the plasma volume.
    lower = np.array(plasma["bbox_mm"]["min"]) * config["units"]["imported_to_cm"]
    upper = np.array(plasma["bbox_mm"]["max"]) * config["units"]["imported_to_cm"]
    center = (lower + upper) / 2
    radius_cm = max(upper[0] - lower[0], upper[1] - lower[1]) / 2
    space = openmc.stats.CylindricalIndependent(
        # Cylindrical volume contains r*dr, so uniform volume needs p(r) ∝ r.
        r=openmc.stats.PowerLaw(0.0, radius_cm, 1.0),
        phi=openmc.stats.Uniform(0.0, 2 * math.pi),
        z=openmc.stats.Uniform(lower[2], upper[2]),
        origin=(center[0], center[1], 0.0),
    )
    # These cell objects identify existing DAGMC volumes for source rejection.
    # They do not add new physical cells to the geometry.
    source_domains = [openmc.Cell(cell_id=cell_id) for cell_id in plasma_ids]
    source = openmc.IndependentSource(
        space=space, angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([14.1e6], [1.0]),
        constraints={"domains": source_domains, "rejection_strategy": "resample"},
    )

    settings = openmc.Settings()
    # The source is prescribed; this is not a criticality/eigenvalue calculation.
    settings.run_mode = "fixed source"
    settings.source = source
    settings.particles = int(batch_particles)
    settings.batches = int(batch_limit)
    settings.seed = seed
    # Lost particles can indicate a geometry problem; stop rather than ignore them.
    settings.max_lost_particles = 1
    settings.rel_max_lost_particles = 1e-9
    settings.output = {"path": str(case_directory), "summary": True, "tallies": False}
    # The timed loop below writes its own checkpoints and final statepoint.
    settings.statepoint = {"batches": [int(batch_limit)]}
    settings.sourcepoint = {"write": False}

    # Collect all wall solids in one tally bin. OpenMC then estimates uncertainty
    # for the whole wall, including correlations between different wall regions.
    wall_filter = openmc.MaterialFilter([wall_material])
    neutron_filter = openmc.ParticleFilter(["neutron"])
    total_flux = openmc.Tally(name="first_wall_total_flux")
    total_flux.filters = [wall_filter, neutron_filter]
    total_flux.scores = ["flux"]
    # Track length adds the weighted neutron path lengths inside the wall.
    # Dividing by wall volume later converts this to volume-averaged scalar flux.
    total_flux.estimator = "tracklength"
    spectrum = openmc.Tally(name="first_wall_energy_flux")
    spectrum.filters = [wall_filter, neutron_filter, openmc.EnergyFilter(energy_edges_eV)]
    spectrum.scores = ["flux"]
    spectrum.estimator = "tracklength"
    model = openmc.Model(geometry, materials, settings, openmc.Tallies([total_flux, spectrum]))
    model.export_to_xml(case_directory)

    # Save the assumptions, geometry identity, and settings with the run so the
    # results can still be understood after parameters in this script change.
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


# %% [markdown]
# ## 4. Define the setup image
#
# Reuse the geometry views made for this exact mesh and save them as `setup.png`.
# The full view gives context; the close-up shows the first wall in red and
# plasma in cyan. The plasma is the source region, and the red wall is the
# tally region. Review this image before interpreting a flux result.

# %%
def make_setup_plot(geometry_paths, run_directory, mesh_sha256):
    """Save a full-model view and a close-up of the source and first wall."""
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


# %% [markdown]
# ## 5. Define the timed transport loop
#
# OpenMC runs one complete batch at a time. After each batch, this helper
# records the current wall flux and uncertainty in `flux_history.csv`.
# Uncertainty is left blank until at least two batches have completed.
#
# A **statepoint** is an OpenMC file containing accumulated tallies and run
# information. Checkpoints save these results periodically; the final file is
# `cases/dt_14p1MeV/statepoint.final.h5`. These files omit source particles and
# are for reading tally results, not a promise that an interrupted run can resume.
#
# The timer stops transport at a batch boundary and saves the final tallies.
# This cell only defines that procedure; the transport cell below starts it.

# %%
def run_timed_transport(model, run_directory, metadata):
    """Run complete batches until the time target or batch limit is reached."""
    if openmc.lib.is_initialized:
        raise RuntimeError("Finalize the existing OpenMC library session before starting this study.")
    case = Path(metadata["case_directory"])
    statepoint = case / "statepoint.final.h5"
    # A monotonic clock measures elapsed time even if the system clock is adjusted.
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
            # The in-memory OpenMC interface lets Python check the clock between
            # batches. "-s" selects the number of OpenMP threads.
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
                        # This is the standard error of the estimated mean flux.
                        deviation = float(live_tally.std_dev.ravel()[0]) / metadata["first_wall_mesh_volume_cm3"]
                        relative_error = deviation / flux if flux > 0 else None
                    writer.writerow([batch, batch * metadata["particles_per_batch"], elapsed,
                                     flux, deviation, relative_error])
                    # Make each completed batch visible in the CSV during the run.
                    stream.flush()
                    if now - last_checkpoint >= metadata["checkpoint_seconds"]:
                        openmc.lib.statepoint_write(str(case / f"checkpoint.{batch}.h5"), write_source=False)
                        last_checkpoint = time.monotonic()
                    if elapsed >= metadata["target_seconds"] and batch >= metadata["minimum_batches"]:
                        metadata["stop_reason"] = "time target reached at a completed batch"
                        break
                else:
                    metadata["stop_reason"] = "maximum batch count reached"
                # Write before finalizing, while the accumulated tallies are in memory.
                openmc.lib.statepoint_write(str(statepoint), write_source=False)
                openmc.lib.simulation_finalize()
        metadata.update(status="completed", elapsed_seconds=time.monotonic() - started,
                        completed_batches=batch, completed_source_histories=batch * metadata["particles_per_batch"],
                        statepoint=str(statepoint))
    except BaseException as error:
        # Record an interrupted or failed run instead of leaving its status "running".
        metadata.update(status="failed or interrupted", elapsed_seconds=time.monotonic() - started,
                        error=f"{type(error).__name__}: {error}")
        write_json(run_directory / "run_summary.json", metadata)
        raise
    write_json(run_directory / "run_summary.json", metadata)
    print(f"Completed {batch} batches in {metadata['elapsed_seconds'] / 60:.2f} minutes.")
    return statepoint


# %% [markdown]
# ## 6. Define the flux calculation and spectrum plot
#
# OpenMC reports the track-length score integrated over the wall, normalized
# per source neutron. Divide by the first-wall mesh volume to obtain the
# volume-averaged scalar flux:
#
# \[
# \overline{\phi}_{\mathrm{per\ source}} =
# \frac{T_{\mathrm{wall}}}{V_{\mathrm{wall}}}
# \qquad [\mathrm{cm}^{-2}\ \mathrm{per\ source\ neutron}].
# \]
#
# Here \(T_{\mathrm{wall}}\) is OpenMC's wall track-length score per source
# neutron and \(V_{\mathrm{wall}}\) is the volume enclosed by the wall mesh.
# If the total plasma source rate is \(S\) neutrons/s, the physical flux is
# \(S\,\overline{\phi}_{\mathrm{per\ source}}\) in n/cm²/s.
#
# Error bars are **one standard deviation of the estimated mean**. They measure
# Monte Carlo sampling uncertainty, not errors in the source, materials, or CAD.
# A longer run generally reduces sampling uncertainty; it does not check those
# physical assumptions or establish mesh convergence.
#
# Each spectrum value is the flux **integrated over its energy bin**. The
# plotted values are not divided by the bin width. Their means should sum to
# the total flux; their uncertainties cannot simply be added in quadrature,
# because energy-bin scores can be correlated.

# %%
def analyze_results(statepoint_path, run_directory, metadata):
    """Convert wall track-length tallies into flux and save tables and a plot."""
    volume_cm3 = metadata["first_wall_mesh_volume_cm3"]
    with openmc.StatePoint(statepoint_path, autolink=False) as statepoint:
        if statepoint.n_realizations < 2:
            raise RuntimeError("At least two tally realizations are needed for uncertainty estimates.")
        total = statepoint.get_tally(name="first_wall_total_flux")
        spectrum = statepoint.get_tally(name="first_wall_energy_flux")
        # The same volume normalization applies to the mean and its uncertainty.
        mean = float(total.mean.ravel()[0]) / volume_cm3
        deviation = float(total.std_dev.ravel()[0]) / volume_cm3
        spectrum_mean = spectrum.mean.ravel() / volume_cm3
        spectrum_deviation = spectrum.std_dev.ravel() / volume_cm3
        bins = spectrum.find_filter(openmc.EnergyFilter).bins
        realizations = int(statepoint.n_realizations)
    # Check that the energy bins cover the flux seen by the separate total tally.
    if not math.isclose(float(spectrum_mean.sum()), mean, rel_tol=1e-9, abs_tol=1e-16):
        raise RuntimeError("The energy-bin fluxes do not sum to the total first-wall flux.")
    result = {"first_wall_volume_cm3": volume_cm3, "flux_per_source_cm2": mean,
              "std_dev_per_source_cm2": deviation,
              "relative_std_dev": deviation / mean if mean > 0 else None,
              "tally_realizations": realizations, "uncertainty": "one standard deviation of the mean"}
    rate = metadata["source_strength_n_per_s"]
    if rate is not None:
        # Apply the physical neutron rate only if the user supplied one.
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
    # Zero cannot appear on a log axis. Keep that lowest bin in the CSV, and plot
    # the positive bins at their geometric center energies, converted to MeV.
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


# %% [markdown]
# ## 7. Prepare this run and inspect the setup
#
# Run this cell to create a new `outputs/first_wall_<timestamp>/` folder.
# It saves the input XML files and a mesh copy under `cases/dt_14p1MeV/`,
# along with the settings and geometry checks needed to identify this run.
# Open `setup.png` in the printed folder to check the source and tally regions.
# **This cell prepares the calculation; the next cell starts neutron transport.**

# %%
# These guards let the file be imported for checks without starting a simulation.
# In a normal notebook kernel, __name__ is "__main__", so these cells run normally.
if __name__ == "__main__":
    model, run_directory, run_metadata = prepare_model(
        target_seconds, particles_per_batch, maximum_batches, threads, source_strength_n_per_s)
    print(f"Inputs and setup plot: {run_directory}")
    print(f"Transport target: {target_seconds / 60:g} minutes; {threads} threads.")


# %% [markdown]
# ## 8. Run neutron transport
#
# With `run_transport = True`, this cell starts the approximately one-hour run.
# Keep the kernel running until it finishes. Batch progress appears in OpenMC's
# output, and `flux_history.csv` records the accumulating flux estimate.
# With `run_transport = False`, this cell skips transport and the analysis
# cell below has no new statepoint to read.

# %%
if __name__ == "__main__":
    statepoint_path = None
    if run_transport:
        statepoint_path = run_timed_transport(model, run_directory, run_metadata)


# %% [markdown]
# ## 9. Read the results
#
# After transport finishes, this cell prints the whole-wall average flux and
# its statistical uncertainty. It saves these files directly in the run folder:
#
# - `first_wall_flux.csv` and `.json`: total flux, uncertainty, and wall volume.
# - `energy_spectrum.csv`: flux and uncertainty for every energy bin.
# - `energy_spectrum.png`: the neutron spectrum with one-standard-deviation bars.
# - `run_summary.json`: run settings, completed histories, and the final result.
#
# The whole-wall average does not show local hot spots or the net current
# crossing a surface. Use it as the average scalar flux for this source and
# geometry, with the source and material assumptions described above.

# %%
if __name__ == "__main__" and statepoint_path is not None:
    flux_result = analyze_results(statepoint_path, run_directory, run_metadata)

# %% [markdown]
# # Neutron flux in the tokamak first wall
#
# This calculation follows 14.1 MeV neutrons born in the plasma and measures
# the **scalar neutron flux averaged over the complete first-wall volume**.
# It also calculates an energy spectrum to show how that flux is distributed
# among neutron energies. Neutrons moving in all directions contribute,
# including neutrons that scatter before reaching or while inside the wall.
# The geometry and material definitions come from `../tokamak_dagmc/`.
#
# Select the **ZClass** kernel and run the cells in order. The transport cell
# runs a fixed number of source histories: `batches * particles_per_batch`.
# First review the parameters and geometry image, then run transport and read
# the results. Set `run_transport = False` to prepare inputs and the setup image
# only. Each execution creates a new output folder and preserves previous runs.
#
# Lengths passed to OpenMC are in **cm** and neutron energies are in **eV**.
# Results are per source neutron unless a physical neutron emission rate is
# supplied. The number of simulated histories controls sampling, rather than
# the physical irradiation time.
#
# ## Imports and file locations
#
# OpenMC builds the transport model; NumPy and Matplotlib handle the results.
# The paths below locate this study and the sibling geometry folder when
# running either the Python script or its notebook. Small shared helpers load
# the existing materials, configuration, and geometry checks.

# %%
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib
# Save figures without needing a desktop, so the script also works over SSH.
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import openmc

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


# %% [markdown]
# ## Run parameters
#
# A history follows one source neutron and its secondary particles. Batches
# group histories so OpenMC can estimate the mean flux and its uncertainty.
# The total number of source histories is
#
# $$N_{\mathrm{source}} = N_{\mathrm{batches}}\,N_{\mathrm{particles\ per\ batch}}.$$
#
# The defaults below give **100 × 5,000 = 500,000 source histories**. Change
# `batches` or `particles_per_batch` to change the sample count. At least two
# batches are needed to estimate uncertainty; two alone give a poor estimate.
# More histories generally reduce Monte Carlo sampling uncertainty.
#
# `threads` sets the number of CPU threads, and `seed` fixes the random-number
# seed to help reproduce a run with the same settings. `source_energy_eV`
# sets the monoenergetic source, with a default of 14.1 MeV for D–T neutrons.
#
# Leave `source_strength_n_per_s` as `None` for flux per source neutron. Supply
# the total plasma neutron emission rate to also calculate flux in n/cm²/s.
# `energy_edges_eV` sets the spectrum bins, including a bin from zero to
# $10^{-5}$ eV and logarithmic bins from $10^{-5}$ eV to 100 MeV.

# %%
particles_per_batch = 5000
batches = 100
threads = 2
seed = 42
source_energy_eV = 14.1e6
source_strength_n_per_s = None  # Total plasma neutron rate [neutrons/s], if known.
run_transport = True

# Include a bin starting at zero, followed by logarithmic bins up to 100 MeV.
energy_edges_eV = np.concatenate(([0.0], np.geomspace(1e-5, 1e8, 101)))

if particles_per_batch <= 0 or batches < 2:
    raise ValueError("Use a positive particle count and at least two batches.")
if threads <= 0 or (source_strength_n_per_s is not None and source_strength_n_per_s <= 0):
    raise ValueError("Thread count and any supplied source strength must be positive.")


# %% [markdown]
# ## Read the checked geometry
#
# Read the geometry configuration, CAD inspection, and DAGMC validation
# report. The shared validation check requires the source CAD, repaired
# geometry, and mesh to match the checked model. This notebook reuses those
# files rather than rebuilding the CAD or mesh.
#
# Identify the first-wall material tag and the plasma cell IDs. The material
# tag selects the tally region, while the cell IDs will confine source sites
# to the plasma. The wall tag must be unique so other components are excluded.
#
# Add the volumes enclosed by the first-wall mesh surfaces to obtain
# `wall_volume_cm3`. Use this volume for tally normalization because OpenMC
# transports particles through the triangular mesh, whose enclosed volume
# differs slightly from the original curved CAD volume.

# %%
config = load_config()
geometry_paths = output_paths(config)
validation = require_validation(config, geometry_paths)
inspection = json.loads((geometry_paths["reports"] / "step_inspection.json").read_text())

for component in config["components"]:
    if component["file"] == "first_wall.step":
        wall = component
for component in inspection["components"]:
    if component["file"] == "Plamsa.step":
        plasma = component

wall_mesh = []
plasma_ids = []
for volume in validation["volume_mesh_checks"]:
    if volume["material"] == wall["tag"]:
        wall_mesh.append(volume)
    if volume["material"] == plasma["tag"]:
        plasma_ids.append(volume["id"])
if not wall_mesh or not plasma_ids:
    raise RuntimeError("The validated mesh is missing the first wall or plasma.")

wall_volume_cm3 = sum(volume["signed_volume_cm3"] for volume in wall_mesh)
if wall_volume_cm3 <= 0:
    raise RuntimeError("The first-wall mesh has a nonpositive volume.")
# A material filter would also count other components if they shared this tag.
if sum(component["tag"] == wall["tag"] for component in config["components"]) != 1:
    raise RuntimeError("The first-wall material tag must be unique to the first wall.")


# %% [markdown]
# ## Materials
#
# Reuse the compositions defined in `../tokamak_dagmc/materials.py`. The first
# wall is tungsten at 19.3 g/cm³. Other components use provisional steel or
# copper/steel bulk compositions. These are modelling assumptions, rather
# than detailed engineering specifications for the tokamak.
#
# The first-wall material object is used by the tally filter below. Check
# that the existing nuclear-data index and a data file for every material
# nuclide are available before exporting the model.

# %%
materials = make_materials(config)
for material in materials:
    if material.name == wall["tag"]:
        wall_material = material

cross_sections = nuclear_data_path(config)
if not cross_sections or not Path(cross_sections).is_file():
    raise RuntimeError("The existing nuclear-data index is unavailable.")
library = openmc.data.DataLibrary.from_xml(cross_sections)
for material in materials:
    for nuclide in material.get_nuclides():
        entry = library.get_by_material(nuclide)
        if entry is None or not Path(entry["path"]).is_file():
            raise RuntimeError(f"Nuclear data are missing for {nuclide}.")


# %% [markdown]
# ## Geometry and run folder
#
# Create `outputs/first_wall_<timestamp>/` for this execution. Put the mesh,
# OpenMC input files, and transport results in `cases/dt_14p1MeV/`. Summary
# tables and figures will go directly in the run folder.
#
# Copy the checked `tokamak.h5m` mesh into the case folder and verify that its
# file hash matches the validation report. DAGMC uses these triangular CAD
# surfaces for particle tracking. Enclose the model in a box with a **vacuum
# boundary**: particles that cross the outer box leave the calculation.

# %%
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
run_directory = STUDY_DIRECTORY / "outputs" / f"first_wall_{stamp}"
case_directory = run_directory / "cases" / "dt_14p1MeV"
case_directory.mkdir(parents=True)
mesh_file = case_directory / "tokamak.h5m"
shutil.copy2(geometry_paths["dagmc"] / "tokamak.h5m", mesh_file)
if file_hash(mesh_file) != validation["h5m_sha256"]:
    raise RuntimeError("The copied mesh does not match the validated geometry.")

dagmc = openmc.DAGMCUniverse(mesh_file)
geometry = openmc.Geometry(dagmc.bounded_universe(
    bounded_type="box", boundary_type="vacuum",
    padding_distance=config["openmc"]["boundary_padding_cm"],
))


# %% [markdown]
# ## Source
#
# Assume **uniform neutron emission per unit plasma volume**, **isotropic
# directions**, and a single source energy of **14.1 MeV**. This represents
# a simple D–T plasma source without a fitted spatial or energy profile.
#
# Use the plasma bounding box to define a cylinder enclosing the source
# region. Sample the axial and azimuthal coordinates uniformly. The radial
# probability density is proportional to $r$, since cylindrical volume
# contains a factor $r\,dr\,d\varphi\,dz$.
#
# Reject sampled points outside the actual DAGMC plasma cells and resample
# them. The accepted points are uniform within the plasma, including its
# toroidal shape. The cylinder itself adds no material or physical boundary.

# %%
lower = np.array(plasma["bbox_mm"]["min"]) * config["units"]["imported_to_cm"]
upper = np.array(plasma["bbox_mm"]["max"]) * config["units"]["imported_to_cm"]
center = (lower + upper) / 2
radius_cm = max(upper[0] - lower[0], upper[1] - lower[1]) / 2
space = openmc.stats.CylindricalIndependent(
    # Cylindrical volume contains r*dr, so uniform volume needs p(r) proportional to r.
    r=openmc.stats.PowerLaw(0.0, radius_cm, 1.0),
    phi=openmc.stats.Uniform(0.0, 2 * math.pi),
    z=openmc.stats.Uniform(lower[2], upper[2]),
    origin=(center[0], center[1], 0.0),
)
# These objects identify existing DAGMC volumes; they do not add physical cells.
source_domains = [openmc.Cell(cell_id=cell_id) for cell_id in plasma_ids]
source = openmc.IndependentSource(
    space=space, angle=openmc.stats.Isotropic(),
    energy=openmc.stats.Discrete([source_energy_eV], [1.0]),
    constraints={"domains": source_domains, "rejection_strategy": "resample"},
)


# %% [markdown]
# ## Settings and tallies
#
# This is a **fixed-source** calculation: the prescribed plasma source starts
# each history. The batch and particle counts set above determine how many
# source histories OpenMC follows. Save the accumulated tallies in a
# statepoint after the final batch, and stop if particles are lost in geometry.
#
# Both tallies use the **track-length estimator**, which accumulates weighted
# neutron path lengths inside the wall. The material filter includes all wall
# solids, and the particle filter restricts the score to neutrons.
#
# - `first_wall_total_flux` scores the complete wall flux in one bin.
# - `first_wall_energy_flux` scores the same quantity in separate energy bins.
#
# The total tally provides the uncertainty of the whole-wall result directly,
# retaining correlations between wall regions. The score is integrated over
# the wall volume; the analysis cells divide it by `wall_volume_cm3`.

# %%
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.source = source
settings.particles = particles_per_batch
settings.batches = batches
settings.seed = seed
# Lost particles can indicate a geometry problem.
settings.max_lost_particles = 1
settings.rel_max_lost_particles = 1e-9
settings.output = {"path": str(case_directory), "summary": True, "tallies": False}
settings.statepoint = {"batches": [batches]}
settings.sourcepoint = {"write": False}

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


# %% [markdown]
# ## Export the model and view the setup
#
# Combine the geometry, materials, source settings, and tallies into an OpenMC
# model and export its XML inputs. Save the run parameters, geometry identity,
# and source assumptions in `run_summary.json`. Copies of the script and
# geometry configuration record what was used for this particular run.
#
# Reuse the geometry views previously made for this exact mesh: a full
# poloidal view and a closer view of the first-wall layers. The **red first
# wall** is the tally region and the **cyan plasma** is the source region.
# Neutrons are emitted throughout the plasma in all directions.
#
# Open `setup.png` in the printed run folder to check these regions before
# starting transport. This cell prepares the model; the next cell runs it.

# %%
model = openmc.Model(geometry, materials, settings, openmc.Tallies([total_flux, spectrum]))
model.export_to_xml(case_directory)

metadata = {
    "status": "prepared", "particles_per_batch": particles_per_batch,
    "batches": batches, "source_histories": batches * particles_per_batch,
    "threads": threads, "seed": seed, "openmc_version": openmc.__version__,
    "python_executable": sys.executable, "h5m_sha256": validation["h5m_sha256"],
    "first_wall_tag": wall["tag"], "first_wall_material_id": wall_material.id,
    "first_wall_cell_ids": [volume["id"] for volume in wall_mesh],
    "first_wall_mesh_volume_cm3": wall_volume_cm3,
    "plasma_cell_ids": plasma_ids, "proposal_radius_cm": radius_cm,
    "proposal_z_cm": [float(lower[2]), float(upper[2])],
    "source_energy_eV": source_energy_eV, "source_isotropic": True,
    "source_strength_n_per_s": source_strength_n_per_s, "cross_sections": cross_sections,
    "geometry_debug": False, "case_directory": str(case_directory),
    "normalization": "track-length tally / actual first-wall mesh volume; per source neutron",
    "assumptions": "Uniform plasma emissivity; provisional materials and coarse CAD surface mesh.",
}
write_json(run_directory / "run_summary.json", metadata)
shutil.copy2(GEOMETRY_DIRECTORY / "config.yaml", run_directory / "geometry_config.yaml")
shutil.copy2(STUDY_DIRECTORY / "first_wall_flux.py", run_directory / "first_wall_flux.py")
for name in ("dagmc_validation.json", "topology_repair.json"):
    shutil.copy2(geometry_paths["reports"] / name, run_directory / name)

plot_report = json.loads((geometry_paths["reports"] / "openmc_plots.json").read_text())
if plot_report.get("h5m_sha256") != validation["h5m_sha256"]:
    raise RuntimeError("Regenerate the geometry plots for the current mesh first.")
figure, axes = plt.subplots(1, 2, figsize=(16, 8))
for axis, name, title in zip(axes, ("tokamak_xz.png", "poloidal_zoom.png"),
                             ("Full tokamak", "First-wall layers")):
    axis.imshow(mpimg.imread(geometry_paths["plots"] / name))
    axis.set_title(title)
    axis.axis("off")
figure.suptitle("First-wall tally: red region; isotropic neutron source in the cyan plasma")
figure.savefig(run_directory / "setup.png", dpi=140, bbox_inches="tight")
plt.close(figure)
print(f"Inputs and setup plot: {run_directory}")
print(f"Transport: {batches} batches × {particles_per_batch} = {batches * particles_per_batch:,} source histories.")


# %% [markdown]
# ## Run neutron transport
#
# With `run_transport = True`, call `openmc.run()` to follow the specified
# number of source histories. OpenMC reports batch progress in its output.
# Runtime depends on the sample count, geometry, materials, and computer.
# Keep the notebook kernel running until transport finishes.
#
# The **statepoint** contains the accumulated tallies and run information.
# The final file is `cases/dt_14p1MeV/statepoint.<batches>.h5`, which is
# `statepoint.100.h5` for the default settings. The analysis cells read this
# file. With `run_transport = False`, this cell and the analysis are skipped.

# %%
statepoint_path = None
if run_transport:
    metadata["status"] = "running"
    write_json(run_directory / "run_summary.json", metadata)
    openmc.run(cwd=case_directory, threads=threads)
    statepoint_path = case_directory / f"statepoint.{batches}.h5"
    metadata.update(status="completed", completed_batches=batches,
                    completed_source_histories=batches * particles_per_batch,
                    statepoint=str(statepoint_path))
    write_json(run_directory / "run_summary.json", metadata)


# %% [markdown]
# ## Read the flux and energy spectrum
#
# OpenMC normalizes the track-length tally per source neutron. Divide the
# wall-integrated tally, $T_{\mathrm{wall}}$, by the mesh volume,
# $V_{\mathrm{wall}}$, to obtain the volume-averaged scalar flux:
#
# $$\overline{\phi}_{\mathrm{per\ source}} =
# \frac{T_{\mathrm{wall}}}{V_{\mathrm{wall}}}
# \qquad [\mathrm{cm}^{-2}\ \mathrm{per\ source\ neutron}].$$
#
# Divide the tally standard deviation by the same volume. If the physical
# plasma source strength is $S$ neutrons/s, multiply both the flux and its
# standard deviation by $S$ to obtain **n/cm²/s**:
#
# $$\overline{\phi} = S\,\overline{\phi}_{\mathrm{per\ source}}.$$
#
# The reported relative standard deviation is the absolute standard deviation
# divided by the mean. Error bars show **one standard deviation of the
# estimated mean**. They describe Monte Carlo sampling uncertainty, rather
# than uncertainty in material compositions, source assumptions, or geometry.
#
# Spectrum values are integrated over each energy bin. Their means should sum
# to the total flux. Their uncertainties cannot be added in quadrature because
# energy bins can be correlated. The check below compares the sum of the
# energy-bin means with the independently scored total tally.

# %%
if statepoint_path is not None:
    with openmc.StatePoint(statepoint_path, autolink=False) as statepoint:
        if statepoint.n_realizations < 2:
            raise RuntimeError("At least two tally realizations are needed for uncertainty estimates.")
        total = statepoint.get_tally(name="first_wall_total_flux")
        spectrum = statepoint.get_tally(name="first_wall_energy_flux")
        mean = float(total.mean.ravel()[0]) / wall_volume_cm3
        deviation = float(total.std_dev.ravel()[0]) / wall_volume_cm3
        spectrum_mean = spectrum.mean.ravel() / wall_volume_cm3
        spectrum_deviation = spectrum.std_dev.ravel() / wall_volume_cm3
        bins = spectrum.find_filter(openmc.EnergyFilter).bins
        realizations = int(statepoint.n_realizations)

    if not math.isclose(float(spectrum_mean.sum()), mean, rel_tol=1e-9, abs_tol=1e-16):
        raise RuntimeError("The energy-bin fluxes do not sum to the total first-wall flux.")
    result = {
        "first_wall_volume_cm3": wall_volume_cm3, "flux_per_source_cm2": mean,
        "std_dev_per_source_cm2": deviation,
        "relative_std_dev": deviation / mean if mean > 0 else None,
        "tally_realizations": realizations, "uncertainty": "one standard deviation of the mean",
    }
    if source_strength_n_per_s is not None:
        result.update(source_strength_n_per_s=source_strength_n_per_s,
                      flux_n_cm2_s=mean * source_strength_n_per_s,
                      std_dev_n_cm2_s=deviation * source_strength_n_per_s)
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


# %% [markdown]
# ## Plot and save the results
#
# Plot the bin-integrated flux against the geometric center of each energy
# bin, converting the horizontal axis to MeV. Both axes are logarithmic. The
# plotted flux is not divided by energy-bin width, so it is not a differential
# spectrum per MeV. Bins starting at zero or having zero estimated flux remain
# in the CSV but are omitted from the plot.
#
# Find the following outputs directly in the run folder:
#
# | File | Contents |
# | --- | --- |
# | `setup.png` | Geometry views showing the source and tally regions. |
# | `first_wall_flux.csv` and `.json` | Whole-wall flux, uncertainty, and volume. |
# | `energy_spectrum.csv` | Bin edges, bin-integrated flux, and uncertainty. |
# | `energy_spectrum.png` | Spectrum with one-standard-deviation error bars. |
# | `run_summary.json` | Run settings, completed history count, and final result. |
#
# The whole-wall average does not show local hot spots or surface current.
# More source histories can reduce sampling uncertainty, but do not establish
# the accuracy of the coarse surface mesh or provisional material assumptions.

# %%
if statepoint_path is not None:
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
    if source_strength_n_per_s is not None:
        print(f"At {source_strength_n_per_s:.6g} n/s: "
              f"{mean * source_strength_n_per_s:.6g} ± {deviation * source_strength_n_per_s:.2g} n/cm^2/s.")
    print(f"Results: {run_directory}")

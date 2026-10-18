# Fusion STEP to OpenMC DAGMC

The seven original files stay in `../step_files/`. Their actual names include
`Plamsa.step` and `Vaccume Vessel.step`. Filename-to-material assignments, units,
mesh profiles, plotting pixels, and test-run parameters are in `config.yaml`.
Material compositions in `materials.py` are editable approximations.

For the one-hour first-wall flux calculation, use
[`../first_wall_flux/first_wall_flux.py`](../first_wall_flux/first_wall_flux.py)
or its matching notebook with `ZClass`. The scripts here rebuild and check the
CAD geometry; they do not need to run before each flux calculation.
Optional diagnostics and previews, with their notebooks, are in
[`archive/`](archive/README.md). Keep the existing STEP files, configuration,
repaired BREP, H5M, validation reports, and setup plots for the flux calculation.

Read [the geometry report](outputs/reports/geometry_report.md) for the executed
checks and current results. The original files have an OpenCascade interface
failure. `repair_step.py` makes a separate BREP copy that shares the existing
blanket/wall and blanket/vessel shell topology. It checks that the geometry and
each body's volume are preserved. It does not cut material or move components.
The original inspection failure remains recorded in `step_inspection.json`.

## Environment

This project uses an isolated Conda clone of `ZClass`, with the CAD packages
added only to that clone. `ZClass`, `OpenMC`, the POPCON files, nuclear-data files,
and shell environment settings were not changed.

Activate the existing project environment:

```bash
cd /home/youeric/openmc-projects/ZachClass/OPENMC/tokamak_dagmc
conda activate "$PWD/.conda-env"
```

To recreate it if `.conda-env` does not exist:

```bash
conda create --prefix "$PWD/.conda-env" --clone ZClass -y
conda activate "$PWD/.conda-env"
python -m pip install -r requirements-cad.txt
```

The clone retains OpenMC 0.16.0 with DAGMC 3.2.4 and MOAB 5.6.0. The CAD tools
are cad_to_dagmc 0.15.1, CadQuery 2.8.0, OpenCascade 7.9.3, and Gmsh 4.15.2.
The H5M writer uses h5py; Python MOAB bindings are not needed. Existing DAGMC
`check_watertight` and `overlap_check` executables are used independently.
The archived `inspect_environment.py` records runtime versions, scheduler availability,
OpenMC DAGMC support, and nuclear-data coverage for all material nuclides.

The clone does not inherit Conda environment variables. `config.yaml` therefore
records the same existing cross-section index used by `ZClass`; an activated
`OPENMC_CROSS_SECTIONS` takes precedence. It is written to materials.xml without
changing global settings. No nuclear data are downloaded. Adjust this path on
another machine, or leave it empty to export inputs without transport.

## Reproduce

From the activated environment above:

```bash
python inspect_step.py
python repair_step.py
python convert_to_dagmc.py
python validate_dagmc.py
python plot_geometry.py
```

`inspect_step.py` archives previous reports. Run the later stages again after
inspection, changing a STEP file, or changing geometry/meshing configuration. Fingerprints prevent
stale inspection, repair, or mesh files from being accepted. Transport controls
and plotting pixel counts can change without rerunning CAD inspection. `repair_step.py`
only accepts the specific interface issue verified for these inputs; it stops
on other errors or mismatched surfaces. If new exports pass inspection directly,
conversion can use the original STEP files without a repair report.

`convert_to_dagmc.py` starts with Gmsh, imprints shared interfaces, checks the
resulting solid count and topology, and writes `outputs/dagmc/tokamak.h5m`.
It retains the assembled BREP, material manifest, conversion log, and H5M
metadata correction record. On a full conversion failure it attempts progressively
larger layer/magnet subsets as diagnostics, without replacing the full model.
The archived `diagnose_interfaces.py` is a diagnostic for the original unmodified
exports; it records the failed original interface and small subset conversions.

All geometric coordinates imported by CadQuery are mm, although the STEP
headers declare metres. The verified conversion is 0.1 cm/mm. The resulting
geometry is 1220 cm wide and 854 cm high. There is no per-part recentering.

Both original-CAD and OpenMC plots work without a desktop. OpenMC cell colors
distinguish the void plasma from the surrounding void. The regular views are
XZ through y=0, XY through z=0, YZ through x=0, XZ through y=100 cm, XY through
z=100 and -200 cm, and a poloidal layer close-up. The fixed offsets/zoom are near
the top of the archived `plot_cad.py` and active `plot_geometry.py` for adjustment.

Open `outputs/plots/tokamak_surfaces.vtk` in ParaView and color by `volume_id`.
This is the actual triangulated H5M surface mesh, rather than a voxel view.
`original_cad_surfaces.vtk` is a separate tessellation of the original CAD, with
component IDs in config order. Shared interfaces are duplicated between
neighbouring display volumes in the H5M visualization.

## Mesh and transport limits

The coarse profile uses 0.5–15 cm surface edge sizes with a 3 cm first-wall
target. The fine option uses 0.2–5 cm and a 1 cm first-wall target. Change
`mesh.profile` in config.yaml, then repeat inspection, repair, conversion, and
validation. Fine meshing can use substantially more memory. All meshing and
OpenMC jobs here use two threads. On a shared cluster, run these commands in an
allocation; the scripts do not submit scheduler jobs.

Validation checks per-volume triangle closure and orientation, degenerate
triangles, material membership and counts, bounds, mesh/CAD volumes, and the
installed DAGMC tools. The overlap checker is non-exhaustive and has a bounded
runtime. A missing tool, failure, or unresolved overlap check blocks transport.
An H5M metadata compatibility correction is needed for the pinned writer:
`FACETING_TOL` must be a sparse tag on a meshset. The converter adds it while
retaining the writer's estimate; it changes no mesh coordinates.

The archived `run_openmc.py` has separate setup/export, transport, and analysis cells.
The default test is 3 batches × 1000 isotropic 14.1 MeV neutrons, with source
rejection into the actual plasma cell and a 40³ neutron-flux mesh. It runs only
after geometry checks pass and the executable and all required nuclear-data
files are available. Set `openmc.run_transport: false` for export only. Inputs
are kept in `outputs/simulation/`, and each transport case has its own
`outputs/simulation/cases/test_<timestamp>/` folder. The CSV tally is track
length per source; the plotted flux also divides by mesh-bin volume.

An exterior vacuum box terminates particles; gaps and the DAGMC implicit
complement are void. The plasma has the reserved `mat:Vacuum` tag, with no
fictional nuclear material named Vacuum. The source samples a bounding
cylinder uniformly in volume and rejects points outside the plasma, producing
a distributed toroidal source without assuming a circular plasma cross section.

Successful plotting initializes the real DAGMC engine, but does not validate
transport. A geometry-debug attempt stopped at the plasma/implicit-complement
boundary. This matches the reported limitation in
[OpenMC issue #3023](https://github.com/openmc-dev/openmc/issues/3023); the failed
case and `geometry_debug_attempt.json` are retained. Geometry-debug is disabled
for the normal test; the independent DAGMC checks remain required.
The small transport test is a diagnostic;
it does not establish mesh/statistical convergence or validate the provisional
material assumptions for SPARC. Keep those limits when interpreting results.

## Notebooks

The `.py` files are the editable source. Corresponding `.ipynb` files were
generated with Jupytext and are not executed copies. After editing a script:

```bash
jupytext --to ipynb plot_geometry.py
```

Run notebooks with the same project environment selected as the kernel and
this directory as the working directory. Each script's execution cells also
work with Run All. Shared helper modules do not need separate notebooks.

## Documentation used

- [cad_to_dagmc documentation](https://fusion-energy.github.io/cad_to_dagmc/)
- [cad_to_dagmc Python API](https://fusion-energy.github.io/cad_to_dagmc/dev/api.html)
- [OpenMC CAD geometry](https://docs.openmc.org/en/stable/usersguide/geometry.html#using-cad-based-geometry)
- [DAGMC tools](https://svalinn.github.io/DAGMC/usersguide/tools.html)
- [DAGMC faceting tag reader](https://github.com/svalinn/DAGMC/blob/develop/src/dagmc/DagMC.cpp)
- [Gmsh API](https://gmsh.info/doc/texinfo/gmsh.html)

The installed package signatures were also inspected: cad_to_dagmc 0.15.1 uses
`imprint='partial'` and `imprint_threads`, which differ from some online examples.

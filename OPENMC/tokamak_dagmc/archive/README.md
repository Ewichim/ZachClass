# Optional tools

These scripts and their matching notebooks were moved here unchanged to keep
the active geometry workflow easier to follow.

| Script | Purpose |
| --- | --- |
| `diagnose_interfaces.py` | Investigates the original STEP interface failure and tries small subset meshes. |
| `run_openmc.py` | Runs the original short 3000-history transport check. |
| `inspect_environment.py` | Records software versions and nuclear-data availability. |
| `plot_cad.py` | Generates views of the original STEP geometry. |
| `write_report.py` | Regenerates the geometry summary from saved reports. |

The active CAD workflow and first-wall flux study do not import these scripts.
Existing geometry, reports, plots, and transport results remain in `../outputs/`.
The separate downloaded examples remain in `../../examples/`.

To use an archived tool again, move its `.py` and `.ipynb` files back to
`tokamak_dagmc/` and run it from there using the CAD environment described in
the project README. The original scripts expect that location for their imports
and paths. Restore both `inspect_environment` and `run_openmc` if you want to
use the environment checker, because it imports a helper from `run_openmc.py`.

Running a restored tool can regenerate or archive outputs; it is not necessary
for the regular first-wall calculation. Use `../../first_wall_flux/first_wall_flux.py`
or its notebook for that calculation.

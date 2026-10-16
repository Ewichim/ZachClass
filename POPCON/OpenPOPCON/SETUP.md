# OpenPOPCON tutorial setup

Use the existing environment:

```bash
conda activate ZClass
cd /home/youeric/openmc-projects/ZachClass/POPCON/OpenPOPCON
jupyter lab tutorial.ipynb
```

Select **Python (ZClass)** and run the cells from the top. The installed
OpenPOPCON is 2.2.0 at commit
`9e70f06df1e8c16df0191730036f5cd00d9e904c`. No equilibrium file, additional
download, or OpenMC nuclear data is needed for this tutorial. First execution
can take longer because Numba compiles the routines.

If that kernel is missing after recreating the environment, register it with
`python -m ipykernel install --sys-prefix --name zclass --display-name 'Python (ZClass)'`.

For a normal script run:

```bash
python tutorial.py
```

The source is `tutorial.py`, with Jupytext percent-format cells and markdown.
After editing it, regenerate the notebook:

```bash
jupytext --to ipynb tutorial.py
```

This regeneration does not execute the notebook. Run all cells to refresh its
visible outputs. Both script and notebook also work from the ZachClass root;
otherwise launch the notebook from this folder.

## Inputs

- `inputs/upstream_SPARC/`: original installed example files, kept unchanged.
- `inputs/fire_h_mode.yaml`: FIRE machine values plus documented teaching assumptions.
- `inputs/plotsettings.yml`: contour variables, levels, and axis choices for FIRE.

Important OpenPOPCON conventions: current in MA; lengths in m; field in T;
density axes in 10^20 m^-3; temperature in keV; powers in MW. The input
temperature range is volume-average Ti. Density fraction uses volume-average
density divided by the Greenwald density. The tutorial explains the distinction
from the usual line-average definition.

The single analytic elongation is set to the published areal value. Ti = Te,
the current profile, impurity species, and small edge temperature floor are
modelling choices. The 30 MW heating cap and beta_N = 3 limit are illustrative.
The script does not claim to enforce all tokamak operating limits.

## Outputs

Each run creates `outputs/<timestamp>/`. This contains setup/profile plots,
POPCON figures, `FIRE_grid.csv`, `confinement_scan.csv`, and `run_summary.json`.
Inputs and full xarray results (`results.nc`) are in `cases/<case_name>/`.
Earlier runs are preserved. Outputs are ignored by Git.

OpenPOPCON marks failed solutions with a large sentinel; the tutorial masks
them. Zero-clipped auxiliary heating is excluded from the selected operating
region. Diagnostic messages about invalid points are expected on this grid.

The CSV grid variables follow the package names; `Paux`, `Pfusion`, `Prad`,
and other power columns are in MW, `tauE` in s, and `betaN` uses beta in percent.
NetCDF variables carry unit attributes. To reopen saved arrays:

```python
import xarray as xr
results = xr.load_dataset("outputs/<run>/cases/02_FIRE_H_mode/results.nc")
```

See `VERIFICATION.md` for the checks performed on the delivered tutorial.

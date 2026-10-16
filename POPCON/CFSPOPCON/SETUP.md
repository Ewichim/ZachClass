# CFSPOPCON tutorial setup

```bash
conda activate ZClass
cd /home/youeric/openmc-projects/ZachClass/POPCON/CFSPOPCON
jupyter lab tutorial.ipynb
```

Select **Python (ZClass)** and run cells in order. The installed versions are
CFSPOPCON 9.0.0 and RADAS `1!1.0.1`. The packages stay in `ZClass`.

If the kernel is missing after recreating the environment, register it with
`python -m ipykernel install --sys-prefix --name zclass --display-name 'Python (ZClass)'`.

## Atomic tables: once per checkout

The needed tables have already been generated on this computer under
`radas_dir/`. They are local files, ignored by Git. A fresh checkout needs:

```bash
radas --config inputs/radas_tutorial.yaml --directory radas_dir
```

RADAS downloads OpenADAS plasma atomic rates and computes tables for helium,
oxygen, tungsten, xenon, and argon. This can take several minutes. It does not
use or change OpenMC nuclear data. The tutorial checks for all five tables and
does not download data during a normal run. Upstream says the atomic data
cannot be distributed with its package; keep the generated data local.

The tutorial atomic grid is 12 densities by 40 temperatures, with the same
rate datasets and ne*tau value as the upstream configuration. This is a coarse
teaching setup, not a convergence study. The original configuration is retained
as `inputs/radas_upstream.yaml` (50 by 100). To try that resolution while
preserving the existing tables, generate a separate folder:

```bash
radas --config inputs/radas_upstream.yaml --directory radas_fine \
  --species helium --species oxygen --species tungsten --species xenon --species argon
```

Then change `data_dir` near the top of `tutorial.py` to `tutorial_dir / "radas_fine"`
and `atomic_config_file` to `tutorial_dir / "inputs/radas_upstream.yaml"`.
Keep this configuration path matched to the tables you generated; the run
summary records it. Regenerate the notebook after edits.
RADAS uses existing raw downloads when available but recomputes requested
tables; do not regenerate over a folder whose old tables you want to preserve.
Check the rate-data usage terms before sharing atomic tables.

RADAS may report nearest-neighbour extrapolation for oxygen charge-exchange
rates outside the source dataset's domain during setup. The POPCON's own
atomic interpolations can also be extrapolated at profile edges. Keep those
messages visible and check ranges before extending the model.

## Tutorial files and modifications

For a script run, use `python tutorial.py`. Edit that file first, then:

```bash
jupytext --to ipynb tutorial.py
```

Regeneration does not run calculations. Run all notebook cells to refresh the
displayed outputs. The script and notebook also support starting from the
ZachClass root; otherwise launch the notebook from this folder.

- `inputs/upstream_SPARC_PRD/`: original input and both plot styles, with license.
- `inputs/fire_h_mode.yaml`: shorter FIRE core recipe and documented assumptions.
- `inputs/plot_fire.yaml`: native CFSPOPCON contour settings.
- `inputs/radas_tutorial.yaml`, `inputs/radas_upstream.yaml`: atomic-data configurations.

The upstream SPARC physical input and 40 by 30 grid are retained. Using coarse
atomic tables means its figure need not exactly match the upstream regression
reference. FIRE deliberately omits SPARC solenoid and divertor settings.

YAML units are A, m, T, keV, MW, and density in 10^19 m^-3. The plot converts
density to 10^20 m^-3. `normalized_beta` defaults to m percent T / MA, so its
numeric value already uses beta in percent. Ti = Te and impurity species are
assumptions; Zeff is calculated, not fixed. Heating and beta caps are examples.

## Outputs

Each run creates `outputs/<timestamp>/`, with plots, a power-budget table,
the FIRE grid CSV, the confinement scan CSV, and `run_summary.json` at its top
level. Case inputs and full NetCDF results are in `cases/<case_name>/`.
Earlier runs are preserved; all run folders are ignored by Git.

The full dataset includes clipped-heating points. The tutorial flags and masks
unphysical or excess-self-heating points before shading the selected operating
region. A successful calculation is not a validation of all physics assumptions.
Reload a full dataset, including its units and named options, with:

```python
from pathlib import Path
from cfspopcon.file_io import read_dataset_from_netcdf
results = read_dataset_from_netcdf(Path("outputs/<run>/cases/02_FIRE_H_mode/results.nc"))
```

See `VERIFICATION.md` for the checks performed on the delivered tutorial.

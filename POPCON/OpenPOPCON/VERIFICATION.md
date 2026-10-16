# Tutorial checks

Checked on 2026-10-15 in Conda environment `ZClass`, OpenPOPCON 2.2.0.

- Ran `tutorial.py` from the ZachClass root through all five cases.
- Generated `tutorial.ipynb` with Jupytext and executed it from its own folder
  with the registered `zclass` kernel. All 19 code cells completed in order,
  with no error outputs and seven inline figures.
- Confirmed that every notebook cell matches the Python source.
- Confirmed 1,023 FIRE grid rows (31 densities by 33 temperatures), five saved
  case folders, CSV/summary agreement, and readable NetCDF results.
- Checked the package's Q definition on valid FIRE points. Required heating
  decreases as H98 increases at the same inspection point, while fusion power
  stays unchanged with the other inputs fixed.
- Checked SHA-256 hashes of the copied upstream inputs against `sources.json`.
- Visually checked the FIRE POPCON and setup/profile plots.
- After the beginner rewrite, checked that all five cases agree with the
  previous tutorial's plasma results. Verified the package-specific heating,
  loss, and radiation accounting explained in the revised markdown.

The executed notebook run is saved locally at
`outputs/20261015_192828_579090/`. Earlier script and notebook runs are preserved.
Outputs are ignored by Git; the executed notebook retains its displayed figures.

Notebook execution command, after activating `ZClass`:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 tutorial.ipynb
```

The agent's restricted filesystem required temporary Matplotlib/Numba/Jupyter
cache locations during checks. Those were command-local settings, not changes
to the tutorial or the Conda environment configuration.

These checks establish that the tutorial runs and saves consistent outputs.
They do not establish grid convergence or validate FIRE's performance claims.

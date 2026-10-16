# Tutorial checks

Checked on 2026-10-15 in Conda environment `ZClass`, CFSPOPCON 9.0.0 and RADAS
`1!1.0.1`. Generated the five required plasma atomic tables using the supplied
12 by 40 tutorial configuration.

- Ran the Python workflow, then generated the notebook with Jupytext and
  executed the final version from its own folder with the `zclass` kernel.
  All 26 code cells completed in order, with no error outputs and five inline
  figures. All notebook cells match the Python source.
- Ran the original SPARC PRD physical input on its 40 by 30 POPCON grid, the
  FIRE input on a 31 by 33 grid, and three confinement cases.
- Confirmed 1,023 FIRE CSV rows, five saved case folders, CSV/summary agreement,
  readable NetCDF results, and correct FIRE minor radius.
- Checked heating balance where the signed auxiliary requirement is positive,
  and checked the package's zero-clipped P_SOL = max(P_in - P_radiation, 0).
- Confirmed decreasing auxiliary heating as H98 increases at a fixed point,
  unchanged fusion power, and a false physical-point flag for the scan point
  with excess self-heating. Large clipped-region Q is not treated as a solution.
- Reloaded and validated all five saved input snapshots, including their
  relative atomic-data paths. Checked copied-source hashes against `sources.json`.
- Visually checked the FIRE operating-limit plot and setup/profile plots.
- After the beginner rewrite, verified that the added single-formula and
  Algorithm/Dataset examples execute, and that saved plasma results in all
  five cases agree with the previous tutorial's results. Checked the heating
  balance described in the new markdown.

The executed notebook run is saved locally at
`outputs/20261015_192828_917990/`. Earlier script and notebook runs are preserved.
Outputs and generated atomic tables are ignored by Git. The notebook retains
its displayed figures.

Notebook execution command, after activating `ZClass`:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 tutorial.ipynb
```

The agent used temporary Matplotlib/Jupyter caches during restricted execution.
These were command-local settings. No nuclear-data settings were changed.

Atomic-data setup emitted oxygen charge-exchange extrapolation warnings at
the edge of its source domain. The tutorial uses coarse atomic tables and
does not claim agreement with the upstream fine-grid regression reference.
Neither atomic-grid/POPCON convergence nor the published FIRE claims have
been validated by these workflow checks.

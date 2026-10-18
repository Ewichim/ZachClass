# First-wall neutron flux

This study uses the validated model in `../tokamak_dagmc/` to calculate neutron
flux averaged over the complete first-wall volume. It adds an energy spectrum
and a one-hour wall-clock target. The original CAD and conversion outputs are
not modified, and no new environment or nuclear-data library is needed.

A short check completed 1000 source histories in two batches, exercised the
time-based stop, and generated the flux CSV and spectrum. Its energy-bin
fluxes sum to the separately tallied total flux. The one-hour inputs have
been exported, but the long run has not been launched.

Run it with the existing class environment:

```bash
cd /home/youeric/openmc-projects/ZachClass/OPENMC/first_wall_flux
conda activate ZClass
python first_wall_flux.py
```

`first_wall_flux.ipynb` contains the same cells: setup and export, transport,
then analysis. Select the ZClass kernel and start the notebook in this folder.
Running all cells starts the one-hour simulation. Change the parameters near
the top of the Python file, then regenerate the notebook with:

```bash
jupytext --to ipynb first_wall_flux.py
```

## Timing

The default target is **3600 seconds**, including OpenMC initialization and
transport. Setup plotting, input export, finalization, and analysis add some
time. The Python driver checks elapsed time between completed batches and then
writes a final statepoint. It can exceed the target by approximately one batch;
it does not terminate OpenMC in the middle of a particle history.

There are 5000 source histories per batch and two OpenMP threads. The previous
small model ran about 480 histories/s, corresponding to roughly 10 seconds per
batch and 1.7 million histories in an hour. This study's tallies and machine load
can change that rate. The actual number of batches and source histories is
recorded, rather than assumed from the estimate.

`maximum_batches=10000` is a safety cap, and at least two batches are completed
to obtain a statistical uncertainty. `checkpoint_seconds=600` saves tally
checkpoints at completed batches about every ten minutes. Older run folders
are preserved.

Use the Python script or notebook to apply the time target: running the exported
XML directly with `openmc` would instead use the maximum batch count. Set
`run_transport=False` to export inputs and the setup plot without starting
transport. On a shared cluster, launch this script inside a compute allocation.

## What is tallied

The source is the same 14.1 MeV isotropic, uniformly distributed plasma source
as in the validated model. Rejection sampling confines accepted source sites
to the actual DAGMC plasma cells. The exterior box has a vacuum boundary.

The tally uses a neutron filter and the unique first-wall material tag. This
includes every first-wall solid and excludes the plasma, blanket, vessel, and
magnets. A single aggregate tally retains correlations between wall regions.
The total tally is calculated separately from its energy bins; uncertainties
from correlated bins are not added in quadrature.

OpenMC's track-length flux score is integrated over the tally volume. The
script divides both the mean and its standard deviation by the **actual
first-wall triangle-mesh volume** recorded in the validation report, giving
the volume-averaged scalar flux in **cm⁻² per source neutron**. It is not the
surface incident current, nor a spatial map of local hot spots.

Set `source_strength_n_per_s` if a physical source rate is known. Both flux
and standard deviation are then multiplied by that rate to give n/cm²/s.
There is no assumed SPARC source strength. A one-hour computing budget is not
a one-hour irradiation time; it cannot determine physical fluence by itself.

## Outputs

Each execution creates:

```text
outputs/first_wall_<timestamp>/
├── run_summary.json
├── setup.png
├── first_wall_flux.py
├── geometry_config.yaml
├── dagmc_validation.json
├── topology_repair.json
├── flux_history.csv
├── first_wall_flux.json
├── first_wall_flux.csv
├── energy_spectrum.csv
├── energy_spectrum.png
└── cases/dt_14p1MeV/
    ├── tokamak.h5m
    ├── geometry.xml
    ├── materials.xml
    ├── settings.xml
    ├── tallies.xml
    ├── summary.h5
    ├── checkpoint.<batch>.h5
    └── statepoint.final.h5
```

The prepared run contains inputs, metadata, and a geometry image before any
transport is launched. Final flux files appear only after transport and
analysis. Flux uncertainties are one standard deviation of the mean. Energy
bins are integrated fluxes, not differential flux per MeV. The CSV includes the
lowest bin starting at zero; that bin is omitted from the logarithmic energy
plot. `flux_history.csv` records the accumulating total flux and its uncertainty.

The tungsten first wall and other material compositions remain provisional.
The geometry uses the existing coarse surface mesh; a longer transport run
reduces sampling uncertainty but does not establish mesh or material accuracy.
DAGMC watertightness, overlap checks, and matching input hashes are required
before setup. Geometry-debug mode is left off because of the documented
[DAGMC boundary-check issue](https://github.com/openmc-dev/openmc/issues/3023).

The driver uses the documented [OpenMC batch API](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.lib.iter_batches.html)
and [statepoint writer](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.lib.statepoint_write.html).
Tally units follow the [OpenMC tally guide](https://docs.openmc.org/en/stable/usersguide/tallies.html).

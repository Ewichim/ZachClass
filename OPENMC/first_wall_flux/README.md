# First-wall neutron flux

This study uses the validated model in `../tokamak_dagmc/` to calculate neutron
flux averaged over the complete first-wall volume and an energy spectrum.
It runs a fixed number of source histories. The original CAD and conversion
outputs are not modified, and no new environment or nuclear-data library is needed.

Run it with the existing class environment:

```bash
cd /home/youeric/openmc-projects/ZachClass/OPENMC/first_wall_flux
conda activate ZClass
python first_wall_flux.py
```

`first_wall_flux.ipynb` contains the same cells: setup and export, transport,
then analysis. Select the ZClass kernel and start the notebook in this folder.
Running all cells starts the fixed-count simulation. Change the parameters near
the top of the Python file, then regenerate the notebook with:

```bash
jupytext --to ipynb first_wall_flux.py
```

## Sample count

The defaults are `batches = 100` and `particles_per_batch = 5000`, giving
**500,000 source histories**. Change these values near the top of the script.
Use at least two batches to estimate statistical uncertainty. OpenMC uses two
OpenMP threads and writes the final statepoint after the last batch.

The transport cell calls `openmc.run()` using the exported inputs. The script,
notebook, and exported XML all use the same batch and particle counts.
Set `run_transport = False` to export inputs and the setup plot without starting
transport. Older run folders are preserved. On a shared cluster, launch this
script inside a compute allocation.

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
There is no assumed SPARC source strength. The number of simulated histories
does not set a physical irradiation time or fluence.

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
    └── statepoint.<batches>.h5
```

The prepared run contains inputs, metadata, and a geometry image before any
transport is launched. Final flux files appear only after transport and
analysis. Flux uncertainties are one standard deviation of the mean. Energy
bins are integrated fluxes, not differential flux per MeV. The CSV includes the
lowest bin starting at zero; that bin is omitted from the logarithmic energy
plot. The run summary records the batch and source-history counts.

The tungsten first wall and other material compositions remain provisional.
The geometry uses the existing coarse surface mesh; a longer transport run
reduces sampling uncertainty but does not establish mesh or material accuracy.
DAGMC watertightness, overlap checks, and matching input hashes are required
before setup. Geometry-debug mode is left off because of the documented
[DAGMC boundary-check issue](https://github.com/openmc-dev/openmc/issues/3023).

The transport cell uses [openmc.run](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.run.html).
Tally units follow the [OpenMC tally guide](https://docs.openmc.org/en/stable/usersguide/tallies.html).

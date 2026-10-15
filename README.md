# ZachClass

Setup for the MIT NSE 22.S906 neutronics and POPCON class projects.

## Use the environment

```bash
conda activate ZClass
jupyter lab
```

Select **Python (ZClass)** in Jupyter. The environment contains OpenMC 0.16.0,
OpenPOPCON 2.2.0, CFS POPCON 9.0.0, and the usual scientific Python and notebook
tools. Both POPCON packages install and import in this environment, so a second environment
was unnecessary. The existing `OpenMC` environment was left unchanged.

## Files

- `NeutronicsProject.pdf` and `POPCONProject.pdf`: assignment instructions.
- [OPENMC/examples](OPENMC/examples/README.md): downloaded CSG geometry examples
  and generated notebooks.
- [OPENMC/data](OPENMC/data/README.md): linked ENDF/B-VIII.1 data and downloaded
  FENDL 3.2 data, excluded from Git.
- [POPCON/references/FIRE](POPCON/references/FIRE/README.md): two FIRE papers and
  a CSV of published machine parameters.
- [setup/installation-checks.txt](setup/installation-checks.txt): import,
  executable, dependency, model-loading, and data-file checks. No simulations
  or POPCON scans were run.
- [setup/downloads.json](setup/downloads.json): download sources and SHA-256 hashes.

OpenPOPCON includes worked examples. To copy them into this folder later:

```bash
openpopcon examples POPCON/examples/openpopcon
```

Package documentation: [OpenPOPCON](https://github.com/hansec/OpenPOPCON) and
[CFS POPCON](https://cfspopcon.readthedocs.io/en/latest/).

## Recreate the environment

On another computer, first change the nuclear-data path in `environment.yml`.
Then run:

```bash
conda env create --file environment.yml
```

For the exact installed Linux packages, use the Conda package list followed by
the pip requirements:

```bash
conda create --name ZClass --file setup/conda-linux-64.txt
conda run -n ZClass python -m pip install -r setup/pip-requirements.txt
```

The exact-package commands do not configure the data path or register the
notebook kernel. Those steps are described in [setup/README.md](setup/README.md).

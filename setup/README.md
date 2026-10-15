# Installation records

`environment.yml` at the project root lists the requested dependencies.
`conda-linux-64.txt` records exact Conda package builds;
`pip-requirements.txt` records the six pip-installed packages.

After recreating the environment, configure the local data path:

```bash
mkdir -p OPENMC/data
ln -s /home/youeric/nuclear_data/endfb-viii.1/endfb-viii.1-hdf5 OPENMC/data/endfb-viii.1
conda env config vars set -n ZClass OPENMC_CROSS_SECTIONS=/home/youeric/ZachClass/OPENMC/data/endfb-viii.1/cross_sections.xml
conda run -n ZClass python -m ipykernel install --sys-prefix --name ZClass --display-name 'Python (ZClass)'
```

Skip the link command if it already exists. Adjust paths on other computers.
For FENDL, download and extract the archive linked in `downloads.json` under
`OPENMC/data/fendl-3.2/`.

`installation-checks.txt` records setup checks only. Model loading and opening
data files do not verify transport results or the physics of the examples.
`downloads.json` records the original downloaded files; generated notebooks
and the manually transcribed FIRE CSV are not upstream downloads.

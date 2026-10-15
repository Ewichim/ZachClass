# Nuclear data

`endfb-viii.1` is a symbolic link to the existing library:

```text
/home/youeric/nuclear_data/endfb-viii.1/endfb-viii.1-hdf5
```

`ZClass` uses `endfb-viii.1/cross_sections.xml` by default. No copy of the
ENDF library was made. OpenMC resolves the link to the original location.

FENDL 3.2 was missing, so the archive from the
[OpenMC data page](https://openmc.org/data/) was downloaded and extracted here.
Its index is `fendl-3.2/fendl-3.2-hdf5/cross_sections.xml`.

To choose FENDL for a shell session, after activating `ZClass`:

```bash
export OPENMC_CROSS_SECTIONS=/home/youeric/ZachClass/OPENMC/data/fendl-3.2/fendl-3.2-hdf5/cross_sections.xml
```

The data and archives are ignored by Git. Recreating the Conda environment
does not download nuclear data or recreate the link.

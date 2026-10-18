# %%
"""Record versions and existing data without changing nuclear-data settings."""

from importlib.metadata import version
import os
from pathlib import Path
import platform
import shutil
import sys

import openmc
import openmc.lib

from materials import make_materials
from run_openmc import check_nuclear_data
from workflow import load_config, output_paths, write_json

config = load_config()
paths = output_paths(config)


def inspect_environment():
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "host": platform.node(),
              "conda_prefix": os.getenv("CONDA_PREFIX"), "openmc_version": openmc.__version__,
              "openmc_dagmc_enabled": openmc.lib._dagmc_enabled(), "versions": {},
              "executables": {}, "thread_limit_in_config": config["mesh"]["threads"]}
    for package in ("cad_to_dagmc", "cadquery", "cadquery-ocp", "gmsh", "h5py", "jupytext"):
        report["versions"][package] = version(package)
    for executable in ("openmc", "check_watertight", "overlap_check", "sbatch", "srun", "qsub"):
        report["executables"][executable] = shutil.which(executable)
    report["nuclear_data"] = check_nuclear_data(make_materials(config))
    write_json(paths["reports"] / "environment.json", report)
    print(report)
    return report


# %%
if __name__ == "__main__":
    environment = inspect_environment()

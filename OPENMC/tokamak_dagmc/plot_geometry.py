# %%
"""OpenMC geometry sections, using cell colors so the void plasma is visible."""

import json
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import openmc

from materials import make_materials
from mesh_checks import read_mesh
from workflow import (build_geometry, load_config, output_paths,
                      file_hash, require_validation, write_json)

config = load_config()
paths = output_paths(config)


def make_plots(geometry, materials, h5m, folder, prefix="tokamak"):
    coordinates, triangles, volumes, senses = read_mesh(h5m)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    center = (lower + upper) / 2
    extent = (upper - lower) * 1.08
    color_for_tag = {component["tag"]: component["color"] for component in config["components"]}
    colors = {volume["id"]: tuple(round(255 * value) for value in to_rgb(color_for_tag[volume["material"]]))
              for volume in volumes}
    # DAGMC creates its implicit complement after the highest explicit cell ID.
    colors[max(volume["id"] for volume in volumes) + 1] = (242, 242, 242)
    colors[10000] = (242, 242, 242)
    sections = [
        (f"{prefix}_xy", "xy", [0, 0, 0], (extent[0], extent[1])),
        (f"{prefix}_xz", "xz", [0, 0, 0], (extent[0], extent[2])),
        (f"{prefix}_yz", "yz", [0, 0, 0], (extent[1], extent[2])),
        (f"{prefix}_xz_y100", "xz", [0, 100, 0], (extent[0], extent[2])),
        (f"{prefix}_xy_z100", "xy", [0, 0, 100], (extent[0], extent[1])),
        (f"{prefix}_xy_zminus200", "xy", [0, 0, -200], (extent[0], extent[1])),
        ("poloidal_zoom" if prefix == "tokamak" else f"{prefix}_zoom", "xz", [325, 0, 0], (220, 180)),
    ]
    plots = openmc.Plots()
    for name, basis, origin, width in sections:
        plot = openmc.SlicePlot()
        plot.filename = name + "_raw"
        plot.basis = basis
        plot.origin = origin
        plot.width = width
        plot.pixels = config["plots"]["pixels"]
        plot.color_by = "cell"
        plot.colors = colors
        plot.background = "white"
        plots.append(plot)
    folder.mkdir(parents=True, exist_ok=True)
    geometry.export_to_xml(folder / "geometry.xml")
    materials.export_to_xml(folder / "materials.xml")
    plots.export_to_xml(folder / "plots.xml")
    # Plot mode does not require nuclear data. It still initializes DAGMC in C++.
    # The Python plotting runner has no thread argument in OpenMC 0.16.
    command = [shutil.which("openmc") or "openmc", "-p", "-s", str(config["openmc"]["threads"])]
    completed = subprocess.run(command, cwd=folder, capture_output=True, text=True)
    (folder / "openmc_plot.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"OpenMC plot initialization failed. Read {folder / 'openmc_plot.log'}")
    for name, basis, origin, width in sections:
        raw = folder / (name + "_raw.png")
        image = mpimg.imread(raw)
        first, second = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[basis]
        bounds = [origin[first] - width[0] / 2, origin[first] + width[0] / 2,
                  origin[second] - width[1] / 2, origin[second] + width[1] / 2]
        figure, axis = plt.subplots(figsize=(9, 9))
        axis.imshow(image, extent=bounds, interpolation="nearest")
        axis.set_xlabel(f"{'xyz'[first]} (cm)")
        axis.set_ylabel(f"{'xyz'[second]} (cm)")
        axis.set_title(f"OpenMC {basis} section: {prefix}")
        legend = [Patch(color=color_for_tag[tag], label="Plasma (Vacuum)" if tag == "Vacuum" else tag)
                  for tag in sorted(set(v["material"] for v in volumes))]
        legend.append(Patch(color="#f2f2f2", label="Void gaps"))
        axis.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1))
        figure.savefig(folder / (name + ".png"), dpi=140, bbox_inches="tight")
        plt.close(figure)
    return [name + ".png" for name, basis, origin, width in sections]


# %%
if __name__ == "__main__":
    require_validation(config, paths)
    geometry, dagmc = build_geometry(config, paths)
    images = make_plots(geometry, make_materials(config), paths["dagmc"] / "tokamak.h5m", paths["plots"])
    write_json(paths["reports"] / "openmc_plots.json", {"success": True, "images": images,
                                                       "h5m_sha256": file_hash(paths["dagmc"] / "tokamak.h5m"),
                                                       "cpp_dagmc_initialized": True,
                                                       "transport_validated": False})

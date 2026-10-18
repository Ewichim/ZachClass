# %%
"""Headless previews of the original CAD, available even if conversion is blocked."""

import json
import math

import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

from inspect_step import import_shape
from workflow import ROOT, load_config, output_paths

config = load_config()
paths = output_paths(config)


def plot_cad():
    components = []
    scale = config["units"]["imported_to_cm"]
    for component in config["components"]:
        shape = import_shape(ROOT / config["step_directory"] / component["file"])
        components.append((component, shape))
    sections = [
        ("xy", 0.0, None), ("xz", 0.0, None), ("yz", 0.0, None),
        ("xz", 1000.0, None), ("xy", 1000.0, None), ("xy", -2000.0, None),
        ("xz", 0.0, (225, 425, -80, 80)),
    ]
    axes_for_basis = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
    for number, (basis, offset_mm, zoom) in enumerate(sections):
        horizontal, vertical, normal = axes_for_basis[basis]
        origin = [0.0, 0.0, 0.0]
        origin[normal] = offset_mm
        direction = [0.0, 0.0, 0.0]
        direction[normal] = 1.0
        plane = gp_Pln(gp_Pnt(*origin), gp_Dir(*direction))
        figure, axis = plt.subplots(figsize=(10, 9))
        for component, shape in components:
            section = BRepAlgoAPI_Section(shape.wrapped, plane, False)
            section.Build()
            if not section.IsDone():
                raise RuntimeError(f"Section failed: {component['file']} {basis}")
            for edge in cq.Shape.cast(section.Shape()).Edges():
                points = []
                for fraction in np.linspace(0.0, 1.0, 120):
                    points.append(edge.positionAt(float(fraction)).toTuple())
                points = np.array(points) * scale
                axis.plot(points[:, horizontal], points[:, vertical],
                          color=component["color"], linewidth=1.2)
        if zoom:
            axis.set_xlim(zoom[:2])
            axis.set_ylim(zoom[2:])
        axis.set_aspect("equal")
        axis.set_xlabel(f"{'xyz'[horizontal]} (cm)")
        axis.set_ylabel(f"{'xyz'[vertical]} (cm)")
        axis.grid(alpha=0.2)
        axis.set_title(f"Original CAD section: {basis}, {'xyz'[normal]}={offset_mm * scale:g} cm\n"
                       "OpenCascade outlines; not an OpenMC geometry validation")
        handles = [Line2D([0], [0], color=c["color"], label=c["tag"])
                   for c, shape in components]
        axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1))
        filename = f"cad_{basis}_{offset_mm * scale:g}cm.png"
        if zoom:
            filename = "cad_poloidal_zoom.png"
        figure.savefig(paths["plots"] / filename, dpi=140, bbox_inches="tight")
        plt.close(figure)

    # A triangle surface mesh for rotation in ParaView, with component and solid IDs.
    vertices = []
    triangles = []
    component_ids = []
    solid_ids = []
    solid_id = 0
    for component_id, (component, shape) in enumerate(components, 1):
        for solid in shape.Solids():
            solid_id += 1
            points, faces = solid.tessellate(5.0, 0.15)
            first_vertex = len(vertices)
            vertices.extend(tuple(v * scale for v in p.toTuple()) for p in points)
            triangles.extend(tuple(v + first_vertex for v in face) for face in faces)
            component_ids.extend([component_id] * len(faces))
            solid_ids.extend([solid_id] * len(faces))
    with (paths["plots"] / "original_cad_surfaces.vtk").open("w") as stream:
        stream.write("# vtk DataFile Version 3.0\nOriginal CAD surfaces, coordinates cm\nASCII\nDATASET POLYDATA\n")
        stream.write(f"POINTS {len(vertices)} double\n")
        for point in vertices:
            stream.write(f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g}\n")
        stream.write(f"POLYGONS {len(triangles)} {4 * len(triangles)}\n")
        for face in triangles:
            stream.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        stream.write(f"CELL_DATA {len(triangles)}\n")
        for name, values in [("component_id", component_ids), ("solid_id", solid_ids)]:
            stream.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
            for value in values:
                stream.write(f"{value}\n")
    print(f"CAD sections and triangle surface VTK written to {paths['plots']}")


# %%
if __name__ == "__main__":
    plot_cad()

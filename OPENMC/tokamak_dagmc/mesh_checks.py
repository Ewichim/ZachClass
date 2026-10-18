"""Read the MOAB H5M sets written by the pinned cad_to_dagmc h5py backend."""

from collections import Counter

import h5py
import numpy as np


def fix_faceting_tag(filename):
    """Correct cad_to_dagmc 0.15.1's tag name/storage for DAGMC 3.2.4."""
    with h5py.File(filename, "r+") as stream:
        tags = stream["tstt/tags"]
        if "FACETING_TOL" in tags:
            return float(tags["FACETING_TOL/values"][0])
        # The writer's estimate is retained; this does not remesh or move nodes.
        tags.copy("FACETING_TOLERANCE", "FACETING_TOL")
        tag = tags["FACETING_TOL"]
        tag.attrs.modify("class", np.int32(1))  # MOAB sparse tag.
        sets = stream["tstt/sets/list"]
        file_set = int(sets.attrs["start_id"]) + len(sets) - 1
        tag["id_list"][:] = [file_set]
        return float(tag["values"][0])


def text_tag(value):
    return value.tobytes().decode().replace("\x00", "")


def read_mesh(filename):
    with h5py.File(filename, "r") as stream:
        mesh = stream["tstt"]
        coordinates = mesh["nodes/coordinates"][:]
        node_start = int(mesh["nodes/coordinates"].attrs["start_id"])
        triangles = mesh["elements/Tri3/connectivity"][:].astype(np.int64) - node_start
        triangle_start = int(mesh["elements/Tri3/connectivity"].attrs["start_id"])
        rows = mesh["sets/list"][:]
        set_start = int(mesh["sets/list"].attrs["start_id"])
        contents = mesh["sets/contents"][:]
        children = mesh["sets/children"][:]
        categories = {}
        for handle, value in zip(mesh["tags/CATEGORY/id_list"][:], mesh["tags/CATEGORY/values"][:]):
            categories[int(handle)] = text_tag(value)
        names = {}
        for handle, value in zip(mesh["tags/NAME/id_list"][:], mesh["tags/NAME/values"][:]):
            names[int(handle)] = text_tag(value)
        ids = {}
        for handle, value in zip(mesh["tags/GLOBAL_ID/id_list"][:], mesh["tags/GLOBAL_ID/values"][:]):
            ids[int(handle)] = int(value)
        senses = {}
        for handle, value in zip(mesh["tags/GEOM_SENSE_2/id_list"][:], mesh["tags/GEOM_SENSE_2/values"][:]):
            senses[int(handle)] = [int(v) for v in value]

    set_contents = {}
    set_children = {}
    for index, row in enumerate(rows):
        handle = set_start + index
        start = int(rows[index - 1, 0]) + 1 if index else 0
        values = [int(v) for v in contents[start:int(row[0]) + 1]]
        # MOAB bit 3 stores (start,count) pairs instead of individual handles.
        if int(row[3]) & 8:
            expanded = []
            for begin, count in zip(values[::2], values[1::2]):
                expanded.extend(range(begin, begin + count))
            values = expanded
        set_contents[handle] = values
        start = int(rows[index - 1, 1]) + 1 if index else 0
        set_children[handle] = [int(v) for v in children[start:int(row[1]) + 1]]

    volume_materials = {}
    for group, name in names.items():
        if name.startswith("mat:") and not name.endswith("_comp"):
            for handle in set_contents[group]:
                if categories.get(handle) == "Volume":
                    if handle in volume_materials:
                        raise ValueError("Volume belongs to multiple material groups.")
                    volume_materials[handle] = name[4:]
    surface_triangles = {}
    for handle, category in categories.items():
        if category == "Surface":
            indices = [v - triangle_start for v in set_contents[handle]
                       if triangle_start <= v < triangle_start + len(triangles)]
            surface_triangles[handle] = triangles[indices]
    volumes = []
    for handle, category in categories.items():
        if category != "Volume":
            continue
        faces = []
        for surface in set_children[handle]:
            face = surface_triangles[surface].copy()
            if senses[surface][1] == handle:
                face = face[:, [0, 2, 1]]
            elif senses[surface][0] != handle:
                raise ValueError("Surface sense disagrees with its parent volume.")
            faces.append(face)
        if not faces:
            raise ValueError("Volume has no surface triangles.")
        volumes.append({"id": ids[handle], "material": volume_materials.get(handle),
                        "triangles": np.vstack(faces)})
    return coordinates, triangles, volumes, senses


def check_volume_mesh(coordinates, volume):
    triangles = volume["triangles"]
    edges = np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    a, b, c = coordinates[triangles[:, 0]], coordinates[triangles[:, 1]], coordinates[triangles[:, 2]]
    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    signed_volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    return {"id": volume["id"], "material": volume["material"],
            "triangle_count": len(triangles), "nonmanifold_or_open_edges": int(np.sum(counts != 2)),
            "degenerate_triangles": int(np.sum(areas <= 1e-12)),
            "signed_volume_cm3": float(signed_volume)}


def export_surface_vtk(filename, coordinates, volumes):
    # Shared interfaces appear once per neighbouring volume in this display file.
    triangle_count = sum(len(v["triangles"]) for v in volumes)
    with filename.open("w") as stream:
        stream.write("# vtk DataFile Version 3.0\nDAGMC surfaces, cm\nASCII\nDATASET POLYDATA\n")
        stream.write(f"POINTS {len(coordinates)} double\n")
        for x, y, z in coordinates:
            stream.write(f"{x:.10g} {y:.10g} {z:.10g}\n")
        stream.write(f"POLYGONS {triangle_count} {4 * triangle_count}\n")
        for volume in volumes:
            for a, b, c in volume["triangles"]:
                stream.write(f"3 {a} {b} {c}\n")
        stream.write(f"CELL_DATA {triangle_count}\nSCALARS volume_id int 1\nLOOKUP_TABLE default\n")
        for volume in volumes:
            for face in volume["triangles"]:
                stream.write(f"{volume['id']}\n")

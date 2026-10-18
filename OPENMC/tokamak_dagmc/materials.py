# %%
"""Editable bulk approximations, not SPARC engineering material specifications."""

import openmc
from pathlib import Path

from workflow import load_config, nuclear_data_path


def make_materials(config=None):
    if config is None:
        config = load_config()
    materials = openmc.Materials()
    data_path = nuclear_data_path(config)
    element_data = data_path if data_path and Path(data_path).is_file() else None
    if data_path:
        materials.cross_sections = data_path
    for component in config["components"]:
        if component["tag"] == "Vacuum":
            continue  # DAGMC mat:Vacuum is an actual void, with no material entry.
        material = openmc.Material(name=component["tag"])
        filename = component["file"]
        if filename == "first_wall.step":
            material.set_density("g/cm3", 19.3)
            material.add_element("W", 1.0, cross_sections=element_data)
        elif filename in ("Blanket.step", "Vaccume Vessel.step"):
            # Generic Fe-Cr-Ni steel; the blanket is shielding, not a breeder model.
            material.set_density("g/cm3", 7.8)
            material.add_element("Fe", 0.70, "wo", cross_sections=element_data)
            material.add_element("Cr", 0.19, "wo", cross_sections=element_data)
            material.add_element("Ni", 0.11, "wo", cross_sections=element_data)
        else:
            # Cu / steel bulk proxy. No claim of a detailed REBCO winding pack.
            material.set_density("g/cm3", 8.4)
            material.add_element("Cu", 0.60, "wo", cross_sections=element_data)
            material.add_element("Fe", 0.28, "wo", cross_sections=element_data)
            material.add_element("Cr", 0.076, "wo", cross_sections=element_data)
            material.add_element("Ni", 0.044, "wo", cross_sections=element_data)
        materials.append(material)
    return materials


# %%
if __name__ == "__main__":
    from workflow import output_paths

    config = load_config()
    paths = output_paths(config)
    make_materials(config).export_to_xml(paths["simulation"] / "materials.xml")

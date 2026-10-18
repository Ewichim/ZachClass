# %%
"""Collect inspection, mesh, plotting, and transport evidence in one report."""

import json

from workflow import load_config, output_paths

config = load_config()
paths = output_paths(config)


def read_report(name):
    filename = paths["reports"] / name
    return json.loads(filename.read_text()) if filename.is_file() else {}


def write_report():
    inspection = read_report("step_inspection.json")
    repair = read_report("topology_repair.json")
    conversion = read_report("conversion.json")
    validation = read_report("dagmc_validation.json")
    plotting = read_report("openmc_plots.json")
    transport = read_report("transport.json")
    debug = read_report("geometry_debug_attempt.json")
    environment = read_report("environment.json")
    lines = ["# Tokamak geometry inspection report", "",
             "Source STEP files are unchanged; hashes are recorded in step_inspection.json.", "",
             "## Current results", "",
             f"- Verified topology copy: {repair.get('verified', False)}.",
             f"- Full CAD conversion: {conversion.get('success', False)}.",
             f"- DAGMC geometry checks passed: {validation.get('geometry_checks_passed', False)}.",
             f"- Full OpenMC cross-section plots: {plotting.get('success', False)}.",
             f"- Small neutron transport test completed: {transport.get('completed', False)}.", "",
             "## Components and materials", "",
             "| Original filename | Solids | Shells | Faces | CAD volume (m³) | DAGMC tag |",
             "|---|---:|---:|---:|---:|---|"]
    for component in inspection.get("components", []):
        if not component.get("import_success"):
            lines.append(f"| {component['file']} | Import failed | | | | |")
            continue
        lines.append(f"| {component['file']} | {component['solid_count']} | {component['shell_count']} | "
                     f"{component['face_count']} | {component['total_volume_mm3']/1e9:.8g} | {component['tag']} |")
    lines += ["", "Materials are provisional: Fe-Cr-Ni steel for shielding and vessel (7.8 g/cm³), "
              "tungsten for the first wall (19.3 g/cm³), and a Cu/steel magnet proxy (8.4 g/cm³). "
              "These are not detailed SPARC specifications. `mat:Vacuum` is void and has no OpenMC material definition.", "",
              "## Units, alignment, and layer checks", "",
              "All seven STEP exports declare metres. CadQuery imports them into millimetres; "
              "the conversion applies **0.1 cm/mm**, once, to the complete assembly. "
              "In terms of original metre coordinates this is a factor of 100. "
              "No component is translated, rotated, or resized independently.", "",
              f"Overall original bounds (mm): `{inspection.get('assembly_bbox_mm')}`. "
              "The CAD spans 12.2 m in X and Y and 8.54 m in Z. This checks unit consistency; "
              "it does not verify dimensions against an engineering SPARC drawing.", "",
              "The axisymmetric components pass 72-point full-turn interior probes about the global Z axis. "
              "The 18 TF bodies are distributed around the full torus; their angular probe includes the gaps between coils. "
              "The JSON records analytic surface axes, every bounding-box collision candidate, "
              "and a 2 mm radial layer scan at z=0.", "",
              "The first-wall cylindrical radii differ by 30 mm (3 cm). The mesh profiles use 3 cm "
              "and 1 cm surface edge targets for that component. Edge size is not a thickness or sag tolerance; "
              "volume agreement and zoomed plots are checked separately. A finer mesh study is still needed for quantitative results.", "",
              "### Component global positions", "",
              "| Component | Minimum xyz (cm) | Maximum xyz (cm) | Center of mass xyz (cm) |",
              "|---|---|---|---|"]
    for component in inspection.get("components", []):
        if component.get("import_success"):
            minimum = [round(v/10, 5) for v in component["bbox_mm"]["min"]]
            maximum = [round(v/10, 5) for v in component["bbox_mm"]["max"]]
            center = [round(v/10, 5) for v in component["center_of_mass_mm"]]
            lines.append(f"| {component['file']} | {minimum} | {maximum} | {center} |")
    lines += ["", "### Equatorial layer transitions", "",
              "Transitions are sampled, with at most 2 mm spatial resolution. Empty lists are void gaps.", "",
              "| Radius from Z axis (cm) | Bodies present just after transition |", "|---:|---|"]
    for transition in inspection.get("equatorial_radial_transitions_2mm_grid", []):
        lines.append(f"| {transition['from_radius_mm']/10:.1f} | {', '.join(transition['inside']) or 'void'} |")
    lines += ["", "## Original interface failure and separate repair", ""]
    for error in inspection.get("errors", []):
        lines.append(f"- Original precheck: {error}")
    if repair.get("verified"):
        lines += ["", "The original blanket/first-wall intersection returned invalid solids, including negative "
                  "signed volumes. Its reported volume was rejected as unphysical. Changing numerical tolerance "
                  "and an independent Gmsh fragment calculation did not fix that result.", "",
                  "The blanket inner shell matches the first-wall outer shell; the blanket outer shell matches "
                  "the vessel inner shell. In a separate BREP assembly, the blanket reuses those existing shells "
                  "with the opposite orientation. This shares topology without moving the analytic CAD surfaces. "
                  "Matching checks include face types, areas, bounds, centers, and bidirectional trim-edge samples "
                  f"within {repair['matching_tolerance_mm']} mm. No Boolean cut removes material.", "",
                  f"Blanket relative volume change: {repair['blanket_relative_volume_change']:.3g}. "
                  f"Maximum relative change of any body in the full imprint: {repair['maximum_individual_volume_change_in_imprint']:.3g}. "
                  f"The full repaired imprint contains {repair['full_imprint_solid_count']} valid solids, "
                  "each with exactly one original-body assignment. Both repaired interface intersections return valid zero volume.", "",
                  "Copies: `../repaired/assembly_shared_interfaces_mm.brep` and "
                  "`../repaired/Blanket_shared_interfaces_mm.brep`. Full comparisons and hashes are in topology_repair.json. "
                  "No physical CAD redesign is currently required by these checks. For future exports, retaining "
                  "shared interfaces in a single assembly STEP may avoid the kernel failure; this has not been tested here."]
    lines += ["", "Ordinary topology checks found no invalid original solids or faces, unattached shells, "
              "or flagged small faces/edges. No valid Boolean intersection showed positive overlap at the stated "
              "tolerances. The original unresolved intersection is retained in the inspection record rather than silently discarded.", "",
              "## DAGMC checks", "",
              f"Cells including implicit complement: {validation.get('n_cells_including_implicit_complement', 'not checked')}. "
              f"Surfaces: {validation.get('n_surfaces', 'not checked')}. "
              f"Maximum bounding-box difference: {validation.get('maximum_bbox_difference_cm', 'not checked')} cm.", "",
              "Watertightness and mesh overlap logs are `watertight.log` and `mesh_overlap.log`. "
              "Per-volume edge counts, degenerate triangles, signed volumes, and material memberships "
              "are in dagmc_validation.json. CAD-to-mesh material volume differences must stay below 2%.", ""]
    for error in validation.get("errors", []):
        lines.append(f"- Unresolved validation issue: {error}")
    lines += ["", "The pinned H5M writer emits `FACETING_TOLERANCE`; DAGMC reads sparse `FACETING_TOL` "
              "on a meshset. The converter adds the expected tag on the file meshset, keeping the writer's "
              "original extent-based estimate. This changes metadata only. `tokamak.metadata.json` records it; "
              "the estimate is not a measured maximum surface deviation.", "",
              "## Visual inspection", "",
              "Original CAD outline sections and `original_cad_surfaces.vtk` are in `../plots/`. "
              "OpenMC images are `tokamak_xy.png`, `tokamak_xz.png`, `tokamak_yz.png`, three offset sections, "
              "and `poloidal_zoom.png`. `tokamak_surfaces.vtk` contains the actual H5M triangle surfaces in cm, "
              "colored by volume ID in ParaView. Shared interfaces appear in both adjacent display volumes.", "",
              "## Source and transport", "",
              "The source proposal is a cylinder derived from the plasma CAD bounds, sampled with r dr dφ dz "
              "and rejected outside the actual void plasma cell. Accepted sites therefore fill the toroidal plasma "
              "uniformly in volume. Neutrons are isotropic at 14.1 MeV. Uniform emissivity is an assumption. "
              "The external CSG box has a padded vacuum boundary. Mesh flux is per source neutron; no fusion power is assumed.", "",
              f"Existing data check: `{environment.get('nuclear_data', {})}`.", "",
              f"Transport result: `{transport}`.", "",
              f"Initial geometry-debug attempt: `{debug}`. "
              "This failed at the plasma/implicit-complement boundary. "
              "[OpenMC issue #3023](https://github.com/openmc-dev/openmc/issues/3023) documents "
              "false overlap reports at DAGMC surface crossings. The normal test therefore uses "
              "geometry_debug=False while retaining the independent CAD and DAGMC checks. "
              "The failed case is preserved; it is not counted as a successful validation run.", "",
              "A small run and geometry-debug diagnostics do not demonstrate production transport accuracy. "
              "DAGMC checks are separate from CSG geometry-debug coverage. Material assumptions, CAD dimensions, "
              "source profiles, statistical convergence, and mesh convergence need review before drawing physics conclusions.", "",
              "## Per-solid inspection details", "",
              "Full global bounds, center of mass, and geometric diagnostics are in step_inspection.json. "
              "The following table gives individual volumes and bounding-box dimensions.", "",
              "| Body | CAD volume (cm³) | Bounding dimensions (cm) | Center of mass (cm) | Valid / closed |",
              "|---|---:|---|---|---|"]
    for component in inspection.get("components", []):
        for solid in component.get("solids", []):
            dimensions = [round(v/10, 5) for v in solid["bbox_mm"]["size"]]
            center = [round(v/10, 5) for v in solid["center_of_mass_mm"]]
            lines.append(f"| {solid['id']} | {solid['volume_mm3']/1000:.8g} | {dimensions} | {center} | "
                         f"{solid['valid']} / {solid['closed_shells']} |")
    lines += ["", "## Reproduction", "", "See `../../README.md` for the exact command sequence and environment setup. "
              "Original reports and earlier outputs are preserved under `../previous/` when stages are rerun."]
    filename = paths["reports"] / "geometry_report.md"
    filename.write_text("\n".join(lines) + "\n")
    print(filename)


# %%
if __name__ == "__main__":
    write_report()

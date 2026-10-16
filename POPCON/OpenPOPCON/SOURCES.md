# Input sources

The beginner tutorial follows the official
[usage instructions](https://github.com/hansec/OpenPOPCON#usage) and the pinned
[example runner](https://github.com/hansec/OpenPOPCON/blob/9e70f06df1e8c16df0191730036f5cd00d9e904c/example_run.py).
Method behaviour, output names, profile IDs, and the power accounting were also
checked against the installed 2.2.0 source. The tutorial distinguishes the
machine inputs, plotting inputs, model object, calculation calls, and saved arrays.

The files in `inputs/upstream_SPARC/` were copied unchanged from the installed
OpenPOPCON 2.2.0 package. Its installation is pinned to commit
`9e70f06df1e8c16df0191730036f5cd00d9e904c`:

- [SPARC input](https://github.com/hansec/OpenPOPCON/blob/9e70f06df1e8c16df0191730036f5cd00d9e904c/openpopcon/resources/examples/SPARC/POPCON_input_example.yaml)
- [SPARC plot settings](https://github.com/hansec/OpenPOPCON/blob/9e70f06df1e8c16df0191730036f5cd00d9e904c/openpopcon/resources/examples/SPARC/plotsettings.yml)

The tutorial uses the upstream physical parameters and grid, with serial
execution instead of parallel execution. Its run folder saves that change.
The package's bundled default confinement scalings are used without edits.

`inputs/fire_h_mode.yaml` and `inputs/plotsettings.yml` were prepared for this
tutorial. FIRE machine values, H98, Zeff target, and approximate profile
peaking come from [Meade et al., 2002, Table I](https://fire.pppl.gov/fec2002_FT2_6_reprint4.pdf),
also saved under `../references/FIRE/`. The current profile, Ti/Te ratio,
impurity species, and profile edge treatment are assumptions described in
the input. No operating temperature is inferred from the published gain.
The scan and heating/beta caps are teaching choices.

`sources.json` records the source URLs and SHA-256 checksums of the copied inputs.

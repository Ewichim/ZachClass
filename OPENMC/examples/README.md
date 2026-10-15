# Downloaded OpenMC examples

Start with `nested_torii/nested_torii.py`. It defines concentric toroidal
shells, a surrounding sphere, and a point source. Its low-density hydrogen
material makes this a geometry benchmark rather than a reactor physics model.
Both installed libraries contain its required nuclide.

`simple_tokamak/simple_tokamak_csg.py` loads the accompanying XML model.
This is a larger CSG tokamak reference. It contains the legacy carbon label
`C0`, which neither installed library provides. Before running transport,
its carbon material definitions need conversion to available carbon isotopes.

Both examples were downloaded unchanged from
[OpenMC's benchmark repository](https://github.com/openmc-dev/openmc-performance-bench/tree/fe41167856ce6a7d1dcd41b9551c77226b2536e6/benchmarks/models).
Their `build_model()` functions create or load a model; they do not start a
simulation. Import and call only the example you want in a fresh kernel.

The `.ipynb` files were generated with Jupytext and have no execution outputs.
These are references for later work, not completed class assignments.

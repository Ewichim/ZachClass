# POPCON tutorials

Start with either notebook; the tutorials are independent.

Both start from basic POPCON terminology and explain the files, package calls,
units, and result variables inside the notebook. The CFS tutorial first runs a
single formula and a small Algorithm/Dataset example before the full case.
File-saving cells are identified separately from plasma calculations.

- [OpenPOPCON](OpenPOPCON/tutorial.ipynb): YAML inputs, basic runs, profile inspection, H98 scan, and selected operating limits.
- [CFSPOPCON](CFSPOPCON/tutorial.ipynb): units and calculation recipes, basic runs, profile inspection, H98 scan, and selected operating limits including L-H access.

Each starts with an upstream SPARC example and then adapts the inputs to a
FIRE H-mode teaching case. Machine parameters come from the 2002 reference
in [references/FIRE](references/FIRE/README.md). Assumed profiles, composition,
and illustrative limits are identified in the inputs and tutorial markdown.
These are tool tutorials, not an assessment of FIRE's published claims.

Use `conda activate ZClass` and the **Python (ZClass)** notebook kernel.
The packages stay installed in that environment. See each folder's `SETUP.md`
for commands and its `SOURCES.md` for the origin of copied inputs.

Edit the `.py` tutorial first, then regenerate its notebook with Jupytext.
Every execution saves a new `outputs/<run>/` folder within its tool folder.
The two tutorials do not yet implement a controlled comparison between tools.

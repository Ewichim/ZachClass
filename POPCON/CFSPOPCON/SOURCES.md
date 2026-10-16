# Input and atomic-data sources

The beginner tutorial follows the progression in the official
[Creating your first POPCON](https://cfspopcon.readthedocs.io/en/latest/doc_sources/getting_started.html)
and [Understanding and writing algorithms](https://cfspopcon.readthedocs.io/en/latest/doc_sources/understanding_algorithms.html)
pages. The [physics glossary](https://cfspopcon.readthedocs.io/en/latest/doc_sources/physics_glossary.html)
provides variable definitions. The explanations and live examples were checked
against the installed 9.0.0 input reader, algorithm methods, and unit definitions.

Original example files were downloaded from CFSPOPCON v9.0.0, commit
`527ce6bd490a003a56e336dfa9a0949cd8d7ed1a`:

- [SPARC PRD input and plot settings](https://github.com/cfs-energy/cfspopcon/tree/527ce6bd490a003a56e336dfa9a0949cd8d7ed1a/example_cases/SPARC_PRD)
- [RADAS configuration](https://github.com/cfs-energy/cfspopcon/blob/527ce6bd490a003a56e336dfa9a0949cd8d7ed1a/radas_config.yaml)
- [Data setup instructions](https://github.com/cfs-energy/cfspopcon/blob/527ce6bd490a003a56e336dfa9a0949cd8d7ed1a/docs/doc_sources/Usage.rst)

`inputs/upstream_SPARC_PRD/` retains the original input, two plot files, and
upstream license. `inputs/radas_upstream.yaml` is the original configuration.
`sources.json` records their URLs and SHA-256 checksums.

`inputs/radas_tutorial.yaml` derives from that configuration. It retains only
helium, oxygen, tungsten, xenon, and argon, and reduces the interpolation grid
from 50 densities by 100 temperatures to 12 by 40. The rate datasets and ne*tau
value are unchanged. Local tables were generated with RADAS `1!1.0.1` using
OpenADAS downloads; raw rates and generated tables remain in `radas_dir/`,
which is ignored by Git. The tutorial does not claim an exact reproduction
of the upstream numerical regression result with this coarse atomic grid.

`inputs/fire_h_mode.yaml` and `inputs/plot_fire.yaml` were prepared for this
tutorial. Published FIRE values come from [Meade et al., 2002, Table I](https://fire.pppl.gov/fec2002_FT2_6_reprint4.pdf),
also saved under `../references/FIRE/`. The shorter recipe uses the installed
package's core formulas and omits SPARC-specific divertor and solenoid inputs.
Ti/Te, impurity concentrations, and identical ion/electron density peaking
are assumptions. Zeff is calculated from the assumed species. The confinement
scan, heating cap, and beta cap are teaching choices.

[![DOI](https://zenodo.org/badge/287222587.svg)](https://zenodo.org/badge/latestdoi/287222587)

NOTE: tracklib has been refactored into a collection of libraries hosted [here](https://github.com/orgs/OpenTrajectoryAnalysis/repositories)

This is a library for downstream analysis of single particle tracking data.

For installation with pip:
```sh
$ pip install git+https://github.com/SGrosse-Holz/tracklib
```

Comprehensive documentation is available, but currently not web-hosted. Refer
to ``doc/sphinx/build/html/index.html`` (update with ``make doc`` from the root
dir)

If you use this library, please cite our original paper[^1]. The looping inference approach presented therein (Bayesian Inference of Loop Dynamics; BILD) is implemented in the ``tracklib.analysis.bild`` module.

[^1]: Gabriele, Brandão, Grosse-Holz, _et al._, __Dynamics of CTCF and cohesin mediated chromatin looping revealed by live-cell imaging__, _bioRχiv_ 2021; [DOI](https://doi.org/10.1101/2021.12.12.472242)

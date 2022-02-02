"""
Bayesian Inference of Loop Dynamics (BILD)

This module provides the implementation of BILD, proposed by `Gabriele,
Brandão, Grosse-Holz, et al.
<biorxiv.org/content/10.1101/2021.12.12.472242v1>`. Since the ideas behind this
scheme are explained in the paper (and its supplementary information), in the
code we only provide technical documentation, assuming knowledge of the
reference text.
"""
from . import util
from .util import Loopingprofile
from . import models
from . import amis
from . import postproc
from .core import *

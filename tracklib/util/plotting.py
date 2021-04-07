"""
Some helper functions for plotting stuff with matplotlib
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def ellipse_from_cov(pos, cov, ax=None, n_std=1, **kwargs):
    """
    Plot an elliptical confidence region

    Parameters
    ----------
    pos : np.array
        the center position of the ellipse, i.e. the point estimate
    cov : np.array
        the corresponding covariance matrix
    n_std : float, optional
        a scaling factor
    kwargs : kwargs
        forwarded to ``matplotlib.patches.Ellipse``

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    w, v = np.linalg.eigh(cov)
    angle = np.arctan2(v[1, -1], v[0, -1]) # dominant eigenvector
    angle_in_degrees = 360/(2*np.pi)*angle

    mykwargs = {
            'facecolor' : 'none',
            'edgecolor' : 'k',
            }
    mykwargs.update(kwargs)

    axes = 2*n_std*np.sqrt(w)
    ell = matplotlib.patches.Ellipse(pos, axes[-1], axes[0], angle_in_degrees, **mykwargs)

    if ax is None:
        ax = plt.gca()
    return ax.add_patch(ell)

.. _traj_meta_fields:

`Trajectory.meta`-info fields used by tracklib
==============================================
Modifiers try to respect the following fields::

    'localization_error', 'parity'

General
-------
`!'localization_error'` : (d,) or (2, d) np.ndarray
    the localization error for each particle and spatial dimension. Use the
    ``(2, d)`` shape to give different errors for a two-locus trajectory.

.. _traj_meta_fields_msd:

Used by analysis.msd
--------------------
`!'MSD'` : np.ndarray
    the MSD of the trajectory. See `tracklib.analysis.msd`.
`!'MSDmeta'` : dict
    `!'N'` : np.ndarray
        the number of sample points for each MSD data point
    `!'alpha'` : float
        scaling at the beginning
    `!'logG'` : float
        y-intercept of log(MSD)
    `!'fit_covariance'` : (2, 2) np.ndarray
        covariance of (alpha, logG) from the fit

Used by analysis.kli
--------------------
`!'looptrace'` : (T,) np.ndarray, dtype=bool, or list of bool
    the ground truth looptrace associated with this trajectory

Used by analysis.kld
--------------------
`!'parity'` : one of {'even', 'odd'}
    parity of the trajectory under time reversal

Used by analysis.chi2
---------------------
`!'chi2scores'` : np.ndarray
    chi2 scores for trajectory snippets

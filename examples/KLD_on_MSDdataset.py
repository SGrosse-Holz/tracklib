# How to:
#  - generate an artificial data set
#  - show the standard plots for it
#  - Calculate Kullback-Leibler divergence
#
# On the last point, note that since we generate this synthetic data set from
# an explicitly time-reversal symmetric process, the ground truth KLD is zero.
# So this can be used to evaluate the bias of the KLD estimator. More
# specifically, in that case one could use tracklib.tools.MSDcontrol to create
# an explicit sister data set.

import os, sys

import numpy as np
from matplotlib import pyplot as plt

import tracklib as tl

def generate_dataset(msd):
    Ts = np.minimum(np.random.geometric(1/100, size=(500,)), len(msd))
    ds = tl.tools.MSDdataset(msd, N=2, Ts=Ts, d=3, subtractMean=True)
    
    # Add a random offset to each trajectory
    # In fact this is pretty unnecessary, except that then plotting the
    # trajectories makes a bit more sense.
    for traj in ds:
        traj._data += np.random.uniform(100, size=(1, 1, traj.d))
        traj._data += np.random.normal(0, 2, size=(traj.N, 1, traj.d))

    return ds

def KLD_for_ns(dataset, ns, processes=16):
    est = tl.analysis.KLDestimator(dataset)
    est.preprocess(lambda traj : traj.relative().abs())
    est.setup(KLDmethod=tl.analysis.KLD_PC, \
              n=ns, k=20, dt=1, \
              bootstraprepeats=20, processes=processes)

    return est.run()

def plot_KLD_results(results):
    ns = np.unique(results['n'])
    violins = [results['KLD'][results['n'] == n] for n in ns]

    plt.violinplot(violins)
    plt.gca().set_xticks(np.arange(len(ns))+1)
    plt.gca().set_xticklabels([str(n) for n in ns])
    plt.title('KLD vs. window size')
    plt.xlabel('n (window size)')
    plt.ylabel('estimated KLD')

def run_full():
    msd = np.sqrt(np.arange(1000))
    ds = generate_dataset(msd)

    KLDres = KLD_for_ns(ds, [5, 10, 20, 40, 80])

    tl.analysis.hist_lengths(ds)
    tl.analysis.hist_distances(ds)
    tl.analysis.plot_trajectories(ds)

    tl.analysis.plot_msds(ds)
    plt.plot(6*msd, 'r', linewidth=2, label='ground truth')
    plt.legend()

    plt.figure()
    plot_KLD_results(KLDres)

    plt.show()

if __name__ == '__main__':
    run_full()

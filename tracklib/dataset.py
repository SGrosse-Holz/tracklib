import os, sys

import numpy as np
import matplotlib.pyplot as plt

from . import util
from .trajectory import Trajectory
from .taggedlist import TaggedList

class Dataset(TaggedList):
    """
    Add some trajectory specific functionality to TaggedList
    """
    def msd(self, tags='_all', giveN=False, memo=True):
        """
        Similar to Trajectory.msd: a memoized method to calculate the ensemble
        MSD. Use memo=False to circumvent memoization.
        """
        tags = self.makeTagsSet(tags)

        msdKnown = hasattr(self, '_msdN')
        if hasattr(self, '_msd_lasttagset'):
            tagsHaveChanged = set(tags) != self._msd_lasttagset
        else:
            tagsHaveChanged = True
        if not msdKnown or tagsHaveChanged or not memo:
            msdNs = [traj.msd(giveN=True, memo=memo) for traj in self.byTag(tags)]

            maxlen = max(len(msdN[0]) for msdN in msdNs)
            emsd = msdNs[0][0]
            npad = [(0, maxlen-len(emsd))] + [(0, 0) for _ in emsd.shape[2:]]
            emsd = np.pad(emsd, npad, constant_values=0)
            eN = np.pad(msdNs[0][1], npad, constant_values=0)
            emsd *= eN

            for msd, N in msdNs[1:]:
                emsd[:len(msd)] += msd*N
                eN[:len(N)] += N
            emsd /= eN
            self._msdN = (emsd, eN)

        self._msd_lasttagset = set(tags)
        if giveN:
            return self._msdN
        else:
            return self._msdN[0]

    # Plotting an overview: all trajectories, color coded by tag; MSDs,
    # individual and ensemble mean; histogram of trajectory lengths
    # Those plots can also be useful alone, so let's have separate functions
    def hist_lengths(self, tags='_all', **kwargs):
        tags = self.makeTagsSet(tags)

        lengths = [len(traj) for traj in self.byTag(tags)]
        
        if 'bins' not in kwargs.keys():
            kwargs['bins'] = 'auto'

        plt.figure()
        h = plt.hist(lengths, **kwargs)
        plt.title("Histogram of trajectory lengths")
        plt.xlabel("Length in frames")
        plt.ylabel("Count")
        return h

    def plot_msds(self, tags='_all', **kwargs):
        tags = self.makeTagsSet(tags)

        ensembleLabel = 'ensemble mean'
        if 'label' in kwargs.keys():
            ensembleLabel = kwargs['label']
        kwargs['label'] = None

        plt.figure()
        lines = []
        for traj in self.byTag(tags):
            msd = traj.msd()
            tmsd = np.arange(len(msd))*traj.dt
            lines.append(plt.loglog(tmsd, msd, **kwargs))
        msd = self.msd(tags=tags)
        tmsd = np.arange(len(msd))*self._data[0].dt # NOTE: we're assuming that all trajectories have the same dt!
        lines.append(plt.loglog(tmsd, msd, color='k', linewidth=2, label='ensemble mean'))
        plt.legend()

        if tags == {'_all'}:
            plt.title('MSDs')
        else:
            plt.title('MSDs for tags {}'.format(str(tags)))
        plt.xlabel("time") # TODO: How do we get a unit description here?
        plt.ylabel("MSD")
        
        return lines

    def plot_trajectories(self, tags='_all', **kwargs):
        """
        Notes:
         - trajectories will be colored by one of the tags they're associated
           with.
         - by default, this uses any logic

        """
        tags = self.makeTagsList(tags)
        flags = {'_all in tags' : False, 'single tag' : False}
        if '_all' in tags:
            tags = list(self.tagset() - {'_all'})
            flags['_all in tags'] = True

        if len(tags) == 1:
            colordict = {tags[0] : None}
            flags['single tag'] = True
        else:
            try:
                if isinstance(kwargs['color'], list):
                    colors = kwargs['color']
                else:
                    colors = [kwargs['color']]
            except KeyError:
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            colordict = {tag : colors[i%len(colors)] for i, tag in enumerate(tags)}

        plt.figure()
        lines = []
        for traj, trajtags in self.byTag(tags, logic=any, giveTags=True):
            for mytag in trajtags & set(tags):
                break # Cheat to get some tag for the trajectory (out of the given ones)
            kwargs['color'] = colordict[mytag]
            lines.append(traj.plot_spatial(**kwargs))

        # Need some faking for the legend
        x0 = sum(plt.xlim())/2
        y0 = sum(plt.ylim())/2
        for tag in tags:
            kwargs['color'] = colordict[tag]
            kwargs['label'] = tag
            if 'linestyle' in kwargs.keys() and isinstance(kwargs['linestyle'], list):
                    kwargs['linestyle'] = kwargs['linestyle'][0]
            plt.plot(x0, y0, **kwargs)

        if not flags['single tag']:
            plt.legend()

        if flags['single tag']:
            plt.title('Trajectories for tag "{}"'.format(tags[0]))
        elif flags['_all in tags']:
            plt.title("Trajectories by tag")
        else:
            plt.title("Trajectories for tags {}".format(str(tags)))
        return lines

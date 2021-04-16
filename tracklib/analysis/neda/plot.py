"""
Visualizing inference runs
"""

from matplotlib import pyplot as plt
import numpy as np

def butterfly(traj, fig=None, title='Example trajectory',
              ylim=[0, None], ylabel='distance', states_cmap='Greens',
             ):
    """
    General overview over an inference run

    Parameters
    ----------
    traj : Trajectory
        should have ``traj.meta['neda']`` meta data, i.e. have gone through
        `neda.main`
    fig : figure handle, optional
        the figure to plot into. Will create a new one by default
    title : string, optional
        the title for the plot
    ylim : [lower, higher], optional
        the y-limits for the plot of the trajectory (can be used to prevent
        outliers from distorting the scale)
    ylabel : string, optional
        label to attach to the y-axis of the trajectory plot
    states_cmap : string, optional
        an identifier for the colormap to use for the display of marginal
        probabilities
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if fig is None:
        fig = plt.figure(figsize=[12, 7])
    axs = fig.subplots(3, 4,
                       gridspec_kw={'height_ratios' : [0.7, 0.3, 1],
                                    'hspace' : 0,
                                    'width_ratios' : [1, 0.3, 0.3, 0.3],
                                    'wspace' : 0,
                                   },
                       sharex='col',
                       sharey='row',
                      )
    
    ref_lt = traj.meta['neda']['final']['mcmcrun'].samples[0]

    # Trajectory
    ax = axs[0, 0]
    
    ax.set_title(title)
    ax.plot(ref_lt.t, traj.abs()[ref_lt.t][:, 0], label='distance', color=colors[0])
#     ax.legend()
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    
    # Loop states
    ax = axs[1, 0]
    
    statedists = np.empty((ref_lt.n, len(traj)))
    statedists[:] = np.nan
    for state in range(statedists.shape[0]):
        statedists[state, ref_lt.t] = np.sum([trace[:] == state
                                              for trace in traj.meta['neda']['final']['mcmcrun'].samples
                                             ], axis=0)
    statedists /= np.sum(statedists, axis=0)
    
    pcm = ax.pcolormesh(np.arange(statedists.shape[1]+1)-0.5,
                        np.arange(statedists.shape[0]+1)-0.5,
                        statedists,
                        cmap=states_cmap,
                       )
    ax.plot(ref_lt.t, np.argmax(statedists[:, ref_lt.t], axis=0), label='MmAP', color=colors[1])
#     ax.legend()
    ax.set_ylim([-0.5, statedists.shape[0]-0.5])
    ax.set_ylabel('loop state')
    ### interesting bug: ax.get_yticks() gives values outside of ylim here...
#     ticks = ax.get_yticks()
#     ylim = ax.get_ylim()
#     ax.set_yticks([tick for tick in ticks if tick % 1 == 0 and tick >= ylim[0] and tick <= ylim[1]])
    ax.set_yticks(np.arange(ref_lt.n))
    
    cax = fig.add_axes([0.05, 0.51, 0.015, 0.1])
    fig.colorbar(pcm, cax=cax, label='p(state)', orientation='vertical', ticks=[0, 1])
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')

    # Waterfall
    ax = axs[2, 0]
    
    ensembles = np.empty((len(traj.meta['neda']['mcmcrun']), len(traj)))
    ensembles[:] = np.nan
    for i, mcmc in enumerate(traj.meta['neda']['mcmcrun']):
        ensembles[i, ref_lt.t] = np.mean([trace[:] for trace in mcmc.samples], axis=0)
    
    pcm = ax.pcolormesh(np.arange(ensembles.shape[1]+1)-0.5,
                        np.arange(ensembles.shape[0]+1)-0.5,
                        ensembles,
                        cmap='viridis',
                       )
    ax.set_ylim([-0.5, ensembles.shape[0]-0.5])
    ax.set_ylabel('iteration')
    ticks = ax.get_yticks()
    ylim = ax.get_ylim()
    ax.set_yticks([tick for tick in ticks if tick % 1 == 0 and tick >= ylim[0] and tick <= ylim[1]])
    
    cax = fig.add_axes([0.05, 0.18, 0.015, 0.25])
    fig.colorbar(pcm, cax=cax, label='mean state', orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')
    
    # X-axis for this column
    ax = axs[2, 0]
    ax.set_xlabel('time [frames]')
    ax.set_xlim([-0.5, len(traj)-0.5])

    # log(q)
    ax = axs[2, 1]
    
    ax.plot(np.array(traj.meta['neda']['prior_params']), np.arange(len(traj.meta['neda']['prior_params'])), marker='o')
    ax.set_title('prior parameter')
    ax.set_xlabel('log(q)')
    ax.invert_xaxis()
    
    # Δevidence
    ax = axs[2, 2]
    
    ax.plot(traj.meta['neda']['evidence_diff'], np.arange(len(traj.meta['neda']['evidence_diff']))+0.5, marker='v', color='g')
    ax.set_title('evidence gain')
    ax.set_xlabel('Δlog P(y)')
    
    # real evidence
    ax = axs[2, 3]
    
    ax.plot(traj.meta['neda']['evidence'], np.arange(len(traj.meta['neda']['evidence'])), marker='^', color='purple')
    ax.set_title('real evidence')
    ax.set_xlabel('log P(y)')
    
    # best iteration
    it_final = traj.meta['neda']['final']['iteration']
    for ax in axs[2, :]:
        ax.axhline(it_final, linestyle='--', color='k')

    # Housekeeping
    axs[2, 0].invert_yaxis()
    axs[2, 1].invert_yaxis() # Bug?
    axs[2, 2].invert_yaxis() # Bug?
    axs[2, 3].invert_yaxis() # Bug?
    
    for ax in list(axs[0, 1:]) + list(axs[1, 1:]):
        ax.axis('off')

    return fig, axs



























































import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

import argparse
import sys
import os
import pdb


from testers import get_states, load_model_path
from helpers import create_loaders
from utils import get_config

from tasks import *

cspaces = [cm.spring, cm.summer, cm.autumn, cm.winter]


def main(args):
    config = get_config(args.model, to_bunch=True)
    net = load_model_path(args.model, config)

    if len(args.dataset) == 0:
        args.dataset = config.dataset

    n_reps = 50
    # don't show these contexts
    context_filter = []
    
    if not config.multimodal:
        _, loader = create_loaders(args.dataset, config, split_test=False, test_size=n_reps, context_filter=context_filter)
        
        x, y, trials = next(iter(loader))

        A = get_states(net, x)
        t_type = type(trials[0])
        if t_type == RSG:
            pca_rsg(args, A, trials, n_reps)
        elif t_type in [DelayProAnti, MemoryProAnti]:
            pca_dmpa(args, A, trials, n_reps)

    else:
        
        cspaces =  [cm.spring, cm.summer, cm.autumn, cm.winter, cm.bone, cm.copper, cm.hot]
        
        _, loader = create_loaders(args.dataset, config, split_test=False, test_size=n_reps, context_filter=context_filter, multimodal_test=True)
        
        x, y, trials = next(iter(loader))

        A = get_states(net, x)
        pca_multimodal(args, A, trials, n_reps)


def pca_multimodal(args, A_uncut, trials, n_reps):
    
    
    #no 'settings' for now i.e. plot full traajcetories for each trial 
    As = [] 
    for idx in range(n_reps):
        As.append(A_uncut[idx])
    
    if args.plot_var:
        svals, A_proj = pca(As,3, svals=True)
    else:
        A_proj = pca(As, 3)
    n_contexts = len(args.dataset)
    

    #let's group trajectories by task first and then by stimulus 
    stimuli_groups = [{} for i in range(n_contexts)]
    for idx in range(n_reps):
        t_type = type(trials[idx])
        if t_type == DelayProAnti or MemoryProAnti:
            stimulus = tuple(trials[idx].stimulus)

        elif t_type == DMProAnti or DelayDMProAnti or DMCProAnti:
            stimulus = trials[idx].stimulus_1 + trials[idx].stimulus_2
            stimulus = tuple(stimulus)
        
        context = trials[idx].context
        if stimulus in stimuli_groups[context]:
            stimuli_groups[context][stimulus].append(A_proj[idx])
        else:
            stimuli_groups[context][stimulus] = [A_proj[idx]]

    context_colors = [
        iter(cspaces[i](np.linspace(0, 1, len(stimuli_groups[i])))) for i in range(n_contexts)
    ]
    
    if args.plot_var:
        fig = plt.figure()
        ax = fig.add_subplot(221, projection='3d')
        ax.grid(False)
        plt.axis('off')

    
        for context, groups in enumerate(stimuli_groups):
            sorted_stimuli = sorted(groups.keys())
            for stimulus in sorted_stimuli:
                v = groups[stimulus]
                
                proj = sum(v) / len(v)
                c = next(context_colors[context])

                t = proj.T

                ax.plot(t[0], t[1], t[2], color=c, lw=1)
                
                #for now use marker to 

                marker_a = 'o'
                marker_b = 's'
                ax.scatter(t[0][0], t[1][0], t[2][0], s=40, color=c, marker=marker_a)
                ax.scatter(t[0][-1], t[1][-1], t[2][-1], s=30, color=c, marker=marker_b)
        
        
        #
        

        # norm = mpl.colors.Normalize(vmin=5, vmax=10)

        # for i in range(n_contexts):
        #     fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cspaces[i]),orientation='horizontal', label=f'context:{i}', shrink=0.5)
        #     fig.colorbar.set_ticks([])

        

        
        ax2=fig.add_subplot(222)
        pca_var_plot(svals, As, relative =True, axis= ax2) 
        

        plt.show()
    



    else: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        plt.axis('off')

    
        for context, groups in enumerate(stimuli_groups):
            sorted_stimuli = sorted(groups.keys())
            for stimulus in sorted_stimuli:
                v = groups[stimulus]
                
                proj = sum(v) / len(v)
                c = next(context_colors[context])

                t = proj.T

                ax.plot(t[0], t[1], t[2], color=c, lw=1)
                
                #for now use marker to 

                marker_a = 'o'
                marker_b = 's'
                ax.scatter(t[0][0], t[1][0], t[2][0], s=40, color=c, marker=marker_a)
                ax.scatter(t[0][-1], t[1][-1], t[2][-1], s=30, color=c, marker=marker_b)
        
        
        #
        

        # norm = mpl.colors.Normalize(vmin=5, vmax=10)

        # for i in range(n_contexts):
        #     fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cspaces[i]),orientation='horizontal', label=f'context:{i}', shrink=0.5)
        #     fig.colorbar.set_ticks([])

        

        plt.show()



def pca_rsg(args, A_uncut, trials, n_reps):

    setting = 'both'

    As = []
    for idx in range(n_reps):
        t_ready, t_set, t_go = trials[idx].rsg
        if setting == 'estimation':
            As.append(A_uncut[idx,t_ready:t_set])
        elif setting == 'prediction':
            As.append(A_uncut[idx,t_set:t_go])
        elif setting == 'both':
            As.append(A_uncut[idx,t_ready:t_go])

    if args.plot_var:
        svals, A_proj = pca(As,3, svals=True)
    else:
        A_proj = pca(As, 3)

    n_contexts = len(args.dataset)
    interval_groups = [{} for i in range(n_contexts)]
    for idx in range(n_reps):
        rsg = trials[idx].rsg
        context = trials[idx].context
        interval = rsg[1] - rsg[0]
        if interval in interval_groups[context]:
            interval_groups[context][interval].append(A_proj[idx])
        else:
            interval_groups[context][interval] = [A_proj[idx]]

    context_colors = [
        iter(cspaces[i](np.linspace(0, 1, len(interval_groups[i])))) for i in range(n_contexts)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    plt.axis('off')
    
    for context, groups in enumerate(interval_groups):
        sorted_intervals = sorted(groups.keys())
        for interval in sorted_intervals:
            v = groups[interval]
            proj = sum(v) / len(v)
            c = next(context_colors[context])

            t = proj.T

            ax.plot(t[0], t[1], t[2], color=c, lw=1)
            if setting == 'estimation':
                marker_a = '^'
                marker_b = 'o'
            elif setting == 'prediction':
                marker_a = 'o'
                marker_b = 's'
            elif setting == 'both':
                marker_a = '^'
                marker_b = 's'
            ax.scatter(t[0][0], t[1][0], t[2][0], s=40, color=c, marker=marker_a)
            ax.scatter(t[0][-1], t[1][-1], t[2][-1], s=30, color=c, marker=marker_b)

    plt.show()

def pca_dmpa(args, A_uncut, trials, n_reps):

    setting = 'nofix'

    As = []
    for idx in range(n_reps):
        t_type = type(trials[idx])
        fix = trials[idx].fix
        stim = trials[idx].stim    
        if t_type is MemoryProAnti:
            memory = trials[idx].memory
        if setting == 'all':
            As.append(A_uncut[idx])
        elif setting == 'nofix':
            As.append(A_uncut[idx][fix:])
        elif setting == 'preparation':
            if t_type is DelayProAnti:
                As.append(A_uncut[idx,fix:stim])
            else:
                As.append(A_uncut[idx,fix:memory])
        elif setting == 'movement':
            if t_type is DelayProAnti:
                As.append(A_uncut[idx,stim:])
            else:
                As.append(A_uncut[idx,memory:])
    
    if args.plot_var:
        svals, A_proj = pca(As,3, svals=True)
    else:
        A_proj = pca(As, 3)

    n_contexts = len(args.dataset)
    stimuli_groups = [{} for i in range(n_contexts)]
    for idx in range(n_reps):
        stimulus = tuple(trials[idx].stimulus)
        context = trials[idx].context
        if stimulus in stimuli_groups[context]:
            stimuli_groups[context][stimulus].append(A_proj[idx])
        else:
            stimuli_groups[context][stimulus] = [A_proj[idx]]

    context_colors = [
        iter(cspaces[i](np.linspace(0, 1, len(stimuli_groups[i])))) for i in range(n_contexts)
    ]
    #add another figure containing variance plots
    if args.plot_var: 
        fig = plt.figure()
        ax = fig.add_subplot(221, projection='3d')
        ax.grid(False)
        plt.axis('off')
        
    
        for context, groups in enumerate(stimuli_groups):
            sorted_stimuli = sorted(groups.keys())
            for stimulus in sorted_stimuli:
                v = groups[stimulus]
                
                proj = sum(v) / len(v)
                c = next(context_colors[context])

                t = proj.T

                ax.plot(t[0], t[1], t[2], color=c, lw=1)
                if setting == 'preparation':
                    marker_a = '^'
                    marker_b = 'o'
                else:
                    marker_a = 'o'
                    marker_b = 's'
                ax.scatter(t[0][0], t[1][0], t[2][0], s=40, color=c, marker=marker_a)
                ax.scatter(t[0][-1], t[1][-1], t[2][-1], s=30, color=c, marker=marker_b)

        ax2=fig.add_subplot(222)
        pca_var_plot(svals, As, relative =True, axis= ax2) 

        plt.show() 
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        plt.axis('off')
        
    
        for context, groups in enumerate(stimuli_groups):
            sorted_stimuli = sorted(groups.keys())
            for stimulus in sorted_stimuli:
                v = groups[stimulus]
                
                proj = sum(v) / len(v)
                c = next(context_colors[context])

                t = proj.T

                ax.plot(t[0], t[1], t[2], color=c, lw=1)
                if setting == 'preparation':
                    marker_a = '^'
                    marker_b = 'o'
                else:
                    marker_a = 'o'
                    marker_b = 's'
                ax.scatter(t[0][0], t[1][0], t[2][0], s=40, color=c, marker=marker_a)
                ax.scatter(t[0][-1], t[1][-1], t[2][-1], s=30, color=c, marker=marker_b)

        plt.show()
    

# As should be either [T, D] (single trajectory) or [[T, D], ...] (batch of trajectories) shaped where
# outer (optional) listing
# T is timesteps
# D is the dimensional space that needs to be reduced

def pca(As, rank, svals=False):
    # can deal with either list of inputs or a single A vector
    #if s_vals, return singular values
    if type(As) is not list:
        As = [As]

    N = len(As)
    # mix up the samples and timesteps, but keep the dimensions
    A = torch.cat(As)
    


    u, s, v = torch.pca_lowrank(A)
    projs = []
    #project As onto 'rank'-dimensional subspace
    for ix in range(N):
        traj = As[ix]
        traj_proj = traj @ v[:, :rank]
        projs.append(traj_proj)

    if svals:
        return s, projs
    else:
        return projs


#takes in vector of singular values of data matrix X [as is returned by torch.pca_lowrank()] and a list of data_amtrices that make up X, and returns a cumulative explained variance vs first m principal components 
#using relative = True returns proportion of var explained by the first m PCs
def pca_var_plot(svals, data_matrices, relative =True, axis=None ):
    data_matrix = torch.cat(data_matrices)
    N = data_matrix.shape[0] #number of samples
    evals = svals ** 2 / (N -1)

    #eigenvalues of data covariance in descending order
    sorted_evals, indices =  torch.sort(evals, descending = True)

    evals = sorted_evals
    print(evals)
    num_of_pcs = torch.count_nonzero(evals) #number of pcs is the number of non-zero eigevalues of the data covariance matrix
    cumulative_var =  torch.zeros(num_of_pcs)
    for i in range(num_of_pcs):
        cumulative_var[i] = torch.sum(evals[:i+1])

    if relative:
        cumulative_var = cumulative_var / cumulative_var[-1] #last entry is sum of non-zero eigenvalues which gives the total variance

    y =  cumulative_var.numpy()
    y = np.concatenate((np.zeros(1),y))
    xr = np.arange(0, len(y))
    axis.plot(xr,y, 'black')
    axis.scatter(np.arange(len(y)),y, marker = 'o',c = 'black')
    plt.xticks(range(0,len(y)))
    plt.xlabel('Number of PCs')
    plt.ylabel('Cumulative explained variance')

    #create image, save it and add it to plot, then delete it 

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=str)
    ap.add_argument('-d', '--dataset', type=str, nargs='+', default=[])
    ap.add_argument('--plot_var', action = 'store_true', help='cumulative relative variance explained by the first m principal components')
    args = ap.parse_args()

    main(args)
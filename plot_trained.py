import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.cm as cm

import random
import pickle
import argparse
import pdb
import json

from helpers import sigmoid
from utils import load_rb, get_config, update_args
from testers import load_model_path, test_model

from tasks import *

# for plotting some instances of a trained model on a specified dataset

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to a model file, to be loaded into pytorch')
# parser.add_argument('-Z', type=int, help='output dimension')
parser.add_argument('-d', '--dataset', nargs='+', help='path to a dataset of trials')
parser.add_argument('--noise', default=0, help='noise to add to trained weights')
parser.add_argument('-r', '--res_noise', default=None, type=float)
parser.add_argument('-m', '--m_noise', default=None, type=float)
parser.add_argument('--x_noise', default=None, type=float)
parser.add_argument('-x', '--reservoir_x_init', default=None, type=str)
parser.add_argument('-a', '--test_all', action='store_true')
parser.add_argument('-n', '--no_plot', action='store_true')
parser.add_argument('-c', '--config', default=None, help='path to config file if custom')
args = parser.parse_args()

if args.config is None:
    config = get_config(args.model, ctype='model')
else:
    config = json.load(open(args.config, 'r'))
config = update_args(args, config)
dsets = config.dataset

net = load_model_path(args.model, config=config)
# assuming config is in the same folder as the model

if args.test_all:
    _, loss = test_model(net, config)
    print('avg summed loss (all):', loss)

if not args.no_plot:
    data, t_losses = test_model(net, config, n_tests=12)
    print('avg losses:')
    for t, j in t_losses.items():
        print(t + ': ' + str(j))
    run_id = '/'.join(args.model.split('/')[-3:-1])

    fig, ax = plt.subplots(3,4,sharex=False, sharey=False, figsize=(12,8))
    for i, ax in enumerate(fig.axes):
        context, ix, trial, x, y, z, loss = data[i]
        xr = np.arange(x.shape[-1])

        ax.axvline(x=0, color='dimgray', alpha = 1)
        ax.axhline(y=0, color='dimgray', alpha = 1)
        ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        if type(trial) in [DelayProAnti, MemoryProAnti]:
            ax.plot(xr, x[0], color='grey', lw=1, ls='--', alpha=.4)
            ax.plot(xr, x[1], color='salmon', lw=1, ls='--', alpha=.4)
            ax.plot(xr, x[2], color='dodgerblue', lw=1, ls='--', alpha=.4)
            ax.plot(xr, y[0], color='grey', lw=1.5, ls=':')
            ax.plot(xr, y[1], color='salmon', lw=1.5, ls=':')
            ax.plot(xr, y[2], color='dodgerblue', lw=1.5, ls=':')
            ax.plot(xr, z[0], color='grey', lw=2)
            ax.plot(xr, z[1], color='salmon', lw=2)
            ax.plot(xr, z[2], color='dodgerblue', lw=2)

        elif type(trial) in [RSG, CSG]:
            ax.plot(xr, y[0], color='coral', alpha=1, lw=1, label='target')
            ax.plot(xr, z[0], color='cornflowerblue', alpha=1, lw=1.5, label='response')
        elif 'bce' in config.loss:
            ax.scatter(xr, y, color='coral', alpha=0.5, s=3, label='target')
            ax.plot(xr, z, color='cornflowerblue', alpha=1, lw=1.5, label='response')

        elif type(trial) is FlipFlop:
            for j in range(trial.dim):
                ax.plot(xr, x[j], color=cols[j], ls='--', lw=.5, alpha=.4)
                ax.plot(xr, y[j], color=cols[j], lw=1.5, ls=':')
                ax.plot(xr, z[j], color=cols[j], lw=2)

        elif type(trial) is DurationDisc:
            ax.plot(xr, x[0], color='grey', lw=.5, ls='--', alpha=.4)
            ax.plot(xr, x[1], color='grey', lw=.5, ls='--', alpha=.4)
            ax.plot(xr, x[2], color='salmon', lw=.5, ls='--', alpha=.7)
            ax.plot(xr, x[3], color='dodgerblue', lw=.5, ls='--', alpha=.7)
            ax.plot(xr, y[0], color='salmon', lw=1.5, ls=':')
            ax.plot(xr, y[1], color='dodgerblue', lw=1.5, ls=':')
            ax.plot(xr, z[0], color='salmon', lw=2)
            ax.plot(xr, z[1], color='dodgerblue', lw=2)

        elif type(trial) is DMCProAnti:
            ax.plot(xr, x[0], color='grey', lw=.5, ls='--', alpha=.4)
            ax.plot(xr, x[1], color='salmon', lw=.5, ls='--', alpha=.4)
            ax.plot(xr, x[2], color='dodgerblue', lw=.5, ls='--', alpha=.7)
            ax.plot(xr, x[3], color='magenta', lw=.5, ls='--', alpha=.7)
            ax.plot(xr, x[4], color='lime', lw=.5, ls='--', alpha=.7)
            ax.plot(xr, y[0], color='grey', lw=1.5, ls=':')
            ax.plot(xr, y[1], color='magenta', lw=1.5, ls=':')
            ax.plot(xr, y[2], color='lime', lw=1.5, ls=':')
            ax.plot(xr, z[0], color='grey', lw=2, ls='-')
            ax.plot(xr, z[1], color='magenta', lw=2, ls='-')
            ax.plot(xr, z[2], color='lime', lw=2, ls='-')

        elif type(trial) is DelayDMProAnti:
            ax.plot(xr, x[0], color='grey', lw=.5, ls='--', alpha=.4)
            if trial.g1 > trial.g2:
                mod1_width = 1.5
                mod2_width = 0.5
            else:
                mod1_width = 0.5
                mod2_width = 1.5
            ax.plot(xr, x[1], color='salmon', lw=mod1_width, ls='--', alpha=.4)
            ax.plot(xr, x[2], color='dodgerblue', lw=mod1_width, ls='--', alpha=.7)
            ax.plot(xr, x[4], color='magenta', lw=mod2_width, ls='--', alpha=.7)
            ax.plot(xr, x[5], color='lime', lw=mod2_width, ls='--', alpha=.7)
            ax.plot(xr, y[0], color='grey', lw=1.5, ls=':')
            ax.plot(xr, y[1], color='magenta', lw=1.5, ls=':')
            ax.plot(xr, y[2], color='lime', lw=1.5, ls=':')
            ax.plot(xr, z[0], color='grey', lw=2, ls='-')
            ax.plot(xr, z[1], color='magenta', lw=2, ls='-')
            ax.plot(xr, z[2], color='lime', lw=2, ls='-')

        ax.tick_params(axis='both', color='white', labelsize=8)
        ax.set_title(f'ctx {context}, trial {ix}, loss {np.round(float(loss), 2)}', size=8)


    fig.text(0.5, 0.04, 'timestep', ha='center', va='center', size=12)
    fig.text(0.03, 0.5, 'value', ha='center', va='center', rotation='vertical', size=14)

    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(f'Final performance: {run_id}', size=14)
    fig.legend(handles, labels, loc='lower right')

    plt.tight_layout(rect=(.04, .06, 1, 0.95))

    plt.show()



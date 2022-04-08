import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import fig_format

import random
import os
import pdb
import json
import sys

from network import M2Net, M2Reservoir
from utils import Bunch, load_rb, get_config

from helpers import get_criteria, create_loaders


def load_model_path(path, config=None):
    if config is None:
        config = get_config(path)
    if type(config) is dict:
        config = Bunch(**config)
    config.model_path = path

    # net = BasicNetwork(config)
    net = M2Net(config)

    net.eval()
    return net

# given a model and a dataset, see how well the model does on it
# works with plot_trained.py
def test_model(net, config, n_tests=128):
    if config.sequential:
        config.sequential = False
    test_set, test_loader = create_loaders(config.dataset, config, split_test=False, test_size=n_tests)
    x, y, trials = next(iter(test_loader))

    criteria = get_criteria(config)

    t_losses = {}
    with torch.no_grad():
        contexts = [t.context for t in trials]
        idxs = [t.n for t in trials]

        # saving each individual loss per sample, per timestep
        losses = np.zeros(len(x))
        outs = []

        for j in range(x.shape[2]):
            # run the step
            net_in = x[:,:,j]
            net_out = net(net_in)
            outs.append(net_out)

        outs = torch.stack(outs, dim=2)
        targets = y
        # pdb.set_trace()
        
        for k in range(len(x)):
            t = trials[k]
            for c in criteria:
                losses[k] += c(outs[k], targets[k], i=trials[k], t_ix=0, single=True).item()
            if t.dname in t_losses:
                t_losses[t.dname].append(losses[k])
            else:
                t_losses[t.dname] = [losses[k]]


        data = list(zip(contexts, idxs, trials, x, y, outs, losses))
        for name in t_losses.keys():
            t_losses[name] = np.mean(t_losses[name])

    return data, t_losses

# returns hidden states as [N, T, H]
# note: this returns hidden states as the last dimension, not timesteps!
def get_states(net, x):
    states = []
    with torch.no_grad():
        net.reset()
        for j in range(x.shape[2]):
            net_in = x[:,:,j]
            net_out, extras = net(net_in, extras=True)
            states.append(extras['x'])

    A = torch.stack(states, dim=1)
    return A

def test_fixed_pts():
    torch.manual_seed(4)
    np.random.seed(3)
    random.seed(0)

    D2 = 3
    D1 = 50
    fixed_pts = 1
    t_len = 1000
    args = Bunch(
        N=500,
        D1=D1,
        D2=D2,
        # fixed_pts=fixed_pts,
        fixed_beta=1.5,
        res_x_seed=0,
        res_seed=0,
        res_init_g=1.5
    )
    reservoir = M2Reservoir(args)

    if fixed_pts > 0:
        # patterns = (2 * np.eye(args.N)-1)[:fixed_pts, :]
        reservoir.add_fixed_points(fixed_pts)
        # print(len(patterns))

    # pdb.set_trace()
    us = np.random.normal(0, .3, (16, D1))
    # us = np.zeros((16, D1))
    # us = np.random.normal(0, 1, (1, D1)) + np.random.normal(0, 0.1, (16, D1))
    us = torch.as_tensor(us, dtype=torch.float)
    vs = []
    for i, u in enumerate(us):
        reservoir.reset()
        trial_vs = []
        for t in range(t_len):
            if t < 500:
                v = reservoir(u)
            else:
                v = reservoir(None)
            trial_vs.append(v.detach())
        print(reservoir.x.detach().numpy()[0,:5])
        trial_vs = torch.cat(trial_vs)
        vs.append(trial_vs.numpy())

    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(16,12))
    cools = plt.cm.cool(np.linspace(0,1,D2))
    xaxis = range(t_len)
    for i, ax in enumerate(axes.ravel()):
        for j in range(D2):
            ax.plot(xaxis, vs[i][:,j], color=cools[j])

    fig_format.hide_frame(*axes.ravel())

    plt.show()

if __name__ == '__main__':
    test_fixed_pts()
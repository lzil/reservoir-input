import numpy as np
import matplotlib.pyplot as plt

import random
import pickle
import pdb
import torch

import argparse
from utils import Bunch

# for plotting some instances over the course of training

from reservoir import Network, Reservoir


args = Bunch(N=10, D=3, O=1, res_init_type='gaussian', res_init_params={'std': 1.5}, reservoir_seed=0)

net = Network(args)

trial_len = 100
t = np.arange(trial_len)

ins = []
outs = []
for i in range(12):
    net.reset()

    #inp = torch.normal(torch.zeros(trial_len), 500*torch.ones(trial_len))
    inp = -100000*torch.ones(trial_len)/4

    out = []
    for j in inp:
        out.append(net(j)[0].detach().item())

    ins.append(inp)
    outs.append(out)



# data_idx = [0]
# data_idx += sorted(random.sample(range(1, len(data) - 1), 10))
# data_idx += [len(data) - 1]

# print(data_idx)
# print(len(data))

# data = [data[i] for i in data_idx]

fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(12,7))

for i, ax in enumerate(fig.axes):

    ax.axvline(x=0, color='dimgray', alpha = 1)
    ax.axhline(y=0, color='dimgray', alpha = 1)
    ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.plot(t, ins[i], color='coral', alpha=0.5, lw=1, label='inp')
    ax.plot(t, outs[i], color='cornflowerblue', alpha=1, lw=1.5, label='out')

    ax.tick_params(axis='both', color='white')

    ax.set_ylim([-2,2])

fig.text(0.5, 0.04, 'timestep', ha='center', va='center')
fig.text(0.06, 0.5, 'value', ha='center', va='center', rotation='vertical')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')

plt.show()
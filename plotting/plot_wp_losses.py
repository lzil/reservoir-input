import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pdb
import os
import sys
import subprocess
sys.path.append('../')

from utils import get_config

import torch
# run_id = '3539456'

# csv_data = pd.concat([csv_data1, csv_data2])

color_scale = ['coral', 'dodgerblue']

# mint wp
csv_path = f'../logs/127/losses_0655086.csv'
df_mint_wp = pd.read_csv(csv_path)
df_mint_wp.test_loss = df_mint_wp.test_loss.apply(lambda x: np.log(float(x[7:-2])))


# dmpa wp
csv_path = f'../logs/137/losses_2317304.csv'
df_dmpa_wp = pd.read_csv(csv_path)
df_dmpa_wp.test_loss = df_dmpa_wp.test_loss.apply(lambda x: np.log(float(x[7:-2])))

# ming backprop
csv_path = f'../logs/38/losses_4516039.csv'
df_mint_bp = pd.read_csv(csv_path)
df_mint_bp.test_loss = df_mint_bp.test_loss.apply(lambda x: np.log(float(x[7:-2])))

# dmpa backprop
csv_path = f'../logs/29/losses_4524667.csv'
df_dmpa_bp = pd.read_csv(csv_path)
df_dmpa_bp.test_loss = df_dmpa_bp.test_loss.apply(lambda x: np.log(float(x[7:-2])))





plt.figure(figsize=(5,4))
ax = plt.gca()
# ax.set_xticklabels([0] + list(Ds), fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.grid(True, which='major', axis='y', lw=1, color='lightgray', alpha=0.4)
ax.grid(False)
ax.set_xlim([-1, 150])
ax.set_ylim([0, 6])

plt.axhline(y=2, color='black', linestyle=':', alpha=.3)

plt.plot(df_mint_wp.test_loss[::5].tolist(), lw=2, color=color_scale[0], label='mint+wp')
plt.plot(df_dmpa_wp.test_loss[::5].tolist(), lw=2, color=color_scale[1], label='dmpa+wp')
plt.plot(df_mint_bp.test_loss[::5].tolist(), lw=2, ls=':', color=color_scale[0], label='mint+bp')
plt.plot(df_dmpa_bp.test_loss[::5].tolist(), lw=2, ls=':', color=color_scale[1], label='dmpa+bp')



ax.legend()
plt.show()


sys.exit(0)



fig, axes = plt.subplots(nrows=len(mnoises), ncols=len(rseeds), sharex=True, sharey=True, figsize=(14,10), squeeze=False)
fig.text(0.07, 0.5, 'loss', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'D', ha='center')

# dt = dt[(dt.dset == 'datasets/rsg-sohn.pkl')]
# dt = dt[(dt.rnoise == 0.01)]
for i, mnoise in enumerate(mnoises):
    for j, rseed in enumerate(rseeds):
        subset = dt[(dt.mnoise == mnoise) & (dt.rseed == rseed)]

        ax = axes[i, j]
        ax.set_xticklabels([0] + list(Ds))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax.tick_params()
        # ax.xaxis.set_ticks_position('bottom')
        # ax.tick_params(which='major', width=1.00, length=4)
        if i == 0:
            ax.set_title('seed = ' + str(rseed))
        if j == 0:
            ax.set_ylabel('noise = ' + str(mnoise))
        
        # subset_all = subset[subset.]
        train_all = subset[subset.tparts == 'all']
        ax.scatter(train_all.D_map, train_all.loss, s=8, alpha=.7, c=color_scale[0], label='train all')
        means = []
        for D in Ds:
            means.append(np.mean(train_all[train_all.D == D]['loss']))
        ax.plot(range(len(Ds)), means, c=color_scale[0], ms=20)
        train_lim = subset[subset.tparts == 'W_f-W_ro']
        ax.scatter(train_lim.D_map, train_lim.loss, s=8, alpha=.7, c=color_scale[1], label='train part')
        means = []
        for D in Ds:
            means.append(np.mean(train_lim[train_lim.D == D]['loss']))
        ax.plot(range(len(Ds)), means, c=color_scale[1], ms=50)

        train_ro = subset[subset.tparts == 'W_ro']
        ax.scatter(train_ro.D_map, train_ro.loss, s=8, alpha=.7, c=color_scale[2], label='train ro')
        means = []
        for D in Ds:
            means.append(np.mean(train_ro[train_ro.D == D]['loss']))
        ax.plot(range(len(Ds)), means, c=color_scale[2], ms=20)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.axhline(y=0, color='dimgray', alpha = 1)
        # ax.axvline(x=-.5, color='dimgray', alpha = 1)
        # ax.tick_params(axis='both', color='white')
        ax.grid(None)
        ax.grid(True, which='major', axis='y', lw=1, color='lightgray', alpha=0.4)
        ax.set_xlim([-.5, len(Ds) - .5])
        ax.set_ylim([0, 20])


# plt.legend()
# plt.show()

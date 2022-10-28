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

# run_id = '3539456'


# run_id = '3540770' # run across many Ds

# csv_path = f'../logs/{run_id}.csv'
# csv_data1 = pd.read_csv(csv_path)

# run_id = '3541548' # across even more Ds

# csv_path = f'../logs/{run_id}.csv'
# csv_data2 = pd.read_csv(csv_path)

# csv_data = pd.concat([csv_data1, csv_data2])

#run_id = '3537318'
#run_id = '6406203'
run_id ='hworkdelayprotest/losses_6406203'

csv_path = f'../logs/{run_id}.csv'
csv_data = pd.read_csv(csv_path)

# vals = []
# for i in csv_data.slurm_id:
#     run_dir = os.path.join(f'../logs/{run_id}', str(i))
#     run_files = os.listdir(run_dir)
#     for f in run_files:
#         if f.startswith('config'):
#             c_file = os.path.join(run_dir, f)
#     config = get_config(c_file, ctype='model')
#     vals.append(config['m_noise'])
# csv_data['mnoise'] = vals

csv_data['tparts'].fillna('all', inplace=True)

cols_to_keep = ['slurm_id', 'N', 'D', 'seed', 'rseed', 'rnoise', 'mnoise', 'dset', 'loss', 'tparts']
dt = csv_data[cols_to_keep]
dt = dt.sort_values(by=['D', 'mnoise', 'rseed'])


# mapping Ds so we can plot it as factor later
Ds = dt['D'].unique()
D_map = dict(zip(Ds, range(len(Ds))))
dt['D_map'] = dt['D'].map(D_map)

dt['D_map'] += np.random.normal(0, .05, len(dt['D_map']))

rnoises = dt['rnoise'].unique()
mnoises = dt['mnoise'].unique()
rseeds = dt['rseed'].unique()

color_scale = ['coral', 'chartreuse', 'skyblue']

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


plt.legend()
plt.show()

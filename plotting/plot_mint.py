import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pdb
import os
import sys
import subprocess
sys.path.append('../')

# import fig_format

from utils import get_config, load_rb
from testers import load_model_path, test_model


# run_id = '4203227'
run_id = '4203794'
# run_id = '4203461'
csv_path = f'logs/{run_id}.csv'
csv_data1 = pd.read_csv(csv_path)

run_id = '4203461'
csv_path = f'logs/{run_id}.csv'
csv_data2 = pd.read_csv(csv_path)


# csv_data['tparts'].fillna('all', inplace=True)

cols_to_keep = ['slurm_id', 'N', 'D1', 'seed', 'rseed', 'loss', 'lr_wp']
dt1 = csv_data1[cols_to_keep]


dt1 = dt1[dt1['lr_wp'] != 1e-8]
# mapping Ds so we can plot it as factor later
dt1 = dt1.sort_values(by=['N', 'D1', 'lr_wp'])
Ds = dt1['D1'].unique()
D_map = dict(zip(Ds, range(len(Ds))))
dt1['D_map'] = dt1['D1'].map(D_map)

dt1['D_map'] += np.random.normal(0, .05, len(dt1['D_map']))

dt1_aggs = dt1.groupby(['N', 'D1'])['loss'].agg(['mean', 'std']).reset_index()



dt2 = csv_data2[cols_to_keep]

dt2 = dt2[dt2['lr_wp'] != 1e-8]
# mapping Ds so we can plot it as factor later
dt2 = dt2.sort_values(by=['N', 'D1', 'lr_wp'])
Ds = dt2['D1'].unique()
D_map = dict(zip(Ds, range(len(Ds))))
dt2['D_map'] = dt2['D1'].map(D_map)

dt2['D_map'] += np.random.normal(0, .05, len(dt2['D_map']))


dt2_aggs = dt2.groupby(['N', 'D1'])['loss'].agg(['mean', 'std']).reset_index()

# print(dt_aggs)

plt.figure(figsize=(5,4))
ax = plt.gca()
ax.set_xticklabels([0] + list(Ds), fontsize=13)
plt.yticks(fontsize=13)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

color_scale = ['salmon', 'coral', 'skyblue', 'salmon', 'cyan', 'magenta']
labels=['N=200', 'N=500']
i = 0
for j, N in enumerate(dt1['N'].unique()):
    # for k, lr in enumerate(dt['lr_wp'].unique()):
        condition = (dt1['N'] == N)
        plt.scatter(dt1[condition]['D_map'], dt1[condition]['loss'], s=8, alpha=.3, c=color_scale[i])
        # pdb.set_trace()
        condition = (dt1_aggs['N'] == N)
        plt.errorbar(range(len(Ds)), dt1_aggs[condition ]['mean'], yerr=[dt1_aggs[condition ]['std'], dt1_aggs[condition ]['std']],c=color_scale[i], lw=3, label=labels[j])
        # plt.errorbar(dt_aggs[condition]['D1'], dt_aggs[condition]['mean'], yerr=[])
        i += 1

for j, N in enumerate(dt2['N'].unique()):
    # for k, lr in enumerate(dt['lr_wp'].unique()):
        condition = (dt2['N'] == N)
        plt.scatter(dt2[condition]['D_map'], dt2[condition]['loss'], s=8, alpha=.3, c=color_scale[i])
        # pdb.set_trace()
        condition = (dt2_aggs['N'] == N)
        plt.errorbar(range(len(Ds)), dt2_aggs[condition ]['mean'], yerr=[dt2_aggs[condition ]['std'], dt2_aggs[condition ]['std']],c=color_scale[i], lw=3, label=labels[j])
        # plt.errorbar(dt_aggs[condition]['D1'], dt_aggs[condition]['mean'], yerr=[])
        i += 1

ax.set_xlabel('D')
ax.set_ylabel('mean squred error')

plt.xlim([-.25, len(Ds) - .5])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(None)
plt.grid(False, which='major', axis='y', lw=1, color='lightgray', alpha=0.4)
plt.grid(False)

plt.legend()
plt.show()


sys.exit(0)

# plt.figure(figsize=(5,4))
# ax = plt.gca()
# ax.plot(range(400), range(400), color='black', lw=.6, linestyle='--')
# plt.axhline(150, c='black', linestyle='--', color='black', lw=.6)

color_scale = ['coral', 'cornflowerblue', 'skyblue']




plt.scatter()

train_all = dt[dt.tparts == 'all']
train_lim = dt[dt.tparts == 'W_f-W_ro']
plt.scatter(train_all.D_map, train_all.loss, s=8, alpha=.3, c=color_scale[0])
plt.scatter(train_lim.D_map, train_lim.loss, s=8, alpha=.3, c=color_scale[1])
q_all = []
q_lim = []
for D in Ds:
    try:
        # q = np.quantile(train_all[train_all.D == D]['loss'], [.25, .5, .75])
        # q = [q[1], q[1] - q[0], q[2] - q[1]]
        mn = np.mean(train_all[train_all.D == D]['loss'])
        sd = np.std(train_all[train_all.D == D]['loss'])
        q = [mn, sd, sd]
        q_all.append(q)
    except IndexError:
        q_all.append([0,0,0])
    try:
        # q = np.quantile(train_lim[train_lim.D == D]['loss'], [.25, .5, .75])
        # q = [q[1], q[1] - q[0], q[2] - q[1]]
        mn = np.mean(train_lim[train_lim.D == D]['loss'])
        sd = np.std(train_lim[train_lim.D == D]['loss'])
        q = [mn, sd, sd]
        q_lim.append(q)
    except IndexError:
        q_lim.append([0,0,0])
q_all = list(zip(*q_all))
q_lim = list(zip(*q_lim))
plt.errorbar(range(len(Ds)), q_all[0], yerr = [q_all[1], q_all[2]], c=color_scale[0], ms=20, elinewidth=2, lw=3, label='trained RNN')
plt.errorbar(range(len(Ds)), q_lim[0], yerr = [q_lim[1], q_lim[2]], c=color_scale[1], ms=20, elinewidth=2, lw=3, label='fixed RNN')


# fig, axes = 

sys.exit(0)




sys.exit(0)

color_scale = ['coral', 'chartreuse', 'skyblue']

fig, axes = plt.subplots(nrows=len(mnoises), ncols=len(dsets), figsize=(14,10), squeeze=False)
fig.text(0.07, 0.5, 'loss', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'D', ha='center')

dt = dt[dt.D == 20]
dt = dt[dt.tparts == 'W_f-W_ro']
dt = dt[(dt.seed == 12) & (dt.rseed > 1) & (dt.rnoise == 0.01)]

# dt = dt[(dt.dset == 'datasets/rsg-sohn.pkl')]
# dt = dt[(dt.rnoise == 0.01)]
for i, mnoise in enumerate(mnoises):
    for j, dset in enumerate(dsets):
        subset = dt[(dt.mnoise == mnoise) & (dt.dset == dset)]
        ax = axes[i, j]

        for iterr in range(len(subset)):

            job_id = subset.iloc[iterr].slurm_id

            model_folder = os.path.join('..', 'logs', run_id, str(job_id))
            model_path = os.path.join(model_folder, 'model_best.pth')
            config = get_config(model_path, ctype='model', to_bunch=True)
            config.m_noise = 0
            net = load_model_path(model_path, config=config)

            data, loss = test_model(net, config, n_tests=200, dset_base='../')
            dset = load_rb(os.path.join('..', config.dataset))

            distr = {}

            for k in range(len(data)):
                dset_idx, x, _, z, _ = data[k]
                r, s, g = dset[dset_idx][2]

                t_first = torch.nonzero(z >= 1)
                if len(t_first) > 0:
                    t_first = t_first[0,0]
                else:
                    t_first = len(x)

                mode = 'offsets'
                if mode == 'offsets':
                    val = t_first - g
                elif mode == 'times':
                    val = t_first
                elif mode == 'intervals':
                    val = t_first - s

                val = np.asarray(val)

                interval = g - s + 5
                if interval not in distr:
                    distr[interval] = [val]
                else:
                    distr[interval].append(val)

            intervals = []
            for k,v in distr.items():
                v_avg = np.mean(v)
                v_std = np.std(v)
                intervals.append((k,v_avg, v_std))

            intervals.sort(key=lambda x: x[0])
            intervals, vals, stds = list(zip(*intervals))
            vals = np.array(vals)
            stds = np.array(stds)

            ax.scatter(intervals, vals, marker='o', color=color_scale[iterr], alpha=0.5)

        x_min, x_max = min(intervals), max(intervals)
        y_min, y_max = min(vals), max(vals)
        xdiff = x_max - x_min
        ydiff = y_max - y_min
        x_min -= .1 * xdiff; y_min -= .1 * ydiff
        x_max += .1 * xdiff; y_max += .1 * ydiff
        ax.plot(range(int(x_max)), range(int(x_max)))
        # plt.fill_between(intervals, offsets - stds, offsets, color='coral', alpha=.5)
        # plt.fill_between(intervals, offsets + stds, offsets, color='coral', alpha=.5)
        ax.set_xlabel('real t_p')

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        print('finished', i, j)

plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pdb

import argparse

from tasks import RSG
from utils import get_config, load_rb
from testers import load_model_path, test_model

cols = ['coral', 'cornflowerblue', 'magenta', 'orchid']

def main(args):
    config = get_config(args.model, to_bunch=True)
    net = load_model_path(args.model, config)

    data, loss = test_model(net, config, n_tests=150)

    ys = [[] for i in range(net.args.T)]
    ys_stim = [[] for i in range(net.args.T)]

    for d in data:
        context, idx, trial, x, y, out, loss = d
        y_ready, y_set, y_go = trial.rsg
        y_prod = np.argmax(out >= 1)
        t_y = y_set - y_ready
        t_p = y_prod - y_set
        t_ym = y_go - y_set
        
        ys[context].append((t_y, t_p))
        ys_stim[context].append((t_y, t_ym))

    ys = [list(zip(*np.array(y))) for y in ys]
    ys_stim = [list(zip(*np.array(y))) for y in ys_stim]

    for i in range(net.args.T):
        plt.scatter(ys_stim[i][0], ys_stim[i][1], marker='o', c='black', s=20, edgecolors=cols[i])
        plt.scatter(ys[i][0], ys[i][1], color=cols[i], s=15)
        
    plt.xlabel('desired t_p')
    plt.ylabel('produced t_p')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='model path')
    args = parser.parse_args()


    main(args)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

import argparse

from tasks import RSG
from utils import get_config, load_rb
from testers import load_model_path, test_model


def main(args):
    # model_path = 'logs/test50/model_best.pth'
    # # model_path = 'logs/test50/model_4628892.pth'
    # model_path = 'logs/test50/model_1746859.pth'
    # model_path = 'logs/train_part1_1,2/model_best.pth'
    # # model_path = 'logs/train_part1_1,2_2/model_best.pth'
    # # model_path = 'logs/train_part_1,2/model_best.pth'
    # model_path = 'logs/rsg_1,2_N300/model_best.pth'

    model_path = args.model
    # model_path = 'logs/train_part1'
    config = get_config(model_path, to_bunch=True)
    # config.dataset = ['datasets/rsg-150-200.pkl', 'datasets/rsg-100-150.pkl']
    # config.dataset = ['datasets/rsg-100-5-150.pkl', 'datasets/rsg-150-5-200.pkl']
    net = load_model_path(model_path, config)

    # dsets = []
    # for d in config.dataset:
    #     dset = load_rb(d)
    #     dsets.append(dset)

    data, loss = test_model(net, config, n_tests=500)
    
    # 2 contexts
    ys_1 = []
    ys_2 = []

    y_ready_set = []
    for d in data:
        # pdb.set_trace()
        y_ready = np.argmax(d[1][0] > 0)
        y_set = np.argmax(d[2] > 0)
        y_target = np.argmax(d[2] >= 1)
        y_prod = np.argmax(d[3] >= 1)
        if d[5].context == 0:
            ys_1.append((y_target - y_set, y_prod - y_set))
        else:
            ys_2.append((y_target - y_set, y_prod - y_set))
        y_ready_set.append((y_set - y_ready, y_target - y_set))


    ys_1 = np.array(ys_1)
    ys_1 = list(zip(*ys_1))

    ys_2 = np.array(ys_2)
    ys_2 = list(zip(*ys_2))

    y_ready_set = np.array(y_ready_set)
    y_ready_set = list(zip(*y_ready_set))
    # sns.relplot(ys[0], ys[1])
    plt.scatter(ys_1[0], ys_1[1], color='salmon')
    if len(ys_2) > 0:
        plt.scatter(ys_2[0], ys_2[1], color='dodgerblue')

    plt.ylabel('produced t_p')
    plt.xlabel('desired t_p')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='model path')
    args = parser.parse_args()


    main(args)
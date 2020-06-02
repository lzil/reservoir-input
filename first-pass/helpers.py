
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import random
import pdb

from reservoir import Network
from utils import Bunch

def test_model(m_dict, dset, n_tests=0):
    bunch = Bunch()
    bunch.N = m_dict['reservoir.J.weight'].shape[0]
    bunch.D = m_dict['reservoir.W_u.weight'].shape[1]
    bunch.O = m_dict['W_f.weight'].shape[1]

    bunch.res_init_type = 'gaussian'
    bunch.res_init_params = {'std': 1.5}
    bunch.reservoir_seed = 0

    net = Network(bunch)
    net.load_state_dict(m_dict)
    net.eval()

    criterion = nn.MSELoss()

    dset_idx = range(len(dset))
    if n_tests != 0:
        dset_idx = sorted(random.sample(range(len(dset)), n_tests))

    xs = []
    ys = []
    zs = []
    losses = []

    ix = 0
    for ix in range(len(dset_idx)):
        trial = dset[dset_idx[ix]]
        # next data sample
        x = torch.from_numpy(trial[0]).float()
        y = torch.from_numpy(trial[1]).float()
        xs.append(x)
        ys.append(y)

        net.reset()

        outs = []

        total_loss = torch.tensor(0.)
        for j in range(x.shape[0]):
            # run the step
            net_in = x[j].unsqueeze(0)
            net_out, val_thal, val_res = net(net_in)

            # this is the desired output
            net_target = y[j].unsqueeze(0)
            outs.append(net_out.item())

            # this is the loss from the step
            step_loss = criterion(net_out.view(1), net_target)

            total_loss += step_loss

        z = np.stack(outs).squeeze()
        zs.append(z)
        losses.append(total_loss.item())

    data = list(zip(dset_idx, xs, ys, zs, losses))

    return data


def get_optimizer(args, train_params):
    optimizer = None
    if args.optimizer == 'adam':
        optimizer = optim.Adam(train_params, lr=args.lr)
    elif args.optimizer == 'lbfgs-pytorch':
        optimizer = optim.LBFGS(train_params, lr=0.75)
    return optimizer

def get_criterion(args):
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    return criterion

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import pdb

import random
from collections import OrderedDict

from utils import load_rb, get_config

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_optimizer(args, train_params):
    op = None
    if args.optimizer == 'adam':
        op = optim.Adam(train_params, lr=args.lr, weight_decay=args.l2_reg)
    elif args.optimizer == 'sgd':
        op = optim.SGD(train_params, lr=args.lr, weight_decay=args.l2_reg)
    elif args.optimizer == 'rmsprop':
        op = optim.RMSprop(train_params, lr=args.lr, weight_decay=args.l2_reg)
    elif args.optimizer == 'lbfgs-pytorch':
        op = optim.LBFGS(train_params, lr=0.75)
    return op

def get_scheduler(args, op):
    if args.s_rate is not None:
        return optim.lr_scheduler.MultiStepLR(op, milestones=[1,2,3], gamma=args.s_rate)
    return None


# dataset that automatically creates trials composed of trial and context data
# input dataset should be in form [(dname, dset), ...]
class TrialDataset(Dataset):
    def __init__(self, datasets, args):
        self.args = args
        
        self.dnames = []    # names of dsets
        self.data = []      # dsets themselves
        self.t_types = []   # task type
        self.lzs = []       # Ls and Zs for task trials
        self.x_ctxs = []    # precomputed context inputs
        self.max_idxs = np.zeros(len(datasets), dtype=int)
        self.t_lens = []
        for i, (dname, ds) in enumerate(datasets):
            self.dnames.append(dname)
            self.data.append(ds)
            # setting context cue for appropriate task
            x_ctx = np.zeros((args.T, ds[0].t_len))
            x_ctx[i] = 1
            self.x_ctxs.append(x_ctx)
            self.t_lens.append(ds[0].t_len)
            # cumulative lengths of data, for indexing
            self.max_idxs[i] = self.max_idxs[i-1] + len(ds)
            # use ds[0] as exemplar to set t_type, L, Z
            self.t_types.append(ds[0].t_type)
            self.lzs.append((ds[0].L, ds[0].Z))

    def __len__(self):
        return self.max_idxs[-1]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[ii] for ii in range(len(self))[idx]]
        # index into the appropriate dataset to get the trial
        context = self.get_context(idx)
        # idx variable now references position within dataset
        if context != 0:
            idx = idx - self.max_idxs[context-1]

        trial = self.data[context][idx]
        x = trial.get_x(self.args)
        x_cts = self.x_ctxs[context]
        # context comes after the stimulus
        x = np.concatenate((x, x_cts))
        y = trial.get_y(self.args)

        trial.context = context
        trial.dname = self.dnames[context]
        trial.lz = self.lzs[context]
        return x, y, trial

    def get_context(self, idx):
        return np.argmax(self.max_idxs > idx)


# turns data samples into stuff that can be run through network
def collater(samples):
    xs, ys, trials = list(zip(*samples))
    # pad xs and ys to be the length of the max-length example
    max_len = np.max([x.shape[-1] for x in xs])
    xs_pad = [np.pad(x, ([0,0],[0,max_len-x.shape[-1]])) for x in xs]
    ys_pad = [np.pad(y, ([0,0],[0,max_len-y.shape[-1]])) for y in ys]
    xs = torch.as_tensor(np.stack(xs_pad), dtype=torch.float)
    ys = torch.as_tensor(np.stack(ys_pad), dtype=torch.float)
    return xs, ys, trials

# creates datasets and dataloaders
def create_loaders(datasets, args, split_test=True, test_size=1, context_filter=[]):
    dsets_train = []
    dsets_test = []
    for i, dpath in enumerate(datasets):
        dset = load_rb(dpath)
        # trim and set name of each dataset
        dname = str(i) + '_' + ':'.join(dpath.split('/')[-1].split('.')[:-1])
        if split_test:
            cutoff = round(.9 * len(dset))
            dsets_train.append([dname, dset[:cutoff]])
            dsets_test.append([dname, dset[cutoff:]])
        else:
            dsets_test.append([dname, dset])

    # creating datasets
    test_set = TrialDataset(dsets_test, args)
    if split_test:
        train_set = TrialDataset(dsets_train, args)

    # TODO: make all this code better
    if args.sequential:
        # helper function for sequential loaders
        def create_subset_loaders(dset, batch_size, drop_last):
            loaders = []
            max_idxs = dset.max_idxs
            for i in range(len(datasets)):
                if i == 0:
                    subset = Subset(dset, range(max_idxs[0]))
                else:
                    subset = Subset(dset, range(max_idxs[i-1], max_idxs[i]))
                loader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collater, drop_last=drop_last)
                loaders.append(loader)
            return loaders
        # create the loaders themselves
        test_loaders = create_subset_loaders(test_set, test_size, False)
        if split_test:
            train_loaders = create_subset_loaders(train_set, args.batch_size, True)
            return (train_set, train_loaders), (test_set, test_loaders)
        return (test_set, test_loaders)
    # filter out some contexts
    elif len(context_filter) != 0:
        def create_context_loaders(dset, batch_size, drop_last):
            max_idxs = dset.max_idxs
            c_range = []
            for i in range(len(datasets)):
                if i in context_filter:
                    continue
                if i == 0:
                    c_range += list(range(max_idxs[0]))
                else:
                    c_range += list(range(max_idxs[i-1], max_idxs[i]))
            subset = Subset(dset, c_range)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collater, drop_last=drop_last)
            return loader
        # create the loaders themselves
        test_loaders = create_context_loaders(test_set, test_size, False)
        if split_test:
            train_loaders = create_context_loaders(train_set, args.batch_size, True)
            return (train_set, train_loaders), (test_set, test_loaders)
        return (test_set, test_loaders)
        
    else:
        # otherwise it's quite simple, create a single dataset and loader
        test_loader = DataLoader(test_set, batch_size=test_size, shuffle=True, collate_fn=collater, drop_last=False)
        if split_test:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collater, drop_last=True)
            return (train_set, train_loader), (test_set, test_loader)
        return (test_set, test_loader)
        


def get_criteria(args):
    criteria = []
    if 'mse' in args.loss:
        # do this in a roundabout way due to truncated bptt
        fn = nn.MSELoss(reduction='sum')
        def mse(o, t, i, t_ix, single=False):
            # last dimension is number of timesteps
            # divide by batch size to avoid doing so logging and in test
            # needs all the contexts to be the same length
            loss = 0.
            if single:
                o = o.unsqueeze(0)
                t = t.unsqueeze(0)
                i = [i]
            for j in range(len(t)):
                length = i[j].t_len
                if t_ix + t.shape[-1] <= length:
                    loss += fn(t[j], o[j])# / length
                elif t_ix < length:
                    t_adj = t[j,:,:length-t_ix]
                    o_adj = o[j,:,:length-t_ix]
                    loss += fn(t_adj, o_adj)# / length
            return args.l1 * loss / args.batch_size
        criteria.append(mse)
    if 'bce' in args.loss:
        weights = args.l3 * torch.ones(1)
        fn = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=weights)
        def bce(o, t, **kwargs):
            return args.l1 * fn(t, o)
        criteria.append(bce)
    if 'mse-e' in args.loss:
        # ONLY FOR RSG AND CSG, WITH [1D] OUTPUT
        # exponential decaying loss from the go time on both sides
        # loss is 1 at go time, 0.5 at set time
        # normalized to the number of timesteps taken
        fn = nn.MSELoss(reduction='none')
        def mse_e(o, t, i, t_ix, single=False):
            loss = 0.
            if single:
                o = o.unsqueeze(0)
                t = t.unsqueeze(0)
                i = [i]
            for j in range(len(t)):
                # last dimension is number of timesteps
                t_len = t.shape[-1]
                xr = torch.arange(t_len, dtype=torch.float)
                # placement of go signal in subset of timesteps
                t_g = i[j].rsg[2] - t_ix
                t_p = i[j].t_p
                # exponential loss centred at go time
                # dropping to 0.25 at set time
                lam = -np.log(4) / t_p
                # left half, only use if go time is to the right
                if t_g > 0:
                    xr[:t_g] = torch.exp(-lam * (xr[:t_g] - t_g))
                # right half, can use regardless because python indexing is nice
                xr[t_g:] = torch.exp(lam * (xr[t_g:] - t_g))
                # normalize, just numerically calculate area
                xr = xr / torch.sum(xr) * t_len
                # only the first dimension matters for rsg and csg output
                loss += torch.dot(xr, fn(o[j][0], t[j][0]))
            return args.l2 * loss / args.batch_size
        criteria.append(mse_e)
    if len(criteria) == 0:
        raise NotImplementedError
    return criteria

def get_activation(name):
    if name == 'exp':
        fn = torch.exp
    elif name == 'relu':
        fn = nn.ReLU()
    elif name == 'sigmoid':
        fn = nn.Sigmoid()
    elif name == 'tanh':
        fn = nn.Tanh()
    elif name == 'none':
        fn = lambda x: x
    return fn

def get_dim(a):
    if hasattr(a, '__iter__'):
        return len(a)
    else:
        return 1

    # return l2 * total_loss

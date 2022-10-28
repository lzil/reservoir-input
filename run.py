import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.optimize import minimize

import os
import argparse
import pdb
import sys
import pickle
import logging
#logging module defines functions and classes which implement a flexible event logging system for applications and libraries
import random
import csv
import math
import json
import copy
import pandas as pd


from utils import log_this, load_rb, get_config, update_args, load_args
from helpers import get_optimizer, get_scheduler, get_criteria, create_loaders

from tasks import *

from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-L', type=int, default=5, help='latent input dimension')
    parser.add_argument('--D1', type=int, default=50, help='u dimension')
    parser.add_argument('--D2', type=int, default=50, help='v dimension')
    parser.add_argument('-N', type=int, default=300, help='number of neurons in reservoir')
    # parser.add_argument('-Z', type=int, default=5, help='output dimension')

    parser.add_argument('--net', type=str, default='M2', choices=['basic', 'M2'])
    parser.add_argument('--train_parts', type=str, nargs='+', default=['M_u', 'M_ro'])
    parser.add_argument('-c', '--config', type=str, default=None, help='use args from config file')
    
    # make sure model_config path is specified if you use any paths! it ensures correct dimensions, bias, etc.
    parser.add_argument('--model_config_path', type=str, default=None, help='config path corresponding to model load path')
    parser.add_argument('--model_path', type=str, default=None, help='start training from certain model. superseded by below')
    parser.add_argument('--M_path', type=str, default=None, help='start training from certain in/out representations')
    parser.add_argument('--res_path', type=str, default=None, help='start training from certain reservoir representation')
    
    # network arguments
    # parser.add_argument('--res_init_type', type=str, default='gaussian', help='')
    parser.add_argument('--res_init_g', type=float, default=1.5) #g that determines variance from which we draw the weights
    parser.add_argument('--res_noise', type=float, default=0) 
    parser.add_argument('--fixed_pts', type=int, default=0, help='number of fixed pts to include as hopfield')
    parser.add_argument('--fixed_beta', type=float, default=1.5, help='beta to make patterns stronger')
    parser.add_argument('--x_noise', type=float, default=0)
    parser.add_argument('--m_noise', type=float, default=0)
    parser.add_argument('--res_bias', action='store_true', help='bias term as part of recurrent connections, with J')
    parser.add_argument('--ff_bias', action='store_true', help='bias in feedforward part of the network, with M_u and M_ro')
    parser.add_argument('--m1_act', type=str, default='none', help='act fn bw M_u and W_u')
    parser.add_argument('--m2_act', type=str, default='none', help='act fn bw W_ro and M_ro')
    parser.add_argument('--out_act', type=str, default='none', help='output activation at the very end of the network')
    parser.add_argument('--net_fb', action='store_true', help='feedback from network output to input')

    # dataset arguments
    parser.add_argument('-d', '--dataset', type=str, default=['datasets/rsg-100-150.pkl'], nargs='+', help='dataset(s) to use. >1 means different contexts')
    #ah I think args.datasets actually stores all the datasets that we feed it in a list
    #yes look the default value is a list with one element!
    #so we can feed it multiple datasets and it will store them in a list
    #which is why in "shortcut for training in designated order" we can get
    #the number of datasets passed in through args.dataset by going len(args.dataset)


    #dataset parameter used to find the dataset .pkl file we created using the json file and the tasks.py etc
    #notice the default dataset is datasets/rsg-100-150.pkl'
    #input is the path to the dataset
    # parser.add_argument('-a', '--add_tasks', type=str, nargs='+', help='add tasks to previously trained reservoir')


    #sequential training
    parser.add_argument('-s', '--sequential', action='store_true', help='sequential training')
    #whether we're training multiple tasks (sequentially - I think we almost always train multiple tasks sequentially)

    parser.add_argument('--owm', action='store_true', help='use orthogonal weight modification')
    parser.add_argument('-o', '--train_order', type=int, nargs='+', default=[], help='ids of tasks to train on, in order if sequential flag is enabled. empty for all')


    # synaptic stabilization and XdG


    # synaptic stabilization
    parser.add_argument('--ss', action='store_true', help='use synaptic stabilization')
    #we're only doing ss for units in u layer for now but we'll extend it to the v units and x units later as we did for xdg
    parser.add_argument('--stabilize_layers', type=str, default='u', help='the hidden layers of the net in which to apply the synaptic stabilization') #later make default ['u', 'v']
    parser.add_argument('--c_strength', type=float, default=1.2, help= 'stabilization strength')
    parser.add_argument('--ss_type', type=str, default='SI', choices=['SI', 'EWC'], help= 'choices for synaptic stabilization; default is synaptic intelligence')
    parser.add_argument('-C', type=float, default=0.01,  help='damping term for synaptic intelligence')
    #after we experiment with different c (C) values, make the optimal-for-default-net-arguments c (C) the default,

    #to run ss and xdg together simply go: --ss --xdg

    #note to self: context dependent gating means context signal as done in Masse and gating with X chosen below
    # context dependent gating (XdG)
    parser.add_argument('--xdg', action ='store_true', help = 'use context dependent gating')

    parser.add_argument('--cs_bias', action='store_true', help='bias in context signal part of the network, with M_u_cs, M_x_cs, M_v_cs')
    
    parser.add_argument('-X', type=float, default=50, help= "gating the activity of X percent of hidden units for each task, chosen randomly")
    #context signal is implemented in network.py just as net_fb is 
    parser.add_argument('--gate_layers', type=str, default=['u','v'], help='the hidden layers of the net in which to apply the gating')
    
    #['u','v'] as default but eventually will work with any hidden layer
    
    #parser.add_argument('--ss_xdg', action='store_true', help='use synaptic stabilization and context dependent gating(XdG)')
    




    parser.add_argument('--seq_threshold', type=float, default=5, help='threshold for having solved a task before moving on to next one')

    parser.add_argument('--same_test', action='store_true', help='use entire dataset for both training and testing')
    #when would you ever want to use the entire dataset for both training and testing?
    
    # training arguments
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop', 'lbfgs'], default='adam')
    parser.add_argument('--k', type=int, default=0, help='k for t-bptt. use 0 for full bptt')

    # adam parameters
    parser.add_argument('--batch_size', type=int, default=1, help='size of minibatch used')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. adam only')
    parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs to train for. adam only')
    parser.add_argument('--conv_type', type=str, choices=['patience', 'grad'], default='patience', help='how to determine convergence. adam only')
    parser.add_argument('--patience', type=int, default=4000, help='stop training if loss doesn\'t decrease. adam only')
    parser.add_argument('--l2_reg', type=float, default=0, help='amount of l2 regularization')
    parser.add_argument('--s_rate', default=None, type=float, help='scheduler rate. dont use for no scheduler')
    parser.add_argument('--loss', type=str, nargs='+', default=['mse'])

    # adam lambdas
    parser.add_argument('--l1', type=float, default=1, help='weight of normal loss')
    parser.add_argument('--l2', type=float, default=1, help='weight of exponential loss')

    # lbfgs parameters
    parser.add_argument('--maxiter', type=int, default=50, help='lbfgs max iterations')

    # seeds
    parser.add_argument('--seed', type=int, help='general purpose seed')
    parser.add_argument('--network_seed', type=int, help='seed for network initialization')
    parser.add_argument('--res_seed', type=int, help='seed for reservoir')
    parser.add_argument('--res_x_seed', type=int, default=0, help='seed for reservoir init hidden states. -1 for zero init')
    parser.add_argument('--res_burn_steps', type=int, default=200, help='number of steps for reservoir to burn in')

    parser.add_argument('-x', '--res_x_init', type=str, default=None, help='other seed options for reservoir')

    # control logging
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--log_interval', type=int, default=50)
    #log_interval is the length of the logging intervals i.e. after how many (is it batches or training) samples we log 
    #the log is shown in terminal and we can see the log_interval on LHS
    parser.add_argument('--log_checkpoint_models', action='store_true')
    parser.add_argument('--log_checkpoint_samples', action='store_true')

    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--slurm_param_path', type=str, default=None)
    parser.add_argument('--slurm_id', type=int, default=None)
    parser.add_argument('--use_cuda', action='store_true')
    #we do say use_cuda in command line but we pass an argument in a different way
    #we specify action "store_true" which store the Boolean True in args.use_cuda
    #see self.device in Trainer object
    #(note: you have to go --use_cuda in command line, )

    args = parser.parse_args()
    return args

def adjust_args(args):
    # don't use logging.info before we initialize the logger!! or else stuff is gonna fail

    # dealing with slurm. do this first!! before anything else
    # needs to be before seed setting, so we can set it
    if args.slurm_id is not None:
        from parameters import apply_parameters
        args = apply_parameters(args.slurm_param_path, args)

    # loading from a config file
    if args.config is not None:
        config = load_args(args.config)
        args = update_args(args, config)

    # setting seeds
    if args.res_seed is None:
        args.res_seed = random.randrange(1e6)
    if args.seed is None:
        args.seed = random.randrange(1e6)
    if args.network_seed is None:
        args.network_seed = random.randrange(1e6)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # TODO
    # in case we are loading from a model
    # if we don't use this we might end up with an error when loading model
    # uses a new seed
    if args.model_path is not None:
        config = get_config(args.model_path)
        args = update_args(args, config, overwrite=None) # overwrite Nones only
        enforce_same = ['N', 'D1', 'D2', 'net', 'res_bias', 'use_reservoir']
        for v in enforce_same:
            if v in config and args.__dict__[v] != config[v]:
                print(f'Warning: based on config, changed {v} from {args.__dict__[v]} -> {config[v]}')
                args.__dict__[v] = config[v]

    # shortcut for specifying train everything including reservoir
    if args.train_parts == ['all']:
        args.train_parts = ['']

    # shortcut for training in designated order
    if args.sequential and len(args.train_order) == 0:
        #train_order default is [], and length of empty list is 0
        #what is in args.dataset exactly
        args.train_order = list(range(len(args.dataset)))
        #gives us the indices of the the different task datasets
        #0 is automatically the first, this allows us to make sure 
        #the orders line up throughout: see trainer
        #args.train_order is the order in which we train on tasks
        #arange gives us integers from 0 to len(args.dataset)-1 inclusive
        #and then we put these integers in a list [0,..., len(args.dataset)-1]

    # TODO
    if 'rsg' in args.dataset[0]:
        args.out_act = 'exp'
    else:
        args.out_act = 'none'

    # number of task variables, latent variables, and output variables
    args.T = len(args.dataset) #i.e. number of different task datasets which is the number of different task variables

    L, Z = 0, 0
    for dset in args.dataset:
        print(dset)
        config = get_config(dset, ctype='dset', to_bunch=True)
        L = max(L, config.L) #so that the net has an input dimension large enough for the task with the task with largest input dimension wise
        Z = max(Z, config.Z)
    args.L = L
    #(remove once bug from Mar 28 found)
    args.Z = Z
    print(args.L)
    # initializing logging before we actually train see Trainer(args) below 

    #notice the idea behind this code is that: on our way to training we've been adding all sorts of information to args specifying exactly what we're doing, how to do it etc and 
    #finally we pass it to Traniner and it does the training and logs the resutls 

    # do this last, because we will be logging previous parameters into the config file
    if not args.no_log:
        #i.e. if we choose to log (default is to log: we have the option to not log by specifying no_log True but by default its False)
        if args.slurm_id is not None:
            #what is  is slurm id?
            log = log_this(args, 'logs', os.path.join(args.name.split('_')[0], args.name.split('_')[1]), checkpoints=args.log_checkpoint_models)
        else:
            log = log_this(args, 'logs', args.name, checkpoints=args.log_checkpoint_models)
            

        logging.basicConfig(format='%(message)s', filename=log.run_log, level=logging.DEBUG)
        #this displays the log contents of log.run_log which is log_path in log_this definition in utils.py
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)
        args.log = log
    else:
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        logging.info('NOT LOGGING THIS RUN.')

    # logging, when loading models from paths
    if args.model_path is not None:
        logging.info(f'Using model path {args.model_path}')
        if args.model_config_path is not None:
            logging.info(f'...with config file {args.model_config_path}')
        else:
            logging.info('...but not using any config file. Errors may ensue due to net param mismatches')

    return args


if __name__ == '__main__':
    #see stack exchange "what does if __name__ == 'main__do "
    args = parse_args()
    args = adjust_args(args)

    trainer = Trainer(args)
    logging.info(f'Initialized trainer. Using device {trainer.device}, optimizer {args.optimizer}.')

    if args.optimizer == 'lbfgs':
        best_loss, n_iters = trainer.optimize_lbfgs()
    elif args.optimizer in ['sgd', 'rmsprop', 'adam']:
        best_loss, n_iters = trainer.train()

    if args.slurm_id is not None:
        # if running many jobs, then we gonna put the results into a csv
        csv_path = os.path.join('logs', args.name.split('_')[0] + '.csv')
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            labels_csv = ['slurm_id', 'N', 'D1', 'D2', 'seed', 'rseed', 'fp', 'fb', 'mnoise', 'rnoise', 'dset', 'niter', 'tparts', 'loss']
            vals_csv = [
                args.slurm_id, args.N, args.D1, args.D2, args.seed,
                args.res_seed, args.fixed_pts, args.fixed_beta, args.m_noise, args.res_noise,
                args.dataset, n_iters, '-'.join(args.train_parts), best_loss
            ]
            if args.optimizer != 'lbfgs':
                labels_csv.extend(['lr', 'epochs'])
                vals_csv.extend([args.lr, args.n_epochs])

            if not csv_exists:
                writer.writerow(labels_csv)
            writer.writerow(vals_csv)

    logging.shutdown()


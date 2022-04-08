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
import random
import csv
import math
import json
import copy
import pandas as pd

# from network import BasicNetwork, Reservoir
from network import M2Net

from utils import log_this, load_rb, get_config, update_args
from helpers import get_optimizer, get_scheduler, get_criteria, create_loaders, collater

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

        trains, tests = create_loaders(self.args.dataset, self.args, split_test=True, test_size=50)

        if self.args.sequential:
            self.train_set, self.train_loaders = trains
            self.test_set, self.test_loaders = tests
            self.train_idx = 0
            self.train_loader = self.train_loaders[self.args.train_order[self.train_idx]]
            self.test_loader = self.test_loaders[self.args.train_order[self.train_idx]]
        else:
            self.train_set, self.train_loader = trains
            self.test_set, self.test_loader = tests
        logging.info(f'Created data loaders using datasets:')
        for ds in self.args.dataset:
            logging.info(f'  {ds}')

        if self.args.sequential:
            logging.info(f'Sequential training. Starting with task {self.train_idx}')

        # self.net = BasicNetwork(self.args)
        self.net = M2Net(self.args)
        # add hopfield net patterns
        if hasattr(self.args, 'fixed_pts') and self.args.fixed_pts > 0:
            self.net.reservoir.add_fixed_points(self.args.fixed_pts)
        self.net.to(self.device)
        
        # print('resetting network')
        # self.net.reset(self.args.res_x_init, device=self.device)

        # getting number of elements of every parameter
        self.n_params = {}
        self.train_params = []
        self.not_train_params = []
        logging.info('Training the following parameters:')
        for k,v in self.net.named_parameters():
            # k is name, v is weight
            found = False
            # filtering just for the parts that will be trained
            for part in self.args.train_parts:
                if part in k:
                    logging.info(f'  {k}')
                    self.n_params[k] = (v.shape, v.numel())
                    self.train_params.append(v)
                    found = True
                    break
            if not found:
                self.not_train_params.append(k)
        logging.info('Not training:')
        for k in self.not_train_params:
            logging.info(f'  {k}')

        self.criteria = get_criteria(self.args)
        self.optimizer = get_optimizer(self.args, self.train_params)
        self.scheduler = get_scheduler(self.args, self.optimizer)
        
        self.log_interval = self.args.log_interval
        if not self.args.no_log:
            self.log = self.args.log
            self.run_id = self.args.log.run_id
            self.vis_samples = []
            self.csv_path = open(os.path.join(self.log.run_dir, f'losses_{self.run_id}.csv'), 'a')
            self.writer = csv.writer(self.csv_path, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            self.writer.writerow(['ix', 'train_loss', 'test_loss'])
            self.plot_checkpoint_path = os.path.join(self.log.run_dir, f'checkpoints_{self.run_id}.pkl')
            self.save_model_path = os.path.join(self.log.run_dir, f'model_{self.run_id}.pth')

    def log_model(self, ix=0, name=None):
        # if we want to save a particular name, just do it and leave
        if name is not None:
            model_path = os.path.join(self.log.run_dir, name)
            if os.path.exists(model_path):
                os.remove(model_path)
            torch.save(self.net.state_dict(), model_path)
            return
        # saving all checkpoints takes too much space so we just save one model at a time, unless we explicitly specify it
        if self.args.log_checkpoint_models:
            self.save_model_path = os.path.join(self.log.checkpoint_dir, f'model_{ix}.pth')
        elif os.path.exists(self.save_model_path):
            os.remove(self.save_model_path)
        torch.save(self.net.state_dict(), self.save_model_path)

    def log_checkpoint(self, ix, x, y, z, train_loss, test_loss):
        self.writer.writerow([ix, train_loss, test_loss])
        self.csv_path.flush()

        self.log_model(ix)

        # we can save individual samples at each checkpoint, that's not too bad space-wise
        if self.args.log_checkpoint_samples:
            self.vis_samples.append([ix, x, y, z, train_loss, test_loss])
            if os.path.exists(self.plot_checkpoint_path):
                os.remove(self.plot_checkpoint_path)
            with open(self.plot_checkpoint_path, 'wb') as f:
                pickle.dump(self.vis_samples, f)

    # runs an iteration where we want to match a certain trajectory
    def run_trial(self, x, y, trial, training=True, extras=False):
        self.net.reset(self.args.res_x_init, device=self.device)
        trial_loss = 0.
        k_loss = 0.
        outs = []
        us = []
        vs = []
        # setting up k for t-BPTT
        if training and self.args.k != 0:
            k = self.args.k
        else:
            # k to full n means normal BPTT
            k = x.shape[2]
        for j in range(x.shape[2]):
            net_in = x[:,:,j]
            net_out, etc = self.net(net_in, extras=True)
            outs.append(net_out)
            us.append(etc['u'])
            vs.append(etc['v'])
            # t-BPTT with parameter k
            if (j+1) % k == 0:
                # the first timestep with which to do BPTT
                k_outs = torch.stack(outs[-k:], dim=2)
                k_targets = y[:,:,j+1-k:j+1]
                for c in self.criteria:
                    k_loss += c(k_outs, k_targets, i=trial, t_ix=j+1-k)
                trial_loss += k_loss.detach().item()
                if training:
                    k_loss.backward()
                    # strategies for continual learning that involve modifying gradients
                    if self.args.sequential and self.train_idx > 0:
                        if self.args.owm:
                            # orthogonal weight modification
                            self.net.M_u.weight.grad = self.P_u @ self.net.M_u.weight.grad @ self.P_s
                            self.net.M_ro.weight.grad = self.P_z @ self.net.M_ro.weight.grad @ self.P_v
                            if self.args.ff_bias:
                                self.net.M_u.bias.grad = self.P_u @ self.net.M_u.bias.grad
                                self.net.M_ro.bias.grad = self.P_z @ self.net.M_ro.bias.grad
                        elif self.args.swt:
                            # keeping sensory and output weights constant after learning first task
                            self.net.M_u.weight.grad[:,:self.args.L] = 0
                            self.net.M_ro.weight.grad[:] = 0
                            if self.args.ff_bias:
                                self.net.M_u.bias.grad[:] = 0
                                self.net.M_ro.bias.grad[:] = 0
                            
                k_loss = 0.
                self.net.reservoir.x = self.net.reservoir.x.detach()

        trial_loss /= x.shape[0]

        if extras:
            net_us = torch.stack(us, dim=2)
            net_vs = torch.stack(vs, dim=2)
            net_outs = torch.stack(outs, dim=2)
            etc = {
                'outs': net_outs,
                'us': net_us,
                'vs': net_vs
            }
            return trial_loss, etc
        return trial_loss

    def train_iteration(self, x, y, trial, ix_callback=None):
        self.optimizer.zero_grad()
        trial_loss, etc = self.run_trial(x, y, trial, extras=True)

        if ix_callback is not None:
            ix_callback(trial_loss, etc)
        self.optimizer.step()

        etc = {
            'ins': x,
            'goals': y,
            'us': etc['us'].detach(),
            'vs': etc['vs'].detach(),
            'outs': etc['outs'].detach()
        }
        return trial_loss, etc

    def test(self):
        with torch.no_grad():
            x, y, trials = next(iter(self.test_loader))
            x, y = x.to(self.device), y.to(self.device)
            loss, etc = self.run_trial(x, y, trials, training=False, extras=True)

        etc = {
            'ins': x,
            'goals': y,
            'us': etc['us'].detach(),
            'vs': etc['vs'].detach(),
            'outs': etc['outs'].detach()
        }

        return loss, etc

    # helper function for sequential training, for testing performance on all tasks
    def test_tasks(self, ids):
        losses = []
        for i in ids:
            self.test_loader = self.test_loaders[self.args.train_order[i]]
            loss, _ = self.test()
            losses.append((i, loss))
        self.test_loader = self.test_loaders[self.train_idx]
        return losses

    def update_P(self, S, states):
        S_new = torch.einsum('ijk,ilk->jl',states,states) / states.shape[0] / states.shape[2]
        S_avg = (S * self.train_idx + S_new) / (self.train_idx + 1)
        alpha = 1e-3
        P = torch.inverse(S_avg / alpha + torch.eye(S_avg.shape[0]))
        return P, S_avg

    def train(self, ix_callback=None):
        ix = 0
        # for convergence testing
        running_min_error = float('inf')
        running_no_min = 0

        running_loss = 0.0
        ending = False

        # for OWM
        if self.args.owm:
            S_s = 0
            S_u = 0
            S_v = 0
            S_z = 0

        for e in range(self.args.n_epochs):
            for epoch_idx, (x, y, info) in enumerate(self.train_loader):
                ix += 1

                x, y = x.to(self.device), y.to(self.device)
                iter_loss, etc = self.train_iteration(x, y, info, ix_callback=ix_callback)

                if iter_loss == -1:
                    logging.info(f'iteration {ix}: is nan. ending')
                    ending = True
                    break

                running_loss += iter_loss

                if ix % self.log_interval == 0:
                    z = etc['outs'].cpu().numpy().squeeze()
                    train_loss = running_loss / self.log_interval
                    test_loss, test_etc = self.test()
                    log_arr = [
                        f'*{ix}',
                        f'train {train_loss:.3f}',
                        f'test {test_loss:.3f}'
                    ]
                    if self.args.sequential:
                        losses = self.test_tasks(ids=range(self.train_idx))
                        for i, loss in losses:
                            log_arr.append(f't{i}: {loss:.3f}')
                    log_str = '\t| '.join(log_arr)
                    logging.info(log_str)

                    if not self.args.no_log:
                        self.log_checkpoint(ix, etc['ins'].cpu().numpy(), etc['goals'].cpu().numpy(), z, train_loss, test_loss)
                    running_loss = 0.0

                    # if training sequentially, move on to the next task
                    # if doing OWM-like updates, do them here
                    if self.args.sequential and test_loss < self.args.seq_threshold:
                        logging.info(f'Successfully trained task {self.train_idx}...')
                        
                        losses = self.test_tasks(ids=range(self.train_idx + 1))
                        for i, loss in losses:
                            logging.info(f'...loss on task {i}: {loss:.3f}')

                        # orthogonal weight modification of M_u and M_ro
                        if self.args.owm:
                            # 0th dimension is test batch size, 2nd dimension is number of timesteps
                            # 1st dimension is the actual vector representation
                            self.P_s, S_s = self.calc_P(S_s, test_etc['ins'])
                            self.P_u, S_u = self.update_P(S_u, test_etc['us'])
                            self.P_v, S_v = self.update_P(S_v, test_etc['vs'])
                            self.P_z, S_z = self.update_P(S_z, test_etc['outs'])
                            logging.info(f'...updated projection matrices for OWM')

                        # done processing prior task, move on to the next one or quit
                        self.train_idx += 1
                        if self.train_idx == len(self.args.train_order):
                            ending = True
                            logging.info(f'...done training all tasks! ending')
                            break
                        logging.info(f'...moving on to task {self.train_idx}.')
                        self.train_loader = self.train_loaders[self.args.train_order[self.train_idx]]
                        self.test_loader = self.test_loaders[self.args.train_order[self.train_idx]]
                        running_min_error = float('inf')
                        running_no_min = 0
                        break

                    # convergence based on no avg loss decrease after patience samples
                    if test_loss < running_min_error:
                        running_no_min = 0
                        running_min_error = test_loss
                        if not self.args.no_log:
                            self.log_model(name='model_best.pth')
                    else:
                        running_no_min += self.log_interval
                    if running_no_min > self.args.patience:
                        logging.info(f'iteration {ix}: no min for {self.args.patience} samples. ending')
                        ending = True
                if ending:
                    break
            logging.info(f'Finished dataset epoch {e+1}')
            if self.scheduler is not None:
                self.scheduler.step()
            if ending:
                break

        if not self.args.no_log and self.args.log_checkpoint_samples:
            # for later visualization of outputs over timesteps
            with open(self.plot_checkpoint_path, 'wb') as f:
                pickle.dump(self.vis_samples, f)

            self.csv_path.close()

        logging.info(f'END | iterations: {(ix // self.log_interval) * self.log_interval} | best loss: {running_min_error}')
        return running_min_error, ix


    def optimize_lbfgs(self):
        xs, ys, trials = collater(self.train_set[:1000])
        xs, ys = xs.to(self.device), ys.to(self.device)

        # xs_test, ys_test, trials_test = collater(self.test_set)
        # so that the callback for scipy.optimize.minimize knows what step it is on
        self.scipy_ix = 0
        vis_samples = []

        # this is what happens every iteration
        # run through all examples (x, y) and get loss, gradient
        def closure(v):
            # setting the parameters in the network with the new values in v
            ind = 0
            for k,nums in self.n_params.items():
                # nums[0] is shape, nums[1] is number of elements
                weight = v[ind:ind+nums[1]].reshape(nums[0])
                self.net.state_dict()[k][:] = torch.Tensor(weight)
                ind += nums[1]

            # res state starting from same random seed for each iteration
            self.net.reset()
            self.net.zero_grad()

            # total_loss = torch.tensor(0.)
            total_loss = self.run_trial(xs, ys, trials, extras=False)
            # total_loss.backward()

            # turn param grads into list
            grad_list = []
            for v in self.train_params:
                grad = v.grad.clone().numpy().reshape(-1)
                grad_list.append(grad)
            vec = np.concatenate(grad_list)
            post = np.float64(vec)

            return total_loss, post

        # callback just does logging
        def callback(xk):
            if self.args.no_log:
                return
            self.scipy_ix += 1
            if self.scipy_ix % self.log_interval == 0:
                sample_n = random.randrange(1000)

                with torch.no_grad():
                    self.net.reset()
                    self.net.zero_grad()
                    # outs = []
                    # total_loss = torch.tensor(0.)

                    # pdb.set_trace()

                    loss, etc = self.test()

                    # x = xs[sample_n,:].reshape(1,1,-1)
                    # y = ys[sample_n,:].reshape(1,1,-1)

                    # trial_loss, etc = self.run_trial(xs_test, ys_test, trials_test, training=False, extras=True)
                    # pdb.set_trace()
                    # for j in range(xs.shape[0]):
                    #     net_in = x[j]
                    #     net_out, etc = self.net(net_in, extras=True)
                    #     outs.append(net_out)
                    #     net_out, step_loss, _ = self.run_iteration(xs[j], ys[j])
                    #     outs.append(net_out.item())
                    #     total_loss += step_loss
                    # z = etc['outs']

                    # z = np.stack(outs).squeeze()
                    self.log_checkpoint(self.scipy_ix, etc['ins'].numpy(), etc['goals'].numpy(), etc['outs'].numpy(), loss, loss)

                    # self.log_checkpoint(self.scipy_ix, xs_test.numpy(), ys_test.numpy(), etc['outs'], total_loss.item(), total_loss.item())

                    logging.info(f'iteration {self.scipy_ix}\t| loss {loss:.3f}')

        # getting the initial values to put into the algorithm
        init_list = []
        for v in self.train_params:
            init_list.append(v.detach().clone().numpy().reshape(-1))
        init = np.concatenate(init_list)

        optim_options = {
            'iprint': self.log_interval,
            'maxiter': self.args.maxiter,
            # 'ftol': 1e-16
        }
        optim = minimize(closure, init, method='L-BFGS-B', jac=True, callback=callback, options=optim_options)

        error_final = optim.fun
        n_iters = optim.nit

        if not self.args.no_log:
            self.log_model(name='model_final.pth')
            if self.args.log_checkpoint_samples:
                with open(self.plot_checkpoint_path, 'wb') as f:
                    pickle.dump(self.vis_samples, f)
            self.csv_path.close()

        return error_final, n_iters

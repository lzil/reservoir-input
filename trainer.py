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
from helpers import get_optimizer, get_scheduler, get_criteria, create_loaders, collater, deriv_tanh

# for PCA variances:
from testers import get_states
from pca import pca_multimodal

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
            
            #loader for fisher information estimation diagonals after training on each task - batch size is fixed at 256 or natural continual learning 
            if self.args.ewc or self.args.ncl:
                # make neater combine into one:
                if self.args.ewc:
                    _2, tests_2 = create_loaders(self.args.dataset, self.args, split_test=True, test_size =256)
                    _2, self.ewc_loaders =tests_2
                    self.ewc_loader = self.ewc_loaders[self.args.train_order[self.train_idx]]
                else:
                    _2, tests_2 = create_loaders(self.args.dataset, self.args, split_test=True, test_size =2048)
                    _2, self.ncl_loaders =tests_2
                    self.ncl_loader = self.ncl_loaders[self.args.train_order[self.train_idx]]

        
        elif self.args.multimodal:
            self.train_set, self.train_loader = trains
            #note trains is as usual, self.test_loaders is a dictionary with keys given by t_types whose values are test_loaders containining examples of t_type trials
            self.test_set, self.test_loaders = tests
            
            #ad hoc code for testing RSG tasks when doing multimodal with variant of RSG
            #tidy later but for now
            # if self.test_set.t_types == ['rsg']:
            #     rsg_trains, rsg_tests= create_loaders(self.args.dataset, self.args, split_test=True, test_size=50, multimodal_rsg=True)
            #     _ , self.rsg_test_loader = rsg_tests
        
        elif self.args.one_mod:
            self.train_set, self.train_loader = trains
            self.test_set, self.og_test_loader = tests

            self.test_loader = self.og_test_loader
      


        else:
            self.train_set, self.train_loader = trains
            self.test_set, self.test_loader = tests
        logging.info(f'Created data loaders using datasets:')
        for ds in self.args.dataset:
            logging.info(f'  {ds}')

        if self.args.sequential:
            logging.info(f'Sequential training. Starting with task {self.train_idx}')

            
            if self.args.xdg:
                logging.info(f'Implementing context-dependent gating. Gating {self.args.X}% of units in layers:')
                for layer in self.args.gate_layers:
                    logging.info(f'  {layer}')
            
            
            if self.args.synaptic_intel or self.args.ewc:
                if self.args.synaptic_intel:
                    stab_method = 'synaptic intelligence'
                elif self.args.ewc:
                    stab_method = 'elastic weight consolidation'
                logging.info(f'Stabilizing trainable parameters with {stab_method}. Hyperparameter values: '.format(stab_method))
                logging.info(f'  Stabilization strength: {self.args.stab_strength}')
                if self.args.synaptic_intel:
                    logging.info((f'  Damping term : {self.args.damp_term}'))

            if self.args.ncl:
                logging.info(f'Implementing natural continual learning with hyperparameter arguments:')
                logging.info(f'')
        
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
        self.train_param_names=[] # list of names of params that we're training e.g. ['M_u.weight', 'M_ro.weight', 'reservoir.J.weight']
        
        logging.info('Training the following parameters:')
        for k,v in self.net.named_parameters():
            # k is name, v is weight
            found = False
            # filtering just for the parprint(self.args.train_parts)
            for part in self.args.train_parts:
                if part in k:
                    logging.info(f'  {k}')
                    self.train_param_names.append(k)
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
            if self.args.sequential or self.args.one_mod or self.args.multimodal:
                task_names = ['t{}'.format(i) for i in range(len(self.args.dataset))]
                self.writer.writerow(['ix', 'train_loss', 'test_loss']+task_names)
            else:
                self.writer.writerow(['ix', 'train_loss', 'test_loss'])
            self.plot_checkpoint_path = os.path.join(self.log.run_dir, f'checkpoints_{self.run_id}.pkl')
            self.save_model_path = os.path.join(self.log.run_dir, f'model_{self.run_id}.pth')
        
        if self.args.xdg:
            self.xdg_gates_path = os.path.join(self.log.run_dir, f'xdg_gates_{self.run_id}.pkl')
            with open(self.xdg_gates_path, 'wb') as f:
                pickle.dump(self.net.task_gates_by_contexts, f)


            
        
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

    def log_checkpoint(self, ix, x, y, z, train_loss, test_loss, seq_losses_all_tasks=None):
        if seq_losses_all_tasks is not None:
            seq_losses = []
            for i in range(len(seq_losses_all_tasks)):
                seq_losses.append(seq_losses_all_tasks[i][1])
            self.writer.writerow([ix, train_loss, test_loss]+seq_losses)
        else:
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

    def run_trial(self, x, y, trial, training=True, extras=False, ewc_fish_estim=False, ncl_fish_estim=False):
        self.net.reset(self.args.res_x_init, device=self.device)
        trial_loss = 0.
        k_loss = 0.
        outs = []
        us = []
        if not training:
            pre_act_xs=[]
        xs = []
        vs = []
        if ncl_fish_estim: 
            # collect vectors for covariance/FIM estimation
            grad_wrt_us = []
            grad_wrt_xs = []
            grad_wrt_zs = []
        # setting up k for t-BPTT
        if training and self.args.k != 0:
            k = self.args.k
        else:
            # k to full n means normal BPTT
            k = x.shape[2]

        if self.args.synaptic_intel and training:
            grads_m_u = 0
            grads_w_u =0
            grads_j = 0
            grads_m_ro = 0
        
        if self.args.rflo:
            tau_reciprocal = torch.tensor(1/self.net.reservoir.tau_x)
            phi_prime = deriv_tanh
            if self.args.train_parts != ['']:
                W_u = self.net.reservoir.W_u.weight.detach()
                
                delta_M_u_time_series = [] # note they've already been scaled by the learning rate
                delta_M_ro_time_series = []
                if self.args.batch_size ==1:

                    self.m_abqjs= torch.zeros((self.args.D1, self.args.L+self.args.T, self.args.N, x.shape[2]))
                    m_abqj_prev = torch.zeros((self.args.D1, self.args.L+self.args.T, self.args.N, x.shape[2]))

                else:
                    self.m_abqjs= torch.zeros((self.args.batch_size, self.args.D1, self.args.L+self.args.T, self.args.N, x.shape[2]))
                    m_abqj_prev = torch.zeros((self.args.batch_size,self.args.D1, self.args.L+self.args.T, self.args.N, x.shape[2]))

            


        for j in range(x.shape[2]):
            net_in = x[:,:,j]
            net_out, etc = self.net(net_in, extras=True,ncl_fish_estim= ncl_fish_estim)
            
            outs.append(net_out)
            
            us.append(etc['u'])
            if not training:
                
                pre_act_xs.append(etc['pre_act_x'])
                
            xs.append(etc['x'])
            vs.append(etc['v'])
            
            
        
            if self.args.rflo and training:


                if self.args.batch_size == 1:
                    bptt = False
                    with torch.no_grad():
                        eps_t = y[:,:,j] - net_out
                    
                    # collect the things you need to compute v_js
                    s_t = net_in 

                    if self.args.train_parts != ['']:
                        with torch.no_grad():
                            u_t = self.net.M_u(s_t)
                    x_t_minus_1 = xs[j-1]
                    v_rflo = self.net.reservoir.J(x_t_minus_1) + self.net.reservoir.W_u(u_t) 
                    # v_rflo is [1,300] at this point

                    
                    # start computing m_abq  for this time step t (j in this code)
                    if self.args.train_parts != ['']:
                        if j == 0: 
                            m_abqj_prev = torch.zeros((self.args.D1, self.args.L+self.args.T, self.args.N))
                        else:
                            m_abqj_prev = self.m_abqjs[:,:,:, j-1]
                        
                        #vectorise (reshape and manipulate things to avoid nested for loops):
                       
                        v_rflo = v_rflo.unsqueeze(2)
                        v_rflo = v_rflo.expand(self.args.D1,self.args.N,self.args.L+self.args.T).transpose(-1,-2)
                        # s_t 
                        # now according to their rule we need s_t at the previous timestep 
                        s_t = x[:, :, j-1]
                        s_t_expanded = s_t.repeat(self.args.D1,1)
                        s_t_expanded = s_t_expanded.unsqueeze(2)
                        s_t_expanded = s_t_expanded.repeat(1,1, self.args.N)
                        # W_u 
                        W_u_clone = self.net.reservoir.W_u.weight.detach().clone()
                        W_u_expanded = W_u_clone.T.unsqueeze(1)
                        W_u_expanded =  W_u_expanded.repeat_interleave(self.args.L+self.args.T, dim=1)

                    
                        
                        self.m_abqjs[:,:,:,j] = tau_reciprocal * phi_prime(v_rflo) * W_u_expanded * s_t_expanded + (1-tau_reciprocal)*m_abqj_prev


                        # for a in range(self.args.D1):
                        #     for b in range(self.args.L):
                        #         for q in range(self.args.N):
                        #             if j  == 0 :
                        #                 m_abqj_prev = 0
                        #             else:
                        #                 m_abqj_prev = self.m_abqjs[a,b,q,j-1]
                        #             # compute m_ab^j(t) - which in code is m_ab^q(j):
                        #             self.m_abqjs[a,b,q,j] = tau_reciprocal*phi_prime(v_rflo[:,q]) * self.net.reservoir.W_u.weight[q,a].detach()*s_t[:,b] + (1 - tau_reciprocal) * m_abqj_prev 
                        
                        # compute delta_M_u(t)
                        #instantiate and populate:
                        with torch.no_grad():
                            delta_M_u_t = torch.zeros((self.args.D1, self.args.L+self.args.T))
                            for a in range(self.args.D1):
                                for b in range(self.args.L+self.args.T):
                                    
                                    delta_M_u_t[a,b] = self.args.M_u_rflo_lr * torch.sum( (self.net.B @ eps_t[0]) * self.m_abqjs[a,b,:,j]) #note take first and (for now) only batch of errors
                            delta_M_u_time_series.append(delta_M_u_t)
                            
                            # compute M_ro 
                            
                            delta_M_ro_t = self.args.M_ro_rflo_lr * eps_t[0].unsqueeze(1) @ xs[j][0].unsqueeze(1).T #turn eps_t[0] from tensor with shape [3] to shape [3,1] 
                            delta_M_ro_time_series.append(delta_M_ro_t)
                            
                else:
                    bptt = False
                    with torch.no_grad():
                        eps_t = y[:,:,j] - net_out
                    
                    # collect the things you need to compute v_js
                    s_t = net_in 

                    if self.args.train_parts != ['']:
                        with torch.no_grad():
                            u_t = self.net.M_u(s_t)
                    x_t_minus_1 = xs[j-1]
                    v_rflo = self.net.reservoir.J(x_t_minus_1) + self.net.reservoir.W_u(u_t)

                    
                    # start computing m_abq  for this time step t (j in this code)
                    if self.args.train_parts != ['']:
                        if j == 0: 
                            m_abqj_prev = torch.zeros((self.args.batch_size, self.args.D1, self.args.L+self.args.T, self.args.N))
                        else:
                            m_abqj_prev = self.m_abqjs[:,:,:,:, j-1]
                        
                        #vectorise (reshape and manipulate things to avoid nested for loops):
                        
                        v_rflo = v_rflo.unsqueeze(2)  # [1,300,1]
                        v_rflo = v_rflo.unsqueeze(1).transpose(-1,-2)
                        v_rflo = v_rflo.repeat(1,self.args.D1,self.args.L+self.args.T,1)
                       
                        # s_t                       [50,4,300]
                        # now according to their rule we need s_t at the previous timestep 
                        s_t = x[:, :, j-1]
                        s_t_expanded = s_t.unsqueeze(1).repeat(1,self.args.D1,1)
                        
                        s_t_expanded = s_t_expanded.unsqueeze(-1).repeat(1,1,1, self.args.N)
                        
                        
                        # W_u 
                        W_u_clone = self.net.reservoir.W_u.weight.detach().clone()
                        W_u_expanded = W_u_clone.T.unsqueeze(1)
                        W_u_expanded = W_u_expanded.unsqueeze(0)
                        W_u_expanded = W_u_expanded.repeat(self.args.batch_size,1,1,1)
                        W_u_expanded =  W_u_expanded.repeat_interleave(self.args.L+self.args.T, dim=2)
                        
                    
                        
                        self.m_abqjs[:,:,:,:,j] = tau_reciprocal * phi_prime(v_rflo) * W_u_expanded * s_t_expanded + (1-tau_reciprocal)*m_abqj_prev

                        
                        # for a in range(self.args.D1):
                        #     for b in range(self.args.L):
                        #         for q in range(self.args.N):
                        #             if j  == 0 :
                        #                 m_abqj_prev = 0
                        #             else:
                        #                 m_abqj_prev = self.m_abqjs[a,b,q,j-1]
                        #             # compute m_ab^j(t) - which in code is m_ab^q(j):
                        #             self.m_abqjs[a,b,q,j] = tau_reciprocal*phi_prime(v_rflo[:,q]) * self.net.reservoir.W_u.weight[q,a].detach()*s_t[:,b] + (1 - tau_reciprocal) * m_abqj_prev 
                        
                        # compute delta_M_u(t)
                        #instantiate and populate:
                        with torch.no_grad():
                            delta_M_u_t = torch.zeros((self.args.batch_size,self.args.D1, self.args.L+self.args.T))
                            
                            
                            batch_B = self.net.B.unsqueeze(0)
                            batch_B = batch_B.repeat(self.args.batch_size,1,1)
                            
                            eps_t = eps_t.unsqueeze(-1)
                            B_mul_eps_t = torch.squeeze(torch.bmm(batch_B,eps_t),-1)
                            
                            for a in range(self.args.D1):
                                for b in range(self.args.L+self.args.T):
                                    
                                    delta_M_u_t[:, a,b] = self.args.M_u_rflo_lr * torch.sum(B_mul_eps_t * self.m_abqjs[:,a,b,:,j]) #note take first and (for now) only batch of errors
                            
                            
                            delta_M_u_time_series.append(delta_M_u_t)
                            
                            # compute M_ro 
                            xs_transpose_batched = xs[j].unsqueeze(-1).transpose(-1,-2)
                            eps_mul_by_xs_transpose = torch.bmm(eps_t,xs_transpose_batched).squeeze(-1)
                            delta_M_ro_t = self.args.M_ro_rflo_lr * eps_mul_by_xs_transpose
                            delta_M_ro_time_series.append(delta_M_ro_t)
                            
            
            # t-BPTT with parameter k
            if (j+1) % k == 0:
                # the first timestep with which to do BPTT
                k_outs = torch.stack(outs[-k:], dim=2)
                k_targets = y[:,:,j+1-k:j+1]
                for c in self.criteria:
                    k_loss += c(k_outs, k_targets, i=trial, t_ix=j+1-k)

                        
                trial_loss += k_loss.detach().item()
                if training and not self.args.rflo:
                    if self.args.synaptic_intel:
                        #this code should run even when training first task as it collects gradient info needed to compute the omegas for the penalties(i.e. don't use additional condition self.train_idx>0)
                        #calculate gradients for loss on current task without SI penalty; save them


                        k_loss.backward()
                        
                        #to do: have code work with biases
                        grads_m_u += self.net.M_u.weight.grad.detach().clone()
                        grads_w_u += self.net.reservoir.W_u.weight.grad.detach().clone()
                        grads_j += self.net.reservoir.J.weight.grad.detach().clone()
                        grads_m_ro += self.net.M_ro.weight.grad.detach().clone()
                        
                        
                        
                        
                        penalty_m_u = torch.sum(self.omega_m_u*(self.net.M_u.weight - self.m_u_prev)**2)
                        # if W_u exists and is trainable then need to stabilize it 
                        penalty_w_u = torch.sum(self.omega_w_u*(self.net.reservoir.W_u.weight - self.w_u_prev)**2)
                        penalty_j= torch.sum(self.omega_j*(self.net.reservoir.J.weight - self.j_prev)**2)
                        
                        penalty_m_ro = torch.sum(self.omega_m_ro * (self.net.M_ro.weight - self.m_ro_prev)**2)
                        # i think we need to times by k so as to add SI for loss at each timestep 

                        # if W_u exists and is trainable then need to stabilize it 
                        if self.args.D1 != 0 and self.args.train_parts == ['']:
                            synaptic_intel_penalty = self.args.stab_strength * (penalty_m_u + penalty_w_u+ penalty_j + penalty_m_ro)
                        else:
                             synaptic_intel_penalty = self.args.stab_strength * (penalty_m_u + penalty_j + penalty_m_ro)
                        
                        
                        synaptic_intel_penalty.backward()

                    elif self.args.ewc  and self.train_idx > 0 and ewc_fish_estim == False:
                        k_loss.backward()
                        penalty_m_u = torch.sum(self.omega_m_u*(self.net.M_u.weight - self.m_u_prev)**2)
                        # if W_u exists and is trainable then we need to stabilize it
                        if self.args.D1 > 0:
                            penalty_w_u = torch.sum(self.omega_w_u*(self.net.reservoir.W_u.weight - self.w_u_prev)**2)
                        penalty_j = torch.sum(self.omega_j*(self.net.reservoir.J.weight - self.j_prev)**2)
                        penalty_m_ro = torch.sum(self.omega_m_ro * (self.net.M_ro.weight - self.m_ro_prev)**2)

                        if self.args.D1 > 0 and self.args.train_parts == ['']:
                            
                            ewc_penalty = self.args.stab_strength * (penalty_m_u + penalty_w_u+ penalty_j + penalty_m_ro)
                        else:
                            ewc_penalty = self.args.stab_strength * (penalty_m_u + penalty_j + penalty_m_ro)
                        ewc_penalty.backward()
                    else:
                        k_loss.backward()
                        if 'sp-bce' in self.args.loss:
                            #clip gradients
                            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)

                    if ncl_fish_estim:
                        if self.args.D1 != 0:
                            grad_wrt_us.append(etc['grad_l_wrt_u'])
                        if self.args.train_parts =='all':
                            grad_wrt_xs.append(etc['grad_l_wrt_x'])
                        #gradient of output wrt the loss
                        
                        grad_wrt_zs.append(net_out.grad.detach().clone())

                k_loss = 0.
                self.net.reservoir.x = self.net.reservoir.x.detach()
        if self.args.owm  and self.train_idx > 0 and training and not self.args.rflo:
            if self.args.owm:
                if self.args.train_parts == [''] and self.args.D1 == 0 and self.args.D2 == 0:
                
                    grad_j= self.net.reservoir.J.weight.grad.detach().clone()
                    grad_m_u = self.net.M_u.weight.grad.detach().clone()
                    grad_w = torch.cat((grad_j, grad_m_u), dim=1)

                    #project input and recurrent weights
                    delta_w = self.P_wq @ grad_w @ self.P_q
                    
                    self.net.M_u.weight.grad = delta_w[:, -self.net.M_u.weight.shape[1]: ]
                    self.net.M_u.weight.grad = self.net.M_u.weight.grad.contiguous() # assigning slices to a grad seems to cause a performance issue, contiguous() fixes it in this case
                    
                    self.net.reservoir.J.weight.grad =  delta_w[:, :self.net.reservoir.J.weight.shape[0]]
                    self.net.reservoir.J.weight.grad = self.net.reservoir.J.weight.grad.contiguous()
                    
                    self.net.M_ro.weight.grad = self.P_z @ self.net.M_ro.weight.grad @ self.P_x
                
                elif self.args.train_parts == ['']  and self.args.D1 > 0 and self.args.D2 ==0:
                     
                     self.net.M_u.weight.grad = self.P_u @ self.net.M_u.weight.grad @ self.net.P_s
                     
                     
                     grad_j= self.net.reservoir.J.weight.grad.detach().clone()
                     grad_w_u = self.net.reservoir.W_u.weight.grad.detach().clone()
                     grad_w =  torch.cat((grad_j,grad_w_u), dim = 1)

                     delta_w = self.P_wq @ grad_w @ self.P_q
                     self.net.W_u.weight.grad = delta_w[:, -self.net.W_u.weight.shape[1]: ]
                     self.net.W_u.weight.grad = self.net.W_u.weight.grad.contiguous()
                     self.net.reservoir.J.weight.grad =  delta_w[:, :self.net.reservoir.J.weight.shape[0]]
                     self.net.reservoir.J.weight.grad = self.net.reservoir.J.weight.grad.contiguous()


                     self.net.M_ro.weight.grad = self.P_z @ self.net.M_ro.weight.grad @ self.P_x
                     
                
                else:
                    self.net.M_u.weight.grad =  self.P_u @ self.net.M_u.weight.grad @ self.P_s
                    self.net.M_ro.weight.grad = self.P_z @ self.net.M_ro.weight.grad @ self.P_x

       
                 
        if training and self.args.rflo:
            if self.args.batch_size > 1:

                self.net.M_u.weight.grad = - 1* torch.sum(sum(delta_M_u_time_series),dim=0)/self.args.batch_size 
                self.net.M_ro.weight.grad = -1* torch.sum(sum(delta_M_ro_time_series), dim=0)/self.args.batch_size 
            else:
                self.net.M_u.weight.grad = - 1* sum(delta_M_u_time_series)/self.args.batch_size 
                self.net.M_ro.weight.grad = -1* sum(delta_M_ro_time_series)/self.args.batch_size 

                
                            
                

        trial_loss /= x.shape[0]
        if self.args.synaptic_intel and training:
            self.grads_list['M_u'].append(grads_m_u)
            self.grads_list['W_u'].append(grads_w_u)
            self.grads_list['J'].append(grads_j)
            self.grads_list['M_ro'].append(grads_m_ro)


        if extras:
            
            net_us = torch.stack(us, dim=2)
            
            net_pre_act_xs = torch.stack(xs,dim=2)
            net_xs = torch.stack(xs, dim =2)
            net_vs = torch.stack(vs, dim=2)
            net_outs = torch.stack(outs, dim=2)

            if ncl_fish_estim:
                if self.args.D1 != 0:
                   grad_wrt_us = torch.stack(grad_wrt_us, dim=2)
                if self.args.train_parts == ['']:
                    grad_wrt_xs = torch.stack(grad_wrt_xs, dim=2)

                grad_wrt_zs = torch.stack(grad_wrt_zs, dim=2)
            

            if not training:
                etc = {
                    'outs': net_outs,
                    'us': net_us,
                    'pre_act_xs': net_pre_act_xs,
                    'xs': net_xs,
                    'vs': net_vs
                }
                return trial_loss, etc
            else:
                if not ncl_fish_estim:
                    etc = {
                        'outs': net_outs,
                        'us': net_us,
                        'xs': net_xs,
                        'vs': net_vs

                    }
                else:
                    etc = {
                        'outs': net_outs,
                        'us': net_us,
                        'xs': net_xs,
                        'pre_act_xs': net_pre_act_xs,
                        'vs': net_vs,
                        'grad_wrt_zs': grad_wrt_zs,
                    }
                    if self.args.D1 !=0:
                        etc['grad_wrt_u'] = grad_wrt_us,
                    if self.args.train_parts == ['']:
                        etc['grad_wrt_xs'] = grad_wrt_xs

                return trial_loss, etc
        
        return trial_loss

  

    def train_iteration(self, x, y, trial, ix_callback=None):
        self.optimizer.zero_grad()
        
        if self.args.synaptic_intel:
            
            m_u_before_step = self.net.M_u.weight.detach().clone()
            w_u_before_step = self.net.reservoir.W_u.weight.detach().clone()
            j_before_step =  self.net.reservoir.J.weight.detach().clone()
            m_ro_before_step = self.net.M_ro.weight.detach().clone()
            
            trial_loss, etc = self.run_trial(x, y, trial, extras=True)
            
            if ix_callback is not None:
                ix_callback(trial_loss, etc)
            self.optimizer.step()

            m_u_change= self.net.M_u.weight.detach().clone() - m_u_before_step
            w_u_change = self.net.reservoir.W_u.weight.detach().clone() - w_u_before_step
            j_change=  self.net.reservoir.J.weight.detach().clone() -j_before_step
            m_ro_change = self.net.M_ro.weight.detach().clone() -m_ro_before_step
            
            
            self.weight_changes_list['M_u'].append(m_u_change)
            self.weight_changes_list['W_u'].append(w_u_change)
            self.weight_changes_list['J'].append(j_change)
            self.weight_changes_list['M_ro'].append(m_ro_change)

            etc = {
                'ins': x,
                'goals': y,
                'us': etc['us'].detach(),
                'vs': etc['vs'].detach(),
                'outs': etc['outs'].detach()
            }
            return trial_loss, etc
        
        
        else:
        
            trial_loss, etc = self.run_trial(x, y, trial, extras=True)

            if ix_callback is not None:
                ix_callback(trial_loss, etc)
            if not self.args.ncl:
                self.optimizer.step()
            else:
                #build momentum and update weights 
                
                self.M_m_ro = self.args.ncl_rho*self.M_m_ro + self.net.M_ro.weight.grad + self.G_m_ro @ (self.net.M_ro.weight - self.m_ro_prev) @ self.A_m_ro
                
                self.net.M_ro.weight.data -= self.args.ncl_lr * (self.p_scalar**2)*self.P_L_m_ro @ self.M_m_ro @ self.P_R_m_ro
                
                


                if self.args.D1 !=0:
                    self.M_m_u = self.args.ncl_rho*self.M_m_u + self.net.M_u.weight.grad + self.G_m_u @ (self.net.M_u.weight - self.m_u_prev) @ self.A_m_u
                    self.net.M_u.weight = nn.Parameter(self.net.M_u.weight.detach().clone() - self.args.ncl_lr * (self.p_scalar**2)*self.P_L_m_u @ self.M_m_u @ self.P_R_m_u)

                    if self.args.train_parts == ['']:
                        
                        grad_j= self.net.reservoir.J.weight.grad.detach().clone()
                        grad_w_u = self.net.reservoir.W_u.weight.grad.detach().clone()
                        grad_w =  torch.cat((grad_j,grad_w_u), dim = 1)

                        w = torch.cat((self.net.reservoir.J.weight,self.net.reservoir.W_u.weight), dim =1 )
                        w_prev = torch.cat((self.j_prev,self.w_u_prev), dim =1)
                        
                        self.M_w = self.args.ncl_rho*self.M_w + grad_w + self.G_w @ (w - w_prev) @ self.A_w
                        delta_w = self.args.ncl_lr*self.p_scalar*self.P_L_w @ self.M_w @ self.P_R_w
                        
                        self.net.W_u.weight=  nn.Parameter(self.net.W_u.weight.detach().clone() - delta_w[:, -self.net.W_u.weight.shape[1]: ])
                        self.net.W_u.weight = self.net.W_u.weight.contiguous()
                        
                        self.net.reservoir.J.weight =  nn.Parameter(self.net.reservoir.J.weight.detach().clone() - delta_w[:, :self.net.reservoir.J.weight.shape[0]])
                        self.net.reservoir.J.weight= self.net.reservoir.J.weight.contiguous()


                else:
                    grad_j= self.net.reservoir.J.weight.grad.detach().clone()
                    grad_m_u = self.net.M_u.weight.grad.detach().clone()
                    grad_w =  torch.cat((grad_j,grad_m_u), dim = 1)
                    w =  torch.cat((self.net.reservoir.J.weight,self.net.M_u.weight), dim =1 )
                    w_prev = torch.cat((self.j_prev,self.m_u_prev), dim =1)

                    self.M_w = self.args.ncl_rho*self.M_w + grad_w + self.G_w @ (w - w_prev) @ self.A_w
                    delta_w = self.args.ncl_lr*(self.p_scalar**2)*self.P_L_w @ self.M_w @ self.P_R_w
                    
                    self.net.M_u.weight = nn.Parameter(self.net.M_u.weight.detach().clone() - delta_w[:, -self.net.M_u.weight.shape[1]: ])
                    self.net.M_u.weight = self.net.M_u.weight.contiguous() # assigning slices to a grad seems to cause a performance issue, contiguous() fixes it in this case
                    
                    self.net.reservoir.J.weight = nn.Parameter(self.net.reservoir.J.weight.detach().clone() - delta_w[:, :self.net.reservoir.J.weight.shape[0]])
                    self.net.reservoir.J.weight = self.net.reservoir.J.weight.contiguous()

            
            
            etc = {'ins': x,
                    'goals': y,
                    'us': etc['us'].detach(),
                    'vs': etc['vs'].detach(),
                    'outs': etc['outs'].detach()
                }

            return trial_loss, etc

    def test(self, current_xdg = True, multimodal_rsg=False,task_idx=None):
        with torch.no_grad():
            
            x, y, trials = next(iter(self.test_loader))
            x, y = x.to(self.device), y.to(self.device)
            #update:
            if self.args.xdg:
                self.net.xdg_test_mode()
                loss, etc = self.run_trial(x, y, trials, training=False, extras=True )
                self.net.xdg_train_mode()
            else:
                loss, etc = self.run_trial(x, y, trials, training=False, extras=True)

        etc = {
            'ins': x,
            'goals': y,
            'us': etc['us'].detach(),
            'pre_act_xs': etc['pre_act_xs'],
            'xs': etc['xs'].detach(),
            'vs': etc['vs'].detach(),
            'outs': etc['outs'].detach()
        }

        return loss, etc

    # default: helper function for sequential training, for testing performance on all tasks
    #if using multimodal: helper function for simultaneous training, for testing performance all all tasks 
    def test_tasks(self, ids=None):
        losses = []
        if self.args.multimodal:
            t_types= self.test_set.t_types
            # quick solution: get context signal ( will always be last self.args.T of input , extract that then position of 1 gives diff tasks)
            for task_type in t_types:
                self.test_loader = self.test_loaders[task_type]
                loss, _ = self.test()
                losses.append((task_type, loss))
            

        elif self.args.one_mod:
            for context, ds in enumerate(self.args.dataset):
                _trains, _tests = create_loaders(self.args.dataset, self.args, split_test=True, test_size=50, subset_loader=True)
                _t_set, one_task_test_loader = _tests
                # create outside of test tasks so we don't ahve to create a loader every time 
                self.test_loader = one_task_test_loader[self.args.train_order[context]]
                loss, _ = self.test()
                losses.append(('task_{}'.format(context), loss))
            self.test_loader = self.og_test_loader


        else:
            for i in ids:
                self.test_loader = self.test_loaders[self.args.train_order[i]]
                if self.args.xdg:
                    if self.train_idx ==0:
                        self.net.xdg_test_mode()
                    else:
                        self.net.xdg_test_mode(current = False, train_idx = i)
                
                loss, _ = self.test()
                #switch net gate back to those of current task being trained before resuming training
                if self.args.xdg:
                    self.net.xdg_train_mode(self.train_idx)
                losses.append((i, loss))
            self.test_loader = self.test_loaders[self.train_idx]
        return losses


    # todo: rename activity covariance - it's actually a general purpose function for estimating the covariance of a layer's activity from a batch - used in ncl and owm
    def owm_task_cov(self,act_batch= None, already_reshaped=False, ncl=False):
        #takes in activity tensor batches with size [batch_size, layer depth, timesteps] and computes covariance of these activities for a single task, averaged over trials and timesteps
        
        #reshape tensor so that all the activities are in the rows, number of rows should be equal to act_acount
        # ncl option will also return the number of timesteps T needed for calculation of the Fisher matrices in ncl

        if not ncl: 
            if not already_reshaped:
                n = act_batch.shape[0] * act_batch.shape[2] #total number of activities 
                act_batch_reshaped = act_batch.transpose(1,2).reshape(-1, act_batch.shape[1])
            #average of outer product of each row with itself
                return torch.mm(act_batch_reshaped.t(), act_batch_reshaped) # don't divide by n for now 
            else:
                return torch.mm(act_batch.t(), act_batch)
        if ncl:
            if not already_reshaped:
                T = act_batch.shape[2]
                n = act_batch.shape[0] * act_batch.shape[2] #total number of activities 
                act_batch_reshaped = act_batch.transpose(1,2).reshape(-1, act_batch.shape[1])
            #average of outer product of each row with itself
                return torch.mm(act_batch_reshaped.t(), act_batch_reshaped)/n, T # don't divide by n for now 
            else:
                return torch.mm(act_batch.t(), act_batch)


    
    # for later:  combineupdate cov and update projections shoul 
    def owm_update_tot_cov(self,total_cov, task_cov=None):
        k = self.train_idx +1
        return ((k-1)/k) * total_cov + (1/k) * task_cov
    def owm_update_proj(self, total_cov=None):
        return torch.inverse(self.args.alpha_owm**(-1) * total_cov + torch.eye(total_cov.shape[0]))
    


    def ncl_kl_div_kfa(self,A,B,C,D):
        #computes the Kroncecker product approximation of a the sum of kronecker products defined by A,B,C, D 
        # i.e. computes X,Y in X * Y approx_eq A*B + C* D
        
        # initialize X,Y by scaled additve approximation:
        pi_opt_sqrd = (torch.trace(B)*torch.trace(C))/ (torch.trace(A) * torch.trace(D))
        pi_opt = torch.sqrt(pi_opt_sqrd)
        X = (A + pi_opt*C)
        Y = (B + (1/pi_opt)*D)


        # kl_div approximation 
        beta = 0.3
        iters = 100
        n = A.shape[0]
        m = B.shape[0]
        for i in range(iters):
            X_inv = torch.inverse(X)
            Y_inv = torch.inverse(Y)
        
            X = (1-beta)*X + (beta/m) * (torch.trace(B@Y_inv)*A + torch.trace(D@Y_inv)*C)
            Y = (1-beta)*Y + (beta/n) * (torch.trace(A@X_inv)*B + torch.trace(C@X_inv)*D)
        return X,Y 
    

    def fim_KFA(self,layer_inputs,loss_grad_wrt_layer_outputs):
        # computes the Kronecker-factored approximation to the Fisher information matrix for a weight matrix
        # output should be a pair of matrices A,G 
        # needs to work with batches with shape [batch_size, layer depth, timesteps]
        
        input_cov, T = self.owm_task_cov(act_batch=layer_inputs,ncl=True )
        output_cov, T = self.owm_task_cov(act_batch =loss_grad_wrt_layer_outputs, ncl=True ) 

        #kronecker factors 
        A = T * input_cov
        G = output_cov
        return A, G


    def pre_inv_stabilize(self,square_matrix, coeff):
        return square_matrix + coeff * torch.eye(square_matrix.shape[0])


    def train(self, ix_callback=None):
        ix = 0
        # for convergence testing
        running_min_error = float('inf')
        running_no_min = 0

        running_loss = 0.0
        ending = False

        # for OWM
        if self.args.owm:
            self.val_trial_count = 0 # number of validation trials over which to average the task covariances; reset before moving onto new task
            task_sigma_s = 0
            task_sigma_u =0
            task_sigma_pre_act_u = 0
            task_sigma_q = 0
            task_sigma_x = 0
            task_sigma_z = 0

            
            self.total_sigma_s= 0
            self.total_sigma_pre_act_u = 0
            self.total_sigma_u = 0
            self.total_sigma_x = 0
            self.total_sigma_q =  0
            self.total_sigma_wq = 0
            self.total_sigma_z = 0

        

            s_ins = []
            us =[]
            rec_xs = []
            rec_xs_post_ac = []
            z_outs = []
            
        elif self.args.ncl:
            #initializing kronecker factors for first task
            if self.args.ncl:

                # initialize previous task thetas:
                self.m_u_prev =  self.net.M_u.weight.detach().clone()
                if self.args.D1 != 0:
                    self.w_u_prev = self.net.reservoir.W_u.weight.detach().clone()
                self.j_prev =  self.net.reservoir.J.weight.detach().clone()
                self.m_ro_prev = self.net.M_ro.weight.detach().clone()

                #intialize momentum holders for each paramter:
                self.M_m_ro = torch.zeros_like(self.m_ro_prev)
                if self.args.D1 !=0:
                    self.M_m_u = torch.zeros_like(self.m_u_prev)
                    if self.args.train_parts == ['']:
                        self.M_w = torch.zeros((self.args.N, self.args.N + self.args.D1))
                else:
                    self.M_w = torch.zeros((self.args.N, self.args.N + self.args.L + self.args.T))

                self.p_scalar =  1/torch.sqrt(torch.tensor(1e7))
                #1/torch.sqrt(torch.tensor((self.args.batch_size * self.args.seq_iters)))# p^-2  is defined as the number of samples seen when learning task k 
                
                # there's always a readout weight matrix and it's always N args.Z by args.N, hence kronecker factors always have the shapes below
                self.A_m_ro, self.G_m_ro = self.p_scalar * torch.eye(self.args.N), self.p_scalar * torch.eye(self.args.Z)
                self.A_m_ro_tilde, self.G_m_ro_tilde = self.ncl_kl_div_kfa(self.A_m_ro,self.G_m_ro, self.args.ncl_alpha*torch.eye(self.args.N), self.args.ncl_alpha*torch.eye(self.args.Z))
                # add ncl_alpha**2 I to matrices before inversions to avoid numerical issues:
                A_m_ro_tilde_padded, G_m_ro_tilde_padded = self.pre_inv_stabilize(self.A_m_ro_tilde, self.args.ncl_alpha**2), self.pre_inv_stabilize(self.G_m_ro_tilde, self.args.ncl_alpha**2)
                
                self.P_L_m_ro, self.P_R_m_ro =torch.inverse(G_m_ro_tilde_padded),  torch.inverse(A_m_ro_tilde_padded) 
                
                if self.args.D1 != 0:
                    self.A_m_u, self.G_m_u = self.p_scalar * torch.eye(self.args.L + self.args.T), self.p_scalar * torch.eye(self.args.D1)
                    self.A_m_u_tilde, self.G_m_u_tilde = self.ncl_kl_div_kfa(self.A_m_u,self.G_m_u, self.args.ncl_alpha*torch.eye(self.args.L + self.args.T), self.args.ncl_alpha*torch.eye(self.args.D1))
                    #cushion before inverting 
                    A_m_u_tilde_padded, G_m_u_tilde_padded = self.pre_inv_stabilize(self.A_m_u_tilde, self.args.ncl_alpha**2), self.pre_inv_stabilize(self.G_m_u_tilde, self.args.ncl_alpha**2)
                    
                    self.P_L_m_u, self.P_R_m_u = torch.inverse(G_m_u_tilde_padded), torch.inverse(A_m_u_tilde_padded) 

                    
                    if self.args.train_parts == ['']:
                        self.A_w, self.G_w = self.p_scalar * torch.eye(self.args.N + self.args.D1), self.p_scalar* torch.eye(self.args.N)
                        self.A_w_tilde, self.G_w_tilde = self.ncl_kl_div_kfa(self.A_w,self.G_w, self.args.ncl_alpha*torch.eye(self.args.N + self.args.D1), self.args.ncl_alpha*torch.eye(self.args.N))


                        A_w_tilde_padded, G_w_tilde_padded = self.pre_inv_stabilize(self.A_w_tilde, self.args.ncl_alpha**2), self.pre_inv_stabilize(self.G_w_tilde, self.args.ncl_alpha**2)

                        self.P_L_w, self.P_R_w = torch.inveres(G_w_tilde_padded), torch.inverse(A_w_tilde_padded)


                else:
                    self.A_w, self.G_w = self.p_scalar * torch.eye(self.args.N+ self.args.L+self.args.T), self.p_scalar * torch.eye(self.args.N)
                    self.A_w_tilde, self.G_w_tilde = self.ncl_kl_div_kfa(self.A_w,self.G_w, self.args.ncl_alpha*torch.eye(self.args.N+ self.args.L+self.args.T), self.args.ncl_alpha*torch.eye(self.args.N))

                    A_w_tilde_padded, G_w_tilde_padded = self.pre_inv_stabilize(self.A_w_tilde, self.args.ncl_alpha**2), self.pre_inv_stabilize(self.G_w_tilde, self.args.ncl_alpha**2)

                    self.P_L_w, self.P_R_w = torch.inverse(G_w_tilde_padded), torch.inverse(A_w_tilde_padded)



                



        if self.args.synaptic_intel:
            self.grads_list = {'M_u':[],'W_u':[], 'J':[], 'M_ro':[]}
            self.weight_changes_list = {'M_u':[],'W_u':[],'J':[], 'M_ro':[]}
            self.omega_m_u =0
            self.omega_w_u = 0
            self.omega_j =0
            self.omega_m_ro =0
            
            self.m_u_prev=0
            self.w_u_prev = 0
            self.j_prev = 0
            self.m_ro_prev=0

        elif self.args.ewc: 
            self.omega_m_u =0
            self.omega_w_u = 0
            self.omega_j =0
            self.omega_m_ro =0
            
            self.m_u_prev=0
            self.w_u_prev = 0
            self.j_prev = 0
            self.m_ro_prev=0

            
            
            
            

                
            
        
        if self.args.xdg:
            #save gates(gate templates to be precise) for this task. We use the same set of gates we used when training task k, for testing task k
            
            
            
            self.net.xdg_train_mode(self.train_idx)


            #save gates so that they are saved to config for plot_trained etc

            

         
        #timer:

        for e in range(self.args.n_epochs):
            for epoch_idx, (x, y, info) in enumerate(self.train_loader):
                ix += 1
                
                x, y = x.to(self.device), y.to(self.device)
                
                    
                

                if self.args.xdg:
                    self.net.xdg_train_mode(self.train_idx)

                
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
                    # print(test_etc['ins'].shape [batch_size, output dimension, task_length]
                    if self.args.owm:
                        
                        s_ins=test_etc['ins']
                        us = test_etc['us']
                        rec_xs = test_etc['pre_act_xs']
                        rec_xs_post_ac = test_etc['xs']
                        #
                        if self.args.out_act is not None or 'rsg' in self.args.dataset[0]:
                            
                            # if outs have been computed via a (non-indentity) activation function, manually compute pre-activation output
                            # reshape input tensor to (batch_size * timesteps, input_size)
                            rec_xs_before_readout = test_etc['xs'].transpose(1, 2).reshape(-1, test_etc['xs'].size(1))
                            z_outs = self.net.M_ro(rec_xs_before_readout)
                            z_outs = z_outs.reshape(test_etc['xs'].size(0), test_etc['xs'].size(2), -1).transpose(1, 2)
                

                        else:
                            z_outs = test_etc['outs']
                        
                        if  self.args.train_parts == [''] and self.args.D1 ==0 and self.args.D2==0:
                            # 0th dimension is the batch element, 2nd dimension is number of timesteps
                            # 1st dimension is the actual vector representation
                            #compute current task covariance
                            
                            # group different direct inputs to reservoir; q is the batch of [x^T , s^T]s - note: use xs preactivation
                            q_xs_ins =  torch.cat((rec_xs, s_ins), dim =1) #torch.Size([50, 307, 300], torch.allclose(q_xs_ins[:,:300, :], rec_xs ) == True

                           
                            task_sigma_q += self.owm_task_cov(q_xs_ins)
                            task_sigma_x += self.owm_task_cov(rec_xs_post_ac)
                            task_sigma_z += self.owm_task_cov(z_outs)
                            self.val_trial_count += z_outs.shape[0] * z_outs.shape[2]

                        elif self.args.train_parts == ['']  and self.args.D1 > 0:
                            q_xs_us = torch.cat((rec_xs, us), dim =1) 
                            
                            
                            task_sigma_s += self.owm_task_cov(s_ins)
                            task_sigma_u += self.owm_task_cov(us)

                            task_sigma_q += self.owm_task_cov(q_xs_us)
                            task_sigma_x += self.owm_task_cov(rec_xs_post_ac)
                            
                            task_sigma_z += self.owm_task_cov(z_outs)
                            self.val_trial_count += z_outs.shape[0] * z_outs.shape[2]


                        elif self.args.train_parts == ['M_u', 'M_ro']:
                            
                            
                            
                            
                            # M_u 
                            task_sigma_s += self.owm_task_cov(s_ins)
                            task_sigma_u += self.owm_task_cov(us)

                            # M_ro
                            task_sigma_x += self.owm_task_cov(rec_xs_post_ac)
                            task_sigma_z += self.owm_task_cov(z_outs)
                            self.val_trial_count += z_outs.shape[0] * z_outs.shape[2]
                        else:
                            raise NotImplementedError('No implentation of OWM for this network architecture yet. D1 and D2 should be zero if training all parts; if u-layer reservoir, D2 should be zero')
                        
                        
                
                    
                    log_arr = [
                        f'*{ix}',
                        f'train {train_loss:.3f}',
                        f'test {test_loss:.3f}'
                    ]
                    

                    if self.args.sequential:
                        seq_losses_all_tasks = self.test_tasks(ids=range(len(self.args.dataset)))
                        losses = self.test_tasks(ids=range(self.train_idx))
                        for i, loss in losses:
                            log_arr.append(f't{i}: {loss:.3f}')
                    log_str = '\t| '.join(log_arr)
                    logging.info(log_str)

                    

                    if not self.args.no_log:
                        if self.args.sequential:
                            self.log_checkpoint(ix, etc['ins'].cpu().numpy(), etc['goals'].cpu().numpy(), z, train_loss, test_loss, seq_losses_all_tasks)
                        else:
                            self.log_checkpoint(ix, etc['ins'].cpu().numpy(), etc['goals'].cpu().numpy(), z, train_loss, test_loss)
                    running_loss = 0.0

                    # if training sequentially, move on to the next task
                    # if doing OWM-like updates, do them here
                    

                    # ^ is XOR operator
                   
                    
                    
                    if (self.args.sequential and test_loss < self.args.seq_threshold and self.args.seq_iters == 0) ^ (self.args.sequential and self.args.seq_iters > 0 and ix == self.args.seq_iters + self.train_idx * self.args.seq_iters):
                        logging.info(f'Successfully trained task {self.train_idx}...')
                    
                        losses = self.test_tasks(ids=range(self.train_idx + 1))
                        

                        for i, loss in losses:
                            logging.info(f'...loss on task {i}: {loss:.3f}')
                        if self.args.synaptic_intel:
                            #current synaptic intel 
                            #update Omegas
                            
                            
                            total_change_m_u = 0
                            total_change_w_u = 0 
                            total_change_j = 0
                            total_change_m_ro  = 0



                            mini_omega_m_u = torch.zeros_like(self.net.M_u.weight).detach()
                            mini_omega_w_u = torch.zeros_like(self.net.reservoir.W_u.weight).detach()
                            mini_omega_j = torch.zeros_like(self.net.reservoir.J.weight).detach()
                            mini_omega_m_ro = torch.zeros_like(self.net.M_ro.weight).detach()

                            for i in range(len(self.grads_list['M_u'])):
                                mini_omega_m_u += self.weight_changes_list['M_u'][i] * self.grads_list['M_u'][i] * -1
                                total_change_m_u += self.weight_changes_list['M_u'][i]
                                
                                mini_omega_m_ro += self.weight_changes_list['M_ro'][i] * self.grads_list['M_ro'][i] * -1
                                total_change_m_ro += self.weight_changes_list['M_ro'][i]
                            
                                if self.args.train_parts == ['']:
                                    mini_omega_w_u += self.weight_changes_list['W_u'][i] * self.grads_list['W_u'][i] * -1
                                    total_change_w_u += self.weight_changes_list['W_u'][i]
                                    mini_omega_j += self.weight_changes_list['J'][i] * self.grads_list['J'][i] * -1
                                    total_change_j += self.weight_changes_list['J'][i]
                    
                            

                            max_input_m_u = mini_omega_m_u/(total_change_m_u**2 +self.args.damp_term)
                            max_input_m_ro = mini_omega_m_ro/(total_change_m_ro**2 +self.args.damp_term)
                            if self.args.train_parts == ['']:
                                max_input_w_u = mini_omega_w_u/(total_change_w_u**2 + self.args.damp_term)
                                max_input_j = mini_omega_j/(total_change_j**2 +self.args.damp_term)
                            
                            
                            
                            


                            task_omega_m_u= torch.maximum(torch.zeros_like(mini_omega_m_u),max_input_m_u)
                            task_omega_m_ro= torch.maximum(torch.zeros_like(mini_omega_m_ro),max_input_m_ro)
                            if self.args.train_parts == ['']:
                                task_omega_w_u = torch.maximum(torch.zeros_like(mini_omega_w_u), max_input_w_u)
                                task_omega_j= torch.maximum(torch.zeros_like(mini_omega_j),max_input_j )
                            

                            #save weights at the end of training this task - right now this is onlu works with weights, when we use biasses we'll go m_u_weight_prev
                            self.m_u_prev =  self.net.M_u.weight.detach().clone()
                            self.w_u_prev = self.net.reservoir.W_u.weight.detach().clone()
                            self.j_prev =  self.net.reservoir.J.weight.detach().clone()
                            self.m_ro_prev = self.net.M_ro.weight.detach().clone()
                            
                            #reset objects used to calculate omegas
                            self.grads_list = {'M_u':[],'W_u':[],'J':[], 'M_ro':[]}
                            self.weight_changes_list = {'M_u':[],'W_u':[],'J':[], 'M_ro':[]}
                            
                            self.omega_m_u += task_omega_m_u
                            self.omega_m_ro += task_omega_m_ro
                            if self.args.train_parts == ['']:
                                self.omega_w_u +=task_omega_w_u
                                self.omega_j += task_omega_j
                            
                        
                        elif self.args.ewc:
                            task_omega_m_u = torch.zeros_like(self.net.M_u.weight).detach()

                            if self.args.D1 > 0:
                                task_omega_w_u = torch.zeros_like(self.net.reservoir.W_u.weight).detach()
                            task_omega_j = torch.zeros_like(self.net.reservoir.J.weight).detach()
                            task_omega_m_ro = torch.zeros_like(self.net.M_ro.weight).detach()


                            num_batches_ewc = 32
                            
                            for i in range(num_batches_ewc):
                                x, y, info = next(iter(self.ewc_loader))
                                trial_loss , etc = self.run_trial(x,y, info, extras=True, ewc_fish_estim=True)
                                # calculate fisher importances for each batch, add, divide by num_of_batches

                                task_omega_m_u += torch.square(self.net.M_u.weight.grad).mean(dim=0)/num_batches_ewc # mean averages over no of samples in a single batch, num_of_batches gives average over number of batches 
                                if self.args.D1 > 0:
                                    task_omega_w_u += torch.square(self.net.reservoir.W_u.weight.grad).mean(dim=0)/num_batches_ewc
                                task_omega_j += torch.square(self.net.reservoir.J.weight.grad).mean(dim=0)/num_batches_ewc
                                
                                task_omega_m_ro += torch.square(self.net.M_ro.weight.grad).mean(dim=0)/num_batches_ewc

                            # theta^(prev)s
                            self.m_u_prev =  self.net.M_u.weight.detach().clone()
                            if self.args.D1 > 0:
                                self.w_u_prev = self.net.reservoir.W_u.weight.detach().clone()
                            self.j_prev =  self.net.reservoir.J.weight.detach().clone()
                            self.m_ro_prev = self.net.M_ro.weight.detach().clone()
                            
                            # update cumulative importances
                            self.omega_m_u += task_omega_m_u
                            if self.args.D1 > 0:
                                self.omega_w_u +=task_omega_w_u
                            self.omega_j += task_omega_j
                            self.omega_m_ro += task_omega_m_ro

                        
                        
                        elif self.args.ncl:
                            x, y, info = next(iter(self.ncl_loader))
                            trial_loss , etc = self.run_trial(x,y, info, extras=True, ncl_fish_estim=True)

                            # update fisher matrix components i.e. the kroncker products A,G for the sets of weights
                            # compute task components A_k, G_k which we use to update the aggregate components A_theta, G_theta

                            A_m_ro_k, G_m_ro_k = self.fim_FKA(etc['xs'], etc['grad_wrt_zs'])
                            self.A_m_ro, self.G_m_ro = self.ncl_kl_div_kfa(self.A_m_ro,self.G_m_ro, A_m_ro_k, G_m_ro_k)

                            # 
                            self.A_m_ro_tilde, self.G_m_ro_tilde = self.ncl_kl_div_kfa(self.A_m_ro,self.G_m_ro, self.args.ncl_alpha*torch.eye(self.args.N), self.args.ncl_alpha*torch.eye(self.args.Z))

                            A_m_ro_tilde_padded, G_m_ro_tilde_padded = self.pre_inv_stabilize(self.A_m_ro_tilde, self.args.ncl_alpha**2), self.pre_inv_stabilize(self.G_m_ro_tilde, self.args.ncl_alpha**2)
                            self.P_L_m_ro, self.P_R_m_ro = torch.inverse(A_m_ro_tilde_padded), torch.inverse(G_m_ro_tilde_padded)

                            if self.args.D1 !=0:
                                # F_M_u is always computed:
                                s_ins= x
                                grad_wrt_us = etc['grad_wrt_u']
                                #kroncker factored of FIM:  F_M_u = A \kron_prod G
                                A_m_u_k, G_m_u_k = self.fim_KFA(s_ins,grad_wrt_us)

                                #update projection matrices 
                                self.A_m_u, self.G_m_u = self.ncl_kl_div_kfa(self.A_m_u,self.G_m_u, A_m_u_k, G_m_u_k)
                                self.A_m_u_tilde, self.G_m_u_tilde = self.ncl_kl_div_kfa(self.A_m_u,self.G_m_u, self.args.ncl_alpha*torch.eye(self.args.L + self.args.T), self.args.ncl_alpha*torch.eye(self.args.D1))
                                A_m_u_tilde_padded, G_m_u_tilde_padded = self.pre_inv_stabilize(self.A_m_u_tilde, self.args.ncl_alpha**2), self.pre_inv_stabilize(self.G_m_u_tilde, self.args.ncl_alpha**2)
                                self.P_L_m_u, self.P_R_m_u = torch.inverse(A_m_u_tilde_padded), torch.inverse(G_m_u_tilde_padded)


                                if self.args.train_parts == '':
                                    # recurrent inputs
                                    us = etc['us'] # how do we know that these are preactivation us? check if ncl doesn't work
                                    rec_xs = etc['pre_act_xs']
                                    q_xs_us = torch.cat((rec_xs, us), dim =1) 
                                    # gradients of recurrent outputs 
                                    grad_xs = etc['grad_wrt_xs']
                                    # F_W
                                    A_w_k, G_w_k = self.fim_KFA(q_xs_us,grad_xs)
                                    self.A_w, self.G_w= self.ncl_kl_div_kfa(self.A_w,self.G_w, A_w_k, G_w_k)
                                    self.A_w_tilde, self.G_w_tilde = self.ncl_kl_div_kfa(self.A_w,self.G_w, self.args.ncl_alpha*torch.eye(self.args.N + self.args.D1), self.args.ncl_alpha*torch.eye(self.args.N))


                                    A_w_tilde_padded, G_w_tilde_padded = self.pre_inv_stabilize(self.A_w_tilde, self.args.ncl_alpha**2), self.pre_inv_stabilize(self.G_w_tilde, self.args.ncl_alpha**2)

                                    self.P_L_w, self.P_R_w = torch.inverse(A_w_tilde_padded, G_w_tilde_padded)


                        
                            else:
                                s_ins= x
                                rec_xs = etc['pre_act_xs']
                                q_xs_ins = torch.cat((rec_xs, us), dim =1) 
                                grad_xs = etc['grad_wrt_xs']
                                A_w_k, G_w_k = self.fim_KFA(q_xs_ins. grad_xs)

                                self.A_w_tilde, self.G_w_tilde = self.ncl_kl_div_kfa(self.A_w,self.G_w, self.args.ncl_alpha*torch.eye(self.args.N+ self.args.L+self.args.T), self.args.ncl_alpha*torch.eye(self.args.N))

                                A_w_tilde_padded, G_w_tilde_padded = self.pre_inv_stabilize(self.A_w_tilde, self.args.ncl_alpha**2), self.pre_inv_stabilize(self.G_w_tilde, self.args.ncl_alpha**2)

                                self.P_L_w, self.P_R_w = torch.inverse(A_w_tilde_padded, G_w_tilde_padded)

                            # update theta_{k-1}s before moving onto task k 
                            self.m_u_prev =  self.net.M_u.weight.detach().clone()
                            self.w_u_prev = self.net.reservoir.W_u.weight.detach().clone()
                            self.j_prev =  self.net.reservoir.J.weight.detach().clone()
                            self.m_ro_prev = self.net.M_ro.weight.detach().clone()
                            
                            

                            
                        
                        if self.args.owm:
                            # if training all parts
                            if self.args.train_parts == [''] and self.args.D1 == 0 and self.args.D2 ==0:
                                self.args.seq_threshold += 0.02
                                

                                # task_sigma_q /= self.val_trial_count
                                # task_sigma_x /= self.val_trial_count
                                # task_sigma_z /= self.val_trial_count
                                

                                
                                # task wq calculated using task_sigma_q 
                                j_end_of_task = self.net.reservoir.J.weight.detach().clone()
                                m_u_end_of_task =  self.net.M_u.weight.detach().clone()
                                W = torch.cat((j_end_of_task, m_u_end_of_task), dim =1)
                                
                                
                            

                                
                            
                                self.total_sigma_x=  self.owm_update_tot_cov(self.total_sigma_x, task_sigma_x)
                                self.total_sigma_q =  self.owm_update_tot_cov(self.total_sigma_q, task_sigma_q)
                                self.total_sigma_wq = W @ self.total_sigma_q @ W.t()
                                self.total_sigma_z = self.owm_update_tot_cov(self.total_sigma_z, task_sigma_z)
                                

                                
                                #update projections  - refactor as function later 
                                self.P_q = self.owm_update_proj(self.total_sigma_q)
                                self.P_wq = torch.inverse( (1/self.args.alpha_owm) * self.total_sigma_wq + torch.eye((self.total_sigma_wq.shape[0])) )
                                self.P_x = self.owm_update_proj(self.total_sigma_x)
                                self.P_z = self.owm_update_proj(self.total_sigma_z)

                                #starter for calc_cov(test_etc['ins'/'us','bs])
                                
                            
                                logging.info(f'...updated projection matrices for OWM')
                                # reset activity collectors for next task 
                                s_ins = []
                                rec_xs = []
                                rec_xs_post_ac = []
                                z_outs = []
                                self.val_trial_count = 0 # number of validation trials over which to average the task covariances; reset before moving onto new task
                                task_sigma_q = 0
                                task_sigma_x = 0
                                task_sigma_z = 0

                            elif self.args.train_parts == ['']  and self.args.D1 > 0:

                                
                                
                                
                                self.total_sigma_s = self.owm_update_tot_cov(self.total_sigma_s, task_sigma_s)
                                self.total_sigma_u = self.owm_update_tot_cov(self.total_sigma_u, task_sigma_u)

                                self.total_sigma_q = self.owm_update_tot_cov(self.total_sigma_q, task_sigma_q)
                               


                                j_end_of_task = self.net.reservoir.J.weight.detach().clone()
                                w_u_end_of_task =  self.net.reservoir.W_u.weight.detach().clone()
                                W = torch.cat((j_end_of_task, w_u_end_of_task ), dim =1)

                                self.total_sigma_wq = W @ self.total_sigma_q @ W.t()

                                self.total_sigma_x = self.owm_update_tot_cov(self.total_sigma_x, task_sigma_x)


                                self.total_sigma_z = self.owm_update_tot_cov(self.total_sigma_z, task_sigma_z)

                                self.P_s = self.owm_update_proj(self.total_sigma_s)
                                self.P_u = self.owm_update_proj(self.total_sigma_u)

                                self.P_q = self.owm_update_proj(self.total_sigma_q)
                                self.P_wq = torch.inverse( (1/self.args.alpha_owm) * self.total_sigma_wq + torch.eye((self.total_sigma_wq.shape[0])) )
                                self.P_x = self.owm_update_proj(self.total_sigma_x)
                                self.P_z = self.owm_update_proj(self.total_sigma_z)

                                self.val_trial_count = 0 # number of validation trials over which to average the task covariances; reset before moving onto new task
                                task_sigma_s = 0
                                task_sigma_u = 0
                                task_sigma_q = 0
                                task_sigma_x = 0
                                task_sigma_z = 0

                            
                            # if doing owm on reservoir with D2 ==0 
                            elif self.args.train_parts != [''] and self.args.D2 ==0 :
                                self.args.seq_threshold += 0.08
                                    # M_u
                                # task_sigma_s /= self.val_trial_count
                                # task_sigma_u /= self.val_trial_count

                                # # M_ro
                                # task_sigma_x /= self.val_trial_count
                                # task_sigma_z /= self.val_trial_count
                                
                                self.total_sigma_s = self.owm_update_tot_cov(self.total_sigma_s, task_sigma_s)
                                
                                self.total_sigma_u = self.owm_update_tot_cov(self.total_sigma_u, task_sigma_u)

                                self.total_sigma_x = self.owm_update_tot_cov(self.total_sigma_x, task_sigma_x)

                                self.total_sigma_z = self.owm_update_tot_cov(self.total_sigma_z, task_sigma_z)
                                
                                
                            
                                #update projections 
                                self.P_s = self.owm_update_proj(self.total_sigma_s)
                                self.P_u = self.owm_update_proj(self.total_sigma_u)

                                self.P_x = self.owm_update_proj(self.total_sigma_x)
                                self.P_z = self.owm_update_proj(self.total_sigma_z)
                                
                                logging.info(f'...updated projection matrices for OWM')
                                
                                # experimental implementation to explicitly enforce orthogonality in reservoir dynamics across tasks 
                                # if self.args.orthog_reservoirs:
                                # #pdate reservoir weights
                                #     pdb.set_trace()
                                #     projected_reservoir_weights_j_w_u = self.P_wq @ W @ self.P_q
                                    
                                    
                                #     self.net.reservoir.J.weight = nn.Parameter(self.net.reservoir.J.weight + projected_reservoir_weights_j_w_u[:, :300], requires_grad =True)
                                #     self.net.reservoir.W_u.weight= nn.Parameter(self.net.reservoir.W_u.weight + projected_reservoir_weights_j_w_u[:, -self.net.reservoir.W_u.weight.shape[1]:], requires_grad =True)



                                
                                self.val_trial_count = 0 # number of validation trials over which to average the task covariances; reset before moving onto new task
                                task_sigma_s = 0
                                task_sigma_u = 0
                                task_sigma_q = 0
                                task_sigma_x = 0
                                task_sigma_z = 0


                                     

    
                        self.train_idx += 1

                            

                        if self.args.xdg or self.args.synaptic_intel or self.args.owm:
                            #reset optimizer and scheduler between tasks as Masse et al do.
                            self.optimizer = get_optimizer(self.args, self.train_params)
                            self.scheduler = get_scheduler(self.args, self.optimizer)
                        
                        

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

                    if self.args.test_loss_stop and test_loss < self.args.test_threshold:
                        logging.info(f'test_loss < test_threshold of {self.args.test_threshold}. ending')
                        ending = True

                    elif running_no_min > self.args.patience:
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
        
        
       
        
        
        # for pca explained variances
        if self.args.pca_vars:
            x,y, trials = next(iter(self.test_loader))
            A = get_states(self.net, x)
            #pca multimodal with plot = False just returns the cumulative variances
            # note: n_reps needs to be the same as 'test size' for the test loader used to generate the A
            pca_vars = pca_multimodal(self.args, A_uncut=A, trials = trials, n_reps=50, plot=False)
            print(pca_vars)

            return running_min_error, ix, losses, pca_vars
        
        elif self.args.sequential:
            mean_loss = 0
            for idx_task_loss in losses:
                mean_loss += idx_task_loss[1]
            mean_loss /= len(self.args.dataset)
            print(f'mean loss {mean_loss}')
            
            return running_min_error, ix , losses , mean_loss
        
        else:
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














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

        if self.args.node_pert:
            if self.args.node_pert_parts == all:
                self.args.node_pert_parts == ['M_u,W_u,J,M_ro']  # perturbation is precisely the same for W_u and J

            # if using a standard optimizer like adam, set node_pert learning rate for each weight to 1
            if self.args.manual_node_pert:
                self.args.node_pert_lr_M_u = 1.
                self.args.node_pert_lr_W_u = 1.
                self.args.node_pert_lr_J = 1.
                self.args.node_pert_lr_M_ro = 1.


            # instantiate distributions from which to draw noises here once, rather than every time run trial is called
            for tp in self.args.node_pert_parts:

                # NP variance should scale with the number of units in the entire network
                unit_count = self.args.D1+self.args.Z+self.args.L+self.args.T + self.args.D2
                
                if tp == 'M_ro':
                    mean = torch.zeros(self.args.Z)
                    # scale noise variance by number of units in layer
                    self.args.node_pert_var_noise_z /= unit_count
                    cov = self.args.node_pert_var_noise_z* torch.eye(self.args.Z)
                    self.z_noise_mvn = torch.distributions.MultivariateNormal(mean,cov)

                elif  tp == 'M_u':
                    mean = torch.zeros(self.args.D1)
                    self.args.node_pert_var_noise_u /= unit_count
                    cov = self.args.node_pert_var_noise_u* torch.eye(self.args.D1)
                    self.u_noise_mvn = torch.distributions.MultivariateNormal(mean,cov)
                    
                elif tp == 'W_u' or 'J':
                    mean = torch.zeros(self.args.N)
                    self.args.node_pert_var_noise_x /= unit_count
                    cov = self.args.node_pert_var_noise_x* torch.eye(self.args.N)
                    self.x_noise_mvn = torch.distributions.MultivariateNormal(mean,cov)

                
                
        
            


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

    def run_trial(self, x, y, trial, training=True, extras=False):
        self.net.reset(self.args.res_x_init, device=self.device)
        trial_loss = 0.
        k_loss = 0.
        outs = []
        us = []
        if not training:
            pre_act_xs=[]
        xs = []
        vs = []
        

        # setting up k for t-BPTT
        if training and self.args.k != 0:
            k = self.args.k
        else:
            # k to full n means normal BPTT
            k = x.shape[2]


        
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

        
        
        if self.args.node_pert:
            # sample al noises for each timestep of trial here
            node_pert_noises =self.node_pert_noise_sampler(x.shape[2])
            if not self.args.node_pert_online:
                for tp in self.args.node_pert_parts:
                    if tp == 'M_ro':
                        time_sum_delta_M_ro = 0  #sum of the (batch of) node perturbation updates hat would be made at each step for the trial 

                    elif  tp == 'M_u':
                        time_sum_delta_M_u = 0
                        
                        
                    elif tp == 'W_u' :
                        time_sum_delta_W_u = 0 
                    else:
                        time_sum_delta_J = 0 
                        
            
            



        for j in range(x.shape[2]):
            net_in = x[:,:,j]
            net_out, etc = self.net(net_in, extras=True)

            # if ncl_fish_estim:
            #     pdb.set_trace()

            outs.append(net_out)
            
            us.append(etc['u'])
            if not training or (self.args.node_pert and 'J'in self.args.node_pert_parts):
                
                pre_act_xs.append(etc['pre_act_x'])
                
            xs.append(etc['x'])
            vs.append(etc['v'])
            
            if self.args.node_pert and training:
                
                # get the perturbations, for all batch elements,for a single timestep
                node_pert_noises_timestep_j = {}
                for key in node_pert_noises.keys():
                    node_pert_noises_timestep_j[key] = node_pert_noises[key][:,:,j]
                # noisy output for a single batch
                with torch.no_grad(): 
                    net_out_noise, etc = self.net(net_in, extras=True, node_pert_noises = node_pert_noises_timestep_j)
                    # noisy_outs.append(net_out_noise)
                # compute the effect of the node perturbation on loss # note you don't want the average loss over batches; you want the loss for each batch
                # so that we can calculate an average node_pert update over batches to get an average node_pert_update for this timestep
                
                error_noiseless = torch.nn.functional.mse_loss(net_out, y[:,:,j], reduction ='none').mean(dim=1) # should be of shape [batch_size]
                error_noise = torch.nn.functional.mse_loss(net_out_noise, y[:,:,j], reduction= 'none').mean(dim=1)


                pert_effect = error_noise - error_noiseless
                
              
                for tp in self.args.node_pert_parts:
                    with torch.no_grad():
                        if tp == 'M_u':
                            # let's see if we can use einstein notation here... batched outer prods (turn out to be mat vec multiplications ) then sum over batches. unsqueeze to faciliate matrix multiplication i.e. so that we have a batch of 2d arrays instead of a batch of 1D arrays 
                            
                            net_in_transposed = torch.einsum('bij->bji',net_in.unsqueeze(2))
                            
                            perf_effect_M_u = pert_effect.unsqueeze(1).unsqueeze(2).expand(self.args.batch_size,self.args.D1,self.args.L+self.args.T)
                            
                            update_batch_M_u = - (self.args.node_pert_lr_M_u/self.args.node_pert_var_noise_u) * perf_effect_M_u * torch.einsum('bij,bkj -> bij', node_pert_noises_timestep_j[tp].unsqueeze(2), net_in_transposed)

                            if not self.args.node_pert_online:
                                time_sum_delta_M_u += update_batch_M_u
                            else:
                                average_update_over_batches = torch.einsum('bij->ij',update_batch_M_u) / self.args.batch_size
                                average_update_over_batches/=x.shape[2]
                                
                                if self.args.manual_node_pert:
                                    self.net.M_u.weight.data = self.net.M_u.weight.data + average_update_over_batches

                                #pass the 'gradient' to the optimizer 
                                else:
                                    self.net.M_u.weight.grad = -1* average_update_over_batches
                        
                        elif tp == 'W_u':
                            
                            us_transposed = torch.einsum('bij,bji',us[j].unsqueeze(2))
                            pert_effect_W_u = pert_effect.unsqueeze(1).unsqueeze(2).expand(self.args.batch_size,self.args.N,self.args.D1)
                            update_batch_W_u = -(self.args.node_pert_lr_W_u/self.args.node_pert_var_noise_x) * pert_effect_W_u * torch.einsum('bij,bkj -> bij', node_pert_noises_timestep_j[tp].unsqueeze(2), us_transposed)

                            if not self.args.node_pert_online:
                                time_sum_delta_W_u += update_batch_W_u

                            else:
                                average_update_over_batches = torch.einsum('bij->ij',update_batch_W_u) / self.args.batch_size
                                average_update_over_batches/=x.shape[2]
                                if self.args.manual_node_pert:
                                    self.net.reservoir.W_u.weight.data = self.net.reservoir.W_u.weight.data + average_update_over_batches
                                else:
                                    self.reservoir.W_u.weight.grad = -1 * average_update_over_batches
                        
                        elif tp =='J':
                            pre_act_xs_transposed = torch.einsum('bij,bji',pre_act_xs[j].unsqueeze(2))
                            pert_effect_J = pert_effect.unsqueeze(1).unsqueeze(2).expand(self.args.batch_size,self.args.N,self.args.N)

                            update_batch_J = - (self.args.node_pert_lr_J/self.args.node_pert_var_noise_x) * pert_effect_J * torch.einsum('bij,bkj -> bij', node_pert_noises_timestep_j[tp].unsqueeze(2),pre_act_xs_transposed) 

                            if not self.args.node_pert_online:
                                time_sum_delta_J += update_batch_J
                            else:
                                average_update_over_batches = torch.einsum('bij->ij',update_batch_J) / self.args.batch_size
                                average_update_over_batches/=x.shape[2]
                                
                                if self.args.manual_node_pert:
                                    self.net.reservoir.J.weight.data = self.net.reservoir.J.weight.data  + average_update_over_batches
                                else:
                                    self.net.reservoir.J.grad = -1 * average_update_over_batches

                        elif tp == 'M_ro': 
                            
                            xs_transposed = torch.einsum('bij->bji',xs[j].unsqueeze(2))
                            pert_effect_M_ro = pert_effect.unsqueeze(1).unsqueeze(2).expand(self.args.batch_size,self.args.Z,self.args.N)
                            update_batch_M_ro = -(self.args.node_pert_lr_M_ro/self.args.node_pert_var_noise_z) *  pert_effect_M_ro * torch.einsum('bij,bkj -> bij', node_pert_noises_timestep_j[tp].unsqueeze(2),xs_transposed)
                            
                            if not self.args.node_pert_online:
                                time_sum_delta_M_ro += update_batch_M_ro

                            else:
                                average_update_over_batches = torch.einsum('bij->ij',update_batch_M_ro) / self.args.batch_size
                                average_update_over_batches/=x.shape[2]

                                

                                if self.args.manual_node_pert:
                                    

                                    self.net.M_ro.weight.data  = self.net.M_ro.weight.data + average_update_over_batches
                                else:
                                    self.net.M_ro.weight.grad = -1 * average_update_over_batches
                    

                
                
        
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
                if training and not self.args.rflo and not self.args.node_pert:
                        
                        k_loss.backward()
                        
                        #gradient clipping for softplus bce-loss
                        if 'sp-bce' in self.args.loss:
                            #clip gradients
                            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)

                        
                            #
                if training and self.args.node_pert and not self.args.node_pert_online:
                    if not self.args.node_pert_online:
                        for tp in self.args.node_pert_parts:
                            if tp == 'M_ro':
                                time_sum_delta_M_ro /= x.shape[2]  #time_sum_delta_M holds the sum of the NP updates that would have been made at each timestep/ sum and and divide by t
                                average_update_over_batches = torch.einsum('bij->ij',update_batch_M_ro) / self.args.batch_size
                                self.net.M_ro.weight.data = self.net.M_ro.weight.data + average_update_over_batches


                            elif  tp == 'M_u':
                                time_sum_delta_M_u /= x.shape[2]
                                average_update_over_batches = torch.einsum('bij->ij',update_batch_M_u) / self.args.batch_size
                                self.net.M_u.weight.data = self.net.M_u.weight.data + average_update_over_batches
                                
                            elif tp == 'W_u' :
                                time_sum_delta_W_u /= x.shape[2]
                                
                                average_update_over_batches = torch.einsum('bij->ij',update_batch_W_u) / self.args.batch_size
                                self.net.reservoir.W_u.weight.data = self.net.reservoir.W_u.weight.data + average_update_over_batches
                            else:
                                time_sum_delta_J /= x.shape[2]
                                
                                average_update_over_batches = torch.einsum('bij->ij',update_batch_J) / self.args.batch_size
                                self.net.reservoir.J.weight.data =self.net.reservoir.J.weight.data + average_update_over_batches
                        



                k_loss = 0.
                self.net.reservoir.x = self.net.reservoir.x.detach()
       
                 
        if training and self.args.rflo:
            if self.args.batch_size > 1:

                self.net.M_u.weight.grad = - 1* torch.sum(sum(delta_M_u_time_series),dim=0)/self.args.batch_size 
                self.net.M_ro.weight.grad = -1* torch.sum(sum(delta_M_ro_time_series), dim=0)/self.args.batch_size 
            else:
                self.net.M_u.weight.grad = - 1* sum(delta_M_u_time_series)/self.args.batch_size 
                self.net.M_ro.weight.grad = -1* sum(delta_M_ro_time_series)/self.args.batch_size 

                
                        
        trial_loss /= x.shape[0]

        if extras:
            
            net_us = torch.stack(us, dim=2)
            
            net_pre_act_xs = torch.stack(xs,dim=2)
            net_xs = torch.stack(xs, dim =2)
            net_vs = torch.stack(vs, dim=2)
            net_outs = torch.stack(outs, dim=2)

            
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
                etc = {
                        'outs': net_outs,
                        'us': net_us,
                        'xs': net_xs,
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

    
    
    # node pertubation helper functions
    def node_pert_noise_sampler(self,timesteps):
        # returns a dictionary of node pertubations for each timestep of a single trial 
        # keys are weight names, values are batches of perturbations i.e. tensors of shape [batch_size, layer_size, timesteps]
        # perturbations for each timesteps are stored in rows i.e. timestep is the row index
        node_pert_noises = {}
        
        # fixed perturbation throughout the trial
        if not self.args.dynamic_pert:
            
            # perturbation fixed throughout all trials (i.e. constant wrt to time and batch index) - this should reduce variance in the estimate of the gradient for that perturbation
            
            for tp in self.args.node_pert_parts:
                if tp == 'M_u':
                    u_noise = self.u_noise_mvn.sample()
                    u_noise = u_noise.view(1,self.args.D1,1).expand(self.args.batch_size, self.args.D1, timesteps)
                    node_pert_noises[tp] = u_noise
                elif tp == 'W_u' or tp == 'J':
                    x_noise = self.x_noise_mvn.sample()
                    x_noise = x_noise.view(1,self.args.N,1).expand(self.args.batch_sizes, self.args.N, timesteps)
                    node_pert_noises[tp] = x_noise
                elif tp == 'M_ro':
                    z_noise = self.z_noise_mvn.sample()
                    z_noise = z_noise.view(1,self.args.Z,1).expand(self.args.batch_size,self.args.Z, timesteps)
                    node_pert_noises[tp] = z_noise


            # perturbation fixed throughout trial but varies over batches
            # for tp in self.args.node_pert_parts:
            #     if tp == 'M_u':
            #         u_noise = self.u_noise_mvn.sample(sample_shape=(self.args.batch_size,))
            #         node_pert_noises[tp] = torch.stack([u_noise for t in range(timesteps)],dim=2)
            #     elif tp == 'W_u' or tp == 'J':
            #         x_noise = self.x_noise_mvn.sample(sample_shape=(self.args.batch_size,))
            #         node_pert_noises[tp] = torch.stack([x_noise for t in range(timesteps)],dim=2)
            #     elif tp == 'M_ro':
            #         z_noise = self.z_noise_mvn.sample(sample_shape=(self.args.batch_size,))
            #         node_pert_noises[tp] = torch.stack([z_noise for t in range(timesteps)],dim=2)
       
        else:
            for tp in self.args.node_pert_parts:
                if tp == 'M_u':
                    node_pert_noises[tp] = torch.stack([self.u_noise_mvn.sample(sample_shape=(self.args.batch_size,)) for t in range(timesteps)],dim=2)
                elif tp == 'W_u' or tp == 'J':
                    node_pert_noises[tp] = torch.stack([self.x_noise_mvn.sample(sample_shape=(self.args.batch_size,)) for t in range(timesteps)],dim=2)
                elif tp == 'M_ro':
                    node_pert_noises[tp] = torch.stack([self.z_noise_mvn.sample(sample_shape=(self.args.batch_size,)) for t in range(timesteps)],dim=2)
    
        return node_pert_noises
            
    
    

    def train(self, ix_callback=None):
        ix = 0
        # for convergence testing
        running_min_error = float('inf')
        running_no_min = 0

        running_loss = 0.0
        ending = False

       

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
                    # print(test_etc['ins'].shape [batch_size, output dimension, task_length]
                   
                        

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














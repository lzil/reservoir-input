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

        if self.args.sequential:
            logging.info(f'Sequential training. Starting with task {self.train_idx}')

            
            if self.args.xdg:
                logging.info(f'Implementing context-dependent gating. Gating {self.args.X}% of units in layers:')
                for layer in self.args.gate_layers:
                    logging.info(f'  {layer}')
                
            
            if self.args.synaptic_intel:
                logging.info(f'Stabilizing trainable parameters with synaptic intelligence. Hyperparameter values: ')
                logging.info(f'  Stabilization strength: {self.args.stab_strength}')
                logging.info((f'  Damping term : {self.args.damp_term}'))
        
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
            if self.args.sequential:
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

        if self.args.synaptic_intel and training:
            grads_m_u = 0
            grads_j = 0
            grads_m_ro = 0
        

        for j in range(x.shape[2]):
            net_in = x[:,:,j]
            net_out, etc = self.net(net_in, extras=True)
            outs.append(net_out)
            us.append(etc['u'])
            if not training:
                
                pre_act_xs.append(etc['pre_act_x'])
                
            xs.append(etc['x'])
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
                    if self.args.synaptic_intel:
                        #calculate gradients for loss on current task without SI penalty; save them


                        k_loss.backward()
                        
                       
                        # self.grads_list['M_u'].append(self.net.M_u.weight.grad.detach().clone())
                        # self.grads_list['J'].append(self.net.reservoir.J.weight.grad.detach().clone())
                        # self.grads_list['M_ro'].append(self.net.M_ro.weight.grad.detach().clone())
                        grads_m_u += self.net.M_u.weight.grad.detach().clone()
                        grads_j += self.net.reservoir.J.weight.grad.detach().clone()
                        grads_m_ro = self.net.M_ro.weight.grad.detach().clone()
                        
                        
                        
                        
                        penalty_m_u = torch.sum(self.omega_m_u*(self.net.M_u.weight - self.m_u_prev)**2)
                        penalty_j= torch.sum(self.omega_j*(self.net.reservoir.J.weight - self.j_prev)**2)
                        penalty_m_ro = torch.sum(self.omega_m_ro * (self.net.M_ro.weight - self.m_ro_prev)**2)
                        # i think we need to times by k so as to add SI for loss at each timestep 
                        
                        synaptic_intel_penalty = self.args.stab_strength * (penalty_m_u + penalty_j + penalty_m_ro)
                        
                    

                        


                        
                        
                        synaptic_intel_penalty.backward()
                    else:
                        
                        k_loss.backward()
                        if 'sp-bce' in self.args.loss:
                            #clip gradients
                            
                            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
        if self.args.owm and self.train_idx > 0 and training:
            if self.args.owm:
                if self.args.train_parts == ['']:
                
                    grad_j= self.net.reservoir.J.weight.grad.detach().clone()
                    grad_m_u = self.net.M_u.weight.grad.detach().clone()
                    grad_w = torch.cat((grad_j, grad_m_u), dim=1)

                    #project input and recurrent weights
                    delta_w = self.P_wq @ grad_w @ self.P_q
                    
                    self.net.M_u.weight.grad = delta_w[:, -self.net.M_u.weight.shape[1]: ]
                    self.net.M_u.weight.grad = self.net.M_u.weight.grad.contiguous() # assigning slices to a grad seems to cause a performance issue, contiguous() fixes it in this case
                    
                    self.net.reservoir.J.weight.grad =  delta_w[:, :300]
                    self.net.reservoir.J.weight.grad = self.net.reservoir.J.weight.grad.contiguous()
                    
                    self.net.M_ro.weight.grad = self.P_z @ self.net.M_ro.weight.grad @ self.P_x
                else:
                    self.net.M_u.weight.grad =  self.P_u @ self.net.M_u.weight.grad @ self.P_s
                    self.net.M_ro.weight.grad = self.P_z @ self.net.M_ro.weight.grad @ self.P_x    
                            
                k_loss = 0.
                self.net.reservoir.x = self.net.reservoir.x.detach()

        trial_loss /= x.shape[0]
        if self.args.synaptic_intel and training:
            self.grads_list['M_u'].append(grads_m_u)
            self.grads_list['J'].append(grads_j)
            self.grads_list['M_ro'].append(grads_m_ro)


        if extras:
            
            net_us = torch.stack(us, dim=2)
            if not training:
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
        
        if self.args.synaptic_intel:
            
            m_u_before_step = self.net.M_u.weight.detach().clone()
            j_before_step =  self.net.reservoir.J.weight.detach().clone()
            m_ro_before_step = self.net.M_ro.weight.detach().clone()
            trial_loss, etc = self.run_trial(x, y, trial, extras=True)
            

            if ix_callback is not None:
                ix_callback(trial_loss, etc)
            self.optimizer.step()
            m_u_change= self.net.M_u.weight.detach().clone() - m_u_before_step
            j_change=  self.net.reservoir.J.weight.detach().clone() -j_before_step
            m_ro_change = self.net.M_ro.weight.detach().clone() -m_ro_before_step
            
            
            self.weight_changes_list['M_u'].append(m_u_change)
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
            self.optimizer.step()

            etc = {
                'ins': x,
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



    def owm_task_cov(self,act_batch= None):
        #takes in activity tensor batches with size [batch_size, layer depth, timesteps] and computes covariance of these activities for a single task, averaged over trials and timesteps
        n = act_batch.shape[0] * act_batch.shape[2] #total number of activities 
        #reshape tensor so that all the activities are in the rows, number of rows should be equal to act_acount
        act_batch_reshaped = act_batch.transpose(1,2).reshape(-1, act_batch.shape[1])
        #average of outer product of each row with itself
        return torch.mm(act_batch_reshaped.t(), act_batch_reshaped) / n
    
    # for later:  combineupdate cov and update projections shoul 
    def owm_update_tot_cov(self,total_cov, task_cov=None):
        k = self.train_idx +1
        return ((k-1)/k) * total_cov + (1/k) * task_cov
    def owm_update_proj(self, total_cov=None):
        return torch.inverse(self.args.alpha_owm**(-1) * total_cov + torch.eye(total_cov.shape[0]))
    
                

    def train(self, ix_callback=None):
        ix = 0
        # for convergence testing
        running_min_error = float('inf')
        running_no_min = 0

        running_loss = 0.0
        ending = False

       # for OWM
        if self.args.owm:
            self.total_sigma_s= 0
            self.total_sigma_pre_act_u = 0
            self.total_sigma_x = 0
            self.total_sigma_q =  0
            self.total_sigma_wq = 0
            self.total_sigma_z = 0

        

            s_ins = []
            us =[]
            rec_xs = []
            rec_xs_post_ac = []
            z_outs = []
            
            
        if self.args.synaptic_intel:
            self.grads_list = {'M_u':[],'J':[], 'M_ro':[]}
            self.weight_changes_list = {'M_u':[],'J':[], 'M_ro':[]}
            self.omega_m_u =0
            self.omega_j =0
            self.omega_m_ro =0
            self.m_u_prev=0
            self.j_prev = 0
            self.m_ro_prev=0
            
            
            
            
            # # OWM for trainable params
            # self.train_params_dict= {}
            # for k,v in self.net.named_parameters():
            #     for part in self.args.train_parts:
            #         if part in k:
            #             self.train_params_dict[k] = v
          

            # #initialise owm ovariances and projection matrices where key is the weight name and value is a list whose  first element is the input activity covariance, 2nd element is output activity covariance, third element is the input (right)projecton matrix, fourth is the output ( left) projection matrix

            # self.covs_and_projs= {}
            # for weight in self.net.state_dict().keys():
            #     #[input cov, output cov, projection ]
            #     self.covs_and_projs[weight] = [0,0,0,0]
                
            
        
        if self.args.xdg:
            #save gates(gate templates to be precise) for this task. We use the same set of gates we used when training task k, for testing task k
            
            
            
            self.net.xdg_train_mode(self.train_idx)


            #save gates so that they are saved to config for plot_trained etc

            

         

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
                    if self.args.multimodal:
                        log_arr = [
                        f'*{ix}',
                        f'train {train_loss:.3f}'
                    ]
                        losses = self.test_tasks()
                        test_loss= 0
                        no_of_tasks = 0
                        for task, loss in losses:
                            no_of_tasks += 1
                            test_loss += loss
                        test_loss /= no_of_tasks
                        log_arr.append(f'test {test_loss:.3f}')
                        for task, loss in losses:
                            log_arr.append(f'{task}: {loss:.3f}')
                    
                    elif self.args.one_mod:
                        log_arr = [
                        f'*{ix}',
                        f'train {train_loss:.3f}'
                    ]
                        losses = self.test_tasks()
                        test_loss= 0
                        no_of_tasks = 0
                        for task, loss in losses:
                            no_of_tasks += 1
                            test_loss += loss
                        test_loss /= no_of_tasks
                        log_arr.append(f'test {test_loss:.3f}')
                        for task, loss in losses:
                            log_arr.append(f'{task}: {loss:.3f}')
                    else:
                        test_loss, test_etc = self.test()
                        # print(test_etc['ins'].shape [batch_size, output dimension, task_length]
                        if self.args.owm:
                        s_ins.append(test_etc['ins'])
                        us.append(test_etc['us'])
                        rec_xs.append(test_etc['pre_act_xs'])
                        rec_xs_post_ac.append(test_etc['xs'])
                        z_outs.append(test_etc['outs'])
                        
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
                    


                    if self.args.sequential and self.args.seq_iters == 0:
                    
                    
                        if self.args.sequential and test_loss < self.args.seq_threshold:
                            logging.info(f'Successfully trained task {self.train_idx}...')
                            
                            losses = self.test_tasks(ids=range(self.train_idx + 1))
                            

                            for i, loss in losses:
                                logging.info(f'...loss on task {i}: {loss:.3f}')
                            if self.args.synaptic_intel:
                                #current synaptic intel 
                                #update Omegas
                                
                                
                                total_change_m_u = 0
                                total_change_j = 0
                                total_change_m_ro  = 0



                                mini_omega_m_u = torch.zeros_like(self.net.M_u.weight).detach()
                                
                                mini_omega_j = torch.zeros_like(self.net.reservoir.J.weight).detach()
                                
                                mini_omega_m_ro = torch.zeros_like(self.net.M_ro.weight).detach()

                                for i in range(len(self.grads_list['M_u'])):
                                    mini_omega_m_u += self.weight_changes_list['M_u'][i] * self.grads_list['M_u'][i] * -1
                                    mini_omega_j += self.weight_changes_list['J'][i] * self.grads_list['J'][i] * -1
                                    mini_omega_m_ro += self.weight_changes_list['M_ro'][i] * self.grads_list['M_ro'][i] * -1

                                    total_change_m_u += self.weight_changes_list['M_u'][i]
                                    total_change_j += self.weight_changes_list['J'][i]
                                    total_change_m_ro += self.weight_changes_list['M_ro'][i]
                                

                                max_input_m_u = mini_omega_m_u/(total_change_m_u**2 +self.args.damp_term)
                                max_input_j = mini_omega_j/(total_change_j**2 +self.args.damp_term)
                                
                                max_input_m_ro = mini_omega_m_ro/(total_change_m_ro**2 +self.args.damp_term)
                                
                                


                                task_omega_m_u= torch.maximum(torch.zeros_like(mini_omega_m_u),max_input_m_u )
                                task_omega_j= torch.maximum(torch.zeros_like(mini_omega_j),max_input_j )
                                task_omega_m_ro= torch.maximum(torch.zeros_like(mini_omega_m_ro),max_input_m_ro)

                                #save weights at the end of training this task - right now this is onlu works with weights, when we use biasses we'll go m_u_weight_prev
                                self.m_u_prev =  self.net.M_u.weight.detach().clone()
                                self.j_prev =  self.net.reservoir.J.weight.detach().clone()
                                self.m_ro_prev = self.net.M_ro.weight.detach().clone()

                                #reset objects used to calculate omegas
                                self.grads_list = {'M_u':[],'J':[], 'M_ro':[]}
                                self.weight_changes_list = {'M_u':[],'J':[], 'M_ro':[]}
                                
                                self.omega_m_u += task_omega_m_u
                                self.omega_j += task_omega_j
                                self.omega_m_ro += task_omega_m_ro
                                

                            

                            # orthogonal weight modification of M_u and M_ro
                            if self.args.owm:
                                 #if training all parts 
                                if self.args.train_parts == ['']:
                                    self.args.seq_threshold += 0.02
                                    # 0th dimension is the batch element, 2nd dimension is number of timesteps
                                    # 1st dimension is the actual vector representation
                                    #compute current task covariances



                                    s_ins = torch.cat(s_ins, dim=0)
                                    
                                    
                                    rec_xs = torch.cat(rec_xs, dim =0 )

                                    rec_xs_post_ac = torch.cat(rec_xs_post_ac, dim =0)
                                    
                                    z_outs = torch.cat(z_outs, dim=0)

                                    # group different direct inputs to reservoir; q is the batch of [x^T , s^T]s - note: use xs preactivation
                                    q_xs_ins =  torch.cat((rec_xs, s_ins), dim =1)  #torch.Size([50, 307, 300], torch.allclose(q_xs_ins[:,:300, :], rec_xs ) == True

                                    task_sigma_q = self.owm_task_cov(q_xs_ins)
                                    task_sigma_x = self.owm_task_cov(rec_xs_post_ac)
                                    task_sigma_z = self.owm_task_cov(z_outs)
                                    

                                    
                                    # task wq calculated using task_sigma_q 
                                    j_end_of_task = self.net.reservoir.J.weight.detach().clone()
                                    m_u_end_of_task =  self.net.M_u.weight.detach().clone()
                                    W = torch.cat((j_end_of_task, m_u_end_of_task), dim =1)
                                    
                                    
                                

                                    #update total covariances - refactor as function later
                                
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
                                
                                # if doing owm on reservoir with D2 ==0 
                                else:
                                    self.args.seq_threshold += 0.08
                                    # 0th dimension is the batch element, 2nd dimension is number of timesteps
                                    # 1st dimension is the actual vector representation
                                    #compute current task covariances



                                    
                                    # task activities 
                                    s_ins = torch.cat(s_ins, dim=0)
                                    us = self.net.M_u(s_ins)



                                    rec_xs = torch.cat(rec_xs, dim =0 )
                                    rec_xs_post_ac = torch.cat(rec_xs_post_ac, dim =0)
                                    z_outs = torch.cat(z_outs, dim=0)
                                    q_xs_ins =  torch.cat((rec_xs, us), dim =1)

                                    task_sigma_s = 0
                                    task_sigma_pre_act_u = 0
                                    task_sigma_q = 0
                                    task_sigma_x =0
                                    task_sigma_z= 0
                                    
                                    # takes care of M_u
                                    task_sigma_s = self.owm_task_cov(s_ins)
                                    task_sigma_u = self.own_task_covs(us)

                                    # M_ro
                                    task_sigma_x = self.owm_task_covs(rec_xs_post_ac)
                                    task_sigma_z = self.owm_task_covs(z_outs)
                                    
                                    
                                    self.total_sigma_s = self.owm_update_tot_cov(self.total_sigma_s, task_sigma_s)
                                    
                                    self.total_sigma_u = self.owm_update_tot_cov(self.total_sigma_u, task_sigma_u)

                                    self.total_sigma_x = self.owm_update_tot_cov(self.total_sigma_x, task_sigma_x)

                                    self.total_sigma_z = self.owm_update_tot_cov(self.total_sigma_z, task_sigma_z)
                                    
                                    
                                
                                    #update projections 
                                    self.P_s = self.owm_update_proj(self.total_sigma_s)
                                    self.P_u = self.owm_update_proj(self.totat_sigma_u)

                                    self.P_x = self.owm_update_proj(self.total_sigma_x)
                                    self.P_z = self.owm_update_proj(self.total_sigma_z)
                                    
                                    logging.info(f'...updated projection matrices for OWM')
                                    
                                    if self.args.orthog_reservoirs:
                                    #pdate reservoir weights
                                        pdb.set_trace()
                                        projected_reservoir_weights_j_w_u = self.P_wq @ W @ self.P_q
                                        
                                        
                                        self.net.reservoir.J.weight = nn.Parameter(self.net.reservoir.J.weight + projected_reservoir_weights_j_w_u[:, :300], requires_grad =True)
                                        self.net.reservoir.W_u.weight= nn.Parameter(self.net.reservoir.W_u.weight + projected_reservoir_weights_j_w_u[:, -self.net.reservoir.W_u.weight.shape[1]:], requires_grad =True)



                                    s_ins = []
                                    
                                    rec_xs = []
                                    rec_xs_post_ac = []
                                    z_outs = []
                                
                                
                                
                                logging.info(f'...updated projection matrices for OWM')
                                
                            # done processing prior task, move on to the next one or quit
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


                    elif self.args.sequential and self.args.seq_iters > 0 and ix == self.args.seq_iters + self.train_idx * self.args.seq_iters:
                        
                        
                        logging.info(f'Successfully trained task {self.train_idx}...')
                            
                        losses = self.test_tasks(ids=range(self.train_idx + 1))
                            

                        for i, loss in losses:
                            logging.info(f'...loss on task {i}: {loss:.3f}')
                        if self.args.synaptic_intel:
                            #current synaptic intel 
                            #update Omegas
                            
                            
                            total_change_m_u = 0
                            total_change_j = 0
                            total_change_m_ro  = 0



                            mini_omega_m_u = torch.zeros_like(self.net.M_u.weight).detach()
                            
                            mini_omega_j = torch.zeros_like(self.net.reservoir.J.weight).detach()
                            
                            mini_omega_m_ro = torch.zeros_like(self.net.M_ro.weight).detach()

                            for i in range(len(self.grads_list['M_u'])):
                                mini_omega_m_u += self.weight_changes_list['M_u'][i] * self.grads_list['M_u'][i] * -1
                                mini_omega_j += self.weight_changes_list['J'][i] * self.grads_list['J'][i] * -1
                                mini_omega_m_ro += self.weight_changes_list['M_ro'][i] * self.grads_list['M_ro'][i] * -1

                                total_change_m_u += self.weight_changes_list['M_u'][i]
                                total_change_j += self.weight_changes_list['J'][i]
                                total_change_m_ro += self.weight_changes_list['M_ro'][i]
                            

                            max_input_m_u = mini_omega_m_u/(total_change_m_u**2 +self.args.damp_term)
                            max_input_j = mini_omega_j/(total_change_j**2 +self.args.damp_term)
                            
                            max_input_m_ro = mini_omega_m_ro/(total_change_m_ro**2 +self.args.damp_term)
                            
                            


                            task_omega_m_u= torch.maximum(torch.zeros_like(mini_omega_m_u),max_input_m_u )
                            task_omega_j= torch.maximum(torch.zeros_like(mini_omega_j),max_input_j )
                            task_omega_m_ro= torch.maximum(torch.zeros_like(mini_omega_m_ro),max_input_m_ro)

                            #save weights at the end of training this task - right now this is onlu works with weights, when we use biasses we'll go m_u_weight_prev
                            self.m_u_prev =  self.net.M_u.weight.detach().clone()
                            self.j_prev =  self.net.reservoir.J.weight.detach().clone()
                            self.m_ro_prev = self.net.M_ro.weight.detach().clone()

                            #reset objects used to calculate omegas
                            self.grads_list = {'M_u':[],'J':[], 'M_ro':[]}
                            self.weight_changes_list = {'M_u':[],'J':[], 'M_ro':[]}
                            
                            self.omega_m_u += task_omega_m_u
                            self.omega_j += task_omega_j
                            self.omega_m_ro += task_omega_m_ro
                            

                        

                        # orthogonal weight modification of M_u and M_ro
                        if self.args.owm:
                            if self.args.owm:
                            # if training all parts
                                if self.args.train_parts == [''] and self.args.D1 ==0 and self.args.D2==0:
                                    # 0th dimension is the batch element, 2nd dimension is number of timesteps
                                        # 1st dimension is the actual vector representation
                                        #compute current task covariances
                                    s_ins = torch.cat(s_ins, dim=0)
                                    
                                    rec_xs = torch.cat(rec_xs, dim =0 )

                                    rec_xs_post_ac = torch.cat(rec_xs_post_ac, dim =0)
                                    
                                    z_outs = torch.cat(z_outs, dim=0)

                                    # group different direct inputs to reservoir; q is the batch of [x^T , s^T]s - note: use xs preactivation
                                    q_xs_ins =  torch.cat((rec_xs, s_ins), dim =1)  #torch.Size([50, 307, 300], torch.allclose(q_xs_ins[:,:300, :], rec_xs ) == True

                                    task_sigma_q = self.owm_task_cov(q_xs_ins)
                                    task_sigma_x = self.owm_task_cov(rec_xs_post_ac)
                                    task_sigma_z = self.owm_task_cov(z_outs)
                                    

                                    
                                    # task wq calculated using task_sigma_q 
                                    j_end_of_task = self.net.reservoir.J.weight.detach().clone()
                                    m_u_end_of_task =  self.net.M_u.weight.detach().clone()
                                    W = torch.cat((j_end_of_task, m_u_end_of_task), dim =1)
                                    
                                    
                                

                                    #update total covariances - refactor as function later
                                
                                    self.total_sigma_x=  self.owm_update_tot_cov(self.total_sigma_x, task_sigma_x)
                                    self.total_sigma_q =  self.owm_update_tot_cov(self.total_sigma_q, task_sigma_q)
                                    self.total_sigma_z = self.owm_update_tot_cov(self.total_sigma_z, task_sigma_z)

                                    total_sigma_wq = W @ self.total_sigma_q @ W.t()
                                    #update projections  - refactor as function later 
                                    self.P_q = self.owm_update_proj(self.total_sigma_q)
                                    self.P_wq = torch.inverse( (1/self.args.alpha_owm) * total_sigma_wq + torch.eye((total_sigma_wq.shape[0])) )
                                    self.P_x = self.owm_update_proj(self.total_sigma_x)
                                    self.P_z = self.owm_update_proj(self.total_sigma_z)

                                    #starter for calc_cov(test_etc['ins'/'us','bs])
                                    
                                
                                    logging.info(f'...updated projection matrices for OWM')
                                    # reset activity collectors for next task 
                                    s_ins = []
                                    rec_xs = []
                                    rec_xs_post_ac = []
                                    z_outs = []
                            # if doing owm on reservoir with D2 ==0 
                                elif self.args.train_parts == ['M_u', 'M_ro']:
                                    self.args.seq_threshold += 0.08
                                    # 0th dimension is the batch element, 2nd dimension is number of timesteps
                                    # 1st dimension is the actual vector representation
                                    #compute current task covariances
                                
                                    # task activities 
                                    s_ins = torch.cat(s_ins, dim=0)
                                    us = torch.cat(us, dim=0)
                                    rec_xs = torch.cat(rec_xs, dim =0 )
                                    rec_xs_post_ac = torch.cat(rec_xs_post_ac, dim =0)
                                    z_outs = torch.cat(z_outs, dim=0)
                                    q_xs_ins =  torch.cat((rec_xs, us), dim =1)

                                    

                                    task_sigma_s = 0
                                    task_sigma_pre_act_u = 0
                                    task_sigma_q = 0
                                    task_sigma_x =0
                                    task_sigma_z= 0
                                    
                                    # 'number' of activities when testing network on this task: (batch_size * timesteps) - 
                                    activ_count = s_ins.shape[0] * s_ins.shape[2]
                                    
                                    for b in range(s_ins.shape[0]):
                                        for timestep in range(s_ins.shape[2]):
                                            with torch.no_grad():
                                                task_sigma_s += torch.outer(s_ins[b, :, timestep],s_ins[b, :, timestep]) 
                                                task_sigma_pre_act_u += torch.outer(self.net.M_u(s_ins[b, :, timestep]), self.net.M_u(s_ins[b, :, timestep]))
                                                task_sigma_q += torch.outer(q_xs_ins[b, :, timestep], q_xs_ins[b, :, timestep] )
                                                task_sigma_x += torch.outer(rec_xs_post_ac[b, :, timestep], rec_xs_post_ac[b, :, timestep]) 
                                                task_sigma_z += torch.outer(z_outs[b, :, timestep], z_outs[b, :, timestep] ) 
                                        
                                    # task wq calculated using task_sigma_q 
                                    j_end_of_task = self.net.reservoir.J.weight.detach().clone()
                                    w_u_end_of_task = self.net.reservoir.W_u.weight.detach().clone()
                                

                                    W = torch.cat((j_end_of_task, w_u_end_of_task), dim =1)
                                    
                                    
                                    #update total covariances - refactor as function later
                                    k = self.train_idx +1
                                    self.total_sigma_s=  ((k-1)/k) * self.total_sigma_s+ (1/k) * task_sigma_s
                                    self.total_sigma_pre_act_u =  ((k-1)/k) * self.total_sigma_pre_act_u + (1/k) * task_sigma_pre_act_u
                                    self.total_sigma_x=  ((k-1)/k) * self.total_sigma_x + (1/k) * task_sigma_x
                                    self.total_sigma_q =  ((k-1)/k) * self.total_sigma_q + (1/k) * task_sigma_q
                                    self.total_sigma_wq =  W @ self.total_sigma_q @ torch.transpose(W, dim0=0, dim1=1) 
                                    self.total_sigma_z =  ((k-1)/k) * self.total_sigma_z + (1/k) * task_sigma_z

                                    
                                    #update projections  - refactor as function later 
                                    self.P_s = torch.inverse( (1/self.args.alpha_owm) * self.total_sigma_s + torch.eye((self.total_sigma_s.shape[0])) )
                                    self.P_u = torch.inverse( (1/self.args.alpha_owm) * self.total_sigma_pre_act_u + torch.eye((self.total_sigma_pre_act_u.shape[0])) )
                                    self.P_q = torch.inverse( (1/self.args.alpha_owm) * self.total_sigma_q + torch.eye((self.total_sigma_q.shape[0])) )
                                    self.P_wq = torch.inverse( (1/self.args.alpha_owm) * self.total_sigma_wq + torch.eye((self.total_sigma_wq.shape[0])) )
                                    self.P_x = torch.inverse( (1/self.args.alpha_owm) * self.total_sigma_x + torch.eye((self.total_sigma_x.shape[0])) )
                                    self.P_z = torch.inverse( (1/self.args.alpha_owm) * self.total_sigma_z + torch.eye((self.total_sigma_z.shape[0])) )
                                    
                                    logging.info(f'...updated projection matrices for OWM')
                                    
                                    if self.args.orthog_reservoirs:
                                    #pdate reservoir weights
                                        pdb.set_trace()
                                        projected_reservoir_weights_j_w_u = self.P_wq @ W @ self.P_q
                                        
                                        
                                        self.net.reservoir.J.weight = nn.Parameter(self.net.reservoir.J.weight + projected_reservoir_weights_j_w_u[:, :300], requires_grad =True)
                                        self.net.reservoir.W_u.weight= nn.Parameter(self.net.reservoir.W_u.weight + projected_reservoir_weights_j_w_u[:, -self.net.reservoir.W_u.weight.shape[1]:], requires_grad =True)



                                    s_ins = []
                                    pre_act_us = []
                                    us=[]
                                    rec_xs = []
                                    rec_xs_post_ac = []
                                    z_outs = []
                                
                            else:
                                raise NotImplementedError('No implentation of OWM for this input configuration yet. D1 and D2 should be zero if training all parts, if u-layer reservoir, should be zero')
                            logging.info(f'...updated projection matrices for OWM')
                            
                        # done processing prior task, move on to the next one or quit
                        self.train_idx += 1

                            

                        if self.args.xdg or self.args.synaptic_intel or self.args.owm:
                            #reset optimizer and scheduler as Masse et al do.
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
        # for pca explained variances
        if self.args.pca_vars:
            x,y, trials = next(iter(self.test_loader))
            A = get_states(self.net, x)
            #pca multimodal with plot = False just returns the cumulative variances
            # note: n_reps needs to be the same as 'test size' for the test loader used to generate the A
            pca_vars = pca_multimodal(self.args, A_uncut=A, trials = trials, n_reps=50, plot=False)
            print(pca_vars)

            return running_min_error, ix, losses, pca_vars
        
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














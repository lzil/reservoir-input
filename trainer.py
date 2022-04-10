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
    #trainer object: trains  model and logs performance

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
        #will  automatically be cuda (a GPU) if 1) it's available and we specify --use_cuda see run.py arguments under control logging

        trains, tests = create_loaders(self.args.dataset, self.args, split_test=True, test_size=50)
        #take .dataset from self.args: it's a list containing the datsets we're going to use  and


        #trains is actually a tuple:(Dataset objects, DataLoader objects)
        #NB: for mutiple datasets(say when sequential training multiple tasks): if args.dataset is a list of datasets and create_loaders stores the corresponding list of training Datasets objects in the first tuple of trains [in the same order as their corresponding .pkl datasets in args.dataset] and the corresponding DataLoaders in the second tuple of trains 
        #similarly for tests

        



        #trains contains the training data and the dataloader separately
            #see "trains" in if statement directly below 

        #read Datasets and Dataloaders in PyTorch Guide!
        #creates the PyTorch data loaders so that we can iterate over datasets
        #args.dataset is the dataset at the (see is the path to the dataset we're using see dataset arguments in run.py 
        #n_trials is often 2000 by default, so we train on 1950 and test on 50
        #batch size is defined at run time
        #we've run it with 5 
        #what's args.dataset
        
        #create_loaders is from helpers and creates our data loaders [see PyTorch guide ] recall data_loader objects allows us to iterate over minibatches(nonoverlapping subsets of the dataset) easily 
        #how it actually creates the loaders is NB


        if self.args.sequential:
            #args.sequential is defined in run.py under dataset arguments
            #if true we do "sequential training"
            
            self.train_set, self.train_loaders = trains #trains is a tuple, first element of tuple is the whole training data set and the second element is the training loader: train_loader and the training data train_set
            

            self.test_set, self.test_loaders = tests
            self.train_idx = 0
            #index of the first task 

            

        
            #and then we set the current train_loader and test loaders to be the ones for the first task(dataset) 
            
            self.train_loader = self.train_loaders[self.args.train_order[self.train_idx]] 
            
            
             
            
            self.test_loader = self.test_loaders[self.args.train_order[self.train_idx]]

        else: #if not doing sequential training
            self.train_set, self.train_loader = trains
            self.test_set, self.test_loader = tests
        logging.info(f'Created data loaders using datasets:')
        #this is what shows in the terminal when its runs this part of the script! 
        for ds in self.args.dataset:
            logging.info(f'  {ds}') #the dataset that come after
            #the "Created data loaders using datasets" above
            #shows us the name of each dataset we're training on 
            #we see a list of all the datasets we're training on 


        if self.args.sequential:
            logging.info(f'Sequential training. Starting with task {self.train_idx}')
            
        # self.net = BasicNetwork(self.args)
        self.net = M2Net(self.args)
        #the model with the input representation layer, reservoir etc
        # add hopfield net patterns
        if hasattr(self.args, 'fixed_pts') and self.args.fixed_pts > 0:
            #syntax: hasattr(object, 'potential_attribute') if the object we input has attribute 'some_attribute' return true
            self.net.reservoir.add_fixed_points(self.args.fixed_pts)
        self.net.to(self.device) #?
        
        # print('resetting network')
        # self.net.reset(self.args.res_x_init, device=self.device)

        
        self.n_params = {}
        #dictionary of named parameters
        self.train_params = []
        # train_params is the list of the parameters we're going to train on 

        self.not_train_params = []
        logging.info('Training the following parameters:')

        #if xdg, train weights from context signal to hidden units so add M_u_cs, M_v_cs and M_x_cs to train_parts:

        if self.args.sequential and self.args.xdg:
            args.train_parts.append('M_u_cs')
            args.train_parts.append('M_x_cs')
            args.train_parts.append('M_v_cs')
            
        #print(self.net.named_parameters)
        #print(args.train_parts)
        for k,v in self.net.named_parameters():
            # k is name, v is weight
            #PT_syntax:
            #named_parameters is a built-in pytorch method
            #returns an iterator over module parameters[an interable storing pairs of the form (parameter_name, parameter)
            #note we have recurse=True by default [see documentation]

            found = False
            # filtering just for the parts that will be trained -we specified which parts in parameters.py
            #
            #for each parameter in named_parameters()
            #check to see if in train_parts or not
            for part in self.args.train_parts:
                #(train_parts is defined in parameters.py - but this isn't too NB)

                #takes the parameters in args.train_parts (the parts we say we want to train when we use run.py see run.py) and makes sure that these are in fact what we train on by adding them to self.train_params

                
                if part in k:
                    logging.info(f'  {k}')
                    self.n_params[k] = (v.shape, v.numel())
                    #tensor.numel returns total number of  elements in input tensor
                    #recall: n_params is a dictionary, key-value pairs
                    self.train_params.append(v)
                    found = True
                    #if the train part is in named parameters then add it to train_params
                    break


            if not found:
                #then this means parameter is not in args.train_parts so we specified not to train it
                self.not_train_params.append(k)
        logging.info('Not training:')
        for k in self.not_train_params:
            logging.info(f'  {k}')



        self.criteria = get_criteria(self.args)
        #what's in self.criteria --> get_criteria defined in helper.py
        #see the definition and explanation in helpers.py:
        #return the list of batch_losses where each batch loss is computed for a particular loss function for this task rsg-100-150
        #criterion as in loss for a particular loss functions - differnt loss fucntions <--> different criteria
        #initially returns a list of the batch_loss of a single batch on a particular loss function by appending the mse on this batch to this list 
        #then if we call it again, on another batch - we can use a different loss function - , it will append the loss on this new batch to the list that has the loss on first batch. So we get a list of losses on different batches.
        

        self.optimizer = get_optimizer(self.args, self.train_params)
        #choose an optimizer for the training that ensues(see "#training arguments " in run.py)
        #default is adam, 

        #and initializes an optimizer object with the arguments we specify at run time and 
        #the training parameters we want to optimize via train_params
        #we can also choose arguments for the optimizers. All of them have a learning rate 




        self.scheduler = get_scheduler(self.args, self.optimizer)
        #gets a scheduler: what is scheduler?
        #it's a method to adjust the learning rate based on different aspects of training/validation during training
    
        #see helpers.py for more

        
        self.log_interval = self.args.log_interval
        #log_interval is the length of the logging intervals fi.e. after how many training samplse samples we test our model and log its performance. It's 50 by default and this can be seen by the output in terminal. we log every 50 samples and do this until all the samples inthe dataset are trained on (for 1 epoch)
        

        if not self.args.no_log:
            #if we're logging: this is the code that populates the log during training!
            self.log = self.args.log
            #log is the log object defined in utils.py with info(stored in attributes) such as run_id, run_dir, the log_path, checkpoint directory
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
        
        #yes see: checkpoints explanation in utils.py
        if self.args.log_checkpoint_models:
            self.save_model_path = os.path.join(self.log.checkpoint_dir, f'model_{ix}.pth')
        elif os.path.exists(self.save_model_path):
            os.remove(self.save_model_path)
        torch.save(self.net.state_dict(), self.save_model_path)
        #this torch.save(neuralnet.state_dict())is what we use to save model's state at different points in traiing!, and we save it to the path self.save_model_path 

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
                #open the file and dump the samples in 

    # runs an iteration where we want to match a certain trajectory
    def run_trial(self, x, y, trial, training=True, extras=False, cs = None, gate_layers= None, train_idx=None,c_strength=None, damp_c=None, big_omega_M_u_weights = None, big_omega_M_ro_weights=None,big_omega_M_u_biases=None, big_omega_M_ro_biases= None, M_u_weights_prev=None,M_ro_weights_prev=None, M_u_biases_prev =None, M_ro_biases_prev=None):
        #this is for a single trial(a single task object i.e. a single training case) which we input in line above
        #trial a.k.a info is the task object a particular instantiation from the class definitions in tasks.py
        #it tells us which task we're doing which is NB because it tells us what the x and y present 
        self.net.reset(self.args.res_x_init, device=self.device)
        #resets the net
        trial_loss = 0.
        #the loss on this single trial
        #need trial loss bc it's an input output pair is a pair of sequences bc it's time series(so the loss at each time point in the trial )
        k_loss = 0. #k_loss bc we calculate the loss for the trial over the last k timesteps of the of the trial - k is the length of the truncation window counting backward from the last timestep of the trial 
        outs = []
        us = []
        vs = []
        # setting up k for t-BPTT
        #k defines the truncation window for BPTT so that we only unroll the network from T-k(inclusive) to T

        




        if training and self.args.k != 0:
            #the k we specify for BPTT, if k != 0, then do truncated bptt with k
            #by default code uses k = 0 and Liang said atm we don't truncate
            #so we use the next code in else statement

            
            k = self.args.k
        else:
            # we don't truncate atm (26 March 2022): so k is task length
            # is k =0 we put it k to full n means normal BPTT: in this case truncation window is the task length k is literally the k in the bptt formula in Bengio's On 
            #yes bc the BPTT update is performed back for n=t_len time steps (so there's no truncation)
            #if k=0 its default or training is False then use 
            k = x.shape[2] 
            #print(k)
            #x.shape[2] is the lenght of the input time-series,it's always t_len and so for the RSG it's 600
            
            
        for j in range(x.shape[2]):
            #x.shape[2] is T the length of the trial.

            #this looks to be the unrolling to compute the losses at each timestep j of the trial(training case)
            #print(f'Am I here?{x.shape}')
            #see get_criteria in goodnotes for an explanation of x's shape: in short x[:, ;, j] is the input to the net at timestep j, we do this for each timestep starting with the 1st and ending with the 600th
            
            net_in = x[:,:,j] # the single input x's value at time j  for j=1,..., T
            #print(f'this is x[:, :, j] {x[:,:,j]}')
            #the input at time j
            #print(f'what is net_in ? {net_in}')
            if self.args.sequential and self.args.xdg:
                net_out, etc = self.net(net_in, cs = cs, gate_layers= self.args.gate_layers, train_idx = self.train_idx, extras = True) 

            else:
                net_out, etc = self.net(net_in, extras = True)
                
            
            #see self.net above, it's the net we're training on 
            #recall we use nets like functions, we called the net on net in, and specified extras=True - this does a forward pass of our net on the net_in (at time j only, yes I think the idea at play here is that of the unrolling the recurrent net in time so in a single time series trial, we treat each input output pair in the time series as a single trial in a feedforward setting and we unroll the dynamics net in time so we have this unrolled feedforward net) (i.e. in def forward , o = net_in),
            #print(f'this is net out {net_out}')



            #see ipad goodnotes get_criteria:
            #net_out for x[:,;,j] is the z(j), we can think of j as a timestep here, 
            #etc stores the values of u where u = self.m1_act(self.M_u(o))
            #and v, etc = self.reservoir(u, extras=True)
            #etc holds here holds the x-states(the recurrently connected units) after this time step i.e x(j)
            # I believe this is the unrolling of the network
            outs.append(net_out) #add th the z(t) at this time step t=j to this list
            #we want a list of all the z(t)'s during the task for each timestep for backprop

            us.append(etc['u']) #store the u for this time step j in us
            vs.append(etc['v']) #store v for this time step k in vs
            #then 
            # t-BPTT with parameter k
            #t stands for truncated, as in truncated BPTT i.e. do BPTT but using only the last k timesteps/;
            #see our notes on BPTT, tBPTT, exploding and vanishing gradients

            # so we keep evaluating our network at each "time" index j=0,....,(task_len-1) 
            #and when we get to the beginning of the truncation window(i.e. when (j+1)%k==0 we start on 
            #t BPTT
            # we start BPTT at j+1==k index timestep in task length i.e. when index is k-1 so that the timestep is k which is ofc by def when we want start BPTT if k
            if (j+1) % k == 0: 
                
                #explanation when not truncating i.e. k=600:
                #note 50 % 600 = 50 
               #therefore this clause will only be triggered the first time j+1=600
               #at which point we've got all the model outputs for the inputs and we stored them in k_outs
                
                
                k_outs = torch.stack(outs[-k:], dim=2) 
                #take the last k outputs from outs concatenates them a long the dimension 2 which is 
                #(see:https://pytorch.org/docs/stable/generated/torch.stack.html)


                
                #we select the last k elements of outs
                #torch.stack concatenates these tensors along the 3rd dimension (dim=2), just as we did with x.shape[2], this 3rd dimension (with index 2) is the time dimension (see ipad), so we have z(t)'s evolution over time the time steps 
                
                k_targets = y[:,:,j+1-k:j+1] 
                # we take the last k target values, if no truncation this is the whole target trajectory for the trial 
                
                #if RSG: k is 600, dpro its 300
                #recall y stores our targets, again 3rd dimenision is the time dimension and we want the targets from time step indices j+1 -k (inclusive) to j+1 (exclusive) i.e. from 0 to k i.e. from 1st time step to kth timestep exclusive (k-1th timestep inclusive)
                #but why not just take the last k targets?

                #new hypothesis 13:10, 11 March:
                #yes we want the last k outputs the outputs in the truncation window but 
                #in order to calculate the loss at the first time step in the truncation window, we need the error 
                for c in self.criteria:
                    k_loss += c(k_outs, k_targets, i=trial, t_ix=j+1-k)
                    

                    if self.args.ss and training:
                        #loss function for either SI or EWC
                        #note we use the ss and xdg loss for training but not testing - if you use it for testing you'll get a growing loss
                        if self.args.ff_bias: #add bbias terms


                            k_loss += c_strength*torch.sum(big_omega_M_u_weights* torch.square(self.net.M_u - M_u_prev)) + c_strength *torch.sum(big_omega_M_ro* torch.square(self.net.M_ro - M_ro_prev))

                        else: 

                            print(c_strength*torch.sum(big_omega_M_u_weights * torch.square(self.net.M_u.weight - M_u_weights_prev)) + c_strength *torch.sum(big_omega_M_ro_weights* torch.square(self.net.M_ro.weight - M_ro_weights_prev)))
                            k_loss +=  c_strength*torch.sum(big_omega_M_u_weights * torch.square(self.net.M_u.weight - M_u_weights_prev)) + c_strength *torch.sum(big_omega_M_ro_weights* torch.square(self.net.M_ro.weight - M_ro_weights_prev))
                            #print(k_loss)
                            # print('no biases just weights')
                            # pdb.set_trace()

                        


        
                        #k_loss += c(k_outs, k_targets, i=trial, t_ix=j+1-k) #if no truncation j+1-k in here is going to be 0 (because we're nested in if j+1 % k ==0)so for '+=' is really just an = (bc its zero initially and then we only add something to it once if no truncation) and now in ss implementation we can just add the quadratic penalyt term 
                        #print(k_loss)
                        # pdb.set_trace()

                    #if we specified mse in args.loss at runtime, then c will only take one value and that will be to calculate the mse for this single trial here between k_outs and k_targets
                    
                    #this will give us the mse and we want t_ix, the truncate index here to be zero bc we've already truncated and we want the mse between each k_outs, k targets

                trial_loss += k_loss.detach().item()
                #k_loss.detach() returns a new tensor, detached from the current graph. The result will never require gradients
                #computation graphs with andrew ng might be important
                #why do we want this: because again we're doing something to the loss function that isn't part of the mathematical definition of the network and so we don't want this operation to be part of the computational graph through which we backprop (see colah's blog on backprop)
                 
                if training:
                    
                    k_loss.backward()  #calculates gradients
                    
                    # strategies for continual learning that involve modifying gradients we computed in line above
                    if self.args.sequential and self.train_idx > 0:
                        #NB: self.train_idx >0 i.e if we're not on training on the first task 
                        if self.args.owm:
                            # orthogonal weight modification
                            self.net.M_u.weight.grad = self.P_u @ self.net.M_u.weight.grad @ self.P_s
                            self.net.M_ro.weight.grad = self.P_z @ self.net.M_ro.weight.grad @ self.P_v
                            if self.args.ff_bias:
                                self.net.M_u.bias.grad = self.P_u @ self.net.M_u.bias.grad
                                self.net.M_ro.bias.grad = self.P_z @ self.net.M_ro.bias.grad
                        elif self.args.swt:
                            # keeping sensory and output weights constant after learning first task
                            #this is important we need this for implementation
                            self.net.M_u.weight.grad[:,:self.args.L] = 0
                            self.net.M_ro.weight.grad[:] = 0
                            if self.args.ff_bias:
                                self.net.M_u.bias.grad[:] = 0
                                self.net.M_ro.bias.grad[:] = 0
                            
                k_loss = 0.
                self.net.reservoir.x = self.net.reservoir.x.detach() #why do we do detach this

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

    def train_iteration(self, x, y, trial, ix_callback=None,  cs=None, gate_layers=None, train_idx =None, damp_c=None, c_strength=None, big_omega_M_u_weights = None, big_omega_M_ro_weights=None, big_omega_M_u_biases=None, big_omega_M_ro_biases= None, M_u_weights_prev=None, M_ro_weights_prev=None, M_u_biases_prev =None, M_ro_biases_prev=None):
        self.optimizer.zero_grad()#put back to zero the gradients bc we want to use the new ones for the trial

        #if self.args.sequential and self.args.ss and self.args.xdg:
            #and then elif for the statement below

        if self.args.sequential and self.args.xdg and self.args.ss:
            trial_loss, etc = self.run_trial(x, y, trial, extras=True, cs=cs, gate_layers=self.args.gate_layers, train_idx= self.train_idx, c_strength=c_strength, big_omega_M_u_weights = big_omega_M_u_weights, big_omega_M_ro_weights=big_omega_M_ro_weights, big_omega_M_u_biases=big_omega_M_u_biases, big_omega_M_ro_biases= big_omega_M_ro_biases, M_u_weights_prev=M_u_weights_prev, M_ro_weights_prev=M_ro_weights_prev, M_u_biases_prev =M_u_biases_prev, M_ro_biases_prev=M_ro_biases_prev)


        elif self.args.sequential and self.args.xdg:

            trial_loss, etc = self.run_trial(x, y, trial, extras=True, cs=cs, gate_layers=self.args.gate_layers, train_idx= self.train_idx)

        

        else:
            #run trial computes trial loss and the gradients for a single trial 
            trial_loss, etc = self.run_trial(x, y, trial, extras=True, cs=cs, gate_layers=self.args.gate_layers)
        
        #computes trial loss and the gradients for the updates


        if ix_callback is not None:
            ix_callback(trial_loss, etc)
        self.optimizer.step() #updates the parameters using the gradients that we computed using self.run_trial directly above

        etc = {
            'ins': x,
            'goals': y,
            'us': etc['us'].detach(),
            'vs': etc['vs'].detach(),
            'outs': etc['outs'].detach()
        }
        return trial_loss, etc

    def test(self, cs=None, gate_layers= None, train_idx = None, c_strength=None, big_omega_M_u_weights = None, big_omega_M_ro_weights=None, big_omega_M_u_biases=None, big_omega_M_ro_biases= None, M_u_weights_prev=None, M_ro_weights_prev=None, M_u_biases_prev =None, M_ro_biases_prev=None):
        with torch.no_grad():
            #note since we're in a no_grad, make sure that you're not running code in the environment that's going to try to compute gradients here bc no gradients are being 'recorded'
            x, y, trials = next(iter(self.test_loader))
            #print(f'these are the trials: {trials}')

            x, y = x.to(self.device), y.to(self.device)

            #if self.args.sequential and self.args.ss and self.args.xdg:
            #and then elif for the statement below

            #if self.args.sequential and self.args.xdg and self.args.ss:
            

            if self.args.sequential and self.args.xdg:
                loss, etc = self.run_trial(x, y, trials, training=False, cs =cs, gate_layers= self.args.gate_layers, train_idx = self.train_idx ,extras=True)

            else:
                loss, etc = self.run_trial(x, y, trials, training=False, cs =cs, extras=True) 


        etc = {
            'ins': x,
            'goals': y,
            'us': etc['us'].detach(),
            'vs': etc['vs'].detach(),
            'outs': etc['outs'].detach()
        }
        

        return loss, etc

    # helper function for sequential training, for testing performance on all tasks
    def test_tasks(self, ids, cs=None, train_idx= None):
        #computes the test losses for each task and appends them to a list losses which will be as long as the number of tasks.
        #the each element of the list is a tuple (task integer id, loss for that task)
        #ids is a list of the task ids(just integers) [0,1,2,.., r-1]  if r tasks of the tasks we're training on 
        losses = []
        for i in ids:
            self.test_loader = self.test_loaders[self.args.train_order[i]]

            #if self.args.sequential and self.args.ss and self.args.xdg:
            #and then elif for the statement below

            if self.args.sequential and self.args.xdg:
                loss, _ = self.test(cs=cs, gate_layers=self.args.gate_layers, train_idx= self.args.train_idx)
            else:
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
        #this is where we actually train using train-iteration 
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

        # for SS (running synaptic importances)
        if self.args.ss:
            
            c_strength = self.args.c_strength
            #instantiate running cumulative importances of parameters across tasks 
            #(notice all the 'running' stuff above :) )

            
            
            
            #for the weights
            big_omega_M_u_weights = torch.zeros_like(self.net.M_u.weight.data)

            big_omega_M_ro_weights = torch.zeros_like(self.net.M_ro.weight.data)
            #for the biases if any:
            if self.args.ff_bias:
                big_omega_M_u_biases= torch.zeros_like(self.net.M_u.bias.data)
                big_omega_M_ro_biases = torch.zeros_like(self.net.M_ro.bias.data)


            #then we're going to pass these into the loss functions throuh train_iteration below and crucially these big omegas will be 0 for the first task so we can use the loss function with the quadratic penalty term from the get go.
                        
            #different ways of calculating importance of paramaters for different tasks
            if self.args.ss_type == 'SI':
                damp_c = self.args.C
                
                #initialise weights prev:
                M_u_weights_prev = torch.zeros_like(self.net.M_u.weight.data)
                M_ro_weights_prev= torch.zeros_like(self.net.M_ro.weight.data)
                if self.args.ff_bias:
                    M_u_biases_prev = torch.zeros_like(self.net.M_u.bias.data)
                    M_ro_biases_prev = torch.zeros_like(self.net.M_ro.bias.data)



                #initialize delta theta_i's and small omegas
                delta_M_u_weights = 0
                delta_M_ro_weights = 0

                w_M_u_weights = 0 
                w_M_ro_weights = 0
                


                if self.args.ff_bias:
                    delta_M_u_biases = 0
                    delta_M_ro_biases = 0
                    if self.args.ff_bias:
                        w_M_u_biases = 0
                        w_M_ro_biases = 0


            #elif self.args.ss_type == 'EWC':

                



        for e in range(self.args.n_epochs):
            #
            # synaptic intelligence requires storing changes in parameter values after iterations, need the value of parameters before training (i.e. their initializations)

    
            for epoch_idx, (x, y, info) in enumerate(self.train_loader):
                #print(f'x in train_loader {x,y}, {info}')
                #bc batch_size is 1: #but we can make it 5 - batch_size is the the number of L_i's (see RNNs notes in ipad) we add up to get the batch loss which we pass to our optimizer.
                #if batch_size is 1: elements of train loader is a list of batches of size 1 i.e. each (x,y, info) is a single input-output pair
                # the batch is (x,y,info) and we number them as we go 
                
                ix += 1

                #a training iteration (see 'RNNs'notes on ipad, page 2 and look for Training and Losses in reservoir-input)
                x, y = x.to(self.device), y.to(self.device)
                
                #storing changes in parameter values after iterations, need the initial value 


                

                # if doing xdg, create intialize and context signal - one hot vector with 1 in first entry, 0 in the rest bc it's for the first task.
                #for now the context signal is only used when doing xdg 
                #NB: cs has to be a parameter for any function that uses the forward pass of the network bc if doing xdg (args.xdg), in network.py cs becomes a parameter for the net that needs to be not None. 

                if self.args.sequential and self.args.xdg:
                    #context signal 
                    cs = torch.zeros((1, self.args.T))
                    cs[0, self.train_idx] = 1
                    #context dependent gating 
                    #pool of units to be gated
                    #print(self.args.gate_layers)
                    #args.gate_layers is assigned ['u', 'v'] by default:
                    gate_layers=self.args.gate_layers

                    # number of units to gate (selected randomly) so that only (1-X)% of units are active for any task 
                    #m=args.*len(gate_pool)
                else:
                    #this line isn't really necessary because cs will be None by default in all the functions.
                    cs=None


                if self.args.sequential and self.args.xdg and self.args.ss:
                    if self.args.ss_type == 'SI':
                        #weights before training step so that we can calculate the change in weights brought about by a single train_iteration ( the t variable for number of batches in Masse paper methods)
                        with torch.no_grad():
                            M_u_bf_weights = self.net.M_u.weight.data #theta(t-1 in Masse paper)
                            M_ro_bf_weights = self.net.M_ro.weight.data
                            if self.args.ff_bias:
                                M_u_bf_biases = self.net.M_u.weight.data
                                M_ro_bf_biases = self.net.M_u.weight.data

                        if self.args.ff_bias:
                            iter_loss, etc = self.train_iteration(x, y, info, ix_callback=ix_callback, cs=cs, gate_layers= self.args.gate_layers, c_strength = c_strength, big_omega_M_u_weights = big_omega_M_u_weights, big_omega_M_ro_weights=big_omega_M_ro_weights,big_omega_M_u_biases=big_omega_M_u_biases, big_omega_M_ro_biases= big_omega_M_ro_biases, M_u_weights_prev=M_u_weights_prev, M_ro_weights_prev=M_ro_weights_prev, M_u_biases_prev =M_u_biases_prev, M_ro_biases_prev=M_ro_biases_prev)

                            with torch.no_grad():
                                delta_M_u_weights += (self.net.M_u.weight.data - M_u_bf_weights)
                                delta_M_ro_weights += (self.net.M_ro.weight.data - M_ro_bf_weights)
                                w_M_u_weights += (self.net.M_u.weight.data - M_u_bf_weights) * (-self.net.M_u.weight.grad)
                                w_M_ro_weights += (self.net.M_ro.weight.data - M_ro_bf_weights) * (-self.net.M_ro.weight.grad)
                                #reset weights before training step after calculating 
                                M_u_bf_weights = 0
                                M_ro_bf_weights = 0 

                                if self.args.ff_bias:
                                    delta_M_u_biases+= (self.net.M_u.bias.data - M_u_bf_biases)
                                    delta_M_ro_biases+= (self.net.M_ro.bias.data - M_u_ro_biases)
                                    w_M_u_biases += (self.net.M_u.bias.data - M_u_bf_biases) * (-self.net.M_u.bias.grad)
                                    w_M_ro_biases += (self.net.M_ro.bias.data - M_u_ro_biases) * (-self.net.M_ro.weight.grad)
                                    #reset weights before training step after calculating 
                                    M_u_bf_biases = 0
                                    M_ro_bf_biases = 0 


                        
                        else:
                            iter_loss, etc = self.train_iteration(x, y, info, ix_callback=ix_callback, cs=cs, gate_layers= self.args.gate_layers, c_strength=c_strength,big_omega_M_u_weights = big_omega_M_u_weights, big_omega_M_ro_weights=big_omega_M_ro_weights,M_u_weights_prev=M_u_weights_prev, M_ro_weights_prev=M_ro_weights_prev)



                            with torch.no_grad():
                                delta_M_u_weights += (self.net.M_u.weight.data - M_u_bf_weights)
                                delta_M_ro_weights += (self.net.M_ro.weight.data - M_ro_bf_weights)
                                w_M_u_weights += (self.net.M_u.weight.data - M_u_bf_weights) * (-self.net.M_u.weight.grad)
                                w_M_ro_weights += (self.net.M_ro.weight.data - M_ro_bf_weights) * (-self.net.M_ro.weight.grad)
                                #reset weights before training step after calculating 
                                M_u_bf_weights = 0
                                M_ro_bf_weights = 0 

                                

                                
                #if doing  ss but no Xdg
                elif self.args.sequential and self.args.ss:
                    if self.args.SI:
                        print(f'implementing synaptic intelligence')
                        with torch.no_grad():
                            M_u_bf_weights = self.net.M_u.weight.data
                            M_ro_bf = self.net.M_u.weight.data

                        iter_loss, etc = self.train_iteration(x, y, info, ix_callback=ix_callback, big_omega_M_u = big_omega_M_u, big_omega_M_ro=big_omega_M_ro)

                        #calculate the w^k_i's for task k while training task k 
                        #we treat the w's for M_ro and the ws for M_u separately for now

                        with torch.no_grad():

                            delta_M_u += (self.net.M_u.weight.data - M_u_bf)
                            delta_M_ro += (self.net.M_ro.weight.data - M_u_ro) 
                            w_M_u += (self.net.M_u.weight.data - M_u_bf) * (-self.net.M_u.weight.grad)
                            w_M_ro += (self.net.M_ro.weight.data - M_u_ro) * (-self.net.M_ro.weight.grad)
                            #reset weights before training step after calculating adding to little omegas
                            M_u_bf = 0
                            M_ro_bf = 0 


                else:
                    iter_loss, etc = self.train_iteration(x, y, info, ix_callback=ix_callback, cs=cs, gate_layers= self.args.gate_layers)
                #print(self.net.M_u.weight-10*torch.ones(50,2)) 
                #great can treat these .weight objects like matrices
                #for synaptic intelligence need the differences in M_u and M_v  before and after each train iteration [so we're going to want to get the parameter values before the self.train_iteration above]
                #print(self.net.M_u.weight.shape)


                if iter_loss == -1:
                    logging.info(f'iteration {ix}: is nan. ending')
                    ending = True
                    break

                running_loss += iter_loss 

                if ix % self.log_interval == 0: 
                    #if it's time to log or stop i.e. if ix the index of the training iteration we're on  is a multiple of the log interval

                    z = etc['outs'].cpu().numpy().squeeze()
                    train_loss = running_loss / self.log_interval
                    #if self.args.sequential and self.args.ss and self.args.xdg:
                        #if self.args.ff_bias:
                            #test_loss, test_etc = self.test(cs=cs, gate_layers= self.args.gate_layers, c_strength = c_strength, big_omega_M_u_weights = big_omega_M_u_weights, big_omega_M_ro_weights=big_omega_M_ro_weights,big_omega_M_u_biases=big_omega_M_u_biases, big_omega_M_ro_biases= big_omega_M_ro_biases, M_u_weights_prev=M_u_weights_prev, M_ro_weights_prev=M_ro_weights_prev, M_u_biases_prev =M_u_biases_prev, M_ro_biases_prev=M_ro_biases_prev)
                        #else:
                            #test_loss, test_etc = self.test(cs=cs, gate_layers= self.args.gate_layers, c_strength=c_strength,big_omega_M_u_weights = big_omega_M_u_weights, big_omega_M_ro_weights=big_omega_M_ro_weights,M_u_weights_prev=M_u_weights_prev, M_ro_weights_prev=M_ro_weights_prev)

                    if self.args.sequential and self.args.xdg:
                        test_loss, test_etc = self.test(cs =cs, gate_layers= self.args.gate_layers, train_idx = self.train_idx)

                    else:
                        test_loss, test_etc = self.test()


                    log_arr = [
                        f'*{ix}',
                        f'train {train_loss:.3f}',
                        f'test {test_loss:.3f}'
                    ]
                    if self.args.sequential:
                        losses = self.test_tasks(ids=range(self.train_idx))
                        #test_tasks is a # helper function for sequential training, for testing performance on all tasks
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
                        #args.seq_threshold 
                        #when training sequentially we move onto next tasks as soon as test_loss is less than args.seq_threshold which has default value 5   
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

                        #synaptic stabilization:

                        
                        # done processing prior task, move on to the next one or quit
                        self.train_idx += 1 #this is us moving onto the next task

                        # if doing XdG, update context signal
                        if self.args.xdg:
                            cs = torch.zeros((1, args.T))
                            cs[0, self.train_idx] = 1

                        #if doing ss, update loss new loss function for new tasks
                        if self.args.ss:
                            if self.args.ss_type == 'SI' :
                                # Omegas (importances) for the task we just finished training on:
                                with torch.no_grad():
                                    omega_M_u_weights= torch.maximum(torch.zeros_like(self.net.M_u.weight.data), torch.div(w_M_u, delta_M_u + damp_c*torch.ones_like(self.net.M_u.weight.data)))
                                    omega_M_ro_weights = torch.maximum(torch.zeros_like(self.net.M_ro.weight.data), torch.div(w_M_ro, delta_M_ro + damp_c*torch.ones_like(self.net.M_ro.weight.data)))
                                    if self.args.ff_bias:
                                        omega_M_u_biases = torch.maximum(torch.zeros_like(self.net.M_u.bias.data), torch.div(w_M_u, delta_M_u + damp_c*torch.ones_like(self.net.M_u.bias.data)))
                                        omega_M_ro_biases = torch.maximum(torch.zeros_like(self.net.M_ro.bias.data), torch.div(w_M_ro, delta_M_ro + damp_c*torch.ones_like(self.net.M_ro.bias.data)))

                            #add the omegas for recently completed task above to cumulative importance and then zero them for the next task
                                
                                    big_omega_M_u_weights += omega_M_u_weights
                                    big_omega_M_ro_weights += omega_M_ro_weights
                                    omega_M_u_weights = 0 
                                    omega_M_ro_weights = 0

                                    # for the loss function in next task we want the theta_prevs which are the parameter values at the end of the training on task k
                                    M_u_weights_prev = self.net.M_u.weight.data
                                    M_ro_weights_prev = self.net.M_ro.weight.data
                                
                                    #zero out small omegas and delta omegas before starting the next task:
                                    delta_M_u_weights= 0
                                    delta_M_ro_weights= 0
                                    w_M_u_weights = 0
                                    w_M_ro_weights = 0

                                    if self.args.ff_bias:
                                        big_omega_M_u_biases += omega_M_u_biases
                                        big_omega_M_ro_biases += omega_M_ro_biases
                                        omega_M_u_biases = 0 
                                        omega_M_ro_biases = 0

                                        M_u_biases_prev = self.net.M_u.bias.data
                                        M_ro_biases_prev = self.net.M_ro.bias.data
                                        delta_M_u_biases= 0
                                        delta_M_ro_biases= 0
                                        w_M_u_biases = 0
                                        w_M_ro_biases = 0
                                
                                        #use the quadratic penalty for the first task but we'll just have the big omega's equal to 0 bc we'll instantiate them as suhc    
                                
                                        # add to cumulative importance and then use cumulative importance for next loss function

                            


                        if self.train_idx == len(self.args.train_order):
                            #i.e. if we've done the last task then we'll jump out of the for loop by breaking so we don't do the things below in the for statement (as to avoid unecessary computations)
                            ending = True
                            logging.info(f'...done training all tasks! ending')
                            break
                            #break terminates the current loop and resumes executiona at the next statement after the current loop. So this will bust us out of "for epoch_idx, (x, y, info) in enumerate(self.train_loader):" , but we'll still be in "for e in range(self.args.n_epochs):"

                        
                        logging.info(f'...moving on to task {self.train_idx}.')
                        self.train_loader = self.train_loaders[self.args.train_order[self.train_idx]]
                        #new train loader for new task 
                        self.test_loader = self.test_loaders
                        [self.args.train_order[self.train_idx]]
                        #test loader for new task
                        running_min_error = float('inf')
                        running_no_min = 0
                        break
                        #is break going to take us to the 

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
            logging.info(f'Finished dataset epoch {e+1}') #e+1 bc zero-indexing in python and we want to print the epoch 1 not epoch 0 !
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

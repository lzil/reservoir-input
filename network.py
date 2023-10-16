import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import pdb
import random
import copy
import sys

from utils import Bunch, load_rb, update_args
from helpers import get_activation

# for easy rng manipulation
class TorchSeed:
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.rng_pt = torch.get_rng_state()
        torch.manual_seed(self.seed)
    def __exit__(self, type, value, traceback):
        torch.set_rng_state(self.rng_pt)


DEFAULT_ARGS = {
    'L': 2,
    'D1': 5,
    'D2': 5,
    'N': 50,
    'Z': 2,

    'use_reservoir': True,
    'res_init_g': 1.5,
    'res_burn_steps': 200,
    'res_noise': 0,

    'ff_bias': True,
    'res_bias': False,

    'm1_act': 'none',
    'm2_act': 'none',
    'out_act': 'none',
    'model_path': None,
    'res_path': None,

    'network_seed': None,
    'res_seed': None,
    'res_x_seed': None
}

class M2Net(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        self.args = update_args(DEFAULT_ARGS, args)
       
        if self.args.network_seed is None:
            self.args.network_seed = random.randrange(1e6)
        
        self.out_act = get_activation(self.args.out_act)
        if  'sp-bce' in self.args.loss:
            self.out_act = nn.Sigmoid()
        
        
        self.m1_act = get_activation(self.args.m1_act)
        self.m2_act = get_activation(self.args.m2_act)

        
        
        if self.args.ncl:
            self.grad_u = 0

        self._init_vars()
        self.reset()


    def _init_vars(self):
        with TorchSeed(self.args.network_seed):
            D1 = self.args.D1 if self.args.D1 != 0 else self.args.N
            D2 = self.args.D2 if self.args.D2 != 0 else self.args.N
            # net feedback into input layer
            if hasattr(self.args, 'net_fb') and self.args.net_fb:
                self.M_u = nn.Linear(self.args.L + self.args.T + self.args.Z, D1, bias=self.args.ff_bias)
            else:
                self.M_u = nn.Linear(self.args.L + self.args.T, D1, bias=self.args.ff_bias)
            self.M_ro = nn.Linear(D2, self.args.Z, bias=self.args.ff_bias)
        self.reservoir = M2Reservoir(self.args)

        # load params for reservoir if they exist
        if self.args.M_path is not None:
            M_params = torch.load(self.args.M_path)
            # TODO load M_params
        if self.args.model_path is not None:
            self.load_state_dict(torch.load(self.args.model_path))
        
        if self.args.xdg:
            #save gates used for each task so that we can use them when testing on previously learnt tasks
            #task gates is a dictionary whose keys are the train_idx/context and the values contain a dictionary of the gates applied for that task whose keys are the layer names
            self.task_gates_by_contexts = {}
            #for each task create a set of gates for each layer a priori so that we can test the training
            for i in range(self.args.T):
                with torch.no_grad():
                    task_gates= {}
                
                    if 'u' in self.args.gate_layers:
                        
                        u_mask_template= torch.ones((self.args.D1))
                        #check that this mask and u in the forward pass are the same size also check to see if u varies with batch size, I think it does

                        #number of units in u layer to be gated
                        u_gate_count= (self.args.X / 100) * self.args.D1
                        #round to nearest integer
                        u_gate_count = round(u_gate_count)
                        #list of indices for:
                        u_indices = [ i for i in range(self.args.D1)]
                        #check that this is a list of indices
                        u_gate_indices = random.sample(u_indices, u_gate_count)
                        #gate
                        u_mask_template[u_gate_indices] = 0
                        u_mask_template= u_mask_template.reshape((1,self.args.D1))
                        task_gates['u']=u_mask_template
                        
                    else: 
                        u_mask_template = None

                
                    if 'v' in self.args.gate_layers:
                        v_size = self.args.D2
                        v_mask_template =torch.ones((v_size))

                        v_gate_count = (self.args.X / 100) * self.args.D2
                        v_gate_count = round(v_gate_count)
                        v_indices = [ i for i in range(v_size)]
                        v_gate_indices = random.sample(v_indices, v_gate_count)
                        v_mask_template[v_gate_indices] = 0
                        #check size with v 
                        v_mask_template = v_mask_template.reshape((1, self.args.D2))
                        task_gates['v']= v_mask_template

                    else:
                        v_mask_template = None

                    if 'x' in self.args.gate_layers:
                        x_mask_template = torch.ones(self.args.N)
                        x_gate_count = (self.args.X / 100) * self.args.N
                        x_gate_count = round(x_gate_count)
                        x_indices = [i for i in range(self.args.N)]
                        x_gate_indices = random.sample(x_indices, x_gate_count)
                        x_mask_template[x_gate_indices] = 0
                        x_mask_template = x_mask_template.reshape((1,self.args.N))
                        task_gates['x'] = x_mask_template 
                    else:
                        self.x_mask_template = None

                    self.task_gates_by_contexts[i] = task_gates

            
            
                          
    

    #resize gates to train batch size: call on net before you train with xdg 
    def xdg_train_mode(self, train_idx=0):
        batch_size = self.args.batch_size
        with torch.no_grad():
            if 'u' in self.args.gate_layers:
                self.u_mask_template = self.task_gates_by_contexts[train_idx]['u']
                self.u_mask = self.u_mask_template.repeat(batch_size,1)
            #assign to the gate that's used in forward pass
            if 'v' in self.args.gate_layers:
                self.v_mask_template = self.task_gates_by_contexts[train_idx]['v']
                self.v_mask = self.v_mask_template.repeat(batch_size,1)
            if 'x' in self.args.gate_layers:
                self.x_mask_template = self.task_gates_by_contexts[train_idx]['x']
                self.x_mask = self.x_mask_template.repeat(batch_size,1)

    #use just before you test with xdg; ensures the right gate is used when testing the network
    def xdg_test_mode(self, current = True, train_idx = None, batch_size = 50, same_context= True, contexts = None):
        
        #this test batch size needs to be equal to the test size. It's 50 by default.
       
        #if testing on the task we're currently training on(i.e. if current), then we don't have to access and use old gates; just use current templates(gates)
        #if not current access task specific mask using train_idx and use this for testing
        #same context: True if same context for each batch, False if different contexts for each batch
        #contexts is a list containing the contexts for each batch
        
        with torch.no_grad():
            if same_context:
                if 'u' in self.args.gate_layers:
                    if current:
                        self.u_mask = self.u_mask_template.repeat(batch_size,1)

                    else:
                        self.u_mask_template = self.task_gates_by_contexts[train_idx]['u']
                        self.u_mask = self.u_mask_template.repeat(batch_size,1)
                    
                if 'v' in self.args.gate_layers:
                    if current:
                        self.v_mask = self.v_mask_template.repeat(batch_size,1)
                    else:
                        self.v_mask_template = self.task_gates_by_contexts[train_idx]['v']
                        self.v_mask = self.v_mask_template.repeat(batch_size,1)
                    
                if 'x' in self.args.gate_layers:
                    if current:
                        self.x_mask = self.x_mask_template.repeat(batch_size,1)
                    else:
                        self.x_mask_template = self.task_gates_by_contexts[train_idx]['x']
                        self.x_mask = self.x_mask_template.repeat(batch_size,1)
                
            else:
                u_masks = [] 
                v_masks = []
                x_masks = []
                
                for context in contexts:

                    if 'u' in self.args.gate_layers:
                        u_masks.append(self.task_gates_by_contexts[context]['u'])
                    if 'v' in self.args.gate_layers:
                        v_masks.append(self.task_gates_by_contexts[context]['v'])
                    if 'x' in self.args.gate_layers:
                        x_masks.append(self.task_gates_by_contexts[context]['x'])
                
                if 'u' in self.args.gate_layers:
                    self.u_mask = torch.squeeze(torch.stack(u_masks))
                if 'v' in self.args.gate_layers:
                    self.v_mask = torch.squeeze(torch.stack(v_masks))
                if 'x' in self.args.gate_layers:
                    self.x_mask = torch.squeeze(torch.stack(x_masks))

    def add_task(self):
        M = self.M_u.weight.data
        self.M_u.weight.data = torch.cat((M, torch.zeros((M.shape[0],1))), dim=1)
        self.args.T += 1


    #hooks for ncl activity gradients
    def save_u_grad(self,grad):
        self.grad_u = grad.detach().clone()
    
    

    def forward(self, o, extras=False, ncl_fish_estim=False, node_pert_noises=None):
        # pass through the forward part
        # o should have shape [batch size, self.args.T + self.args.L]
        # ncl_fish_estim is going to be a list of strings containing the names of the  weights to which ncl is applied
        # it will add the derivatives of te loss w.r.t the activities (needed for the fish estims) to the extras 
        
        if ncl_fish_estim:
            
            self.grad_list_z = []

        if hasattr(self.args, 'net_fb') and self.args.net_fb:
            self.z = self.z.expand(o.shape[0], self.z.shape[1])
            oz = torch.cat((o, self.z), dim=1)
            
            u = self.m1_act(self.M_u(oz))

        elif ncl_fish_estim and self.args.D1 != 0:
            grad_u = None
            def save_grad_u(grad):
                nonlocal grad_u 
                grad_u = grad
            
            u_pre_act = self.M_u(o)
            handle_u = u_pre_act.register_hook(save_grad_u)
            u = self.m1_act(u_pre_act)
            # remove hook 
            handle_u.remove()

        elif node_pert_noises is not None and 'M_u' in node_pert_noises.keys() :
            u = self.m1_act(self.M_u(o) + node_pert_noises['M_u'])
        
        
        else:
            
            u = self.m1_act(self.M_u(o))

        if self.args.xdg and ('u' in self.args.gate_layers):
                
            u = torch.mul(u, self.u_mask)

       
                
        if extras:
            if self.args.xdg and 'x' in self.args.gate_layers:
                v, etc = self.reservoir(u, extras=True, x_mask = self.x_mask)
            else:
                v, etc = self.reservoir(u, extras=True,ncl_fish_estim = ncl_fish_estim,node_pert_noises = node_pert_noises)

        else: 
            if self.args.xdg and 'x' in self.args.gate_layers:
                v = self.reservoir(u, extras=False, x_mask = self.x_mask)
            else:
                v = self.reservoir(u, extras=False)
        
        if self.args.xdg and ('v' in self.args.gate_layers):
                v = v * self.v_mask
        
        if  node_pert_noises is not None  and ('M_ro'in node_pert_noises.keys()):
            z = self.M_ro(self.m2_act(v)) +node_pert_noises['M_ro']
        else:
            z = self.M_ro(self.m2_act(v))
        
        self.z = self.out_act(z)

        if not extras:
            return self.z
        elif self.args.use_reservoir:
            if not ncl_fish_estim:
                return self.z, {'u': u, 'x': etc['x'],'pre_act_x': etc['pre_act_x'] ,'v': v}
            elif ncl_fish_estim:
                self.z.retain_grad() 
                if self.args.train_parts == ['']:
                    if self.args.D1 != 0:
                        return self.z, {'u': u, 'x': etc['x'],'pre_act_x': etc['pre_act_x'] ,'v': v, 'grad_l_wrt_x':etc['grad_l_wrt_x'], 'grad_l_wrt_u':grad_u}
                    else:
                        return self.z, {'u': u, 'x': etc['x'],'pre_act_x': etc['pre_act_x'] ,'v': v, 'grad_l_wrt_x': etc['grad_l_wrt_x']}
                else:
                    return self.z, {'u': u, 'x': etc['x'],'pre_act_x': etc['pre_act_x'] ,'v': v, 'grad_l_wrt_u':grad_u}
        else:
            return self.z, {'u': u, 'v': v}

    def reset(self, res_state=None, device=None):
        self.z = torch.zeros((1, self.args.Z))
        if self.args.use_reservoir:
            self.reservoir.reset(res_state=res_state, device=device)

class M2Reservoir(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        self.args = update_args(DEFAULT_ARGS, args)

        if self.args.res_seed is None:
            self.args.res_seed = random.randrange(1e6)
        if self.args.res_x_seed is None:
            self.args.res_x_seed = np.random.randint(1e6)

        self.tau_x = 10
        self.activation = torch.tanh

        # use second set of dynamics equations as in jazayeri papers
        self.dynamics_mode = 0
        # storage variable for gradient of loss wrt to hidden units before act. func applied
        self.grad_x = 0
        self._init_vars()
        self.reset()

    def _init_vars(self):
        if self.args.res_path is not None:
            self.load_state_dict(torch.load(self.args.res_path))
        else:
            with TorchSeed(self.args.res_seed):
                if self.args.D1 == 0:
                    # go straight from the input to the network
                    self.W_u = nn.Identity()
                else:
                    # use representation layer in between as division bw trained / untrained parts
                    self.W_u = nn.Linear(self.args.D1, self.args.N, bias=False)
                    torch.nn.init.normal_(self.W_u.weight.data, std=self.args.res_init_g / np.sqrt(self.args.D1))

                # recurrent weights
                self.J = nn.Linear(self.args.N, self.args.N, bias=self.args.res_bias)
                torch.nn.init.normal_(self.J.weight.data, std=self.args.res_init_g / np.sqrt(self.args.N))

                if self.args.D2 == 0:
                    # go straight to output
                    self.W_ro = nn.Identity()
                else:
                    # use low-D representation layer bw output
                    self.W_ro = nn.Linear(self.args.N, self.args.D2, bias=self.args.res_bias)
                    torch.nn.init.normal_(self.W_ro.weight.data, std=self.args.res_init_g / np.sqrt(self.args.D2))

    # add designated fixed points using hopfield network
    def add_fixed_points(self, n_patterns):
        patterns = (2 * torch.eye(self.args.N)-1)[:n_patterns, :]
        W_patt = torch.zeros((self.args.N, self.args.N))
        for p in patterns:
            p_tensor = torch.as_tensor(p)
            W_patt += torch.outer(p_tensor, p_tensor)
        self.J.weight.data += self.args.fixed_beta * W_patt / self.args.N / n_patterns

        # pdb.set_trace()

    def burn_in(self, steps):
        for i in range(steps):
            g = torch.tanh(self.J(self.x))
            delta_x = (-self.x + g) / self.tau_x
            self.x = self.x + delta_x
        self.x.detach_()

    def save_grad_x(self,grad):
        self.grad_x = grad.detach().clone()

    # extras currently doesn't do anything. maybe add x val, etc.
    def forward(self, u=None, extras=False, x_mask=None, ncl_fish_estim=False, node_pert_noises = None):
        
        

        if self.dynamics_mode == 0:
            #gate xs if specified
            
            if u is None:
                g = self.activation(self.J(self.x))

            elif ncl_fish_estim and self.args.train_parts == ['']:
                pre_act_x = self.J(self.x) + self.W_u(u)
                pre_act_x.register_hook(self.save_grad_x)
                g = self.activation(pre_act_x)
            
            elif node_pert_noises is not None and ('W_u'in node_pert_noises.keys() or 'J' in node_pert_noises.keys()):
                if 'W_u' in self.args.node_pert_parts:
                    x_pert = node_pert_noises['W_u']
                else:
                    x_pert = node_pert_noises['J']
                g = self.activation(self.J(self.x) + self.W_u(u)+ x_pert)
            
            else:
                g = self.activation(self.J(self.x) + self.W_u(u))
            # adding any inherent reservoir noise
            if self.args.res_noise > 0:
                g = g + torch.normal(torch.zeros_like(g), self.args.res_noise)
            delta_x = (-self.x + g) / self.tau_x
            
            if not ncl_fish_estim: 
                pre_act_x = self.x.detach()
            
            self.x = self.x + delta_x
            
            if self.args.xdg and ('x' in self.args.gate_layers):
                self.x = self.x * x_mask
            
            v = self.W_ro(self.x)

        elif self.dynamics_mode == 1:
            if u is None:
                g = self.J(self.r)
            else:
                g = self.J(self.r) + self.W_u(u)
            if self.args.res_noise > 0:
                gn = g + torch.normal(torch.zeros_like(g), self.args.res_noise)
            else:
                gn = g
            delta_x = (-self.x + gn) / self.tau_x
            self.x = self.x + delta_x
            self.r = self.activation(self.x)

            v = self.W_ro(self.r)

        if extras:
            if not ncl_fish_estim:
                etc = {'x': self.x.detach(), 'pre_act_x': pre_act_x.detach()}
            else:
                if self.args.train_parts == ['']:
                    
                    etc = {'x': self.x.detach(), 'pre_act_x': pre_act_x.detach(), 'grad_l_wrt_x': self.grad_x}
                else:
                     etc = {'x': self.x.detach(), 'pre_act_x': pre_act_x.detach()}
            
            return v, etc
        return v

    def reset(self, res_state=None, burn_in=True, device=None):
        if res_state is None:
            # load specified hidden state from seed
            res_state = self.args.res_x_seed

        if type(res_state) is np.ndarray:
            # load an actual particular hidden state
            # if there's an error here then highly possible that res_state has wrong form
            self.x = torch.as_tensor(res_state).float()
        elif type(res_state) is torch.Tensor:
            self.x = res_state
        elif res_state == 'zero' or res_state == -1:
            # reset to 0
            self.x = torch.zeros((1, self.args.N))
        elif res_state == 'random' or res_state == -2:
            # reset to totally random value without using reservoir seed
            self.x = torch.normal(0, 1, (1, self.args.N))
        elif type(res_state) is int and res_state >= 0:
            # if any seed set, set the net to that seed and burn in
            with TorchSeed(res_state):
                self.x = torch.normal(0, 1, (1, self.args.N))
        else:
            print('not any of these types, something went wrong')
            pdb.set_trace()

        if device is not None:
            self.x = self.x.to(device)

        if self.dynamics_mode == 1:
            self.r = self.activation(self.x)

        if burn_in:
            self.burn_in(self.args.res_burn_steps)

# creates reservoir with embedded hopfield patterns
def hopfield_reservoir(N, g, patterns, beta):
    W = torch.zeros((N, N))
    W_rand = torch.normal(torch.zeros_like(W), g / np.sqrt(N))
    W += W_rand
    for p in patterns:
        p_tensor = torch.as_tensor(p)
        W_patt = torch.outer(p_tensor, p_tensor) / N
        W += beta * W_patt

    return W

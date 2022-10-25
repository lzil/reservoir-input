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
    def __init__(self, args=DEFAULT_ARGS, train_idx=None):
        super().__init__()
        self.args = update_args(DEFAULT_ARGS, args)
       
        if self.args.network_seed is None:
            self.args.network_seed = random.randrange(1e6)

        self.out_act = get_activation(self.args.out_act)
        self.m1_act = get_activation(self.args.m1_act)
        self.m2_act = get_activation(self.args.m2_act)
        self.train_idx= train_idx

        self._init_vars() #NB intialise variables
        self.reset()
        #if you want ot pass things into net (like train_idx) put before self.reset()
        
        

        

        

    def _init_vars(self):
        with TorchSeed(self.args.network_seed):
            D1 = self.args.D1 if self.args.D1 != 0 else self.args.N
            D2 = self.args.D2 if self.args.D2 != 0 else self.args.N
            # net feedback into input layer
            #print(self.args)
            #note net feeback is 'off' by default
            if hasattr(self.args, 'net_fb') and self.args.net_fb:
            
                self.M_u = nn.Linear(self.args.L + self.args.T + self.args.Z, D1, bias=self.args.ff_bias)
                #adding the number of tasks args.T to facilitate the rule input
                # we augment the sensory input with a rule input telling us which task is being done see OWM diagram with rule input

                
                    
            else:
                self.M_u = nn.Linear(self.args.L + self.args.T, D1, bias=self.args.ff_bias)
                #e.g T is the number of datasets

            

            self.M_ro = nn.Linear(D2, self.args.Z, bias=self.args.ff_bias)
        self.reservoir = M2Reservoir(self.args)

        
        if self.args.xdg:
            with torch.no_grad():
                gate_pool_loading_bay=[]
                self.args.gate_layers = ['u', 'v', 'x'] if self.args.train_parts == 'all' else self.args.gate_layers 
                
                if 'x' in self.args.gate_layers:
                    self.gate_x = True

                else: 
                    self.gate_x= False

                for layer in self.args.gate_layers: 
                    #i.e. (for i in ['u','v'])
                    # #we want a list of all of us indices [ ('u', [0,0]),..., ('u', [0,D1-1] ) (D1-1 bc zero-indexing) and a list of all of vs in indices#[('v',[0,0]),..., ('v', [0, D2-1])]
                    if layer == 'u':
                        u_idxs = [ (layer, [0,i])  for i in range(self.args.D1)]
                        gate_pool_loading_bay.append(u_idxs)
                    elif layer =='v':
                        v_idxs = [ (layer, [0,i]) for i in range(self.args.D2)]
                        gate_pool_loading_bay.append(v_idxs)
                    elif layer == 'x':
                        x_idxs = [ (layer, [0,i]) for i in range(self.args.N)]
                        gate_pool_loading_bay.append(x_idxs)
                
                gate_pool = [layer_idx for layer_idx_list in gate_pool_loading_bay for layer_idx in layer_idx_list]

                    #print(f'this is the loading bay:{gate_pool_loading_bay}')
                    # #then concatenate lists, result is our gating pool so assign to variable called gate_pool
                    #https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists

                    
                    #then calculate m using args.X
                    # m = len(gate_pool)*X/100 
                    # note m isn't necessarily going to be an integer so need to convert it
                m= len(gate_pool)*self.args.X/100
                
                    #m is the number of units to select (from all units in network that are in gate layers) and gate so that X% of network is gated for any one task 
                    #note 


                    #then choose m things randomly from gate pool for each task: 
                    #note to self for one task we want to choose randomly the same m units each time. 
                


                    #we want the random sample to be the same each time until train_idx changes (so that for a particular train_idx the gated units are the same every time we call the net) so need to set the same seed and have it change only when we move to the next task

                random.seed(self.train_idx)
                    #print(f'train_idx is {train_idx}')
                gated_units = random.sample(gate_pool, k=round(m))
                    # .sample bc we want to sample without replacement as opposed .choice
                    #print(f'these are my gated units {gated_units}')
                    #from these m units the ones with 'u' labels to index the us for gating
                    #and the ones with 'v' labels to select the vs for gating: 
                    #note u is a tensor with shape [1,D1], v is a tensor with shape [1,D2]

                u_gate_idxs = []
                v_gate_idxs = []
                x_gate_idxs = []

                for layer, idx in gated_units:
                    #print(f'{layer,idx}')

                    if layer == 'u': 
                        u_gate_idxs.append(idx)
                    elif layer == 'v':
                        v_gate_idxs.append(idx)
                    elif layer == 'x':
                        x_gate_idxs.append(idx)
                
                print(f'these are the u_gate_idx{u_gate_idxs}')
                print(torch.ones(1,D1))
                
                #print(f'these are the v_gate_idx{v_gate_idxs}')
                # print(f'these are the x_gate_idx{x_gate_idxs}')

                #create masks for gating:
                    
                for layer in self.args.gate_layers:
                    if layer == 'u':
                        self.u_gater = torch.ones((1,D1))
                        for idx in u_gate_idxs:
                            self.u_gater[0][idx[1]]*=0

                    elif layer == 'v':
                        self.v_gater = torch.ones((1,D2))
                        for idx in v_gate_idxs:
                            self.v_gater[0][idx[1]]*=0

                    elif layer == 'x':
                        self.x_gater = torch.ones((1,N))
                        for idx in x_gate_idxs:
                            self.x_gater[0][idx[1]]*=0

                

                
                    


                    




        




        # load params for reservoir if they exist
        if self.args.M_path is not None:
            M_params = torch.load(self.args.M_path)
            # TODO load M_params
        if self.args.model_path is not None:
            self.load_state_dict(torch.load(self.args.model_path))

    def add_task(self):
        M = self.M_u.weight.data
        self.M_u.weight.data = torch.cat((M, torch.zeros((M.shape[0],1))), dim=1) #horizontal concatenation 
        self.args.T += 1

    def forward(self, o, extras=False) :
        # pass through the forward part
        # o should have shape [batch size, self.args.T + self.args.L]

        
        #leave net feedback for now 
        if hasattr(self.args, 'net_fb') and self.args.net_fb:
            #net feedback from output to input
            self.z = self.z.expand(o.shape[0], self.z.shape[1])
            oz = torch.cat((o, self.z), dim=1)
            u = self.m1_act(self.M_u(oz))
            

            if self.args.sequential and self.args.xdg:
    
                u = self.m1_act(self.M_u(o))
                u = u * self.u_gater

                if extras:
                        v, etc = self.reservoir(u, extras=True, gating=gate_x)
                        v = v * self.v_gater
                else:
                        v = self.reservoir(u, extras=False, gating=gate_x)
                        v = v * self.v_gater

                
            else:
                u = self.m1_act(self.M_u(o))
                if extras:
                        v, etc = self.reservoir(u, extras=True)
                else:
                        v = self.reservoir(u, extras=False)
        






        # no net feeback - this is what runs by default
        else:                        
            #print(f'yas{o.shape}')
            if self.args.sequential and self.args.xdg:

                u = self.m1_act(self.M_u(o))
                u = u * self.u_gater

                if extras:
                        v, etc = self.reservoir(u, extras=True, gating=self.gate_x)
                        v = v * self.v_gater
                else:
                        v = self.reservoir(u, extras=False, gating=self.gate_x)
                        v = v * self.v_gater


            else:
                u = self.m1_act(self.M_u(o))
                print(f'this is uuuuu{u.size()}')
                if extras:
                        v, etc = self.reservoir(u, extras=True)
                else:
                        v = self.reservoir(u, extras=False)
        

        z = self.M_ro(self.m2_act(v))
        self.z = self.out_act(z)

        if not extras:
            return self.z
        elif self.args.use_reservoir:
            return self.z, {'u': u, 'x': etc['x'], 'v': v, }
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

    # extras currently doesn't do anything. maybe add x val, etc.
    def forward(self, u=None, extras=False, gating=False):
        
        if self.dynamics_mode == 0:
            #if doing xdg project context signal onto hidden units
            if u is None:
                
                    g = self.activation(self.J(self.x))
            else:

                     g = self.activation(self.J(self.x) + self.W_u(u))

            # adding any inherent reservoir noise
            if self.args.res_noise > 0:
                g = g + torch.normal(torch.zeros_like(g), self.args.res_noise)
            delta_x = (-self.x + g) / self.tau_x
            
            self.x = self.x + delta_x

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
            etc = {'x': self.x.detach()}
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


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import pdb

import random
from collections import OrderedDict

from utils import load_rb, get_config

from tasks import * #can we do wildcard imports?

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
        #input is a list of [(dataset's name, dataset itself)]

        self.args = args
        self.dnames = []    # names of dsets
        self.data = []      # dsets themselves
        self.t_types = []   # task type
        self.lzs = []       # Ls and Zs for task trials
        self.x_ctxs = []    # precomputed context inputs
        self.max_idxs = np.zeros(len(datasets), dtype=int)
        self.t_lens = []
        self.T = len(self.data)
        for i, (dname, ds) in enumerate(datasets):
            self.dnames.append(dname)
            self.data.append(ds)
            # setting context cue for appropriate task
            x_ctx = np.zeros((args.T, ds[0].t_len))
            x_ctx[i] = 1
            self.x_ctxs.append(x_ctx) #rule input for the datasets 
            self.t_lens.append(ds[0].t_len)
            # cumulative lengths of data, for indexing
            self.max_idxs[i] = self.max_idxs[i-1] + len(ds)
            # use ds[0] as exemplar to set t_type, L, Z
            self.t_types.append(ds[0].t_type)
            self.lzs.append((ds[0].L, ds[0].Z))

        if self.args.multimodal:
            self.t_types =[]
            self.fixation = True #start by assuming at least one task requires fixation; include a shared fixation input by default
            self.fixation_task_count = 0 #no.of fixation tasks
            self.max_t_len = 0

            #the L,Z dimension for the augmented inputs and outputs, respectively, without counting the fixation input
            self.tot_L_sans_fix = 0 
            self.tot_Z_sans_fix = 0

            #dictionary of input and output locations(the first index in the shells) for different task groups
            self.input_modality_locations={}
            self.output_modality_locations={}
            
            # pass through datasets in order to collect info needed to augment inputs and outputs in get_item
            for ds in self.data:
                #check whether task has a fixation modality
                task_has_fix = ds[0].has_fix
                t_type = ds[0].t_type
                print(f'this is a t_type:{t_type}')
                task_has_fix= ds[0].has_fix
            
                task_L =ds[0].L
                task_Z= ds[0].Z

                #if we haven't already seen this t_type before (recall if tasks/datasets have the same t_types we want to use the same modalities)
                # TODOS : what about DMPA - we want it to share the same modality
                
                #if you haven't seen the task before
                if t_type not in self.t_types:
                    self.t_types.append(t_type)

                    if task_has_fix:
                        self.fixation_task_count +=1
                        self.tot_L_sans_fix += (ds[0].L -1 )
                        self.tot_Z_sans_fix += (ds[0].Z -1 )
                    else:
                        self.tot_L_sans_fix += (ds[0].L )
                        self.tot_Z_sans_fix += (ds[0].Z)

                #the length of all trials in multimodal
                #if say an rsg task has length 300 but max is 600, we'll just put zeros after t=300
                if ds[0].t_len > self.max_t_len:
                    self.max_t_len = ds[0].t_len
            
             #where you'd input (new) task modalities if you hadn't already encountered the task
            if self.fixation_task_count==0:
                    self.fixation = False #no tasks that require fixation
            
            #if there is at least one fixation-modality-requiring task 
            if self.fixation:
                #tell us from which index of the shells [inclusive] to start inputting modalities as we go
                #note: for all the tasks we consider, a task has a fixation modality in input if and only if it has a fixation modality in output
                
                #if 
                self.modality_input_index = 1
                self.modality_output_index = 1 

                self.L_mm=self.tot_L_sans_fix +1 + self.T
                self.Z_mm=self.tot_Z_sans_fix +1
                
                
            else:
                self.modality_input_index = 0
                self.modality_output_index = 0
                
                self.L_mm=self.tot_L_sans_fix 
                self.Z_mm=self.tot_Z_sans_fix 
                
                
            
            



            #t_types=[]


            
            
            #self.multimodal_trials=[]
            

            # self.x_shell = np.zeros((self.tot_L_sans_fix, self.max_t_len))
            # self.y_shell =np.zeros((self.tot_Z_sans_fix, self.max_t_len))

            

            
                
                


    def __len__(self):
        #__len_returns the number of samples in the dataset hence the following:
        return self.max_idxs[-1] #the sum of the lengths of the datasets

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[ii] for ii in range(len(self))[idx]]
        # index into the appropriate dataset to get the trial
        context = self.get_context(idx)
        # idx variable now references position within dataset
        if context != 0:
            idx = idx - self.max_idxs[context-1]
        # 'context' is the dataset we're indexing into i.e. which dataset
        trial = self.data[context][idx]
        x = trial.get_x(self.args)
        
        y = trial.get_y(self.args)
        

        x_cts = self.x_ctxs[context]
        #augment x and y and then add 
        if self.args.multimodal:
            task_len = x.shape[1] 
            self.x_shell = np.zeros((self.tot_L_sans_fix, task_len))
            self.y_shell =np.zeros((self.tot_Z_sans_fix, task_len))

            if self.fixation:
                    #if one of the tasks requires fixations
                    #tell us from which index of the shells [inclusive] to start inputting modalities as we go
                    #tells us from which index of the shells [inclusive] to start inputting modalities as we go
                    self.modality_input_index = 1
                    self.modality_output_index = 1
                    #shared fixation
                    self.fix_shell=np.zeros((1,task_len))
                    #we just add first row of x (y) to this x_fix and keep remainining
                    self.x_shell = np.concatenate((self.fix_shell , self.x_shell), axis=0)
                    self.y_shell =np.concatenate((self.fix_shell, self.y_shell), axis=0)
            
            else:
                self.modality_input_index = 0
                self.modality_output_index = 0

            #get location in shells for input and output of different tasks
            self.t_types = []
            for ds in self.data:
                task_has_fix = ds[0].has_fix
                t_type = ds[0].t_type
                task_has_fix= ds[0].has_fix
            
                task_L =ds[0].L
                task_Z= ds[0].Z
                
                
                if t_type not in self.t_types:
                    self.t_types.append(t_type)
                    self.input_modality_locations[t_type]=self.modality_input_index
                    self.output_modality_locations[t_type]= self.modality_output_index


                    #update where we'd next input a new task 
                    if task_has_fix:
                        self.modality_input_index+= task_L -1 
                        self.modality_output_index += task_Z-1
                    else:
                        self.modality_input_index+= task_L 
                        self.modality_output_index += task_Z








            t_type = trial.t_type
            if self.x_shell.shape[1] > x.shape[1]:
                    x_pad_len = self.x_shell.shape[1] - x.shape[1]
                    #x_pad = np.zeros((x.shape[0],x_pad_len))
                    #x = np.concatenate((x,x_pad), axis=1)
                    #need context to be on throughout the trial even if input is zero and output are 'off' 
                    x_cts_pad_len = self.x_shell.shape[1]-x_cts.shape[1]
                    x_cts_pad = np.tile(x_cts[:,0].reshape((self.args.T,1)), reps = x_cts_pad_len )
                    x_cts = np.concatenate((x_cts,x_cts_pad),axis=1)
                
                    #y_pad_len = self.y_shell.shape[1] - y.shape[1] 
                    #y_pad = np.zeros((y.shape[0],y_pad_len))
                    #y = np.concatenate((y,y_pad),axis=1)
            
            if trial.has_fix:
                self.x_shell[0] = x[0]
                
                self.y_shell[0] = y[0]
                #subtract no. of fixation inputs
                self.x_shell[self.input_modality_locations[t_type]: self.input_modality_locations[t_type]+trial.L-1, :] = x[1:,:]
                self.y_shell[self.output_modality_locations[t_type]: self.output_modality_locations[t_type]+trial.Z-1, :] = y[1:,:]

            else:
                if self.fixation:
                    #if its rsg say, mm fixation is just zeros
                    self.x_shell[0]=np.zeros((1,self.max_t_len))
                    self.y_shell[0]=np.zeros((1,self.max_t_len))
                    self.x_shell[self.input_modality_locations[t_type]: self.input_modality_locations[t_type]+trial.L, :] = x[0:,:]

                    self.y_shell[self.output_modality_locations[t_type]: self.output_modality_locations[t_type]+trial.Z, :] = y[0:,:]

            x = self.x_shell
            y = self.y_shell 

        # context comes after the stimulus
        x = np.concatenate((x, x_cts))
        #'under' stimulus 
      

        trial.context = context
        trial.dname = self.dnames[context]
        if self.args.multimodal:
            trial.lz = (self.L_mm, self.Z_mm)
        else:
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

    #similar idea to args.sequential,creates separate test loaders for each task 
    elif args.multimodal:
        def create_subset_loaders(dset, batch_size, drop_last):
            loaders = {}
            max_idxs = dset.max_idxs
            # e.g of dset_types is ['dmc-pro', 'rsg', 'memory-pro']
            for task_type in dset.t_types:
                #indices of examples of type task_type
                
                task_idxs = [i for i, trial in enumerate(dset) if trial[2].t_type== task_type]
                subset = Subset(dset,task_idxs)
                loader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collater, drop_last=drop_last)
                loaders[task_type]=loader
            return loaders
        # create the loaders themselves
        test_loaders = create_subset_loaders(test_set, test_size, False)
        if split_test:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collater, drop_last=True)
            return (train_set, train_loader), (test_set, test_loaders)
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


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
    #optim has a whole bunch of different optimizers see PyTorch guide
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
#a scheduler is a method to adjust the learning rate based on different aspects of training/validation during training
#see: https://pytorch.org/docs/stable/optim.html for more 

#we use a MultiStepLR


# dataset that automatically creates trials composed of trial and context data
# input dataset should be in form [(dname, dset), ...] - we create these input datasets in create loaders:

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
        #see get_criteria in ipad: enumerate(datasets) = [(0,dataset_1),...,(n-1,dataset_n)]
        dset = load_rb(dpath)

        #loads the dataset from a pkl file with path dpath
        # trim and set name of each dataset
        dname = str(i) + '_' + ':'.join(dpath.split('/')[-1].split('.')[:-1])
        if split_test:
            #i.e. if there's a train test split:
            
            cutoff = round(.9 * len(dset)) 
            #what does round: rounds to nearest integer
            
            #train_tests split is done below 
            dsets_train.append([dname, dset[:cutoff]])
            #take from 0-index row, to cutoff-1 index row
            dsets_test.append([dname, dset[cutoff:]])
            #take from cutoff index row to the last row inclusive
            #notice there's an order
        else:
            #notice if no train_test split then it's just test bc it doesn't 'make sense(or at least we're not interested in doing that) to train and not test
            dsets_test.append([dname, dset])

    # creating datasets
    test_set = TrialDataset(dsets_test, args)
    #TrialDataset is a different function 
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
        #args.loss has the loss function we specify in command line at 
        #run time, by default it's  "mse"


        # do this in a roundabout way due to truncated bptt
        #(how it this roundabout?)
        fn = nn.MSELoss(reduction='sum')
        #the loss function is MSE but bc we've specified 'reduction'= sum, we don't divide the whole MSE by n
        
        #how nn.MSELoss works: it creates a criterion that measures the MSE (squared L2 norm) between each element in the x input x and target y. [i.e. they mean the model's current outputs for the seen inputs x vs the observed outputs y for these seen inputs]
        #we'd go fn=loss(input, target)
        #then (eventually:)
        #loss.backward()

        #I think we work out an loss function MSE for each input-output pair bc recall:
        #our data is a sequence w.r.t time: so 
        #for a single training case we need the l2 norm between observed output and current model output on each input point in the sequence.
        # I think this is what's happening...


        def mse(o, t, i, t_ix, single=False):
            #calculates the MSE between o and t 

            # last dimension is number of timesteps
            #(expl.: we're referring to last dimension of o and t, they're of shape 1 and they're used to the number of time steps in the sequences - the sequences are the rows of the o and t variables
            #(o is model output for inputs, t are the corresponding targets)


            # divide by batch size to avoid doing so logging and in test(?)

            # needs all the contexts to be the same length(?) - ask Liang
            
            #what are o, t, i and t_ix
            #o is the vector of inputs of the batch [see above: "How nn.MSELoss works"]
            #t is the vector of "t"argets the corresponding observed outputs(observed outputs are called 'targets')for these inputs in o

            # what's in i ?
            #i is an array of task objects for a single dataset(or maybe multiple) including the possibility of a single trial
            #but it's cool bc it can store a whole bunch of different trials which may or may not correspond to different tasks



            loss = 0.   #a notice initialize loss as 0 but going '0.' to ensure that it's a float
            if single:
                
                #i.e. MSE has the functionality to compute the loss on a single case (a single input output pair) as opposed to on a batch but the code is built to deal with batches of inputs o's (for model outputs), and batches of targets so we need to do the unsqueezing that's seen below to represent a single output as batch containing only 1 training case.

                # a single task 

                
                o = o.unsqueeze(0)
                #torch.unsqueeze() returns a new tensor with a dimension of size one inserted at the specified position, so zero 
                #e.g. if o had shape(2,1) initially, after this we'd have it of shape (1,2,1) bc #we've inserted a dimension of size 1 at the 0 index entry of the o's shape vector

                #reason we want this is so that we can multi-dimensional inputs i.e. each input is of shape (1,2) and
                #o is actually a vector of inputs so we need to be able to index them: so if o has shape (5,2,1)
                #in this case, this means it has 5 inputs of shape (2,1) and we can
                #access the ith-indexed input of shape (2,1) of o by going o[i]
                #the reason we do this is bc the 

                #also the length len is always the entry in first dimension of shape tuple (bc it's the length of the list of lists of ... lists)
                #e.g. suppose x=torch.rand(4,5) --> x.shape is torch.Size([4,5])
                #and len(x) is 4
                #if we go x=x.unsqueze(0) we get x.shape as torch.Size([1,4,5]) and
                #len(x) is 1

                #so if single is False , yes len(t) is the number of rows of t
                #which is the number of targets



                t = t.unsqueeze(0)
                i = [i] # a single-element list
            for j in range(len(t)):
                
                #for each target sequence [see above], based on t's def above, len(t) gives us the number of training cases in the batch

                #read first:
                #our target for an input isn't just a point, it's a sequnce of points bc time series! So we t[j] is the target time series with index j in t, and o[j] is the correspdonding time series outputted by our model for the same input. [the jth rows t and o, respectively]
                #so the length directly below is the length of these time series which should be the length of task
                #so when we use the loss function on fn(t[j], o[j]) we're working out the loss on a single training case.
                #doing it for each j in range(len(j)) allows us to do it for each training case we input

                #we get a total loss for this batch and then we divided by batch size


                length = i[j].t_len 
                
                #let's see if i[j] is a single trial - it is 
                


                #for RSG task the t_len=600 by default. 
                #again it's the number of timesteps for the task 
                #i stores task objects that look like  <tasks.RSG object at 0x13c768370> if we print them them for an rsg dataset.
                #but why do we have multiple task objects for a single dataset

                #what's t_ix?: <t_ix stand for ’truncate index” as in how much to truncate everything by.

                
                #the -e stands for “exponential”, it’s a loss that Liang tried while trying to find ways to improve performance on RSG. I have some comments on the definition, in helpers.py where the code for it is. in general I don’t think it’s necessary, usually mse works just fine

              
                #does each i[j] correspond to a single training case in the dataset:



                if t_ix + t.shape[-1] <= length:
                    #so if we specify a valid truncate index (value)
                    #by valid we mean one that actually activates truncation i.e. if t_ix positive this condition is False, the next one i
                    #if we specify t_ix =0, t.shape[-1] will equal length task length (by def)
                    #and so we won't truncate

                    #if we specify a negative truncate index which doesn't make sense, this condition ensures that no truncating happens and code runs anyway



                    #this looks to be the condition that, if satisfied, means we do 'regular' BPTT rather than truncated BPTT see below
                    #t.shape[-1]is last element of the shape tuple for a target i.e. which is the "length" of the trial # for RSG this is t_len
                    #length also looks to contain t_len (will know for sure once we figure out what i[j] holds)
                    #so if t_ix + t_len is <= length then we do regular BPTT it seems (confirm?)
                    #and length= t_len in this case, and if we call
                    #this MSE function with t_ix gt 0 we get truncated bptt vs if we call it with t_ix =0 we get normal bptt
                    #bc in this case, I think that t_ix will equal length
                    
                    #now we use our loss function on a *single* training case: t[j] is a jth target vector and o[j] is jth produced vector 
                    #loss on a time-series training case
                    loss += fn(t[j], o[j])# / length
                    
                
                elif t_ix < length:

                    #this looks to be the condition that triggers doing truncated BPTT rather than BPTT
                    #t_ix stands for truncate index - yes: see: https://stats.stackexchange.com/questions/255203/can-someone-please-explain-the-truncated-back-propagation-through-time-algorithm 
                    
                    t_adj = t[j,:,:length-t_ix] #I think t_ix os how much we truncate it? bc we're taking the jth target in the batch, all the entries in rows
                    #and then for the columns, we take all the entries up to the entry with index length-t_ix exclusive(exclusive bc of how python slicing works), 

                    

                    o_adj = o[j,:,:length-t_ix]
                    loss += fn(t_adj, o_adj)# / length
            return args.l1 * loss / args.batch_size
            #this is what mse returns
            #but args.l1 is an adam lambda, default value is 1 but let's dig into it 

            #we then append the mse for this batch to the criteria list
        criteria.append(mse)


    if 'bce' in args.loss:
        #BCE stands for binary cross entropy
        weights = args.l3 * torch.ones(1)
        fn = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=weights)
        def bce(o, t, **kwargs):
            return args.l1 * fn(t, o)
        criteria.append(bce)
    if 'mse-e' in args.loss:

        #the -e stands for “exponential”, it’s a loss that Liang tried while trying to find ways to improve performance on RSG. I have some comments on the definition, in helpers.py where the code for it is. in general I don’t think it’s necessary, usually mse works just fine

        # ONLY FOR RSG AND CSG, WITH [1D] OUTPUT
        # exponential decaying loss from the go time on both sides
        # loss is 1 at go time, 0.5 at set time
        # normalized to the number of timesteps taken
        fn = nn.MSELoss(reduction='none') #reduction= 'none' 
        #means we stores the losses between elements of t and y as elements of list i.e. we don't sum or average them 
        #see documentation for more info
        def mse_e(o, t, i, t_ix, single=False):
            loss = 0.
            if single:
                o = o.unsqueeze(0)
                t = t.unsqueeze(0)
                i = [i]
            for j in range(len(t)): #for each target
                
                # last dimension is number of timesteps #i.e the number of timesteps in the time series
                t_len = t.shape[-1]
                xr = torch.arange(t_len, dtype=torch.float)
                #will return a 1D tensor with values from the 
                #interval[0, t_len] and step=1 (by default)

                # placement of go signal in subset of timesteps
                t_g = i[j].rsg[2] - t_ix
                #rsg[2] is third element of .rsg attribute
                #of an RSG task object 
                #it's the go_time 



                #what is t_ix? 

                t_p = i[j].t_p #t_p is the length of produced interval for the 
                #task j in i 
                #ah i[j] is a particular task object
                #or is it a dataset for a particular task object? might be bc why is the first dimension of i have size equal to the number of targets (bc we go i[j] and j is in len(t))? 


                #e.g. see RSG task class def., i[j] has the attributes
                #so i is a list of different task objects?
                #different RSG tasks, say with some CSG tasks !

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
    return criteria #return the list of batch_losses where each batch loss is computed for a particular loss function for this task rsg-100-150
    #criterion as in loss for a particular loss functions - differnt loss fucntions <--> different criteria
    #initially returns a list of the batch_loss of a single batch on a particular loss function by appending the mse on this batch to this list 
    #then if we call it again, on another batch - we can use a different loss function - , it will append the loss on this new batch to the list that has the loss on first batch. So we get a list of losses on different batches.

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

import numpy as np
from scipy.stats import norm

import pickle
import os
import sys
import json
import pdb
import random
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import collections as matcoll

import argparse

# from motifs import gen_fn_motifs
from utils import update_args, load_args, load_rb, Bunch

eps = 1e-6

mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = .5

cols = ['coral', 'cornflowerblue', 'magenta', 'orchid']

# dset_id is the name of the dataset (as saved)
# n is the index of the trial in the dataset
class Task:
    def __init__(self, t_len, dset_id=None, n=None):
        self.t_len = t_len
        self.dset_id = dset_id
        self.n = n

        self.L = 0
        self.Z = 0

    def get_x(self):
        pass

    def get_y(self):
        pass

class RSG(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= False 
        if args.intervals is None:
            t_o = np.random.randint(args.min_t, args.max_t)
        else:
            t_o = random.choice(args.intervals)
        t_p = int(t_o * args.gain)
        ready_time = np.random.randint(args.p_len * 2, args.max_ready)
        set_time = ready_time + t_o
        go_time = set_time + t_p

        self.t_type = args.t_type
        self.p_len = args.p_len
        self.rsg = (ready_time, set_time, go_time)
        self.t_o = t_o
        self.t_p = t_p

        self.L = 1
        self.Z = 1

    def get_x(self, args=None):
        rt, st, gt = self.rsg
        # ready pulse
        x_ready = np.zeros(self.t_len)
        x_ready[rt:rt+self.p_len] = 1
        # set pulse
        x_set = np.zeros(self.t_len)
        x_set[st:st+self.p_len] = 1
        # insert set pulse
        x = np.zeros((1, self.t_len))
        x[0] = x_set
        # perceptual shift
        if args is not None and args.m_noise != 0:
            x_ready = shift_x(x_ready, args.m_noise, self.t_o)
        x[0] += x_ready
        # noisy up/down corruption
        if args is not None and args.x_noise != 0:
            x = corrupt_x(args, x)
        return x

    def get_y(self, args=None):
        y = np.arange(self.t_len)
        slope = 1 / self.t_p
        y = y * slope - self.rsg[1] * slope
        # so the output value is not too large
        y = np.clip(y, 0, 1.5)
        # RSG output is only 1D
        y = y.reshape(1, self.t_len)
        return y


class BinaryRSG(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= False 
        if args.intervals is None:
            t_o = np.random.randint(args.min_t, args.max_t)
        else:
            t_o = random.choice(args.intervals)
        t_p = int(t_o * args.gain)
        ready_time = np.random.randint(args.p_len * 2, args.max_ready)
        set_time = ready_time + t_o
        go_time = set_time + t_p

        self.t_type = args.t_type
        self.p_len = args.p_len
        self.rsg = (ready_time, set_time, go_time)
        self.t_o = t_o
        self.t_p = t_p

        self.L = 1
        self.Z = 1

    def get_x(self, args=None):
        rt, st, gt = self.rsg
        # ready pulse
        x_ready = np.zeros(self.t_len)
        x_ready[rt:rt+self.p_len] = 1
        # set pulse
        x_set = np.zeros(self.t_len)
        x_set[st:st+self.p_len] = 1
        # insert set pulse
        x = np.zeros((1, self.t_len))
        x[0] = x_set
        # perceptual shift
        if args is not None and args.m_noise != 0:
            x_ready = shift_x(x_ready, args.m_noise, self.t_o)
        x[0] += x_ready
        # noisy up/down corruption
        if args is not None and args.x_noise != 0:
            x = corrupt_x(args, x)
        return x

    def get_y(self, args=None):
        y = np.zeros((1, self.t_len))
        #y either 0 or 1, we want it to 1 exactly at  go time and thereafter
        rt, st, gt = self.rsg
        y[0, gt:] = 1 
        return y
    


class CSG(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        if args.intervals is None:
            t_p = np.random.randint(args.min_t, args.max_t)
            t_percentile = (t_p - args.min_t) / (args.max_t - args.min_t)
        else:
            ix = np.random.randint(len(args.intervals))
            t_p = args.intervals[ix]
            t_percentile = ix / len(args.intervals)
        cue_time = np.random.randint(args.p_len * 2, args.max_cue)
        set_time = cue_time + np.random.randint(args.p_len * 2, args.max_cue)
        go_time = set_time + t_p
        assert go_time < self.t_len

        self.t_type = args.t_type
        self.p_len = args.p_len
        self.t_percentile = t_percentile
        self.csg = (cue_time, set_time, go_time)
        self.t_p = t_p

        self.L = 1
        self.Z = 1

    def get_x(self, args=None):
        x = np.zeros((1, self.t_len))
        ct, st, gt = self.csg
        x[0, ct:ct+self.p_len] = 0.5 + 0.5 * self.t_percentile
        x[0, st:st+self.p_len] = 1
        return x

    def get_y(self, args=None):
        y = np.arange(self.t_len)
        slope = 1 / self.t_p
        y = y * slope - self.csg[1] * slope
        y = np.clip(y, 0, 1.5)
        y = y.reshape(1, -1)
        return y

class DelayProAnti(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= True 
        if args.angles is None:
            theta = np.random.random() * 2 * np.pi
        else:
            theta = np.random.choice(args.angles) * np.pi / 180
        stimulus = [np.cos(theta), np.sin(theta)]

        self.t_type = args.t_type
        assert self.t_type in ['delay-pro', 'delay-anti']
        self.stimulus = stimulus
        self.fix = args.fix_t
        
        self.stim = self.fix + args.stim_t
        #stim_t is duration of stimulus period after fixation period
        # when the stimulus period ends and fixation drops to zero and go period begins

        self.L = 3
        self.Z = 3

    def get_x(self, args=None):
        x = np.zeros((3, self.t_len))
        # 0 is fixation, the remaining channels are the directional stimulus
        x[0,:self.stim] = 1
        #up to but not including self.stim, fixate
        x[1,self.fix:] = self.stimulus[0]
        #from and including self.fix time show stimulus until end
        x[2,self.fix:] = self.stimulus[1]
        #from
        return x

    def get_y(self, args=None):
        y = np.zeros((3, self.t_len))
        y[0,:self.stim] = 1
        #when stimulus period ends (at t=self.stim), input stimulus on the output channels
        y[1,self.stim:] = self.stimulus[0]
        y[2,self.stim:] = self.stimulus[1]
        if self.t_type.endswith('anti'):
            y[1:,] = -y[1:,] 
            
        return y

class MemoryProAnti(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= True 
        if args.angles is None:
            theta = np.random.random() * 2 * np.pi
        else:
            theta = np.random.choice(args.angles) * np.pi / 180
        stimulus = [np.cos(theta), np.sin(theta)]

        self.t_type = args.t_type
        assert self.t_type in ['memory-pro', 'memory-anti']
        self.stimulus = stimulus
        self.fix = args.fix_t
        self.stim = self.fix + args.stim_t
        self.memory = self.stim + args.memory_t

        self.L = 3
        self.Z = 3

    def get_x(self, args=None):
        x = np.zeros((3, self.t_len))
        x[0,:self.memory] = 1
        x[1,self.fix:self.stim] = self.stimulus[0]
        x[2,self.fix:self.stim] = self.stimulus[1]
        return x

    def get_y(self, args=None):
        y = np.zeros((3, self.t_len))
        y[0,:self.memory] = 1
        y[1,self.memory:] = self.stimulus[0]
        y[2,self.memory:] = self.stimulus[1]
        # reversing output stimulus for anti condition
        if self.t_type.endswith('anti'):
            y[1:,] = -y[1:,]
        return y
    
# fixation input never goes off and network should respond as soon as the stimulus appears (for now it's the 
class RTProAnti(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= True 
        if args.angles is None:
            theta = np.random.random() * 2 * np.pi
        else:
            theta = np.random.choice(args.angles) * np.pi / 180
        stimulus = [np.cos(theta), np.sin(theta)]

        self.t_type = args.t_type
        assert self.t_type in ['rt-pro', 'rt-anti']
        self.stimulus = stimulus
        self.fix = args.fix_t 
        self.rt_buffer = args.react_t # amount of time between whent then fixation appears and when the stimulus appears
        
        self.stim = self.fix + self.rt_buffer # when the stimulus appears 
        #stim_t is duration of stimulus period after fixation period
        #so that stim is when the stimulus period ends and fixation drops to zero and go period begins

        self.L = 3
        self.Z = 3

    def get_x(self, args=None):
        x = np.zeros((3, self.t_len))
        # fixation input never goes off 
        x[0,self.fix:] = 1
        #up to but not including self.stim, fixate
        x[1,self.stim:] = self.stimulus[0]
        #from and including self.fix time show stimulus until end
        x[2,self.stim:] = self.stimulus[1]
        #from
        return x

    def get_y(self, args=None):
        y = np.zeros((3, self.t_len))
        y[0,self.fix:] = 1
        #when stimulus period ends (at t=self.stim), input stimulus on the output channels
        y[1,self.stim:] = self.stimulus[0]
        y[2,self.stim:] = self.stimulus[1]
        if self.t_type.endswith('anti'):
            y[1:,] = -y[1:,] 
            
        return y


class DelayCopy(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)

        self.s_len = self.t_len // 2
        x_r = np.arange(self.s_len)

        x = np.zeros((args.dim, self.s_len))
            
        freqs = np.random.uniform(args.f_range[0], args.f_range[1], (args.dim, args.n_freqs))
        amps = np.random.uniform(-args.amp, args.amp, (args.dim, args.n_freqs))

        for i in range(args.dim):
            for j in range(args.n_freqs):
                x[i] = x[i] + amps[i,j] * np.sin(1/freqs[i,j] * x_r) / np.sqrt(args.n_freqs)

        self.t_type = args.t_type
        self.dim = args.dim
        self.pattern = x

        self.L = args.dim
        self.Z = args.dim

    def get_x(self, args=None):
        x = np.zeros((self.dim, self.t_len))
        x[:self.dim, :self.s_len] = self.pattern
        return x

    def get_y(self, args=None):
        y = np.zeros((self.dim, self.t_len))
        y[:self.dim, self.s_len:] = self.pattern
        return y

class FlipFlop(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)

        keys = []
        for i in range(args.dim):
            cum_xlen = 0
            # add new dimension
            keys.append([])
            while cum_xlen < self.t_len:
                cum_xlen += np.random.geometric(args.geop) + args.p_len
                if cum_xlen < self.t_len:
                    sign = np.random.choice([-1, 1])
                    keys[i].append(sign * (cum_xlen - args.p_len))

        self.t_type = args.t_type
        self.p_len = args.p_len
        self.dim = args.dim
        self.keys = keys

        self.L = args.dim
        self.Z = args.dim

    def get_x(self, args=None):
        x = np.zeros((self.dim, self.t_len))
        for i in range(self.dim):
            for idx in self.keys[i]:
                x[i, abs(idx):abs(idx)+self.p_len] = np.sign(idx)
        return x

    def get_y(self, args=None):
        y = np.zeros((self.dim, self.t_len))
        for i in range(self.dim):
            for j in range(len(self.keys[i])):
                # the desired key we care about
                idx = self.keys[i][j]
                # the sign to assign to this one
                sign = np.sign(idx)
                if j == len(self.keys[i]) - 1:
                    y[i, np.abs(idx):] = sign
                else:
                    idxs = np.abs(self.keys[i][j:j+2])
                    y[i, idxs[0]:idxs[1]] = sign
        return y

class DurationDisc(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)

        s1_t = np.random.randint(args.tau, args.sep_t - args.max_d - args.tau)
        s1_len, s2_len = np.random.randint(args.min_d, args.max_d, 2)
        s2_t = np.random.randint(args.sep_t + args.tau, args.cue_t - args.max_d - args.tau)

        self.t_type = args.t_type
        self.s1 = [s1_t, s1_len]
        self.s2 = [s2_t, s2_len]
        self.cue_id = np.random.choice([1, -1])
        self.direction = (self.s1[1] < self.s2[1]) ^ (self.cue_id == 1)
        self.cue_t = args.cue_t
        self.select_t = args.select_t

        self.L = 4
        self.Z = 2

    def get_x(self, args=None):
        x = np.zeros((4, self.t_len))
        s1, s1l = self.s1
        s2, s2l = self.s2
        x[0, s1:s1+s1l] = 1
        x[1, s2:s2+s2l] = 1
        if self.cue_id == 1:
            x[2, self.cue_t:] = 1
        else:
            x[3, self.cue_t:] = 1
        return x

    def get_y(self, args=None):
        y = np.zeros((2, self.t_len))
        if self.direction:
            y[0, self.select_t:] = 1
        else:
            y[1, self.select_t:] = 1
        return y



# DM 1 and 2  - 1 as in appears in modality 1.
class DM1ProAnti(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= True 
        #stimulus_1
        if args.angles is None:
            theta_1=np.random.random()*2*np.pi
        else:
            theta_1 = np.random.choice(args.angles)*np.pi/180
            #randomly sammple a value from 0 to arg.angles and convert from degrees to radians
        
        self.stimulus_1=[np.cos(theta_1),np.sin(theta_1)]

        #stimulus 2
        theta_2= np.random.uniform(low=theta_1 + np.pi * 0.5,high= theta_1 + np.pi*1.5)
        self.stimulus_2= [np.cos(theta_2),np.sin(theta_2)]
        #check 1(delete once checked): angles are what they're supposed to be

        self.t_type = args.t_type
        assert self.t_type in ['dm1-pro', 'dm1-anti']


        gamma_mean = np.random.uniform(.8, 1.2)
        coherence_arr= [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08]
        
        coherence=np.random.choice(coherence_arr)
        coherence=np.random.choice(coherence_arr)
        
        self.g1 = gamma_mean + coherence
        self.g2 = gamma_mean - coherence

        #duration of stimulus 1
        self.fix = args.fix_t # fixaton duration and self.fix can also be point when self.fix ends
    
        #stim_t= random.choice([400, 800, 1600])
        #used fixed stim_t
        self.stim = self.fix + args.stim_t

        

        



        self.L = 13
        self.Z = 3
    
    def get_x(self,args=None):
        x=np.zeros((13,self.t_len))
        
        x[0, :self.stim]=1

        # modality 1 
        #stimulus 1
        x[1, :]=self.stimulus_1[0]
        x[2, :] = self.stimulus_1[1]
        x[3, :] =self.g1
        #stimulus 2
        x[4, :]=self.stimulus_2[0]
        x[5, :] =self.stimulus_2[1]
        x[6, :] =self.g2


        #x[7: .. ], ... x[12:] are just zero rows
        
        
        return x 


    def get_y(self,args=None):
        y=np.zeros((3,self.t_len))
        
        y[0,:self.stim]= 1
        #fixate until stim period ends (i.e until when go period begins)

        if self.g1 > self.g2:
            y[1,self.stim:] = self.stimulus_1[0]
            y[2,self.stim:] = self.stimulus_1[1]
        
        elif self.g1 < self.g2:
            y[1,self.stim:] = self.stimulus_2[0]
            y[2,self.stim:] =self.stimulus_2[1]



        
        if self.t_type.endswith('anti'):
            if self.g1 > self.g2:
                y[1,self.stim:] = self.stimulus_2[0]
                y[2,self.stim:] = self.stimulus_2[1]
        
            elif self.g1 < self.g2:
                y[1,self.stim:] = self.stimulus_1[0]
                y[2,self.stim:] =self.stimulus_1[1]
            
        return y



class DM2ProAnti(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= True 
        #stimulus_1
        if args.angles is None:
            theta_1=np.random.random()*2*np.pi
        else:
            theta_1 = np.random.choice(args.angles)*np.pi/180
            #randomly sammple a value from 0 to arg.angles and convert from degrees to radians
        
        self.stimulus_1=[np.cos(theta_1),np.sin(theta_1)]

        #stimulus 2
        theta_2= np.random.uniform(low=theta_1 + np.pi * 0.5,high= theta_1 + np.pi*1.5)
        self.stimulus_2= [np.cos(theta_2),np.sin(theta_2)]
        #check 1(delete once checked): angles are what they're supposed to be

        self.t_type = args.t_type
        assert self.t_type in ['dm2-pro', 'dm2-anti']


        gamma_mean = np.random.uniform(.8, 1.2)
        coherence_arr= [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08]
        
        coherence=np.random.choice(coherence_arr)
        coherence=np.random.choice(coherence_arr)
        
        self.g1 = gamma_mean + coherence
        self.g2 = gamma_mean - coherence

        #duration of stimulus 1
        self.fix = args.fix_t # fixaton duration and self.fix can also be point when self.fix ends
    
        #stim_t= random.choice([400, 800, 1600])
        #used fixed stim_t
        self.stim = self.fix + args.stim_t

        

        



        self.L = 13
        self.Z = 3
    
    def get_x(self,args=None):
        x=np.zeros((13,self.t_len))
        
        x[0, :self.stim]=1
        
        
        # modality 1: x[1: .. ], ... x[6:] are just zero rows

        # modality 2
        #stimulus 1
        x[7, :]=self.stimulus_1[0]
        x[8, :] = self.stimulus_1[1]
        x[9, :] =self.g1
        #stimulus 2
        x[10, :]=self.stimulus_2[0]
        x[11, :] =self.stimulus_2[1]
        x[12, :] =self.g2


       
        
        
        return x 


    def get_y(self,args=None):
        y=np.zeros((3,self.t_len))
        
        y[0,:self.stim]= 1
        #fixate until stim period ends (i.e until when go period begins)

        if self.g1 > self.g2:
            y[1,self.stim:] = self.stimulus_1[0]
            y[2,self.stim:] = self.stimulus_1[1]
        
        elif self.g1 < self.g2:
            y[1,self.stim:] = self.stimulus_2[0]
            y[2,self.stim:] =self.stimulus_2[1]



        
        if self.t_type.endswith('anti'):
            if self.g1 > self.g2:
                y[1,self.stim:] = self.stimulus_2[0]
                y[2,self.stim:] = self.stimulus_2[1]
        
            elif self.g1 < self.g2:
                y[1,self.stim:] = self.stimulus_1[0]
                y[2,self.stim:] =self.stimulus_1[1]
            
        return y

# combines two DM inputs so two modalities (two random dot motion screens)
# if CtxDM1  task is to ignore what happens in the second screen (hence the '1' in 'CtxDM1'), respond in 
#direction of motion with greater coherence in modality 1, ignore modality 2 
class CtxDM(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= True 
        #stimulus_1
        if args.angles is None:
            theta_1=np.random.random()*2*np.pi
        else:
            theta_1 = np.random.choice(args.angles)*np.pi/180
            #randomly sammple a value from 0 to arg.angles and convert from degrees to radians
        
        self.stimulus_1=[np.cos(theta_1),np.sin(theta_1)]

        #stimulus 2
        theta_2= np.random.uniform(low=theta_1 + np.pi * 0.5,high= theta_1 + np.pi*1.5)
        self.stimulus_2= [np.cos(theta_2),np.sin(theta_2)]
        #check 1(delete once checked): angles are what they're supposed to be

        self.t_type = args.t_type
        


        
        coherence_arr= [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08]
        

        gamma_mean_1_mod_1= np.random.uniform(.8, 1.2) 
        gamma_mean_1_mod_2= np.random.uniform(.8, 1.2) 

        gamma_mean_2_mod_1= np.random.uniform(.8, 1.2) 
        gamma_mean_2_mod_2= np.random.uniform(.8, 1.2) 

        
        
        coherence_mod_1 =np.random.choice(coherence_arr)
        coherence_mod_2 =np.random.choice(coherence_arr)
        
        self.g1_mod_1= gamma_mean_1_mod_1 + coherence_mod_1  #g1 as in stimulus 1
        self.g1_mod_2= gamma_mean_1_mod_2 + coherence_mod_2 

        self.g2_mod_1= gamma_mean_2_mod_1 + coherence_mod_1  
        self.g2_mod_2= gamma_mean_2_mod_2 + coherence_mod_2


       



        #duration of stimulus 1
        self.fix = args.fix_t # fixaton duration and self.fix can also be point when self.fix ends
    
        #stim_t= random.choice([400, 800, 1600])
        #used fixed stim_t
        self.stim = self.fix + args.stim_t

        

        



        self.L = 13
        self.Z = 3
    
    def get_x(self,args=None):
        x=np.zeros((13,self.t_len))
        
        x[0, :self.stim]=1
        # modality 1

        #stimulus 1
        x[1, :]=self.stimulus_1[0]
        x[2, :] = self.stimulus_1[1]
        x[3, :] = self.g1_mod_1

        #stimulus 2

        x[4, :]=self.stimulus_2[0]
        x[5, :] =self.stimulus_2[1]
        x[6, :] = self.g2_mod_1


        # modality 2

        #stimulus 1

        x[7, :]=self.stimulus_1[0]
        x[8, :] = self.stimulus_1[1]
        x[9, :] =self.g1_mod_2

        #stimulus 2
        x[10, :]=self.stimulus_2[0]
        x[11, :] =self.stimulus_2[1]
        x[12, :] =self.g2_mod_2
        return x 


    def get_y(self,args=None):
        y=np.zeros((3,self.t_len))
        
        y[0,:self.stim]= 1
        #fixate until stim period ends (i.e until when go period begins)
        if self.t_type.endswith('1'):
            if self.g1_mod_1 > self.g2_mod_1:
                y[1,self.stim:] = self.stimulus_1[0]
                y[2,self.stim:] = self.stimulus_1[1]
            
            elif self.g1_mod_1 < self.g2_mod_1:
                y[1,self.stim:] = self.stimulus_2[0]
                y[2,self.stim:] =self.stimulus_2[1]

        elif self.t_type.endswith('2'):
            if self.g1_mod_2 > self.g2_mod_2:
                y[1,self.stim:] = self.stimulus_1[0]
                y[2,self.stim:] = self.stimulus_1[1]
        
            elif self.g1_mod_2  < self.g2_mod_2:
                y[1,self.stim:] = self.stimulus_2[0]
                y[2,self.stim:] =self.stimulus_2[1]
            
        return y


# mutlisensory integration task
class MultiSenDM(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= True 
        #stimulus_1
        if args.angles is None:
            theta_1=np.random.random()*2*np.pi
        else:
            theta_1 = np.random.choice(args.angles)*np.pi/180
            #randomly sammple a value from 0 to arg.angles and convert from degrees to radians
        
        self.stimulus_1=[np.cos(theta_1),np.sin(theta_1)]

        #stimulus 2
        theta_2= np.random.uniform(low=theta_1 + np.pi * 0.5,high= theta_1 + np.pi*1.5)
        self.stimulus_2= [np.cos(theta_2),np.sin(theta_2)]
        #check 1(delete once checked): angles are what they're supposed to be

        self.t_type = args.t_type
        


        
        coherence_arr= [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08]
        

        gamma = np.random.uniform(.8, 1.2) 
        coherence =np.random.choice(coherence_arr)
        
        self.gamma_1 = gamma + coherence
        self.gamma_2 = gamma - coherence

        delta_1 = self.union_of_uniforms_sampler()
        delta_2 = self.union_of_uniforms_sampler()

        
        
        self.g1_mod_1= self.gamma_1 * (1 + delta_1)
        self.g1_mod_2= self.gamma_1*(1-delta_1)

        self.g2_mod_1= self.gamma_2 * (1+delta_2)
        self.g2_mod_2= self.gamma_2 * (1+delta_2)


       



        #duration of stimulus 1
        self.fix = args.fix_t # fixaton duration and self.fix can also be point when self.fix ends
    
        #stim_t= random.choice([400, 800, 1600])
        #used fixed stim_t
        self.stim = self.fix + args.stim_t

        

        



        self.L = 13
        self.Z = 3


    def union_of_uniforms_sampler(self):
        sample = np.random.rand(1)  # draw from U(0,1)

        #map sample to one of two intervals
        if sample < 0.5:
            sample = 0.1 + sample * 0.3 # map to a sample from U(0.1,0.4)
        else:
            sample = -0.4 + (sample - 0.5) * 0.3 #map to a sample from U(-0.4,-0.1)
        return sample
    
    def get_x(self,args=None):
        x=np.zeros((13,self.t_len))
        
        x[0, :self.stim]=1
        # modality 1

        #stimulus 1
        x[1, :]=self.stimulus_1[0]
        x[2, :] = self.stimulus_1[1]
        x[3, :] = self.g1_mod_1

        #stimulus 2

        x[4, :]=self.stimulus_2[0]
        x[5, :] =self.stimulus_2[1]
        x[6, :] = self.g2_mod_1


        # modality 2

        #stimulus 1

        x[7, :]=self.stimulus_1[0]
        x[8, :] = self.stimulus_1[1]
        x[9, :] =self.g1_mod_2

        #stimulus 2
        x[10, :]=self.stimulus_2[0]
        x[11, :] =self.stimulus_2[1]
        x[12, :] =self.g2_mod_2
        return x 


    def get_y(self,args=None):
        y=np.zeros((3,self.t_len))
        
        y[0,:self.stim]= 1
        #fixate until stim period ends (i.e until when go period begins)
        
        if self.gamma_1> self.gamma_2:
            y[1,self.stim:] = self.stimulus_1[0]
            y[2,self.stim:] = self.stimulus_1[1]
        
        elif self.gamma_1 < self.gamma_2:
            y[1,self.stim:] = self.stimulus_2[0]
            y[2,self.stim:] =self.stimulus_2[1]

        
        return y



class DMProAnti(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= True 
        #stimulus_1
        if args.angles is None:
            theta_1=np.random.random()*2*np.pi
        else:
            theta_1 = np.random.choice(args.angles)*np.pi/180
            #randomly sammple a value from 0 to arg.angles and convert from degrees to radians
        
        self.stimulus_1=[np.cos(theta_1),np.sin(theta_1)]

        #stimulus 2
        theta_2= np.random.uniform(low=theta_1 + np.pi * 0.5,high= theta_1 + np.pi*1.5)
        self.stimulus_2= [np.cos(theta_2),np.sin(theta_2)]
        #check 1(delete once checked): angles are what they're supposed to be

        self.t_type = args.t_type
        assert self.t_type in ['dm-pro', 'dm-anti']


        gamma_mean = np.random.uniform(.8, 1.2)
        coherence_arr= [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08]
        
        coherence=np.random.choice(coherence_arr)
        coherence=np.random.choice(coherence_arr)
        
        self.g1 = gamma_mean + coherence
        self.g2 = gamma_mean - coherence

        #duration of stimulus 1
        self.fix = args.fix_t # fixaton duration and self.fix can also be point when self.fix ends
    
        #stim_t= random.choice([400, 800, 1600])
        #used fixed stim_t
        self.stim = self.fix + args.stim_t

        

        



        self.L = 7
        self.Z = 3
    
    def get_x(self,args=None):
        x=np.zeros((7,self.t_len))
        
        x[0, :self.stim]=1
        #stimulus 1
        x[1, :]=self.stimulus_1[0]
        x[2, :] = self.stimulus_1[1]
        x[3, :] =self.g1
        #stimulus 2
        x[4, :]=self.stimulus_2[0]
        x[5, :] =self.stimulus_2[1]
        x[6, :] =self.g2
        
        #I think  I have a way to update the matrices if a new task modality added that keeps the old weights and keeps them in the right position and still keeps the dimension D1 the same.
        return x 


    def get_y(self,args=None):
        y=np.zeros((3,self.t_len))
        
        y[0,:self.stim]= 1
        #fixate until stim period ends (i.e until when go period begins)

        if self.g1 > self.g2:
            y[1,self.stim:] = self.stimulus_1[0]
            y[2,self.stim:] = self.stimulus_1[1]
        
        elif self.g1 < self.g2:
            y[1,self.stim:] = self.stimulus_2[0]
            y[2,self.stim:] =self.stimulus_2[1]



        
        if self.t_type.endswith('anti'):
            if self.g1 > self.g2:
                y[1,self.stim:] = self.stimulus_2[0]
                y[2,self.stim:] = self.stimulus_2[1]
        
            elif self.g1 < self.g2:
                y[1,self.stim:] = self.stimulus_1[0]
                y[2,self.stim:] =self.stimulus_1[1]
            
        return y



class DelayDM(Task):
    def __init__(self, args, dset_id=None, n=None):
        
        
        super().__init__(args.t_len, dset_id, n)
        #stimulus_1
        self.has_fix= True 
        if args.angles is None:
            theta_1=np.random.random()*2*np.pi
        else:
            theta_1 = np.random.choice(args.angles)*np.pi/180
            #randomly sammple a value from 0 to arg.angles and convert from degrees to radians
        
        self.stimulus_1=[np.cos(theta_1),np.sin(theta_1)]

        #stimulus 2
        theta_2= np.random.uniform(low=theta_1 + np.pi * 0.5,high= theta_1 + np.pi*1.5)
        self.stimulus_2= [np.cos(theta_2),np.sin(theta_2)]
        #check 1(delete once checked): angles are what they're supposed to be

        self.t_type = args.t_type
        


        gamma_mean = np.random.uniform(.8, 1.2)
        coherence_arr= [-0.32, -0.16, -0.08, 0.08, 0.16, 0.32]
        
        coherence=np.random.choice(coherence_arr)
        
        self.g1 = gamma_mean + coherence
        self.g2 = gamma_mean - coherence

        #duration of stimulus 1
        self.fix = args.fix_t # fixaton duration and self.fix can also be point when self.fix ends
    
        
        

        self.stim1=args.stim_t1 #we'll define args.stim_t1
        #we show stimulus 1 up until this point self.stim_1
        self.delay1=self.stim1+args.delay_t1
        self.stim_2=self.delay1+args.stim_t2 
        #note:delay_t1 is the length of the delay self.delay1 is when the first delay ends in the trial

        self.stim=self.stim_2+args.delay_t2 #generate saccade after second delay 
        #second delay is the delay after 2nd stimulus is shown
        
        
        #point where

        



        self.L = 7
        self.Z = 3
    
    def get_x(self,args=None):
        x=np.zeros((7,self.t_len))
        
        x[0,:self.stim]=1
        #stimulus 1
        x[1, :self.stim1]=self.stimulus_1[0]
        x[2, :self.stim1] = self.stimulus_1[1]
        x[3, :self.stim1] =self.g1
        #stimulus 2
        x[4, self.delay1:self.stim_2]=self.stimulus_2[0]
        x[5, self.delay1:self.stim_2:] =self.stimulus_2[1]
        x[6, self.delay1:self.stim_2] =self.g2
        
        #I think  I have a way to update the matrices if a new task modality added that keeps the old weights and keeps them in the right position and still keeps the dimension D1 the same.
        
        return x 


    def get_y(self,args=None):
        y=np.zeros((3,self.t_len))
        
        y[0,:self.stim]= 1
        #fixate until stim period ends (i.e until when go period begins
        
        if self.g1 > self.g2:
            y[1,self.stim:] = self.stimulus_1[0]
            y[2,self.stim:] = self.stimulus_1[1]
        
        elif self.g1 < self.g2:
            y[1,self.stim:] = self.stimulus_2[0]
            y[2,self.stim:] =self.stimulus_2[1]



        
        
            
        return y



class DelayDM1(DelayDM):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args, dset_id, n)
        self.L = 13 #  two modalities 
        self.Z = 3
    
    
    def get_x(self,args =None):
        
        
        x=np.zeros((13,self.t_len))
        
        
        
        #stimulus 1

        x[1, :self.stim1]=self.stimulus_1[0]
        x[2, :self.stim1] = self.stimulus_1[1]
        x[3, :self.stim1] =self.g1
        #stimulus 2
        x[4, self.delay1:self.stim_2]=self.stimulus_2[0]
        x[5, self.delay1:self.stim_2:] =self.stimulus_2[1]
        x[6, self.delay1:self.stim_2] =self.g2
        
        return x 


    def get_y(self,args=None):
        y=np.zeros((3,self.t_len))
        
        y[0,:self.stim]= 1
        #fixate until stim period ends (i.e until when go period begins)


       
        if self.g1 > self.g2:
            y[1,self.stim:] = self.stimulus_1[0]
            y[2,self.stim:] = self.stimulus_1[1]
        
        elif self.g1 < self.g2:
            y[1,self.stim:] = self.stimulus_2[0]
            y[2,self.stim:] =self.stimulus_2[1]

        return y 

class DelayDM2(DelayDM):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args, dset_id, n)
        self.L = 13 #  two modalities 
        self.Z = 3
    
    
    def get_x(self,args =None):
        
        
        x=np.zeros((13,self.t_len))
        
        
        
        #stimulus 1

        x[1+6, :self.stim1]=self.stimulus_1[0]
        x[2+6, :self.stim1] = self.stimulus_1[1]
        x[3+6, :self.stim1] =self.g1
        #stimulus 2
        x[4+6, self.delay1:self.stim_2]=self.stimulus_2[0]
        x[5+6, self.delay1:self.stim_2:] =self.stimulus_2[1]
        x[6+6, self.delay1:self.stim_2] =self.g2

        return x 


    def get_y(self,args=None):
        y=np.zeros((3,self.t_len))
        
        y[0,:self.stim]= 1
        #fixate until stim period ends (i.e until when go period begins)


       
        if self.g1 > self.g2:
            y[1,self.stim:] = self.stimulus_1[0]
            y[2,self.stim:] = self.stimulus_1[1]
        
        elif self.g1 < self.g2:
            y[1,self.stim:] = self.stimulus_2[0]
            y[2,self.stim:] =self.stimulus_2[1]

        return y 


class CtxDelayDM(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        
        self.L = 13
        self.Z = 3
        self.has_fix= True 
        #stimulus_1
        if args.angles is None:
            theta_1=np.random.random()*2*np.pi
        else:
            theta_1 = np.random.choice(args.angles)*np.pi/180
            #randomly sammple a value from 0 to arg.angles and convert from degrees to radians
        
        self.stimulus_1=[np.cos(theta_1),np.sin(theta_1)]

        #stimulus 2
        theta_2= np.random.uniform(low=theta_1 + np.pi * 0.5,high= theta_1 + np.pi*1.5)
        self.stimulus_2= [np.cos(theta_2),np.sin(theta_2)]
        #check 1(delete once checked): angles are what they're supposed to be

        self.t_type = args.t_type
        


        
        coherence_arr= [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08]
        

        gamma_mean_1_mod_1= np.random.uniform(.8, 1.2) 
        gamma_mean_1_mod_2= np.random.uniform(.8, 1.2) 

        gamma_mean_2_mod_1= np.random.uniform(.8, 1.2) 
        gamma_mean_2_mod_2= np.random.uniform(.8, 1.2) 

        
        
        coherence_mod_1 =np.random.choice(coherence_arr)
        coherence_mod_2 =np.random.choice(coherence_arr)
        
        self.g1_mod_1= gamma_mean_1_mod_1 + coherence_mod_1  #g1 as in stimulus 1
        self.g1_mod_2= gamma_mean_1_mod_2 + coherence_mod_2 

        self.g2_mod_1= gamma_mean_2_mod_1 + coherence_mod_1  
        self.g2_mod_2= gamma_mean_2_mod_2 + coherence_mod_2

        # duration of stimulus 1
        self.fix = args.fix_t # fixaton duration and self.fix can also be point when self.fix ends
    
        
        

        self.stim1= args.stim_t1 #we'll define args.stim_t1
        #we show stimulus 1 up until this point self.stim_1
        self.delay1=self.stim1+args.delay_t1
        self.stim_2=self.delay1+args.stim_t2 
        #note:delay_t1 is the length of the delay self.delay1 is when the first delay ends in the trial

        self.stim=self.stim_2+args.delay_t2

    
    def get_x(self,args=None):
        x=np.zeros((13,self.t_len))
        
        x[0, :self.stim]=1
        # modality 1

        #stimulus 1
        x[1, :self.stim1]=self.stimulus_1[0]
        x[2, :self.stim1] = self.stimulus_1[1]
        x[3, :self.stim1] = self.g1_mod_1

        #stimulus 2

        x[4,self.delay1 :self.stim_2]=self.stimulus_2[0]
        x[5,self.delay1 :self.stim_2] =self.stimulus_2[1]
        x[6, self.delay1:self.stim_2] = self.g2_mod_1


        # modality 2

        #stimulus 1

        x[7, :self.stim1]=self.stimulus_1[0]
        x[8, :self.stim1] = self.stimulus_1[1]
        x[9, :self.stim1] =self.g1_mod_2

        #stimulus 2
        x[10, self.delay1:self.stim_2]=self.stimulus_2[0]
        x[11, self.delay1:self.stim_2] =self.stimulus_2[1]
        x[12, self.delay1:self.stim_2] =self.g2_mod_2
        return x 


    def get_y(self,args=None):
        y=np.zeros((3,self.t_len))
        
        y[0,:self.stim]= 1
        #fixate until stim period ends (i.e until when go period begins)
        if self.t_type.endswith('1'):
            if self.g1_mod_1 > self.g2_mod_1:
                y[1,self.stim:] = self.stimulus_1[0]
                y[2,self.stim:] = self.stimulus_1[1]
            
            elif self.g1_mod_1 < self.g2_mod_1:
                y[1,self.stim:] = self.stimulus_2[0]
                y[2,self.stim:] =self.stimulus_2[1]

        elif self.t_type.endswith('2'):
            if self.g1_mod_2 > self.g2_mod_2:
                y[1,self.stim:] = self.stimulus_1[0]
                y[2,self.stim:] = self.stimulus_1[1]
        
            elif self.g1_mod_2  < self.g2_mod_2:
                y[1,self.stim:] = self.stimulus_2[0]
                y[2,self.stim:] =self.stimulus_2[1]
            
        return y

#DMC stands for delay match to category , DMS delay match to stimuli (bc same direction) N stands for not - we'll use pro for match and anti for not match 



# can generate a DNMC by using dmc-anti
class DMCProAnti(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        #stimulus_1
        self.has_fix= True #info for mm task has a fixation input

        angles_arr = [18,54,90,126,162,198,234,270,306,342]
        self.theta_1=np.random.choice(angles_arr)
        self.theta_1*=np.pi/180
        self.stimulus_1=[np.cos(self.theta_1),np.sin(self.theta_1)]

        #stimulus 2
        self.theta_2=np.random.choice(angles_arr)
        
        self.theta_2*=np.pi/180

        self.stimulus_2= [np.cos(self.theta_2),np.sin(self.theta_2)]

        
        
        #check 1(delete once checked): angles are what they're supposed to be

        self.t_type = args.t_type
        assert self.t_type in ['dmc-pro', 'dmc-anti']

        
        

        #duration of stimulus 1
        self.fix = args.fix_t # fixaton duration and self.fix can also be point when self.fix ends
    
        
        

        self.stim1=self.fix+args.stim_t1 #we'll define args.stim_t1
        #we show stimulus 1 up until this point self.stim_1
        self.delay1=self.stim1+args.delay_t1
        self.stim_2=self.delay1+args.stim_t2 
        #note:delay_t1 is the length of the delay self.delay1 is when the first delay ends in the trial

        self.stim=self.stim_2 
        
        




        self.L = 5
        self.Z = 3
    
    def get_x(self,args=None):
        x=np.zeros((5,self.t_len))
        
        x[0,:self.stim]=1
        #stimulus 1
        x[1, self.fix:self.stim1]=self.stimulus_1[0]
        x[2,self.fix:self.stim1] = self.stimulus_1[1]
        
        #stimulus 2
        x[3, self.delay1:self.stim_2]=self.stimulus_2[0]
        x[4,self.delay1:self.stim_2] =self.stimulus_2[1]
        
        
        #I think  I have a way to update the matrices if a new task modality added that keeps the old weights and keeps them in the right position and still keeps the dimension D1 the same.
        
       

        return x 


    def get_y(self,args=None):
        y=np.zeros((3,self.t_len))
        
        if self.t_type.endswith('pro'):
            if ((0<=self.theta_1 <= np.pi) and (0<=self.theta_2<=np.pi) ) or ((np.pi <self.theta_1 <= 2*np.pi) and (np.pi<self.theta_2<= 2*np.pi)):
                y[0,:self.stim]= 1
                y[1,self.stim:] = self.stimulus_2[0]
                y[2,self.stim:] =self.stimulus_2[1]
        #stop fixating and responsd after self.stim if in same category otherwise hold fixation
        
            else:
                y[0]=1
            

        elif self.t_type.endswith('anti'):
            #if two angles aren't in the same category
            if ((0<=self.theta_1 <= np.pi) and (np.pi<self.theta_2<= 2*np.pi)) or  ((np.pi <self.theta_1 <= 2*np.pi) and (0<=self.theta_2<=np.pi)):
                y[0,:self.stim]= 1
                y[1,self.stim:] = self.stimulus_2[0]
                y[2,self.stim:] =self.stimulus_2[1]
            
        #stop fixating after self.stim if in same category otherwise hold fixation
        
            else:
                y[0]=1
                

            
        return y
    

# Mod Cog tasks
# default task args will be an anti-clockwise shift but we'll include a displacement direction argument --displ_direction clockwise 
class DelayGo_IntL(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        self.has_fix= True 
        if args.angles is None:
            theta = np.random.random() * 2 * np.pi
        else:
            theta = np.random.choice(args.angles) * np.pi / 180
        stimulus = [np.cos(theta), np.sin(theta)]


        # new arg for endpoints of uniform distribution of memory period lengths

        memory_t = np.random.uniform(args.min_mem_t,args.max_mem_t)
        
        self.t_type = args.t_type
        assert self.t_type in ['memory-pro', 'memory-anti']
        self.stimulus = stimulus
        self.fix = args.fix_t
        self.stim = self.fix + args.stim_t
        self.memory = self.stim + memory_t

        assert args.displacement_direction in ['anti-clockwise', 'clockwise']
        # now shift theta based on length of delay
        delta_theta = args.delay_scalar * memory_t
        
        
        
        if args.displacement_direction == 'anti-clockwise':
            theta_out = theta + delta_theta
        else: #clockwise
            theta_out = theta - delta_theta
        
        theta_out = theta_out % (2 * np.pi) 
        self.response = [np.cos(theta_out), np.sin(theta_out)]

        self.L = 3
        self.Z = 3

    def get_x(self, args=None):
        x = np.zeros((3, self.t_len))
        x[0,:self.memory] = 1
        x[1,self.fix:self.stim] = self.stimulus[0]
        x[2,self.fix:self.stim] = self.stimulus[1]
        return x

    def get_y(self, args=None):
        y = np.zeros((3, self.t_len))
        y[0,:self.memory] = 1
        y[1,self.memory:] = self.response[0]
        y[2,self.memory:] = self.response[1]
        # reversing output stimulus for anti condition
        if self.t_type.endswith('anti'):
            y[1:,] = -y[1:,]
        return y








# ways to add noise to x
def corrupt_x(args, x):
    x += np.random.normal(scale=args.x_noise, size=x.shape)
    return x

def shift_x(x, m_noise, t_p):
    #ways to add noise to what the reservoir observes
    #i.e slightly noisy signal
    if m_noise == 0:
        return x
    disp = int(np.random.normal(0, m_noise*t_p/50))
    x = np.roll(x, disp)
    return x

def create_dataset(args):
    t_type = args.t_type
    n_trials = args.n_trials
    
    if t_type == 'binary-rsg':
        assert args.max_ready + args.max_t + int(args.max_t * args.gain) < args.t_len
        TaskObj = BinaryRSG
    elif t_type.startswith('rsg'):
        assert args.max_ready + args.max_t + int(args.max_t * args.gain) < args.t_len
        TaskObj = RSG
    elif t_type.startswith('csg'):
        TaskObj = CSG
    elif t_type == 'delay-copy':
        TaskObj = DelayCopy
    elif t_type == 'flip-flop':
        TaskObj = FlipFlop
    elif t_type == 'delay-pro' or t_type == 'delay-anti':
        assert args.fix_t + args.stim_t < args.t_len
        TaskObj = DelayProAnti
    elif t_type == 'memory-pro' or t_type == 'memory-anti':
        assert args.fix_t + args.stim_t + args.memory_t < args.t_len
        TaskObj = MemoryProAnti
    elif t_type == 'rt-pro' or t_type == 'rt-anti':
        assert args.fix_t + args.stim_t < args.t_len
        TaskObj = RTProAnti
    elif t_type == 'ctx-delay-dm1' or 'ctx-delay-dm2':
        TaskObj = CtxDelayDM
    elif t_type == 'delay-dm1':
        TaskObj = DelayDM1
    elif t_type == 'delay-dm2':
        TaskObj = DelayDM2
    elif t_type == 'delay-dm':
        TaskObj = DelayDM
    
    elif t_type == 'dm-pro' or 'dm-anti':
        TaskObj = DMProAnti
    elif t_type =='multisen-dm':
        TaskObj = MultiSenDM
    
    elif t_type == 'ctx-dm1' or 'ctx-dm2':
        TaskObj = CtxDM
    elif t_type == 'dm1-pro' or 'dm1-anti':
        TaskObj =DM1ProAnti
    elif t_type == 'dm2-pro' or 'dm2-anti':
        TaskObj =DM2ProAnti
    
    elif t_type == 'delay-go-interval':
        TaskObj = DelayGo_IntL
    
    elif t_type == 'dur-disc':
        assert args.tau + args.max_d <= args.sep_t
        assert args.sep_t + args.tau + args.max_d <= args.cue_t
        TaskObj = DurationDisc
    else:
        raise NotImplementedError

    trials = []
    for n in range(n_trials):
        trial = TaskObj(args, dset_id=args.name, n=n)
        
        args.L = trial.L
        args.Z = trial.Z
        trials.append(trial)
        #add an input-output pair to trials 
        #and recall the input pairs are generated randomly
        #so they'll likely be distinct input-output pairs
        # this is our dataset of stored in "trials"
        #we will train on these

    return trials, args
    

# turn task_args argument into usable argument variables
# lots of defaults are written down here
def get_task_args(args):
    #takes int args after we passed in the config file etc
    tarr = args.task_args
    

    targs = Bunch()
    #create an empty bunched dictionary to be populated

    if args.t_type.startswith('rsg') or args.t_type.startswith('binary'):
        targs.has_fix = get_tval(tarr,'has_fix',False, bool)
        targs.t_len = get_tval(tarr, 'l', 600, int)
        targs.p_len = get_tval(tarr, 'pl', 5, int)
        targs.gain = get_tval(tarr, 'gain', 1, float)
        targs.max_ready = get_tval(tarr, 'max_ready', 80, int)
        
        #recall: we can specify interval in command line parse using -i
        if args.intervals is None:
            targs.min_t = get_tval(tarr, 'gt', targs.p_len * 4, int)
            targs.max_t = get_tval(tarr, 'lt', targs.t_len // 2 - targs.p_len * 4 - targs.max_ready, int)
        else:
            targs.max_t = max(args.intervals)
            targs.min_t = min(args.intervals)

    elif args.t_type.startswith('csg'):
        targs.t_len = get_tval(tarr, 'l', 600, int)
        targs.p_len = get_tval(tarr, 'pl', 5, int)
        targs.max_cue = get_tval(tarr, 'max_cue', 100, int)
        targs.max_set = get_tval(tarr, 'max_set', 300, int)
        if args.intervals is None:
            targs.min_t = get_tval(tarr, 'gt', targs.p_len * 4, int)
            targs.max_t = get_tval(tarr, 'lt', targs.t_len // 2 - targs.p_len * 4, int)

    elif args.t_type == 'delay-copy':
        targs.t_len = get_tval(tarr, 'l', 500, int)
        targs.dim = get_tval(tarr, 'dim', 2, int)
        targs.n_freqs = get_tval(tarr, 'n_freqs', 20, int)
        targs.f_range = get_tval(tarr, 'f_range', [10, 40], float, n_vals=2)
        targs.amp = get_tval(tarr, 'amp', 1, float)

    elif args.t_type == 'flip-flop':
        targs.t_len = get_tval(tarr, 'l', 500, int)
        targs.dim = get_tval(tarr, 'dim', 3, int)
        targs.p_len = get_tval(tarr, 'pl', 5, int)
        targs.geop = get_tval(tarr, 'p', .02, float)

    elif args.t_type == 'delay-pro' or args.t_type == 'delay-anti':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 150, int)

    elif args.t_type == 'rt-pro' or args.t_type == 'rt-anti':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.react_t = get_tval(tarr,'react',20, int) #time in between fix_t and stimulus onset
        targs.stim_t = get_tval(tarr, 'stim', 150, int)

    elif args.t_type == 'memory-pro' or args.t_type == 'memory-anti':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 100, int)
        targs.memory_t = get_tval(tarr, 'memory', 50, int)

    elif args.t_type == 'dur-disc':
        targs.t_len = get_tval(tarr, 'l', 600, int)
        targs.tau = get_tval(tarr, 'tau', 10, int)
        targs.min_d = get_tval(tarr, 'gt', 10, int)
        targs.max_d = get_tval(tarr, 'lt', 80, int)
        targs.sep_t = get_tval(tarr, 'sep_t', 150, int)
        targs.cue_t = get_tval(tarr, 'cue_t', 400, int)
        targs.select_t = get_tval(tarr, 'select_t', 440, int)

    elif args.t_type == 'dm-pro' or args.t_type == 'dm-anti':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        #default value of t_len is 300 according to this but doesn't do anything atm
        #bc for now t_len in dm is determined by stimulus duration
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 100, int)
        targs.has_fix = get_tval(tarr, 'has_fix', True, bool)
    
    elif args.t_type == 'dm1-pro' or args.t_type == 'dm1-anti':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        #default value of t_len is 300 according to this but doesn't do anything atm
        #bc for now t_len in dm is determined by stimulus duration
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 100, int)
        targs.has_fix = get_tval(tarr, 'has_fix', True, bool)

    elif args.t_type == 'dm2-pro' or args.t_type == 'dm2-anti':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        #default value of t_len is 300 according to this but doesn't do anything atm
        #bc for now t_len in dm is determined by stimulus duration
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 100, int)
        targs.has_fix = get_tval(tarr, 'has_fix', True, bool)
    
    elif args.t_type == 'delay-dm' or args.t_type =='delay-dm1' or args.t_type == 'delay-dm2' or args.t_type =='ctx-delay-dm1' or args.t_type == 'ctx-delay-dm2':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        #default value of t_len is 300 according to this but doesn't do anything atm
        #bc for now t_len in dm is determined by stimulus duration
        targs.fix_t = get_tval(tarr, 'fix', 33, int)
        targs.stim_t1 = get_tval(tarr, 'stimt1', 75, int)
        targs.delay_t1=get_tval(tarr,'delayl1',15, int)
        targs.stim_t2 = get_tval(tarr, 'stimt2', 75, int)
        targs.delay_t2=get_tval(tarr,'delayl1', 15, int)
        targs.has_fix = get_tval(tarr, 'has_fix', True, bool)


    elif args.t_type == 'ctx-dm1' or args.t_type == 'ctx-dm2':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 100, int)
        targs.has_fix = get_tval(tarr, 'has_fix', True, bool)


    elif args.t_type == 'multisen-dm':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 100, int)
        targs.has_fix = get_tval(tarr, 'has_fix', True, bool)


    #use dmc anti for DNMC 
    elif args.t_type == 'dmc-pro' or args.t_type == 'dmc-anti':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        #default value of t_len is 300 according to this but doesn't do anything atm
        #bc for now t_len in dm is determined by stimulus duration
        targs.fix_t = get_tval(tarr, 'fix', 20, int)
        targs.stim_t1 = get_tval(tarr, 'stimt1', 100, int)
        targs.delay_t1=get_tval(tarr,'delayl1',20, int)
        targs.stim_t2 = get_tval(tarr, 'stimt2', 100, int)
        targs.has_fix = get_tval(tarr, 'has_fix', True, bool)



    elif args.t_type == 'delay-go-interval':
        targs.has_fix= get_tval(tarr,'has_fix',True, bool)
        targs.t_len = get_tval(tarr, 'l', 300, int)
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 100, int)
        
        targs.min__mem_t = get_tval(tarr, 'min__mem_t', 0, int)
        targs.max__mem_t = get_tval(tarr, 'max__mem_t', 50, int)
        targs.displacement_direction = get_tval(tarr,'displacement_direction','anti-clockwise', str)
    
    return targs
    #we get this targs dictionary

# get particular value(s) given name and casting type
def get_tval(targs, name, default, dtype, n_vals=1):
    #note get_tval also works on tarr(works on any dictionary), we use 
    #it on tarr in definition of get_task_args,
    #where tarr = ["lt", "150", "gt", "100"]

    
    if name in targs:
        
        # set parameter(s) if set in command line
        idx = targs.index(name)
        # list.index() method returns the position(index)
        #at the first occurrence of the specified value
        
        if n_vals == 1: # one value to set
            val = dtype(targs[idx + 1])
        else: # multiple values to set
            vals = []
            for i in range(1, n_vals+1):
                vals.append(dtype(targs[idx + i]))
    else:
        # if parameter is not set in command line, set it to default
        val = default
    return val
    #e.g of this in action for tarr in definition of get_task_args
    #we go:
    #takes int args after we passed in the config file etc
    #tarr = args.task_args
    #where tarr=[ "lt","150","gt", "100"]
    #and for example in next line when we go
    #targs.t_len = get_tval(tarr, 'l', 600, int)
    #default= 600 
    # name='l' is  not in tarr so get_tval returns the 600





def save_dataset(dset, name, config=None):
    fname = os.path.join('datasets', name + '.pkl')
    #creates the file name for the pickle file where we'll save our dataset see below
    with open(fname, 'wb') as f:
        pickle.dump(dset, f) #this is what puts the dataset, that we made using create_dataset, in the file name fname
    
    gname = os.path.join('datasets', 'configs', name + '.json')
    #point of these dataset config json file is to record which arguments we used
    #to create a dataset
    #the dataset itself is saved in a pickle file at fname, see block above
    if config is not None:
        with open(gname, 'w') as f:
            json.dump(config.to_json(), f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='load', choices=['create', 'load'])
    parser.add_argument('name')
    parser.add_argument('-c', '--config', default=None, help='create from a config file')

    # general dataset arguments
    parser.add_argument('-t', '--t_type', default='rsg', help='type of trial to create')
    parser.add_argument('-n', '--n_trials', type=int, default=2000)

    # task-specific arguments
    parser.add_argument('-a', '--task_args', nargs='*', default=[], help='terms to specify parameters of trial type')
    # rsg intervals t_o t_p
    


    parser.add_argument('-i', '--intervals', nargs='*', type=int, default=None, help='select from rsg intervals')
    #this -i command: -i 46 
    #let's us specify t_o and t_p as having a particular length
    #e.g. -i 46 will both t_o and t_p as having 46
    #does this by overwriting max_t and min_t with 46 [see rsg-i_test.json]
    #also 
    #for all the trials in the dataset (rt will still be random though)

    #task_args for rsg are lt some_value, gt some value 
    #if you feed two values to -i e.g. -i 46 49 : some trials will have t_o=t_p=46 and other trials will have t_o=t_p=49


    # delay memory pro anti preset angles
    parser.add_argument('--angles', nargs='*', type=float, default=None, help='angles in degrees for dmpa tasks')
    
    

    args = parser.parse_args()
    if args.config is not None:
        #args.config is defined above "--config", it allows you to input 
        # config file(.json)from the command line that already exists
        # and if it's not none it's a json file that contains
        # {"mode":create(or load), "name": "rsg-100-150", "config": null, }
        #but if you don't input a config file, as we've done when running python3 tasks.py create rsg-123-157 -t rsg -a lt 157 gt 123
        #it will create[hence the create command] one using the arguments you specified
        # if using config file, load args from config, ignore everything else
        config_args = load_args(args.config)
        #load.args extracts the parameters from the json file(e.g. rsg-100-150.json)

        #it bunches them and stores them in config args
        #now config args is the dictionary in the json file 
        #
        
        del config_args.name
        del config_args.config
        #we delete these these arguments as we don't need them anymore 
        #and want to be economical with storage, processing etc
        #we add these args to args
        args = update_args(args, config_args)
    else:
        #if not loading from a config file:

        # add task-specific arguments. shouldn't need to do this if loading from config file
        task_args = get_task_args(args)

        args = update_args(args, task_args)

    args.argv = ' '.join(sys.argv)

    if args.mode == 'create':
        # create and save a dataset
        dset, config = create_dataset(args) 
        #see def of create dataset: we store
        #trials and args in dset, config
        save_dataset(dset, args.name, config=config)
    elif args.mode == 'load':
        # visualize a dataset
        #note in visualization 
        #the label at top of figure: is (rt, st, gt) [t_0]
        #t_p. Unless specified otherwise t_o =t_p


        dset = load_rb(args.name)
        t_type = type(dset[0])
        xr = np.arange(dset[0].t_len)

        samples = random.sample(dset, 12)
        fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(10,6))
        for i, ax in enumerate(fig.axes):
            ax.axvline(x=0, color='dimgray', alpha = 1)
            ax.axhline(y=0, color='dimgray', alpha = 1)
            ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
            ax.tick_params(axis='both', color='white')
            #ax.set_title(sample[i][2])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            trial = samples[i]
            trial_x = trial.get_x()

            
            trial_y = trial.get_y()

            if t_type in [RSG, CSG]:
                trial_x = np.sum(trial_x, axis=0)

                trial_y = trial_y[0]
                ml, sl, bl = ax.stem(xr, trial_x, use_line_collection=True, linefmt='coral', label='ready/set')
                ml.set_markerfacecolor('coral')
                ml.set_markeredgecolor('coral')
                if t_type == 'rsg-bin':
                    ml, sl, bl = ax.stem(xr, [1], use_line_collection=True, linefmt='dodgerblue', label='go')
                    ml.set_markerfacecolor('dodgerblue')
                    ml.set_markeredgecolor('dodgerblue')
                else:
                    ax.plot(xr, trial_y, color='dodgerblue', label='go', lw=2)
                    if t_type is RSG:
                        ax.set_title(f'{trial.rsg}: [{trial.t_o}, {trial.t_p}] ', fontsize=9)

            elif t_type is BinaryRSG:
                trial_x = np.sum(trial_x, axis=0)

                trial_y = trial_y[0]
                ml, sl, bl = ax.stem(xr, trial_x, use_line_collection=True, linefmt='coral', label='ready/set')
                ml.set_markerfacecolor('coral')
                ml.set_markeredgecolor('coral')
                if t_type == 'rsg-bin':
                    ml, sl, bl = ax.stem(xr, [1], use_line_collection=True, linefmt='dodgerblue', label='go')
                    ml.set_markerfacecolor('dodgerblue')
                    ml.set_markeredgecolor('dodgerblue')
                else:
                    ax.plot(xr, trial_y, color='dodgerblue', label='go', lw=2)
                    if t_type is RSG:
                        ax.set_title(f'{trial.rsg}: [{trial.t_o}, {trial.t_p}] ', fontsize=9)


            elif t_type is DelayCopy:
                for j in range(trial.dim):
                    ax.plot(xr, trial_x[j], color=cols[j], ls='--', lw=1)
                    ax.plot(xr, trial_y[j], color=cols[j], lw=1)

            elif t_type is FlipFlop:
                for j in range(trial.dim):
                    ax.plot(xr, trial_x[j], color=cols[j], lw=.5, ls='--', alpha=.9)
                    ax.plot(xr, trial_y[j], color=cols[j], lw=1)

            elif t_type in [DelayProAnti, MemoryProAnti, RTProAnti,DelayGo_IntL]:
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--', alpha=.6)
                ax.plot(xr, trial_x[1], color='salmon', lw=1, ls='--', alpha=.6)
                ax.plot(xr, trial_x[2], color='dodgerblue', lw=1, ls='--', alpha=.6)
                ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)

            elif t_type is DurationDisc:
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--')
                ax.plot(xr, trial_x[1], color='grey', lw=1, ls='--')
                ax.plot(xr, trial_x[2], color='salmon', lw=1, ls='--')
                ax.plot(xr, trial_x[3], color='dodgerblue', lw=1, ls='--')
                ax.plot(xr, trial_y[0], color='salmon', lw=1.5)
                ax.plot(xr, trial_y[1], color='dodgerblue', lw=1.5)

            elif t_type is DMProAnti:
                xr=np.arange(trial.t_len)
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--', alpha=.6)
                #stimulus 1
                if trial.t_type.endswith('pro'):
                    if trial.g1 > trial.g2:
                        ax.plot(xr, trial_x[1], color='salmon', lw=1*trial.g1, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1*trial.g1, ls='--', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[4], color='magenta', lw=1*trial.g2, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[5], color='lime', lw=1*trial.g2, ls='dotted', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                        ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)

                    elif trial.g1 < trial.g2:
                        ax.plot(xr, trial_x[1], color='salmon', lw=1*trial.g1, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1*trial.g1, ls='dotted', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[4], color='magenta', lw=1*trial.g2, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[5], color='lime', lw=1*trial.g2, ls='--', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)
                elif trial.t_type.endswith('anti'):
                    if trial.g1 > trial.g2:
                        ax.plot(xr, trial_x[1], color='salmon', lw=1*trial.g1, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1*trial.g1, ls='--', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[4], color='magenta', lw=1*trial.g2, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[5], color='lime', lw=1*trial.g2, ls='dotted', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)

                    elif trial.g1 < trial.g2:
                        ax.plot(xr, trial_x[1], color='salmon', lw=1*trial.g1, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1*trial.g1, ls='dotted', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[4], color='magenta', lw=1*trial.g2, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[5], color='lime', lw=1*trial.g2, ls='--', alpha=.6)


                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                        ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)

            elif t_type is DM1ProAnti:
                xr=np.arange(trial.t_len)
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--', alpha=.6)
                #stimulus 1
                if trial.t_type.endswith('pro'):
                    if trial.g1 > trial.g2:
                        ax.plot(xr, trial_x[1], color='salmon', lw=1*trial.g1, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1*trial.g1, ls='--', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[4], color='magenta', lw=1*trial.g2, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[5], color='lime', lw=1*trial.g2, ls='dotted', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                        ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)

                    elif trial.g1 < trial.g2:
                        ax.plot(xr, trial_x[1], color='salmon', lw=1*trial.g1, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1*trial.g1, ls='dotted', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[4], color='magenta', lw=1*trial.g2, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[5], color='lime', lw=1*trial.g2, ls='--', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)
                elif trial.t_type.endswith('anti'):
                    if trial.g1 > trial.g2:
                        ax.plot(xr, trial_x[1], color='salmon', lw=1*trial.g1, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1*trial.g1, ls='--', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[4], color='magenta', lw=1*trial.g2, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[5], color='lime', lw=1*trial.g2, ls='dotted', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)

                    elif trial.g1 < trial.g2:
                        ax.plot(xr, trial_x[1], color='salmon', lw=1*trial.g1, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1*trial.g1, ls='dotted', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[4], color='magenta', lw=1*trial.g2, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[5], color='lime', lw=1*trial.g2, ls='--', alpha=.6)


                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                        ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)


            elif t_type is DM2ProAnti:
                xr=np.arange(trial.t_len)
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--', alpha=.6)
                #stimulus 1
                if trial.t_type.endswith('pro'):
                    if trial.g1 > trial.g2:
                        ax.plot(xr, trial_x[7], color='salmon', lw=1*trial.g1, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[8], color='dodgerblue', lw=1*trial.g1, ls='--', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[10], color='magenta', lw=1*trial.g2, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[11], color='lime', lw=1*trial.g2, ls='dotted', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                        ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)

                    elif trial.g1 < trial.g2:
                        ax.plot(xr, trial_x[7], color='salmon', lw=1*trial.g1, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[8], color='dodgerblue', lw=1*trial.g1, ls='dotted', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[10], color='magenta', lw=1*trial.g2, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[11], color='lime', lw=1*trial.g2, ls='--', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)
                elif trial.t_type.endswith('anti'):
                    if trial.g1 > trial.g2:
                        ax.plot(xr, trial_x[7], color='salmon', lw=1*trial.g1, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[8], color='dodgerblue', lw=1*trial.g1, ls='--', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[10], color='magenta', lw=1*trial.g2, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[11], color='lime', lw=1*trial.g2, ls='dotted', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)

                    elif trial.g1 < trial.g2:
                        ax.plot(xr, trial_x[7], color='salmon', lw=1*trial.g1, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[8], color='dodgerblue', lw=1*trial.g1, ls='dotted', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[10], color='magenta', lw=1*trial.g2, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[11], color='lime', lw=1*trial.g2, ls='--', alpha=.6)


                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                        ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)



            elif t_type is CtxDM or t_type is CtxDelayDM:
                xr=np.arange(trial.t_len)
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--', alpha=.6)
                #stimulus 1
                if trial.t_type.endswith('1'):
                    if trial.g1_mod_1 > trial.g2_mod_1:
                        # FOR later: currently not plotting input modality 2 as in CtxDM1, task is to ignore what happens in mod 2
                        #  - as there is stuff happening in modality 2 - should we plot it as well?
                        ax.plot(xr, trial_x[1], color='salmon', lw=1*trial.g1_mod_1, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1*trial.g1_mod_1, ls='--', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[4], color='magenta', lw=1*trial.g2_mod_1, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[5], color='lime', lw=1*trial.g2_mod_1, ls='dotted', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                        ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)

                    elif trial.g1_mod_1 < trial.g2_mod_1:
                        ax.plot(xr, trial_x[1], color='salmon', lw=1*trial.g1_mod_1, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1*trial.g1_mod_1, ls='dotted', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[4], color='magenta', lw=1*trial.g2_mod_1, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[5], color='lime', lw=1*trial.g2_mod_1, ls='--', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)
                elif trial.t_type.endswith('2'):
                    if trial.g1_mod_2 > trial.g2_mod_2:
                        ax.plot(xr, trial_x[7], color='salmon', lw=1*trial.g1_mod_2, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[8], color='dodgerblue', lw=1*trial.g1_mod_2, ls='--', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[10], color='magenta', lw=1*trial.g2_mod_2, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[11], color='lime', lw=1*trial.g2_mod_2, ls='dotted', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)

                    elif trial.g1_mod_2 < trial.g2_mod_2:
                        ax.plot(xr, trial_x[7], color='salmon', lw=1*trial.g1_mod_2, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[8], color='dodgerblue', lw=1*trial.g1_mod_2, ls='dotted', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[10], color='magenta', lw=1*trial.g2_mod_2, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[11], color='lime', lw=1*trial.g2_mod_2, ls='--', alpha=.6)


                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                        ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)

            
            elif t_type is MultiSenDM:
                xr=np.arange(trial.t_len)
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--', alpha=.6)
                #stimulus 1
                
                if trial.gamma_1> trial.gamma_2:
                    ax.plot(xr, trial_x[7], color='salmon', lw=1*trial.gamma_1, ls='--', alpha=.6)
                    ax.plot(xr, trial_x[8], color='dodgerblue', lw=1*trial.gamma_1, ls='--', alpha=.6)
                    #stimulus 2
                    ax.plot(xr, trial_x[10], color='magenta', lw=1*trial.gamma_2, ls='dotted', alpha=.6)
                    ax.plot(xr, trial_x[11], color='lime', lw=1*trial.gamma_2, ls='dotted', alpha=.6)

                    ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                    ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                    ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)

                elif trial.gamma_1 < trial.gamma_2:
                    ax.plot(xr, trial_x[7], color='salmon', lw=1*trial.gamma_1, ls='dotted', alpha=.6)
                    ax.plot(xr, trial_x[8], color='dodgerblue', lw=1*trial.gamma_1, ls='dotted', alpha=.6)
                    #stimulus 2
                    ax.plot(xr, trial_x[10], color='magenta', lw=1*trial.gamma_2, ls='--', alpha=.6)
                    ax.plot(xr, trial_x[11], color='lime', lw=1*trial.gamma_2, ls='--', alpha=.6)

                    ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                    ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                    ax.plot(xr, trial_y[2], color='lime', lw=1.5)
                    



            elif t_type is DMCProAnti:
                xr=np.arange(trial.t_len)
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--', alpha=.6)
                #stimulus 1

                if trial.t_type.endswith('pro'):
                    if ((0<=trial.theta_1 <= np.pi) and (0<=trial.theta_2<=np.pi) ) or ((np.pi <trial.theta_1 <= 2*np.pi) and (np.pi<trial.theta_2<= 2*np.pi)):
                        #if match both solid
                        ax.plot(xr, trial_x[1], color='salmon', lw=1, ls='solid', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1, ls='solid', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[3], color='magenta', lw=1, ls='solid', alpha=.6)
                        ax.plot(xr, trial_x[4], color='lime', lw=1, ls='solid', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)

                    else:
                        #if they don't match
                        ax.plot(xr, trial_x[1], color='salmon', lw=1, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1, ls='--', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[3], color='magenta', lw=1, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[4], color='lime', lw=1, ls='dotted', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)
                
                elif trial.t_type.endswith('anti'):
                    #if angles aren't in same categories
                    if ((0<=trial.theta_1 <= np.pi) and (np.pi<trial.theta_2<= 2*np.pi)) or  ((np.pi <trial.theta_1 <= 2*np.pi) and (0<=trial.theta_2<=np.pi)):
                        #if they don't match
                        ax.plot(xr, trial_x[1], color='salmon', lw=1, ls='--', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1, ls='--', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[3], color='magenta', lw=1, ls='dotted', alpha=.6)
                        ax.plot(xr, trial_x[4], color='lime', lw=1, ls='dotted', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)

                    else:
                        ax.plot(xr, trial_x[1], color='salmon', lw=1, ls='solid', alpha=.6)
                        ax.plot(xr, trial_x[2], color='dodgerblue', lw=1, ls='solid', alpha=.6)
                        #stimulus 2
                        ax.plot(xr, trial_x[3], color='magenta', lw=1, ls='solid', alpha=.6)
                        ax.plot(xr, trial_x[4], color='lime', lw=1, ls='solid', alpha=.6)

                        ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                        ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                        ax.plot(xr, trial_y[2], color='lime', lw=1.5)
                
                    


            elif t_type is DelayDM or t_type is DelayDM1 or t_type is DelayDM2:
                xr=np.arange(trial.t_len)
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--', alpha=.6)
                #stimulus 1
                
                # takes us to modality 2 
                if trial.t_type.endswith('2'):
                    g = 6 
                else:
                    g=0
                
                if trial.g1 > trial.g2:
                    ax.plot(xr, trial_x[1+g], color='salmon', lw=1*trial.g1, ls='--', alpha=.6)
                    ax.plot(xr, trial_x[2+g], color='dodgerblue', lw=1*trial.g1, ls='--', alpha=.6)
                    #stimulus 2
                    ax.plot(xr, trial_x[4+g], color='magenta', lw=1*trial.g2, ls='dotted', alpha=.6)
                    ax.plot(xr, trial_x[5+g], color='lime', lw=1*trial.g2, ls='dotted', alpha=.6)

                    ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                    ax.plot(xr, trial_y[1], color='salmon', lw=1.5)
                    ax.plot(xr, trial_y[2], color='dodgerblue', lw=1.5)


                elif trial.g1 < trial.g2:
                    ax.plot(xr, trial_x[1+g], color='salmon', lw=1*trial.g1, ls='dotted', alpha=.6)
                    ax.plot(xr, trial_x[2+g], color='dodgerblue', lw=1*trial.g1, ls='dotted', alpha=.6)
                    #stimulus 2
                    ax.plot(xr, trial_x[4+g], color='magenta', lw=1*trial.g2, ls='--', alpha=.6)
                    ax.plot(xr, trial_x[5+g], color='lime', lw=1*trial.g2, ls='--', alpha=.6)

                    ax.plot(xr, trial_y[0], color='grey', lw=1.5)
                    ax.plot(xr, trial_y[1], color='magenta', lw=1.5)
                    ax.plot(xr, trial_y[2], color='lime', lw=1.5)

                


            elif t_type is DurationDisc:
                ax.plot(xr, trial_x[0], color='grey', lw=1, ls='--')
                ax.plot(xr, trial_x[1], color='grey', lw=1, ls='--')
                ax.plot(xr, trial_x[2], color='salmon', lw=1, ls='--')
                ax.plot(xr, trial_x[3], color='dodgerblue', lw=1, ls='--')
                ax.plot(xr, trial_y[0], color='salmon', lw=1.5)
                ax.plot(xr, trial_y[1], color='dodgerblue', lw=1.5)

        handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='lower center')
        plt.show()








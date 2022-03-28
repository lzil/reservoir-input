import os
import numpy as np

#import tensorflow as tf

import yaml
import logging
import time
import json
import csv
import pickle
import copy
import pdb
import re
# import pandas as pd

class LogObject(object):
    #this is our log, we define it as empty object and then populate it using log_this function
    #see log_this, this is the blank canvas for the log we see in terminal when we run the code
    pass

# turn arbitrary file into args to be used
def load_args(path=None, to_bunch=True):
    #we use it to turn json file into args in tasks.py
    if path:
        try:
            # maybe it's yaml
            config = yaml.safe_load(open(path))
        except:
            # maybe it's json
            config = json.load(open(path, 'r'))
    else:
        config = {}
    if to_bunch:
        try:
            return Bunch(config)
        except:
            print('Bunchify failed!')
    else:
        return config

# combine two args, overwriting with the second
# overwrite can take three possible values: True, False, or None (in which case overwrite Nones only)
def update_args(args, new_args, overwrite=True, to_bunch=True):
    dic = args if type(args) is dict else vars(args)
    new_dic = new_args if type(new_args) is dict else vars(new_args)
    for k in new_dic.keys():
        if overwrite is True or k not in dic or (dic[k] is None and overwrite is None) :
            dic[k] = new_dic[k]
    if to_bunch:
        return Bunch(dic)
    return dic


# produce run id and create log directory
def log_this(config, log_dir, log_name=None, checkpoints=False, use_id=True):
    #so the configs parameter takes args as argument and we can name the log using the at run time argument --name
    #we can name our log by passing in a log_name 
    run_id = str(int(time.time() * 100))[-7:]
    #we're using a method from python's time module to generate a run.id
    #run_id is to identify a particular run of the the network (i.e. a a particular run.py )
    #.time() literally returns the number of seconds passed since "epoch"
    #which isn't a training epoch but a fixed arbitrary time in history
    #for unix systems its January 1, 1970 00:00:00
    config.run_id = run_id
    #we assign this run_id to 
    print('\n=== Logging ===', flush=True)
    
    if log_name is None or len(log_name) == 0:
        #i.e. if you don't provide a name (or if you provide one of length zero by passing in nothing after --name at runt time) for the log use run_id for
        #the --name we provide when we run from terminal is this log id
        log_name = run_id
    print(f'Run id: {run_id} with name {log_name}', flush=True)

    run_dir = os.path.join(log_dir, log_name)
    #creates a path to the log folder(directory) using the paths in log_dir and log_name, in which to stores the log_file which has path log_path
    #stores the path in run_dir variable, named so bc its the directory(dir) (a dir is a folder) for this run 
    #geeks for geeks has good explanation 
    os.makedirs(run_dir, exist_ok=True) 
    #creates a directory in the specified path which is the path stored in run_dir 

    print(f'Log folder: {run_dir}', flush=True)

    log_path = os.path.join(run_dir, f'log_{run_id}.log')
    #the path to the log file i.e. where its stored, we can find them in the logs folder
    print(f'Log file: {log_path}', flush=True)

    if checkpoints:
        #what are checkpoints?
            #from control logging in run.py, if you specify --log_checkpoint_models or --log_checkpoint_samples, at run time then checkpoints will become true and we'll checkpoint things.
            #which 'things' depends on which argument we specify:
            #so our model evolves as we train it on batches. Just like in a video game, we can create save point (checkpoints) where we save our model's current state at different points in training. e.g. maybe we want to have access to whatever our model is after training on 50 batches, or 100 batches etc.
            #NB the checkpoints coincide with the log intervals which is ofc desirable we can see how our model is performing at a particular stage of training and we can actually access the model at that state
                #these checkpoints are stored model_{number of last training sample trained on}.pth
            #also we automatically(i.e. don't need to specify log_checkpoint_models) store the model path to the best model found during training 
        #we don't checkpoint by default when we run, see log_checkpoint_models in #control logging run.py and section before if __name__ = '__main__': 
        checkpoint_dir = os.path.join(run_dir, f'checkpoints_{run_id}')
        #make the path to where we save the checkpoints. the folder has name checkpoints_run_id where run_id is ofc whatever the run_id for the run we're checkpointing is 
        os.makedirs(checkpoint_dir, exist_ok=True) #actually create the folder(directory) for the checkpoints
        print(f'Logging checkpoints to: {checkpoint_dir}', flush=True)
        #
    else:
        checkpoint_dir = None

    # might want to send stdout here later too 
    
    #what's stdout?
    path_config = os.path.join(run_dir, f'config_{run_id}.json')
    #the config files store the setting on which we ran the net, so that we have a record of the 'config'urations
    with open(path_config, 'w', encoding='utf-8') as f:
        json.dump(vars(config), f, indent=4)
        print(f'Config file saved to: {path_config}', flush=True)

    log = LogObject()
    #what's a log object: (it's define at top of script)
    #we define attributes for the log object 
    log.checkpoint_dir = checkpoint_dir
    log.run_dir = run_dir
    log.run_log = log_path
    log.run_id = run_id

    print('===============\n', flush=True)
    return log
    #but we actually populate it using code in trainer.py: see if not self.args.no_log: in trainer.py


# https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/?in=user-97991
class Bunch:
    def __init__(self, *args, **kwds):
        if len(args) > 0:
            if type(args[0]) is dict:
                self.__dict__ = copy.deepcopy(args[0])
            else:
                for k,v in args[0].__dict__.items():
                    self.__dict__[k] = copy.deepcopy(v)
                # self.__dict__.update(args[0].__dict__)
        self.__dict__.update(kwds)

    def __repr__(self):
        return 'Bunch(' + str(self.__dict__) + ')'

    def to_json(self):
        return copy.deepcopy(self.__dict__)

def load_rb(path):
    with open(path, 'rb') as f:
        #opens pickle file (see: https://www.pythonforbeginners.com/files/with-statement-in-python)
        qs = pickle.load(f)
    return qs

def lrange(l, p=0.1):
    return np.linspace(0, (l-1) * p, l)


# get config dictionary from the model path
def get_config(path, ctype='model', to_bunch=False):
    head, tail = os.path.split(path)
    if ctype == 'dset':
        fname = '.'.join(tail.split('.')[:-1]) + '.json'
        c_folder = os.path.join(head, 'configs')
        if os.path.isfile(os.path.join(c_folder, fname)):
            c_path = os.path.join(head, 'configs', fname)
        else:
            raise NotImplementedError

    elif ctype == 'model':
        if tail == 'model_best.pth' or 'test' in tail:
            for i in os.listdir(head):
                if i.startswith('config'):
                    c_path = os.path.join(head, i)
                    break
        else:
            folders = head.split('/')
            if folders[-1].startswith('checkpoints_'):
                run_id = folders[-1].split('_')[-1]
                c_path = os.path.join(*folders[:-1], 'config_'+run_id+'.json')
            else:
                run_id = re.split('_|\.', tail)[1]
                c_path = os.path.join(head, 'config_'+run_id+'.json')
        if not os.path.isfile(c_path):
            raise NotImplementedError
    else:
        raise NotImplementedError
    with open(c_path, 'r') as f:
        config = json.load(f)
    if to_bunch:
        return Bunch(**config)
    else:
        return config


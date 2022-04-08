import numpy as np
import torch
import matplotlib.pyplot as plt

import argparse
import pdb
import sys

sys.path.append('../')

# from testers import load_model, test_model
from network import M2Net
from utils import Bunch, load_rb, get_config


parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+')
# parser.add_argument('dset')
args = parser.parse_args()

nets = []
for path in args.models:
    config = get_config(path)
    # with open(path, 'rb') as f:
    net = M2Net(config)
    net.load_state_dict(torch.load(path))
    nets.append(net)
    
pdb.set_trace()


    
# J = m_dict['W_f.weight']
# v = J.std()
# shp = J.shape
# m_dict['W_f.weight'] += torch.normal(0, v * .01, shp)

# J = m_dict['W_ro.weight']
# v = J.std()
# shp = J.shape
# m_dict['W_ro.weight'] += torch.normal(0, v * .01, shp)


# dset = load_rb(args.dset)

# test_data = test_model(m_dict, dset, n_tests=200)
# i, xs, ys, zs, losses = list(zip(*test_data))

# print(np.mean(losses))





# bunch = Bunch()
# bunch.N = 250
# bunch.D = 250
# bunch.O = 1

# bunch.res_init_type = 'gaussian'
# bunch.res_init_params = {'std': 1.5}
# bunch.reservoir_seed = 0
# net = Network(bunch)

# nums = ['146', '152', '158', '164', '170', '176', '182', '188']
# #dsets = ['15', '25', '35', '45', '55', '65', '75', '85']
# dsets = ['logs/2625461/'+i+'/model.pth' for i in nums]
# models = []
# for i in dsets:
#     with open(i, 'rb') as f:
#         models.append(torch.load(f))

# pdb.set_trace()
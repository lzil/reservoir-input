from itertools import product
import os
import json
import argparse
import random

def create_parameters(debug):

    mapping = {}
    ix = 1

    D1s = [10, 50, 100, 200]
    Ns = [200, 500]

    lr = 1e-3
    n_epochs = 20
    patience = 5000
    batch_size = 3

    # usually have this off but if we wanna check models, set it on
    log_checkpoint_models = False

    n_seeds = 2
    n_rseeds = 3

    m_noises = [0, 2, 4, 10]
    # m_noises = [0]
    r_noises = [0.01]
    # train_parts = [['M_u', 'M_ro']]
    train_parts = [['all'], ['M_u', 'M_ro']]

    datasets = [
        ['datasets/rsg-100-150.pkl', 'datasets/rsg-150-200.pkl'],
        # ['datasets/delaypro.pkl', 'datasets/delayanti.pkl', 'datasets/memorypro.pkl', 'datasets/memoryanti.pkl']
    ]
    losses = [
        # ['mse-e'],
        ['mse']
    ]

    if debug:
        datasets = [['datasets/delaypro.pkl']]
        Ns = [80]
        D1s = [20]
        n_seeds = 1
        n_rseeds = 1
        m_noises = [0]
        r_noises = [0]
        n_epochs = 2
        patience = 1000
        batch_size = 1
        train_parts = [['M_u', 'M_ro']]

    seed_offset = 20
    rseed_offset = 20
    seed_samples = [i + seed_offset for i in range(n_seeds)]
    rseed_samples = [i + rseed_offset for i in range(n_rseeds)]

    for (d, nN, nD1, loss, rnoise, mnoise, tp, nseed, rseed) in product(datasets, Ns, D1s, losses, r_noises, m_noises, train_parts, range(n_seeds), range(n_rseeds)):
        if nD1 > nN:
            continue
        run_params = {}
        run_params['dataset'] = d
        run_params['loss'] = loss

        run_params['D1'] = nD1
        run_params['N'] = nN

        # these parameters only useful when training with adam
        run_params['optimizer'] = 'adam'
        run_params['lr'] = lr
        run_params['n_epochs'] = n_epochs
        run_params['patience'] = patience
        run_params['batch_size'] = batch_size

        run_params['train_parts'] = tp

        run_params['res_noise'] = rnoise
        run_params['m_noise'] = mnoise

        run_params['seed'] = 0
        run_params['network_seed'] = seed_samples[nseed]
        run_params['res_seed'] = rseed_samples[rseed]
        run_params['res_x_seed'] = 0

        run_params['log_checkpoint_models'] = log_checkpoint_models

        mapping[ix] = run_params
        ix += 1

    n_commands = ix - 1

    if debug:
        name = 'debug'
    else:
        name = 'params'
    fname = os.path.join('slurm_params', name + '.json')
    with open(fname, 'w') as f:
        json.dump(mapping, f, indent=2)
    if debug:
        print(f'Produced {n_commands} run commands in {fname}. Use with:\nsbatch --array=1-{n_commands} slurm_debug.sbatch')
    else:
        print(f'Produced {n_commands} run commands in {fname}. Use with:\nsbatch --array=1-{n_commands} slurm_train.sbatch')

    return mapping


def apply_parameters(filename, args):
    dic = vars(args)
    with open(filename, 'r') as f:
        mapping = json.load(f)
    for k,v in mapping[str(args.slurm_id)].items():
        dic[k] = v
    return args


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--debug', action='store_true')
    args = p.parse_args()

    create_parameters(args.debug)

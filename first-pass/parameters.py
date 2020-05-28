from itertools import product
import os
import json
import argparse

def create_parameters(name):

    mapping = {}
    ix = 1

    Ds = [10, 50, 100, 250]
    Ns = [100, 250]

    trial_lens = [100]
    singles = [10]
    lrs = [1e-2, 1e-3]
    n_epochs = 100

    patience = 4000

    n_seeds = 2
    n_rseeds = 2

    #n_commands = len(Ds) * len(Ns) * len(trial_lens) * len(singles) * len(lrs) * n_seeds

    for (nD, nN, tl, s, lr, seed, rseed) in product(Ds, Ns, trial_lens, singles, lrs, range(n_seeds), range(n_rseeds)):
        if nD > nN:
            continue
        run_params = {}
        run_params['dataset'] = f'data/rsg_tl100_l20_sc1.pkl'
        run_params['D'] = nD
        run_params['N'] = nN
        run_params['lr'] = lr
        run_params['n_epochs'] = n_epochs
        run_params['patience'] = patience

        run_params['reservoir_seed'] = rseed

        mapping[ix] = run_params
        ix += 1

    n_commands = ix - 1

    fname = os.path.join('slurm_params', name + '.json')
    with open(fname, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f'Produced {n_commands} run commands in {fname}. Use with `sbatch --array=1-{n_commands} slurm_train.sbatch`.')

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
    p.add_argument('-n', '--name', type=str, default='params')
    args = p.parse_args()

    create_parameters(args.name)
# reservoir-input
This is a project about how you can optimize the inputs of a recurrent reservoir to perform a diverse array of tasks. We're now implementing many tasks.


## quick file guide
- `run.py`: to train/run the network. contains all the options, using argparse
- `network.py`: defines the network with pytorch
- `tasks.py`: defines the tasks:
    - RSG: ready-set-go task from Sohn et al
    - CSG: cue-set-go task from Wang et al
    - DelayProAnti, MemoryProAnti: tasks from Duncker et al
    - FlipFlop: flip flop task from Sussillo and Barak


## to run something
Start with `python run.py --no_log` to see if it works by default. Most likely, you'll need to make sure the default dataset exists; you can do that by seeing the section below.
You can also choose to run with any dataset, with `python run.py --no_log -d datasets/custom_dataset.pkl`.

Without the `--no_log` option, logs will be generated in a custom manner in the `logs/` folder; details are in `utils.py`, but they're not particularly important.

Once the code above works, try `python run.py -d datasets/rsg-100-150.pkl --name test_run --n_epochs 2 --batch_size 5 -N 100`.

### to train using node perturbation
Start with `python run.py -d datasets/rsg-100-150.pkl --node_pert --D2 0`
By default, this will train the feedforward weight matrices M_u, and M_ro while leaving the rest of the weight matrices fixed throughout training.
This is because the argument for node_pert_parts - which the weight matrices to which node perturbation will be applied - has a default argument of ['M_u', 'M_ro'].
`node_pert_parts` can be any nonempty subset of `['M_u','W_u','J','M_ro']`. Weight matrices not in the subset will be left fixed throughout training.

Each trainable weight has noise variance and a learning rate hyperparameters. Together these determine the step size of the NP update.
For example, the default values of `--node_pert_lr_M_u` and `--node_pert_var_noise_u` are 1e-3 and 1e-1, respectively.

Notes: 
- For now, this implementation works only for `--D2 0` and when NP is the only learning rule in use.




## to create tasks
### dataset creation
`python tasks.py create rsg_1 -t rsg -n 200` creates an RSG dataset at `datasets/rsg_1.pkl` with 200 trials, using the default parameters.

`python tasks.py create dpro_1 -t delaypro -n 100 --angles 30 60 100` creates a DelayPro dataset at `datasets/dpro_1.pkl` with 100 trials, using angles of 30, 60, or 100.

`python tasks.py create rsg_2 -t rsg -a gt 100 lt 150`
creates an RSG dataset at `datasets/rsg_2.pkl` with 2000 trials, with intervals between 100 and 150.

`python tasks.py create danti_1 -c datasets/config/delayanti.json` creates a DelayAnti dataset at `datasets/danti_1.pkl` with 2000 trials, with parameters taken from the config file.

### dataset visualization
`python tasks.py load datasets/rsg_1.pkl` loads some examples from the dataset


## contact
Email <liang.zhou.18@ucl.ac.uk> with any questions.
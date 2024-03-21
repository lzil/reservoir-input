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


### notes for reinforcement learning implementations

- Our code conforms with the  'immediate-evaluative' convention where $r_t$ denotes the reward received for taking action $a_{t}$ in state $s_t$ as in our implementation For the tasks we consider, the immediate consequence of an action is assumed to be determined instantly and does not depend on the state to which the agent transitions after the action. This convention is fairly standard ([Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf), [Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf)). 
- In a stimulus-response task of length T time-steps, we assume the agent has an action $a_T$ to select at the final time-step T in state $s_T$ and so the terminal state is $s_{T+1}$ . This has several consequences:
    - By definition the value of the terminal state is 0 and thus so is it's TD residual for an episodic task.
    - In turn when calculating the generalized advantage estimate of action $a_T$ we have
      $$\hat{A}_{T}={\sum^{\infty}_{l=0}{(\gamma\lambda})^l}\delta^V_{T+l}=\delta^V_{T}=r_{T}+\gamma \hat{V}(s_{T+1})-\hat{V}(s_{T})=r_{T} -\hat{V}(s_{T})$$
    - As there is an action to be taken at time T, $V(s_T)=\mathbb{E}[R_{T}|s_{T}]$; so that the target for a value function approximator's output for state $s_T$ is simply $r_T$.

### to train using node perturbation
Start with `python run.py -d datasets/rsg-100-150.pkl --node_pert --D2 0 --node_pert_manual`
By default, this will train the feedforward weight matrices M_u, and M_ro while leaving the rest of the weight matrices fixed throughout training.
This is because the argument for node_pert_parts - which the weight matrices to which node perturbation will be applied - has a default argument of ['M_u', 'M_ro'].
`node_pert_parts` can be any nonempty subset of `['M_u','W_u','J','M_ro']`. Weight matrices not in the subset will be left fixed throughout training.
Using `--node_pert_manual` means that the standard node perturbation update will be implemented instead of the update being passed to an optimizer (e.g. Adam). 
--node_pert_mode_online 


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


## task library 
### Cognitive task battery from Yang et al
The following task arguments (like '`rsg`' above) can be used to create datasets of continuous-direction versions of the tasks described by Yang:
- Go task family:
    - `delay-pro`: Go 
    - `memory-pro`: DlyGo 
    - `rt-pro`: RT Go (reaction time task)
- Anti task family:
    - `delay-anti`: same as the 'delay-pro' except the expected response is a saccade in the opposite direction of that of the stimulus.
    - `memory-anti`: DlyAnti
    - `rt-anti`: RT Go (reaction time task)
- DM family:
    - `dm1-pro`: DM1 (simplified RDM task with two directions,and their respective coherences,shown through modality 1; objective: respond in direction with greater coherence )
    - `dm2-pro`: DM2 (same but through modality 2)
    - `ctx-dm1`: Ctx DM 1 (two modalities, each receives a distinct RDM task -  direction pairs are the same in each RDM task, but coherences differ ; objective: ignore modality 2, respond in the direction with strongest coherence in stimulus 1 )
    - `ctx-dm2`: Ctx DM 2 
    - `multisen-dm`: MultSen DM 
- Dly DM family;
    - `delay-dm1`: Dly DM 1 (same as DM1 but the stimuli are shown separately with a delay between them and a delay after the second stimulus is shown before the response period)
    - `delay-dm2`: Dly DM 2 
    - `ctx-delay-dm1`: Ctx Dly DM 1 
    - `ctx-delay-dm2`: Ctx Dly DM 2
- Matching family:
    - `dmc-pro` : DMC (delay-match-to-category; two directional stimuli shown one at a time with a delay period betwen them - respond in same direction as second stimulus if stimuli point towards the same hemicircle; fixate otherwise)
    - `dmc-anti`: DNMC (respond in direction same direction of second stimulus, if stimuli don't fall in same category; fixate otherwise)
    (DMS has been omitted as an exact match for continuous random directions is highly improbable - in this setting a network could achieve near perfect performance by always choosing to respond in the direcion of the second stimulus).



### `Mod cog` task battery from Khona et al 
- Integration tasks
    - `delay-go-interval`: extension of `DlyGo`(`memory-pro` in our code) where the expected output is displaced by an amount proportional to the length of the delay period; 
        - by default, `displacement-direction` is anti-clockwise; the pass argument `clockwise` for a clockwise displacement
        - the constant of proportionality can be accessed via the argument `delay-scalar`. 
        - the delay period length is variable and randomly sampled from U(`min_mem_t`, `max_mem_t`) in each trial. 
    -
    



## contact
Email <liang.zhou.18@ucl.ac.uk> with any questions.

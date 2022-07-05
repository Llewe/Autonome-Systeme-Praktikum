# Autonome-Systeme-Praktikum

## Setup
1. Create env `conda create -n l-ki python=3.9`
2. Switch to env `conda activate l-ki` (depending on os)
3. Install packages `pip install -r requirements.txt`

### Setup Unity Env (Environment Executable)
> https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Executable.md

## Application Parameter
### Required:
- `-env <type>` set the enviroment (`gym` or `unity`)
- `-tag` give the run an individual tag/name 

### Optional
- `-d` use the demo ppo
- `-eval` run evaluation of trained model
- `-cpu` forces to use the cpu 
- `-env_n` name the domain name 
- `-agent` (ppo, random agent) set the agent type 

### Hyperparamters (if not given, default values are selected)
- `-e <number>` number of episodes in this run 
- `-us <number>` number of steps until update (n_steps/update_timestep)
- `-g <number>` discount factor
- `-lr_a <number>` learn rate of the actor
- `-lr_c <number>` learn rate of the critic
- `-ke <number>` "--k_epochs"
- `-e_clip <number>` eps_clip
- `-a_std <number>` action_std
- `-a_std_rate` action standard deviation decay rate 
- `-a_std_freq` action standard deviation decay frequency
- `-a_std_min` minimum action standard deviation

### Environment Parameter
- `--env_rngMass` default: `False` enable random mass
- `--env_mass <float>` default: `1.` set mass if random is disabled
- `--env_minMass <float>` default: `0.1` min mass if random is enabled
- `--env_maxMass <float>` default: `20.` max mass if random is enabled
- `--env_rngGravity` default: `False` enable random gravity
- `--env_gravity <float>` default: `9.81` set gravity if random is disabled
- `--env_minGravity <float>` default: `4.` min gravity if random is enabled
- `--env_maxGravity <float>` default: `105.` max gravity if random is enabled
- `--env_rngScale` default: `False` enable random ball scale
- `--env_individualScale` default: `False` enable individual ball scale for x y z axis
- `--env_scale <float>` default: `1.` set scale if random and individual scale is disabled
- `--env_scale_x <float>` default: `1.` set scale for x axis if random scale is disabled and individual scale is enabled
- `--env_scale_y <float>` default: `1.` set scale for y axis if random scale is disabled and individual scale is enabled
- `--env_scale_z <float>` default: `1.` set scale for z axis if random scale is disabled and individual scale is enabled
- `--env_minScale <float>` default: `0.2` min scale if random is enabled (works also combined with individual scale)
- `--env_maxScale <float>` default: `5.` max scale if random is enabled (works also combined with individual scale)
- `--env_video` default: `False` enable graphical output
- `--env_timeScale <float>` default: `20.` time speedup


## Check Carbon Footprint
Load webpage `carbonboard --filepath="./out/emissions.csv"`

## Check Model Logs - TensorBoard
```
tensorboard --logdir ./logs --port 9238
```

## Slurm
```
sbatch --partition=All --cpus-per-task=4 slurm_runner.sh
```

### Fix return code 127
 Error Message
 > mlagents_envs.exception.UnityEnvironmentException: Environment shut down with return code 127
 

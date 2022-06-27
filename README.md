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
### Optional
- `-d` use the demo ppo
- `-r` enable replay mode
- `-m <name>` storage name of the model
- `-e <number>` training episode number 
- `-us <number>` number of steps until update (n_steps/update_timestep)
- `-g <number>` discount factor
- `-lr_a <number>` learn rate of the actor
- `-lr_c <number>` learn rate of the critic
- `-ke <number>` "--k_epochs"
- `-e_clip <number>` eps_clip
- `-a_std <number>` action_std

## Check Carbon Footprint
Load webpage `carbonboard --filepath="./out/emissions.csv"`

## Check Model Logs - TensorBoard
```
tensorboard --logdir ./logs --port 9238
```

## Slurm
sbatch --partition=All --cpus-per-task=4 slurm_runner.sh

 ### Fix return code 127
 Error Message
 > mlagents_envs.exception.UnityEnvironmentException: Environment shut down with return code 127
 
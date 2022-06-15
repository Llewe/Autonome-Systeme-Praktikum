# Autonome-Systeme-Praktikum

## Setup
1. Create env `conda create -n l-ki python=3.9`
2. Switch to env `conda activate l-ki` (depending on os)
3. Install packages `pip install -r requirements.txt`

### Setup Unity Env (Environment Executable)
> https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Executable.md

## Application Parameter
- `-m <name>` storage name of the model
- `-d` use the demo ppo
- `-r` enable replay mode

## Check Carbon Footprint
Load webpage `carbonboard --filepath="./out/emissions.csv"`

## Check Model Logs - TensorBoard
```
tensorboard --logdir ./logs --port 9238
```
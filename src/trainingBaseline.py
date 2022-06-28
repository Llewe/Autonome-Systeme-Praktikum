from pyexpat import model
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import platform

def trainBaselinePPO(args, env):
    osName = platform.node()
    currentTimeInSec = int(round(datetime.now().timestamp()))
    logDir = f"runs/logs/{args.tag}/{osName}-{currentTimeInSec}"
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    
    
    agent = PPO(
        policy=MlpPolicy,
        env=env,
        learning_rate=args.lr_actor,
        gamma=args.gamma,
        clip_range= args.epsilon_clip,
        verbose=1,
        tensorboard_log=logDir
        )
    agent.learn(total_timesteps=args.episodes,tb_log_name=args.tag)
    
    env.close()
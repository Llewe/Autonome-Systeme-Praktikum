from pyexpat import model
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from torch.utils.tensorboard import SummaryWriter
import os

def getLogDir():
    logDir = f"out/logs/stable_baseline"
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    return logDir


def trainBaselinePPO(args, env):
    agent = PPO(
        policy=MlpPolicy,
        env=env,
        learning_rate=args.lr_actor,
        gamma=args.gamma,
        clip_range= args.epsilon_clip,
        verbose=1,
        tensorboard_log=getLogDir()
        )
    agent.learn(total_timesteps=args.episodes,tb_log_name=args.tag)
    
    env.close()
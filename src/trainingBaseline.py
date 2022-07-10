import glob
import imp
from pyexpat import model
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from torch.utils.tensorboard import SummaryWriter
from gym import Wrapper
import os
from datetime import datetime
import numpy as np
import platform
import torch
from src.training import CONST_LOG_ACTION_FREQUENCY


class ActionLogger(Wrapper):
    def __init__(self, env,logger,total_steps):
        super().__init__(env)
        self.logger = logger
        self.time_step = 0
        self.plot_histogram_step=0
        self.total_steps = total_steps
        self.actions = []
        
        
    def step(self, action):
        self.time_step +=1
        self.actions.append(action)
        
        # plot action distribution
        if self.time_step % (0.2 * self.total_steps) == 1: # number of sessions
            action_freq = np.array(self.actions)
            self.actions.clear()
            self.logger.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(action_freq), global_step = self.plot_histogram_step)    
            self.plot_histogram_step += 1
        
        return super().step(action)
    def close(self):
        action_freq = np.array(self.actions)
        self.logger.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(action_freq), global_step = self.plot_histogram_step)
        return super().close()

def trainBaselinePPO(args, env, output_dir, folderPath):
    osName = platform.node()
    currentTimeInSec = int(round(datetime.now().timestamp()))
    logDir = f"runs/logs/{args.tag}/{osName}-{currentTimeInSec}"
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    actionLogDir = f"runs/logs/{args.tag}-actionlogger/{osName}-{currentTimeInSec}"
    if not os.path.exists(actionLogDir):
        os.makedirs(actionLogDir)
        
    logWriter = SummaryWriter(log_dir=actionLogDir)
    
    agent = PPO(
        policy=MlpPolicy,
        env=ActionLogger(env,logWriter,args.episodes),
        learning_rate=args.lr_actor,
        gamma=args.gamma,
        clip_range= args.epsilon_clip,
        verbose=1,
        tensorboard_log=logDir
        )
    agent.learn(total_timesteps=args.episodes,tb_log_name=args.tag)
    
    
    # save model
    modelPath = output_dir + "/models" + folderPath 
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    agent.save(os.path.join(modelPath, "checkpoint-1"))
    
    env.close()
    logWriter.close()
    
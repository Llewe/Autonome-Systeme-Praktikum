from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

import os

def trainingDemo(env,name,episodes):
    modelDir = f"models/{name}"
    logDir = f"logs/{name}"
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    if not os.path.exists(logDir):
        os.makedirs(logDir)
        
    model = PPO(
        policy=MlpPolicy,
        env=env,
        batch_size=128,
        learning_rate=3e-04,
        gae_lambda=0.99,
        gamma=0.9999,
        clip_range=0.1,
        vf_coef=0.19,
        verbose=1,
        tensorboard_log=logDir
        )
    model.learn(total_timesteps=episodes,reset_num_timesteps=False, tb_log_name="PPO")
    print(f"Saving model '{modelDir}/model'")
    model.save(f"{modelDir}/model")
    print("Saved model")
    env.close()
    
def replayDemo(env,name,repeatMode=False):
    modelDir = f"models/{name}"
    if not os.path.exists(modelDir):
        print(f"Model '{modelDir}' doesn't exist.")
        return
    
    obs = env.reset()
    model = PPO.load(f"{modelDir}/model", env=env)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            if repeatMode:
                print("Resetting env")
                env.reset()
            else:
                env.close()
                return
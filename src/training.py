import json
import os
import numpy as np
import gym
from src.PPO import PPO
import ray
from ray import tune, register_env
from src.envBuilder import buildFromArgs, createGymEnv, createUnityEnv


@ray.remote
class PPOTraining:
    
    def __init__(self, env, checkpoint_path, agent):
        print(env)
        self.env = createUnityEnv()
        register_env("unity_env", createUnityEnv())
        self.checkpoint_path = checkpoint_path
        self.agent = agent
        # start environment
        # no graphics: faster, no visual rendering 
        #env = createUnityEnv(no_graphics=True)
        #env = gym.make('CartPole-v1')

        #state_dim = self.env.observation_space[0].shape[0]
        
        #action_dim = self.env.action_space.shape[0]
    
        ''' 
        # create directory and file to save checkpoint to
        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.checkpoint_path = os.path.join(directory, 'net_{}_{}'.format('logs', 0))
        '''
      
        # create PPO driven agent with hyperparameters
        '''agent = PPO(state_dim, 
            action_dim, 
            args.lr_actor, 
            args.lr_critic, 
            args.gamma, 
            args.k_epochs, 
            args.epsilon_clip, 
            args.action_std
            )


        self.searchSpace = {
            #"env": self.env,
            "agent": self.agent,
            "checkpoint_path": self.checkpoint_path,
            "render": True,
            "nr_episodes": 400,
            "update_timestep": 64,
            "K_epochs": 10,
            "eps_clip": 0.2,
            "gamma": 0.99,
            "lr_actor": 0.0003,
            "lr_critic": 0.001,
            "action_std_decay_rate": 0.05,
            "min_action_std": 0.1,
            "action_std_decay_freq": int(2.5e5),
            "save_model_freq": int(1e1)
        }
        
    def getSearchSpace(self):
        return self.searchSpace
    '''
    def episode(self, nr_episode=0, render=False, update_timestep=4000, action_std_decay_rate = 0.01, min_action_std = 0.001, action_std_decay_freq = int(2.5e5), save_model_freq = int(1e1)):
       
        state = self.env.reset()
        total_return = 0
        done = False
        time_step = 0
        print("episode:" + nr_episode)
        while not done:
            if render:
                self.env.render()
                    
            # 1. Select action according to policy
            action = self.agent.select_action(state)
        
            # 2. Execute selected action
            next_state, reward, done, _ = self.env.step(action)
            
            # 3. Update buffer
            self.agent.buffer.rewards.append(reward)         
            self.agent.buffer.is_terminals.append(done)

            # 4. Integrate new experience into self.agent
            if time_step % update_timestep == 1:      
                self.agent.update()
                
                
            if time_step % action_std_decay_freq == 1:
                self.agent.decay_action_std(action_std_decay_rate, min_action_std)
            
            state = next_state
            total_return += reward
            time_step += 1

            if time_step % save_model_freq == 0:
                self.agent.save(self.checkpoint_path)
            
        print(nr_episode, ":", total_return)
        self.env.close()
    
        return total_return


    """
    at the moment without multiple instances at once
    """
    
    def training(self, config):
        print("env:")
        print(self.env)
        for nr_episode in range(1):
            print(config)
            self.episode(nr_episode, render=config["render"], update_timestep=config["update_timestep"], action_std_decay_rate=config["action_std_decay_rate"], min_action_std=config["min_action_std"], action_std_decay_freq=config["action_std_decay_freq"], save_model_freq=config["save_model_freq"])
            
            
  
    def startTraining(self, config, checkpoint_dir=None):
                
        
        # train agent with hyperparameter tuning

        analysis = tune.run(self.training, config=config, num_samples= 1, resources_per_trial={"cpu": 2, "gpu": 1})
        
        # print ray tune analysis
     
        print(analysis.get_best_config(metric="score", mode="min"))
        
        #close environment
        #self.env.close()
        
        return analysis
        

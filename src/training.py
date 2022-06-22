import os
import numpy as np
import gym
from src.PPO import PPO
import ray
from ray import tune
from src.envBuilder import buildFromArgs, createGymEnv, createUnityEnv


@ray.remote
class PPOTraining:
    
    def __init__(self, args):
        
  
        self.env = buildFromArgs(args)

        self.args = args
        # start environment
        # no graphics: faster, no visual rendering 
        #env = createUnityEnv(no_graphics=True)
        #env = gym.make('CartPole-v1')

        state_dim = self.env.observation_space[0].shape[0]
        
        action_dim = self.env.action_space.shape[0]
    

        # create directory and file to save checkpoint to
        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.checkpoint_path = os.path.join(directory, 'net_{}_{}'.format('logs', 0))
        
      
        # create PPO driven agent with hyperparameters
        self.agent = PPO(state_dim, 
            action_dim, 
            args.lr_actor, 
            args.lr_critic, 
            args.gamma, 
            args.k_epochs, 
            args.epsilon_clip, 
            args.action_std
            )


        self.searchSpace = {
            "env": self.env,
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
        
    
    def episode(env, checkpoint_path, agent,nr_episode=0, render=False, update_timestep=4000, action_std_decay_rate = 0.01, min_action_std = 0.001, action_std_decay_freq = int(2.5e5), save_model_freq = int(1e1)):
        state = env.reset()
        total_return = 0
        done = False
        time_step = 0
        print("episode:" + nr_episode)
        while not done:
            if render:
                env.render()
                    
            # 1. Select action according to policy
            action = agent.select_action(state)
        
            # 2. Execute selected action
            next_state, reward, done, _ = env.step(action)
            
            # 3. Update buffer
            agent.buffer.rewards.append(reward)         
            agent.buffer.is_terminals.append(done)

            # 4. Integrate new experience into agent
            if time_step % update_timestep == 1:      
                agent.update()
                
                
            if time_step % action_std_decay_freq == 1:
                agent.decay_action_std(action_std_decay_rate, min_action_std)
            
            state = next_state
            total_return += reward
            time_step += 1

            if time_step % save_model_freq == 0:
                agent.save(checkpoint_path)
            
        print(nr_episode, ":", total_return)
    
        return total_return


    """
    at the moment without multiple instances at once
    """

    def training(self, config):
        
        for nr_episode in range(config["nr_episodes"]):
            self.episode(nr_episode, config["env"], config["agent"], config["checkpoint_path"], config["render"], config["update_timestep"], config["action_std_decay_rate"], config["min_action_std"], config["action_std_decay_freq"], config["save_model_freq"])
  
  
    def startTraining(self, i):
            
        print(i)
        # train agent with hyperparameter tuning
        analysis = tune.run(self.training, config=self.searchSpace)

        
        # print ray tune analysis
        print(analysis.get_best_config(metric="score", mode="min"))

        #close environment
        #self.env.close()
        
        return "test"
        

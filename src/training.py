import os
import numpy as np
import gym
from src.PPO import PPO
import matplotlib.pyplot as plt
from src.envBuilder import createGymEnv, createUnityEnv

def episode(env, checkpoint_path, agent,nr_episode=0, update_timestep=4000, render=False, action_std_decay_rate = 0.01, min_action_std = 0.001, action_std_decay_freq = int(2.5e5), save_model_freq = int(1e1)):
    state = env.reset()
    total_return = 0
    done = False
    time_step = 0
   

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
def training(env, checkpoint_path, agent,nr_episodes, update_timestep, action_std_decay_rate, min_action_std, action_std_decay_freq, save_model_freq):
    list_total_return = []
    for nr_episode in range(nr_episodes):
        total_return = episode(env, checkpoint_path, agent,nr_episode,True, update_timestep, action_std_decay_rate, min_action_std, action_std_decay_freq, save_model_freq)
        list_total_return.append(total_return)
    plt.plot(list_total_return)
    plt.show()
        
    
def startTraining():            

    params = {}
    params["has_continuous_action_space"] = True
    params["update_timestep"] = 64    
    params["K_epochs"] = 30     # should probably be between [3, 30]          
    params["eps_clip"] = 0.2    # should probably be between [0.1, 0.3]          
    params["gamma"] = 0.99      # probably 0.99 at its best          
    params["lr_actor"] = 0.0003      
    params["lr_critic"] = 0.001 
    params["action_std"] = 0.6     
    params["nr_episodes"] = 500
    params["action_std_decay_rate"] = 0.05          # action standard deviation decay rate
    params["min_action_std"] = 0.1                 # minimum action standard deviation
    params["action_std_decay_freq"] = int(2.5e5)    # action standard deviation decay frequency
    params["save_model_freq"] = int(1e1)            # save model to checkpoint frequency 

    # start environment
    # no graphics: faster, no visual rendering 
    env = createUnityEnv(no_graphics=True)
    #env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space[0].shape[0]

    if params["has_continuous_action_space"]:
        action_dim = env.action_space.shape[0]
    
    else:
        action_dim = env.action_space.n

    # create directory and file to save checkpoint to
    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = os.path.join(directory, 'net_{}_{}'.format('logs', 0))

    
    # create PPO driven agent with hyperparameters
    agent = PPO(state_dim, 
                action_dim, 
                params["lr_actor"], 
                params["lr_critic"], 
                params["gamma"], 
                params["K_epochs"], 
                params["eps_clip"], 
                params["has_continuous_action_space"], 
                params["action_std"])
    
    # train agent
    training(env=env, checkpoint_path=checkpoint_path, agent=agent, nr_episodes=params["nr_episodes"], update_timestep = params["update_timestep"], action_std_decay_rate=params["action_std_decay_rate"], min_action_std=params["min_action_std"], action_std_decay_freq=params["action_std_decay_freq"], save_model_freq=params["save_model_freq"])

    #close environment
    env.close()
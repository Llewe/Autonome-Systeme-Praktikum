import os
import numpy as np
import gym
from src.PPO import PPO
from src.envBuilder import createGymEnv, createUnityEnv

def episode(env, checkpoint_path, agent,nr_episode=0, render=False):
    state = env.reset()
    total_return = 0
    discounted_return = 0
    done = False
    time_step = 0
    action_std_decay_rate = 0.01
    min_action_std = 0.001
    action_std_decay_freq = int(2.5e5)
    save_model_freq = int(1e1)
   

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
        if time_step % 10 == 1:      
            agent.update()
              
              
        if time_step % action_std_decay_freq == 1:
            agent.decay_action_std(action_std_decay_rate, min_action_std)
        
        state = next_state
        discounted_return += (0.99**time_step) * reward
        total_return += reward
        time_step += 1

        if time_step % save_model_freq == 0:
            agent.save(checkpoint_path)
        
    print(nr_episode, ":", total_return)
   
    return total_return
 

"""
at the moment without multiple instances at once
"""
def training(env, checkpoint_path, agent,episodes):
    for nr_episode in range(episodes):
        episode(env, checkpoint_path, agent,nr_episode,True)
    
    
def startTraining():            

    params = {}
    params["has_continuous_action_space"] = True
    params["update_timestep"] = 1000    
    params["K_epochs"] = 30     # [3, 30]          
    params["eps_clip"] = 0.1    # [0.1, 0.3]          
    params["gamma"] = 0.99      # 0.99            
    params["lr_actor"] = 0.0000003      
    params["lr_critic"] = 0.000001 
    params["action_std"] = 0.6     
    params["episodes"] = 10000
    
   # print("training environment name : " + env_name)

    env = createUnityEnv(no_graphics=True)
    #env = gym.make('CartPole-v1')
    state_dim = env.observation_space[0].shape[0]

    if params["has_continuous_action_space"]:
        action_dim = env.action_space.shape[0]
    
    else:
        action_dim = env.action_space.n

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + 'testenv' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    
    checkpoint_path = os.path.join(directory, 'net_{}_{}'.format('test', 0))

    

    agent = PPO(state_dim, 
                action_dim, 
                params["lr_actor"], 
                params["lr_critic"], 
                params["gamma"], 
                params["K_epochs"], 
                params["eps_clip"], 
                params["has_continuous_action_space"], 
                params["action_std"])

    training(env=env, checkpoint_path=checkpoint_path, agent=agent, episodes=params["episodes"])
    env.close()
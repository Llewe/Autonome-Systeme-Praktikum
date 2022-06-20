import numpy as np
import gym
from src.PPO import PPO
from src.envBuilder import createGymEnv, createUnityEnv

def episode(env,agent,nr_episode=0, render=False):
    state = env.reset()
    undiscounted_return = 0
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
        if time_step % 4000 == 1:
            agent.update()
        
        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    return undiscounted_return

"""
at the moment without multiple instances at once
"""
def training(env,agent,episodes,render):
    for nr_episode in range(episodes):
        episode(env,agent,nr_episode,render)
    
    
def startTraining(args,env):            

    params = {}
    params["has_continuous_action_space"] = True
    params["update_timestep"] = args.u_step
    params["K_epochs"] = args.k_epochs               
    params["eps_clip"] = args.epsilon_clip        
    params["gamma"] = args.gamma      
    params["lr_actor"] = args.lr_actor     
    params["lr_critic"] = args.lr_critic
    params["action_std"] = args.action_std  

   # print("training environment name : " + env_name)
    
    state_dim = env.observation_space.shape[0]
    if params["has_continuous_action_space"]:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
        
    agent = PPO(state_dim, 
                action_dim, 
                params["lr_actor"], 
                params["lr_critic"], 
                params["gamma"], 
                params["K_epochs"], 
                params["eps_clip"], 
                params["has_continuous_action_space"], 
                params["action_std"])

    training(env=env, agent=agent, episodes=args.episodes,render=args.replay)
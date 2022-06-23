import os
import numpy as np
import gym
from src.PPO import PPO
import matplotlib.pyplot as plt
from src.envBuilder import createGymEnv, createUnityEnv
from torch.utils.tensorboard import SummaryWriter

time_step = 0

def episode(env, checkpoint_path,
            agent,
            writer,
            nr_episode=0,
            render=False,
            update_timestep=20, 
            action_std_decay_rate = 0.01,
            min_action_std = 0.001,
            action_std_decay_freq = int(2.5e5), 
            save_model_freq = int(1e1)):
    state = env.reset()
    total_return = 0
    done = False
   
    global time_step
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
    writer.add_scalar("reward x episode", total_return, nr_episode)
    
    return total_return


"""
at the moment without multiple instances at once
"""

def training(env, checkpoint_path, agent, nr_episodes, render, update_timestep, action_std_decay_rate, min_action_std, action_std_decay_freq, save_model_freq, writer):
    totalReward = 0.0
    for nr_episode in range(nr_episodes):
        totalReward += episode(env, checkpoint_path, agent, writer, nr_episode, render, update_timestep, action_std_decay_rate, min_action_std, action_std_decay_freq, save_model_freq)
    return totalReward
       

       
    
def startTraining(args,env):            

    params = {}
    params["update_timestep"] = args.u_step #64
    params["K_epochs"] = args.k_epochs  #30         # should probably be between [3, 30]                       
    params["eps_clip"] = args.epsilon_clip #0.2     # should probably be between [0.1, 0.3]    
    params["gamma"] = args.gamma #0.99              # probably 0.99 at its best  
    params["lr_actor"] = args.lr_actor #0.0003    
    params["lr_critic"] = args.lr_critic #0.001
    params["action_std"] = args.action_std #0.6  
    params["action_std_decay_rate"] = 0.0005         # action standard deviation decay rate
    params["min_action_std"] = 0.001                  # minimum action standard deviation
    params["action_std_decay_freq"] = int(2.5e5)    # action standard deviation decay frequency
    params["save_model_freq"] = int(1e1)            # save model to checkpoint frequency 

    # start environment
    # no graphics: faster, no visual rendering 
    #env = createUnityEnv(no_graphics=True)
    #env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space[0].shape[0]

    action_dim = env.action_space.shape[0]

    # create directory and file to save checkpoint to
    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = os.path.join(directory, 'net_{}_{}'.format('logs', 0))
    
    writer = SummaryWriter()
    
    # create PPO driven agent with hyperparameters
    agent = PPO(state_dim, 
                action_dim, 
                params["lr_actor"], 
                params["lr_critic"], 
                params["gamma"], 
                params["K_epochs"], 
                params["eps_clip"], 
                params["action_std"])
    
    # train agent
    endReward = training(env=env, checkpoint_path=checkpoint_path, agent=agent, nr_episodes=args.episodes, update_timestep = params["update_timestep"], action_std_decay_rate=params["action_std_decay_rate"], min_action_std=params["min_action_std"], action_std_decay_freq=params["action_std_decay_freq"], save_model_freq=params["save_model_freq"], render=args.replay, writer=writer)
    meanRewardOverall = endReward/time_step
    writer.add_hparams( dict(params),{'hparam/endReward':meanRewardOverall})

    #close writer
    writer.close()

    #close environment
    env.close()


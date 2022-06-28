from ast import arg
import numpy as np
from src.algorithms.RandomAgent import RandomAgent
from src.PPO import PPO
from mlagents_envs.base_env import ActionTuple
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import platform

import os

CONST_LOG_ACTION_STD = "training/action_std x timestep"
CONST_LOG_EPISODE_REWARD = "training/reward x episode"
CONST_LOG_TIMESTEP_REWARD = "training/reward x timestep"
CONST_LOG_HYPER_PARAMETERS = "training/h_param"
CONST_LOG_ACTION_FREQUENCY = "training/action_frequency"

def trainingUnityVec(env,
                agent,
                logWriter,
                nr_episodes,
                update_timestep, 
                action_std_decay_rate,
                min_action_std,
                action_std_decay_freq,
                simCount):
    time_step = 0
    
    # name of the "unity behavior"
    bName = list(env.behavior_specs)[0]
    
    for nr_episode in range(nr_episodes):
        # it should be possible, that this line is not needed but for now we will reset the env at the start of every episode
        env.reset()
        activeEnvs, termEnvs = env.get_steps(bName)
        
        reward_episode = np.full(simCount, 0.0)
        
        dones = np.full(simCount,False)
        waiting =  np.full(simCount,False)
        
        while not all(dones):
            for index in activeEnvs:
                activeEnv = activeEnvs[index]
                simId = activeEnv.agent_id
                if not dones[simId]:# and not waiting[simId]:
                    action = agent.select_action(activeEnv.obs,simId)
                    # Convert action to a "unity" readable action
                    action = ActionTuple(np.array([action], dtype=np.float32))
                    
                    env.set_action_for_agent(bName,simId, action)
                    
                    waiting[simId] = True
              
            # execute the steps in unity
            env.step()
            activeEnvs, termEnvs = env.get_steps(bName)
            
            for simId in range(simCount):
                if not dones[simId]:# and waiting[simId]:
                    if simId not in activeEnvs and simId not in termEnvs:
                        continue
                    
                    reward = 0
                    if simId in activeEnvs: # The agent requested a decision
                        reward += activeEnvs[simId].reward
                    if simId in termEnvs: # The agent terminated its episode
                        reward += termEnvs[simId].reward
                        dones[simId] = True
                    reward_episode[simId] += reward
                    
                    agent.save_action_reward(reward,dones[simId],simId)
                    #waiting[simId] = False
                    
            # 4. Integrate new experience into agent
            if time_step % update_timestep == 1:
                agent.update()
              
            if time_step % action_std_decay_freq == 1:
                action_std = agent.decay_action_std(action_std_decay_rate, min_action_std)
                logWriter.add_scalar(CONST_LOG_ACTION_STD, action_std, nr_episode)
     
            time_step += 1
            
        reward_mean_episode = np.mean(reward_episode)
        print(nr_episode, ":", reward_mean_episode)
        logWriter.add_scalar(CONST_LOG_EPISODE_REWARD, reward_mean_episode, nr_episode)  


"""
train ppo with an unity env. This training will only use one simulation/agent
"""
def trainingUnity(env,
                agent,
                logWriter,
                nr_episodes,
                update_timestep, 
                action_std_decay_rate,
                min_action_std,
                action_std_decay_freq):
    
    # for plotting of action distribution
    action_dist = []
    plot_histogram_step = 0
    
    
    time_step = 0

    # name of the "unity behavior"
    bName = list(env.behavior_specs)[0]
    
    # this uses only one env at a time => we only use env 0
    envId = 0
    
    for nr_episode in range(nr_episodes):
        env.reset()
        activeEnvs, termEnvs = env.get_steps(bName)
        
        reward_episode = 0
        done = False
        
        while not done:
            # Generate 
            # an action for all envs
            action = agent.select_action(activeEnvs[envId].obs)
            
            # add action for plotting
            action_dist.append(action)
           
            # Convert action to a "unity" readable action
            action = ActionTuple(np.array([action], dtype=np.float32))
            
            # Set the actions
            env.set_action_for_agent(bName,envId, action)

            # Move the simulation forward
            env.step()

            # Get the new simulation results
            activeEnvs, termEnvs = env.get_steps(bName)
        
            reward = 0 
            if envId in activeEnvs: # The agent requested a decision
                reward += activeEnvs[envId].reward
            if envId in termEnvs: # The agent terminated its episode
                reward += termEnvs[envId].reward
                done = True
        
            if envId not in activeEnvs and envId not in termEnvs:
                print("Carefull enviId wasn't present")
            
            # 3. Update buffer
            agent.save_action_reward(reward,done)
  
            # 4. Integrate new experience into agent
            if time_step % update_timestep == 1:      
                agent.update()
                
            if time_step % action_std_decay_freq == 1:
                action_std = agent.decay_action_std(action_std_decay_rate, min_action_std)
                logWriter.add_scalar(CONST_LOG_ACTION_STD, action_std, time_step)     
                  
            reward_episode += reward
            logWriter.add_scalar(CONST_LOG_TIMESTEP_REWARD, reward, time_step)
            time_step += 1
            
        print(nr_episode, ":", reward_episode)
        logWriter.add_scalar(CONST_LOG_EPISODE_REWARD, reward_episode, time_step)

        # plot action distribution
        if nr_episode % (0.2 * nr_episodes) == 1: # number of sessions
            action_freq = np.array(action_dist)
            logWriter.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(action_freq), global_step = plot_histogram_step)    
            plot_histogram_step += 1
        

def trainingGym(env,
                agent,
                logWriter,
                nr_episodes,
                update_timestep, 
                action_std_decay_rate,
                min_action_std,
                action_std_decay_freq):
    
    time_step = 0
    
    # plot action distribution
    action_dist = []
    plot_histogram_step = 0
    
    for nr_episode in range(nr_episodes):
        state = env.reset()
        reward_episode = 0
        
        done = False
        while not done:
            # 1. Select action according to policy
            action = agent.select_action(state)
            
            # add action for plotting
            action_dist.append(action)
            
            # 2. Execute selected action
            next_state, reward, done, _ = env.step(action)
            
            # 3. Update buffer
            agent.save_action_reward(reward,done)
    
            # 4. Integrate new experience into agent
            if time_step % update_timestep == 1:      
                agent.update()
                
                
            if time_step % action_std_decay_freq == 1:
                action_std = agent.decay_action_std(action_std_decay_rate, min_action_std)
                logWriter.add_scalar(CONST_LOG_ACTION_STD, action_std, time_step)
            
            state = next_state
            reward_episode += reward
            logWriter.add_scalar(CONST_LOG_TIMESTEP_REWARD, reward, time_step)
            time_step += 1
            
        print(nr_episode, ":", reward_episode)
        logWriter.add_scalar(CONST_LOG_EPISODE_REWARD, reward_episode, time_step)
        
        # plot action distribution
        if nr_episode % (0.2 * nr_episodes) == 1: # number of sessions
            action_freq = np.array(action_dist)
            logWriter.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(action_freq), global_step = plot_histogram_step)    
            plot_histogram_step += 1
  
def startTraining(args, env, state_dim, action_dim, simCount):            
    osName = platform.node()
    currentTimeInSec = int(round(datetime.now().timestamp()))
    logDir = f"runs/logs/{args.tag}/{osName}-{currentTimeInSec}"
    if not os.path.exists(logDir):
        os.makedirs(logDir)

    logWriter = SummaryWriter(log_dir=logDir)
    
    logWriter.add_text(CONST_LOG_HYPER_PARAMETERS,str(args))

    device = torch.device('cpu')
    if(torch.cuda.is_available() and not args.force_cpu): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    

    if (args.agent == "ppo"):
        # create PPO driven agent with hyperparameters
        agent = PPO(state_dim, 
                action_dim,
                args.lr_actor, 
                args.lr_critic, 
                args.gamma, 
                args.k_epochs, 
                args.epsilon_clip, 
                args.action_std,
                device,
                simCount)
    elif (args.agent == "random_agent"):
        agent = RandomAgent(state_dim,action_dim)
    else:
        print(f"the agend isn't know: {args.agent}")

    if args.env == "unity":
        if simCount == 1:
            trainingUnity(env,
                    agent,
                    logWriter,
                    args.episodes,
                    args.update_timestep, 
                    args.action_std_decay_rate,
                    args.min_action_std,
                    args.action_std_decay_freq)
        else:
            trainingUnityVec(env,
                    agent,
                    logWriter,
                    args.episodes,
                    args.update_timestep, 
                    args.action_std_decay_rate,
                    args.min_action_std,
                    args.action_std_decay_freq,
                    simCount)
            
    else:
        trainingGym(env,
                agent,
                logWriter,
                args.episodes,
                args.update_timestep, 
                args.action_std_decay_rate,
                args.min_action_std,
                args.action_std_decay_freq)

    #close writer
    logWriter.close()

    #close environment
    env.close()


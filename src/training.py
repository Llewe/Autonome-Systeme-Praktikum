from ast import arg
import numpy as np
from src.algorithms.RandomAgent import RandomAgent
from src.PPO import PPO
from mlagents_envs.base_env import ActionTuple
import torch
from torch.utils.tensorboard import SummaryWriter

import os

CONST_LOG_ACTION_STD = "training/action_std x timestep"
CONST_LOG_EPISODE_REWARD = "training/return x episode"
CONST_LOG_HYPER_PARAMETERS = "training/h_param"
CONST_LOG_ACTION_FREQUENCY = "training/action_frequency"
CONST_CHECKPOINT_COUNT = 5  # change the number of model savings

def trainingUnityVec(env,
                     agent,
                     logWriter,
                     nr_episodes,
                     update_timestep,
                     action_std_decay_rate,
                     min_action_std,
                     action_std_decay_freq,
                     simCount,
                     modelPath):
    print("this function is not implemented yet")
    pass


"""
train ppo with an unity env. This training will only use one simulation/agent
"""
def trainingUnity(env,
                  agent,
                  logWriter,
                  max_timesteps,
                  update_timestep,
                  action_std_decay_rate,
                  min_action_std,
                  action_std_decay_freq, 
                  modelPath):

    # for plotting of action distribution
    action_dist = []
    plot_histogram_step = 0
    saving_interval = (1/CONST_CHECKPOINT_COUNT) * max_timesteps
    model_checkpoint = saving_interval

    time_step = 0

    # name of the "unity behavior"
    bName = list(env.behavior_specs)[0]

    # this uses only one env at a time => we only use env 0
    envId = 0
    
    nr_episode = 0
    
    while time_step < max_timesteps:
        env.reset()
        activeEnvs, termEnvs = env.get_steps(bName)

        reward_episode = 0
        done = False

        while not done:
            # Generate
            # an action for all envs
            action = agent.select_action(activeEnvs[envId].obs)

            # clip action space
            action = np.nan_to_num(action)
            action = np.clip(action, -1, 1)

            # add action for plotting
            action_dist.append(action)

            # Convert action to a "unity" readable action
            action = ActionTuple(np.array([action], dtype=np.float32))

            # Set the actions
            env.set_action_for_agent(bName, envId, action)

            # Move the simulation forward
            env.step()

            # Get the new simulation results
            activeEnvs, termEnvs = env.get_steps(bName)

            reward = 0
            if envId in activeEnvs:  # The agent requested a decision
                reward += activeEnvs[envId].reward
            if envId in termEnvs:  # The agent terminated its episode
                reward += termEnvs[envId].reward
                done = True

            if envId not in activeEnvs and envId not in termEnvs:
                print("Carefull enviId wasn't present")

            # 3. Update buffer
            agent.save_action_reward(reward, done)

            # 4. Integrate new experience into agent
            if time_step % update_timestep == 1:
                agent.update()
                
            if time_step % action_std_decay_freq == 1:
                action_std = agent.decay_action_std(
                    action_std_decay_rate, min_action_std)
                logWriter.add_scalar(CONST_LOG_ACTION_STD,
                                     action_std, time_step)

            reward_episode += reward
            time_step += 1
          
        # next timestep after last checkpoint when episode is finished
        if ((model_checkpoint <= time_step) or (time_step >= max_timesteps)):
            # save model
            agent.save(modelPath, time_step)
            # add action log 
            logWriter.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(
                np.array(action_dist)), global_step=plot_histogram_step)
            # clear action buffer after histogram session
            action_dist *= 0
            plot_histogram_step += 1
            model_checkpoint += saving_interval
       
        
        print(nr_episode, ":", reward_episode)
        logWriter.add_scalar(CONST_LOG_EPISODE_REWARD,
                             reward_episode, time_step)
        
        nr_episode+=1

    if len(action_dist) > 0:
        logWriter.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(
            np.array(action_dist)), global_step=plot_histogram_step + 1)


def trainingGym(env,
                agent,
                logWriter,
                max_timesteps,
                update_timestep,
                action_std_decay_rate,
                min_action_std,
                action_std_decay_freq, 
                modelPath):

    time_step = 0

    # plot action distribution
    action_dist = []
    plot_histogram_step = 0
    saving_interval = (1/CONST_CHECKPOINT_COUNT) * max_timesteps
    model_checkpoint = saving_interval

    nr_episode = 0
    
    while time_step < max_timesteps:
        state = env.reset()
        reward_episode = 0

        done = False
        while not done:
            # 1. Select action according to policy
            action = agent.select_action(state)

            # clip action space
            action = np.nan_to_num(action)
            action = np.clip(action, -1, 1)

            # add action for plotting
            action_dist.append(action)

            # 2. Execute selected action
            next_state, reward, done, _ = env.step(action)

            # 3. Update buffer
            agent.save_action_reward(reward, done)

            # 4. Integrate new experience into agent
            if time_step % update_timestep == 1:
                agent.update()
      

            if time_step % action_std_decay_freq == 1:
                action_std = agent.decay_action_std(
                    action_std_decay_rate, min_action_std)
                logWriter.add_scalar(CONST_LOG_ACTION_STD,
                                     action_std, time_step)

            state = next_state
            reward_episode += reward
            time_step += 1
        
        # next timestep after last checkpoint when episode is finished
        if ((model_checkpoint <= time_step) or (time_step >= max_timesteps)):
            # save model
            agent.save(modelPath, time_step)
            # add action log 
            logWriter.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(
                np.array(action_dist)), global_step=plot_histogram_step)
            # clear action buffer after histogram session
            action_dist *= 0
            plot_histogram_step += 1
            model_checkpoint += saving_interval
        

        print(nr_episode, ":", reward_episode)
        logWriter.add_scalar(CONST_LOG_EPISODE_REWARD,
                             reward_episode, time_step)
        
        nr_episode+=1
    
    if len(action_dist) > 0:
        logWriter.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(
            np.array(action_dist)), global_step=plot_histogram_step + 1)




def startTraining(args, env, state_dim, action_dim, simCount, output_dir, folderPath):
    
    # create path for saving model     
    modelPath = output_dir + "/models" + folderPath 
    
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    
    # create log path
    logPath = output_dir + "/training-logs" + folderPath
    
    if not os.path.exists(logPath):
        os.makedirs(logPath)

    logWriter = SummaryWriter(log_dir=logPath)

    logWriter.add_text(CONST_LOG_HYPER_PARAMETERS, str(args))

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
                    logWriter,
                    simCount)
    elif (args.agent == "random_agent"):
        agent = RandomAgent(state_dim, action_dim)
    else:
        print(f"the agend isn't know: {args.agent}")

    if args.env == "unity":
        if simCount == 1:
            trainingUnity(env,
                          agent,
                          logWriter,
                          args.max_timesteps,
                          args.update_timestep,
                          args.action_std_decay_rate,
                          args.min_action_std,
                          args.action_std_decay_freq,
                          modelPath)
        else:
            trainingUnityVec(env,
                             agent,
                             logWriter,
                             args.max_timesteps,
                             args.update_timestep,
                             args.action_std_decay_rate,
                             args.min_action_std,
                             args.action_std_decay_freq,
                             simCount,
                             modelPath)

    else:
        trainingGym(env,
                    agent,
                    logWriter,
                    args.max_timesteps,
                    args.update_timestep,
                    args.action_std_decay_rate,
                    args.min_action_std,
                    args.action_std_decay_freq,
                    modelPath)

    # close writer
    logWriter.close()

    # close environment
    env.close()

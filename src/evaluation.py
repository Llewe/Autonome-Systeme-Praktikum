from ast import arg
import numpy as np
from src.algorithms.RandomAgent import RandomAgent
from src.PPO import PPO
from mlagents_envs.base_env import ActionTuple
import torch
from torch.utils.tensorboard import SummaryWriter

import os
import glob


CONST_LOG_EPISODE_REWARD = "eval/return x episode"
CONST_LOG_HYPER_PARAMETERS = "eval/h_param"
CONST_LOG_ACTION_FREQUENCY = "eval/action_frequency"
CONST_LOG_AVG_REWARD = "eval/average_reward"


"""
test ppo with a unity env. 
Multiple simulations/agents

"""

def testUnityVec(env,
                     agent,
                     logWriter,
                     nr_episodes,
                     simCount):
    time_step = 0

    # name of the "unity behavior"
    bName = list(env.behavior_specs)[0]

    for nr_episode in range(nr_episodes):
        # it should be possible, that this line is not needed but for now we will reset the env at the start of every episode
        env.reset()
        activeEnvs, termEnvs = env.get_steps(bName)

        reward_episode = np.full(simCount, 0.0)

        dones = np.full(simCount, False)
        waiting = np.full(simCount, False)

        while not all(dones):
            for index in activeEnvs:
                activeEnv = activeEnvs[index]
                simId = activeEnv.agent_id
                if not dones[simId]:  # and not waiting[simId]:
                    action = agent.select_action(activeEnv.obs, simId)
                    # Convert action to a "unity" readable action
                    action = ActionTuple(np.array([action], dtype=np.float32))

                    env.set_action_for_agent(bName, simId, action)

                    waiting[simId] = True

            # execute the steps in unity
            env.step()
            activeEnvs, termEnvs = env.get_steps(bName)

            for simId in range(simCount):
                if not dones[simId]:  # and waiting[simId]:
                    if simId not in activeEnvs and simId not in termEnvs:
                        continue

                    reward = 0
                    if simId in activeEnvs:  # The agent requested a decision
                        reward += activeEnvs[simId].reward
                    if simId in termEnvs:  # The agent terminated its episode
                        reward += termEnvs[simId].reward
                        dones[simId] = True
                    reward_episode[simId] += reward

                    agent.save_action_reward(reward, dones[simId], simId)
                    # waiting[simId] = False

            time_step += 1

        reward_mean_episode = np.mean(reward_episode)
        print(nr_episode, ":", reward_mean_episode)
        logWriter.add_scalar(CONST_LOG_EPISODE_REWARD,
                             reward_mean_episode, nr_episode)

"""
test ppo with a unity env. 
One simulation/agent

"""

def testUnity(env,
              agent,
              logWriter,
              nr_episodes):

    # for plotting of action distribution
    action_dist = []
    total_rewards = 0
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

            reward_episode += reward
            time_step += 1

        total_rewards += reward_episode

        print(nr_episode, ":", reward_episode)
        logWriter.add_scalar(CONST_LOG_EPISODE_REWARD,
                             reward_episode, time_step)

        # plot action distribution
        if nr_episode % (0.1 * nr_episodes) == 0:  # number of sessions
            action_freq = np.array(action_dist)
            logWriter.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(
                action_freq), global_step=plot_histogram_step)
            # clear action buffer after histogram session
            action_dist = []
            plot_histogram_step += 1

    logWriter.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(
        action_freq), global_step=plot_histogram_step + 1)

    average_reward = round((total_rewards / nr_episodes), 2)
    print("average test reward : " + str(average_reward))
    logWriter.add_text(CONST_LOG_AVG_REWARD, str(average_reward))


"""
test ppo with gym env. 
One simulation/agent

"""

def testGym(env,
                agent,
                logWriter,
                nr_episodes):

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

            # clip action space
            action = np.nan_to_num(action)
            action = np.clip(action, -1, 1)

            # add action for plotting
            action_dist.append(action)

            # 2. Execute selected action
            next_state, reward, done, _ = env.step(action)

            state = next_state
            reward_episode += reward
            time_step += 1

        print(nr_episode, ":", reward_episode)
        logWriter.add_scalar(CONST_LOG_EPISODE_REWARD,
                             reward_episode, time_step)

        # plot action distribution
        if nr_episode % (0.2 * nr_episodes) == 1:  # number of sessions
            action_freq = np.array(action_dist)
            logWriter.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(
                action_freq), global_step=plot_histogram_step)
            # clear action buffer after histogram session
            action_dist = []
            plot_histogram_step += 1

    logWriter.add_histogram(CONST_LOG_ACTION_FREQUENCY, torch.from_numpy(
        action_freq), global_step=plot_histogram_step + 1)

"""
Scans for the biggest number in a list of path strings.
The number must start after a "-" and end with a "."
"""
def findNewestPath(paths):
    newestTime = 0
    newestPath = ""
    for p in paths:
        time = int(p[(p.rfind("-")+1):].split(".")[0])
        if (time > newestTime):
            newestTime = time
            newestPath = p
    return newestPath
    
def startEval(args, env, state_dim, action_dim, simCount, output_dir, folderPath):
    
     # create log path
    logPath = output_dir + "/eval-logs" + folderPath
    
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

        # load latest model with specified environment and tag
        modelDir = glob.glob(output_dir + f"/models/{args.env}/{args.env_name}/{args.tag}/*")
     
        if len(modelDir) > 0:
            print("Info: Directory contains multiple entries. Choosing the latest entry, which may not be your intended model.")
        
        latestModel = findNewestPath(modelDir)
        checkpointList = glob.glob(latestModel + r'\*pth')
        
        modelPath = findNewestPath(checkpointList)
        
        print(checkpointList)
        print(modelPath)
        
        # load model  
        print("loading network from : " + modelPath)
        agent.load(modelPath)

    elif (args.agent == "random_agent"):
        agent = RandomAgent(state_dim, action_dim)
    else:
        print(f"unknown agent: {args.agent}")

    if args.env == "unity":
        if simCount == 1:

            testUnity(env,
                      agent,
                      logWriter,
                      args.episodes)
        else:
            testUnityVec(env,
                    agent,
                    logWriter,
                    args.episodes,
                    simCount)

    else:
        testGym(env,
                agent,
                logWriter,
                args.episodes)

    # close writer
    logWriter.close()

    # close environment
    env.close()

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


CONST_LOG_EPISODE_REWARD = "testing/return x episode"
CONST_LOG_HYPER_PARAMETERS = "testing/h_param"
CONST_LOG_ACTION_FREQUENCY = "testing/action_frequency"

"""
test ppo with an unity env. 
One simulation/agent

"""


def testUnity(env,
              agent,
              logWriter,
              nr_episodes,
              update_timestep,
              action_std_decay_rate,
              min_action_std,
              action_std_decay_freq):

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


def startTesting(args, env, state_dim, action_dim, simCount):
    osName = platform.node()
    currentTimeInSec = int(round(datetime.now().timestamp()))
    logDir = f"tests/logs/{args.tag}/{osName}-{currentTimeInSec}"

    if not os.path.exists(logDir):
        os.makedirs(logDir)

    logWriter = SummaryWriter(log_dir=logDir)

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

        dir_name = "ppo"
        directory = "PPO_preTrained" + '/' + dir_name + '/'
        checkpoint_path = directory + "PPO_train.pth"
        print("loading network from : " + checkpoint_path)

        agent.load(checkpoint_path)

    elif (args.agent == "random_agent"):
        agent = RandomAgent(state_dim, action_dim)
    else:
        print(f"the agend isn't know: {args.agent}")

    if args.env == "unity":
        if simCount == 1:

            testUnity(env,
                      agent,
                      logWriter,
                      args.episodes,
                      args.update_timestep,
                      args.action_std_decay_rate,
                      args.min_action_std,
                      args.action_std_decay_freq)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    # close writer
    logWriter.close()

    # close environment
    env.close()

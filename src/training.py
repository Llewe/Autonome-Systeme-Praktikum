from operator import le
import os
from tkinter.tix import Tree
import numpy as np
import gym
from src.PPO import PPO
from src.PPOVec import PPOVec
import matplotlib.pyplot as plt
from mlagents_envs.base_env import ActionTuple
from src.envBuilder import createGymEnv, createUnityEnv, createPureUnityEnv

def episodeVec(env, checkpoint_path, agent,nr_episode=0, render=False, update_timestep=4000, action_std_decay_rate = 0.01, min_action_std = 0.001, action_std_decay_freq = int(2.5e5), save_model_freq = int(1e1)):
    env.reset()
    
    behavior_name = list(env.behavior_specs)[0]
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    done = False # all simulations done?
    simulationCount = len(decision_steps) # in 3DBall 12 Boxes => 12 simulations
    
    episode_rewards = np.full(simulationCount, 0.0)
    time_step = 0
    
    waiting =  np.full( simulationCount,False)
    
    dones = np.full( simulationCount,False)
    
    while not done:
        for index in decision_steps:
            decition_step = decision_steps[index]
            simId = decition_step.agent_id
            if not dones[simId]:# and not waiting[simId]:
                action = agent.select_action(decition_step.obs,simId)

                action = ActionTuple(np.array([action], dtype=np.float32))
                
                env.set_action_for_agent(behavior_name,simId, action)
                
                waiting[simId] = True
                
                
        # execute the steps in unity

        env.step()

        # Get the new simulation results
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        for simId in range(simulationCount):
            if not dones[simId]:# and waiting[simId]:
                if simId not in decision_steps and simId not in terminal_steps:
                    continue
                
                reward = 0
                if simId in decision_steps: # The agent requested a decision
                    reward += decision_steps[simId].reward
                if simId in terminal_steps: # The agent terminated its episode
                    reward += terminal_steps[simId].reward
                    dones[simId] = True
                episode_rewards[simId] += reward
                
                agent.save_action_reward(reward,dones[simId],simId)
                #waiting[simId] = False

        # 4. Integrate new experience into agent
        if time_step % update_timestep == 1:
          #  print(f"doneActions{doneActions}, savedActions{savedActions}, finalActions{finalActions}")  
            agent.update()
              
        if time_step % action_std_decay_freq == 1:
            agent.decay_action_std(action_std_decay_rate, min_action_std)


    #     if time_step % save_model_freq == 0:
    #        agent.save(checkpoint_path)
            
        time_step += 1
        # no more active agents => done
        
        if all(dones):
            done = True

            
   # print(f"Total rewards for episode {nr_episode} is {np.mean(episode_rewards)}")
    
    return np.mean(episode_rewards)
    

def episode(env, checkpoint_path, agent,nr_episode=0, render=False, update_timestep=4000, action_std_decay_rate = 0.01, min_action_std = 0.001, action_std_decay_freq = int(2.5e5), save_model_freq = int(1e1)):
    env.reset()
    
    behavior_name = list(env.behavior_specs)[0]
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1 # -1 indicates not yet tracking
    done = False # For the tracked_agent
    episode_rewards = 0 # For the tracked_agent
    time_step = 0
    while not done:
        # Track the first agent we see if not tracking
        # Note : len(decision_steps) = [number of agents that requested a decision]
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[3]
        elif len(decision_steps) == 0:
            break
        # Generate an action for all agents
        action = agent.select_action(decision_steps[tracked_agent].obs)
        # print(f"action {type(action)} values: {action}")
        action = ActionTuple(np.array([action], dtype=np.float32))
        # Set the actions
        env.set_action_for_agent(behavior_name,tracked_agent, action)

        # Move the simulation forward
        env.step()

        # Get the new simulation results
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        
        if tracked_agent in decision_steps: # The agent requested a decision
            reward = decision_steps[tracked_agent].reward
        if tracked_agent in terminal_steps: # The agent terminated its episode
            reward = terminal_steps[tracked_agent].reward
            done = True
            
        # 3. Update buffer
        agent.buffer.rewards.append(reward)         
        agent.buffer.is_terminals.append(done)
  
        # 4. Integrate new experience into agent
        if time_step % update_timestep == 1:      
            agent.update()
              
              
        if time_step % action_std_decay_freq == 1:
            agent.decay_action_std(action_std_decay_rate, min_action_std)
        
        episode_rewards += reward
        time_step += 1

        if time_step % save_model_freq == 0:
            agent.save(checkpoint_path)
            
   # print(f"Total rewards for episode {nr_episode} is {episode_rewards}")
    
    return episode_rewards


"""
at the moment without multiple instances at once
"""

def training(env, checkpoint_path, agent, nr_episodes, render, update_timestep, action_std_decay_rate, min_action_std, action_std_decay_freq, save_model_freq):
    list_total_return = []
    meanReward = 0
    for nr_episode in range(nr_episodes):
        meanReward += episodeVec(env, checkpoint_path, agent, nr_episode, render, update_timestep, action_std_decay_rate, min_action_std, action_std_decay_freq, save_model_freq)
        if (nr_episode % 20 == 0):
            print(f"Total rewards for episode {nr_episode} is {meanReward/20.0}") 
            meanReward = 0
       

       
    
def startTraining(args,env):            

    params = {}
    params["has_continuous_action_space"] = True
    params["update_timestep"] = args.u_step #64
    params["K_epochs"] = args.k_epochs  #30         # should probably be between [3, 30]                       
    params["eps_clip"] = args.epsilon_clip #0.2     # should probably be between [0.1, 0.3]    
    params["gamma"] = args.gamma #0.99              # probably 0.99 at its best  
    params["lr_actor"] = args.lr_actor #0.0003    
    params["lr_critic"] = args.lr_critic #0.001
    params["action_std"] = args.action_std #0.6  
    params["action_std_decay_rate"] = 0.00005          # action standard deviation decay rate
    params["min_action_std"] = 0.1                  # minimum action standard deviation
    params["action_std_decay_freq"] = int(2.5e5)    # action standard deviation decay frequency
    params["save_model_freq"] = int(1e1)            # save model to checkpoint frequency 

    # start environment
    # no graphics: faster, no visual rendering 
    #env = createUnityEnv(no_graphics=True)
    #env = gym.make('CartPole-v1')
    
   # state_dim = env.observation_space.shape[0]

   # if params["has_continuous_action_space"]:
  #      action_dim = env.action_space.shape[0]
    
   # else:
   #     action_dim = env.action_space.n

    # create directory and file to save checkpoint to
    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = os.path.join(directory, 'net_{}_{}'.format('logs', 0))

    
    # create PPO driven agent with hyperparameters
    agent = PPOVec(8,#state_dim, 
                2,#action_dim, 
                params["lr_actor"], 
                params["lr_critic"], 
                params["gamma"], 
                params["K_epochs"], 
                params["eps_clip"], 
                params["action_std"],12)
    #print(f"state_dim {state_dim}, action_dim {action_dim}")
    # train agent
    training(env=env, checkpoint_path=checkpoint_path, agent=agent, nr_episodes=args.episodes, update_timestep = params["update_timestep"], action_std_decay_rate=params["action_std_decay_rate"], min_action_std=params["min_action_std"], action_std_decay_freq=params["action_std_decay_freq"], save_model_freq=params["save_model_freq"], render=args.replay)

    #close environment
    env.close()

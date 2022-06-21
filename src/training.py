import os
from tkinter.tix import Tree
import numpy as np
import gym
from src.PPO import PPO
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
    dones = np.full( simulationCount,False)

    time_step = 0
    
    while not done:
        # Calculate action for every simulation
        for i in range(simulationCount):
            if i in decision_steps: #simulation still active?
                action = agent.select_action(decision_steps.obs[0][i])
                action = ActionTuple(np.array([action], dtype=np.float32)) # "convert" action
                
                env.set_action_for_agent(behavior_name,i, action)
        
        # execute the steps in unity
        env.step()

        # Get the new simulation results
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        # calculate the reward of each agent
        for i in range(simulationCount):
            # only update if agent was active and did a step

            reward = 0
            if i in decision_steps: # The agent requested a decision
                reward  = decision_steps[i].reward
            if i in terminal_steps: # The agent terminated its episode
                reward  = terminal_steps[i].reward
                dones[i] = True
            episode_rewards[i] += reward
                # 3. Update buffer
            agent.buffer.rewards.append(reward)         
            agent.buffer.is_terminals.append(dones[i])

        
        # 4. Integrate new experience into agent
        if time_step % update_timestep == 1:      
            agent.update()
              
        if time_step % action_std_decay_freq == 1:
            agent.decay_action_std(action_std_decay_rate, min_action_std)


        if time_step % save_model_freq == 0:
            agent.save(checkpoint_path)
            
        time_step += 1
        # no more active agents => done
        if len(decision_steps) == 0:
            done = True

            
    print(f"Total rewards for episode {nr_episode} is {np.mean(episode_rewards)}")
    
    return episode_rewards
    

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
            print(f"set tracked_agent {tracked_agent}, lenght = {len(decision_steps)}")
        elif len(decision_steps) == 0:
            print(f"decision_length = {len(decision_steps)}")
            break
 
        # Generate an action for all agents
        # print(f"decision_steps {decision_steps.obs}")
        action = agent.select_action(decision_steps.obs[0][tracked_agent])
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
            
    print(f"Total rewards for episode {nr_episode} is {episode_rewards}")
    
    return episode_rewards


"""
at the moment without multiple instances at once
"""

def training(env, checkpoint_path, agent, nr_episodes, render, update_timestep, action_std_decay_rate, min_action_std, action_std_decay_freq, save_model_freq):
    list_total_return = []
    for nr_episode in range(nr_episodes):
        episodeVec(env, checkpoint_path, agent, nr_episode, render, update_timestep, action_std_decay_rate, min_action_std, action_std_decay_freq, save_model_freq)
       

       
    
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
    params["action_std_decay_rate"] = 0.0005          # action standard deviation decay rate
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
    agent = PPO(8,#state_dim, 
                2,#action_dim, 
                params["lr_actor"], 
                params["lr_critic"], 
                params["gamma"], 
                params["K_epochs"], 
                params["eps_clip"], 
                params["has_continuous_action_space"], 
                params["action_std"])
    #print(f"state_dim {state_dim}, action_dim {action_dim}")
    # train agent
    training(env=env, checkpoint_path=checkpoint_path, agent=agent, nr_episodes=args.episodes, update_timestep = params["update_timestep"], action_std_decay_rate=params["action_std_decay_rate"], min_action_std=params["min_action_std"], action_std_decay_freq=params["action_std_decay_freq"], save_model_freq=params["save_model_freq"], render=args.replay)

    #close environment
    env.close()

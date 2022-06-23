import os
import numpy as np
from src.PPO import PPO
from mlagents_envs.base_env import ActionTuple

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
    


""" Trains one episode for an env. There must be only one env to observe
"""
def episode(env, checkpoint_path, agent,nr_episode=0, render=False, update_timestep=4000, action_std_decay_rate = 0.01, min_action_std = 0.001, action_std_decay_freq = int(2.5e5), save_model_freq = int(1e1)):
    env.reset()
    
    # name of the "unity behavior"
    bName = list(env.behavior_specs)[0]
    
    activeEnvs, termEnvs = env.get_steps(bName)
    
    # this uses only one env at a time => we only use env 0
    envId = 0 #
    done = False
    
    episode_rewards = 0
    time_step = 0
    while not done:
        
        # Generate an action for all envs
        action = agent.select_action(activeEnvs[envId].obs)
        
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
            
    return episode_rewards


"""
at the moment without multiple instances at once
"""

def training(env, checkpoint_path, agent, nr_episodes, render, update_timestep, action_std_decay_rate, min_action_std, action_std_decay_freq, save_model_freq):
    list_total_return = []
    for nr_episode in range(nr_episodes):
        reward = episode(env, checkpoint_path, agent, nr_episode, render, update_timestep, action_std_decay_rate, min_action_std, action_std_decay_freq, save_model_freq)
        list_total_return.append(reward)
        if len(list_total_return) == 20:
            print(f"Total rewards for episode {nr_episode-19}-{nr_episode+1} is {np.mean(list_total_return)}")
            list_total_return.clear()
       

       
    
def startTraining(args,env,state_dim,action_dim):            

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
    print(f"state_dim {state_dim}, action_dim {action_dim}")
    # train agent
    training(env=env,
             checkpoint_path=checkpoint_path,
             agent=agent, nr_episodes=args.episodes,
             update_timestep = params["update_timestep"],
             action_std_decay_rate=params["action_std_decay_rate"],
             min_action_std=params["min_action_std"],
             action_std_decay_freq=params["action_std_decay_freq"],
             save_model_freq=params["save_model_freq"],
             render=args.replay)

    #close environment
    env.close()

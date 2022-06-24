import numpy as np
from src.PPO import PPO
from mlagents_envs.base_env import ActionTuple
import torch
from torch.utils.tensorboard import SummaryWriter

CONST_LOG_ACTION_STD = "action_std"
CONST_LOG_EPISODE_REWARD = "reward x episode"
CONST_LOG_HYPER_PARAMETERS = "h_param"

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
                action_std = agent.decay_action_std(action_std_decay_rate, min_action_std)
                logWriter.add_scalar(CONST_LOG_ACTION_STD, action_std, nr_episode)     
                  
            reward_episode += reward
            time_step += 1

        print(nr_episode, ":", reward_episode)
        logWriter.add_scalar(CONST_LOG_EPISODE_REWARD, reward_episode, nr_episode)

def trainingGym(env,
                agent,
                logWriter,
                nr_episodes,
                update_timestep, 
                action_std_decay_rate,
                min_action_std,
                action_std_decay_freq):
    
    time_step = 0
    for nr_episode in range(nr_episodes):
        state = env.reset()
        reward_episode = 0
        
        done = False
        while not done:
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
                action_std = agent.decay_action_std(action_std_decay_rate, min_action_std)
                logWriter.add_scalar(CONST_LOG_ACTION_STD, action_std, nr_episode)
            
            state = next_state
            reward_episode += reward
            time_step += 1
            
        print(nr_episode, ":", reward_episode)
        logWriter.add_scalar(CONST_LOG_EPISODE_REWARD, reward_episode, nr_episode)
  
def startTraining(args,env,state_dim,action_dim, forceCpu = False):            
 
    logWriter = SummaryWriter()
    logWriter.add_text(CONST_LOG_HYPER_PARAMETERS,str(args))
    
    
    device = torch.device('cpu')
    if(torch.cuda.is_available() and not forceCpu): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    
    # create PPO driven agent with hyperparameters
    agent = PPO(state_dim, 
                action_dim,
                args.lr_actor, 
                args.lr_critic, 
                args.gamma, 
                args.k_epochs, 
                args.epsilon_clip, 
                args.action_std,
                device)

    if args.env == "unity":
        trainingUnity(env,
                agent,
                logWriter,
                args.episodes,
                args.update_timestep, 
                args.action_std_decay_rate,
                args.min_action_std,
                args.action_std_decay_freq)
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


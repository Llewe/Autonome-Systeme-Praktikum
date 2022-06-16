import numpy as np
import gym
from PPO import PPO

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
def training(env,agent,episodes):
    for nr_episode in range(episodes):
        episode(env,agent,nr_episode,True)
    
    
if __name__ == '__main__':
    env_name = "MountainCarContinuous-v0"
    has_continuous_action_space = True  # continuous action space; else discrete
    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    print("training environment name : " + env_name)

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
        
    # initialize a PPO agent
    agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    training(env=env, agent=agent, episodes=K_epochs)
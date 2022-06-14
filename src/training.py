def episode(env,agent,nr_episode=0, render=False):
    state = env.reset()
    undiscounted_return = 0
    done = False
    time_step = 0
    while not done:
        if render:
            env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
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
    
    
    
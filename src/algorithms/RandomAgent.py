import numpy as np

class RandomAgent:
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state, simulation=0):
        return np.random.uniform(-1.0,1.0,self.action_dim)

    """
    actions are always random
    return 1 otherwise logging would fail
    """
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        return 1
    """
    empty methods so that we can use the default training
    """
    def save_action_reward(self, reward, is_terminal, simulation=0):
        pass
    """
    empty methods so that we can use the default training
    """
    def update(self):
        pass
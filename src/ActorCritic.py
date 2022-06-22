import torch
import torch.nn as nn
import numpy as np

from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # not sure if needed ""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        self.actor = nn.Sequential(
                        layer_init(nn.Linear(state_dim, 64)),
                        nn.Tanh(),
                        layer_init(nn.Linear(64, 64)),
                        nn.Tanh(),
                        layer_init(nn.Linear(64, action_dim),std=0.01)
                    )
        # critic
        self.critic = nn.Sequential(
                        layer_init(nn.Linear(state_dim, 64)),
                        nn.Tanh(),
                        layer_init(nn.Linear(64, 64)),
                        nn.Tanh(),
                        layer_init(nn.Linear(64, 1),std=1.0)
                    )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
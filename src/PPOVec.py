import numpy
import torch
import torch.nn as nn
from src.ActorCritic import ActorCritic

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class TempBuffer:
    def __init__(self):
        self.action = {}
        self.state = {}
        self.logprob = {}

class PPOVec:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.06,simulations=1):
        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        self.tempBuffer = numpy.full(simulations,TempBuffer())
        
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ],eps=1e-5)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
        
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        else:
            pass #???
        self.set_action_std(self.action_std)     
              
    def update(self):
        
        # Monte Carlo estimate of returns
      
        rewards = []
       
        if self.buffer.is_terminals[-1]:
            discounted_reward = 0
        else:
            discounted_reward = self.policy_old.critic(self.buffer.states[-1]).item()

        for reward in reversed(self.buffer.rewards):
            
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
       
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
       
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
          
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
     
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
          
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
           
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # clear buffer
        self.buffer.clear()
        self.resetBuffer()
    
    def resetBuffer(self):
        for i in range(len(self.tempBuffer)):
            self.tempBuffer[i].state = None
            
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
    def select_action(self,state,simulation):
        with torch.no_grad():
            state = torch.FloatTensor(numpy.array(state)).to(device)
            action, action_logprob = self.policy_old.act(state)
        self.tempBuffer[simulation].state = state
        self.tempBuffer[simulation].action = action
        self.tempBuffer[simulation].logprob = action_logprob
        return action.detach().cpu().numpy().flatten()
    
    def save_action_reward(self,reward,is_terminal,simulation):
        if self.tempBuffer[simulation].state is None:
            
            print(f"deleted")
            return
        self.buffer.states.append(self.tempBuffer[simulation].state)
        self.buffer.actions.append(self.tempBuffer[simulation].action)
        self.buffer.logprobs.append(self.tempBuffer[simulation].logprob)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_terminal)
        
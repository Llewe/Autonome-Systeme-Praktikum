from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import gym

import os

"""
create a gym domain
"""
def createGymEnv(name='MountainCarContinuous-v0'):
    env = gym.make(name)    
    return env
"""
create a unity domain
"""
def createUnityEnv(name='3DBall1',no_graphics=True,time_scale=20.):
  rootDir = os.getcwd()
  unityEnvDir = os.path.join(rootDir, "unity-env",name)
  if not os.path.exists(unityEnvDir):
    print(f"Unit-Env file '{unityEnvDir}' doesn't exist.")
    exit()
  
  unityExe = os.path.abspath(os.path.join(unityEnvDir, "asp"))
  
  if(no_graphics):
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale = time_scale)
    envUnity = UnityEnvironment(file_name=unityExe,no_graphics=no_graphics, side_channels=[channel])
  else:
    envUnity = UnityEnvironment(file_name=unityExe,no_graphics=no_graphics, side_channels=[])

  env = UnityToGymWrapper(envUnity)

  return env

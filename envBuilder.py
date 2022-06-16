from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
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
def createUnityEnv(name='3DBall1_IndividualScale',no_graphics=True,time_scale=20.,):
  rootDir = os.getcwd()
  unityEnvDir = os.path.join(rootDir, "unity-env",name)
  if not os.path.exists(unityEnvDir):
    print(f"Unit-Env file '{unityEnvDir}' doesn't exist.")
    exit()
  
  unityExe = os.path.abspath(os.path.join(unityEnvDir, "asp"))
  
  if(no_graphics):
    envUnity = UnityEnvironment(file_name=unityExe,no_graphics=no_graphics, side_channels=[channel])
    channel.set_configuration_parameters(time_scale = time_scale)
  else:
    
    channel = EngineConfigurationChannel()
    env_parameters = EnvironmentParametersChannel()
    envUnity = UnityEnvironment(file_name=unityExe,no_graphics=no_graphics, side_channels=[channel,env_parameters])
    channel.set_configuration_parameters(time_scale = 1)
    #env_parameters.set_uniform_sampler_parameters("scale", 0.2,5,1)
    #env_parameters.set_uniform_sampler_parameters("scale_x", 0.2,1,1)
    #env_parameters.set_uniform_sampler_parameters("scale_y", 4,5,1)
    #env_parameters.set_uniform_sampler_parameters("scale_z", 4,5,1)
    #env_parameters.set_gaussian_sampler_parameters("mass", 0.1,20,1)
    env_parameters.set_float_parameter("scale_x",3)
    env_parameters.set_float_parameter("scale_y",2)
    env_parameters.set_float_parameter("scale_z",2)
    #env_parameters.set_uniform_sampler_parameters("gravity", 4,105,1)

  env = UnityToGymWrapper(envUnity)

  return env


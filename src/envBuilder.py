from random import Random, random
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import gym
import sys

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
def createUnityEnv(name='3DEllipsoid1-Bouncy',no_graphics=True,time_scale=20.,rngMass=False,rngGravity=False,rngScale=False,individualScale=False):
  rootDir = os.getcwd()
  unityEnvDir = os.path.join(rootDir, "unity-env",name)
  if not os.path.exists(unityEnvDir):
    print(f"Unit-Env file '{unityEnvDir}' doesn't exist.")
    exit()
  
  unityExe = os.path.abspath(unityEnvDir)
  
  envChannel = EnvironmentParametersChannel()
  
  if(no_graphics):
    channel = EngineConfigurationChannel()
    envUnity = UnityEnvironment(file_name=unityExe,no_graphics=no_graphics, side_channels=[channel,envChannel])
    channel.set_configuration_parameters(time_scale = time_scale)
  else:
    envUnity = UnityEnvironment(file_name=unityExe,no_graphics=no_graphics, side_channels=[envChannel])
    
  # Recommended parameter bounds: https://unity-technologies.github.io/ml-agents/Learning-Environment-Examples/#3dball-3d-balance-ball
  
  
  if rngMass:
    envChannel.set_gaussian_sampler_parameters("mass", 0.1,20,12)
  
  if rngGravity:
    envChannel.set_uniform_sampler_parameters("gravity", 4,105,123124)
  
  if rngScale:
    if individualScale:
      envChannel.set_uniform_sampler_parameters("scale_x", 0.2,5,31512)
      envChannel.set_uniform_sampler_parameters("scale_y",0.2,5,1231526)
      envChannel.set_uniform_sampler_parameters("scale_z",0.2,5,344684)
    else:
      envChannel.set_uniform_sampler_parameters("scale", 0.2,5,5463343)
    
  env = UnityToGymWrapper(envUnity, allow_multiple_obs=True)

  return env


def buildFromArgs(args):
  if args.env == "gym":
    env = createGymEnv()
  elif args.env == "unity":
    noGraphics = not args.replay
    env = createUnityEnv(no_graphics=noGraphics)
  else:
    print("unknown env. Falling back to gym env")
    env = createGymEnv()
  return env
  

import imp
from random import Random, random
from mlagents_envs.environment import UnityEnvironment, BehaviorSpec
from mlagents_envs.base_env import ActionSpec
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
    
  env = UnityToGymWrapper(envUnity)

  return env


def createPureUnityEnv(name='3DBall12',no_graphics=True,time_scale=20.,rngMass=False,rngGravity=False,rngScale=False,individualScale=False):
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

  
  envUnity.reset() # this reset is necessary otherwise there aren't any behavior specs present
  behavior_names = envUnity.behavior_specs.keys()

  envUnity.behavior_specs
  print("....")
  print(behavior_names)
  print(len(behavior_names))
  print(envUnity.behavior_specs)
  print("....")
  
  
  for specs in envUnity.behavior_specs.values():
    observation_specs = specs.observation_specs
    action_spec  = specs.action_spec
    print("....")
    print("....")
    print(observation_specs)
    print(len(observation_specs))
    print(observation_specs[0].shape)
    print(observation_specs[0].shape[0])
    print(observation_specs[0].dimension_property)
    print(observation_specs[0].observation_type)
    print("....")
    print("....")
    print(action_spec)
    print(action_spec.continuous_size)
  
  print("....")
  #observation_shapes
  #action_size

  #    state_dim = env.observation_space[0].shape[0]
  # if params["has_continuous_action_space"]:
  # action_dim = env.action_space.shape[0]
  return envUnity#, observation_specs[0].shape[0],action_spec.continuous_size


def buildFromArgs(args):
  if args.env == "gym":
    env = createGymEnv()
  elif args.env == "unity":
    noGraphics = not args.replay
    #env = createPureUnityEnv()
    env = createPureUnityEnv()
  else:
    print("unknown env. Falling back to gym env")
    env = createGymEnv()
  return env
  

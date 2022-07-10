from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import gym

import random

import os

"""
create a gym domain
"""
def createGymEnv(name='MountainCarContinuous-v0'):
    env = gym.make(name)   
    return env, env.observation_space.shape[0], env.action_space.shape[0], 1

"""
create a unity domain
"""
def createUnityEnv(args):
  rngMin = 0
  rngMax = 100000
  
  rootDir = os.getcwd()
  unityEnvDir = os.path.join(rootDir, "unity-env",args.env_name)
  if not os.path.exists(unityEnvDir):
    print(f"Unit-Env file '{unityEnvDir}' doesn't exist.")
    exit()
  
  unityExe = os.path.abspath(unityEnvDir)
  
  # Setup Channels
  channel = EngineConfigurationChannel()
  envChannel = EnvironmentParametersChannel()
  
  envUnity = UnityEnvironment(file_name=unityExe,no_graphics=not args.env_video, side_channels=[channel,envChannel], seed=random.randint(-10000,10000))
  
  channel.set_configuration_parameters(time_scale = args.env_timeScale)

  if args.env_rngMass:
    envChannel.set_uniform_sampler_parameters("mass",min_value = args.env_minMass, max_value=args.env_maxMass,seed=random.randint(rngMin,rngMax))
  else:
    envChannel.set_float_parameter("mass",args.env_mass)

  if args.env_rngGravity:
    envChannel.set_uniform_sampler_parameters("gravity",min_value = args.env_minGravity, max_value=args.env_maxGravity,seed=random.randint(rngMin,rngMax))
  else:
    envChannel.set_float_parameter("gravity",args.env_gravity)
    
  if args.env_rngScale:
    if args.env_individualScale:
      envChannel.set_uniform_sampler_parameters("scale_x",min_value = args.env_minScale, max_value=args.env_maxScale,seed=random.randint(rngMin,rngMax))
      envChannel.set_uniform_sampler_parameters("scale_y",min_value = args.env_minScale, max_value=args.env_maxScale,seed=random.randint(rngMin,rngMax))
      envChannel.set_uniform_sampler_parameters("scale_z",min_value = args.env_minScale, max_value=args.env_maxScale,seed=random.randint(rngMin,rngMax))
    else:
      envChannel.set_uniform_sampler_parameters("scale",min_value = args.env_minScale, max_value=args.env_maxScale,seed=random.randint(rngMin,rngMax))
  else:
    if args.env_individualScale:
      envChannel.set_float_parameter("scale_x",args.env_scale_x)
      envChannel.set_float_parameter("scale_y",args.env_scale_y)
      envChannel.set_float_parameter("scale_z",args.env_scale_z)
    else:
      envChannel.set_float_parameter("scale",args.env_scale)

      
  # this reset is necessary otherwise there aren't any behavior specs present
  envUnity.reset()
  
  bName = list(envUnity.behavior_specs)[0]

  # we only use on behavior => always first one is our desired behavior
  behavior_specs = list(envUnity.behavior_specs.values())[0]
  
  obsDim = behavior_specs.observation_specs[0].shape[0]
  actDim = behavior_specs.action_spec.continuous_size
  
  activeEnvs, termEnvs = envUnity.get_steps(bName)  
  simCount = len(activeEnvs) # in 3DBall 12 Boxes => 12 simulations

  return envUnity, obsDim, actDim, simCount

"""
create a unity domain with the gym wrapper
"""
def createUnityGymEnv(args):
  envUnity,_,_,_ = createUnityEnv(args)
  env = UnityToGymWrapper(envUnity)
  return env, env.observation_space.shape[0], env.action_space.shape[0], 1

"""
create an env from args
"""
def buildFromArgs(args):
  if args.env == "gym":
    env, obsDim, actDim, simCount = createGymEnv()
  elif args.env == "unity":
    env, obsDim, actDim, simCount = createUnityEnv(args)
  elif args.env == "unity-gym":
    env, obsDim, actDim, simCount = createUnityGymEnv(args)
  else:
    print("unknown env. Falling back to gym env")
    env, obsDim,actDim = createGymEnv()
  return env, obsDim, actDim, simCount
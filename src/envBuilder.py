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
    return env, env.observation_space.shape[0], env.action_space.shape[0], 1

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
def createUnityGymEnv(name='3DEllipsoid1-Bouncy',no_graphics=True,time_scale=20.,rngMass=False,rngGravity=False,rngScale=False,individualScale=False):
  envUnity,_,_ = createUnityEnv(name,no_graphics,time_scale,rngMass,rngGravity,rngScale,individualScale)
  env = UnityToGymWrapper(envUnity)
  return env, env.observation_space.shape[0], env.action_space.shape[0], 1

"""
create an env from args
"""
def buildFromArgs(args):
  name = args.env_name
  if args.env == "gym":
    env, obsDim, actDim, simCount = createGymEnv()
  elif args.env == "unity":
    noGraphics = not args.replay
    env, obsDim, actDim, simCount = createUnityEnv(name=name,no_graphics=noGraphics)
  elif args.env == "unity-gym":
    noGraphics = not args.replay
    env, obsDim, actDim, simCount = createUnityGymEnv(name=name,no_graphics=noGraphics)
  else:
    print("unknown env. Falling back to gym env")
    env, obsDim,actDim = createGymEnv()
  return env, obsDim, actDim, simCount
  

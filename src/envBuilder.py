from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import gym


"""
create a gym domain
"""
def createGymEnv(name='MountainCarContinuous-v0'):
    env = gym.make(name)    
    return env
"""
create a unreal domain
"""
def createUnrealEnv(name='3DBall'):
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale = 2.0)
    env = UnityEnvironment(file_name="3DBall", seed=1, side_channels=[channel])
  #  channel = EnvironmentParametersChannel()
  #  channel.set_float_parameter("parameter_1", 2.0)
    return env

from ast import arg
import pickle
from src.PPO import PPO
from codecarbon import OfflineEmissionsTracker
from src.trainingDemo import replayDemo,trainingDemo
from src.envBuilder import buildFromArgs, createUnityEnv
from src.training import PPOTraining
import ray
from ray.tune.registry import register_env
import os
import argparse

def parseArguments():
    parser = argparse.ArgumentParser("L-KI")
    parser.add_argument("-d",   "--demo",           action="store_true",            help="use the demo ppo")
    parser.add_argument("-r",   "--replay",         action="store_true",            help="enable replay mode")
    parser.add_argument("-env", "--env",            default="gym",  type=str,       help="set the enviroment (gym,unity)")
    
    parser.add_argument("-m", "--model",            type=str, required=True, help="name of the model")
    
    # parameter similar to stable_baselines3
    parser.add_argument("-e", "--episodes",         default=10000,  type=int,           help="training episode number")
    parser.add_argument("-us", "--u_step",          default=4000,   type=int,           help="number of steps until update (n_steps/update_timestep)")
    parser.add_argument("-g", "--gamma",            default=0.99,   type=float,         help="discount factor")
    
    # our parameters
    parser.add_argument("-lr_a", "--lr_actor",          default=3e-04,  type=float,         help="learn rate of the actor")
    parser.add_argument("-lr_c", "--lr_critic",         default=3e-04,  type=float,         help="learn rate of the critic")
    parser.add_argument("-ke", "--k_epochs",            default=80,   type=int,         help="")
    parser.add_argument("-e_clip", "--epsilon_clip",    default=0.2,   type=float,         help="eps_clip")
    parser.add_argument("-a_std", "--action_std",       default=0.6,   type=float,         help="")
    
    return parser.parse_args()
    

#tracker = OfflineEmissionsTracker(output_dir="./out/", country_iso_code="DEU") # project_name="L-KI"
#tracker.start()

args = parseArguments()

modelName = args.model

if args.demo:
    if args.replay:
        env = buildFromArgs(args)
        replayDemo(env,modelName,True)
    else:
        env = buildFromArgs(args)
        trainingDemo(env,modelName,args.episodes)     
else:
    if args.replay:
        print("Replay mode for non-demo not implemented yet")
    else:
    
        
        #Trainng mode
        
    
        
            # create directory and file to save checkpoint to
        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        checkpoint_path = os.path.join(directory, 'net_{}_{}'.format('logs', 0))
        
        #env = buildFromArgs(args)
        register_env("unity_3DBall", lambda args: createUnityEnv(args))
     
        
        state_dim = 8#env.observation_space[0].shape[0]
        
        action_dim = 2# env.action_space.shape[0]
       
        # create PPO driven agent with hyperparameters
        agent = PPO(state_dim, 
            action_dim, 
            args.lr_actor, 
            args.lr_critic, 
            args.gamma, 
            args.k_epochs, 
            args.epsilon_clip, 
            args.action_std
            )
      
        
        config = {
            "nr_episodes": 100,
            "render": True,
            "update_timestep": 64,
            "K_epochs": 10,
            "eps_clip": 0.2,
            "gamma": 0.99,
            "lr_actor": 0.0003,
            "lr_critic": 0.001,
            "action_std_decay_rate": 0.05,
            "min_action_std": 0.1,
            "action_std_decay_freq": int(2.5e5),
            "save_model_freq": int(1e1)
        }

        ray.init()
        ppoTraining = PPOTraining.remote(env = "unity_3DBall", checkpoint_path=checkpoint_path, agent = agent)
        
        results = ray.get(ppoTraining.startTraining.remote(config))
        print("results=",results)

        if ray.is_initialized():
            ray.shutdown()
            
#tracker.stop()
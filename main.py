from ast import arg
from codecarbon import OfflineEmissionsTracker
from src.trainingDemo import replayDemo,trainingDemo
from src.envBuilder import buildFromArgs
from src.training import startTraining

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

env, obsDim,actDim = buildFromArgs(args)

if args.demo:
    if args.replay:
        replayDemo(env,modelName,True)
    else:
        trainingDemo(env,modelName,args.episodes)     
else:
    if args.replay:
        print("Replay mode for non-demo not implemented yet")
    else:
        #Trainng mode
        startTraining(args,env, obsDim,actDim)
    
#tracker.stop()
from codecarbon import OfflineEmissionsTracker
from src.trainingDemo import replayDemo,trainingDemo
from src.envBuilder import buildFromArgs
from src.training import startTraining

import argparse

def parseArguments():
    parser = argparse.ArgumentParser("L-KI")
    parser.add_argument("-d",   "--demo",           action="store_true",            help="use the demo ppo")
    parser.add_argument("-r",   "--replay",         action="store_true",            help="enable replay mode")
    parser.add_argument("-cpu",   "--force_cpu",    action="store_true",            help="forces to use the cpu")
    parser.add_argument("-env", "--env",            default="unity",  type=str,     help="set the enviroment (gym,unity)")
    parser.add_argument("-env_n", "--env_name",     default="3DBall1",  type=str,     help="name the domain name")
    
    parser.add_argument("-tag", "--tag",            type=str, required=True, help="name/tag of the run")
    
    #hyperparameter
    parser.add_argument("-e", "--episodes",                         default=100000,  type=int,           help="training episode number")
    parser.add_argument("-us", "--update_timestep",                 default=1000,   type=int,           help="number of steps until update (n_steps/update_timestep) e.g. max_ep_len * 4")
    parser.add_argument("-g", "--gamma",                            default=0.99,   type=float,         help="discount factor, probably 0.99 at its best")
    parser.add_argument("-lr_a", "--lr_actor",                      default=1e-03,  type=float,         help="learn rate of the actor")
    parser.add_argument("-lr_c", "--lr_critic",                     default=2e-03,  type=float,         help="learn rate of the critic")
    parser.add_argument("-ke", "--k_epochs",                        default=15,     type=int,           help="should probably be between [3, 30]")
    parser.add_argument("-e_clip", "--epsilon_clip",                default=0.3,    type=float,         help="should probably be between [0.1, 0.3]")
    parser.add_argument("-a_std", "--action_std",                   default=0.8,    type=float,         help="")
    parser.add_argument("-a_std_rate", "--action_std_decay_rate",   default=5e-4,   type=float,         help="action standard deviation decay rate")
    parser.add_argument("-a_std_freq", "--action_std_decay_freq",   default=1e3,    type=int,           help="action standard deviation decay frequency")
    parser.add_argument("-a_std_min", "--min_action_std",           default=1e-3,   type=float,         help="minimum action standard deviation")
    
    return parser.parse_args()
    

tracker = OfflineEmissionsTracker(output_dir="./out/", country_iso_code="DEU") # project_name="L-KI"
tracker.start()

args = parseArguments()

modelName = args.tag

env, obsDim, actDim, simCount = buildFromArgs(args)

if args.demo:
    if args.replay:
        replayDemo(env,modelName,True)
    else:
        trainingDemo(env,modelName,args.episodes)     
else:
    if args.replay:
        print("Replay mode for non-demo not implemented yet")
    else:
        #Training mode
        startTraining(args, env, obsDim, actDim, simCount)
    
tracker.stop()
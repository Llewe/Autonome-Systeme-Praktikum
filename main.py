from pickle import TRUE
from xmlrpc.client import boolean
from codecarbon import OfflineEmissionsTracker
from src.envBuilder import buildFromArgs
from src.training import startTraining
from src.trainingBaseline import trainBaselinePPO
from src.evaluation import startEval
from datetime import datetime

import platform
import argparse

def parseArguments():
    parser = argparse.ArgumentParser("L-KI")
    parser.add_argument("-d",   "--demo",           action="store_true",            help="use the demo ppo")
    parser.add_argument("-eval",   "--evaluation",  action="store_true",            help="run evaluation of trained model")
    parser.add_argument("-cpu",   "--force_cpu",    action="store_true",            help="forces to use the cpu")
    parser.add_argument("-agent",   "--agent",      default="ppo",  type=str,       help="set the agent type here. (ppo,random_agent,ppo-baseline)")
    parser.add_argument("-tag", "--tag",            type=str, required=True,        help="name/tag of the run")
    
    #hyperparameter
    parser.add_argument("-e", "--episodes",                         default=1000,   type=int,           help="episode number")
    parser.add_argument("-ts", "--max_timesteps",                   default=1200000,type=int,           help="number of timesteps in training")
    parser.add_argument("-us", "--update_timestep",                 default=1000,   type=int,           help="number of steps until update (n_steps/update_timestep) e.g. max_ep_len * 4")
    parser.add_argument("-g", "--gamma",                            default=0.99,   type=float,         help="discount factor, probably 0.99 at its best")
    parser.add_argument("-lr_a", "--lr_actor",                      default=1e-04,  type=float,         help="learn rate of the actor")
    parser.add_argument("-lr_c", "--lr_critic",                     default=1e-04,  type=float,         help="learn rate of the critic")
    parser.add_argument("-ke", "--k_epochs",                        default=15,     type=int,           help="should probably be between [3, 30]")
    parser.add_argument("-e_clip", "--epsilon_clip",                default=0.3,    type=float,         help="should probably be between [0.1, 0.3]")
    parser.add_argument("-a_std", "--action_std",                   default=0.5,    type=float,         help="")
    parser.add_argument("-a_std_rate", "--action_std_decay_rate",   default=5e-4,   type=float,         help="action standard deviation decay rate")
    parser.add_argument("-a_std_freq", "--action_std_decay_freq",   default=1000,   type=int,           help="action standard deviation decay frequency")
    parser.add_argument("-a_std_min", "--min_action_std",           default=1e-2,   type=float,         help="minimum action standard deviation")
    
    #env parameter
    parser.add_argument("-env", "--env",            default="unity",  type=str,     help="set the enviroment (gym,unity)")
    parser.add_argument("-env_n", "--env_name",     default="3DBall1",  type=str,   help="name the domain name")
    
    # Recommended parameter bounds: https://unity-technologies.github.io/ml-agents/Learning-Environment-Examples/#3dball-3d-balance-ball
    parser.add_argument("--env_rngMass",        action="store_true",        help="enable random mass")
    parser.add_argument("--env_mass",           default=1., type=float,      help="set mass if random is disabled")
    parser.add_argument("--env_minMass",        default=0.1, type=float,    help="min mass if random is enabled")
    parser.add_argument("--env_maxMass",        default=20., type=float,    help="max mass if random is enabled")
    
    parser.add_argument("--env_rngGravity",     action="store_true",        help="enable random gravity")
    parser.add_argument("--env_gravity",        default=9.81, type=float,   help="set gravity if random is disabled")
    parser.add_argument("--env_minGravity",     default=4., type=float,     help="min gravity if random is enabled")
    parser.add_argument("--env_maxGravity",     default=105., type=float,   help="max gravity if random is enabled")

    parser.add_argument("--env_rngScale",       action="store_true",        help="enable random ball scale")
    parser.add_argument("--env_individualScale",action="store_true",        help="enable individual ball scale for x y z axis")
    parser.add_argument("--env_scale",          default=1., type=float,      help="set scale if random and individual scale is disabled")
    parser.add_argument("--env_scale_x",        default=1., type=float,      help="set scale for x axis if random scale is disabled and individual scale is enabled")
    parser.add_argument("--env_scale_y",        default=1., type=float,      help="set scale for y axis if random scale is disabled and individual scale is enabled")
    parser.add_argument("--env_scale_z",        default=1., type=float,      help="set scale for z axis if random scale is disabled and individual scale is enabled")
    parser.add_argument("--env_scale_max_dev",  default=0., type=float,      help="set the maximum difference betwen x y z scales. If 0 no maximum is set")
    parser.add_argument("--env_minScale",       default=0.2,type=float,     help="min scale if random is enabled (works also combined with individual scale)")
    parser.add_argument("--env_maxScale",       default=5., type=float,     help="max scale if random is enabled (works also combined with individual scale)")
    parser.add_argument("--env_color_r",       default=0.85,type=float,     help="set ball color (Red) value must be between [0-1]")
    parser.add_argument("--env_color_g",       default=0.33,type=float,     help="set Ball color (Green) value must be between [0-1]")
    parser.add_argument("--env_color_b",       default=0.93,type=float,     help="set ball color (Blue) value must be between [0-1]")
    parser.add_argument("--env_color_a",       default=1.,  type=float,     help="set ball color (Alpha) value must be between [0-1]")
    parser.add_argument("--env_bounciness",     default=1., type=float,      help="set bounciness of the ball [0-1]")
    parser.add_argument("--env_dynamicFriction",default=0., type=float,      help="set dynamic friction of the ball [0-1] (more information: https://docs.unity3d.com/ScriptReference/PhysicMaterial.html)")
    parser.add_argument("--env_staticFriction", default=0., type=float,      help="set static friction of the ball [0-1] (more information: https://docs.unity3d.com/ScriptReference/PhysicMaterial.html)")
    
    parser.add_argument("--env_rngBounciness",  action="store_true",        help="enable random bounciness")
    parser.add_argument("--env_minBounce",       default=0.1,type=float,     help="min bounciness if random is enabled")
    parser.add_argument("--env_maxBounce",       default=1., type=float,     help="max bounciness if random is enabled")
    
 
    parser.add_argument("--env_video",          action="store_true",        help="enable graphical output")
    parser.add_argument("--env_timeScale",      default=20., type=float,    help="time speedup")
 
    return parser.parse_args()

tracker = OfflineEmissionsTracker(output_dir="./out/", country_iso_code="DEU") # project_name="L-KI"
tracker.start()

args = parseArguments()

modelName = args.tag

env, obsDim, actDim, simCount = buildFromArgs(args)

# create output folder path
osName = platform.node()
currentTimeInSec = int(round(datetime.now().timestamp()))

output_dir = "generated"
folderPath = f"/{args.env}/{args.env_name}/{args.tag}/{osName}-{currentTimeInSec}"

if args.demo:
        trainBaselinePPO(args, env, output_dir, folderPath)
else:
    if args.evaluation:     
        startEval(args, env, obsDim, actDim, simCount, output_dir, folderPath)
    else:
        #Training mode
        startTraining(args, env, obsDim, actDim, simCount, output_dir, folderPath)
    
tracker.stop()
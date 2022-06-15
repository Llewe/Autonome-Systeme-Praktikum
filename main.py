from codecarbon import OfflineEmissionsTracker
from src.trainingDemo import replayDemo,trainingDemo
import src.envBuilder as envBuilder

import argparse

tracker = OfflineEmissionsTracker(output_dir="./out/", country_iso_code="DEU") # project_name="L-KI"
tracker.start()

# use argparse https://stackoverflow.com/a/42929351/11723520
parser = argparse.ArgumentParser("L-KI")
parser.add_argument("-m", "--model", required=True, help="name of the model", type=str)
parser.add_argument("-d", "--demo",  help="use the demo ppo", action="store_true")
parser.add_argument("-r", "--replay", help="enable replay mode", action="store_true")

args = parser.parse_args()

modelName = args.model

if args.demo:
    if args.replay:
        env = envBuilder.createUnityEnv(no_graphics=False)
        replayDemo(env,modelName)
    else:
        env = envBuilder.createUnityEnv()
        trainingDemo(env,modelName,1000000)     

tracker.stop()
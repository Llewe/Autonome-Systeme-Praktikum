from src.envBuilder import createGymEnv
from src.training import training

from codecarbon import OfflineEmissionsTracker
import argparse

tracker = OfflineEmissionsTracker(output_dir="./out/", country_iso_code="DEU")
tracker.start()
# training

# use argparse https://stackoverflow.com/a/42929351/11723520
parser = argparse.ArgumentParser("simple_example")
parser.add_argument("-t", "--test", required=True, help="test parameter", type=int)

args = parser.parse_args()

tracker.stop()
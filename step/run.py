import os
import sys
from argparse import ArgumentParser

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../.."))
from basicts import launch_training


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    # parser.add_argument("-c", "--cfg", default="step/TSFormer_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="step/STEP_METR-LA.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="step/TSFormer_PEMS04.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="step/STEP_PEMS04.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="step/TSFormer_PEMS-BAY.py", help="training config")
    parser.add_argument("-c", "--cfg", default="step/STEP_PEMS-BAY.py", help="training config")
    parser.add_argument("--gpus", default="0", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    launch_training(args.cfg, args.gpus)

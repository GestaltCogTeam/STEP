import os
import sys
from argparse import ArgumentParser

from easytorch import launch_training


def parse_args():
    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-c', '--cfg', help='training config', required=True)
    parser.add_argument('--node-rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--gpus', help='visible gpus', type=str)
    parser.add_argument('--tf32', help='enable tf32 on Ampere device', action='store_true')
    return parser.parse_args()


def main():
    # work dir
    path = os.getcwd()
    sys.path.append(path)

    # parse arguments
    args = parse_args()

    # train
    launch_training(args.cfg, args.gpus, args.tf32, args.node_rank)


if __name__ == '__main__':
    main()

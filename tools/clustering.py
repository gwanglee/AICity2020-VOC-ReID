import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from lib.config import cfg
from lib.data import make_data_loader
from lib.modeling import build_model

def main():
    device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/debug.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

## build network
    # model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
    #                  cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
    #                  cfg)

    num_classes = 6*5   # num cameras = 6 * 5 bg for camera
    model = build_model(cfg, num_classes)


if __name__ == "__main__":
    main()
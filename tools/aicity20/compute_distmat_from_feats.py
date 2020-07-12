import torch
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--src_dir", default="./output/aicity20/0410-test/r50-320-circle", help="path to config file", type=str
    )
    parser.add_argument(
        "--src_npy", default=None, help="path to config file", type=str
    )

    args = parser.parse_args()
    src_dir = args.src_dir
    src_npy = args.src_npy

    if src_npy is not None:
        input_npy_path = src_npy
    else:
        input_npy_path = os.path.join(src_dir, 'feats.npy')

    print('load npy: ' + input_npy_path)
    feat = np.load(input_npy_path)

    feat = torch.tensor(feat, device='cpu')
    all_num = len(feat)
    distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
              torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    distmat.addmm_(1, -2, feat, feat.t())   # this is euclidean distance!
    distmat = distmat.cpu().numpy()

    np.save(input_npy_path[:-4] + '_dist.npy', distmat)


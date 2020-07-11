import torch
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--src_dir", default="./output/aicity20/0410-test/r50-320-circle", help="path to config file", type=str
    )
    args = parser.parse_args()
    src_dir = args.src_dir

    feat = np.load(os.path.join(src_dir, 'feats.npy'))

    feat = torch.tensor(feat, device='cpu')
    all_num = len(feat)
    distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
              torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    distmat.addmm_(1, -2, feat, feat.t())   # this is euclidean distance!
    distmat = distmat.cpu().numpy()

    np.save(src_dir + '/' + 'feat_distmat', distmat)


    # feat = np.load(os.path.join(src_dir, 'feats_norm.npy'))
    #
    # feat = torch.tensor(feat, device='cpu')
    # all_num = len(feat)
    # distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
    #           torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    # distmat.addmm_(1, -2, feat, feat.t())
    # distmat = distmat.cpu().numpy()
    #
    # np.save(src_dir + '/' + 'feat_distmat', distmat)
    # np.save(src_dir + '/' + 'feat_distmat_2', distmat)

    # ####
    #
    # feat = np.load(os.path.join(src_dir, 'feats_ori.npy'))
    #
    # feat = torch.tensor(feat, device='cpu')
    # all_num = len(feat)
    # distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
    #           torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    # distmat.addmm_(1, -2, feat, feat.t())
    # distmat = distmat.cpu().numpy()
    #
    # np.save(src_dir + '/' + 'feat_distmat_ori', distmat)
    # np.save(src_dir + '/' + 'feat_distmat_ori_2', distmat)


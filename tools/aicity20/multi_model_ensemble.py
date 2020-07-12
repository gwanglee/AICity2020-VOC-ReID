import numpy as np
import sys
sys.path.append('.')
from lib.data.datasets.aicity20 import AICity20
from tools.aicity20.submit import write_result_with_track
from tools.aicity20.submit import write_result


if __name__ == '__main__':
    # dataset = AICity20('/home/zxy/data/ReID/vehicle')

    distmat_path = ['./output/visda/trained/resnet50_warmup_multi_step_cm0.3_cs100_1_rerb_xbm8/distmat.npy',
                   './output/visda/trained/resnet101_warmup_multi_step_cm0.3_cs100_1_rerb_xbm8/distmat.npy',
                   './output/visda/trained/resnet50_warmup_multi_step_cm0.3_cs100_2/distmat.npy',
                   './output/visda/trained/resnext101_warmup_multi_step_cm0.3_cs100_1_rerb_xbm8/distmat.npy',
                   ]
    # distmat_path = ['./output/visda/base-ensemble/dist_mat_1.npy',
    #                 './output/visda/base-ensemble/dist_mat_2.npy',
    #

                    # './output/visda/base-ensemble-0704/r50-E40/distmat.npy',
        #             './output/visda/base-ensemble-0704/r101-E40/distmat.npy',
        #             './output/visda/base-ensemble-0704/rx101-E40/distmat.npy',
    #             ]

    #cam_distmat = np.load('./output/aicity20/0407-ReCamID/distmat_submit.npy')
    #ori_distmat = np.load('./output/aicity20/0409-ensemble/ReTypeID/distmat_submit.npy')
    distmat = []
    for path in distmat_path:
        distmat.append(np.load(path))
    distmat = sum(distmat) / len(distmat)
    #distmat = distmat - 0.1 * cam_distmat - 0.1 * ori_distmat

    indices = np.argsort(distmat, axis=1)
    # write_result_with_track(indices, './output/aicity20/submit/', dataset.test_tracks)
    write_result(indices, './output/visda/trained')

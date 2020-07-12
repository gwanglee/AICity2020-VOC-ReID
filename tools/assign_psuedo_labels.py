import numpy as np
import cv2
import os

DEBUG = True
USE_RERANK = True

if __name__ == '__main__':
    if USE_RERANK:
        npy_path = '/Users/gglee/Downloads/dist_after_rerank.npy'
    else:
        npy_path = '/Users/gglee/Downloads/feat_distmat.npy'

    # 1. how did I rerank?
    # 2. use clustering here also

    distmat = np.load(npy_path)

    print(distmat.shape)
    print(np.amax(distmat), np.amin(distmat), np.mean(distmat))

    in_th = np.argwhere(distmat < np.mean(distmat)/2.0)
    print('num elm', len(in_th))

    # assert distmat[100][50] == distmat[50][100]
    # assert distmat[22][4412] == distmat[4412][22]

    if USE_RERANK:
        val_pair = sorted([(distmat[i[0], i[1]], i[0], i[1]) for i in in_th])
    else:
        val_pair = sorted([(distmat[i[0], i[1]], i[0], i[1]) for i in in_th if i[0] < i[1]])

    if DEBUG:
        # how much of distnace is different???
        img_path = '/Users/gglee/Documents/VisDA/data/challenge_datasets/target_training/image_train'

        for i in range(100):
            dist, x, y = val_pair[i]
            # dist, x, y = val_pair[int(i*len(val_pair)/100)]
            img1 = cv2.imread(os.path.join(img_path, '{:05d}.jpg'.format(x)))

            if USE_RERANK:
                img2 = cv2.imread(os.path.join(img_path, '{:05d}.jpg'.format(y+distmat.shape[0])))
            else:
                img2 = cv2.imread(os.path.join(img_path, '{:05d}.jpg'.format(y)))

            img  = cv2.hconcat([img1, img2])
            cv2.line(img, (64, 0), (64, 128), (255, 255, 255))

            PX, PY = 10, 100
            str_dist = '{:.4f}'.format(dist)
            cv2.putText(img, str_dist, (PX-1, PY), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))
            cv2.putText(img, str_dist, (PX+1, PY), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))
            cv2.putText(img, str_dist, (PX, PY-1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))
            cv2.putText(img, str_dist, (PX, PY+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))
            cv2.putText(img, str_dist, (PX, PY), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))

            cv2.imshow('matching pair', img)
            cv2.waitKey(-1)

    print(len(val_pair))
    print(val_pair[:100])

    # now cluster close elements together

    clusters = dict()
    for v in val_pair:
        i, j = v[1], v[2]
        if i in clusters:
            clusters[i].append(j)
        elif j in clusters:
            clusters[j].append(i)
        else:
            clusters[i] = [j]

    print(len(clusters))
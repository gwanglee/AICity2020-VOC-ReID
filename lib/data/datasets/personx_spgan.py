from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import os
import warnings

# from ..dataset import ImageDataset
from .bases import BaseImageDataset

# https://kaiyangzhou.github.io/deep-person-reid/user_guide.html#use-your-own-dataset

class PersonX_Spgan(BaseImageDataset):
    """Person X.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'personx_spgan'
    dataset_url = 'https://drive.google.com/open?id=18qIbI1XiG2n36qCTS-Te-2XATxiHNVDj'

    def __init__(self, root='', verbose=True, **kwargs):
        super(PersonX_Spgan, self).__init__()

        train_path = os.path.join(root, 'personX_spgan/image_train')
        query_path = os.path.join(root, 'target_validation/image_query')
        gallery_path = os.path.join(root, 'target_validation/image_gallery')

        train = self.process_dir(train_path)
        query = self.process_dir(query_path, './lib/data/datasets/index_validation_query.txt')
        gallery = self.process_dir(gallery_path, './lib/data/datasets/index_validation_gallery.txt')

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        from random import random
        for i, q in enumerate(self.query):
            if random > 0.95:
                print(i, q)

        for i, g in enumerate(self.gallery):
            if random > 0.95:
                print(i, g)

        if verbose:
            print("=> PersonX-SPGAN loaded")


    def process_dir(self, dir_path, list_path=None):

        data = []
        if list_path is None:
            for img in os.listdir(dir_path):
                if not img.startswith('.') and img.endswith('.jpg'):
                    splitted  = img.split('_')
                    pid, cid = int(splitted[0]), int(splitted[1][1])

                    data.append((os.path.join(dir_path, img), pid, cid))
        else:
            with open(list_path, 'r') as rf:
                lines = rf.readlines()
                for line in lines:
                    words = line.strip().split()
                    data.append((os.path.join(dir_path, words[0]), int(words[2]), int(words[1])))


        return data
        #
        #
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        #
        # pid_container = set()
        # for img_path in img_paths:
        #     pid, _ = map(int, pattern.search(img_path).groups())
        #     if pid == -1:
        #         continue # junk images are just ignored
        #     pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}
        #
        # data = []
        # for img_path in img_paths:
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     if pid == -1:
        #         continue # junk images are just ignored
        #     assert 0 <= pid <= 1501 # pid == 0 means background
        #     assert 1 <= camid <= 6
        #     camid -= 1 # index starts from 0
        #     if relabel:
        #         pid = pid2label[pid]
        #     data.append((img_path, pid, camid))
        #
        # return data

from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import os
import warnings

from ..dataset import ImageDataset

# https://kaiyangzhou.github.io/deep-person-reid/user_guide.html#use-your-own-dataset

class PersonX(ImageDataset):
    """Person X.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'personx'
    dataset_url = 'https://drive.google.com/open?id=18qIbI1XiG2n36qCTS-Te-2XATxiHNVDj'

    def __init__(self, root='',  **kwargs):

        train_path = os.path.join(root, 'personX/image_train')
        query_path = os.path.join(root, 'target_validation/image_query')
        gallery_path = os.path.join(root, 'target_validation/image_gallery')

        train = self.process_dir(train_path)
        query = self.process_dir(query_path)
        gallery = self.process_dir(gallery_path)

        # self.root = osp.abspath(osp.expanduser(root))
        # self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        #
        # # allow alternative directory structure
        # self.data_dir = self.dataset_dir
        # data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        # if osp.isdir(data_dir):
        #     self.data_dir = data_dir
        # else:
        #     warnings.warn(
        #         'The current data structure is deprecated. Please '
        #         'put data folders such as "bounding_box_train" under '
        #         '"Market-1501-v15.09.15".'
        #     )
        #
        # self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.query_dir = osp.join(self.data_dir, 'query')
        # self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        # self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        # self.market1501_500k = market1501_500k
        #
        # required_files = [
        #     self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        # ]
        # if self.market1501_500k:
        #     required_files.append(self.extra_gallery_dir)
        # self.check_before_run(required_files)
        #
        # train = self.process_dir(self.train_dir, relabel=True)
        # query = self.process_dir(self.query_dir, relabel=False)
        # gallery = self.process_dir(self.gallery_dir, relabel=False)
        # if self.market1501_500k:
        #     gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(PersonX, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):

        data = []
        for img in os.listdir(dir_path):
            if not img.startswith('.') and img.endswith('.jpg'):
                splitted  = img.split('_')
                pid, cid = int(splitted[0])-1, int(splitted[1][1])

                data.append((os.path.join(dir_path, img), pid, cid))

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

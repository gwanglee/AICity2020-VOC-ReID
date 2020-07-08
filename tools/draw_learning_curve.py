import os
import re
from matplotlib import pyplot as plt

log_dir = './output/visda/trained'

# subs = os.listdir(log_dir)
#
# for s in subs:
#     cur = os.path.join(log_dir, s)
#     if os.path.isdir(cur):
#         with open(os.path.join(cur, 'log.txt'), 'r'), as rf:
#             logs = rf.readlines()
#             for l in logs:
#                 if ' Loss: ' in l:
#                     words = l.strip().split(' ')


lines = ['2020-07-08 05:44:08,092 reid_baseline.train INFO: Epoch[11] Iteration[100/316] Loss: 7.967, data time: 0.005s, model time: 0.322s',
         '2020-07-08 05:44:42,417 reid_baseline.train INFO: Epoch[11] Iteration[200/316] Loss: 7.472, data time: 0.006s, model time: 0.319s',
         '2020-07-08 05:45:16,956 reid_baseline.train INFO: Epoch[11] Iteration[300/316] Loss: 6.944, data time: 0.006s, model time: 0.323s',
         '2020-07-08 05:45:22,040 reid_baseline.train INFO: epoch takes 111.746s',
         '2020-07-08 05:45:22,664 reid_baseline.train INFO: Epoch[12] lr=3.50e-04',
         '/pytorch/torch/csrc/utils/tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.',
         '/pytorch/torch/csrc/utils/tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.',
         '/pytorch/torch/csrc/utils/tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.',
         '/pytorch/torch/csrc/utils/tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.',
         '/pytorch/torch/csrc/utils/tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.',
         '/pytorch/torch/csrc/utils/tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.',
         '/pytorch/torch/csrc/utils/tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.',
         '/pytorch/torch/csrc/utils/tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.',
         '2020-07-08 05:46:00,587 reid_baseline.train INFO: Epoch[12] Iteration[100/316] Loss: 7.174, data time: 0.005s, model time: 0.331s',
         '2020-07-08 05:46:34,847 reid_baseline.train INFO: Epoch[12] Iteration[200/316] Loss: 7.250, data time: 0.006s, model time: 0.330s',
         '2020-07-08 05:46:00,587 reid_baseline.train INFO: Epoch[12] Iteration[100/316] Loss: 7.174, data time: 0.005s, model time: 0.331s',
         '2020-07-08 05:47:09,230 reid_baseline.train INFO: Epoch[12] Iteration[300/316] Loss: 5.924, data time: 0.005s, model time: 0.323s',
         '2020-07-08 05:46:00,587 reid_baseline.train INFO: Epoch[12] Iteration[100/316] Loss: 7.174, data time: 0.005s, model time: 0.331s',
         '2020-07-08 05:47:14,270 reid_baseline.train INFO: epoch takes 111.606s',]


def extract(logs):
    res_loss = []
    res_map = []

    lr = 0.0
    for l in logs:
        l = l.strip()
        regex_epoch = re.compile("Epoch\[\d+\]")
        epochs = regex_epoch.search(l)
        if epochs:
            e = int(re.compile('\d+').search(epochs.group(0)).group(0))
            if 'lr=' in l:
                lr = float(l.split('=')[-1])
            elif 'Loss:' in l:
                loss = float(l.split(' ')[7][:-1])
                res_loss.append({'epoch': e, 'lr': lr, 'loss': loss})
            elif 'mAP' in l:
                mAP = float(l.split(' ')[5][1:-1])
                res_map.append({'epoch': e, 'mAP': loss})

    return res_loss, res_map


data = extract(lines)

def draw(data_loss, data_map):
    epochs = [d['epoch'] for d in data_loss]
    losses = [d['loss'] for d in data_loss]
    lr = [d['lr'] for d in data_loss]

    epochs2 = [d['epoch'] for d in data_map]
    mAP = [d['mAP'] for d in data_map]

    plt.figure()
    plt.plot(losses, 'o-')
    plt.xticks([i for i in range(len(epochs))], epochs)
    plt.title('loss')

    plt.figure()
    plt.plot(lr, 'x-')
    plt.xticks([i for i in range(len(epochs))], epochs)
    plt.title('lr')

    plt.show()

    plt.figure()
    plt.plot(lr, '*-')
    plt.xticks([i for i in range(len(epochs2))], epochs2)
    plt.title('mAP')

    plt.show()

draw(data)

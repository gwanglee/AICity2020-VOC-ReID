import os
import re
import numpy as np
from matplotlib import pyplot as plt
import sys

def extract_loss(logs):
    res_loss = []
    res_map = {}

    lr = 0.0
    cur_epoch = -1
    best = 0.0

    for l in logs:
        l = l.strip()
        if not l.startswith('2020'):
            continue

        regex_epoch = re.compile("Epoch\[\d+\]")
        epochs = regex_epoch.search(l)

        if epochs:
            e = int(re.compile('\d+').search(epochs.group(0)).group(0))
            if 'lr=' in l:
                lr = float(l.split('=')[-1])
                cur_epoch = e

        if 'Loss:' in l:
            loss = float(l.split(' ')[7][:-1])
            res_loss.append({'epoch': cur_epoch, 'lr': lr, 'loss': loss})
        if 'mAP' in l:
            if 'best' in l:     # last line
                best = float(l.split(':')[-1][:-1])

            else:
                mAP = float(l.split(' ')[5][:-1])
                if not cur_epoch in res_map:
                    res_map[cur_epoch] = {'mAP': mAP}

        if 'Rank-10' in l:
            res_map[cur_epoch]['Rank-10'] = float(l.split(':')[-1][:-1])
        elif 'Rank-5' in l:
            res_map[cur_epoch]['Rank-5'] = float(l.split(':')[-1][:-1])
        elif 'Rank-1' in l:
            res_map[cur_epoch]['Rank-1'] = float(l.split(':')[-1][:-1])

    return res_loss, res_map, best


def analyze(log_path):
    with open(log_path, 'r') as rf:
        lines = rf.readlines()

    configs = {
        'DATALOADER': dict(),
        'DATASETS' : dict(),
        'INPUT': dict(),
        'MODEL': dict(),
        'OUTPUT_DIR': dict(),
        'SOLVER': dict()
               }

    key = None
    for l in lines:
        line = l.strip()
        if line[:4] != '2020':
            if line.split(':')[0] in configs:
                cur_key  = line.split(':')[0]
                key = cur_key
                if cur_key == 'OUTPUT_DIR':
                    configs[key]['OUTPUT_DIR'] = line.split(':')[1]
            else:
                if key is not None:
                    body = line.split(':')[1]
                    body = re.sub('[\'\"]', '', body).strip()
                    configs[key][line.split(':')[0]] = body

    loss, map, best_map = extract_loss(lines)

    # for k in configs:
    #     print(k)
    #     print('\t', configs[k])
    #
    # for l in loss:
    #     print(l)
    #
    # for m in map:
    #     print(m, map[m])

    ap = [map[m]['mAP'] for m in map]
    r1 = [map[m]['Rank-1'] for m in map]
    r5 = [map[m]['Rank-5'] for m in map]
    r10 = [map[m]['Rank-10'] for m in map]

    # make text output
    title_row = []
    data_row = []

    title_row.append('Best mAP')
    data_row.append('{}'.format(best_map))

    title_row.append('mAP (avg_5)')
    data_row.append('{:.2f}'.format((sum([ap[-i] for i in range(1, 6)]))/5.0))
    title_row.append('Rank-1 (avg_5)')
    data_row.append('{:.2f}'.format((sum([r1[-i] for i in range(1, 6)]))/5.0))
    title_row.append('Rank-5 (avg_5)')
    data_row.append('{:.2f}'.format((sum([r5[-i] for i in range(1, 6)]))/5.0))
    title_row.append('Rank-10 (avg_5)')
    data_row.append('{:.2f}'.format((sum([r10[-i] for i in range(1, 6)]))/5.0))

    title_row.append('mAP')
    data_row.append('{}'.format(ap[-1]))
    title_row.append('Rank-1')
    data_row.append('{}'.format(r1[-1]))
    title_row.append('Rank-5')
    data_row.append('{}'.format(r5[-1]))
    title_row.append('Rank-10')
    data_row.append('{}'.format(r10[-1]))

    for cur in configs:
        for c in configs[cur]:
            title_row.append('{}.{}'.format(cur, c))
            data_row.append('{}'.format(configs[cur][c]))


    # print(title_row)
    # print(data_row)

    # print('best: ', best_map)

    # draw loss plot
    lr = [np.log(l['lr']) for l in loss]
    e = [l['epoch'] for l in loss]
    loss = [l['loss'] for l in loss]

    plt.plot(loss)
    plt.figure()
    plt.plot(lr)

    plt.figure()
    # draw accuracies

    plt.plot(ap)
    plt.plot(r1)
    plt.plot(r5)
    plt.plot(r10)
    # plt.show()

    return title_row, data_row


if __name__ == '__main__':
    args = sys.argv

    if len(args) >= 3:
        root_path = sys.argv[1]
        output_path = sys.argv[2]
    elif len(args) == 2:
        root_path = sys.argv[1]
        output_path = 'logs.txt'
    else:
        root_path = '/Users/gglee/Downloads/output/visda/base-ensemble'
        output_path = 'logs.txt'


    title_written = False

    title = None
    data = list()
    num_cols = 0

    with open('logs.txt', 'w') as wf:
        for p in os.listdir(root_path):
            if os.path.isdir(os.path.join(root_path, p)):
                if os.path.exists(os.path.join(root_path, p, 'log.txt')):
                    t, d = analyze(os.path.join(root_path, p, 'log.txt'))

                    if title is None:
                        title = t
                        num_cols = len(title)

                    assert len(d) == num_cols, 'different num cols'
                    data.append(d)

        to_remove = []
        for i in range(num_cols):
            cur_col_data = set()
            for d in data:
                cur_col_data.add(d[i])
            if len(cur_col_data) == 1:
                to_remove.append(i)


        def get_str(src, idx_to_remove):
            return '\t'.join([w for i, w in enumerate(src) if i not in idx_to_remove])

        wf.write(get_str(title, to_remove) + '\n')
        for d in data:
            wf.write(get_str(d, to_remove) + '\n')
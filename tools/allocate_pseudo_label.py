import os
import numpy as np
import networkx as nx
import metis

if __name__ == '__main__':
    npy_path = '/Users/gglee/Documents/VisDA/code/AICity2020-VOC-ReID/output/visda/base-ensemble-0704/r50-E40/distmat.npy'
    distmat = np.load(npy_path)
    print(distmat)
    print(distmat.shape)
    print(np.amin(distmat), np.amax(distmat))

    # does metis minimize graph weights?
    # assume that: it minimized weight cut, maximize weight in remaining graph

    # 1. remove edges below threshold
    #       - is it distance or similarity? -> seems like euclidean distance
    #       - distance is normalized??
    #       - Need to change distance to similarity -> take inverse? or 1-distance?

    simmat = 1.0 - distmat
    print(simmat)

    simmat[simmat < 0.5] = 0.0
    print(simmat)
    print(np.count_nonzero(simmat), np.count_nonzero(simmat)/377.0/377.0)

    simmat = np.asarray(distmat[:, :377]*10000, dtype=np.int)

    print(simmat.shape)

    G = nx.from_numpy_matrix(simmat)
    G.graph['edge_weight_attr']='weight'

    print(G)
    [cost, parts] = metis.part_graph(G, nparts=30, recursive=True)

    print(cost)
    print(parts)
    print(len(parts), min(parts), max(parts))

    print([(n, np.count_nonzero(parts == n)) for n in range(30)])

    counts = [0 for n in range(30)]
    for p in parts:
        counts[p] += 1
    print(counts)
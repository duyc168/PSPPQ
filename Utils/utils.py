import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


def CalcMapk(qB, rB, queryL, retrievalL, k, pre_smat):
    num_query = queryL.shape[0]
    # print("query num:{}".format(num_query))
    map = 0
    S = cosine_similarity(qB, rB) + pre_smat

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        ind = np.argsort(-S[iter])
        gnd = gnd[ind]
        gnd = gnd[0:k]
        tsum = np.sum(gnd)
        tsum = int(tsum)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map, S
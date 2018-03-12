import zerorpc
from indexPQ import *
import numpy as np


index = IVFPQ_multiGpu(ngpu=1)
index.build_index()


class SearchRPC(object):
    def search(self, query, nprobe):
        q = np.asarray(query, dtype=np.float32)
        #D, I = index.search(q.reshape(-1, q.shape[0]))
        D, I = index.search(q, nprobe)
        return D, I

s = zerorpc.Server(SearchRPC(), pool_size=2, heartbeat=None)
s.bind("tcp://0.0.0.0:8281")
s.run()

import zerorpc
from indexPQ import *
import numpy as np


index = IVFPQ_multiGpu(ngpu=3)
index.build_index()


class SearchRPC(object):
    def search(self, query):
        q = np.asarray(query, dtype=np.float32)
        #D, I = index.search(q.reshape(-1, q.shape[0]))
        D, I = index.search(q)
        return D, I

s = zerorpc.Server(SearchRPC())
s.bind("tcp://0.0.0.0:8282")
s.run()

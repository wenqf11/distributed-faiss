# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python2

import os
import time
import numpy as np
import pdb
from math import sqrt
import faiss

class IVFPQ_cpu():
    train_data_path="/root/dataset/sift1m/sift_learn.fvecs"
    base_data_path="/root/dataset/sift1m/sift_base.fvecs"
    m = 8                             # number of bytes per vector
    nlist = 4096
    
    def ivecs_read(fname):
        a = np.fromfile(fname, dtype='int32')
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy()

    def fvecs_read(fname):
    	return ivecs_read(fname).view('float32')

    def load_data(self):
        self.xt = fvecs_read(self.train_data_path)
        self.xb = fvecs_read(self.base_data_path)

    def build_index(self):
	    nt, d = self.xt.shape
	    quantizer = faiss.IndexFlatL2(d)  # this remains the same
	    index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, 8)
	    index.train(xt)
	    index.add(xb)
	    index.nprobe = self.nprobe 
	    print("finish building index")

    def search(self, query):
        t0 = time.time()
        D, I = index.search(query, 100)
        t1 = time.time()
        print("search uses  %.4f s" %(t1-t0))


class IVFPQ_gpu():
    train_data_path="/root/dataset/sift1m/sift_learn.fvecs"
    base_data_path="/root/dataset/sift1m/sift_base.fvecs"
    m = 8                             # number of bytes per vector
    nlist = 4096
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    co.usePrecomputed = False
    
    def ivecs_read(fname):
        a = np.fromfile(fname, dtype='int32')
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy()

    def fvecs_read(fname):
    	return ivecs_read(fname).view('float32')

    def load_data(self):
        self.xt = fvecs_read(self.train_data_path)
        self.xb = fvecs_read(self.base_data_path)

    def build_index(self):
	    nt, d = self.xt.shape
        index = faiss.index_factory(d, "IVF4096,PQ64")
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        index.setNumProbes(nprobe)

	    index.train(xt)
	    index.add(xb)

	    print("finish building index")

    def search(self, query):
        t0 = time.time()
        D, I = index.search(query, 100)
        t1 = time.time()
        print("search uses  %.4f s" %(t1-t0))



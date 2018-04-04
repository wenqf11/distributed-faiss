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

tempmem = -1
#tempmem = (2048+1024+512)*1024*1024

class IVFPQ_cpu():
    m = 8                   
    nlist = 4096
    nprobe = 128

    def __init__(self, xt_path="/root/dataset/sift1m/sift_learn.fvecs", xb_path="/root/dataset/sift1m/sift_base.fvecs"):
        self.xt = self.fvecs_read(xt_path)
        self.xb = self.fvecs_read(xb_path)

    def ivecs_read(self, fname):
        a = np.fromfile(fname, dtype='int32')
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy()

    def fvecs_read(self, fname):
    	return self.ivecs_read(fname).view('float32')

    def build_index(self):
	    nt, d = self.xt.shape
	    quantizer = faiss.IndexFlatL2(d)  # this remains the same
	    self.index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, 8)
	    self.index.train(self.xt)
	    self.index.add(self.xb)
	    self.index.nprobe = self.nprobe 
	    print("finish building index")

    def search(self, query):
        print("start searching")
        t0 = time.time()
        D, I = self.index.search(query, 100)
        t1 = time.time()
        print("search uses  %.4f s" %(t1-t0))
        return D.tolist()[0], I.tolist()[0]


class IVFPQ_gpu():
    m = 8
    nlist = 4096
    nprobe = 128 

    def __init__(self, xt_path="/home/wenqingfu/project/faiss/sift1M/sift_learn.fvecs", xb_path="/home/wenqingfu/project/faiss/sift1M/sift_base.fvecs"):
        self.xt = self.fvecs_read(xt_path)
        self.xb = self.fvecs_read(xb_path)
        self.res = faiss.StandardGpuResources()
        self.co = faiss.GpuClonerOptions()
        self.co.useFloat16 = True
        self.co.usePrecomputed = False

    def ivecs_read(self, fname):
        a = np.fromfile(fname, dtype='int32')
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy()

    def fvecs_read(self, fname):
    	return self.ivecs_read(fname).view('float32')

    def build_index(self):
        nt, d = self.xt.shape
        index = faiss.index_factory(d, "IVF4096,PQ64")
        self.index = faiss.index_cpu_to_gpu(self.res, 0, index, self.co)
        self.index.setNumProbes(self.nprobe)
        
        self.index.train(self.xt)
        self.index.add(self.xb)
        
        print("finish building index")

    def search(self, query):
        t0 = time.time()
        D, I = self.index.search(query, 100)
        t1 = time.time()
        print("search uses  %.4f s" %(t1-t0))
        return D.tolist()[0], I.tolist()[0]


class IVFPQ_multiGpu():
    m = 8
    nlist = 4096
    nprobe = 128
    db_start = 33
    db_end = 67

    def __init__(self, xt_path="/root/dataset/sift1b/bigann_learn.bvecs", xb_path="/root/dataset/sift1b/bigann_base.bvecs", ngpu=3):
        self.xt = self.mmap_bvecs(xt_path)
        self.xb = self.mmap_bvecs(xb_path)
        self.xt = self.sanitize(self.xt[:1000000])
        self.xb = self.sanitize(self.xb[self.db_start*1000*1000:self.db_end*1000*1000])
        self.gpu_resources = []
        for i in range(0,ngpu):
            res = faiss.StandardGpuResources()
            if tempmem >= 0:
                res.setTempMemory(tempmem)
                print("set tempemm to %d" % tempmem)
            self.gpu_resources.append(res)
        self.vres = faiss.GpuResourcesVector()
        self.vdev = faiss.IntVector()

        for i in range(0, ngpu):
            self.vdev.push_back(i)
            self.vres.push_back(self.gpu_resources[i])

        self.co = faiss.GpuMultipleClonerOptions()
        self.co.useFloat16 = True 
        self.co.useFloat16CoarseQuantizer = False
        self.co.usePrecomputed = False 
        self.co.indicesOptions = 0
        self.co.verbose = True
        self.co.shard = True  

        self.ps = faiss.GpuParameterSpace()

    def ivecs_read(self, fname):
        a = np.fromfile(fname, dtype='int32')
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy()

    def fvecs_read(self, fname):
        return self.ivecs_read(fname).view('float32')
    
    def mmap_bvecs(self, fname):
        x = np.memmap(fname, dtype='uint8', mode='r')
        d = x[:4].view('int32')[0]
        return x.reshape(-1, d + 4)[:, 4:]

    def sanitize(self, x):
        """ convert array to a c-contiguous float array """
        return np.ascontiguousarray(x.astype('float32'))

    def build_index(self):
        nt, d = self.xt.shape
        index = faiss.index_factory(d, "IVF4096,PQ64")
        self.index = faiss.index_cpu_to_gpu_multiple(self.vres, self.vdev, index, self.co)
        
        self.index.train(self.xt)
        self.index.add(self.xb)
        
        self.ps.initialize(self.index)
        self.ps.set_index_parameter(self.index, 'nprobe', self.nprobe)

        print("finish building index")

    def search(self, query, nprobe):
        self.ps.set_index_parameter(self.index, 'nprobe', nprobe)
        t0 = time.time()
        D, I = self.index.search(query, 100)
        t1 = time.time()
        print("search uses  %.4f s" %(t1-t0))
        #return D.tolist()[0], I.tolist()[0]
        return D.tolist(), I.tolist()

import zerorpc
from multiprocessing.pool import Pool
from multiprocessing.dummy import Pool as ThreadPool
from indexPQ import *


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))

path = "/home/wenqingfu/data/sift1b/bigann"

#xq = mmap_bvecs(path+"_query.bvecs")
#xq = fvecs_read("/home/wenqingfu/project/faiss/sift1M/sift_query.fvecs")

dbsize = 100

xq = mmap_bvecs(path+"_query.bvecs")
gt = ivecs_read('/home/wenqingfu/data/sift1b/gnd/idx_%dM.ivecs' % dbsize)

xq = sanitize(xq)

nq, d = xq.shape

xq = xq.tolist()

def call_rpc(arg):
    rpc = zerorpc.Client(arg[0], heartbeat=None, timeout=100)
    return rpc.search(xq, int(arg[1]))

for lnprobe in range(10):
    nprobe = 1 << lnprobe
    #ps.set_index_parameter(index, 'nprobe', nprobe)
    t0 = time.time()
    pool = Pool(3)
    result = pool.map(call_rpc, [
        ["tcp://166.111.80.138:8281", str(nprobe)],
        ["tcp://166.111.80.130:8281", str(nprobe)],
        ["tcp://192.168.3.101:8281", str(nprobe)]
    ])
    t1 = time.time()
    print("nprobe=%4d %.3f s recalls=" % (nprobe, t1 - t0), end="")
    
    D, I = result[0]
    D2, I2 = result[1]
    D3, I3 = result[2]
    I = np.asarray(I, dtype=np.int32)
    I2 = np.asarray(I2, dtype=np.int32)
    I3 = np.asarray(I3, dtype=np.int32)
    I2 = I2 + 33000000
    I3 = I3 + 67000000

    for rank in 1, 10, 100:
        n_ok = (I[:, :rank] == gt[:, :1]).sum()
        n_ok = n_ok + (I2[:, :rank] == gt[:, :1]).sum()
        n_ok = n_ok + (I3[:, :rank] == gt[:, :1]).sum()
        print("%.4f" % (n_ok / float(nq)),end=" ")
    print()

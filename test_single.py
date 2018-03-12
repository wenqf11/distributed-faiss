from indexPQ import *
import zerorpc

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

path = "/home/wenqingfu/sift1b/bigann"
rpc = zerorpc.Client("tcp://127.0.0.1:8281", timeout=1000000, heartbeat=None)

xq = mmap_bvecs(path+"_query.bvecs")
#xq = fvecs_read("/home/wenqingfu/project/faiss/sift1M/sift_query.fvecs")

dbsize = 50
path = "/home/wenqingfu/sift1b/bigann"

xq = mmap_bvecs(path+"_query.bvecs")
gt = ivecs_read('/home/wenqingfu/sift1b/gnd/idx_%dM.ivecs' % dbsize)

xq = sanitize(xq)

nq, d = xq.shape

for lnprobe in range(10):
    nprobe = 1 << lnprobe
    #ps.set_index_parameter(index, 'nprobe', nprobe)
    t0 = time.time()
    D, I = rpc.search(xq.tolist(), nprobe)
    t1 = time.time()
    I = np.asarray(I, dtype=np.int32)

    print("nprobe=%4d %.3f s recalls=" % (nprobe, t1 - t0), end="")
    for rank in 1, 10, 100:
        n_ok = (I[:, :rank] == gt[:, :1]).sum()
        print("%.4f" % (n_ok / float(nq)),end=" ")
    print()

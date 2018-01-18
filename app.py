from sanic import Sanic
from sanic.response import json
import zerorpc
import numpy as np


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
rpc = zerorpc.Client()
rpc.connect("tcp://127.0.0.1:8282")
xq = mmap_bvecs(path+"_query.bvecs")
#xq = fvecs_read("/home/wenqingfu/project/faiss/sift1M/sift_query.fvecs")
print(rpc.search(xq[0,:].tolist()))

app = Sanic()

@app.route("/")
async def test(request):
    rpc.hello("RPC")
    return json({"hello": "world"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

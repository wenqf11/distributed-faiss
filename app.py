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


rpc = zerorpc.Client()
rpc.connect("tcp://127.0.0.1:8282")
xq = fvecs_read("/root/dataset/sift1m/sift_query.fvecs")
print(rpc.search(xq[0,:].tolist()))

app = Sanic()

@app.route("/")
async def test(request):
    rpc.hello("RPC")
    return json({"hello": "world"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

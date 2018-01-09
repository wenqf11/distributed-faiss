from indexPQ import *



index = IVFPQ_gpu()

xq = index.fvecs_read("/root/dataset/sift1m/sift_query.fvecs")
index.build_index()

D, I = index.search(xq[0:1, :])

print(D)
print(I)

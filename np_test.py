import numpy as np


p = np.array([0.9, 0.1])
pprime = np.array([0.5, 0.5])
t = np.array([[0.9, 0.1],
              [0.1, 0.9]])
# eigenvalues, eigenvectors = np.linalg.eig(t)
# print(eigenvectors)

print(p@t@t@t@t@t@t@t@t@t@t)
print(pprime@t@t@t@t@t@t@t)
# P = np.array([0,1,2,3,4,5])
# print(P.reshape(2,3))
# A = np.array([[0,1,2,3,4,5],
#               [6,7,8,9,10,11],
#               [12,13,14,15,16,17],
#               [18,19,20,21,22,23],
# [24,25,26,27,28,29],
# [30,31,32,33,34,35]
#               ])
# A_test=A.reshape(2,3,2,3)
# print(A_test[0,:,0,:])
# P_test = P.reshape(2,3)
# print(P_test[0,:])
# print(P@A)
# testforward = (P@A).reshape(2,3)
# print(testforward[0,:])
# probacond = np.array([[0.1,0.9], [0.8,0.2], [0.3,0.7]])
# probacond = np.repeat(probacond, 3, axis=1)
# print(probacond[0])
# testf=(probacond[0]*(P@A)).reshape(2,3)
# print(testf[0,:])

# c = np.array([[0.05061065, 0.17434422],
#        [0.17434431, 0.60070082]])
# p = c.sum(axis=1)
# t = (c.T / p).T
# print(p, t)

# cu = np.array([[1,2],[3,4]])
#
# nbc_x = 2
# nbc_u = 2
#
# x = np.array([0,1,1,1,0,1,0,1,0,1])
# psi = np.array([[0.1,0.9], [0.1,0.9], [0.1,0.9], [0.9,0.1], [0.1,0.9], [0.9,0.1], [0.1,0.9], [0.9,0.1], [0.1,0.9], [0.9,0.1]])
# test = (x[..., np.newaxis] == np.indices((nbc_x,)))[...,np.newaxis]
# test2 = psi[:,np.newaxis,:] * (x[..., np.newaxis] == np.indices((nbc_x,)))[...,np.newaxis]
#
# test3 = test2.sum(axis=0)
# psi_sum = psi.sum(axis=0)
#
#
#
# px = test3/psi_sum[np.newaxis,...]
# print(px)
# print(cu)
#
# def build_mat(cu, px, nbc_x, nbc_u):
#     c_res = np.zeros((nbc_x,nbc_u,nbc_x,nbc_u))
#     for i in range(nbc_x):
#         for k in range(nbc_u):
#             for j in range(nbc_x):
#                 for l in range(nbc_u):
#                     c_res[i,k,j,l] = cu[k,l]*px[i,k]*px[j,l]
#     return c_res
#
# textpx = px[...,np.newaxis, np.newaxis]*px[np.newaxis, np.newaxis,...]
# print(textpx)
# testf = cu[np.newaxis,:,np.newaxis,:]*textpx
# print(testf)
#
# print(build_mat(cu, px, nbc_x, nbc_u))




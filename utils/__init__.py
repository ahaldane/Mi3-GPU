import numpy as np

def printsome(a, prec=6):
    return np.array2string(a.flatten()[:5], precision=prec, sign=' ')[1:-1]

def getLq(x):
    L = int(((1+np.sqrt(1+8*x.shape[0]))//2) + 0.5)
    q = int(np.sqrt(x.shape[1]) + 0.5)
    return L, q

def unimarg(bimarg):
    L, q = getLq(bimarg)
    ff = bimarg.reshape((L*(L-1)//2,q,q))
    f = np.array([np.sum(ff[0],axis=1)] +
                 [np.sum(ff[n],axis=0) for n in range(L-1)])
    return f/np.sum(f, axis=1, keepdims=True) # correct any fp errors

def indep_bimarg(bimarg):
    f = unimarg(bimarg)
    L = f.shape[0]
    return np.array([np.outer(f[i], f[j]).flatten() for i in range(L-1)
                                                    for j in range(i+1,L)])

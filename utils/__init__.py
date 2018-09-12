from numpy import array2string

def printsome(a, prec=6):
    return array2string(a.flatten()[:5], precision=prec, sign=' ')[1:-1]

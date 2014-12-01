#!/usr/bin/env python2
from __future__ import with_statement
from scipy import *
import sys, types
import itertools

class Opener:
    def __init__(self, fileobj, rw="rt"):
        self.fileobj = fileobj
        self.rw = rw
        self.f = None

    def __enter__(self):
        if isinstance(self.fileobj, types.StringType):
            self.f = open(self.fileobj, self.rw)
            return self.f
        elif hasattr(self.fileobj, 'read') or hasattr(self.fileobj, 'write'):
            if self.rw != self.fileobj.mode: #error? XXX
                raise Exception("File is already open ({}), but in wrong mode ({})".format(self.fileobj.mode, self.rw))
            return self.fileobj

    def __exit__(self, e, fileobj, t):
        if self.f != None:
            self.f.close()
        return False

def getHeader(fn):
    with Opener(fn, 'rt') as f:
        header = list(itertools.takewhile(lambda s: s.startswith('#'), f))
    return header

def parseHeader(hd):
    etable = []
    tree = None
    info = []
    param = None
    for line in hd:
        if line.startswith('#ETABLE '):
            etable.append(line)
        elif line.startswith('#TREE '):
            tree = line[len('#TREE '):]
        elif line.startswith('#PARAM '):
            terms = [kv.split(':') for kv in line[len('#PARAM '):].split(',')]
            param = dict([(k.strip(), eval(v)) for k,v in terms])
        elif line.startswith('#INFO '):
            info.append(line)
    
    etable = array([[float(x) for x in line[len('#ETABLE '):].split()] 
                                   for line in etable])
    
    return info, param, etable, tree

def loadSites(fn, names=None): #optimized for fast loading, assumes ASCII
    if isinstance(fn, types.StringType):
        f = open(fn, 'rt')
    else:
        f = fn

    pos = f.tell()
    header = []
    l = f.readline()
    while l.startswith('#'):
        header.append(l)
        pos = f.tell()
        l = f.readline()
    L = len(l[:-1])
    f.seek(pos)

    try:
        info, param, etable, tree = parseHeader(header)
    except Exception as e:
        print e
        info, param, etable, tree = None,None,None,None
    if names == None:
        if param == None or 'alpha' not in param:
            raise Exception("Could not determine names of alphabet")
        names = param['alpha']

    nucNums = -ones(256, uint8) #nucNums is a map from ascii to base number
    nucNums[frombuffer(names, uint8)] = arange(len(names))

    #make this load in chunks, and do binary load XXX
    chunksize = 4096
    chunks = []
    while 1:
        dat = fromfile(f, dtype=uint8, count=chunksize*(L+1))

        if (dat.size % (L+1)) != 0: #check if end of file messed up
            if dat[-1] == ord("\n") and ((dat.size-1) % (L+1)) == 0:
                dat = dat[:-1] #account for newline at eof
            else:
                raise Exception("Unexpected characters at eof") 

        if len(dat) == 0: #are we done?
            break

        dat = dat.reshape(dat.size/(L+1), L+1)

        if any(dat[:,-1] != ord('\n')):
            badline = sum([c.shape[0] for c in chunks])
            badline += argwhere(dat[:,-1] != ord('\n'))[0]
            raise Exception("Sequence {} has different length".format(badline))

        dat = nucNums[dat[:,:-1]] #requires intermediate cast to int....

        if any(dat.flatten() < 0):
            badpos = argwhere(dat.flatten() < 0)[0]
            badchar = dat.flatten()[badpos]
            if badchar == ord('\n'):
                badline = sum([c.shape[0] for c in chunks])
                badline += badpos/L
                raise Exception("Sequence {} has different length".format(badline))
            else:
                raise Exception("Invalid residue: {0}".format(badchar))
            
        chunks.append(dat)

    seqs = concatenate(chunks)
    f.close()
    return seqs, (names,info,param,etable,tree)

def writeSites(fn, seqs, names, info=None, param=None, etable=None, etree=None):
    with Opener(fn, 'w') as f:
        if param != None:
            pstr = ",".join("{0}: {1}".format(k,repr(v)) for k,v in param.iteritems())
            f.write('#PARAM {0}\n'.format(pstr))
        if info != None:
            f.write('#INFO {0}\n'.format(info))
        if etable != None:
            for r in etable:
                f.write('#ETABLE {0}\n'.format(" ".join(str(c) for c in r)))
        if etree != None:
            f.write('#INFO {0}\n'.format(etree))

        chunksize = 4096
        alphabet = array([ord(c) for c in names] + [ord('\n')], dtype='<u1')
        s = empty((chunksize, seqs.shape[1]+1), dtype=intp)
        s[:,-1] = len(names)
        for i in range(0,seqs.shape[0], chunksize):
            s[:,:-1] = seqs[i:i+chunksize,:]
            alphabet[s].tofile(f)
        if i+chunksize != seqs.shape[0]:
            s[:seqs.shape[0]-i-chunksize,:-1] = seqs[i+chunksize:,:]
            alphabet[s[:seqs.shape[0]-i-chunksize,:]].tofile(f)

def getCounts(seqs, nBases):
    nSeq, seqLen = seqs.shape
    bins = arange(nBases+1, dtype='int')
    counts = zeros((seqLen, nBases), dtype='int')
    for i in range(seqLen):
        counts[i,:] = histogram(seqs[:,i], bins)[0]
    return counts # index as [pos, res]

def getFreqs(seq, nBases):
    return getCounts(seq, nBases).astype('float')/seq.shape[0]

def loadSitesGen(fn, names):
    nucNums = -ones(256, int) #nucNums is a map from ascii to base number
    nucNums[frombuffer(names, uint8)] = arange(len(names))

    data = []
    with Opener(fn, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            yield nucNums[frombuffer(line[:-1], uint8)]

def loadSitesGradual(fn, names, mapper=None):
    #for use when memory is an issue
    if mapper == None:
        mapper = lambda x: x

    nucNums = -ones(256, int) #nucNums is a map from ascii to base number
    nucNums[frombuffer(names, uint8)] = arange(len(names))
    
    data = []
    with Opener(fn, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            seq = nucNums[frombuffer(line[:-1], uint8)]
            data.append(mapper(seq))

    return array(data)

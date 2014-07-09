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

    l = f.next()
    header = []
    while l.startswith('#'):
        header.append(l)
        l = f.next()
    seqs = [l.strip()] + [l.strip() for l in f]
    if seqs[-1] == '':
        del seqs[-1]
    
    if isinstance(fn, types.StringType):
        f.close()
    
    try:
        info, param, etable, tree = parseHeader(header)
    except Exception as e:
        print e
        info, param, etable, tree = None,None,None,None
    if names == None:
        if param == None or 'alpha' not in param:
            raise Exception("Could not determine names of alphabet")
        names = param['alpha']

    nSeqs, seqLen = len(seqs), len(seqs[0])
    if any([len(s) != seqLen for s in seqs]): #check that file is OK
        raise Exception("Error: Sequences have different lengths")

    nucNums = -ones(256, int) #nucNums is a map from ascii to base number
    nucNums[frombuffer(names, uint8)] = arange(len(names))
    
    bases = frombuffer("".join(seqs), uint8).reshape((nSeqs, seqLen))
    seqTable = nucNums[bases]

    badBases = seqTable < 0
    if any(badBases):
        raise Exception("Invalid residue(s): {0}".format(" ".join(chr(i) for i in set(bases[badBases]))))

    return seqTable, (names,info,param,etable,tree)

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

def writeSites(fn, seqs, names, info=None, param=None, etable=None, etree=None):
    bs = []
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

        for s in seqs:
            f.write("".join(names[i] for i in s))
            f.write("\n")

def getCounts(seqs, nBases):
    nSeq, seqLen = seqs.shape
    bins = arange(nBases+1, dtype='int')
    counts = zeros((seqLen, nBases), dtype='int')
    for i in range(seqLen):
        counts[i,:] = histogram(seqs[:,i], bins)[0]
    return counts # index as [pos, res]

def getFreqs(seq, nBases):
    return getCounts(seq, nBases).astype('float')/seq.shape[0]

#unoptimized but clearer version of loadSites:
#def loadSites(fn, names):
#    with Opener(fn, 'rt') as f:
#        dat = f.read()
#        if dat[0] == '#': #skip comment in first line if present
#            dat = dat[dat.index('\n'):]
#        seqs = dat.split()
#    
#    nucNums = dict(zip(names, range(len(names))))
#    bs = []
#    for s in seqs:
#        bs.append(array([nucNums[n] for n in s]))
#    return array(bs)

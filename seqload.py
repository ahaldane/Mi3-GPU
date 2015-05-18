#!/usr/bin/env python2
from __future__ import with_statement
from scipy import *
import sys
import json

class Opener:
    def __init__(self, fileobj, rw="rt"):
        self.fileobj = fileobj
        self.rw = rw
        self.f = None

    def __enter__(self):
        if isinstance(self.fileobj, basestring):
            self.f = open(self.fileobj, self.rw)
            return self.f
        elif hasattr(self.fileobj, 'read') or hasattr(self.fileobj, 'write'):
            if self.rw != self.fileobj.mode: #error? XXX
                raise Exception(("File is already open ({}), but in wrong mode "
                                "({})").format(self.fileobj.mode, self.rw))
            return self.fileobj

    def __exit__(self, e, fileobj, t):
        if self.f != None:
            self.f.close()
        return False

def loadSeqs(fn, names=None): 
    with Opener(fn) as f:
        gen = loadSeqsChunked(f, names)
        param, headers = gen.next()
        seqs = concatenate([s for s in gen])
    return seqs, param, headers

def mapSeqs(fn, names, mapper):
    if mapper == None:
        mapper = lambda x: x
    
    with Opener(fn) as f:
        gen = loadSeqsChunked(f, names)
        param, headers = gen.next()
        seqs = concatenate([mapper(s) for s in gen])
    return seqs, param, headers

def parseHeader(hd):
    param = {}
    headers = {}
    for line in hd:
        if line[1:].isspace():
            continue
        elif line.startswith('#PARAM '):
            param = json.loads(line[len('#PARAM '):])
        elif line.startswith('# '):
            headers['comments'].append(line[2:])
        else: #assumes first word is header category
            wordend = line.find(' ')
            if wordend == -1:
                head = line[1:]
            else:
                head = line[1:wordend]
            headers[head] = headers.get(head,[]) + [line[len(head)+2:]]
            
    return param, headers

#optimized for fast loading, assumes ASCII
def loadSeqsChunked(f, names=None, chunksize=None): 
    #read header
    pos = f.tell()
    header = []
    l = f.readline()
    while l.startswith('#'):
        header.append(l)
        pos = f.tell()
        l = f.readline()
    L = len(l[:-1]) + (1 if l[-1] != '\n' else 0)
    f.seek(pos)
    
    #get alphabet
    param, headers = parseHeader(header)
    if names == None:
        if param == {} or 'alpha' not in param:
            raise Exception("Could not determine names of alphabet")
        names = param['alpha']
    param['alpha'] = names

    yield param, headers
    
    #set up translation table
    nucNums = 255*ones(256, uint8) #nucNums is a map from ascii to base number
    nucNums[frombuffer(names, uint8)] = arange(len(names))
    
    #do translation
    def translateSeqs(seqmat, pos):
        seqmat = seqmat.reshape(seqmat.size/(L+1), L+1)
        
        #sanity checks on the sequences
        if any(seqmat[:,-1] != ord('\n')):
            badline = pos[0] + argwhere(seqmat[:,-1] != ord('\n'))[0] 
            raise Exception("Sequence {} has different length".format(badline))

        #fancy indexing casts indices to intp.... annoying slowdown
        seqs = nucNums[seqmat[:,:-1]] 

        if any(seqs.flatten() == 255):
            badpos = argwhere(seqs.flatten() == 255)[0]
            badchar = chr(seqmat.flatten()[badpos])
            if badchar == '\n':
                badline = pos[0] + badpos/L
                raise Exception("Sequence {} has wrong length".format(badline))
            else:
                raise Exception("Invalid residue: {0}".format(badchar))
        pos[0] += seqmat.shape[0]
        return seqs

    #load in chunks
    if chunksize is None:
        chunksize = 4*1024*1024/(L+1)
    dat = fromfile(f, dtype=uint8, count=chunksize*(L+1))
    pos = [0]
    while dat.size == chunksize*(L+1):
        yield translateSeqs(dat, pos)
        dat = fromfile(f, dtype=uint8, count=chunksize*(L+1))
    
    #process last partial chunk if present
    if dat.size != 0:
        #correct for extra/missing newline at end of file
        if (dat.size % (L+1)) != 0: 
            if dat[-1] == ord("\n") and ((dat.size-1) % (L+1)) == 0:
                dat = dat[:-1] #account for newline at eof
            elif ((dat.size+1) % (L+1)) == 0:
                dat = concatenate([dat, [ord("\n")]])
            else:
                raise Exception("Unexpected characters at eof") 
        yield translateSeqs(dat, pos)

def writeSeqs(fn, seqs, names, param=None, headers=None, noheader=False):
    with Opener(fn, 'w') as f:
        writeSeqsF(f, seqs, names, param, headers, noheader)

def writeSeqsF(f, seqs, names, param=None, headers=None, noheader=False):
    if not noheader:
        param = param if param != None else {}
        param['alpha'] = names
        f.write('#PARAM {0}\n'.format(json.dumps(param)))
        headers = headers if headers != None else []
        for head in headers:
            for line in headers[head]:
                f.write('#{} {0}\n'.format(head, line))

    chunksize = 1024*1024
    alphabet = array([ord(c) for c in names] + [ord('\n')], dtype='<u1')
    s = empty((chunksize, seqs.shape[1]+1), dtype=intp)
    s[:,-1] = len(names)
    i = -chunksize # in case len(seqs) < chunksize
    for i in range(0,seqs.shape[0]-chunksize, chunksize):
        s[:,:-1] = seqs[i:i+chunksize,:]
        alphabet[s].tofile(f)
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

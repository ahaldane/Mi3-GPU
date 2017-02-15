#!/usr/bin/env python2
#
#Copyright 2016 Allan Haldane.

#This file is part of IvoGPU.

#IvoGPU is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#IvoGPU is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with IvoGPU.  If not, see <http://www.gnu.org/licenses/>.

#Contact: allan.haldane _AT_ gmail.com
from __future__ import with_statement
from scipy import *
import sys
import json
import seqtools
import bz2, io

class Opener:
    def __init__(self, fileobj, rw="r", zipf=None):
        self.fileobj = fileobj
        if rw not in "rwa":
            raise Exception("File mode must be r,w, or a")
        self.rw = rw
        self.f = None
        self.zipf = zipf
    
    def __enter__(self):
        if isinstance(self.fileobj, basestring):
            if self.rw in 'ra' and self.zipf is not False:
                magic = "\x42\x5a\x68"  # for bz2
                f = open(self.fileobj, self.rw + 'b')
                start = f.read(len(magic))
                f.close()
                if start == magic:
                    self.f = bz2.BZ2File(self.fileobj, self.rw+'b')
                elif self.zipf is True:
                    raise Exception("bz2 file expected")
            elif self.rw == 'w' and self.zipf:
                self.f = bz2.BZ2File(self.fileobj, self.rw+'b')

            if self.f is None:
                self.f = open(self.fileobj, self.rw + 'b')

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

def loadSeqs(fn, names=None, zipf=None): 
    with Opener(fn, zipf=zipf) as f:
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
        seqs = concatenate([mapper(s, (param, headers)) for s in gen])
    return seqs, param, headers

def reduceSeqs(fn, reduc, startval=None, names=None):
    with Opener(fn) as f:
        gen = loadSeqsChunked(f, names)
        param, headers = gen.next()
        val = startval if startval != None else gen.next()
        for s in gen:
            val = reduc(val, s, (param, headers))
    return val, param, headers

def parseHeader(hd):
    param = {}
    headers = {}
    for line in hd:
        if line[1:].isspace():
            continue
        elif line.startswith('#PARAM '):
            param = json.loads(line[len('#PARAM '):])
        elif line.startswith('# '):
            headers['comments'] = headers.get('comments',[]) + [line[2:]]
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
    
    #load in chunks
    if chunksize is None:
        chunksize = 4*1024*1024/(L+1)

    pos = 0
    while True:
        dat = frombuffer(f.read(chunksize*(L+1)), dtype=uint8)
        if dat.size != chunksize*(L+1):
            break
        seqmat = dat.reshape(dat.size/(L+1), L+1)
        seqtools.translateascii(seqmat, names, pos)
        pos += seqmat.shape[0]
        yield seqmat[:,:-1]

    if dat.size == 0:
        return
    
    #process last partial chunk if present
    #correct for extra/missing newline at end of file
    if (dat.size % (L+1)) != 0: 
        if dat[-1] == ord("\n") and ((dat.size-1) % (L+1)) == 0:
            dat = dat[:-1] #account for newline at eof
        elif ((dat.size+1) % (L+1)) == 0:
            dat = concatenate([dat, [ord("\n")]])
        else:
            raise Exception("Unexpected characters at eof") 
    seqmat = dat.reshape(dat.size/(L+1), L+1)
    seqtools.translateascii(seqmat, names, pos)
    yield seqmat[:,:-1]

def writeSeqs(fn, seqs, names, param=None, headers=None, noheader=False, zipf=None):
    with Opener(fn, 'w', zipf) as f:
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
        # could be sped up: s is uneccesarily cast to int32/64
        f.write(alphabet[s])
    s[:seqs.shape[0]-i-chunksize,:-1] = seqs[i+chunksize:,:]
    f.write(alphabet[s[:seqs.shape[0]-i-chunksize,:]])

def getCounts(seqs, nBases):
    nSeq, seqLen = seqs.shape
    bins = arange(nBases+1, dtype='int')
    counts = zeros((seqLen, nBases), dtype='int')
    for i in range(seqLen):
        counts[i,:] = histogram(seqs[:,i], bins)[0]
    return counts # index as [pos, res]

def getFreqs(seq, nBases):
    return getCounts(seq, nBases).astype('float')/seq.shape[0]

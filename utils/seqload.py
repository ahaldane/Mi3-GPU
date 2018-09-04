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
import numpy as np
import sys
import json, bz2, io

def translateascii_python(seqmat, names, pos):
    #set up translation table
    nucNums = 255*ones(256, uint8) #nucNums maps from ascii to base number
    nucNums[frombuffer(names, uint8)] = arange(len(names))
    
    #sanity checks on the sequences
    if any(seqmat[:,-1] != ord('\n')):
        badline = pos + argwhere(seqmat[:,-1] != ord('\n'))[0] 
        raise Exception("Sequence {} has different length".format(badline))

    #fancy indexing casts indices to intp.... annoying slowdown
    seqs = nucNums[seqmat[:,:-1]] 

    if np.any(seqs == 255):
        badpos = argwhere(seqs.flatten() == 255)[0]
        badchar = chr(seqmat[:,:-1].flatten()[badpos])
        if badchar == '\n':
            badline = pos + badpos/L
            raise Exception("Sequence {} has wrong length".format(badline))
        else:
            raise Exception("Invalid residue: {0}".format(badchar))
    return seqs

try:
    import seqtools
    def translateascii(seqmat, names, pos):
        # translates in-place. seqmat must be writeable
        seqtools.translateascii(seqmat, names, pos)
        return seqmat[:,:-1]
except:
    translateascii = translateascii_python

try:
    from Bio.Alphabet import IUPAC
    prot_alpha = '-' + IUPAC.protein.letters
except:
    prot_alpha = None

class Opener:
    """
    Generalization of the file object, to accepting either a filename or an
    already opened file descriptor, and also detect and reads/write bzip2
    compressed files.
    """

    def __init__(self, fileobj, rw="r", zipf=None):
        """
        Constructor

        Parameters
        ----------
        fileobj : string or file object
            Filename or file opject to open.
        rw : string
            Read/write mode. Used to open new files, and also used to check
            already opened files.
        zipf : boolean or None
            If None, autodetect bzip2 content when reading. If True use bzip2
            compression. If False, don't compress.
        """
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

def loadSeqs(fn, names=prot_alpha, zipf=None): 
    """
    Load sequence from a file.

    Parameters
    ----------
    fn : string or file object
      The file to read from. This should contain sequences, one per line.
      It may start with a set of lines beginning with ``#`` which encode
      parameter and heading info. The line starting with ``#PARAM `` will
      be read as json specifying MSA parameters, lines starting with ``#WORD``
      will be read as a heading group "WORD", and lines starting with ``# ``
      will be read as comments.
    names : string
      The alphabet used to interpret the sequences. Should be ASCII.
    zipf : boolean or None
      Whether the file is compressed with bzip. If `None`, this is autodetected.

    Returns
    -------
    seqs : ndarray
      The sequences read. This is in the form of a uint8 array where each
      value corresponds an index in `names`.
    param : dictionary
        Parameter values from the ``#PARAM`` line.
    headers : dictionary of lists of strings
        Header values. The dictionary keys are the header group ``WORD`` as
        described above. Comments are under the key ``'comments'``
        
    """
    with Opener(fn, zipf=zipf) as f:
        gen = loadSeqsChunked(f, names)
        param, headers = gen.next()
        seqs = concatenate([s for s in gen])
    return seqs, param, headers

def mapSeqs(fn, mapper, names=prot_alpha, zipf=None):
    """
    Load sequence from a file and process them with a mapping function.

    This avoids storing all sequences in memory.

    See loadSeqs for description of other parameters.

    Parameters
    ----------
    mapper : function
      This function should take two arguments: A set of sequences as returned
      by `loadSeqs`, and a tuple (param, headers) like returned by `loadSeqs`.

    Returns
    -------
    vals : ndarray
      Array containing the values returned by the mapper function for each
      sequence.
    """
    if mapper == None:
        mapper = lambda x: x
    
    with Opener(fn) as f:
        gen = loadSeqsChunked(f, names)
        param, headers = gen.next()
        seqs = concatenate([mapper(s, (param, headers)) for s in gen])
    return seqs, param, headers

def reduceSeqs(fn, reduc, startval=None, names=prot_alpha):
    """
    Like mapseqs but does a reduction.
    """
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

def readbytes(f, count):
    if isinstance(f, file):
        dat = np.fromfile(f, dtype=uint8, count=count)
    else:
        dat = frombuffer(f.read(count), dtype=uint8)
        dat.flags.writeable = True
    return dat

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
        dat = readbytes(f, chunksize*(L+1))
        if dat.size != chunksize*(L+1):
            break
        seqmat = dat.reshape(dat.size/(L+1), L+1)
        seqs = translateascii(seqmat, names, pos)
        pos += seqmat.shape[0]
        yield seqs

    if dat.size == 0:
        return
    
    #process last partial chunk if present
    #correct for extra/missing newline at end of file
    if (dat.size % (L+1)) != 0: 
        if dat[-1] == ord("\n") and ((dat.size-1) % (L+1)) == 0:
            dat = dat[:-1] #account for newline at eof
        elif ((dat.size+1) % (L+1)) == 0:
            dat = concatenate([dat, [uint8(ord("\n"))]])
        else:
            raise Exception("Unexpected characters at eof") 
    seqmat = dat.reshape(dat.size/(L+1), L+1)
    seqs = translateascii(seqmat, names, pos)
    yield seqs

def writeSeqs(fn, seqs, names=prot_alpha, param={'alpha': prot_alpha}, 
              headers=None, noheader=False, zipf=None):
    """
    Write sequences to a file.

    Parameters
    ----------
    fn : string or file object
        File to write to.
    seqs : ndarray
        Sequences to write. Should be a 2d uint8 array.
    names : string
        Alphabet to use to write the sequences, corresponding to the index
        values in the seqs array.
    param : dictionary
        Dictionary of values to write to the ``#PARAM`` line of the file. See
        loadSeqs.
    headers : dictionary of lists of strings
        Dictionary of header info. See loadSeqs.
    noheader : boolean
        If True, don't write any param or header info, only sequences.
    zipf : boolean or None
        Whether to compress the file using bzip2
    """
    with Opener(fn, 'w', zipf) as f:
        if not noheader:
            param = param if param != None else {}
            param['alpha'] = names
            f.write('#PARAM {0}\n'.format(json.dumps(param)))
            headers = headers if headers != None else []
            for head in headers:
                for line in headers[head]:
                    f.write('#{} {0}\n'.format(head, line))

        chunksize = 4*1024*1024/(seqs.shape[1]+1)
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
    """
    Helper function which computes the counts of each letter at each position.
    """
    nSeq, seqLen = seqs.shape
    bins = arange(nBases+1, dtype='int')
    counts = zeros((seqLen, nBases), dtype='int')
    for i in range(seqLen):
        counts[i,:] = histogram(seqs[:,i], bins)[0]
    return counts # index as [pos, res]

def getFreqs(seq, nBases):
    """
    Helper function which computes the univariate marginals of an MSA
    """
    return getCounts(seq, nBases).astype('float')/seq.shape[0]

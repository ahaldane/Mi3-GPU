#!/usr/bin/env python3
#
#Copyright 2020 Allan Haldane.

#This file is part of Mi3-GPU.

#Mi3-GPU is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#Mi3-GPU is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with Mi3-GPU.  If not, see <http://www.gnu.org/licenses/>.

#Contact: allan.haldane _AT_ gmail.com
import numpy as np
import sys, os
import json, bz2, io
from pathlib import Path


from mi3gpu.utils.potts_common import alpha21
prot_alpha = alpha21.encode('ascii')

class Opener:
    """
    Generalization of the file object, to accepting either a filename or an
    already opened file descriptor, and also detect and reads/write bzip2
    compressed files.

    Makes binary files only, and is unbuffered.
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
        if rw.rstrip('b') not in "rwa":
            raise Exception("File mode must be r,w, or a")
        self.rw = rw
        self.f = None
        self.zipf = zipf

    def __enter__(self):
        if isinstance(self.fileobj, (str, Path)):
            rw = self.rw.strip('b').rstrip('t')
            if rw in 'ra' and self.zipf is not False:
                magic = b"\x42\x5a\x68"  # for bz2
                with open(self.fileobj, rw + 'b') as f:
                    start = f.read(len(magic))
                if start == magic:
                    self.f = bz2.BZ2File(self.fileobj, rw+'b')
                elif self.zipf is True:
                    raise Exception("bz2 file expected")
            elif rw == 'w' and self.zipf:
                self.f = bz2.BZ2File(self.fileobj, rw+'b')

            if self.f is None:
                self.f = open(self.fileobj, rw + 'b', buffering=0)

            return self.f

        elif hasattr(self.fileobj, 'read') or hasattr(self.fileobj, 'write'):
            if self.rw != self.fileobj.mode: #error? XXX
                self.fileobj = os.fdopen(self.fileobj.fileno(), self.rw)
            return self.fileobj
        else:
            raise ValueError("invalid filename or file object")

    def __exit__(self, e, fileobj, t):
        if self.f != None:
            self.f.close()
        return False

def loadSeqs(fn, alpha=prot_alpha, zipf=None):
    """
    Load sequence from a file.

    Parameters
    ----------
    fn : string or file object
      The file to read from. This should contain sequences, one per line,
      optionally with the sequence id at the start of the line. If ids are
      given, all the sequence ids must be padded with whitespace to the same
      length. The file may start with a set of lines beginning with ``#`` which
      encode parameter and heading comments. A line starting with ``#PARAM ``
      will be read as json specifying MSA parameters if present, and the only
      currently supported param is 'alpha' which is the alphabet to be used
      if the alpha option below is None. All other comments are ignored.
    alpha : string
      The alphabet used to interpret the sequences. Should be ASCII.
    zipf : boolean or None
      Whether the file is compressed with bzip. If `None`, this is autodetected.

    Returns
    -------
    seqs : ndarray
      The sequences, converted to a uint8 array where each value corresponds an
      index in `alpha`. This allows vectorized MSA manipulation using numpy
      tools.
    ids : ndarray
      The ids of the sequences as a numpy array of byte strings, if present in
      the file, or None
    headers : list
        List of header lines starting with '#' from the file, with leading '#'
        removed.

    """
    with Opener(fn, zipf=zipf) as f:
        gen = loadSeqsChunked(f, alpha)
        headers, alpha = next(gen)
        seqs, ids = zip(*gen)

    seqs = np.concatenate(seqs)
    if ids[0] is not None:
        ids = np.concatenate(ids)
    else:
        ids = None

    return seqs, ids, headers

def mapSeqs(fn, mapper, alpha=prot_alpha, zipf=None):
    """
    Load sequence from a file and process them with a mapping function.

    This avoids storing all sequences in memory.

    See loadSeqs for description of other parameters.

    Parameters
    ----------
    mapper : function
      This function should take two arguments: A set of sequences as returned
      by `loadSeqs`, and a tuple (ids, headers) like returned by `loadSeqs`.

    Returns
    -------
    vals : ndarray
      Array containing the values returned by the mapper function for each
      sequence.
    """
    if mapper == None:
        mapper = lambda x: x

    with Opener(fn) as f:
        gen = loadSeqsChunked(f, alpha)
        headers, alpha = next(gen)
        seqs = np.concatenate([mapper(s, (i, headers)) for s,i in gen])
    return seqs, headers

def reduceSeqs(fn, reduce_func, startval=None, alpha=prot_alpha):
    """
    Like mapseqs but does a reduction.
    """
    with Opener(fn) as f:
        gen = loadSeqsChunked(f, alpha)
        headers, alpha = next(gen)
        val = startval if startval != None else next(gen)
        for s,i in gen:
            val = reduce_func(val, s, i, headers, alpha)
    return val, headers

def readbytes(f, count, buf=None):
    # efficiently read binary data
    if buf is None:
        buf = np.empty(count, dtype=np.uint8)
    elif buf.size*buf.itemsize < count:
        raise ValueError("buf must be at least {} bytes, got {}".format(count,
                                                        buf.size*buf.itemsize))
    return buf[:f.readinto(buf) or 0]

def chunk_firstpass(dat, idL, lineL, pos):
    if dat.size%(lineL+1) != 0:  # +1 for newline
        raise ValueError('internal error: dat size must be multiple of L')
    dat = dat.reshape((dat.size//(lineL+1), lineL+1))

    badlines = np.where(dat[:,-1] != ord('\n'))[0]
    if len(badlines) != 0:
        raise ValueError("Sequence {} has wrong length (expected {})".format(
                          pos + 1 + badlines[0], lineL))
    dat = dat[:,:-1]

    ids = None
    if idL != 0:
        ids, dat = dat[:,:idL], dat[:,idL:]

        badids = np.where(ids[:,-1] != ord(' '))[0]
        if len(badids) != 0:
            raise ValueError("Sequence {} has wrong id field length. Expected "
                             "id length {}".format(pos + 1 + badids[0], idL))

        ids = ids.flatten().view('S{}'.format(ids.shape[1]))
        ids = np.char.strip(ids)

    return dat, ids

def translateascii_python(seqmat, alpha, pos):
    #set up translation table
    nucNums = np.full(256, 255, np.uint8)  #maps from ascii to base number
    nucNums[np.frombuffer(alpha, np.uint8)] = np.arange(len(alpha))

    # fancy indexing casts indices to intp and makes a copy => slowdown.
    # (C version in seqtools, further below, avoids this)
    seqs = nucNums[seqmat]
    L = seqs.shape[1]

    # check for errors
    badline, badcol = np.where(seqs == 255)
    if len(badline) > 0:
        badline, badcol = badline[0], badcol[0]

        badchar = chr(seqmat[badline, badcol])

        if badchar == '\n':
            raise ValueError("Sequence {} has length {} (expected {})".format(
                             badline + pos + 1, badcol, L))
        else:
            raise ValueError("Invalid residue '{}' in sequence {} position "
                             "{}".format(badchar, badline + pos + 1, badcol))
    return seqs

try:
    import mi3gpu.utils.seqtools as seqtools

    def translateascii(seqmat, alpha, pos):
        # translates in-place. seqmat must be writeable
        seqtools.translateascii(seqmat, alpha, pos)
        return seqmat
except:
    translateascii = translateascii_python

def loadSeqsChunked(f, alpha=None, chunksize=None):
    #read header
    pos = f.tell()
    header = []
    l = f.readline()
    while l.isspace() or l.startswith(b'#'):
        if not l.isspace():
            header.append(l[1:].decode('utf-8'))
        pos = f.tell()
        l = f.readline()
    f.seek(pos)

    # get length of first sequence line (should be same for all)
    firstline = l.rstrip(b'\n')
    lineL = len(firstline)

    # each line may be a sequence id, whitespace, then the sequence
    idL = firstline.rfind(b' ') + 1

    #get alphabet if in header
    if alpha is None:
        for h in header:
            if h.startswith("PARAM"):
               param = json.loads(line[len('PARAM '):])
               alpha = param.get('alpha', None)
        if alpha is None:
            raise Exception("Could not determine alphabet")
    if isinstance(alpha, str):
        alpha = alpha.encode('ascii')

    yield header, alpha

    #load in chunks
    if chunksize is None:
        chunksize = 4*1024*1024//(lineL+1)  # about 4MB
    buf = np.empty(chunksize*(lineL+1), dtype=np.uint8)

    pos = 0
    ids = None
    while True:
        dat = readbytes(f, chunksize*(lineL+1), buf)

        # simple attempt to account for stockholm terminator
        if dat.size > 2 and dat[-2] == ord('/'):
            if dat[-1] == ord('/'):
                dat = dat[:-2]
            elif dat.size > 3 and dat[-1] == ord('\n') and  dat[-3] == ord('/'):
                dat = dat[:-3]

        if dat.size != chunksize*(lineL+1):
            break # reached end of file

        seqmat, ids = chunk_firstpass(dat, idL, lineL, pos)
        seqs = translateascii(seqmat.copy(), alpha, pos)
        pos += seqmat.shape[0]
        yield seqs, ids

    if dat.size == 0:
        return

    #process last partial chunk if present

    #correct for extra/missing newline at end of file
    if (dat.size % (lineL+1)) != 0:
        if dat[-1] == ord("\n") and ((dat.size-1) % (lineL+1)) == 0:
            # remove one newline at eof
            dat = dat[:-1]
        elif ((dat.size+1) % (lineL+1)) == 0:
            # add one newline at eof. Use the fact that dat is a view of buf
            # and that buf is guaranteed to have space up to multiple of lineL+1
            dat = buf[:dat.size+1]
            dat[-1] = np.uint8(ord("\n"))
        else:
            # some kind of problem with sequences.
            # try processing dat up to last multiple of lineL+1 to get a good
            # exception message.
            line_mul = dat.size - (dat.size % (lineL+1))
            seqmat, ids = chunk_firstpass(dat[:line_mul], idL, lineL, pos)
            # if previous line didn't raise, raise now. Must be problem at eof.
            bad_tail = (b"".join(dat[line_mul:].view("S1"))).decode('utf-8')
            raise Exception("Unexpected characters at eof: {}".format(
                                                                repr(bad_tail)))


    seqmat, ids = chunk_firstpass(dat, idL, lineL, pos)
    seqs = translateascii(seqmat, alpha, pos)
    yield seqs, ids

def writeSeqs(fn, seqs, alpha=prot_alpha, ids=None,
              headers=None, write_param=False, zipf=None):
    """
    Write sequences to a file.

    Parameters
    ----------
    fn : string or file object
        File to write to.
    seqs : ndarray
        Sequences to write. Should be a 2d uint8 array.
    alpha : string
        Alphabet to use to write the sequences, corresponding to the index
        values in the seqs array.
    ids : iterable
        sequence ids, either bytes or str. Must be same length as sequences.
    headers : dictionary of lists of strings
        Dictionary of header info. See loadSeqs.
    write_param : boolean
        If True, write alphabet PARAM header
    zipf : boolean or None
        Whether to compress the file using bzip2
    """
    if isinstance(alpha, str):
        alpha = alpha.encode('ascii')

    # reformat ids to be whitespace-padded if given
    idL = 0
    if ids is not None:
        ids = [x if isinstance(x, bytes) else x.encode('utf-8') for x in ids]
        idL = max(len(x) for x in ids) + 1
        ids = np.array([x.ljust(idL) for x in ids], dtype='S')
        ids = ids[:,None].view('u1')

    # set up buffers for translation to ascii
    seqL = seqs.shape[1]
    bufL = idL + seqL + 1
    chunksize = 4*1024*1024//(bufL+1)
    buf = np.empty((chunksize, bufL), dtype=np.uint8)
    buf[:,-1] = ord('\n') # add newline at end of each row

    # prepare for alphabet translation.
    # could be sped up: s is uneccesarily cast to intp in 'take' below
    alphabet = np.array(list(alpha), dtype=np.uint8)

    with Opener(fn, 'wb', zipf) as f:
        if write_param:
            param = {'alpha': alpha.decode('utf-8')}
            f.write('#PARAM {0}\n'.format(json.dumps(param)).encode('utf-8'))
        if headers != None:
            headers = [h if h.startswith('#') else '#' + h for h in headers]
            f.write("\n".join(headers))

        i = -chunksize # in case len(seqs) < chunksize
        for i in range(0,seqs.shape[0]-chunksize, chunksize):
            if idL != 0:
                buf[:,:idL] = ids[i:i+chunksize,:]
            np.take(alphabet, seqs[i:i+chunksize,:], out=buf[:,idL:-1])
            f.write(buf)

        # process final chunk
        buf = buf[:seqs.shape[0]-i-chunksize,:]
        if idL != 0:
            buf[:,:idL] = ids[i+chunksize:,:]
        np.take(alphabet, seqs[i+chunksize:,:], out=buf[:,idL:-1])
        f.write(buf)

def getCounts(seqs, q):
    """
    Helper function which computes the counts of each letter at each position.
    """
    nSeq, seqLen = seqs.shape
    bins = np.arange(q+1, dtype='int')
    counts = np.zeros((seqLen, q), dtype='int')
    for i in range(seqLen):
        counts[i,:] = histogram(seqs[:,i], bins)[0]
    return counts # index as [pos, res]

def getFreqs(seq, q):
    """
    Helper function which computes the univariate marginals of an MSA
    """
    return getCounts(seq, q).astype('float')//seq.shape[0]
